import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

from cnn_model import SimpleCNN
from dmlpffn_model import DMLPFFN
from vae_model import VAE   # ✅ USE NEW VAE


# -------------------------------------------------
# Optional Focal Loss
# -------------------------------------------------

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


# -------------------------------------------------
# Model Builder
# -------------------------------------------------

def build_model(model_type, in_ch, num_classes):

    if model_type == "cnn":
        return SimpleCNN(num_channels=in_ch, num_classes=num_classes)

    elif model_type == "dmlpffn":
        return DMLPFFN(in_channels=in_ch, num_classes=num_classes)

    else:
        raise ValueError("Invalid model_type")


# -------------------------------------------------
# Training Function
# -------------------------------------------------

def train_model(
        dataloaders,
        in_ch,
        num_classes,
        model_type="cnn",
        optimizer_type="AdamW",
        loss_type="CrossEntropyLoss",
        lr=1e-4,
        weight_decay=5e-5,
        epochs=40,
        patience=10,
        scheduler_type="CosineAnnealingWarmRestarts",
        use_genai=True
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_model(model_type, in_ch, num_classes).to(device)

    # -------------------------------------------------
    # GenAI INIT (Conv VAE)
    # -------------------------------------------------

    if use_genai:
        gen_model = VAE(
            in_channels=in_ch,
            latent_dim=64,
            patch_size=9
        ).to(device)

        optimizer_gen = torch.optim.Adam(gen_model.parameters(), lr=1e-3)

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )

    # -------------------------------------------------
    # Scheduler
    # -------------------------------------------------

    scheduler = None
    if scheduler_type == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    # -------------------------------------------------
    # Class weights
    # -------------------------------------------------

    y_train = []
    for _, yb in dataloaders["train"]:
        y_train.extend(yb.numpy())

    y_train = np.array(y_train)
    classes_present = np.unique(y_train)

    if len(classes_present) == num_classes:
        weights = compute_class_weight("balanced", classes_present, y_train)
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        print("Class weights:", weights)
    else:
        print("Warning: missing classes → disabling class weights")
        class_weights = None

    # -------------------------------------------------
    # Weighted sampler
    # -------------------------------------------------

    class_counts = np.bincount(y_train)
    class_counts[class_counts == 0] = 1

    sample_weights = 1.0 / class_counts[y_train]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    dataloaders["train"] = torch.utils.data.DataLoader(
        dataloaders["train"].dataset,
        batch_size=dataloaders["train"].batch_size,
        sampler=sampler
    )

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------

    if loss_type == "CrossEntropyLoss":
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    else:
        criterion = FocalLoss()

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------

    best_acc = 0
    patience_counter = 0
    best_state = None

    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for xb, yb in dataloaders["train"]:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            # ---------------- GenAI Forward ----------------
            if use_genai:
                x_recon, mu, logvar = gen_model(xb)

                # 🔥 Fusion
                xb_fused = xb + 0.3 * x_recon
            else:
                xb_fused = xb

            # ---------------- Model Forward ----------------
            outputs = model(xb_fused)
            loss_cls = criterion(outputs, yb)

            # ---------------- GenAI Loss ----------------
            if use_genai:
                recon_loss = nn.functional.mse_loss(x_recon, xb)

                kl_loss = -0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )

                loss = loss_cls + 0.3 * (recon_loss + kl_loss)
            else:
                loss = loss_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if use_genai:
                optimizer_gen.step()
                optimizer_gen.zero_grad()

            train_loss += loss.item() * xb.size(0)

            preds = outputs.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss /= total
        train_acc = correct / total

        # ---------------- Validation ----------------

        model.eval()

        val_loss = 0
        y_true = []
        y_pred = []

        with torch.no_grad():

            for xb, yb in dataloaders["val"]:

                xb = xb.to(device)
                yb = yb.to(device)

                outputs = model(xb)
                loss = criterion(outputs, yb)

                val_loss += loss.item() * xb.size(0)

                preds = outputs.argmax(1).cpu()
                y_pred.append(preds)
                y_true.append(yb.cpu())

        val_loss /= len(dataloaders["val"].dataset)

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()

        val_acc = accuracy_score(y_true, y_pred)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

        if scheduler:
            scheduler.step()

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, history
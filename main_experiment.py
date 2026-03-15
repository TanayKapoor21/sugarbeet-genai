import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import cfg
from data_preprocessing import (
    load_sugarbeet_dataset,
    spectral_normalize,
    train_val_test_split
)

from advanced_dataset import AdvancedHSIDataset
from training_pipeline import train_model
from evaluation_metrics import evaluate_model
from genai_augmentor import GenerativeAugmentor


# -------------------------------------------------------
# Create dataloaders
# -------------------------------------------------------

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test):

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }

    dataloaders = {
        k: DataLoader(
            AdvancedHSIDataset(*v, augment=(k == "train")),
            batch_size=cfg.batch_size,
            shuffle=(k == "train")
        )
        for k, v in splits.items()
    }

    return dataloaders


# -------------------------------------------------------
# Training wrapper
# -------------------------------------------------------

def train_pipeline(name, dataloaders, input_dim, model_type):

    print(f"\n=========== Training {name} ===========")

    model, history = train_model(
        dataloaders,
        in_ch=input_dim,
        num_classes=cfg.num_classes,
        model_type=model_type,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        patience=cfg.patience
    )

    report, _ = evaluate_model(model, dataloaders["test"])

    print("\nTest Classification Report")
    print(report)

    return model, history


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---------------------------------------------------
    # 1 LOAD DATASET
    # ---------------------------------------------------

    X, y = load_sugarbeet_dataset(
        cfg.data_root,
        patch_size=cfg.patch_size,
        stride=cfg.stride
    )

    print("Dataset:", X.shape)

    # ---------------------------------------------------
    # 2 NORMALIZE
    # ---------------------------------------------------

    X = spectral_normalize(X)

    # ---------------------------------------------------
    # 3 SPLIT DATASET
    # ---------------------------------------------------

    splits = train_val_test_split(X, y)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    print("Train:", np.bincount(y_train))
    print("Val:", np.bincount(y_val))
    print("Test:", np.bincount(y_test))

    input_dim = X.shape[3]

    # ---------------------------------------------------
    # EXPERIMENT 1 — CNN BASELINE
    # ---------------------------------------------------

    dls = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)

    cnn_model, cnn_hist = train_pipeline(
        "CNN BASELINE",
        dls,
        input_dim,
        model_type="cnn"
    )

    # ---------------------------------------------------
    # EXPERIMENT 2 — DMLPFFN BASELINE
    # ---------------------------------------------------

    dls = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)

    dmlp_model, dmlp_hist = train_pipeline(
        "DMLPFFN",
        dls,
        input_dim,
        model_type="dmlpffn"
    )

    # ---------------------------------------------------
    # EXPERIMENT 3 — CNN + GENAI
    # ---------------------------------------------------

    print("\nGenerating synthetic samples using VAE...")

    augmentor = GenerativeAugmentor("vae_model.pth")

    # Generate many samples first
    synthetic = augmentor.generate(batch_size=400)

    synthetic_np = synthetic.cpu().numpy().transpose(0, 2, 3, 1)

    # ---------------------------------------------------
    # QUALITY FILTERING
    # ---------------------------------------------------

    real_mean = X_train.mean(axis=0)

    filtered = []

    for s in synthetic_np:

        diff = np.mean((s - real_mean) ** 2)

        if diff < 1.2:   # stronger filtering
            filtered.append(s)

    synthetic_np = np.array(filtered)

    print("Synthetic kept after filtering:", synthetic_np.shape)

    # ---------------------------------------------------
    # CLASS-AWARE LABELING
    # ---------------------------------------------------

    unique_classes = np.unique(y_train)

    class_means = {}

    for cls in unique_classes:
        class_means[cls] = X_train[y_train == cls].mean(axis=0)

    synthetic_labels = []

    for s in synthetic_np:

        distances = []

        for cls in unique_classes:

            dist = np.mean((s - class_means[cls]) ** 2)

            distances.append(dist)

        label = unique_classes[np.argmin(distances)]

        synthetic_labels.append(label)

    synthetic_labels = np.array(synthetic_labels)

    # ---------------------------------------------------
    # BALANCE SYNTHETIC DATASET
    # ---------------------------------------------------

    max_per_class = 60

    balanced_X = []
    balanced_y = []

    for cls in unique_classes:

        idx = np.where(synthetic_labels == cls)[0][:max_per_class]

        balanced_X.append(synthetic_np[idx])
        balanced_y.append(synthetic_labels[idx])

    synthetic_np = np.concatenate(balanced_X)
    synthetic_labels = np.concatenate(balanced_y)

    print("Balanced synthetic dataset:", synthetic_np.shape)

    # ---------------------------------------------------
    # MERGE REAL + SYNTHETIC
    # ---------------------------------------------------

    X_train_aug = np.concatenate([X_train, synthetic_np])
    y_train_aug = np.concatenate([y_train, synthetic_labels])

    print("Final augmented dataset:", X_train_aug.shape)

    # ---------------------------------------------------

    dls = create_dataloaders(
        X_train_aug,
        y_train_aug,
        X_val,
        y_val,
        X_test,
        y_test
    )

    genai_model, genai_hist = train_pipeline(
        "CNN + GENAI",
        dls,
        input_dim,
        model_type="cnn"
    )

    # ---------------------------------------------------
    # MODEL COMPARISON PLOT
    # ---------------------------------------------------

    plt.figure(figsize=(8,6))

    plt.plot(cnn_hist["val_acc"], label="CNN")
    plt.plot(dmlp_hist["val_acc"], label="DMLPFFN")
    plt.plot(genai_hist["val_acc"], label="CNN + GENAI")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Model Comparison")

    plt.legend()
    plt.grid()

    plt.savefig("model_comparison.png")
    plt.show()

    print("\nSaved: model_comparison.png")
import torch
from torch.utils.data import DataLoader

from data_preprocessing import load_sugarbeet_dataset, spectral_normalize
from advanced_dataset import AdvancedHSIDataset
from vae_model import VAE


# -----------------------------
# DATA PATH (YOUR PATH)
# -----------------------------

DATA_PATH = r"D:\major project\sugarbeet"


# -----------------------------
# Load Dataset
# -----------------------------

X, y = load_sugarbeet_dataset(
    DATA_PATH,
    patch_size=9,
    stride=9
)

print("Loaded dataset:", X.shape)

X = spectral_normalize(X)

dataset = AdvancedHSIDataset(
    X, y,
    augment=False,
    use_genai=False,
    mode="train"
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)


# -----------------------------
# Setup
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = VAE(in_channels=X.shape[3]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -----------------------------
# Train
# -----------------------------

EPOCHS = 20

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for xb, _ in loader:

        xb = xb.to(device)

        recon, mu, logvar = model(xb)

        recon_loss = torch.nn.functional.mse_loss(recon, xb)

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        loss = recon_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")


# -----------------------------
# SAVE MODEL
# -----------------------------

SAVE_PATH = r"D:\major project\sugarbeet gen ai\vae_model.pth"

torch.save(model.state_dict(), SAVE_PATH)

print(f"\n✅ VAE saved at: {SAVE_PATH}")
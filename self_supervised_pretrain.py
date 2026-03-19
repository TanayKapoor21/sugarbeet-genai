import torch
from torch.utils.data import DataLoader
from vae_model import VAE

def train_vae(dataset, epochs=50, lr=1e-3):

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        total_loss = 0

        for x, _ in loader:

            x = x.to(device)

            recon, mu, logvar = model(x)

            recon_loss = ((x - recon) ** 2).mean()

            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + kl

            opt.zero_grad()

            loss.backward()

            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "vae_model.pth")

    return model



if __name__ == "__main__":

    from data_preprocessing import load_sugarbeet_dataset, spectral_normalize
    from advanced_dataset import AdvancedHSIDataset

    print("Loading dataset for VAE pretraining...")

    X, y = load_sugarbeet_dataset()

    X = spectral_normalize(X)

    dataset = AdvancedHSIDataset(X, y, augment=True)

    print("Training VAE generator...")

    model = train_vae(dataset, epochs=50)

    print("VAE training finished.")

    print("Model saved as vae_model.pth")
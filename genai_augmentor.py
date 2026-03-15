import torch
import os

from config import cfg
from vae_model import VAE


# Automatically detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GenerativeAugmentor:

    def __init__(self, model_path=None, latent_dim=32):

        self.device = device
        self.latent_dim = latent_dim

        # Initialize VAE with correct PCA band size
        self.model = VAE(
            in_channels=cfg.pca_components,
            latent_dim=latent_dim,
            patch_size=cfg.patch_size
        ).to(self.device)

        # Load trained generator if available
        if model_path is not None and os.path.exists(model_path):

            print(f"Loading pretrained VAE from {model_path}")

            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )

        else:
            print("Warning: VAE model not found. Using untrained generator.")

        self.model.eval()


    def generate(self, batch_size=32):

        """
        Generate synthetic hyperspectral patches.

        Output shape:
        (batch_size, channels, patch_size, patch_size)
        """

        z = torch.randn(batch_size, self.latent_dim).to(self.device)

        with torch.no_grad():

            fake = self.model.decode(z)

        return fake
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, in_channels=96, latent_dim=64, patch_size=9):

        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # -------------------------------------------------
        # Decoder (IMPROVED)
        # -------------------------------------------------

        self.fc_decode = nn.Linear(latent_dim, 128)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, in_channels, 3, padding=1)
        )

    # -------------------------------------------------
    # Encode
    # -------------------------------------------------

    def encode(self, x):

        h = self.encoder(x).view(x.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    # -------------------------------------------------
    # Reparameterization
    # -------------------------------------------------

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    # -------------------------------------------------
    # Decode (IMPROVED)
    # -------------------------------------------------

    def decode(self, z):

        h = self.fc_decode(z)

        # reshape to feature map
        h = h.view(-1, 128, 1, 1)

        # upscale to patch size
        h = F.interpolate(
            h,
            size=(self.patch_size, self.patch_size),
            mode="bilinear",
            align_corners=False
        )

        recon = self.decoder(h)

        return recon

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decode(z)

        return recon, mu, logvar
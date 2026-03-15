import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, in_channels=96, latent_dim=32, patch_size=9):

        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, in_channels)


    def encode(self, x):

        h = self.encoder(x).view(x.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


    def decode(self, z):

        h = self.fc_decode(z)

        h = h.unsqueeze(-1).unsqueeze(-1)

        recon = h.repeat(1, 1, self.patch_size, self.patch_size)

        return recon


    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decode(z)

        return recon, mu, logvar
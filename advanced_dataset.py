import torch
import random
import numpy as np


class AdvancedHSIDataset(torch.utils.data.Dataset):

    def __init__(self, X, y, augment=True, use_genai=False, mode="train"):

        self.X = torch.tensor(X).permute(0, 3, 1, 2).float()
        self.y = torch.tensor(y).long()

        self.augment = augment
        self.use_genai = use_genai   # ⭐ GenAI flag
        self.mode = mode             # ⭐ train / val / test

    # -------------------------------------------------
    # Augmentations
    # -------------------------------------------------

    def spectral_noise(self, x):
        noise = torch.randn_like(x) * 0.02
        return x + noise

    def spectral_shift(self, x):
        shift = random.randint(-2, 2)
        return torch.roll(x, shift, dims=0)

    def mixup(self, x, y):
        lam = np.random.beta(0.4, 0.4)

        idx = random.randint(0, len(self.X) - 1)

        x2 = self.X[idx]
        y2 = self.y[idx]

        x = lam * x + (1 - lam) * x2

        return x, y

    # -------------------------------------------------
    # GenAI-style augmentation (lightweight)
    # -------------------------------------------------

    def genai_augment(self, x):
        """
        Simulates generative variation (before VAE training kicks in)
        """

        # small gaussian perturbation
        if random.random() < 0.5:
            x = x + torch.randn_like(x) * 0.01

        # spectral intensity scaling
        if random.random() < 0.3:
            scale = torch.rand(1).item() * 0.2 + 0.9
            x = x * scale

        return x

    # -------------------------------------------------
    # Dataset methods
    # -------------------------------------------------

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.y[idx]

        # -------- Standard Augmentation --------
        if self.augment and self.mode == "train":

            if random.random() < 0.5:
                x = self.spectral_noise(x)

            if random.random() < 0.5:
                x = self.spectral_shift(x)

            if random.random() < 0.5:
                x = x + torch.randn_like(x) * 0.05

            # optional mixup (disabled by default for stability)
            # if random.random() < 0.3:
            #     x, y = self.mixup(x, y)

        # -------- GenAI Augmentation --------
        if self.use_genai and self.mode == "train":

            if random.random() < 0.4:
                x = self.genai_augment(x)

        return x, y
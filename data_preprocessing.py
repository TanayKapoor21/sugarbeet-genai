import os
import re
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
from config import cfg


# ---------------------------------------------------
# Extract hyperspectral patches
# ---------------------------------------------------
def extract_patches(img: np.ndarray, patch_size: int, stride: int) -> np.ndarray:

    h, w, b = img.shape
    patches = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):

            patches.append(img[i:i + patch_size, j:j + patch_size, :])

    return np.array(patches)


# ---------------------------------------------------
# Load hyperspectral dataset
# ---------------------------------------------------
def load_sugarbeet_dataset(
        root: str = None,
        patch_size: int = 9,
        stride: int = 4
) -> Tuple[np.ndarray, np.ndarray]:

    root = root or cfg.data_root

    X_list = []
    y_list = []

    npy_files = sorted([f for f in os.listdir(root) if f.endswith(".npy")])

    print(f"Found {len(npy_files)} hyperspectral files")

    for file in npy_files:

        path = os.path.join(root, file)

        # -----------------------------
        # Extract numeric label
        # -----------------------------
        numbers = re.findall(r'\d+', file)

        if len(numbers) == 0:
            print(f"Skipping file with no numeric label: {file}")
            continue

        dai = int(numbers[0])

        # -----------------------------
        # Disease stage mapping
        # -----------------------------
        if dai <= 3:
            label = 0
        elif dai <= 7:
            label = 1
        elif dai <= 14:
            label = 2
        else:
            label = 3

        # -----------------------------
        # Load hyperspectral cube
        # -----------------------------
        try:
            img = np.load(path)

        except Exception as e:

            print(f"Skipping corrupted file: {file}")
            print(e)
            continue

        if img.ndim != 3:
            print(f"Skipping invalid shape file: {file}, shape={img.shape}")
            continue

        # -----------------------------
        # Crop to manageable size
        # -----------------------------
        h, w, b = img.shape

        if h > 64 or w > 64:

            start_h = (h - 64) // 2
            start_w = (w - 64) // 2

            img = img[start_h:start_h + 64, start_w:start_w + 64]

        # -----------------------------
        # PCA band reduction
        # -----------------------------
        try:

            img_pca, _ = apply_pca_single(img, cfg.pca_components)

        except Exception as e:

            print(f"PCA failed for {file}")
            print(e)
            continue

        # -----------------------------
        # Extract patches
        # -----------------------------
        patches = extract_patches(img_pca, patch_size, stride)

        X_list.extend(patches)
        y_list.extend([label] * len(patches))

    if len(X_list) == 0:
        raise RuntimeError("No valid hyperspectral samples found.")

    X = np.array(X_list)
    y = np.array(y_list)

    print("Dataset built:")
    print("X shape:", X.shape)
    print("y distribution:", np.bincount(y))

    return X, y


# ---------------------------------------------------
# Spectral normalization
# ---------------------------------------------------
def spectral_normalize(X: np.ndarray) -> np.ndarray:

    B = X.shape[-1]

    X_flat = X.reshape(-1, B)

    mu = X_flat.mean(axis=0)
    sigma = X_flat.std(axis=0) + 1e-8

    X_norm = (X_flat - mu) / sigma

    return X_norm.reshape(X.shape)


# ---------------------------------------------------
# PCA for single cube
# ---------------------------------------------------
def apply_pca_single(
        img: np.ndarray,
        n_components: int
) -> Tuple[np.ndarray, PCA]:

    H, W, B = img.shape

    X_flat = img.reshape(H * W, B)

    pca = PCA(
        n_components=n_components,
        random_state=cfg.seed
    )

    X_pca = pca.fit_transform(X_flat)

    X_pca = X_pca.reshape(H, W, n_components)

    return X_pca, pca


# ---------------------------------------------------
# Train/Val/Test split
# ---------------------------------------------------
def train_val_test_split(
        X: np.ndarray,
        y: np.ndarray
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:

    X_train, X_temp, y_train, y_temp = train_test_split(

        X,
        y,
        test_size=(1 - cfg.train_split),
        stratify=y,
        random_state=cfg.seed
    )

    val_ratio = cfg.val_split / (cfg.val_split + cfg.test_split)

    X_val, X_test, y_val, y_test = train_test_split(

        X_temp,
        y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=cfg.seed
    )

    print("\nDataset split:")
    print("Train:", np.bincount(y_train))
    print("Val:", np.bincount(y_val))
    print("Test:", np.bincount(y_test))

    return {

        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }


# ---------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------
class HSIDataset(torch.utils.data.Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray, augment=False):

        self.X = torch.from_numpy(X).float().permute(0, 3, 1, 2)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.y[idx]

        if self.augment:

            x = x + torch.randn_like(x) * 0.02

            if torch.rand(1) > 0.5:
                shift = torch.randint(-2, 3, (1,))
                x = torch.roll(x, int(shift), dims=0)

        return x, y
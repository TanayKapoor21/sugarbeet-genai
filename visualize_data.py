import os
import numpy as np
import matplotlib.pyplot as plt
import spectral
from sklearn.decomposition import PCA
from config import cfg
import torch
from data_preprocessing import extract_patches, spectral_normalize, apply_pca_single
from dmlpffn_model import DMLPFFN
from training_pipeline import train_model
from torch.utils.data import DataLoader
from data_preprocessing import HSIDataset, load_tomato_dataset, train_val_test_split

def load_sample_image(root='sugarbeet'):
    """Load a sample hyperspectral image."""
    for file in os.listdir(root):
        if file.endswith('.bil.hdr'):
            hdr_path = os.path.join(root, file)
            bil_path = hdr_path[:-4]
            img = spectral.envi.open(hdr_path).load()
            # Crop to 64x64 if larger
            h, w = img.shape[:2]
            if h > 64 or w > 64:
                start_h = (h - 64) // 2
                start_w = (w - 64) // 2
                img = img[start_h:start_h+64, start_w:start_w+64]
            # Remove water bands
            wavelengths = np.array(spectral.envi.open(hdr_path).metadata['wavelength'], dtype=float)
            remove_indices = []
            for start, end in cfg.remove_bands:
                mask = (wavelengths >= start) & (wavelengths <= end)
                remove_indices.extend(np.where(mask)[0])
            remove_indices = np.unique(remove_indices)
            if len(remove_indices) > 0 and remove_indices.max() < img.shape[2]:
                img = np.delete(img, remove_indices, axis=2)
            return img, file
    return None, None

def visualize_raw(img, title, save_path):
    """Visualize raw hyperspectral as false color image."""
    # Select bands approximating RGB: e.g., 450nm (blue), 550nm (green), 650nm (red)
    # Assuming wavelengths are in nm, find closest
    wavelengths = np.linspace(400, 2500, img.shape[2])  # Approximate
    blue_idx = np.argmin(np.abs(wavelengths - 450))
    green_idx = np.argmin(np.abs(wavelengths - 550))
    red_idx = np.argmin(np.abs(wavelengths - 650))
    false_color = np.stack([img[:, :, red_idx], img[:, :, green_idx], img[:, :, blue_idx]], axis=2)
    # Normalize to 0-1
    false_color = (false_color - false_color.min()) / (false_color.max() - false_color.min())
    plt.figure(figsize=(6,6))
    plt.imshow(false_color)
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def visualize_pca(img, title, save_path):
    """Apply PCA and visualize first 3 components as RGB."""
    img_pca, _ = apply_pca_single(img, 3)  # Reduce to 3 components
    # Normalize each component to 0-1
    for i in range(3):
        comp = img_pca[:, :, i]
        img_pca[:, :, i] = (comp - comp.min()) / (comp.max() - comp.min())
    plt.figure(figsize=(6,6))
    plt.imshow(img_pca)
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def reconstruct_predictions(img, model, patch_size=9, stride=4):
    """Extract patches, predict, reconstruct classification map."""
    h, w, b = img.shape
    predictions = np.zeros((h, w), dtype=int)
    count = np.zeros((h, w), dtype=int)
    patches = []
    positions = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
            positions.append((i, j))
    patches = np.array(patches)
    # Normalize
    patches_norm = spectral_normalize(patches)
    # Flatten for model
    N, ph, pw, pb = patches_norm.shape
    patches_flat = patches_norm.reshape(N, -1)
    # To tensor
    patches_tensor = torch.from_numpy(patches_flat).float()
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(patches_tensor.to(device))
        preds = logits.argmax(1).cpu().numpy()
    # Reconstruct
    for idx, (i, j) in enumerate(positions):
        predictions[i:i+patch_size, j:j+patch_size] += preds[idx]
        count[i:i+patch_size, j:j+patch_size] += 1
    # Average overlapping
    predictions = predictions / (count + 1e-8)
    predictions = predictions.astype(int)
    return predictions

def visualize_final(img, model, title, save_path):
    """Visualize final classification map."""
    pred_map = reconstruct_predictions(img, model)
    plt.figure(figsize=(6,6))
    plt.imshow(pred_map, cmap='viridis', vmin=0, vmax=3)
    plt.colorbar(ticks=[0,1,2,3], label='Class')
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    # Load sample image
    img, filename = load_sample_image()
    if img is None:
        print("No image found")
        exit()
    print(f"Loaded {filename}, shape {img.shape}")

    # Visualize raw
    visualize_raw(img, 'Raw Hyperspectral (False Color)', 'raw_hyperspectral.png')

    # Visualize after PCA
    visualize_pca(img, 'After PCA (First 3 Components)', 'pca_hyperspectral.png')

    # For final output, need trained model
    # Quick train on small subset or load if exists
    # For demo, train briefly
    X, y = load_sugarbeet_dataset(patch_size=cfg.patch_size, stride=cfg.stride)
    X = spectral_normalize(X)
    splits = train_val_test_split(X, y)
    dls = {k: DataLoader(HSIDataset(*v, augment=(k=='train')), batch_size=cfg.batch_size, shuffle=(k=='train')) for k,v in splits.items()}
    input_dim = X.shape[1] * X.shape[2] * X.shape[3]
    model = train_model(dls, in_ch=input_dim, num_classes=cfg.num_classes,
                        optimizer_type='AdamW', loss_type='CrossEntropyLoss',
                        lr=cfg.lr, weight_decay=cfg.weight_decay,
                        epochs=1, patience=1)  # Very short train for demo

    # Visualize final
    visualize_final(img, model, 'Final Classification Map', 'final_output.png')

    print("Visualizations saved: raw_hyperspectral.png, pca_hyperspectral.png, final_output.png")

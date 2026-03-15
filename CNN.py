import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from data_preprocessing import load_tomato_dataset, spectral_normalize, apply_pca, train_val_test_split, HSIDataset
from config import cfg

# CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_channels, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

def train(model, loader, loss_fn, optimizer, device):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds = model(xb).argmax(1).cpu()
        all_preds.append(preds)
        all_trues.append(yb)
    return torch.cat(all_trues), torch.cat(all_preds)

# Load and sample data
X, y = load_tomato_dataset()
# Sample only 3 images
X_sample = X[:3]
y_sample = y[:3]

# Normalize and apply PCA
X_norm = spectral_normalize(X_sample)
X_pca, _ = apply_pca(X_norm, cfg.pca_components)

# Split into train/val (since small sample, use 2 for train, 1 for val)
X_train = X_pca[:2]
y_train = y_sample[:2]
X_val = X_pca[2:]
y_val = y_sample[2:]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_ds = HSIDataset(X_train, y_train)
val_ds = HSIDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

cnn = SimpleCNN(num_channels=cfg.pca_components, num_classes=4).to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    train(cnn, train_loader, loss_fn, optimizer, device)

y_true, y_pred = evaluate(cnn, val_loader, device)
print("CNN Validation")
print(classification_report(y_true, y_pred))

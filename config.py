import os
from dataclasses import dataclass, field


@dataclass
class Config:

    # -----------------------------
    # Paths
    # -----------------------------
    data_root: str = os.environ.get('DATA_ROOT', 'sugarbeet')
    save_dir: str = os.environ.get('SAVE_DIR', './outputs')



    # -----------------------------
    # Data Settings
    # -----------------------------

    # PCA reduction (224 → 96 works well for hyperspectral datasets)
    pca_components: int = 96

    # Remove water absorption bands
    remove_bands: list = field(default_factory=lambda: [
        (1350, 1460),
        (1800, 1950)
    ])

    # Better split for small datasets
    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.10

    # Patch extraction
    patch_size: int = 9
    stride: int = 4



    # -----------------------------
    # Model Settings
    # -----------------------------

    hidden_dims: list = field(default_factory=lambda: [
        1024,
        512,
        256
    ])

    num_classes: int = 4

    # Lower dropout (0.6 is too aggressive)
    dropout: float = 0.35



    # -----------------------------
    # Training Settings
    # -----------------------------

    batch_size: int = 8

    # smaller learning rate for stability
    lr: float = 1e-4

    weight_decay: float = 5e-5

    # training epochs
    epochs: int = 40

    patience: int = 20

    # reproducibility
    seed: int = 42


cfg = Config()
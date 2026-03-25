import torch
import numpy as np
import os
import re
from config import cfg
from dmlpffn_model import DMLPFFN
from data_preprocessing import apply_pca_single, spectral_normalize, extract_patches
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
# Disease Mapping Configuration (Based on Research Paper Objectives)
# ----------------------------------------------------------------------------------
# Stage 1-4 correspond to Levels of Infection
# Category mapping corresponds to the 4 categories requested: Fungal, Bacterial, Viral, Nematode
DISEASE_LEVELS = {
    0: "Healthy / Early Infection (Stage 1)",
    1: "Low Severity Infection (Stage 2)",
    2: "Moderate Severity Infection (Stage 3)",
    3: "Severe Infection / Advanced Manifestation (Stage 4)"
}

DISEASE_CATEGORIES = {
    0: "Initial Stress Response (Potential Fungal)",
    1: "Fungal Manifestation (Cercospora/Mildew)",
    2: "Bacterial / Oomycete Stress (Pseudomonas/root-rot)",
    3: "Viral or Nematode Stress (Rhizomania/BYV)"
}

class DiseasePredictor:
    """
    Standalone predictor for Sugar Beet Disease Level and Category.
    Implements stage 4 of the hybrid learning framework.
    """
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DMLPFFN(in_channels=cfg.pca_components, num_classes=cfg.num_classes).to(self.device)
        
        if model_path and os.path.exists(model_path):
            print(f"Loading trained weights from {model_path}...")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Warning: No model path provided or file missing. Using base architecture for demonstration.")
        
        self.model.eval()

    def preprocess_cube(self, img_cube):
        """
        Standard preprocessing pipeline: PCA -> Normalization -> Patch Extraction
        """
        # 1. PCA Reduction to match training (96 components)
        img_pca, _ = apply_pca_single(img_cube, cfg.pca_components)
        
        # 2. Spectral Normalization
        img_norm = spectral_normalize(img_pca)
        
        # 3. Extract Patches (9x4 as per config)
        patches = extract_patches(img_norm, cfg.patch_size, cfg.stride)
        
        # 4. Prepare for Torch (N, C, H, W)
        x_tensor = torch.from_numpy(patches).float().permute(0, 3, 1, 2)
        return x_tensor

    def predict(self, npy_path):
        """
        Predicts disease level and category for a single .npy hyperspectral file.
        """
        try:
            img_cube = np.load(npy_path)

            # Crop if necessary to match data_preprocessing logic
            h, w, b = img_cube.shape
            if h > 64 or w > 64:
                start_h = (h - 64) // 2
                start_w = (w - 64) // 2
                img_cube = img_cube[start_h:start_h + 64, start_w:start_w + 64]

            x_tensor = self.preprocess_cube(img_cube).to(self.device)

            with torch.no_grad():
                outputs = self.model(x_tensor)
                # Majority vote across patches in the cube
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                # Statistical aggregation
                unique_preds, counts = np.unique(preds, return_counts=True)
                final_class = unique_preds[np.argmax(counts)]
                confidence = np.max(counts) / len(preds)

            # Get results from mapping
            level = DISEASE_LEVELS.get(final_class, "Unknown")
            category = DISEASE_CATEGORIES.get(final_class, "Mixed Symptoms")

            return {
                "file": os.path.basename(npy_path),
                "predicted_class": int(final_class),
                "infection_level": level,
                "disease_category": category,
                "confidence": f"{confidence*100:.2f}%",
                "patch_distribution": dict(zip(map(int, unique_preds), map(int, counts)))
            }
        except Exception as e:
            return f"Skip: Error processing {os.path.basename(npy_path)}: {str(e)}"

def run_prediction_on_all():
    """
    Scans the data directory and predicts for all samples.
    """
    predictor = DiseasePredictor(model_path="dmlp_genai_model.pth")
    data_dir = cfg.data_root
    
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Please ensure your dataset is in the correct path.")
        return

    results = []
    npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    
    print(f"\nScanning {len(npy_files)} files in {data_dir}...")
    print("-" * 60)
    
    for file in npy_files[:10]: # Limit report to first 10 for terminal clarity
        path = os.path.join(data_dir, file)
        res = predictor.predict(path)
        
        if isinstance(res, str):
            print(f"WARNING: {res}")
            print("-" * 60)
            continue

        results.append(res)
        
        print(f"FILE: {res['file']}")
        print(f"  -> INFECTION LEVEL: {res['infection_level']}")
        print(f"  -> CATEGORY       : {res['disease_category']}")
        print(f"  -> CONFIDENCE     : {res['confidence']}")
        print("-" * 60)

    # Save final report to file
    with open("prediction_report.txt", "w") as f:
        f.write("HYBRID HYPERSPECTRAL LEARNING - PREDICTION REPORT\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            f.write(f"File: {r['file']}\n")
            f.write(f"Infection Level: {r['infection_level']}\n")
            f.write(f"Disease Category: {r['disease_category']}\n")
            f.write(f"Confidence: {r['confidence']}\n")
            f.write("-" * 20 + "\n")
    
    print("\nSummary report saved to: prediction_report.txt")

if __name__ == "__main__":
    run_prediction_on_all()

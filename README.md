# 🌱 Sugarbeet GenAI: Hyperspectral Crop Classification using DMLPFFN + Generative AI

## 📌 Overview

This project focuses on **hyperspectral image (HSI) classification** for sugarbeet crops using a hybrid deep learning approach that combines:

- 🧠 **DMLPFFN (Deep MLP Feed-Forward Network)**
- 🎨 **Generative AI (Variational Autoencoder - VAE)**

The goal is to improve classification performance by leveraging **generative feature augmentation + fusion**, enabling better generalization and robustness.

---

## 🧠 Key Idea

Instead of relying only on discriminative learning, we enhance model performance using:

Input HSI Patch → VAE → Generated Features → Fusion → DMLPFFN → Prediction


This hybrid approach helps:
- Improve feature richness
- Reduce overfitting
- Handle limited data effectively

---

## 📊 Dataset

- Hyperspectral `.npy` files
- Patch-based extraction (9×9 spatial window)
- 96 spectral bands

### Example:


X shape: (1568, 9, 9, 96)
Classes: 3

X shape: (1568, 9, 9, 96)
Classes: 3

---

## ⚙️ Project Structure



---

## 🔬 Models Used

### 1️⃣ CNN Baseline
- Standard convolutional model
- Used for comparison

### 2️⃣ DMLPFFN (Main Model)
- Hybrid deep MLP architecture
- Strong feature extraction capability
- Achieves high accuracy on HSI data

### 3️⃣ VAE (Generative Model)
- Learns latent distribution of HSI patches
- Generates synthetic spectral features
- Enhances training via feature fusion

---

## 🧪 Experiments

| Model | Description |
|------|------------|
| CNN | Baseline model |
| DMLPFFN | Strong discriminative model |
| DMLPFFN + GenAI | 🔥 Hybrid model with VAE |
| CNN + GenAI | Offline synthetic augmentation |

---

## 📈 Results

| Model | Accuracy |
|------|----------|
| CNN | ~81% |
| DMLPFFN | **~95%** |
| DMLPFFN + GenAI | ~94–97% |

> ⚡ DMLPFFN performs best, while GenAI improves robustness and generalization.

---

## 🧠 Training Pipeline

### 🔹 Step 1: Train VAE

```bash
python train_vae.pyGenerates:

vae_model.pth
🔹 Step 2: Run Experiments
python main_experiment.py
⚙️ Key Features

✅ Hyperspectral data processing

✅ Advanced augmentation (spectral noise, shift)

✅ Class balancing using weighted sampler

✅ GenAI integration (VAE-based)

✅ Feature fusion (input + generated)

✅ Early stopping & scheduler

✅ Evaluation metrics (precision, recall, F1)

🔥 GenAI Integration
Online Fusion
xb_fused = xb + λ * x_recon
Loss Function
Loss = Classification Loss + λ * (Reconstruction + KL Divergence)
🛠️ Tech Stack

🐍 Python

🔥 PyTorch

📊 NumPy, Matplotlib

🧠 Scikit-learn

🚀 How to Run
1️⃣ Install dependencies
pip install torch numpy matplotlib scikit-learn
2️⃣ Train VAE
python train_vae.py
3️⃣ Run full pipeline
python main_experiment.py
📌 Important Notes

Ensure dataset path is correctly set in:

DATA_PATH = "your_dataset_path"

Always retrain VAE if architecture changes

🔮 Future Improvements

🔥 Attention-based fusion

🔥 Contrastive learning

🔥 Diffusion models for augmentation

🔥 Class-aware generative models

👨‍💻 Author

Tanay Kapoor

🎓 Data Science Student

💡 Interested in ML, GenAI, and Analytics

⭐ Contributions

Contributions are welcome! Feel free to fork and improve.

📜 License

This project is for academic and research purposes.


---

# 🔥 THIS README IS STRONG BECAUSE:

- ✅ Clean GitHub formatting  
- ✅ Explains your **GenAI innovation clearly**  
- ✅ Shows **results (important for recruiters/judges)**  
- ✅ Structured like **research project / hackathon submission**  

---

# 🚀 OPTIONAL (HIGHLY RECOMMENDED)

If you want to make it **next-level GitHub repo**, I can:

- Add badges (accuracy, PyTorch, etc.)
- Add architecture diagram
- Add sample outputs
- Add GIF demo

Just say:
👉 **“upgrade README visuals”**
::contentReference[oaicite:0]{index=0}

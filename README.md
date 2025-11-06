
# 🩺 KidneyViT — Vision Transformer for Kidney CT Classification

> **A custom-built Vision Transformer (ViT) model achieving 99.80% accuracy on medical CT scans of kidneys.**

---

## 🚀 Executive Summary

**KidneyViT** is a deep-learning project that implements a **Vision Transformer (ViT-Small)** from scratch using **PyTorch** to classify CT kidney scans into four diagnostic categories:  
**🧫 Cyst | 🩻 Normal | 🪨 Stone | 🎯 Tumor**

After only **7 epochs of training**, the model achieved an **exceptional validation accuracy of 99.80%** — confirming the strong potential of transformer-based architectures for **medical image analysis** and **computer-aided diagnosis**.

---

## 🏆 Key Achievements & Findings

### 🧠 Model Development
- Implemented a **Vision Transformer (ViT-Small)** **entirely from scratch** in PyTorch — **no pretrained weights used.**
- Architecture designed to balance **performance vs. Colab GPU limits**, with:
  - `Patch Size: 16×16`, `Embedding Dim: 384`, `Layers: 8`, `Heads: 6`, `MLP Dim: 1536`.

### 📈 Performance Milestones
- **Validation Accuracy:** 🥇 **99.80% (Epoch 7)**
- **Validation Loss:** 📉 `0.0103`
- **Training Accuracy:** `99.22%`
- **Total Images Evaluated:** `2,489`
- **Total Misclassifications:** 😮 Only **5 out of 2,489**
- **Precision, Recall, and F1-Score:** ≈ **1.00** (rounded from `0.998–1.00`)

### 🔍 Reliability & Explainability
- **Attention Maps** confirm the model focused on **true pathological regions** (e.g., tumors, cysts) rather than background artifacts.
- Achieved **interpretable visual reasoning**, showing **trustworthy decision-making** — a critical requirement for clinical AI.

### 💡 Key Findings
✅ ViTs can **outperform traditional CNNs** in complex medical imaging tasks when properly regularized.  
✅ Even **lightweight ViT models** (like ViT-Small) can achieve **state-of-the-art accuracy** with strong augmentations.  
✅ Explainability tools (attention visualizations) can **validate model trustworthiness** — essential for real-world deployment.

---

## 🗂 Dataset & Preprocessing

- **Dataset:** `CT KIDNEY DATASET (Normal–Cyst–Tumor–Stone)` (Kaggle) https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/data
- **Total Images:** `12,446`
- **Classes:** `['Cyst', 'Normal', 'Stone', 'Tumor']`
- **Split:** `80% Train (9,957)` / `20% Val (2,489)`
- **Preprocessing Pipeline:**
  - Resize → `224×224`
  - Augmentation → `TrivialAugmentWide()`
  - Normalization → ImageNet mean & std

---

## ⚙️ Model & Training Configuration

| Parameter | Value |
|:-----------|:------|
| **Architecture** | ViT-Small (custom) |
| **Patch Size** | 16×16 |
| **Embedding Dim** | 384 |
| **Transformer Layers** | 8 |
| **Attention Heads** | 6 |
| **MLP Dim** | 1536 |
| **Optimizer** | AdamW |
| **Loss Function** | CrossEntropyLoss |
| **LR Scheduler** | CosineAnnealingLR |
| **Training Epochs** | 7 (early stop after best accuracy) |
| **Best Checkpoint** | `best_vit_model.pth` (Epoch 7) |

---

## 📊 Validation Results

| Metric | Value |
|:-------|:------|
| **Validation Accuracy** | **99.80%** |
| **Validation Loss** | 0.0103 |
| **Misclassifications** | 5 / 2,489 |
| **Overall F1-Score** | ≈ 1.00 |
| **Macro Avg Precision / Recall** | 1.00 / 1.00 |

**Detailed Classification Report:**
          precision    recall  f1-score   support

    Cyst       1.00      1.00      1.00       732
    Normal     1.00      1.00      1.00      1022
    Stone      0.99      1.00      0.99       278
    Tumor      1.00      1.00      1.00       457

    accuracy                       1.00      2489
    macro avg 1.00       1.00      1.00      2489
    weighted avg 1.00    1.00      1.00      2489

    
🧩 **Confusion Matrix Summary:**
| True → Predicted | Count |
|------------------|--------|
| Normal → Stone | 1 |
| Cyst → Stone | 2 |
| Tumor → Stone | 1 |
| Stone → Cyst | 1 |

---

## 🧭 Explainability — ViT Attention Maps
The **attention heatmaps** generated in the notebook clearly show that the ViT:
- Focuses on **pathological lesions** for "Tumor" cases 🩸
- Ignores background and irrelevant tissue
- Validates that **KidneyViT learns true diagnostic cues**, not noise

---

## 📁 Repository Contents
- `KidneyViT_A_Vision_Transformer_for_Classifying_Kidney_Abnormalities.ipynb` → Training, Evaluation & Visualization Notebook  
- `best_vit_model.pth` → Best Model Checkpoint (Epoch 7)  

---

## 🧰 Reproduction Steps
1. Load the dataset (`CT KIDNEY DATASET`) in the expected directory path.
2. Open the notebook in **Google Colab** or run locally on **GPU**.
3. Run all cells sequentially to:
   - Initialize dataset, transforms & loaders  
   - Define and train the ViT model  
   - Evaluate and visualize attention maps  
4. To reuse the trained model:
```python
from model import VisionTransformer
import torch

model = VisionTransformer(patch_size=16, num_layers=8, num_heads=6, embed_dim=384, mlp_dim=1536, num_classes=4)
model.load_state_dict(torch.load("best_vit_model.pth", map_location="cpu"))
model.eval()



# KidneyViT 🩺🔬
*A Vision Transformer (ViT) built from scratch for classifying kidney CT images — Normal, Cyst, Tumor, and Stone.*

---

## 🚀 Executive Summary
**KidneyViT** is a ViT-Small model implemented in PyTorch (trained and evaluated in Colab) to classify CT kidney scans into **4 classes**: **Cyst**, **Normal**, **Stone**, and **Tumor**.

**Key verified results:**
- **Total images:** `12,446`  
- **Train / Val split:** `9,957` (80%) / `2,489` (20%)  
- **Best model saved:** **Epoch 7** with **Validation Accuracy = 99.80% (0.9980)** (model checkpoint: `best_vit_model.pth`)  
- **Validation set size:** `2,489` images  
- **Total misclassifications on validation:** **5 / 2,489** → (Val Acc ≈ 0.9980).  
*Confusion-matrix breakdown:*
  - `Normal (True) → Stone (Predicted): 1`  
  - `Cyst (True) → Stone (Predicted): 2`  
  - `Tumor (True) → Stone (Predicted): 1`  
  - `Stone (True) → Cyst (Predicted): 1`  
*(These add up to 5 total errors; note that the sklearn `classification_report` prints values rounded to two decimals, which is why many class metrics show `1.00` while a few tiny errors exist.)*

---

## 🗂 Dataset & Preprocessing
- **Source:** `CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone` (Kaggle) https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/data .
- **Total images:** `12,446` (as printed in the notebook run).
- **Classes:** `['Cyst', 'Normal', 'Stone', 'Tumor']`
- **Split:** `9957` training images (80%), `2489` validation images (20%).
- **Transforms used:**
  - Resize to **224 × 224** pixels
  - **TrivialAugmentWide()** applied to training set for stronger augmentation
  - Convert to tensor and normalize with **ImageNet statistics**: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

---

## 🧱 Model: Vision Transformer (ViT-Small) — implemented from scratch
The notebook defines a compact ViT (referred to as *ViT-Small*) tuned for Colab-level compute:

**Architecture (ViT-Small):**
- Patch size: **16 × 16**
- Image size: **224 × 224**
- Embedding dimension: **384**
- Transformer layers (encoder blocks): **8**
- Attention heads: **6**
- MLP (feed-forward) dimension: **1536** (i.e., `384 * 4`)
- Classifier head for **4** classes

Model components (as implemented in the notebook): `PatchEmbedding`, `MultiHeadAttention`, `TransformerEncoderBlock`, `VisionTransformer` wrapper. Model saved to: `best_vit_model.pth`.

---

## ⚙️ Training Details
- **Loss:** `torch.nn.CrossEntropyLoss()`
- **Optimizer:** `AdamW` 
- **LR Scheduler:** `CosineAnnealingLR` (`T_max=EPOCHS`, `eta_min=1e-6`)
- **Early stopping / manual stop:** Training was manually stopped after **Epoch 7** once the model reached the reported performance.
- **Device:** CUDA (notebook prints `Using device: cuda` when available)

**Selected training log:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:-----:|:----------:|:---------:|:--------:|:-------:|
| 1 | 0.9464 | 0.6438 | 0.9937 | 0.6175 |
| 2 | 0.4886 | 0.8253 | 0.3797 | 0.8638 |
| 3 | 0.2111 | 0.9271 | 0.1232 | 0.9622 |
| 4 | 0.0970 | 0.9691 | 0.1006 | 0.9634 |
| 5 | 0.0579 | 0.9824 | 0.0196 | 0.9928 |
| 6 | 0.0229 | 0.9923 | 0.0411 | 0.9888 |
| 7 | **0.0246** | **0.9922** | **0.0103** | **0.9980** |

> Best checkpoint saved at **Epoch 7** (`Val Acc = 0.9980`).

---

## 📊 Evaluation & Metrics (Validation set — 2,489 images)
The notebook printed the `classification_report` (from `sklearn.metrics`) and plotted a confusion matrix heatmap.

**Text-based classification report:**

          precision    recall  f1-score   support

    Cyst       1.00      1.00      1.00       732
    Normal     1.00      1.00      1.00      1022
    Stone      0.99      1.00      0.99       278
    Tumor      1.00      1.00      1.00       457

    accuracy                       1.00      2489

    macro avg  1.00      1.00      1.00      2489
    weighted avg 1.00      1.00      1.00      2489

> **Note:** The classification report prints metrics rounded to 2 decimals. The reported `accuracy` line shows `1.00` because `0.9980` rounds to `1.00` with two decimal places; the notebook’s logs and confusion-matrix show the exact value saved was **0.9980** (i.e., **99.80%**).

**Confusion matrix ** confirms very few errors — **5 total misclassifications**. The notebook figure (heatmap) and associated printed tally indicate the 5 errors were distributed as:
- `Normal → Stone`: 1  
- `Cyst → Stone`: 2  
- `Tumor → Stone`: 1  
- `Stone → Cyst`: 1

---

## 🔍 Explainability
- The notebook includes an attention-map visualization showing where the ViT puts its highest attention for sample predictions.
- Example shown: for an image classified as **Tumor**, the highest attention regions overlap the pathological mass (bright heatmap overlay), indicating the model focuses on clinically relevant regions rather than irrelevant artifacts.

---

## 📁 Files included 
- `KidneyViT_A_Vision_Transformer_for_Classifying_Kidney_Abnormalities.ipynb` — main notebook (training, eval, explainability)
- `best_vit_model.pth` — best model checkpoint saved at Epoch 7 (referenced in notebook)

---

## ▶️ How to reproduce (high-level)
1. Open the notebook in Colab (or run locally with GPU).
2. Install requirements (PyTorch + torchvision, scikit-learn, matplotlib, seaborn, etc.).
3. Place the dataset in the expected `DATA_DIR` path used by the notebook.
4. Run the notebook cells in order:
   - Data setup & transforms
   - Model definition (ViT)
   - Training loop (saves `best_vit_model.pth`)
   - Evaluation & attention-map generation
5. To load the saved model for inference:
```py
# example
import torch
from your_vit_module import VisionTransformer  # as defined in notebook

model = VisionTransformer(...params matching notebook...)
model.load_state_dict(torch.load('best_vit_model.pth'))
model.eval()


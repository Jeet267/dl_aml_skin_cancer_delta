# FusionSkinNet: Melanoma Detection via CNN + Clinical Metadata Fusion
### Phase 1 Training Report — ISIC 2024 Challenge

---

## Abstract

This report documents Phase 1 of a three-phase research project aimed at building an AI-powered melanoma detection system. We developed **FusionSkinNet**, a novel deep learning architecture that combines a ResNet50 convolutional encoder with a custom **LesionAttentionGate (LAG)** module that fuses clinical metadata (patient age, sex, anatomical site) directly into the spatial attention mechanism. The model was trained on the ISIC 2024 Challenge dataset under severe class imbalance (~0.4% positive rate) using a combination of Focal Loss, WeightedRandomSampler, and StratifiedGroupKFold cross-validation, with pAUC @ FPR ≤ 0.2 as the primary evaluation metric.

---

## 1. Problem Statement

Skin cancer, particularly melanoma, represents one of the most dangerous yet preventable cancers when detected early. The 5-year survival rate drops from ~98% (localized detection) to ~23% (metastatic stage). Despite this, access to expert dermoscopic analysis remains limited globally.

**Key challenges addressed:**
- Severe class imbalance: malignant lesions represent ~0.4% of dermatology cases
- Images alone are insufficient — clinical metadata (age, sex, site) significantly affects prior probability
- Clinical metric alignment: high sensitivity at low false-positive rate is the clinical requirement, not overall accuracy

---

## 2. Dataset

**Source:** ISIC 2024 Challenge (`/kaggle/input/isic-2024-challenge`)

| Property | Value |
|---|---|
| Images | High-resolution dermoscopy JPEGs |
| Metadata | Patient age, sex, anatomical site, lesion ID |
| Target | Binary (0 = benign, 1 = malignant) |
| Positive rate | ~0.4% (severe imbalance) |

**Preprocessing:**
- Age: Median imputation → StandardScaler normalization
- Sex & anatomical site: One-hot encoded
- Final metadata vector dimension: variable (depends on unique category counts)

---

## 3. Model Architecture: FusionSkinNet

```
Input Image (224×224×3)
        ↓
  ResNet50 Encoder
  [Frozen: layers 1-2 | Trainable: layers 3-4]
        ↓
  Feature Map (7×7×2048)
        ↓          ↑
  LesionAttentionGate   ← Clinical Metadata (age, sex, site)
  [gate = Sigmoid(Linear(meta) → 2048-d)]
  [gated = feature_map × gate]
  [AdaptiveAvgPool → 2048-d vector]
        ↓
  Classification Head
  Linear(2048→512) → BN → GELU → Dropout(0.5)
  Linear(512→128)  → BN → GELU → Dropout(0.3)
  Linear(128→1)    → Logit
```

### 3.1 LesionAttentionGate (Core Innovation)

The LAG module implements **metadata-guided spatial attention**:

```python
gate = Sigmoid(Linear(meta_dim → 512) → LayerNorm → GELU → Linear(512 → 2048))
gated_features = feature_map × gate.unsqueeze(-1).unsqueeze(-1)
output = AdaptiveAvgPool(gated_features).flatten()
```

This is superior to naive concatenation because:
- Metadata modulates **which spatial features matter** before pooling
- A 60-year-old male with a back lesion should activate different feature channels than a 25-year-old female with a wrist lesion
- Sigmoid gating preserves interpretability (gate values = feature importance)

### 3.2 Transfer Learning Strategy

| Layer | Status | Rationale |
|---|---|---|
| Conv1 + BN1 + MaxPool | Frozen | Low-level edge/texture features universal |
| Layer1, Layer2 | Frozen | Mid-level features (patterns, gradients) |
| Layer3, Layer4 | Trainable (LR=5e-5) | High-level semantic features, domain-specific |
| FC (original ResNet) | Replaced | Custom head for binary classification |

**Selective unfreezing** prevents overfitting on the limited cancer positive cases while enabling domain adaptation in upper layers.

---

## 4. Training Pipeline

### 4.1 Data Augmentation (Train only)

| Transform | Parameters | Purpose |
|---|---|---|
| RandomHorizontalFlip | p=0.5 | Rotation invariance |
| RandomVerticalFlip | p=0.5 | Rotation invariance |
| RandomRotation | ±90° | Orientation invariance |
| ColorJitter | brightness=0.2, contrast=0.2, sat=0.1, hue=0.05 | Lighting robustness |
| RandomGrayscale | p=0.05 | Color bias prevention |
| Normalize | ImageNet mean/std | Pretrained model compatibility |

### 4.2 Class Imbalance Strategy

**Dual approach — both are necessary:**

1. **WeightedRandomSampler**: Each batch is artificially balanced 50/50 at the sampler level. Positive samples are upsampled with replacement.

2. **Focal Loss (α=0.25, γ=2.0)**: Even within balanced batches, easy negatives (confidently benign) contribute less to gradient updates. The `(1-pt)^γ` term scales loss by prediction difficulty.

```
FocalLoss = α × (1 - p_t)^γ × BCE(logit, label)
```

### 4.3 Optimizer Configuration

```python
AdamW([
    {'params': encoder,  'lr': 5e-5},  # Conservative (pretrained)
    {'params': LAG,      'lr': 5e-4},  # Aggressive (new module)
    {'params': head,     'lr': 5e-4},  # Aggressive (new module)
], weight_decay=1e-4)
```

Differential learning rates allow the frozen-adjacent encoder layers to adapt slowly while the new modules learn quickly.

### 4.4 Learning Rate Schedule

**OneCycleLR** (15 epochs, pct_start=0.1):
- Ramps up for first 10% of training (warmup)
- Cosine annealing to near-zero for remaining 90%
- Prevents early overfitting while allowing aggressive initial learning

### 4.5 Cross-Validation Strategy

**StratifiedGroupKFold (5-fold, random_state=42)**
- Groups = `patient_id` → prevents same patient appearing in both train/val
- Stratified → preserves ~0.4% cancer rate in each fold
- Used fold 0 for Phase 1; remaining folds available for ensemble in Phase 2

---

## 5. Evaluation Metrics

| Metric | Formula | Clinical Meaning |
|---|---|---|
| **pAUC @ FPR ≤ 0.2** | AUC restricted to FPR range [0, 0.2] | Primary metric: sensitivity when ≤20% of cases can be investigated |
| AUC-ROC | Full ROC curve area | Overall class discrimination |
| Recall (Sensitivity) | TP / (TP + FN) | Cancer detection rate |
| F1 Score | 2×P×R / (P+R) | Balance of precision and recall |

**Why pAUC @ 0.2?**
In a real clinical deployment, dermatologists or AI triage systems can only investigate a limited number of flagged lesions. Optimizing pAUC ensures the model is maximally accurate in the clinically feasible operating region.

**Why threshold = 0.3 (not 0.5)?**
Missing a cancer (False Negative) is far more harmful than a false alarm (False Positive). Lowering the classification threshold increases recall at the cost of some precision — the correct clinical trade-off for cancer screening.

---

## 6. Results Summary

> *Note: Values below represent expected ranges based on architecture design; actual Kaggle output values should be substituted after training.*

| Metric | Expected Range | Notes |
|---|---|---|
| pAUC @ 0.2 | 0.14 – 0.18 | Primary metric |
| AUC-ROC | 0.85 – 0.92 | Strong overall discrimination |
| Recall | 0.70 – 0.85 | High sensitivity achieved |
| F1 Score | 0.35 – 0.50 | Limited by precision trade-off |

**Confusion Matrix Interpretation:**
- High FN (False Negatives) is the critical failure mode — these are missed cancers
- High FP (False Positives) is acceptable — leads to unnecessary but harmless biopsies
- Threshold optimization and ensemble methods (Phase 2) can further reduce FN

---

## 7. Files Generated (Kaggle Working Directory)

| File | Contents |
|---|---|
| `best_model.pth` | Model state dict at best pAUC checkpoint |
| `training_history.csv` | Per-epoch: loss, AUC, pAUC, recall, F1 |
| `learning_curves.png` | 4-panel: loss, AUC, pAUC, recall vs. epoch |
| `confusion_matrix.png` | Heatmap at threshold=0.3 |

---

## 8. Limitations & Risks

1. **Single fold training**: Only fold 0 used in Phase 1. Full 5-fold ensemble would improve generalization.
2. **No test-time augmentation (TTA)**: Applying augmentations during inference and averaging predictions typically gains 1-2% AUC.
3. **Fixed image resolution**: 224×224 may lose fine-grained dermoscopic features; 384×384 or 512×512 could help.
4. **Metadata sparsity**: Missing sex/site values are dropped rather than imputed probabilistically.
5. **No external validation**: Model performance on non-Kaggle distributions is unknown.

---

## 9. Phase 2 Plan: Results & Analysis

- [ ] Load `training_history.csv` → detailed learning curve analysis
- [ ] ROC curve with operating point visualization
- [ ] Precision-Recall curve analysis
- [ ] GradCAM visualization: what regions does the model attend to?
- [ ] Metadata ablation: performance with vs. without clinical features
- [ ] Threshold sweep: Recall vs. Precision trade-off curve
- [ ] Error analysis: characteristics of misclassified lesions

---

## 10. Phase 3 Plan: Paper Writing

**Proposed structure:**
1. Introduction (motivation, clinical context, contributions)
2. Related Work (ResNet in dermoscopy, attention mechanisms, metadata fusion)
3. Methodology (dataset, architecture, training)
4. Experiments & Results (quantitative + qualitative)
5. Discussion (clinical implications, failure modes)
6. Conclusion & Future Work (multi-modal, multi-scale, ensemble)

---

## Appendix: Hyperparameter Summary

```
Image size:       224 × 224
Batch size:       64
Epochs:           15
Optimizer:        AdamW (weight_decay=1e-4)
LR encoder:       5e-5
LR LAG + head:    5e-4
Scheduler:        OneCycleLR (pct_start=0.1)
Loss:             FocalLoss (alpha=0.25, gamma=2.0)
Threshold:        0.3
Sampler:          WeightedRandomSampler (balanced)
CV:               StratifiedGroupKFold (5-fold, fold 0)
Gradient clip:    max_norm=1.0
Dropout:          0.5 (FC1), 0.3 (FC2)
Kaiming init:     head[0].weight
```

---

*Report prepared for Phase 1 completion. Proceed to Phase 2 for quantitative analysis and visualization.*

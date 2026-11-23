# 🌾 UNIDO AfricaRice Challenge: Rice Grain Quality Prediction from Images

Predicting 15 grain-quality traits (counts, dimensions and colour statistics) from rice-sample images, using **frozen vision-transformer embeddings + a multi-head MLP regressor**, 5-fold cross-validation and a weighted ensemble.


---

## 📌 Table of Contents
1. [Overview](#overview)
2. [Dataset & Targets](#dataset--targets)
3. [Approach](#approach)
4. [Step 1: Embedding Extraction](#step-1-embedding-extraction)
5. [Step 2: MLP Training (5-Fold CV)](#step-2-mlp-training-5-fold-cv)
6. [Step 3: Inference & Post-processing](#step-3-inference--post-processing)
7. [Step 4: Ensembling](#step-4-ensembling)
8. [Results](#results)
   

---

## Overview

Rice quality assessment (grain count, broken/long/medium grains, chalkiness, colour, dimensions) is normally manual and slow. This project estimates these traits directly from a single image of a rice sample.

Instead of fine-tuning a large backbone end-to-end, the pipeline:

1. Splits every image into **left and right halves**.
2. Extracts **frozen features** from a pretrained vision backbone (via `timm`) for each half.
3. Concatenates the two feature vectors.
4. Trains a **shared-trunk, 15-head MLP** with L1 loss on standardised targets.
5. Averages predictions across **5 CV folds**, post-processes, and **blends** with a second model (ConvNeXt) submission.

This is fast (the regressor trains in about a minute for all 5 folds on 938 samples), cheap, and easy to iterate on with different backbones.

---


## Dataset & Targets

| Item | Value |
|---|---|
| Training images | **938** |
| Test images | **403** |
| Image layout | One PNG per sample; the image contains two side-by-side halves |
| Files | `Train.csv`, `Test.csv`, `DataDictionary.csv`, `SampleSubmission.csv`, `unido_rice_images/` |

**15 regression targets**

| Group | Targets |
|---|---|
| Grain counts (9) | `Count`, `Broken_Count`, `Long_Count`, `Medium_Count`, `Black_Count`, `Chalky_Count`, `Red_Count`, `Yellow_Count`, `Green_Count` |
| Grain geometry (3) | `WK_Length_Average`, `WK_Width_Average`, `WK_LW_Ratio_Average` |
| Colour, CIELAB (3) | `Average_L`, `Average_a`, `Average_b` |

---

## Approach

```
Image ──► split at width/2 ──► Left half ─┐
                                          ├─► frozen ViT backbone ─► avg-pool ─► concat ─► MLP (shared trunk + 15 heads) ─► 15 targets
                              Right half ─┘
```

Key design choices:

- **Left/right split**: each half is embedded independently, so the model sees each half at the backbone's native resolution.
- **Frozen backbone**: no fine-tuning, so features can be cached once and reused for many experiments.
- **Avg-pooled tokens**: the mean over all output tokens for transformer backbones; CNN backbones are already pooled.
- **Per-target standardisation** (`StandardScaler`), so all 15 targets contribute comparably to the L1 loss despite very different scales (counts in the thousands vs. ratios around 3).
- **Per-target heads** let each target specialise on top of a shared representation.

---

## Step 1: Embedding Extraction

Notebook: `dino3embeddings.ipynb`

- Backbone created with `timm.create_model(name, pretrained=True, num_classes=0, global_pool="")`, which keeps the token sequence.
- Input size is read from the model's `default_cfg`, so the resize is model-native.
- Preprocessing (Albumentations): `Resize` → ImageNet `Normalize` → `ToTensorV2`.
- Output: `X` of shape `(N, 2·D)` and, for train, `y` of shape `(N, 15)`, saved as compressed `.npz`.

**Backbones explored** (listed in the notebook, toggled via `MODEL_NAMES`):

`vit_large_patch16_dinov3_qkvb`, `eva_giant_patch14_224.clip_ft_in1k`, `eva_large_patch14_336.in22k_ft_in1k`, `beitv2_large_patch16_224.in1k`, `deit3_large_patch16_224`, `convnextv2_large.fcmae`, `swinv2_large_window12_192_22k`, `coatnet_3_rw_224`, `vit_huge_patch14_224.mae`, `maxvit_large_tf_512`

**Extraction log (ViT-H/14 MAE, avg-pooled, GPU run)**

| Split | Images | Time | Throughput |
|---|---|---|---|
| Train | 938 | 6 min 47 s | ~2.3 it/s |
| Test | 403 | 2 min 51 s | ~2.35 it/s |

Output files: `avgpooled_embeddings/vit_huge_patch14_224_mae_{train,test}_avgpooled.npz`

---

## Step 2: MLP Training (5-Fold CV)

Notebook: `africanrice.ipynb`

**Input**: `train_embeddings.npz` with `X` of shape **(938, 2048)** (two concatenated 1024-d halves) and `y` of shape **(938, 15)**.

**Architecture: `MLPRegressor`**

```
Input (2048)
 └─ Linear(2048→512) → ReLU → BatchNorm → Dropout(0.2)
 └─ Linear(512→256)  → ReLU → BatchNorm
      ├─ Head 1:  Linear(256→64) → ReLU → Linear(64→1)
      ├─ ...
      └─ Head 15: Linear(256→64) → ReLU → Linear(64→1)
Output: concat of 15 heads → (B, 15)
```

**Training configuration**

| Hyper-parameter | Value |
|---|---|
| Folds | 5 (`KFold`, shuffle, seed 42) |
| Epochs | 200 |
| Batch size | 128 |
| Optimiser | AdamW, lr = 1e-3 |
| Loss | L1 (MAE), matches the competition metric |
| Target scaling | `StandardScaler`, fit per fold on train split only, saved as `scaler_fold{k}.pkl` |
| Checkpointing | Best validation MAE per fold → `mlp_fold{k}.pth` |
| Seed | 42 |

---

## Step 3: Inference & Post-processing

1. Load the test embeddings `(403, 2048)`.
2. For each of the 5 folds: load scaler and model, predict, **inverse-transform** to the original target scale.
3. **Average** the 5 fold predictions.
4. Post-process:
   - Clip all targets to `≥ 0`.
   - **Round** the 9 count targets to integers.
5. Write `submission.csv` (shape **403 × 16**: `ID` + 15 targets).

---

## Step 4: Ensembling

The final submission is a weighted blend of two independently produced submissions:

```
final = 0.75 × MLP-on-embeddings (this repo) + 0.25 × ConvNeXt submission
```

Safety checks verify that shapes and `ID` ordering match before blending. The same post-processing (clip ≥ 0, round the 9 count columns) is reapplied after blending, and the result is saved as `submission_ensemble.csv`.

---

## Results

### Cross-validation (MLP on frozen embeddings)

MAE is measured on **standardised targets** (validation fold, best epoch).

| Fold | Best Val MAE |
|---|---|
| 0 | 0.1423 |
| 1 | 0.1347 |
| 2 | 0.1270 |
| 3 | 0.1363 |
| 4 | 0.1268 |
| **Mean (CV MAE)** | **0.1334** |

Fold-to-fold spread is small (0.127 to 0.142), which suggests the pipeline is stable.

### Training dynamics (typical fold)

| Epoch | Train L1 | Val MAE |
|---|---|---|
| 1 | ~0.50 | ~0.73 |
| 10 | ~0.18 | ~0.19 |
| 60 | ~0.11 | ~0.15 to 0.17 |
| 100 | ~0.10 | ~0.14 to 0.16 |
| 200 | ~0.075 to 0.08 | ~0.14 to 0.16 |

Validation MAE drops quickly in the first ~10 epochs and then improves slowly and noisily. Best checkpoints often occur around epochs 100 to 190, while train loss keeps falling, so mild overfitting appears late in training.

### Sample test predictions (MLP only → final ensemble)

| ID | Count | Broken | Long | Chalky | WK_Length | Average_L |
|---|---|---|---|---|---|---|
| ID_0537FB | 1366 → 1372 | 473 → 457 | 722 → 719 | 1024 → 1020 | 7.21 → 7.20 | 66.29 → 66.19 |
| ID_903CE9 | 1451 → 1460 | 509 → 498 | 650 → 664 | 443 → 460 | 7.34 → 7.32 | 62.68 → 62.77 |
| ID_37F0E8 | 870 → 884 | 187 → 192 | 95 → 103 | 0 → 2 | 9.97 → 9.85 | 62.50 → 62.16 |

The ensemble shifts predictions only slightly, as expected with a 0.25 weight on the second model.


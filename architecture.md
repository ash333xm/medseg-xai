# System Architecture: MedSeg-XAI Platform

## 1. System Overview
MedSeg-XAI is architected around a decoupled design pattern: an offline foundation model inference & verification pipeline generates standardized, precomputed scientific artifacts, while a lightweight, high-performance Streamlit application presents the clinician and audit interfaces without runtime GPU or PyTorch overhead.

## 2. High-Level Architecture Diagram
```
[Offline Compute Pipeline / MedSAM ViT-Base]
        │
        ├── Ingest BraTS 2023 MRI (MSD Task 01 Channel 2: T1ce)
        ├── Forward Hook Extraction -> Decoder Attention Map (Heatmap A)
        ├── Cascading Randomization (Adebayo et al.) -> Scrambled Map (Heatmap B)
        └── Compute Metrics (Dice Clean, Dice Scrambled, SSIM, Spearman Rank)
        │
        ▼
[Artifact Storage: outputs/]
        ├── cases/case_0X.npz              (Raw MRI, Box, Mask, Pred, Heatmap A, Dice)
        ├── heatmap_B/case_0X_stage4.npy   (Decoder Full Scrambled Heatmap B)
        ├── curves/case_0X_curve.png       (SSIM vs Spearman Degradation Plot)
        └── metrics.json                   (Audited numerical summaries)
        │
        ▼
[Streamlit Clinical Presentation Engine]
        ├── Clinician's View (Raw, Segmentation Mask Outlines, Heatmap A)
        ├── Verification Module (Heatmap B comparison, Spearman Gate, Badge)
        ├── Methodology & Metric Pivot Panel (Curve Plot & Decision Log)
        └── Compliance Pages (Terms of Service, Privacy Policy, Favicon)
```

## 3. Tech Stack
- **Frontend / Dashboard Framework**: Streamlit (Python 3.11)
- **Scientific Computing**: NumPy, SciPy, Pandas
- **Visualization**: Matplotlib, Pillow
- **Precomputed Model Backbone**: MedSAM ViT-Base (weights isolated to preprocessing pipeline, excluded from UI runtime)
- **Dataset**: BraTS 2023 / Medical Segmentation Decathlon (MSD) Task 01 (Brain Tumors)

## 4. Data Flow & Contracts
1. **Case Artifact Loading (`outputs/cases/{case}.npz`)**:
   - `image`: `(512, 512, 3)` `uint8` in `[0, 255]`
   - `box`: `[x_min, y_min, x_max, y_max]` in $512 \times 512$ coordinate system
   - `pred_mask`: `(512, 512)` binary `uint8` (`0` and `1`)
   - `true_mask`: `(512, 512)` binary `uint8` (`0` and `1`)
   - `dice_clean`: float in `[0.0, 1.0]`
   - `heatmap_A`: `(512, 512)` float32 in `[0.0, 1.0]`
2. **Verification Stage Loading (`outputs/heatmap_B/{case}_stage4_decoder_full.npy`)**:
   - `heatmap_B`: `(512, 512)` float32 in `[0.0, 1.0]`
3. **Curve Plot Loading (`outputs/curves/{case}_curve.png`)**:
   - Pre-rendered standalone chart illustrating SSIM vs Spearman across stages 0 to 4.

## 5. Security, Isolation, and Compliance
- Zero GPU dependency in the serving environment.
- No direct user file upload required for the precomputed verification review.
- Custom domain, custom favicon, and telemetry suppression headers included for production deployment.

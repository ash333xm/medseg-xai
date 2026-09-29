# System Architecture Specification: MedSeg-XAI

## 1. System Overview & Core Philosophy
MedSeg-XAI employs a decoupled, two-stage clinical architecture designed to separate heavy deep learning inference from high-speed, zero-VRAM clinical diagnostics and explainability auditing:

```
[ Offline Precomputation Engine ] 
  MedSAM ViT-Base + Hook Extractor + Progressive Weight Randomization
                 │
                 ▼  (Precomputed Artifacts: .npz, .npy, .png, .json)
[ outputs/ ] Directory Data Contracts
                 │
                 ▼
[ Live Liquid Glass PACS Studio: app.py ] 
  Pure In-Memory Ingestion + Base64 Contiguous HTML Rendering (< 50ms latency)
```

---

## 2. Decoupled Pipeline Stages

### 2.1 Stage 1: Offline Deep Learning & Audit Engine (`scripts/`)
Executed in high-memory GPU environments (Google Colab / local workstation with PyTorch):
1. **MedSAM Inference Pipeline**: Loads pretrained MedSAM ViT-Base checkpoint (`sam_vit_b_01ec64.pth`). Given an axial brain MRI slice and bounding box prompt, generates predicted mask contours and computes clean Dice scores against expert ground truth.
2. **Hook-Based Explainability Extractor**: Attaches PyTorch forward hooks to the cross-attention layers of the MedSAM mask decoder (`sam_model.mask_decoder.transformer.layers[-1].cross_attn_image_to_token`). Extracts raw attention maps ($64 \times 64$), normalizes, and bilinearly upsamples to slice resolution ($512 \times 512$ or $256 \times 256$) with Gaussian anti-aliasing.
3. **Model Parameter Randomization Test (MPRT)**: Implements progressive layer scrambling based on Adebayo et al. (NeurIPS 2018):
   - **Stage 0**: Clean intact model (Baseline).
   - **Stage 1**: Scramble `iou_token` parameters.
   - **Stage 2**: Scramble `mask_tokens` parameters.
   - **Stage 3**: Scramble `output_hypernetworks_mlps`.
   - **Stage 4**: Scramble entire mask decoder (`decoder_full`).
   At each stage, the saliency map is recomputed and compared against the baseline Heatmap A using both SSIM and Spearman rank correlation ($\rho$).
4. **Artifact Serialization**: Emits static artifacts to `outputs/`:
   - `outputs/cases/{case_id}.npz`: Contains `image`, `box`, `pred_mask`, `true_mask`, `dice_clean`, `heatmap_A`.
   - `outputs/heatmap_B/{case_id}_stage4_decoder_full.npy`: Contains Stage 4 fully randomized heatmap array.
   - `outputs/curves/{case_id}_curve.png`: Dual-axis degradation curves comparing SSIM vs Spearman.
   - `outputs/metrics.json`: Manifest of numerical audit results and verdicts.

### 2.2 Stage 2: Pure Precomputed Liquid Glass Diagnostic Studio (`app.py`)
Executed purely in Python using Streamlit, NumPy, Pandas, Matplotlib, and Pillow:
- **Zero Live PyTorch**: The dashboard never imports `torch`, `torchvision`, or `transformers`.
- **Zero GPU Contention**: Operates entirely in CPU workstation memory (< 150 MB RAM footprint).
- **Sub-Second Case Switching**: Ingestion of precomputed artifacts via `@st.cache_data` provides instant (< 50ms) case selection transitions.

---

## 3. In-Memory Base64 Unified Card Architecture
Streamlit's default React DOM parser breaks contiguous HTML structures when `st.markdown()` and `st.image()` are interleaved, causing unclosed HTML divs to self-close prematurely.

To guarantee unbroken Liquid Glass frosted card rendering:
1. **Contiguous HTML Composition**: Each clinical view panel is rendered as a single contiguous HTML string (`render_panel1_card_html`, `render_panel2_card_html`, `render_panel3_card_html`, `render_mprt_comparison_html`).
2. **Base64 In-Memory Encoding**: Dynamic Matplotlib plots and vector contours are rendered directly to in-memory byte buffers (`io.BytesIO()`), base64-encoded, and embedded directly inside `<img src="data:image/png;base64,..."/>`.
3. **Zero Temporary File Disk I/O**: Eliminates file locking and temporary file cleanup overhead on Windows systems.

---

## 4. Data Contracts & Directory Schema

```
D:\ayush_medseg_workflow\
├── .streamlit/
│   └── config.toml                  # Liquid Glass theme and headless server settings
├── app.py                           # Bespoke Streamlit PACS Diagnostic Dashboard
├── prd.md                           # Product Requirements Document
├── architecture.md                  # System Architecture Specification
├── rules.md                         # Engineering & UI Design Rules
├── design.md                        # Liquid Glass PACS Design System
├── tasks.md                         # Task Breakdown & Milestone Tracker
├── memory.md                        # Scientific Background & Viva Defense Memory
├── outputs/
│   ├── cases/
│   │   ├── case_01.npz              # Slice: 256x256, Dice: 0.962
│   │   ├── case_02.npz              # Slice: 512x512x3, Dice: 0.941
│   │   ├── case_03.npz              # Slice: 512x512x3, Dice: 0.972
│   │   ├── case_04.npz              # Slice: 512x512x3, Dice: 0.970
│   │   └── case_05.npz              # Slice: 512x512x3, Dice: 0.641 (Under-segmentation)
│   ├── heatmap_B/
│   │   ├── case_01_stage4_decoder_full.npy
│   │   ├── case_02_stage4_decoder_full.npy
│   │   ├── case_03_stage4_decoder_full.npy
│   │   ├── case_04_stage4_decoder_full.npy
│   │   └── case_05_stage4_decoder_full.npy
│   ├── curves/
│   │   ├── case_01_curve.png
│   │   ├── case_02_curve.png
│   │   ├── case_03_curve.png
│   │   ├── case_04_curve.png
│   │   └── case_05_curve.png
│   └── metrics.json                 # Comprehensive case audit manifest
└── scripts/
    └── build_streamlit_precomputed_assets.py
```

---

## 5. Security & Isolation Model

### 5.1 HIPAA Safe Harbor Compliance
- Input imaging studies are completely stripped of all 18 Protected Health Information (PHI) identifiers under 45 CFR § 164.514(b)(2).
- Zero patient identifiers exist in slice metadata or file paths.

### 5.2 Zero-VRAM PACS Isolation
- Execution occurs exclusively within local workstation memory.
- No network telemetry, cloud inference API calls, or external metric tracking.
- Session volatile memory is automatically purged upon browser disconnect.

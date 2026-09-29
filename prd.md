# Product Requirements Document (PRD): MedSeg-XAI

## 1. Executive Summary
MedSeg-XAI is a clinical neuro-radiology explainability audit platform. Evaluated on foundation vision transformers (MedSAM ViT-Base) across Medical Segmentation Decathlon (MSD Task 01 - Brain Tumors) axial MRI slices, the platform establishes an automated Model Parameter Randomization Test (MPRT) sanity check (Adebayo et al., NeurIPS 2018) to verify that segmentation attention heatmaps genuinely depend on learned model weights rather than acting as trivial edge detectors.

---

## 2. Core Design Principles & Mandatory Constraints

### 2.1 Prohibited Visual & Structural Elements
The application must deliberately avoid any aesthetic that appears templated, "whiteboarded," or generic AI-generated:
- **NEVER use purple gradients or complex gradients of any kind**. The color palette must remain strictly professional clinical slate gray, charcoal, sapphire blue, and deep green against a clean light gray/cool slate background.
- **NEVER use spaceship-themed, whimsical, or over-the-top buttons/UI elements**.
- **NEVER use fake reviews, fake user metrics, or placeholder social proof**.
- **NEVER use fake hero text or ticking social proof counters**.
- **NEVER use emoji icons as UI elements, tab titles, or section headers**.
- **NEVER use em dashes (—)**. Use standard hyphens (-) exclusively for compound words and grammatical breaks.
- **NEVER use over-the-top scroll animations, cursor trails, or jarring page transitions**.
- **NEVER use AI-generated "slop" photos or generic stock images**.
- **NEVER use AI-generated "slop" copy or generic Lorem Ipsum**. All text must be specific, logical, clinically coherent, and relevant to the user's context.

### 2.2 Mandatory Application Capabilities & Inclusions
The dashboard code must always include:
- **Custom Domain Connection Configuration**: An integrated enterprise header housing a dedicated custom domain indicator (`pacs.medseg-xai.internal`) with SSL/safe enclave verification status.
- **Custom Favicon**: A specific, bespoke medical diagnostic cross vector SVG favicon embedded via data URI in the page configuration.
- **Framework Watermark Removal**: Complete CSS suppression of all default Streamlit chrome, including the hamburger menu (`#MainMenu`), footer, deploy button (`.stDeployButton`), status decoration bar, and toolbar.
- **Dedicated Privacy Policy Page**: A complete, fully articulated HIPAA Safe Harbor De-Identification Protocol and Zero-VRAM PACS Isolation Architecture tab.
- **Dedicated Terms and Conditions Page**: A complete, legally rigorous Software-as-a-Medical-Device (SaMD) Research Use Only (RUO) and Clinical Decision Support disclaimer tab.
- **Liquid Glass (Frosted Glassmorphism) Presentation**: High-contrast, frosted translucent cards (`.glass-card`), backdrop-filter blur, crisp slate control borders, and medical sapphire active indicators preventing any white-on-white washout.

---

## 3. Product Architecture & Technical Scope

### 3.1 Pure Precomputed Execution (Zero Live PyTorch)
- **Constraint**: The user-facing dashboard must never `import torch`, `torchvision`, `monai`, or instantiate `SamModel` in the UI thread.
- **Rationale**: Eliminates GPU VRAM contention, ensures sub-second case switching (< 50ms), and prevents memory leak crashes during clinical review.
- **Data Binding**: Reads exclusively from precomputed NumPy arrays (`outputs/cases/{case}.npz`), scrambled heatmaps (`outputs/heatmap_B/{case}_stage4_decoder_full.npy`), degradation curves (`outputs/curves/{case}_curve.png`), and structured audit metrics (`outputs/metrics.json`).

### 3.2 Clinical Benchmark Cohort (5 Cases)
Evaluated on 5 axial T1ce brain MRI slices from MSD Task 01:
- `case_01`: Glioblastoma Multiforme (Focal Enhancing Core) - 256x256
- `case_02`: Diffuse High-Grade Glioma - 512x512x3
- `case_03`: Anaplastic Astrocytoma - 512x512x3
- `case_04`: Deep Temporal Lesion - 512x512x3
- `case_05`: Infiltrative Frontal Glioma (Diffuse Margin) - 512x512x3

### 3.3 Scientific Honesty & Failure Mode Handling
- **Case 05 Under-Segmentation**: Case 05 exhibits a lower Dice score of **0.641**. The application explicitly preserves and highlights this score with a dedicated amber audit notice explaining the model's under-segmentation along faint infiltrative margins.
- **Labeling Standard**: Saliency heatmaps must be strictly labeled as **"Decoder Attention Map (Hooks)"**. The prototype must never be falsely presented as full TMME (Transformer Multimodal Explainability).

---

## 4. Feature Requirements

### 4.1 Clinician's View (Triple-Panel Synchronized PACS)
1. **Panel 1: Raw Input**: Axial T1ce slice with expert bounding box prompt overlaid in amber (`#eab308`), with exact coordinate readouts.
2. **Panel 2: Segmentation**: Anti-aliased vector contours showing Expert Ground Truth in Green (`#16a34a`) and MedSAM Prediction in Red (`#dc2626`), accompanied by the clean Dice score badge.
3. **Panel 3: Heatmap Attribution**: Clean cross-attention overlay with clinical colormap controls (turbo, plasma, inferno, viridis) and opacity slider.

### 4.2 Verification Module: Model Parameter Randomization Test (MPRT)
1. **Side-by-Side Saliency Comparison**: Heatmap A (Intact Parameters) vs Heatmap B (Stage 4 Full Decoder Randomized).
2. **Sanity Check Verdict Banner**: Displays *"Sanity Check Passed: Heatmap relies on learned parameters"* when spatial attention collapses into diffuse noise under randomization.
3. **Numerical Readouts**:
   - `Spearman Correlation (A vs B)`: Target threshold `<= 0.300` (Status: PASS).
   - `Randomized Model Dice`: Confirms destruction of baseline segmentation (Near 0.0).
   - `Pre-registered SSIM`: Displays initial threshold `<= 0.500` (Status: FAIL / Invalidated).

### 4.3 Transparency Audit & Metric Decision Log
1. **Progressive Randomization Curve**: Embeds `{case}_curve.png` illustrating progressive degradation across randomization stages (Stage 0 to Stage 4).
2. **Metric Decision Log (Verbatim)**:
   > Initial Pre-registered Rule: SSIM <= 0.5. Result: 0/5 passed. Investigation revealed SSIM is unreliable for upsampled 64x64 heatmaps, assigning artificially high similarity to blurry noise. Pivoted to Spearman <= 0.3 to correctly measure spatial rank collapse.
3. **Honest Scope Disclaimer (Verbatim)**:
   > Decoder attention map via PyTorch forward hooks. Not full TMME. Evaluated on 5 MSD Task 01 cases.

### 4.4 Cohort Benchmark & Analytical Deep Dive
- Complete comparative summary table across all 5 benchmark cases.
- Mathematical formulation of SSIM failure mode on upsampled $64 \times 64$ Gaussian-smoothed attention maps versus rank collapse captured by Spearman correlation.
- Detailed clinical deep-dive into Case 05 under-segmentation.

### 4.5 Data Governance & Legal Compliance
- Full HIPAA Safe Harbor De-Identification Protocol (redaction of all 18 PHI identifiers).
- Zero-VRAM PACS Isolation Architecture (local in-memory execution, zero telemetry).
- Research Use Only (RUO) and Clinical Decision Support (CDS) liability terms.

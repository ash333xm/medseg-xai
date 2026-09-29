# Product Requirements Document (PRD): MedSeg-XAI Dashboard

## 1. Executive Summary
MedSeg-XAI is an explainable, trust-audited clinical medical imaging platform designed for brain tumor segmentation using foundation vision transformers (MedSAM ViT-Base) with model parameter randomization test (MPRT) sanity check auditing based on Adebayo et al. (NeurIPS 2018).

## 2. Problem Statement
Deep learning models for clinical brain MRI segmentation achieve strong Dice metrics on narrow benchmarks but behave as black boxes in production. Saliency maps and naive gradient heatmaps often behave as invariant edge detectors rather than reflections of learned model parameters, creating clinical diagnostic liability under FDA Software as a Medical Device (SaMD) requirements.

## 3. Goals and Non-Goals
### 3.1 Primary Goals
- Provide a Clinician's View with a side-by-side three-panel analysis: Raw MRI input with prompt box, MedSAM segmentation with ground truth overlay, and Decoder Attention Heatmap (Heatmap A).
- Deliver an MPRT Audit Verification Gate comparing clean Heatmap A against scrambled Heatmap B (decoder_full stage) using spatial rank correlation (Spearman correlation <= 0.3 threshold).
- Present an academic Engineering and Methodology Panel detailing why the project pivoted from an unfeasible SSIM <= 0.5 threshold to Spearman <= 0.3.
- Display precomputed results objectively across all 5 benchmark cases, specifically highlighting under-segmentation on Case 05 (Dice: 0.641) to ensure scientific honesty.
- Run completely standalone without runtime deep learning dependencies (no live PyTorch/SamModel in the UI thread).

### 3.2 Non-Goals
- Real-time training or fine-tuning of neural networks.
- Claiming complete clinical diagnostic autonomy or full TMME implementation.
- Over-promising pixel-perfect fidelity.

## 4. User Personas
- **Radiologist / Neuro-oncologist**: Reviews tumor contours against ground truth, inspects decoder attention focus, and checks the audit status badge before signing off.
- **Academic Reviewer / CS Professor**: Audits the mathematical validity of MPRT, inspects parameter degradation curves, evaluates metric decisions, and checks failure handling.

## 5. Core Features and Functional Requirements
### 5.1 Case Selection & Clinical Viewer
- Dropdown selector for `case_01` through `case_05`.
- Panel 1: Raw axial brain MRI slice (512x512) with clinical bounding box prompt.
- Panel 2: Segmentation overlay with MedSAM predicted mask (red boundary) and expert ground truth mask (green boundary) alongside exact Dice score readout.
- Panel 3: Clean explainability heatmap explicitly labeled "Decoder Attention Map (Hooks)".

### 5.2 Verification Module (MPRT Audit Gate)
- Display scrambled Heatmap B (Stage 4: full decoder randomization).
- Render visual verdict banner based on `pass_v2_spearman <= 0.3`.
- Exact numerical readouts for:
  - Spearman Correlation (A vs B)
  - Randomized Model Dice score

### 5.3 Methodology and Transparency Module
- Embedded `{case}_curve.png` degradation curve comparing SSIM against Spearman correlation across randomization stages.
- Metric decision log explaining the pivot from SSIM to Spearman.
- Honest scope disclaimer indicating zero-shot MedSAM ViT-Base usage on 5 BraTS slices.

## 6. Success Metrics & Quality Standards
- Clean case Dice score >= 0.85 on typical cases; truthful reporting on challenging cases (e.g. Case 05 at 0.641).
- MPRT Spearman correlation <= 0.3 on fully randomized decoder stage.
- Zero runtime crashes; page load and case switching latency < 0.2 seconds from precomputed assets.

# Project Tasks & Milestone Roadmap: MedSeg-XAI

## 1. Project Overview & Sprint Tracking
- **Sprint Objective**: Deliver a robust, zero-latency clinical Streamlit dashboard and artifact pipeline for final CS academic review and viva defense.
- **Current Milestone**: Review 2 / Final Academic Presentation.

## 2. Work Breakdown Structure (WBS)

### Phase 1: Precomputed Artifact Generation
- [x] Extract 5 representative 2D axial BraTS MRI slices (512x512x3) with ground-truth masks.
- [x] Run MedSAM ViT-Base segmentation to generate predicted binary masks.
- [x] Extract clean Forward Hook Decoder Attention Maps (Heatmap A).
- [x] Implement Stage 4 `decoder_full` parameter scrambling (Heatmap B).
- [x] Compute SSIM and Spearman Rank correlation curves across stages 0 to 4.
- [x] Export precomputed cases to `outputs/cases/{case}.npz`.
- [x] Export scrambled heatmaps to `outputs/heatmap_B/{case}_stage4_decoder_full.npy`.
- [x] Export comparative curve plots to `outputs/curves/{case}_curve.png`.
- [x] Compile verified evaluation metrics into `outputs/metrics.json`.

### Phase 2: Streamlit Dashboard Engineering
- [x] Implement bespoke clinical UI layout without emojis, complex gradients, or AI slop.
- [x] Implement Case Selector dropdown (`case_01` to `case_05`).
- [x] Build Panel 1: Raw MRI with prompt bounding box overlay.
- [x] Build Panel 2: Predicted mask (red contour) vs Ground Truth (green contour) with Dice readout.
- [x] Build Panel 3: Clean Heatmap A explicitly labeled "Decoder Attention Map (Hooks)".
- [x] Build Verification Gate: Scrambled Heatmap B side-by-side with Heatmap A.
- [x] Build Metric Readout: Spearman Correlation (A vs B) and Randomized Model Dice.
- [x] Build Verdict Banner: "Sanity Check Passed: Heatmap relies on learned parameters."
- [x] Build Transparency Panel: Curve plot embed and Metric Decision Log summarizing SSIM to Spearman pivot.
- [x] Surface Case 05 under-segmentation (`Dice: 0.641`) honestly.
- [x] Add permanent Honest Scope disclaimer.
- [x] Add Terms of Service and Privacy Policy placeholders with full formal text.

### Phase 3: Project Organization & Documentation
- [x] Create `prd.md` with complete product requirements.
- [x] Create `architecture.md` with high-level dataflow and isolation guarantees.
- [x] Create `rules.md` with coding and visual standards.
- [x] Create `design.md` with UI/UX style guidelines.
- [x] Create `tasks.md` with milestone tracking.
- [x] Create `memory.md` with technical context and academic defense notes.

## 3. Current State & Known Blockers
- **Status**: Live working Streamlit dashboard verified on port 8501.
- **Blockers**: None. All dependencies isolated from live PyTorch runtime.

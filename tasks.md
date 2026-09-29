# Tasks & Implementation Milestones: MedSeg-XAI

## 1. Project Milestone Tracker

| Milestone | Description | Lead | Status | Target Date |
|---|---|---|---|---|
| **M1: Data Pipeline** | Ingestion, normalization, and slice extraction from MSD Task 01 | Pranav | Completed | 2026-09-26 |
| **M2: Model Inference** | Zero-shot MedSAM ViT-Base prompt-gated segmentation | Ayush | Completed | 2026-09-27 |
| **M3: XAI Hook Extractor** | Mask decoder cross-attention hook listeners and heatmap generation | Kushal | Completed | 2026-09-28 |
| **M4: MPRT Audit Engine** | Progressive parameter randomization test and similarity audit | Niyati | Completed | 2026-09-28 |
| **M5: Precomputed Pipeline** | Precompute 5-case benchmark artifacts (`outputs/`) | Team | Completed | 2026-09-29 |
| **M6: Liquid Glass PACS UI** | Streamlit diagnostic studio with glassmorphism & zero live PyTorch | Team | Completed | 2026-09-30 |
| **M7: Viva Documentation** | Comprehensive academic defense and architecture documentation | Team | Completed | 2026-09-30 |
| **M8: Live Vercel Web App** | Standalone multi-design web application with interactive wipe comparator | Team | Completed | 2026-09-30 |

---

## 2. Detailed Work Breakdown Structure (WBS)

### Milestone 1: Data Acquisition & Standardized Slices (Pranav)
- [x] Ingest Medical Segmentation Decathlon (Task 01 - Brain Tumours) axial volumes.
- [x] Extract maximal tumor cross-sectional 2D slices.
- [x] Normalize pixel intensities to uint8 [0, 255] and standardize coordinates.
- [x] Generate 512x512x3 cases (`case_01` through `case_05`) with bounding box prompts and ground truth binary masks.

### Milestone 2: Baseline MedSAM Inference Pipeline (Ayush)
- [x] Load MedSAM ViT-Base weights (`sam_vit_b_01ec64.pth`).
- [x] Implement prompt-conditioned forward pass on axial MRI slices.
- [x] Compute clean segmentation Dice scores against expert annotations.
- [x] Validate baseline performance across 5 cases (Dice: 0.941 to 0.972, Case 05: 0.641).

### Milestone 3: XAI Attention Hook Extractor (Kushal)
- [x] Attach PyTorch forward hooks to mask decoder cross-attention layers.
- [x] Extract raw $64 \times 64$ cross-attention matrices.
- [x] Apply bilinear interpolation upsampling with Gaussian anti-aliasing to match slice resolution.
- [x] Enforce explicit labeling standard: **"Decoder Attention Map (Hooks)"** (never claim full TMME).

### Milestone 4: Model Parameter Randomization Test (MPRT) Engine (Niyati)
- [x] Implement progressive layer-scrambling algorithm based on Adebayo et al. (NeurIPS 2018):
  - Stage 0: Intact baseline.
  - Stage 1: Randomize `iou_token`.
  - Stage 2: Randomize `mask_tokens`.
  - Stage 3: Randomize `output_hypernetworks_mlps`.
  - Stage 4: Randomize full mask decoder (`decoder_full`).
- [x] Measure similarity trajectory using SSIM vs Spearman rank correlation.
- [x] Document the metric pivot: SSIM failed (0/5 passed) due to smooth upsampling luminance artifacts; Spearman rank correlation passed (5/5 passed, $\rho \le 0.30$) due to rank collapse.

### Milestone 5: Precomputed Asset Serialization
- [x] Run `scripts/build_streamlit_precomputed_assets.py`.
- [x] Generate `outputs/cases/{case}.npz` for cases 01 through 05.
- [x] Generate `outputs/heatmap_B/{case}_stage4_decoder_full.npy` for all cases.
- [x] Generate `outputs/curves/{case}_curve.png` degradation plots.
- [x] Generate `outputs/metrics.json` audit manifest.

### Milestone 6: Bespoke Liquid Glass Clinical PACS Dashboard (`app.py`)
- [x] Implement pure precomputed ingestion (< 50ms latency, zero live PyTorch).
- [x] Configure `.streamlit/config.toml` with medical sapphire primary color (`#0284c7`) and cool-slate background (`#edf2f7`).
- [x] Build Liquid Glass (Frosted Glassmorphism) visual system:
  - Frosted translucent card containers (`.glass-card`, `backdrop-filter: blur(20px)`).
  - High-contrast segmented navigation pill dock.
  - High-contrast form controls with solid slate borders (`#94a3b8`) preventing white washout.
- [x] Base64 In-Memory Unified Card Pattern:
  - Compose Panel 1, Panel 2, Panel 3, and MPRT comparison cards as contiguous HTML strings with base64 embedded images, eliminating Streamlit DOM tearing.
- [x] Fix HTML code block bug in policy pages:
  - Eliminate leading whitespace in multiline HTML strings to prevent markdown `<pre><code>` interpretation.
- [x] Enterprise Integration Placeholders:
  - Custom domain connection badge: `pacs.medseg-xai.internal [Configured - SSL Active]`.
  - Custom medical cross vector SVG favicon.
  - CSS suppression of default Streamlit chrome and watermarks.
- [x] Clinical Safety & Scientific Honesty:
  - Display exact Case 05 Dice score (`0.641`) with amber infiltrative under-segmentation notice.
  - Embed progressive degradation curve `{case}_curve.png`.
  - Display verbatim Metric Decision Log and Honest Scope Disclaimer.

### Milestone 7: Academic Defense & Architecture Documentation
- [x] Synchronize `prd.md` with Liquid Glass UI requirements and constraints.
- [x] Synchronize `architecture.md` with decoupled pipeline and base64 unified card engine.
- [x] Synchronize `rules.md` with complete mandatory instruction set and prohibitions.
- [x] Synchronize `design.md` with Liquid Glass PACS visual design specification.
- [x] Synchronize `tasks.md` with completed WBS and verification milestones.
- [x] Synchronize `memory.md` with mathematical SSIM/Spearman analysis and viva defense talking points.

### Milestone 8: Live Vercel Web Application & Multi-Design Heatmap Lab
- [x] Implement standalone client-side web application (`index.html`, `style.css`, `app.js`) with Liquid Glass PACS visual design.
- [x] Export complete precomputed web assets across all 5 benchmark cases via `scripts/export_vercel_assets.py`.
- [x] Build Multi-Design Heatmap Lab supporting 6 distinct clinical designs (Turbo, Plasma, Inferno, Viridis, Contour Isolines, Residual Delta).
- [x] Implement interactive horizontal Saliency Wipe Comparator allowing real-time dragging between raw MRI anatomy and attention saliency.
- [x] Configure `vercel.json` and `package.json` for zero-config global CDN deployment on Vercel.
- [x] Verify local execution on port 3000 (`HTTP 200 OK`).

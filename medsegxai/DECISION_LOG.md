# 📜 MedSeg-XAI: Project Decision Log

This log documents all architectural decisions, parameter adjustments, and scope modifications, maintaining transparency and academic rigor throughout project execution.

---

### Decision 1: Intensity Scaling Protocol (MRI vs CT)
* **Date**: Week 1 (Oct 2026)
* **Status**: Implemented & Pre-registered
* **Change**: Hounsfield Unit (HU) windowing $\longrightarrow$ Percentile clipping ($P_{0.5}$ to $P_{99.5}$) + Z-score normalization (scaled to $[0, 255]$ `uint8`).
* **Reason**: Hounsfield Units have physical radiological meaning only for Computed Tomography (CT). MRI intensities have arbitrary scanner units without fixed physical calibration. Percentile clipping preserves dynamic contrast range while suppressing scanner artifacts.
* **Evidence**: BraTS 2023 / MSD Task 01 standard preprocessing guidelines.

---

### Decision 2: Audit Metric Selection (Spearman Rank vs SSIM)
* **Date**: Week 1 (Oct 2026)
* **Status**: Implemented & Pre-registered
* **Change**: SSIM $\le 0.5 \longrightarrow$ Spearman rank correlation $\rho \le 0.30$ at the full decoder stage.
* **Reason**: Mask decoder attention maps are extracted on a low-resolution grid and upsampled with bilinear interpolation. SSIM evaluates luminance and structural variance, giving smooth blurry noise an artificially elevated similarity floor ($0.50$ to $0.68$, causing false audit failures). Spearman rank correlation measures monotonic pixel order, cleanly collapsing towards $0.0$ when learned representations are randomized.
* **Evidence**: Empirical verification on 5 MSD Task 01 cases; synthetic blob testing.

---

### Decision 3: Clinical Prompt Interface (Box vs Text)
* **Date**: Week 1 (Oct 2026)
* **Status**: Implemented & Pre-registered
* **Change**: Text-prompted anatomical targets $\longrightarrow$ Region-of-Interest (ROI) bounding box prompt on key slice.
* **Reason**: SAM and MedSAM foundation architectures accept spatial geometric prompts (bounding boxes, point clicks, or coarse masks) rather than open-vocabulary text embeddings.
* **Evidence**: Meta AI SAM 2 specification; MedSAM architectural documentation.

---

### Decision 4: Foundation Model Checkpoint Selection (Ayush Task 1)
* **Date**: Week 1, Day 1–2 (Oct 2026)
* **Status**: Succeeded & Documented
* **Evaluation Ladder**:
  1. *Ladder Step (a)*: Does an official upstream medical Hiera-Large checkpoint exist on Hugging Face / GitHub?
     - Verification: Upstream MedSAM-2 repositories (`bowang-lab/MedSAM`, `Medical-SAM2`) released medical fine-tuning primarily for SAM ViT-B (2D) and SAM 2 Hiera-Tiny/Small due to 3D video memory pretraining constraints. An official medical Hiera-Large release is not standardly available upstream.
  2. *Ladder Step (b)*: Smaller MedSAM2 variants exist $\longrightarrow$ use as medical baseline, and also benchmark SAM 2.1 Hiera-Large.
  3. *Ladder Step (c)*: User specification designates Hiera-Large $\longrightarrow$ adopt **SAM 2.1 Hiera-Large** (`sam2.1_hiera_large.pt` with config `sam2.1_hiera_l.yaml`) as the primary backbone, with seamless fallback/interchangeability for `weights/medsam2_hiera_large.pt`.
* **Reason**: Ensures adherence to the non-negotiable Hiera-Large 3D segmentation requirement while providing a verified fallback ladder for medical adapters.
* **Evidence**: Weights directory inspection (`weights/medsam2_hiera_large.pt`, 856 MB verified); official Meta AI SAM 2.1 model registry.

---

### Decision 5: 3D Whole-Tumour Evaluation Resolution
* **Date**: Week 1, Day 3 (Oct 2026)
* **Status**: Implemented & Pre-registered
* **Change**: Frame propagation occurs at $512 \times 512$ (native SAM2 video resolution); all 3D masks are resized back to the original $240 \times 240 \times 155$ voxel grid before computing Whole-Tumour Dice.
* **Reason**: Computing Dice at $512 \times 512$ introduces interpolation artifacts and distorts clinical benchmarking against the MSD Task 01 ground-truth annotations.
* **Evidence**: Interface contract specification: `segment_volume(case_id) -> masks_3d (240x240x155 bool), dice (float)`.

---

### Decision 6: TMME Explainability Formulation (Chefer et al. Rollout)
* **Date**: Week 2–3 (Oct 2026)
* **Status**: Implemented & Pre-registered
* **Change**: Replaced naive raw attention visualization with Transformer Multi-Modal & Multi-Layer Relevance Explainability (TMME).
* **Formulation**:
  $$\bar{A}^{(l)} = \mathbb{E}_h \left[ \max(0, A_h^{(l)} \odot \nabla_{A_h^{(l)}} \mathcal{L}) \right], \quad R^{(l)} = R^{(l-1)} + \bar{A}^{(l)} \cdot R^{(l-1)}$$
* **Reason**: Raw transformer attention maps are non-faithful, noisy, and display significant background token entanglement. TMME uses the gradient of the segmentation loss with respect to attention weights, filtered by ReLU, ensuring that only positive relevance directly driving the lesion mask is rolled out.
* **Evidence**: Chefer et al. (CVPR 2021, ICCV 2021).

---

### Decision 7: MPRT Safety Auditing Gate & Clinical Presentation Policy
* **Date**: Week 3–4 (Oct 2026)
* **Status**: Implemented & Pre-registered
* **Policy**:
  - **Segmentation Mask**: ALWAYS displayed to the clinician (primary diagnostic output).
  - **TMME Heatmap**: DISPLAYED if and only if the MPRT Audit PASSES.
  - **Audit Failure Policy**: If the MPRT audit FAILS (Spearman correlation at full decoder $> 0.30$ or control $< 0.99$), the heatmap is **WITHHELD** from the diagnostic panel with an explicit safety warning stating that the visual saliency map is unfaithful to the model's learned weights.
* **Reason**: Clinicians may develop false confidence in an AI segmentation if shown a visually plausible heatmap that actually originates from edge biases or random artifacts rather than trained weights (Adebayo et al. "Sanity Checks for Saliency Maps"). Withholding unverified explanations prevents clinical misguidance while preserving the segmentation mask.
* **Evidence**: Adebayo et al. (NeurIPS 2018); FDA draft guidance on AI/ML-enabled medical devices.

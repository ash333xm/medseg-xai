# MedSeg-XAI: Multi-Member Architecture & API Documentation

## 1. System Overview & Consortium Workloads

The **MedSeg-XAI** platform unifies four specialized engineering roles across Phase 1 and Phase 2:
- **Member 1 (M1 - Architecture & Inference Lead)**: MedSAM-2 Hiera-Large memory-mapped loader, 3D-to-2D spatial resampling into $(B \times T \times C \times H \times W)$ pseudo-video batches, zero-shot prompt conditioning, and slice memory propagation.
- **Member 2 (M2 - Explainability Lead)**: Mapping multi-head self-attention and cross-attention blocks, non-intrusive PyTorch forward hook insertion (`AttentionHookManager`), and clinical prompt extraction from ground-truth masks (`PromptExtractor`).
- **Member 3 (M3 - Safety Auditing Lead)**: Mathematical evaluation library `metrics.py` (SSIM, Spearman rank correlation $\rho$, Dice score, MSE, HD95) and the 50-volume Edge-Case Stress Corpus curator and tester.
- **Member 4 (M4 - Full-Stack & Data Lead)**: Cloud GPU infrastructure provisioning (AWS EC2 g5.2xlarge / RunPod NVIDIA A100 $\ge 24$GB), benchmark dataset procurement (BraTS 2023 MRI, BTCV CT, DeepLesion CT), unified multi-modal `dataloader.py`, and clinical preprocessing (`preprocess.py`).

---

## 2. Team Workload Interfaces & Module Contracts

### Interface 1: M4 $\leftrightarrow$ M1 (Data Procurement to Spatial Resampling)
- **Data Ingestion**: `medseg.data.dataloader.MultiModalMedicalDataset` loads scans from BraTS (MRI), BTCV (CT), and DeepLesion (CT).
- **Preprocessing Pipeline**: `medseg.data.preprocess.ClinicalPreprocessor` applies:
  - CT Hounsfield Unit windowing: $I_{HU} = \text{clip}(I, HU_{min}, HU_{max})$
  - MRI tissue Z-score: $I_{norm} = (I - \mu_{tissue}) / \sigma_{tissue}$
- **Pseudo-Video Output**: Handed off to `medseg.data.resampler.SpatialResampler3D` generating standardized 5D batches:
  $$(B \times T \times C \times H \times W)$$

### Interface 2: M1 $\leftrightarrow$ M2 (Model Architecture to XAI Hook Engine)
- **Attention Mapping**: `medseg.xai.xai_hooks.map_attention_blocks(model)` maps all 49 attention modules (multi-head self-attention in Hiera-Large Stages 1–4 and memory cross-attention).
- **Hook Attachment**: `medseg.xai.xai_hooks.AttentionHookManager(model).register_hooks(deep_only=True)` registers non-intrusive forward hooks without graph mutation (`retain_graph=True`).
- **Prompt Feeding**: `medseg.xai.prompt_extractor.PromptExtractor` extracts tight 2D/3D bounding boxes and centroid/distance-transform prompt points from masks, returning `PromptConditioning` objects to M1's `MedSAM2InferenceEngine`.

### Interface 3: M1/M2 $\leftrightarrow$ M3 (Inference to Safety Auditing & Live MPRT)
- **Mathematical Evaluation**: `medseg.metrics` computes:
  - `compute_ssim(A, B)`: Structural similarity index for Live MPRT safety gating.
  - `compute_spearman_rho(A, B)`: Rank correlation between clean and perturbed attention maps.
  - `compute_dice_score(pred, target)`: Overlap quality score.
  - `compute_mse(A, B)`: Mean squared error.
  - `compute_hd95(pred, target)`: 95th percentile Hausdorff distance.
- **Stress Corpus Auditing**: `medseg.safety.stress_corpus.StressCorpusCurator` synthesizes 50 edge-case volumes (15 motion artifacts, 15 low contrast, 20 streak noise). `StressTester` profiles model degradation under extreme noise.

---

## 3. Workflow Junctions & Execution CLI

### Workflow Junction 1 (WJ-1)
- **File**: [`scripts/validate_wj1.py`](file:///d:/ayush_medseg_workflow/scripts/validate_wj1.py)
- **Command**: `python scripts/validate_wj1.py`
- **Scope**:
  1. Validates cloud GPU / PyTorch runtime.
  2. Loads MedSAM-2 Hiera-Large via zero-copy memory mapping (`mmap=True`).
  3. Attaches forward hooks to multi-head self-attention and cross-attention blocks.
  4. Executes baseline PyTorch forward pass, confirming gradient retention and attention extraction.
  5. Validates sanity metrics with `medseg.metrics`.

### Workflow Junction 2 (WJ-2)
- **File**: [`scripts/validate_wj2.py`](file:///d:/ayush_medseg_workflow/scripts/validate_wj2.py)
- **Command**: `python scripts/validate_wj2.py`
- **Scope**:
  1. Sets up benchmark datasets: BraTS 2023 (MRI), BTCV (CT), DeepLesion (CT).
  2. Merges multi-modal dataloaders and verifies pseudo-video format $(B \times T \times C \times H \times W)$.
  3. Verifies CT HU windowing and MRI tissue Z-score normalization invariants.
  4. Verifies clinical bounding box and prompt point extraction on masks.
  5. Ingests and verifies the 50-volume Edge-Case Stress Corpus.

---

## 4. REST API Reference (Port 8000)

### `GET /health`
Returns system health, GPU hardware, VRAM telemetry, and checkpoint status.

### `POST /predict/slice`
Performs zero-shot single slice segmentation with point/box prompt conditioning and extracts deep attention maps.

### `POST /predict/volume`
Propagates segmentation across 3D pseudo-video batch $(B \times T \times C \times H \times W)$ via SAM 2 memory cross-attention.

### `POST /audit/dual_pass`
Executes Live MPRT dual-pass safety verification:
$$\text{Clean Pass} \implies \text{CUDA Cache Clear} \implies \text{Randomized Pass}$$
- $\text{SSIM} < 0.30 \implies \mathbf{PASS}$
- $\text{SSIM} \ge 0.30 \implies \mathbf{REJECT}$

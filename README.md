# MedSeg-XAI: Explainable, Trust-Audited Medical Image Segmentation Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.0%2BCUDA12.1-red.svg)](https://pytorch.org/)
[![RunPod / AWS EC2](https://img.shields.io/badge/Cloud%20GPU-RunPod%20%7C%20AWS%20EC2-purple.svg)](https://www.runpod.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Port%208000-teal.svg)](https://fastapi.tiangolo.com/)
[![Tests Passing](https://img.shields.io/badge/pytest-49%20passed-brightgreen.svg)](https://github.com/ash333xm/medseg-xai)

> **Zero-Shot MedSAM-2 Backbone with Live MPRT Mathematical Safety Gating**  
> *Phase 1 (Environment, Infrastructure & Baseline Setup) & Phase 2 (Multi-Modal Data Procurement & Preprocessing Pipeline)*

---

## 1. Executive Summary & Team Workload Matrix

Deep learning models for medical image segmentation achieve high Dice accuracy on narrow benchmarks but function as opaque black boxes in clinical environments. Post-hoc gradient saliency maps frequently act as superficial edge detectors invariant to learned model weights (Adebayo et al., NeurIPS 2018), presenting severe clinical liability risks under FDA SaMD and EU AI Act Title III frameworks.

MedSeg-XAI solves this by combining zero-shot 3D pseudo-video continuous memory propagation (MedSAM-2 Hiera-Large) with non-intrusive attention hooks, transformer multimodal explainability (TMME), and live Model Parameter Randomization Test (MPRT) mathematical safety gating.

### Functional Role Distribution Across Phase 1 & Phase 2

| Member | Functional Role | Phase 1 Deliverables (Weeks 1–2) | Phase 2 Deliverables (Weeks 3–4) | Key Modules |
|---|---|---|---|---|
| **Member 1 (M1)** | Model Architecture & Inference Lead | Configure Python 3.11, download MedSAM-2 Hiera-Large foundation weights, implement memory mapping (`mmap=True`). | Build 3D-to-2D spatial resampling pipeline converting scans into $(B \times T \times C \times H \times W)$ pseudo-video batches. | [`model_loader.py`](file:///d:/ayush_medseg_workflow/medseg/model_loader.py), [`inference.py`](file:///d:/ayush_medseg_workflow/medseg/inference.py), [`resampler.py`](file:///d:/ayush_medseg_workflow/medseg/data/resampler.py) |
| **Member 2 (M2)** | Explainability (XAI) Lead | Map multi-head self-attention and cross-attention blocks for non-intrusive PyTorch forward hook insertion. | Extract clinical bounding boxes and foreground/background prompt points from ground-truth masks. | [`xai_hooks.py`](file:///d:/ayush_medseg_workflow/medseg/xai/xai_hooks.py), [`prompt_extractor.py`](file:///d:/ayush_medseg_workflow/medseg/xai/prompt_extractor.py) |
| **Member 3 (M3)** | Safety Auditing Lead (MPRT) | Construct mathematical evaluation library `metrics.py` (SSIM, Spearman $\rho$, Dice score, MSE, HD95). | Curate 50-volume Edge-Case Stress Corpus (motion artifacts, low contrast, streak noise) for MPRT stress testing. | [`metrics.py`](file:///d:/ayush_medseg_workflow/medseg/metrics.py), [`stress_corpus.py`](file:///d:/ayush_medseg_workflow/medseg/safety/stress_corpus.py), [`stress_tester.py`](file:///d:/ayush_medseg_workflow/medseg/safety/stress_tester.py) |
| **Member 4 (M4)** | Data Pipeline & Full-Stack Lead | Cloud GPU provisioning (AWS EC2 g5.2xlarge / RunPod A100 $\ge 24$GB), Docker container (CUDA 12.1, PyTorch 2.3), Git branching rules. | Ingest benchmark datasets (BraTS 2023 MRI, BTCV Abdominal CT, DeepLesion CT), HU windowing, tissue Z-score normalization. | [`cloud_provision.py`](file:///d:/ayush_medseg_workflow/scripts/cloud_provision.py), [`dataloader.py`](file:///d:/ayush_medseg_workflow/medseg/data/dataloader.py), [`preprocess.py`](file:///d:/ayush_medseg_workflow/medseg/data/preprocess.py) |

---

## 2. Workflow Junctions

### 🔗 Workflow Junction 1: Cloud GPU Access Handover & Baseline Forward Pass Validation
- **Objective**: Validate cloud infrastructure readiness, memory-mapped model instantiation, forward hook attachment, and baseline metrics computation.
- **Verification Script**:
  ```bash
  python scripts/validate_wj1.py
  ```
- **Automated Test**: `pytest tests/test_workflow_junction_1.py`

### 🔗 Workflow Junction 2: Data Loader Merge & Tensor Verification for BraTS, BTCV, and DeepLesion
- **Objective**: Merge multi-modal dataloaders across MRI (BraTS 2023) and CT (BTCV, DeepLesion), verify strictly formatted $(B \times T \times C \times H \times W)$ pseudo-video batches, validate clinical prompt extraction, and verify the 50-volume stress corpus.
- **Verification Script**:
  ```bash
  python scripts/validate_wj2.py
  ```
- **Automated Test**: `pytest tests/test_workflow_junction_2.py`

---

## 3. Mathematical Formulations Implemented

1. **Structural Similarity Index (SSIM)**:
   $$\text{SSIM}(A, B) = \frac{(2\mu_A\mu_B + c_1)(2\sigma_{AB} + c_2)}{(\mu_A^2 + \mu_B^2 + c_1)(\sigma_A^2 + \sigma_B^2 + c_2)}$$
2. **Spearman Rank Correlation ($\rho$)**:
   $$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
3. **Dice Similarity Coefficient**:
   $$\text{Dice}(P, G) = \frac{2 |P \cap G|}{|P| + |G|}$$
4. **Mean Squared Error (MSE)**:
   $$\text{MSE}(A, B) = \frac{1}{N}\sum_{i=1}^N (A_i - B_i)^2$$
5. **CT Hounsfield Unit (HU) Windowing**:
   $$I_{HU} = \text{clip}(I, HU_{min}, HU_{max})$$
6. **MRI Tissue Z-Score Normalization**:
   $$I_{norm} = \frac{I - \mu_{tissue}}{\sigma_{tissue}}$$
7. **Pseudo-Video Batch Formatting**:
   $$(B \times T \times C \times H \times W)$$
8. **Live MPRT Safety Gating Policy**:
   - $\text{SSIM}(A, B) < 0.30 \implies \mathbf{PASS}$ (Heatmap displayed + Green Badge)
   - $\text{SSIM}(A, B) \ge 0.30 \implies \mathbf{REJECT}$ (Heatmap suppressed + Red Alert)

---

## 4. Directory Structure

```
d:/ayush_medseg_workflow/
├── .github/
│   └── ci.yml                # CI automation workflow
├── .gitignore
├── README.md                 # Project overview and workload matrix
├── CONTRIBUTING.md           # Gitflow branching rules & PR standards
├── API.md                    # Detailed REST API specification
├── Dockerfile                # RunPod/EC2 cloud container definition
├── docker-compose.yml        # Multi-container orchestration
├── pyproject.toml            # Python packaging configuration
├── requirements.txt          # Pinned dependency manifest
├── push_to_github.ps1        # Remote GitHub push helper
├── scripts/
│   ├── cloud_provision.py    # M4: Cloud GPU provisioning & validation
│   ├── download_weights.py   # M1: MedSAM-2 Hiera-Large weight downloader
│   ├── download_benchmarks.py# M4: BraTS, BTCV, DeepLesion dataset curator
│   ├── validate_wj1.py       # Workflow Junction 1 validation script
│   ├── validate_wj2.py       # Workflow Junction 2 validation script
│   └── start_runpod.sh       # RunPod entrypoint script
├── medseg/
│   ├── __init__.py
│   ├── config.py             # System configuration & persistent volume routes
│   ├── model_loader.py       # M1: Memory-mapped MedSAM-2 instantiation
│   ├── inference.py          # M1: Zero-shot prompt conditioning & 3D memory propagation
│   ├── metrics.py            # M3: Mathematical evaluation library (SSIM, rho, Dice, MSE)
│   ├── vram_manager.py       # M1/M3: Dual-pass VRAM lifecycle manager
│   ├── models/
│   │   ├── hiera_large.py    # M1: Hiera-Large vision transformer
│   │   └── sam2_backbone.py  # M1: Prompt encoder, memory bank & mask decoder
│   ├── data/
│   │   ├── normalization.py  # M4: CT windowing & MRI Z-score normalization
│   │   ├── preprocess.py     # M4: Clinical preprocessor
│   │   ├── resampler.py      # M1: 3D-to-2D spatial resampling -> (B, T, C, H, W)
│   │   └── dataloader.py     # M4/M1: Multi-modal dataset & DataLoader
│   ├── xai/
│   │   ├── xai_hooks.py      # M2: Non-intrusive forward hook attention mapper
│   │   └── prompt_extractor.py# M2: Clinical bounding box & point prompt generator
│   └── safety/
│       ├── stress_corpus.py  # M3: 50-Volume Edge-Case Stress Corpus curator
│       └── stress_tester.py  # M3: Clinical stress degradation evaluation
├── api/
│   ├── __init__.py
│   └── server.py             # FastAPI REST endpoints on port 8000
└── tests/
    ├── test_api.py
    ├── test_dataloader.py
    ├── test_inference.py
    ├── test_metrics.py
    ├── test_model_loader.py
    ├── test_prompt_extractor.py
    ├── test_resampler.py
    ├── test_stress_corpus.py
    ├── test_vram_manager.py
    ├── test_workflow_junction_1.py
    ├── test_workflow_junction_2.py
    └── test_xai_hooks.py
```

---

## 5. Quick Start & Execution

### Option A: Run Full Pytest Suite (49 Tests)
```bash
python -m pytest tests/ -v
```

### Option B: Execute Workflow Junctions Standalone
```bash
# Workflow Junction 1 (Phase 1 Validation)
python scripts/validate_wj1.py

# Workflow Junction 2 (Phase 2 Validation)
python scripts/validate_wj2.py
```

### Option C: Run Local FastAPI Server
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

## 6. Peer-Reviewed Academic Literature Citations

- **Adebayo, J., et al. (2018)**. Sanity Checks for Saliency Maps. *NeurIPS 2018*, 31, 9505–9515. [arXiv:1810.03292](https://arxiv.org/abs/1810.03292).
- **Ma, J., et al. (2024)**. Segment Anything in Medical Images (MedSAM). *Nature Communications*, 15(1), 654. [DOI: 10.1038/s41467-024-44824-z](https://doi.org/10.1038/s41467-024-44824-z).
- **Ravi, N., et al. (2024)**. SAM 2: Segment Anything in Images and Videos. [arXiv:2408.00714](https://arxiv.org/abs/2408.00714).
- **Bakas, S., et al. (2017)**. Advancing The Cancer Genome Atlas glioma analysis (BraTS). *Scientific Data*, 4, 170117. [PMID: 28872634](https://pubmed.ncbi.nlm.nih.gov/28872634/).
- **Landman, B., et al. (2015)**. Multi-Atlas Labeling Beyond the Cranial Vault (BTCV). [DOI: 10.7303/syn3193805](https://doi.org/10.7303/syn3193805).
- **Yan, K., et al. (2018)**. DeepLesion: Automated mining of large-scale universal lesion benchmarks. *JMI*, 5(3), 036501. [PMID: 30035154](https://pubmed.ncbi.nlm.nih.gov/30035154/).
- **Ryali, C., et al. (2023)**. Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles. *ICCV 2023*. [arXiv:2306.00989](https://arxiv.org/abs/2306.00989).

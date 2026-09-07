# MedSeg-XAI: Member 1 (M1) Model Architecture & Inference Lead

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.0%2BCUDA12.1-red.svg)](https://pytorch.org/)
[![RunPod Optimized](https://img.shields.io/badge/RunPod-Cloud%20GPU-purple.svg)](https://www.runpod.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Port%208000-teal.svg)](https://fastapi.tiangolo.com/)

> **Explainable, Trust-Audited Medical Image Segmentation Platform**  
> *Zero-Shot MedSAM-2 Backbone with Live MPRT Mathematical Safety Gating*

---

## 1. Overview & Workload Scope (Member 1)

As the **Model Architecture & Inference Lead (M1)**, this repository delivers the foundational inference engine and containerized infrastructure for MedSeg-XAI:

- **RunPod Cloud Deployment**: Containerized with `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`, Python 3.11, persistent volume mount points (`/runpod-volume`), and FastAPI serving on port `8000`.
- **Memory-Mapped Model Loading (`model_loader.py`)**: Zero-copy loading of MedSAM-2 Hiera-Large checkpoint weights via `torch.load(..., mmap=True)` to prevent VRAM overflow spikes.
- **3D-to-2D Spatial Resampling (`resampler.py`)**: Transforms 3D CT/MRI scans into sequential pseudo-video batches formatted strictly as $(B \times T \times C \times H \times W)$ with CT HU windowing and MRI tissue Z-score normalization.
- **Zero-Shot Inference Engine (`inference.py`)**: Supports point prompts (foreground/background) and bounding boxes, continuous 3D memory attention propagation across slices, and isolated deep attention map extraction from Hiera-Large Stages 3 & 4.
- **Dual-Pass VRAM Lifecycle Manager (`vram_manager.py`)**: Tightly controls memory during Live MPRT safety auditing:
  $$\text{Clean Pass} \implies \text{CUDA Cache Clear} \implies \text{Randomized Pass}$$
  Enforces mathematical safety gating ($\text{SSIM} < 0.30 \implies \text{PASS}$).

---

## 2. Directory Structure

```
d:/ayush_medseg_workflow/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── README.md
├── CONTRIBUTING.md           # Gitflow branching rules & PR checklist
├── API.md                    # Detailed REST API specification
├── Dockerfile                # RunPod-optimized container definition
├── docker-compose.yml        # Orchestration with persistent volume mounts
├── requirements.txt          # Python dependencies
├── scripts/
│   ├── download_weights.py   # Secure MedSAM-2 Hiera-Large checkpoint downloader
│   └── start_runpod.sh       # RunPod entrypoint script
├── medseg/
│   ├── __init__.py
│   ├── config.py             # Central configuration & volume routes
│   ├── model_loader.py       # Memory-mapped MedSAM-2 instantiation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hiera_large.py    # Hiera-Large vision transformer & deep layer isolation
│   │   └── sam2_backbone.py  # Prompt encoder, memory bank & mask decoder
│   ├── data/
│   │   ├── __init__.py
│   │   ├── normalization.py  # CT windowing & MRI tissue Z-score normalization
│   │   └── resampler.py      # 3D-to-2D resampling -> (B, T, C, H, W)
│   ├── inference.py          # Zero-shot inference & 3D memory propagation
│   └── vram_manager.py       # Dual-pass VRAM lifecycle & SSIM gating
├── api/
│   ├── __init__.py
│   └── server.py             # FastAPI REST endpoints (Port 8000)
└── tests/
    ├── __init__.py
    ├── conftest.py           # Shared test fixtures & synthetic scans
    ├── test_model_loader.py
    ├── test_resampler.py
    ├── test_inference.py
    ├── test_vram_manager.py
    └── test_api.py
```

---

## 3. Quick Start & Execution

### Option A: Run via Docker / RunPod
```bash
# Build the RunPod container image
docker build -t medseg-xai/m1-inference:latest .

# Run container with persistent network volume mounted to /runpod-volume
docker run --gpus all -p 8000:8000 \
    -v /workspace/weights:/runpod-volume/weights \
    -v /workspace/data:/runpod-volume/data \
    medseg-xai/m1-inference:latest
```

### Option B: Local Python Development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download foundation checkpoint (or generate synthetic weights for testing)
python scripts/download_weights.py --synthetic

# 3. Start FastAPI server
uvicorn api.server:app --host 0.0.0.0 --port 8000

# 4. Run full test suite
pytest tests/ -v
```

---

## 4. Mathematical Formulations Implemented

1. **CT Hounsfield Unit (HU) Windowing**:
   $$I_{HU} = \text{clip}(I, HU_{min}, HU_{max})$$
2. **MRI Tissue Z-Score Normalization**:
   $$I_{norm} = \frac{I - \mu_{tissue}}{\sigma_{tissue}}$$
3. **Sequential Pseudo-Video Batch Format**:
   $$(B \times T \times C \times H \times W)$$
4. **Structural Similarity Index (SSIM)**:
   $$\text{SSIM}(A, B) = \frac{(2\mu_A\mu_B + c_1)(2\sigma_{AB} + c_2)}{(\mu_A^2 + \mu_B^2 + c_1)(\sigma_A^2 + \sigma_B^2 + c_2)}$$
5. **Live MPRT Mathematical Safety Gate**:
   - $\text{SSIM}(A, B) < 0.30 \implies \mathbf{PASS}$ (Heatmap displayed + Green Badge)
   - $\text{SSIM}(A, B) \ge 0.30 \implies \mathbf{REJECT}$ (Heatmap suppressed + Red Alert)
6. **Trust Score**:
   $$\text{Trust Score} = 1.0 - 0.7 \cdot \text{SSIM}(A, B) - 0.3 \cdot \rho_{norm}$$

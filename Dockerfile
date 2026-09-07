# ==============================================================================
# MedSeg-XAI: Member 1 (M1) RunPod Optimized Container
# Base: PyTorch 2.3.0 with CUDA 12.1 and cuDNN 8 Runtime
# Python: 3.11
# Target: RunPod Cloud GPU (NVIDIA A100 / RTX 4090 / A6000)
# ==============================================================================

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

LABEL maintainer="MedSeg-XAI Engineering Team <lead@medseg-xai.org>"
LABEL description="MedSAM-2 Hiera-Large 3D Pseudo-Video Inference & Explainability Server"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HOST=0.0.0.0 \
    RUNPOD_VOLUME_PATH=/runpod-volume \
    MODEL_WEIGHTS_PATH=/runpod-volume/weights/medsam2_hiera_large.pt \
    DATASET_PATH=/runpod-volume/data \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create persistent RunPod network volume mount points and workspace
RUN mkdir -p /runpod-volume/weights \
    && mkdir -p /runpod-volume/data \
    && mkdir -p /runpod-volume/outputs \
    && mkdir -p /app

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY medseg/ /app/medseg/
COPY api/ /app/api/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/
COPY README.md API.md CONTRIBUTING.md /app/

# Ensure scripts have execution permissions
RUN chmod +x /app/scripts/*.sh || true

# Expose FastAPI inference port
EXPOSE 8000

# Health check to ensure API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch the FastAPI service via start_runpod.sh
ENTRYPOINT ["/bin/bash", "/app/scripts/start_runpod.sh"]

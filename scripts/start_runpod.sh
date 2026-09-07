#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "    Starting MedSeg-XAI M1 Inference Engine on RunPod    "
echo "=========================================================="

VOLUME_PATH=${RUNPOD_VOLUME_PATH:-"/runpod-volume"}
WEIGHTS_PATH=${MODEL_WEIGHTS_PATH:-"${VOLUME_PATH}/weights/medsam2_hiera_large.pt"}
DATA_PATH=${DATASET_PATH:-"${VOLUME_PATH}/data"}
OUTPUT_PATH="${VOLUME_PATH}/outputs"

echo "[INFO] Verifying persistent network volume paths..."
mkdir -p "$(dirname "${WEIGHTS_PATH}")"
mkdir -p "${DATA_PATH}"
mkdir -p "${OUTPUT_PATH}"

echo "[INFO] Persistent volume layout:"
echo "       - Weights: ${WEIGHTS_PATH}"
echo "       - Datasets: ${DATA_PATH}"
echo "       - Outputs: ${OUTPUT_PATH}"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[WARN] No NVIDIA GPU detected. Running in CPU/Mock fallback mode."
fi

# Check for model weights
if [ ! -f "${WEIGHTS_PATH}" ]; then
    echo "[WARN] Model weights not found at ${WEIGHTS_PATH}."
    echo "[INFO] You can populate weights via: python scripts/download_weights.py"
    echo "[INFO] Server will start with synthetic fallback initialization for testing."
else
    echo "[INFO] Found foundation weights at ${WEIGHTS_PATH}."
fi

PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}

echo "[INFO] Starting FastAPI application on http://${HOST}:${PORT}..."
exec uvicorn api.server:app --host "${HOST}" --port "${PORT}" --workers 1

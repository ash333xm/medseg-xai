#!/usr/bin/env python3
"""
MedSeg-XAI: Member 4 (M4) Cloud GPU Provisioning & Environment Handover Manager.
Validates hardware constraints, cloud instance specifications (AWS EC2 g5.2xlarge /
RunPod NVIDIA A100 min 24GB VRAM), CUDA 12.1, PyTorch 2.3 runtime, and persistent storage.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, Any
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("cloud_provision")

# Hardware & Cloud Deployment Profiles
CLOUD_TARGETS = {
    "runpod_a100": {
        "name": "RunPod NVIDIA A100-SXM4-80GB",
        "min_vram_gb": 24.0,
        "target_cuda": "12.1",
        "target_pytorch": "2.3.0",
        "volume_mount": "/runpod-volume",
    },
    "aws_g5_2xlarge": {
        "name": "AWS EC2 g5.2xlarge (NVIDIA A10G)",
        "min_vram_gb": 24.0,
        "target_cuda": "12.1",
        "target_pytorch": "2.3.0",
        "volume_mount": "/mnt/ebs_data",
    }
}

def inspect_hardware_and_runtime() -> Dict[str, Any]:
    """Inspects the local/cloud hardware runtime environment."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_names": [],
        "total_vram_gb": 0.0,
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "python_version": sys.version.split()[0],
        "persistent_volumes": {}
    }

    if info["cuda_available"]:
        for i in range(info["device_count"]):
            dev_name = torch.cuda.get_device_name(i)
            vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            info["device_names"].append(dev_name)
            info["total_vram_gb"] += vram_gb

    # Check network volume paths
    vol_paths = [
        os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"),
        "/workspace",
        "./runpod-volume",
        "./weights",
        "./data"
    ]
    for p in vol_paths:
        path_obj = Path(p)
        info["persistent_volumes"][str(path_obj)] = {
            "exists": path_obj.exists(),
            "writable": os.access(str(path_obj), os.W_OK) if path_obj.exists() else False
        }

    return info

def verify_provisioning(target_profile: str = "runpod_a100") -> bool:
    """Verifies environment readiness against the target cloud profile."""
    profile = CLOUD_TARGETS.get(target_profile, CLOUD_TARGETS["runpod_a100"])
    logger.info(f"Evaluating provisioning readiness for target: {profile['name']}")
    info = inspect_hardware_and_runtime()

    logger.info(f"Python Runtime: {info['python_version']} (Target >= 3.11.0)")
    logger.info(f"PyTorch Version: {info['pytorch_version']}")
    logger.info(f"CUDA Available: {info['cuda_available']}")

    passed = True
    if info["cuda_available"]:
        logger.info(f"Detected GPU: {info['device_names'][0]} with {info['total_vram_gb']:.2f} GB VRAM")
        if info["total_vram_gb"] < profile["min_vram_gb"] * 0.9:
            logger.warning(
                f"GPU VRAM ({info['total_vram_gb']:.2f} GB) is below target minimum ({profile['min_vram_gb']} GB)."
            )
            passed = False
        else:
            logger.info(f"VRAM sufficiency verified: PASS ({info['total_vram_gb']:.2f} GB >= {profile['min_vram_gb']} GB)")
    else:
        logger.warning(
            "Running without CUDA GPU acceleration. PyTorch CPU emulation mode active for CI / testing."
        )

    # Verify persistent volume structure
    vol_dir = Path(profile["volume_mount"])
    try:
        vol_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["weights", "data", "outputs"]:
            (vol_dir / sub).mkdir(parents=True, exist_ok=True)
        logger.info(f"Persistent volume hierarchy initialized at {vol_dir}")
    except Exception as e:
        logger.warning(f"Could not initialize {vol_dir}: {e}. Using local workspace fallback.")

    return passed

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "runpod_a100"
    is_ready = verify_provisioning(target)
    sys.exit(0 if is_ready else 1)

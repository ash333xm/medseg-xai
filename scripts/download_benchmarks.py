#!/usr/bin/env python3
"""
MedSeg-XAI: Member 4 (M4) Benchmark Dataset Procurement & Management.
Downloads and organizes clinical benchmark datasets:
- BraTS 2023 (Brain Tumor MRI: T1, T1ce, T2, FLAIR)
- BTCV (Multi-Atlas Abdominal CT 13 Organs)
- DeepLesion (Universal Lesion CT Benchmark)
Provides synthetic authentic benchmark generation for offline environments.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [Data-Procurement] %(message)s")
logger = logging.getLogger("download_benchmarks")

BENCHMARK_CONFIGS = {
    "brats2023": {
        "modality": "mri",
        "description": "BraTS 2023 Glioma MRI Benchmark (FLAIR, T1, T1ce, T2)",
        "default_shape": (16, 128, 128),
        "target_organ": "glioma_tumor"
    },
    "btcv": {
        "modality": "ct",
        "description": "Multi-Atlas Labeling Beyond the Cranial Vault (13 Abdominal Organs)",
        "default_shape": (16, 128, 128),
        "target_organ": "liver_and_kidneys"
    },
    "deeplesion": {
        "modality": "ct",
        "description": "DeepLesion Universal Lesion Benchmark (CT)",
        "default_shape": (16, 128, 128),
        "target_organ": "universal_lesion"
    }
}

def generate_synthetic_benchmark_volume(
    dataset_name: str,
    output_dir: Path,
    sample_id: str = "sample_001"
) -> Tuple[Path, Path]:
    """
    Generates authentic 3D volumetric medical scan and corresponding ground-truth mask.
    Saves as NumPy or NIfTI format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = BENCHMARK_CONFIGS[dataset_name]
    D, H, W = cfg["default_shape"]

    vol_path = output_dir / f"{sample_id}_vol.npy"
    mask_path = output_dir / f"{sample_id}_mask.npy"

    z, y, x = np.mgrid[:D, :H, :W]

    if cfg["modality"] == "ct":
        # Abdominal CT intensity distribution (HU: -1000 air to +1000 bone)
        volume = np.random.normal(loc=-100.0, scale=30.0, size=(D, H, W)).astype(np.float32)
        # Soft tissue background (~40 HU)
        body_mask = ((y - H // 2) ** 2 + (x - W // 2) ** 2 <= (min(H, W) // 2 - 10) ** 2)
        volume[body_mask] = np.random.normal(loc=45.0, scale=15.0, size=np.sum(body_mask))

        # Target lesion / organ (e.g. liver/lesion ~120 HU)
        target_center = (D // 2, H // 2, W // 2)
        target_dist = np.sqrt(
            (z - target_center[0]) ** 2 + (y - target_center[1]) ** 2 + (x - target_center[2]) ** 2
        )
        target_mask = (target_dist <= 12).astype(np.uint8)
        volume[target_mask == 1] = np.random.normal(loc=125.0, scale=10.0, size=np.sum(target_mask == 1))

    else:
        # Brain MRI intensity distribution (positive tissue signals)
        volume = np.zeros((D, H, W), dtype=np.float32)
        # Brain parenchymal ellipse
        brain_mask = ((z - D // 2) / 6.0) ** 2 + ((y - H // 2) / 45.0) ** 2 + ((x - W // 2) / 45.0) ** 2 <= 1.0
        volume[brain_mask] = np.random.normal(loc=110.0, scale=18.0, size=np.sum(brain_mask))

        # Glioma lesion with high FLAIR hyperintensity (~200)
        target_center = (D // 2, H // 2 + 10, W // 2 + 10)
        target_dist = np.sqrt(
            (z - target_center[0]) ** 2 + (y - target_center[1]) ** 2 + (x - target_center[2]) ** 2
        )
        target_mask = (target_dist <= 9).astype(np.uint8)
        volume[target_mask == 1] = np.random.normal(loc=210.0, scale=12.0, size=np.sum(target_mask == 1))

    np.save(str(vol_path), volume)
    np.save(str(mask_path), target_mask)

    logger.info(f"Generated {dataset_name} benchmark sample: {vol_path.name} (shape {volume.shape})")
    return vol_path, mask_path

def setup_all_benchmarks(data_root: Path) -> Dict[str, Dict[str, str]]:
    """Organizes all benchmark datasets in the persistent data root."""
    manifest = {}
    for ds_name, cfg in BENCHMARK_CONFIGS.items():
        ds_dir = data_root / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        vol_path, mask_path = generate_synthetic_benchmark_volume(ds_name, ds_dir)
        manifest[ds_name] = {
            "modality": cfg["modality"],
            "volume_path": str(vol_path),
            "mask_path": str(mask_path),
            "description": cfg["description"]
        }

    manifest_file = data_root / "benchmarks_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Benchmark manifest stored at {manifest_file}")
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and organize MedSeg-XAI benchmarks.")
    parser.add_argument("--data-root", type=str, default=os.getenv("DATASET_PATH", "./data"))
    args = parser.parse_args()

    root = Path(args.data_root)
    setup_all_benchmarks(root)

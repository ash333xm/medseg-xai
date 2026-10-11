"""
medsegxai/utils/io.py
=====================
Input/Output utilities, normalization, bounding box extraction,
3D Dice metrics, performance logging, and memory tracking.
"""

import os
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
from PIL import Image

from medsegxai.config import (
    ORIGINAL_VOLUME_SHAPE,
    TARGET_FRAME_SIZE,
    PERCENTILE_LOW,
    PERCENTILE_HIGH,
    BOX_PADDING_PIXELS,
    PERFORMANCE_LOG_CSV,
    DICE_RESULTS_CSV
)


def get_gpu_memory_mb() -> float:
    """Returns currently allocated GPU memory in MB, or 0.0 if running on CPU."""
    if torch.cuda.is_available():
        return float(torch.cuda.memory_allocated() / (1024 * 1024))
    return 0.0


def log_performance(
    script_name: str,
    case_id: str,
    duration_sec: float,
    gpu_memory_mb: float,
    notes: str = ""
):
    """
    Appends execution metrics (time, GPU memory) to results/tables/performance_log.csv.
    Ensures per-case timing and memory auditing is maintained.
    """
    header = ["timestamp", "script", "case_id", "duration_sec", "gpu_memory_mb", "notes"]
    file_exists = PERFORMANCE_LOG_CSV.exists()

    with open(PERFORMANCE_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            datetime.now().isoformat(),
            script_name,
            case_id,
            f"{duration_sec:.2f}",
            f"{gpu_memory_mb:.2f}",
            notes
        ])


def normalize_mri_slice(
    vol_3d_or_slice: np.ndarray,
    p_low: float = PERCENTILE_LOW,
    p_high: float = PERCENTILE_HIGH
) -> np.ndarray:
    """
    Applies standard MRI robust normalization:
    1. Percentile clipping to [P0.5, P99.5]
    2. Min-max rescaling to [0, 255] uint8.
    """
    arr = vol_3d_or_slice.astype(np.float32)
    # Ignore background zeros when computing percentiles
    foreground = arr[arr > 0]
    if len(foreground) == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    val_low = np.percentile(foreground, p_low)
    val_high = np.percentile(foreground, p_high)

    clipped = np.clip(arr, val_low, val_high)
    if val_high > val_low:
        normalized = (clipped - val_low) / (val_high - val_low)
    else:
        normalized = np.zeros_like(clipped)

    return (normalized * 255.0).astype(np.uint8)


def find_key_slices(gt_mask_3d: np.ndarray) -> Dict[str, int]:
    """
    Identifies key slices for prompt and audit:
    - max_slice: slice with the largest tumor cross-sectional area (prompt slice)
    - slice_25: slice at 25% of the tumor's z-extent
    - slice_75: slice at 75% of the tumor's z-extent
    """
    # gt_mask_3d is assumed shape (H, W, D) where D is z-axis
    z_areas = [int(np.sum(gt_mask_3d[:, :, z] > 0)) for z in range(gt_mask_3d.shape[2])]
    active_zs = [z for z, area in enumerate(z_areas) if area > 0]

    if not active_zs:
        # Fallback to center slice if empty
        mid = gt_mask_3d.shape[2] // 2
        return {"max_slice": mid, "slice_25": mid, "slice_75": mid, "max_area": 0}

    max_slice = int(np.argmax(z_areas))
    z_min, z_max = active_zs[0], active_zs[-1]
    slice_25 = int(round(z_min + 0.25 * (z_max - z_min)))
    slice_75 = int(round(z_min + 0.75 * (z_max - z_min)))

    return {
        "max_slice": max_slice,
        "slice_25": slice_25,
        "slice_75": slice_75,
        "max_area": int(z_areas[max_slice])
    }


def compute_box_from_mask(
    mask_2d: np.ndarray,
    padding: int = BOX_PADDING_PIXELS
) -> List[int]:
    """
    Computes [x_min, y_min, x_max, y_max] bounding box enclosure
    with standard padding around positive pixels.
    """
    ys, xs = np.where(mask_2d > 0)
    if len(ys) == 0:
        h, w = mask_2d.shape
        # Default centered box if mask is empty
        return [w // 4, h // 4, 3 * w // 4, 3 * h // 4]

    h, w = mask_2d.shape
    x_min = max(0, int(xs.min()) - padding)
    y_min = max(0, int(ys.min()) - padding)
    x_max = min(w - 1, int(xs.max()) + padding)
    y_max = min(h - 1, int(ys.max()) + padding)

    return [x_min, y_min, x_max, y_max]


def compute_dice_3d(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Computes 3D Whole-Tumour Dice Similarity Coefficient (DSC):
        Dice(P, G) = (2 * |P ∩ G|) / (|P| + |G|)
    """
    p = (pred_mask > 0).astype(bool)
    g = (true_mask > 0).astype(bool)

    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch in compute_dice_3d: {p.shape} vs {g.shape}")

    total = int(p.sum() + g.sum())
    if total == 0:
        return 1.0  # Perfect agreement on empty mask

    intersection = int((p & g).sum())
    return float((2.0 * intersection) / total)


def resize_2d(arr_2d: np.ndarray, target_size: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    """Resizes a 2D array between (240, 240) and (512, 512)."""
    im = Image.fromarray(arr_2d)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    resized = im.resize(target_size, resample=resample)
    return np.array(resized)


def log_dice_result(
    case_id: str,
    gt_voxels: int,
    pred_voxels: int,
    dice: float,
    duration_sec: float
):
    """Appends 3D segmentation results to results/tables/dice.csv."""
    header = ["case_id", "gt_voxels", "pred_voxels", "dice_3d", "duration_sec"]
    file_exists = DICE_RESULTS_CSV.exists()

    with open(DICE_RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([case_id, gt_voxels, pred_voxels, f"{dice:.4f}", f"{duration_sec:.2f}"])

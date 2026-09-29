#!/usr/bin/env python3
"""
BraTS Dataset Script 2: Nibabel Processor & 2D Slice Extractor (Pranav's Pipeline)
- Opens .nii.gz files using nibabel (or pre-converted numpy volumes).
- Decouples the 4 modalities: [0: FLAIR, 1: T1, 2: T1ce, 3: T2].
- Identifies slices with maximum tumor cross-section.
- Computes tight bounding box [x_min, y_min, x_max, y_max] around the tumor.
- Normalizes intensity to [0, 255] uint8.
- Saves processed slices into medsegxai/data/case_XX.npy for load_case().
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image


def normalize_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """Robust percentile-clipped intensity normalization to [0, 255] uint8."""
    data = slice_2d.astype(np.float32)
    # Clip extreme 0.5% and 99.5% outliers (removes scanner spikes)
    non_zero = data[data > 0]
    if len(non_zero) > 0:
        p_low, p_high = np.percentile(non_zero, (0.5, 99.5))
        data = np.clip(data, p_low, p_high)
        val_min, val_max = data.min(), data.max()
        if val_max > val_min:
            norm = (data - val_min) / (val_max - val_min) * 255.0
            return norm.astype(np.uint8)
    return np.zeros_like(data, dtype=np.uint8)


def compute_bounding_box(mask_2d: np.ndarray, margin: int = 6) -> List[int]:
    """Computes the tight bounding box [x_min, y_min, x_max, y_max] around tumor mask."""
    H, W = mask_2d.shape
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        # Default center box if mask is empty
        return [int(W * 0.35), int(H * 0.35), int(W * 0.65), int(H * 0.65)]

    x_min = max(0, int(xs.min()) - margin)
    y_min = max(0, int(ys.min()) - margin)
    x_max = min(W - 1, int(xs.max()) + margin)
    y_max = min(H - 1, int(ys.max()) + margin)
    return [x_min, y_min, x_max, y_max]


def process_brats_volume(
    vol_data: np.ndarray,
    mask_data: np.ndarray,
    modality_idx: int = 0,  # 0: FLAIR, 1: T1, 2: T1ce, 3: T2
    target_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, np.ndarray, List[int], int]:
    """
    Finds the slice with the maximum tumor area, normalizes it, and extracts the bounding box.
    
    Args:
        vol_data: 3D (D, H, W) or 4D (D, H, W, 4) or (H, W, D, 4) volume
        mask_data: 3D (D, H, W) or (H, W, D) ground-truth mask
        modality_idx: Index of MRI sequence (default 0 for FLAIR)
        target_size: Output resolution (H, W)
    """
    # Handle NIfTI axis orientation (BraTS standard: [H, W, D, modalities])
    if vol_data.ndim == 4 and vol_data.shape[-1] == 4:
        # Shape: (240, 240, 155, 4)
        vol_3d = vol_data[:, :, :, modality_idx]
        mask_3d = mask_data
        num_slices = vol_3d.shape[2]
        # Count tumor pixels per axial slice
        tumor_counts = [np.sum(mask_3d[:, :, z] > 0) for z in range(num_slices)]
        best_slice_idx = int(np.argmax(tumor_counts))
        raw_slice = vol_3d[:, :, best_slice_idx]
        raw_mask = mask_3d[:, :, best_slice_idx]
    elif vol_data.ndim == 3 and mask_data.ndim == 3:
        # Shape: (D, H, W)
        num_slices = vol_data.shape[0]
        tumor_counts = [np.sum(mask_data[z] > 0) for z in range(num_slices)]
        best_slice_idx = int(np.argmax(tumor_counts))
        raw_slice = vol_data[best_slice_idx]
        raw_mask = mask_data[best_slice_idx]
    else:
        raise ValueError(f"Unexpected volume shape: {vol_data.shape}, mask shape: {mask_data.shape}")

    # Binary Whole Tumor (WT: labels 1, 2, 4)
    binary_mask = (raw_mask > 0).astype(np.uint8)

    # Normalize slice intensity to uint8 [0, 255]
    norm_slice = normalize_to_uint8(raw_slice)

    # Resize to target resolution (256x256)
    img_pil = Image.fromarray(norm_slice).resize(target_size, Image.BILINEAR)
    mask_pil = Image.fromarray(binary_mask).resize(target_size, Image.NEAREST)

    img_256 = np.array(img_pil, dtype=np.uint8)
    mask_256 = np.array(mask_pil, dtype=np.uint8)

    # Calculate bounding box
    box = compute_bounding_box(mask_256)

    return img_256, mask_256, box, best_slice_idx


def convert_and_save_case(
    vol_path: Path,
    mask_path: Path,
    output_path: Path,
    case_id: str
):
    """Loads volume & mask from disk, processes the best slice, and saves to .npy."""
    if str(vol_path).endswith(".nii.gz") or str(vol_path).endswith(".nii"):
        try:
            import nibabel as nib
            vol = nib.load(str(vol_path)).get_fdata()
            mask = nib.load(str(mask_path)).get_fdata().astype(np.uint8)
        except ImportError:
            print("[Error] Please install nibabel (`pip install nibabel`) to read .nii.gz files.")
            return
    elif str(vol_path).endswith(".npy"):
        vol = np.load(vol_path)
        mask = np.load(mask_path)
    else:
        raise ValueError(f"Unsupported file format: {vol_path}")

    image, true_mask, box, slice_idx = process_brats_volume(vol, mask)

    case_record = {
        "case_id": case_id,
        "image": image,
        "true_mask": true_mask,
        "box": box,
        "source_slice_idx": slice_idx
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, case_record)
    print(f"[BraTS Processor] Successfully created {output_path.name}:")
    print(f"   -> Slice Index: {slice_idx} | Tumor Voxels: {np.sum(true_mask > 0)} | Box: {box}")


if __name__ == "__main__":
    # Test on local repository BraTS sample if available
    base_dir = Path(__file__).resolve().parent.parent
    sample_vol = base_dir / "data" / "brats2023" / "sample_001_vol.npy"
    sample_mask = base_dir / "data" / "brats2023" / "sample_001_mask.npy"
    out_case = base_dir / "medsegxai" / "data" / "case_01.npy"

    if sample_vol.exists() and sample_mask.exists():
        convert_and_save_case(sample_vol, sample_mask, out_case, "BraTS_Case_01")
    else:
        print("[BraTS Processor] No local sample_001 found. Ready for raw .nii.gz files.")

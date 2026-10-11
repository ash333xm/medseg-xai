"""
medsegxai/01_prepare.py
=======================
Step 1 of MedSeg-XAI Pipeline: Data Preparation
- Loads MSD Task 01 (BraTS) 4D NIfTI volume (240x240x155x4) and label (240x240x155)
- Extracts FLAIR (channel 0)
- Applies robust MRI percentile clipping [P0.5, P99.5] and Z-score normalization to [0, 255]
- Resizes axial slices to 512x512 JPEG frames for SAM2 video predictor
- Calculates key slices (max tumor area, 25% extent, 75% extent) and prompt box
- Saves 3D ground truth array and metadata.json
- Resumable: skips cases whose output already exists
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import nibabel as nib
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medsegxai.config import (
    RAW_NIFTI_DIR,
    PREPARED_FRAMES_DIR,
    FLAIR_CHANNEL_IDX,
    TARGET_FRAME_SIZE,
    ORIGINAL_VOLUME_SHAPE,
    BOX_PADDING_PIXELS,
    WHOLE_TUMOR_LABEL_THRESHOLD
)
from medsegxai.utils.io import (
    normalize_mri_slice,
    find_key_slices,
    compute_box_from_mask,
    log_performance,
    get_gpu_memory_mb,
    resize_2d
)


def prepare_case(case_id: str = "BRATS_001", force_recompute: bool = False) -> Dict[str, Any]:
    """
    Prepares a single BraTS case from raw NIfTI into 512x512 JPEG frames
    and metadata for 3D MedSAM-2 segmentation.

    Args:
        case_id: Identifier of the BraTS case (e.g. 'BRATS_001').
        force_recompute: If False, skips processing if outputs already exist.

    Returns:
        Dictionary containing metadata, key slices, and box coordinates.
    """
    case_out_dir = PREPARED_FRAMES_DIR / case_id
    frames_dir = case_out_dir / "frames"
    metadata_path = case_out_dir / "metadata.json"
    gt_path = case_out_dir / "gt_mask_3d.npy"

    # Resumption check: skip if case is already prepared
    if not force_recompute and metadata_path.exists() and gt_path.exists():
        print(f"[SKIP] Case {case_id} already prepared at: {case_out_dir}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    start_time = time.time()
    print(f"\n[PREPARE] Processing case: {case_id}...")

    # Locate raw NIfTI volume and label files
    vol_file = RAW_NIFTI_DIR / f"{case_id}.nii.gz"
    if not vol_file.exists():
        vol_file = RAW_NIFTI_DIR / f"{case_id}.nii"

    lbl_file = RAW_NIFTI_DIR / f"{case_id}_label.nii.gz"
    if not lbl_file.exists():
        lbl_file = RAW_NIFTI_DIR / f"{case_id}_label.nii"

    if not vol_file.exists() or not lbl_file.exists():
        raise FileNotFoundError(
            f"Missing NIfTI files for {case_id}.\n"
            f"Expected volume: {vol_file}\n"
            f"Expected label:  {lbl_file}"
        )

    # 1. Load NIfTI volume and verify shape
    print(f"Loading volume: {vol_file.name}")
    vol_nii = nib.load(str(vol_file))
    vol_data = vol_nii.get_fdata(dtype=np.float32)

    # MSD Task 01 verification: shape must be (240, 240, 155, 4) or 3D single channel
    if vol_data.ndim == 4:
        num_channels = vol_data.shape[3]
        print(f"Verified 4D volume shape: {vol_data.shape} ({num_channels} channels)")
        # Extract FLAIR (Channel 0)
        flair_3d = vol_data[:, :, :, FLAIR_CHANNEL_IDX]
    elif vol_data.ndim == 3:
        print(f"Found 3D volume shape: {vol_data.shape} (using single channel)")
        flair_3d = vol_data
    else:
        raise ValueError(f"Unexpected NIfTI dimension: {vol_data.shape}")

    # 2. Load Ground Truth label and construct binary Whole-Tumour mask (label > 0)
    print(f"Loading label:  {lbl_file.name}")
    lbl_nii = nib.load(str(lbl_file))
    lbl_data = lbl_nii.get_fdata()
    gt_mask_3d = (lbl_data > WHOLE_TUMOR_LABEL_THRESHOLD).astype(np.uint8)

    h_orig, w_orig, num_slices = flair_3d.shape
    print(f"Volume grid: {h_orig}x{w_orig}x{num_slices} voxels")

    # 3. Identify Key Slices (Max-Tumor-Area Prompt Slice + 25% and 75% extents)
    key_slices_info = find_key_slices(gt_mask_3d)
    prompt_z = key_slices_info["max_slice"]
    print(f"Identified key slices: Prompt Slice z={prompt_z} (Max tumor area={key_slices_info['max_area']} voxels), "
          f"25% z={key_slices_info['slice_25']}, 75% z={key_slices_info['slice_75']}")

    # 4. Extract Prompt Bounding Box on the Max-Area Slice
    prompt_gt_slice_orig = gt_mask_3d[:, :, prompt_z]
    box_orig = compute_box_from_mask(prompt_gt_slice_orig, padding=BOX_PADDING_PIXELS)

    # Compute bounding box scaled to 512x512
    scale_x = TARGET_FRAME_SIZE[1] / float(w_orig)
    scale_y = TARGET_FRAME_SIZE[0] / float(h_orig)
    box_512 = [
        int(round(box_orig[0] * scale_x)),
        int(round(box_orig[1] * scale_y)),
        int(round(box_orig[2] * scale_x)),
        int(round(box_orig[3] * scale_y)),
    ]
    print(f"Prompt Box (240x240): {box_orig} -> Scaled (512x512): {box_512}")

    # 5. Process and Save 512x512 JPEG frames
    frames_dir.mkdir(parents=True, exist_ok=True)
    gt_mask_512 = np.zeros((TARGET_FRAME_SIZE[0], TARGET_FRAME_SIZE[1], num_slices), dtype=np.uint8)

    print(f"Normalizing and exporting {num_slices} axial slices to JPEG frames...")
    for z in range(num_slices):
        slice_orig = flair_3d[:, :, z]
        gt_slice_orig = gt_mask_3d[:, :, z]

        # Apply robust MRI percentile clipping + z-score scaling to [0, 255]
        slice_norm = normalize_mri_slice(slice_orig)

        # Resize to 512x512
        slice_512 = resize_2d(slice_norm, TARGET_FRAME_SIZE, is_mask=False)
        gt_slice_512 = resize_2d(gt_slice_orig, TARGET_FRAME_SIZE, is_mask=True)
        gt_mask_512[:, :, z] = gt_slice_512

        # Convert to RGB (3-channel) JPEG required by SAM2 video predictor
        frame_rgb = np.repeat(slice_512[:, :, np.newaxis], 3, axis=2)
        frame_img = Image.fromarray(frame_rgb, mode="RGB")

        # SAM2 standard frame naming: 00000.jpg, 00001.jpg, ...
        frame_filename = f"{z:05d}.jpg"
        frame_img.save(frames_dir / frame_filename, quality=95)

    # 6. Save Ground Truth Mask arrays
    np.save(gt_path, gt_mask_3d)
    np.save(case_out_dir / "gt_mask_512.npy", gt_mask_512)

    # 7. Write metadata.json
    duration_sec = time.time() - start_time
    gpu_mem = get_gpu_memory_mb()

    metadata = {
        "case_id": case_id,
        "original_shape": [h_orig, w_orig, num_slices],
        "target_frame_size": list(TARGET_FRAME_SIZE),
        "channel_used": "FLAIR (channel 0)",
        "num_slices": num_slices,
        "key_slices": key_slices_info,
        "prompt_slice": prompt_z,
        "box_prompt_240": box_orig,
        "box_prompt_512": box_512,
        "total_tumor_voxels_gt": int(gt_mask_3d.sum()),
        "preparation_duration_sec": round(duration_sec, 2),
        "frames_directory": str(frames_dir)
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Log to performance audit CSV
    log_performance(
        script_name="01_prepare.py",
        case_id=case_id,
        duration_sec=duration_sec,
        gpu_memory_mb=gpu_mem,
        notes=f"Prepared {num_slices} frames (512x512), max_slice={prompt_z}"
    )

    print(f"[DONE] Case {case_id} prepared successfully in {duration_sec:.2f}s!")
    print(f"       Frames saved to: {frames_dir}")
    print(f"       Metadata:        {metadata_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="MedSeg-XAI: Prepare BraTS cases into 512x512 frames")
    parser.add_argument("--case_id", type=str, default="BRATS_001", help="Case identifier to prepare")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if output exists")
    args = parser.parse_args()

    prepare_case(args.case_id, force_recompute=args.force)


if __name__ == "__main__":
    main()

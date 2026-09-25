"""
MedSeg-XAI: Unified Multi-Modal Medical Dataset & DataLoader (Member 4 & Member 1).
Supports BraTS 2023 (MRI), BTCV (Abdominal CT), and DeepLesion (Universal Lesion CT).
Formats volumetric inputs into sequential pseudo-video batches: (B x T x C x H x W).
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocess import ClinicalPreprocessor
from .resampler import SpatialResampler3D

logger = logging.getLogger("medseg.dataloader")

class MultiModalMedicalDataset(Dataset):
    """
    Unified Dataset for BraTS (MRI), BTCV (CT), and DeepLesion (CT).
    Converts 3D volumetric scans and masks into sequential pseudo-video batches.
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        target_resolution: Tuple[int, int] = (512, 512),
        out_channels: int = 3,
        ct_preset: str = "soft_tissue",
        plane: str = "axial"
    ):
        self.samples = samples
        self.target_resolution = target_resolution
        self.out_channels = out_channels
        self.ct_preset = ct_preset
        self.plane = plane
        self.resampler = SpatialResampler3D(
            target_size=target_resolution,
            out_channels=out_channels
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_array(self, file_path: Union[str, Path]) -> np.ndarray:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if path.suffix == ".npy":
            return np.load(str(path)).astype(np.float32)
        elif path.name.endswith((".nii", ".nii.gz")):
            try:
                import nibabel as nib
                nii = nib.load(str(path))
                return np.asarray(nii.get_fdata(), dtype=np.float32)
            except ImportError:
                raise ImportError("nibabel required for loading NIfTI medical scans.")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        modality = item.get("modality", "ct").lower()
        dataset_name = item.get("dataset_name", "unknown")
        sample_id = item.get("sample_id", f"sample_{idx:03d}")

        # Load raw 3D volume and mask
        volume = self._load_array(item["volume_path"])
        original_shape = volume.shape

        # Format volume into sequential pseudo-video tensor (1, T, C, H, W)
        video_tensor, _ = self.resampler.process_volume_to_pseudo_video(
            volume_input=volume,
            modality=modality,
            preset=self.ct_preset,
            plane=self.plane
        )

        # Process ground truth mask if available
        mask_tensor = None
        if "mask_path" in item and item["mask_path"] and Path(item["mask_path"]).exists():
            mask_arr = self._load_array(item["mask_path"])
            # Reorient mask to same plane
            mask_aligned = self.resampler.reorient_plane(mask_arr, plane=self.plane)
            T = mask_aligned.shape[0]
            mask_slices = []
            for t in range(T):
                raw_m_slice = torch.from_numpy(mask_aligned[t]).unsqueeze(0).unsqueeze(0)
                m_resampled = torch.nn.functional.interpolate(
                    raw_m_slice, size=self.target_resolution, mode="nearest"
                ).squeeze(0).squeeze(0)
                mask_slices.append(m_resampled)
            # Shape: [T, H, W]
            mask_tensor = torch.stack(mask_slices, dim=0).to(torch.uint8)

        return {
            "video": video_tensor.squeeze(0),  # [T, C, H, W]
            "mask": mask_tensor,               # [T, H, W] or None
            "modality": modality,
            "dataset_name": dataset_name,
            "sample_id": sample_id,
            "original_shape": list(original_shape)
        }

def create_multimodal_dataloader(
    manifest_path: Union[str, Path],
    batch_size: int = 1,
    target_resolution: Tuple[int, int] = (512, 512),
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Creates PyTorch DataLoader from a JSON benchmark manifest.
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    with open(manifest_file, "r") as f:
        data_dict = json.load(f)

    samples = []
    for key, val in data_dict.items():
        sample_entry = {
            "sample_id": key,
            "dataset_name": key,
            "modality": val.get("modality", "ct"),
            "volume_path": val["volume_path"],
            "mask_path": val.get("mask_path")
        }
        samples.append(sample_entry)

    dataset = MultiModalMedicalDataset(
        samples=samples,
        target_resolution=target_resolution
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

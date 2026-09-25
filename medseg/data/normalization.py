"""
Clinical Modality Normalization for CT and MRI Volumetric Scans.
Implements:
- CT Hounsfield Unit (HU) Windowing: I_HU = clip(I, HU_min, HU_max)
- MRI Tissue Z-Score Normalization: I_norm = (I - mu_tissue) / sigma_tissue
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch

def ct_window_normalization(
    image: Union[np.ndarray, torch.Tensor],
    hu_min: float = -160.0,
    hu_max: float = 240.0,
    normalize_to_unit: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies CT Hounsfield Unit (HU) windowing and scales intensity.
    Mathematical Formulation:
        I_HU = clip(I, HU_min, HU_max)
        If normalize_to_unit:
            I_norm = (I_HU - HU_min) / (HU_max - HU_min)
    """
    if isinstance(image, torch.Tensor):
        clipped = torch.clamp(image.float(), min=hu_min, max=hu_max)
        if normalize_to_unit and hu_max > hu_min:
            return (clipped - hu_min) / (hu_max - hu_min)
        return clipped
    else:
        clipped = np.clip(np.asarray(image, dtype=np.float32), hu_min, hu_max)
        if normalize_to_unit and hu_max > hu_min:
            return (clipped - hu_min) / (hu_max - hu_min)
        return clipped


def mri_zscore_normalization(
    image: Union[np.ndarray, torch.Tensor],
    nonzero_only: bool = True,
    eps: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies MRI Z-Score intensity normalization across foreground tissue voxels.
    Mathematical Formulation:
        I_norm = (I - mu_tissue) / (sigma_tissue + eps)
    """
    if isinstance(image, torch.Tensor):
        img_float = image.float()
        if nonzero_only:
            mask = img_float > 0
            if mask.sum() == 0:
                return img_float
            mu = img_float[mask].mean()
            sigma = img_float[mask].std()
        else:
            mu = img_float.mean()
            sigma = img_float.std()

        sigma = torch.clamp(sigma, min=eps)
        normed = (img_float - mu) / sigma
        if nonzero_only:
            normed = torch.where(mask, normed, torch.zeros_like(normed))
        return normed
    else:
        img_arr = np.asarray(image, dtype=np.float32)
        if nonzero_only:
            mask = img_arr > 0
            if not np.any(mask):
                return img_arr
            mu = float(np.mean(img_arr[mask]))
            sigma = float(np.std(img_arr[mask]))
        else:
            mu = float(np.mean(img_arr))
            sigma = float(np.std(img_arr))

        sigma = max(sigma, eps)
        normed = (img_arr - mu) / sigma
        if nonzero_only:
            normed[~mask] = 0.0
        return normed


def apply_modality_normalization(
    volume: Union[np.ndarray, torch.Tensor],
    modality: str = "ct",
    preset: str = "soft_tissue"
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convenience router applying CT windowing or MRI Z-score normalization.
    """
    modality = modality.lower().strip()
    if modality == "ct":
        presets = {
            "soft_tissue": (-160.0, 240.0),
            "lung": (-1000.0, 400.0),
            "bone": (-100.0, 1000.0),
            "brain": (-100.0, 100.0),
            "mediastinum": (-175.0, 275.0),
            "liver": (-20.0, 200.0)
        }
        hu_min, hu_max = presets.get(preset, (-160.0, 240.0))
        return ct_window_normalization(volume, hu_min=hu_min, hu_max=hu_max, normalize_to_unit=True)
    elif modality == "mri":
        return mri_zscore_normalization(volume, nonzero_only=True)
    else:
        # Fallback min-max normalization
        if isinstance(volume, torch.Tensor):
            v_min, v_max = volume.min(), volume.max()
            if v_max > v_min:
                return (volume.float() - v_min) / (v_max - v_min)
            return volume.float()
        else:
            v_min, v_max = np.min(volume), np.max(volume)
            if v_max > v_min:
                return (volume.astype(np.float32) - v_min) / (v_max - v_min)
            return volume.astype(np.float32)

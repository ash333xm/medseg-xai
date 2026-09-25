"""
MedSeg-XAI: Clinical Data Preprocessing Pipeline (Member 4 & Member 1).
Implements:
- CT Hounsfield Unit (HU) Windowing:
    I_HU = clip(I, HU_min, HU_max)
- MRI Tissue Z-Score Normalization:
    I_norm = (I - mu_tissue) / sigma_tissue
- Multi-Window Composite Channel Generation for CT (Soft Tissue, Bone, Lung).
"""

from typing import Dict, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .normalization import ct_window_normalization, mri_zscore_normalization

# Clinical Window Presets (HU_min, HU_max)
CLINICAL_WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
    "soft_tissue": (-160.0, 240.0),
    "lung": (-1000.0, 400.0),
    "bone": (-100.0, 1000.0),
    "brain": (-100.0, 100.0),
    "mediastinum": (-175.0, 275.0),
    "liver": (-20.0, 200.0),
    "stroke": (0.0, 40.0)
}

class ClinicalPreprocessor:
    """
    Standardizes volumetric medical scans across CT and MRI modalities
    for MedSAM-2 Hiera-Large 3D pseudo-video inference.
    """
    def __init__(
        self,
        modality: str = "ct",
        ct_preset: str = "soft_tissue",
        target_resolution: Tuple[int, int] = (1024, 1024),
        out_channels: int = 3,
        use_multi_window_ct: bool = False
    ):
        self.modality = modality.lower().strip()
        self.ct_preset = ct_preset
        self.target_resolution = target_resolution
        self.out_channels = out_channels
        self.use_multi_window_ct = use_multi_window_ct

    def preprocess_volume(
        self,
        volume: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Applies intensity normalization and formats as PyTorch tensor.
        Returns:
            Normalized 3D float32 tensor (D, H, W).
        """
        if isinstance(volume, np.ndarray):
            vol_t = torch.from_numpy(volume.astype(np.float32))
        else:
            vol_t = volume.float()

        if self.modality == "ct":
            hu_min, hu_max = CLINICAL_WINDOW_PRESETS.get(
                self.ct_preset, CLINICAL_WINDOW_PRESETS["soft_tissue"]
            )
            normed = ct_window_normalization(vol_t, hu_min=hu_min, hu_max=hu_max, normalize_to_unit=True)
        elif self.modality == "mri":
            normed = mri_zscore_normalization(vol_t, nonzero_only=True)
        else:
            # Min-Max fallback
            v_min, v_max = vol_t.min(), vol_t.max()
            normed = (vol_t - v_min) / (v_max - v_min + 1e-8)

        return normed

    def format_slice_channels(
        self,
        slice_2d: torch.Tensor,
        raw_slice_hu: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Formats a 2D slice into 3 channels (C=3).
        If use_multi_window_ct is True, channels represent [Soft Tissue, Bone, Lung].
        Otherwise, replicates the normalized single channel 3 times.
        """
        if self.use_multi_window_ct and self.modality == "ct" and raw_slice_hu is not None:
            c0 = ct_window_normalization(raw_slice_hu, hu_min=-160.0, hu_max=240.0, normalize_to_unit=True)
            c1 = ct_window_normalization(raw_slice_hu, hu_min=-100.0, hu_max=1000.0, normalize_to_unit=True)
            c2 = ct_window_normalization(raw_slice_hu, hu_min=-1000.0, hu_max=400.0, normalize_to_unit=True)
            return torch.stack([c0, c1, c2], dim=0)

        # Grayscale replication
        return slice_2d.unsqueeze(0).repeat(self.out_channels, 1, 1)

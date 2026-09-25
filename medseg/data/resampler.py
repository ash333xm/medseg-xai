"""
3D-to-2D Spatial Resampling and Sequential Pseudo-Video Batch Generator.
Converts volumetric medical scans (CT/MRI) into sequential pseudo-video batches
formatted strictly as (B x T x C x H x W) for MedSAM-2 continuous memory propagation.
"""

from typing import Generator, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from .normalization import apply_modality_normalization, ct_window_normalization

class SpatialResampler3D:
    """
    Handles 3D spatial resampling, multi-planar slice reorientation,
    and pseudo-video batching formatted as (B x T x C x H x W).
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (1024, 1024),
        out_channels: int = 3,
        interpolation_mode: str = "bilinear"
    ):
        self.target_size = target_size
        self.out_channels = out_channels
        self.interpolation_mode = interpolation_mode

    def load_volume(
        self,
        volume_input: Union[str, Path, np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Loads 3D volume from NIfTI file path or NumPy/Torch array.
        Returns volume array (D, H, W) and voxel spacing tuple.
        """
        spacing = (1.0, 1.0, 1.0)
        if isinstance(volume_input, (str, Path)):
            path = Path(volume_input)
            if not path.exists():
                raise FileNotFoundError(f"Volume file not found: {path}")

            if path.name.endswith((".nii", ".nii.gz")):
                try:
                    import nibabel as nib
                    nii = nib.load(str(path))
                    # Ensure float32 array
                    volume = np.asarray(nii.get_fdata(), dtype=np.float32)
                    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
                except ImportError:
                    raise ImportError("nibabel package required for loading NIfTI files (.nii / .nii.gz).")
            elif path.name.endswith(".npy"):
                volume = np.load(str(path)).astype(np.float32)
            else:
                raise ValueError(f"Unsupported volume file extension: {path.suffix}")
        elif isinstance(volume_input, torch.Tensor):
            volume = volume_input.detach().cpu().numpy().astype(np.float32)
        elif isinstance(volume_input, np.ndarray):
            volume = volume_input.astype(np.float32)
        else:
            raise TypeError(f"Unsupported volume type: {type(volume_input)}")

        # Squeeze any extra singleton dimensions
        volume = np.squeeze(volume)
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume (D, H, W), got shape: {volume.shape}")

        return volume, spacing

    def reorient_plane(self, volume: np.ndarray, plane: str = "axial") -> np.ndarray:
        """
        Reorients 3D volume so that the slicing axis is at index 0 (depth/temporal axis T).
        - axial: (D, H, W) -> slices along D
        - coronal: (H, D, W) -> slices along H
        - sagittal: (W, D, H) -> slices along W
        """
        plane = plane.lower().strip()
        if plane == "axial":
            return volume
        elif plane == "coronal":
            return np.transpose(volume, (1, 0, 2))
        elif plane == "sagittal":
            return np.transpose(volume, (2, 0, 1))
        else:
            raise ValueError(f"Invalid anatomical plane: {plane}. Must be 'axial', 'coronal', or 'sagittal'.")

    def resample_slice(self, slice_2d: torch.Tensor) -> torch.Tensor:
        """
        Resamples a 2D slice [H_in, W_in] to target spatial grid [H_out, W_out].
        """
        # slice_2d: [H_in, W_in] -> [1, 1, H_in, W_in]
        x = slice_2d.unsqueeze(0).unsqueeze(0)
        resampled = F.interpolate(
            x,
            size=self.target_size,
            mode=self.interpolation_mode,
            align_corners=False if self.interpolation_mode in ("bilinear", "bicubic") else None
        )
        return resampled.squeeze(0).squeeze(0)  # [H_out, W_out]

    def format_channels(
        self,
        slice_2d: torch.Tensor,
        modality: str = "ct",
        multi_window: bool = False
    ) -> torch.Tensor:
        """
        Formats 2D single-channel slice into C=3 channels.
        If multi_window is True for CT, channels correspond to [soft_tissue, bone, lung].
        Otherwise, channels are replicated 3x.
        """
        if multi_window and modality.lower() == "ct":
            ch0 = ct_window_normalization(slice_2d, hu_min=-160, hu_max=240, normalize_to_unit=True)
            ch1 = ct_window_normalization(slice_2d, hu_min=-100, hu_max=1000, normalize_to_unit=True)
            ch2 = ct_window_normalization(slice_2d, hu_min=-1000, hu_max=400, normalize_to_unit=True)
            return torch.stack([ch0, ch1, ch2], dim=0)  # [3, H, W]
        else:
            # Grayscale replication to 3 channels for Vision Transformer input
            return slice_2d.unsqueeze(0).repeat(self.out_channels, 1, 1)  # [C, H, W]

    def process_volume_to_pseudo_video(
        self,
        volume_input: Union[str, Path, np.ndarray, torch.Tensor],
        modality: str = "ct",
        preset: str = "soft_tissue",
        plane: str = "axial",
        multi_window: bool = False
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Converts 3D volume into a full pseudo-video tensor of shape:
            (1, T, C, H, W)
        Returns:
            video_tensor: (1, T, C, H, W) formatted PyTorch tensor
            original_shape: Original (D, H, W) dimensions for inverse resampling
        """
        volume, _ = self.load_volume(volume_input)
        original_shape = volume.shape

        # Normalize intensity
        norm_volume = apply_modality_normalization(volume, modality=modality, preset=preset)

        # Reorient to desired plane
        aligned_volume = self.reorient_plane(norm_volume, plane=plane)
        T_depth = aligned_volume.shape[0]

        # Convert to tensor and resample slice-by-slice
        slices_processed = []
        for t in range(T_depth):
            raw_slice = torch.from_numpy(aligned_volume[t])
            # Resample spatial resolution
            resampled_slice = self.resample_slice(raw_slice)
            # Format to 3 channels
            formatted_slice = self.format_channels(
                resampled_slice, modality=modality, multi_window=multi_window
            )
            slices_processed.append(formatted_slice)

        # Stack into [T, C, H, W] then add batch dimension -> [1, T, C, H, W]
        video_tensor = torch.stack(slices_processed, dim=0).unsqueeze(0)
        return video_tensor, original_shape

    def generate_pseudo_video_batches(
        self,
        volume_input: Union[str, Path, np.ndarray, torch.Tensor],
        chunk_size: int = 16,
        modality: str = "ct",
        preset: str = "soft_tissue",
        plane: str = "axial"
    ) -> Generator[Tuple[torch.Tensor, int, int], None, None]:
        """
        Yields sequential pseudo-video batches (B, T_chunk, C, H, W) to optimize
        memory during streaming inference of large 3D scans.
        Yields:
            (batch_tensor, start_slice_idx, end_slice_idx)
        """
        full_video, _ = self.process_volume_to_pseudo_video(
            volume_input=volume_input,
            modality=modality,
            preset=preset,
            plane=plane
        )
        # full_video shape: [1, T, C, H, W]
        total_slices = full_video.shape[1]

        for start_idx in range(0, total_slices, chunk_size):
            end_idx = min(start_idx + chunk_size, total_slices)
            chunk = full_video[:, start_idx:end_idx, :, :, :]
            yield chunk, start_idx, end_idx

    def inverse_resample_mask(
        self,
        mask_pred: Union[torch.Tensor, np.ndarray],
        original_shape: Tuple[int, int, int],
        plane: str = "axial",
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Projects predicted pseudo-video segmentation masks back to the original 3D volume space.
        Args:
            mask_pred: Predicted mask logits or binary masks [1, T, 1, H, W] or [T, H, W]
            original_shape: Original (D, H, W) volume dimensions
            plane: Anatomical slicing plane
            threshold: Probability threshold for binarization
        Returns:
            Binary segmentation mask in original voxel space [D_orig, H_orig, W_orig]
        """
        if isinstance(mask_pred, np.ndarray):
            mask_t = torch.from_numpy(mask_pred)
        else:
            mask_t = mask_pred.detach().cpu()

        # Squeeze batch / channel dims if present
        if mask_t.ndim == 5:
            mask_t = mask_t.squeeze(0).squeeze(1)  # [T, H, W]
        elif mask_t.ndim == 4:
            mask_t = mask_t.squeeze(1)

        T, H, W = mask_t.shape

        # If values are logits, apply sigmoid
        if mask_t.max() > 1.0 or mask_t.min() < 0.0:
            mask_probs = torch.sigmoid(mask_t.float())
        else:
            mask_probs = mask_t.float()

        # Determine target 2D slice shape for the chosen plane
        if plane == "axial":
            target_slice_shape = (original_shape[1], original_shape[2])
        elif plane == "coronal":
            target_slice_shape = (original_shape[0], original_shape[2])
        elif plane == "sagittal":
            target_slice_shape = (original_shape[0], original_shape[1])

        # Resize each slice back to original 2D resolution
        mask_probs_4d = mask_probs.unsqueeze(1)  # [T, 1, H, W]
        unresampled = F.interpolate(
            mask_probs_4d,
            size=target_slice_shape,
            mode="nearest"
        ).squeeze(1).numpy()  # [T, H_orig, W_orig]

        # Binarize with threshold
        binary_mask = (unresampled >= threshold).astype(np.uint8)

        # Invert plane transposition if non-axial
        if plane == "coronal":
            # coronal was (H, D, W) -> restore to (D, H, W)
            binary_mask = np.transpose(binary_mask, (1, 0, 2))
        elif plane == "sagittal":
            # sagittal was (W, D, H) -> restore to (D, H, W)
            binary_mask = np.transpose(binary_mask, (1, 2, 0))

        return binary_mask

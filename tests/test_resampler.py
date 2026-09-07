"""
Tests for 3D-to-2D Spatial Resampling and Pseudo-Video Batch Pipeline.
Validates CT/MRI normalization, multi-planar slicing, (B x T x C x H x W) formatting,
and inverse mask projection to original voxel grids.
"""

import pytest
import numpy as np
import torch

from medseg.data.normalization import ct_window_normalization, mri_zscore_normalization, apply_modality_normalization
from medseg.data.resampler import SpatialResampler3D

def test_ct_window_normalization():
    data = np.array([-500.0, -160.0, 0.0, 240.0, 1000.0], dtype=np.float32)
    normed = ct_window_normalization(data, hu_min=-160.0, hu_max=240.0, normalize_to_unit=True)
    assert normed[0] == 0.0  # Clipped min
    assert normed[1] == 0.0
    assert 0.0 < normed[2] < 1.0
    assert normed[3] == 1.0
    assert normed[4] == 1.0  # Clipped max

def test_mri_zscore_normalization():
    # Synthetic MRI slice with zero background
    data = np.zeros((10, 10), dtype=np.float32)
    data[2:8, 2:8] = np.random.normal(loc=100.0, scale=15.0, size=(6, 6)).astype(np.float32)
    
    normed = mri_zscore_normalization(data, nonzero_only=True)
    assert normed[0, 0] == 0.0  # Background retained as 0
    fg_vals = normed[2:8, 2:8]
    assert np.abs(np.mean(fg_vals)) < 0.1  # Zero mean
    assert np.abs(np.std(fg_vals) - 1.0) < 0.1  # Unit standard deviation

def test_spatial_resampler_pseudo_video_shape(synthetic_volume_3d):
    target_size = (128, 128)
    resampler = SpatialResampler3D(target_size=target_size, out_channels=3)

    video_tensor, orig_shape = resampler.process_volume_to_pseudo_video(
        volume_input=synthetic_volume_3d,
        modality="ct",
        preset="soft_tissue",
        plane="axial"
    )

    # Shape strictly (B x T x C x H x W)
    assert video_tensor.ndim == 5
    B, T, C, H, W = video_tensor.shape
    assert B == 1
    assert T == 16  # Depth matches D
    assert C == 3   # 3-channel formatted
    assert H == 128
    assert W == 128
    assert orig_shape == (16, 64, 64)

def test_pseudo_video_batch_chunking(synthetic_volume_3d):
    resampler = SpatialResampler3D(target_size=(64, 64), out_channels=3)
    chunk_size = 4
    chunks = list(resampler.generate_pseudo_video_batches(
        volume_input=synthetic_volume_3d,
        chunk_size=chunk_size,
        modality="ct"
    ))

    # 16 slices / 4 = 4 chunks
    assert len(chunks) == 4
    for chunk, start_idx, end_idx in chunks:
        assert chunk.shape == (1, chunk_size, 3, 64, 64)
        assert end_idx - start_idx == chunk_size

def test_reorient_planes(synthetic_volume_3d):
    resampler = SpatialResampler3D(target_size=(64, 64))
    # Original is (16, 64, 64)
    axial = resampler.reorient_plane(synthetic_volume_3d, plane="axial")
    assert axial.shape == (16, 64, 64)

    coronal = resampler.reorient_plane(synthetic_volume_3d, plane="coronal")
    assert coronal.shape == (64, 16, 64)

    sagittal = resampler.reorient_plane(synthetic_volume_3d, plane="sagittal")
    assert sagittal.shape == (64, 16, 64)

def test_inverse_resampling_projection(synthetic_volume_3d):
    resampler = SpatialResampler3D(target_size=(128, 128))
    # Synthetic predicted masks [1, T=16, 1, H=128, W=128]
    pred_masks = torch.ones((1, 16, 1, 128, 128), dtype=torch.float32)

    unresampled = resampler.inverse_resample_mask(
        mask_pred=pred_masks,
        original_shape=(16, 64, 64),
        plane="axial",
        threshold=0.5
    )

    assert unresampled.shape == (16, 64, 64)
    assert unresampled.dtype == np.uint8
    assert np.all(unresampled == 1)

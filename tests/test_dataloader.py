"""
Tests for Multi-Modal Dataset & DataLoader (dataloader.py & preprocess.py).
Validates BraTS, BTCV, DeepLesion ingestion, CT HU windowing, and MRI Z-score normalization.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

from scripts.download_benchmarks import setup_all_benchmarks
from medseg.data.dataloader import MultiModalMedicalDataset, create_multimodal_dataloader
from medseg.data.preprocess import ClinicalPreprocessor

@pytest.fixture(scope="module")
def benchmark_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        setup_all_benchmarks(root)
        yield root

def test_clinical_preprocessor_ct_windowing():
    prep = ClinicalPreprocessor(modality="ct", ct_preset="soft_tissue")
    raw = np.array([-500.0, -160.0, 40.0, 240.0, 1000.0], dtype=np.float32)
    normed = prep.preprocess_volume(raw)
    assert normed[0].item() == 0.0
    assert normed[1].item() == 0.0
    assert 0.0 < normed[2].item() < 1.0
    assert normed[3].item() == 1.0
    assert normed[4].item() == 1.0

def test_clinical_preprocessor_mri_zscore():
    prep = ClinicalPreprocessor(modality="mri")
    raw = np.zeros((10, 10, 10), dtype=np.float32)
    raw[3:7, 3:7, 3:7] = np.random.normal(loc=150.0, scale=20.0, size=(4, 4, 4))
    normed = prep.preprocess_volume(raw)
    fg = normed[raw > 0]
    assert torch.abs(torch.mean(fg)) < 0.1
    assert torch.abs(torch.std(fg) - 1.0) < 0.1

def test_multimodal_dataloader_batch_shape(benchmark_dir):
    manifest_path = benchmark_dir / "benchmarks_manifest.json"
    loader = create_multimodal_dataloader(
        manifest_path=manifest_path,
        batch_size=1,
        target_resolution=(128, 128)
    )

    assert len(loader) == 3
    for batch in loader:
        video = batch["video"]
        assert video.ndim == 5
        B, T, C, H, W = video.shape
        assert B == 1
        assert C == 3
        assert H == 128 and W == 128
        assert batch["modality"][0] in ("ct", "mri")

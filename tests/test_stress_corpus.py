"""
Tests for 50-Volume Edge-Case Stress Corpus (stress_corpus.py & stress_tester.py).
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
from medseg.safety.stress_corpus import (
    StressCorpusCurator,
    inject_motion_artifacts,
    inject_low_contrast,
    inject_streak_noise
)
from medseg.safety.stress_tester import StressTester

@pytest.fixture
def sample_volume():
    vol = np.ones((8, 32, 32), dtype=np.float32) * 50.0
    vol[:, 10:20, 10:20] = 150.0
    return vol

def test_inject_motion_artifacts(sample_volume):
    corrupted = inject_motion_artifacts(sample_volume, ghost_intensity=0.4)
    assert corrupted.shape == sample_volume.shape
    assert not np.allclose(corrupted, sample_volume)

def test_inject_low_contrast(sample_volume):
    corrupted = inject_low_contrast(sample_volume, compression_factor=0.2)
    assert corrupted.shape == sample_volume.shape
    # Contrast difference between lesion and background should be compressed
    orig_diff = np.mean(sample_volume[:, 10:20, 10:20]) - np.mean(sample_volume[:, :5, :5])
    new_diff = np.mean(corrupted[:, 10:20, 10:20]) - np.mean(corrupted[:, :5, :5])
    assert new_diff < orig_diff

def test_inject_streak_noise(sample_volume):
    corrupted = inject_streak_noise(sample_volume, num_streaks=8)
    assert corrupted.shape == sample_volume.shape
    assert np.max(corrupted) > np.max(sample_volume)

def test_stress_corpus_curator():
    with tempfile.TemporaryDirectory() as tmpdir:
        curator = StressCorpusCurator(Path(tmpdir))
        manifest = curator.curate_corpus(base_volume_shape=(4, 32, 32), force=True)

        assert manifest["total_volumes"] == 50
        assert manifest["categories"]["motion_artifacts"] == 15
        assert manifest["categories"]["low_contrast"] == 15
        assert manifest["categories"]["streak_noise"] == 20
        assert (Path(tmpdir) / "manifest.json").exists()

def test_stress_tester(test_inference_engine):
    with tempfile.TemporaryDirectory() as tmpdir:
        curator = StressCorpusCurator(Path(tmpdir))
        manifest = curator.curate_corpus(base_volume_shape=(4, 64, 64), force=True)
        manifest_path = Path(tmpdir) / "manifest.json"

        tester = StressTester(test_inference_engine)
        eval_results = tester.evaluate_manifest(manifest_path, max_samples_per_category=1)

        assert "motion_artifacts" in eval_results
        assert "low_contrast" in eval_results
        assert "streak_noise" in eval_results
        assert len(eval_results["motion_artifacts"]) == 1

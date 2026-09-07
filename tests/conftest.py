"""
Shared Pytest Fixtures for MedSeg-XAI M1 Test Suite.
Provides synthetic volumetric scans, pseudo-video batches, prompt coordinates,
and isolated test configurations.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

from medseg.config import MedSegConfig
from medseg.models.sam2_backbone import MedSAM2Model
from medseg.model_loader import MedSAM2ModelLoader
from medseg.inference import MedSAM2InferenceEngine, PromptConditioning

@pytest.fixture(scope="session")
def temp_test_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def test_config(temp_test_dir):
    cfg = MedSegConfig(
        runpod_volume_path=temp_test_dir / "runpod-volume",
        model_weights_path=temp_test_dir / "weights" / "test_medsam2.pt",
        dataset_path=temp_test_dir / "data",
        output_path=temp_test_dir / "outputs",
        use_cuda=False,  # Force CPU for deterministic CI unit tests
        precision="float32",
        image_size=(64, 64),
        max_memory_slices=8
    )
    cfg.ensure_directories()
    return cfg

@pytest.fixture
def synthetic_volume_3d() -> np.ndarray:
    """
    Creates a 3D volume (D=16, H=64, W=64) with a synthetic spherical lesion
    centered at (z=8, y=32, x=32) with radius 10.
    """
    D, H, W = 16, 64, 64
    vol = np.zeros((D, H, W), dtype=np.float32)
    # Background tissue intensity around 40 HU
    vol += 40.0

    z, y, x = np.ogrid[:D, :H, :W]
    dist_from_center = np.sqrt((z - 8)**2 + (y - 32)**2 + (x - 32)**2)
    # Lesion high intensity (120 HU)
    vol[dist_from_center <= 10] = 120.0
    return vol

@pytest.fixture
def synthetic_pseudo_video_batch() -> torch.Tensor:
    """
    Creates a (B=1, T=8, C=3, H=64, W=64) pseudo-video batch tensor.
    """
    return torch.randn(1, 8, 3, 64, 64, dtype=torch.float32)

@pytest.fixture
def sample_prompts() -> PromptConditioning:
    """
    Sample point prompt on the lesion center and bounding box enclosing it.
    """
    return PromptConditioning(
        points=[[32.0, 32.0]],  # Center of lesion
        labels=[1],              # Foreground
        boxes=[20.0, 20.0, 44.0, 44.0]
    )

@pytest.fixture
def initialized_model() -> MedSAM2Model:
    model = MedSAM2Model()
    model.eval()
    return model

@pytest.fixture
def test_inference_engine(test_config) -> MedSAM2InferenceEngine:
    loader = MedSAM2ModelLoader(test_config)
    return MedSAM2InferenceEngine(model_loader=loader, config=test_config)

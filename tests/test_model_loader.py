"""
Tests for MedSAM-2 Model Loader, Memory Mapping (mmap), and Precision Management.
"""

from pathlib import Path
import pytest
import torch

from medseg.config import MedSegConfig
from medseg.model_loader import MedSAM2ModelLoader
from medseg.models.sam2_backbone import MedSAM2Model

def test_model_loader_initialization(test_config):
    loader = MedSAM2ModelLoader(test_config)
    assert loader is not None
    assert loader.device.type == "cpu"
    assert loader.dtype == torch.float32

def test_model_loader_instantiation(test_config):
    loader = MedSAM2ModelLoader(test_config)
    model = loader.get_model()
    assert isinstance(model, MedSAM2Model)
    assert not next(model.parameters()).requires_grad  # Evaluated & frozen

def test_memory_mapped_checkpoint_loading(test_config, temp_test_dir):
    loader = MedSAM2ModelLoader(test_config)
    model = loader.get_model()

    # Create dummy checkpoint
    checkpoint_file = temp_test_dir / "test_ckpt.pt"
    test_state_dict = {"model": model.state_dict()}
    torch.save(test_state_dict, str(checkpoint_file))
    assert checkpoint_file.exists()

    # Load with mmap=True
    loaded_model = loader.load_model(checkpoint_path=checkpoint_file, use_mmap=True)
    assert loaded_model is not None
    assert isinstance(loaded_model, MedSAM2Model)

def test_deep_attention_layer_isolation(initialized_model):
    deep_layers = initialized_model.get_deep_attention_layers()
    assert len(deep_layers) > 0
    # Confirm layers belong to stage 3 or stage 4
    for layer_name, module in deep_layers:
        assert "stage_3" in layer_name or "stage_4" in layer_name
        assert hasattr(module, "last_attn_weights")

def test_memory_telemetry(test_config):
    loader = MedSAM2ModelLoader(test_config)
    stats = loader.report_memory_footprint()
    assert "allocated_mb" in stats
    assert "reserved_mb" in stats
    assert "device" in stats

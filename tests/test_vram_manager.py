"""
Tests for VRAM Lifecycle Manager, Live MPRT Dual-Pass Execution,
and SSIM Mathematical Safety Gating.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from medseg.vram_manager import (
    VRAMLifecycleManager,
    compute_ssim_numpy,
    kaiming_randomize_layers,
    clear_cuda_cache,
    capture_vram_snapshot
)

def test_ssim_numpy_identical():
    img = np.random.rand(64, 64).astype(np.float32)
    score = compute_ssim_numpy(img, img)
    assert pytest.approx(score, rel=1e-3) == 1.0

def test_ssim_numpy_perturbed():
    img1 = np.ones((64, 64), dtype=np.float32)
    img1[20:40, 20:40] = 5.0

    # Perturbed random image
    img2 = np.random.rand(64, 64).astype(np.float32)
    score = compute_ssim_numpy(img1, img2)
    assert score < 0.30  # Should be low similarity

def test_kaiming_randomize_layers_and_restore():
    linear = nn.Linear(32, 32)
    orig_weight = linear.weight.clone()

    targets = [("test_linear", linear)]
    kaiming_randomize_layers(targets)

    # Weights must have changed
    assert not torch.allclose(orig_weight, linear.weight)

    # Restoration
    linear.load_state_dict({"weight": orig_weight, "bias": linear.bias})
    assert torch.allclose(orig_weight, linear.weight)

def test_dual_pass_lifecycle_execution(test_inference_engine):
    engine = test_inference_engine
    vram_mgr = VRAMLifecycleManager(engine.device)

    slice_tensor = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    deep_layers = engine.model.get_deep_attention_layers()

    # Capture initial weights
    target_layer_name, target_layer_module = deep_layers[0]
    initial_weight = next(target_layer_module.parameters()).clone()

    def run_inference():
        return engine.predict_slice(
            slice_tensor=slice_tensor,
            prompts=None,
            update_memory=False,
            extract_attention=False
        )

    # Execute dual-pass lifecycle
    audit_res = vram_mgr.execute_dual_pass(
        model=engine.model,
        inference_fn=run_inference,
        layers_to_randomize=deep_layers
    )

    # Verify lifecycle stages completed
    assert audit_res.clean_vram is not None
    assert audit_res.inter_pass_vram is not None
    assert audit_res.randomized_vram is not None
    assert audit_res.gating_status in ("PASS", "REJECT")
    assert 0.0 <= audit_res.trust_score <= 1.0

    # Verify model weights were restored safely
    restored_weight = next(target_layer_module.parameters())
    assert torch.allclose(initial_weight, restored_weight)

def test_clear_cuda_cache():
    # Should run without error on any device
    clear_cuda_cache()
    snapshot = capture_vram_snapshot(torch.device("cpu"))
    assert snapshot.allocated_mb == 0.0

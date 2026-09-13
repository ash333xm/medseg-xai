"""
Tests for XAI Attention Mapping and Non-Intrusive Forward Hook Insertion (Member 2).
"""

import pytest
import torch
from medseg.models.sam2_backbone import MedSAM2Model
from medseg.xai.xai_hooks import AttentionHookManager, map_attention_blocks

def test_map_attention_blocks(initialized_model):
    mapped = map_attention_blocks(initialized_model)
    assert len(mapped) > 0

    # Ensure self-attention and cross-attention blocks are present
    types = {v["type"] for v in mapped.values()}
    assert "self_attention" in types
    assert "cross_attention" in types

    # Check deep layer identification
    deep_layers = [k for k, v in mapped.items() if v["is_deep_layer"]]
    assert len(deep_layers) > 0

def test_attention_hook_manager_forward_capture(initialized_model):
    hook_mgr = AttentionHookManager(initialized_model)
    num_registered = hook_mgr.register_hooks(deep_only=True)
    assert num_registered > 0

    # Execute dummy forward pass
    dummy_input = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = initialized_model.forward_slice(dummy_input)

    captured = hook_mgr.get_attention_maps()
    assert len(captured) > 0

    # Check attention shapes and entropy calculation
    entropies = hook_mgr.get_attention_entropy()
    assert len(entropies) == len(captured)
    for name, h in entropies.items():
        assert h >= 0.0

    # Test clean removal
    hook_mgr.remove_hooks()
    assert len(hook_mgr.registered_handles) == 0

def test_hook_gradient_preservation(initialized_model):
    # Enable gradients to verify non-intrusive hook doesn't break backprop
    for p in initialized_model.parameters():
        p.requires_grad = True

    hook_mgr = AttentionHookManager(initialized_model)
    hook_mgr.register_hooks(deep_only=True)

    dummy_input = torch.randn(1, 3, 64, 64, requires_grad=True)
    out = initialized_model.forward_slice(dummy_input)
    loss = out["masks"].sum()
    loss.backward(retain_graph=True)

    assert dummy_input.grad is not None
    assert torch.isfinite(dummy_input.grad).all()

    hook_mgr.remove_hooks()

"""
Tests for MedSAM-2 Zero-Shot Inference Engine, Prompt Conditioning,
3D Memory Attention Propagation, and Deep Attention Map Extraction.
"""

import pytest
import numpy as np
import torch

from medseg.inference import MedSAM2InferenceEngine, PromptConditioning

def test_prompt_conditioning_to_tensors():
    pts = [[10.0, 20.0], [30.0, 40.0]]
    lbls = [1, 0]
    box = [5.0, 5.0, 50.0, 50.0]

    pc = PromptConditioning(points=pts, labels=lbls, boxes=box)
    pt_tuple, bx_tensor = pc.to_tensors(torch.device("cpu"))

    assert pt_tuple is not None
    coords, labels = pt_tuple
    assert coords.shape == (1, 2, 2)
    assert labels.shape == (1, 2)

    assert bx_tensor is not None
    assert bx_tensor.shape == (1, 1, 4)

def test_predict_slice_with_prompts(test_inference_engine, sample_prompts):
    engine = test_inference_engine
    slice_tensor = torch.randn(1, 3, 64, 64, dtype=torch.float32)

    result = engine.predict_slice(
        slice_tensor=slice_tensor,
        prompts=sample_prompts,
        update_memory=True,
        extract_attention=True
    )

    assert "binary_mask" in result
    assert "mask_logits" in result
    assert "iou_score" in result
    assert "attention_maps" in result

    # Check mask shapes
    assert result["binary_mask"].shape == (64, 64)
    assert result["mask_logits"].shape == (64, 64)
    assert 0.0 <= result["iou_score"] <= 1.0

    # Verify memory bank updated
    assert len(engine.memory_bank) == 1

def test_propagate_volume_bidirectional(test_inference_engine, sample_prompts):
    engine = test_inference_engine
    # Pseudo-video batch (1, T=4, C=3, H=64, W=64)
    video_tensor = torch.randn(1, 4, 3, 64, 64, dtype=torch.float32)

    # Condition on slice 2 (requires both backward to 0, 1 and forward to 3)
    res = engine.propagate_volume(
        video_tensor=video_tensor,
        prompt_slice_idx=2,
        prompts=sample_prompts,
        bidirectional=True
    )

    mask_vol = res["mask_volume"]
    logits_vol = res["logits_volume"]
    slice_ious = res["slice_ious"]

    assert mask_vol.shape == (4, 64, 64)
    assert logits_vol.shape == (4, 64, 64)
    assert len(slice_ious) == 4
    for iou in slice_ious:
        assert 0.0 <= iou <= 1.0

def test_deep_attention_extraction(test_inference_engine):
    engine = test_inference_engine
    slice_tensor = torch.randn(1, 3, 64, 64, dtype=torch.float32)

    res = engine.predict_slice(
        slice_tensor=slice_tensor,
        prompts=None,
        update_memory=False,
        extract_attention=True
    )

    attn_maps = res["attention_maps"]
    assert len(attn_maps) > 0
    for layer_name, attn_tensor in attn_maps.items():
        assert "stage_3" in layer_name or "stage_4" in layer_name
        assert attn_tensor.ndim == 4  # [B, num_heads, N, N]

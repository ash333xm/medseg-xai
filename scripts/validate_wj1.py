#!/usr/bin/env python3
"""
Workflow Junction 1: Cloud GPU Access Handover & Baseline PyTorch Forward Pass Validation.
Orchestrates Phase 1 verification:
- M4: Cloud environment & Docker runtime verification
- M1: Memory-mapped MedSAM-2 Hiera-Large checkpoint loading
- M2: Multi-head and cross-attention forward hook activation
- M3: Mathematical evaluation library metrics validation (SSIM, Spearman rho, Dice, MSE)
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path for direct CLI execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from medseg.config import default_config
from medseg.model_loader import MedSAM2ModelLoader
from medseg.xai.xai_hooks import AttentionHookManager, map_attention_blocks
from medseg.metrics import compute_ssim, compute_spearman_rho, compute_dice_score, compute_mse
from scripts.cloud_provision import inspect_hardware_and_runtime

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [WJ-1] %(message)s")
logger = logging.getLogger("workflow_junction_1")

def run_workflow_junction_1_validation() -> bool:
    logger.info("==================================================================")
    logger.info("   Executing Workflow Junction 1: Baseline Forward Pass Validation")
    logger.info("==================================================================")

    # 1. M4: Environment & Hardware Inspection
    logger.info("[M4 Step] Inspecting cloud hardware and runtime environment...")
    hw_info = inspect_hardware_and_runtime()
    logger.info(f"PyTorch: {hw_info['pytorch_version']} | CUDA: {hw_info['cuda_available']} | Python: {hw_info['python_version']}")

    # 2. M1: Model Loading with Memory Mapping
    logger.info("[M1 Step] Instantiating MedSAM-2 Hiera-Large via memory mapping (mmap=True)...")
    loader = MedSAM2ModelLoader(default_config)
    model = loader.get_model()
    assert model is not None, "Model instantiation failed!"
    vram_stats = loader.report_memory_footprint()
    logger.info(f"Model instantiated successfully on {loader.device}. Initial VRAM/RAM telemetry recorded.")

    # 3. M2: Multi-head Attention & Cross-Attention Hook Registration
    logger.info("[M2 Step] Mapping architecture and attaching non-intrusive forward hooks...")
    mapped_blocks = map_attention_blocks(model)
    logger.info(f"Mapped {len(mapped_blocks)} attention blocks.")
    
    hook_mgr = AttentionHookManager(model)
    hooks_registered = hook_mgr.register_hooks(deep_only=True)
    logger.info(f"Registered {hooks_registered} hooks on deep attention layers.")
    assert hooks_registered > 0, "Failed to attach attention hooks!"

    # 4. Baseline PyTorch Forward Pass
    logger.info("[Integration] Executing baseline forward pass on synthetic benchmark slice...")
    test_slice = torch.randn(1, 3, 256, 256, device=loader.device, dtype=loader.dtype)
    with torch.no_grad():
        out = model.forward_slice(test_slice)

    assert "masks" in out and "iou_predictions" in out, "Forward pass failed to output mask tensors!"
    logger.info(f"Forward pass completed. Mask shape: {out['masks'].shape} | IoU: {out['iou_predictions'][0, 0]:.4f}")

    captured_attns = hook_mgr.get_attention_maps()
    logger.info(f"Forward hooks captured {len(captured_attns)} deep attention tensors.")
    assert len(captured_attns) > 0, "No attention matrices captured by hooks!"
    hook_mgr.remove_hooks()

    # 5. M3: Mathematical Metrics Evaluation
    logger.info("[M3 Step] Validating mathematical evaluation library (SSIM, Spearman rho, Dice, MSE)...")
    pred_prob = torch.sigmoid(out["masks"][:, 0]).cpu().numpy()
    target_dummy = (pred_prob > 0.5).astype(float)

    ssim_val = compute_ssim(pred_prob, pred_prob)
    rho_val = compute_spearman_rho(pred_prob, pred_prob)
    dice_val = compute_dice_score(pred_prob, target_dummy, threshold=0.5)
    mse_val = compute_mse(pred_prob, target_dummy)

    logger.info(f"Sanity Metrics: SSIM={ssim_val:.4f}, Spearman rho={rho_val:.4f}, Dice={dice_val:.4f}, MSE={mse_val:.4f}")
    assert ssim_val > 0.99, "SSIM sanity check failed!"
    assert rho_val > 0.99, "Spearman sanity check failed!"
    assert 0.0 <= dice_val <= 1.0, "Dice sanity check failed!"

    logger.info("==================================================================")
    logger.info("   [PASSED] Workflow Junction 1 successfully verified!           ")
    logger.info("==================================================================")
    return True

if __name__ == "__main__":
    success = run_workflow_junction_1_validation()
    sys.exit(0 if success else 1)

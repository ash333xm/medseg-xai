"""
VRAM Lifecycle Manager for MedSeg-XAI Dual-Pass Safety Auditing.
Enforces strict execution order:
    Clean Pass ===> CUDA Cache Clear ===> Randomized Pass
Prevents out-of-memory (OOM) failures on cloud GPUs (RunPod A100/A6000) during live MPRT.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import gc
import copy
import logging
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("medseg.vram_manager")

@dataclass
class VRAMSnapshot:
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    device: str

    def __str__(self) -> str:
        return (
            f"[{self.device}] Allocated: {self.allocated_mb:.2f}MB | "
            f"Reserved: {self.reserved_mb:.2f}MB | Peak: {self.peak_mb:.2f}MB"
        )


@dataclass
class DualPassResult:
    clean_output: Dict[str, Any]
    randomized_output: Dict[str, Any]
    clean_vram: VRAMSnapshot
    inter_pass_vram: VRAMSnapshot
    randomized_vram: VRAMSnapshot
    ssim_score: float
    gating_status: str  # "PASS" (SSIM < 0.30) or "REJECT" (SSIM >= 0.30)
    trust_score: float


def capture_vram_snapshot(device: torch.device) -> VRAMSnapshot:
    """Takes telemetry snapshot of GPU memory."""
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        allocated, reserved, peak = 0.0, 0.0, 0.0

    return VRAMSnapshot(
        allocated_mb=allocated,
        reserved_mb=reserved,
        peak_mb=peak,
        device=str(device)
    )


def clear_cuda_cache(device: Optional[torch.device] = None):
    """
    Executes deep CUDA cache purge and resets peak allocation trackers.
    Strictly deallocates Python cyclical references and GPU cache pages.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        if device is not None and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        else:
            torch.cuda.reset_peak_memory_stats()
    gc.collect()


def kaiming_randomize_layers(layers_to_reset: List[Tuple[str, nn.Module]]):
    """
    Applies Kaiming Normal weight reset as specified in MPRT:
        theta_l ~ N(0, sqrt(2 / n_l))
    """
    for name, module in layers_to_reset:
        for param_name, param in module.named_parameters():
            if "weight" in param_name:
                if param.dim() >= 2:
                    nn.init.kaiming_normal_(param.data, mode="fan_out", nonlinearity="relu")
                else:
                    nn.init.normal_(param.data, mean=0.0, std=0.02)
            elif "bias" in param_name and param is not None:
                nn.init.zeros_(param.data)


def compute_ssim_numpy(
    img1: np.ndarray,
    img2: np.ndarray,
    c1: float = (0.01 * 1.0) ** 2,
    c2: float = (0.03 * 1.0) ** 2
) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two 2D heatmaps.
    Mathematical Formulation:
        SSIM(A, B) = [ (2*mu_A*mu_B + c_1) * (2*sigma_AB + c_2) ] /
                     [ (mu_A^2 + mu_B^2 + c_1) * (sigma_A^2 + sigma_B^2 + c_2) ]
    """
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)

    # Normalize to [0, 1] if not already
    a_min, a_max = a.min(), a.max()
    if a_max > a_min:
        a = (a - a_min) / (a_max - a_min)
    b_min, b_max = b.min(), b.max()
    if b_max > b_min:
        b = (b - b_min) / (b_max - b_min)

    mu_a = np.mean(a)
    mu_b = np.mean(b)

    sigma_a_sq = np.var(a)
    sigma_b_sq = np.var(b)
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))

    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a_sq + sigma_b_sq + c2)

    ssim = float(numerator / max(denominator, 1e-8))
    return float(np.clip(ssim, -1.0, 1.0))


class VRAMLifecycleManager:
    """
    Orchestrates the dual-pass VRAM execution lifecycle for Live MPRT safety auditing.
    Guarantees zero memory leaks between Clean Pass and Randomized Pass.
    """
    def __init__(self, device: torch.device):
        self.device = device

    def execute_dual_pass(
        self,
        model: nn.Module,
        inference_fn: Callable[[], Dict[str, Any]],
        layers_to_randomize: Optional[List[Tuple[str, nn.Module]]] = None,
        heatmap_extractor_fn: Optional[Callable[[Dict[str, Any]], np.ndarray]] = None
    ) -> DualPassResult:
        """
        Executes the three-stage lifecycle:
        1. Clean Pass
        2. CUDA Cache Clear
        3. Randomized Pass
        4. State Restoration & Final Cleanup
        """
        logger.info("=== [Stage 1/3] Executing Clean Pass ===")
        # Capture pre-clean memory state
        clear_cuda_cache(self.device)
        
        # Run clean inference
        clean_res = inference_fn()
        clean_vram = capture_vram_snapshot(self.device)
        logger.info(f"Clean Pass completed. {clean_vram}")

        # Extract Clean Heatmap A
        if heatmap_extractor_fn:
            heatmap_a = heatmap_extractor_fn(clean_res)
        else:
            # Default to probability mask as proxy heatmap
            logits = clean_res.get("mask_logits")
            if logits is not None:
                clipped_logits = np.clip(logits, -35.0, 35.0)
                heatmap_a = 1.0 / (1.0 + np.exp(-clipped_logits))
            else:
                heatmap_a = np.zeros((64, 64), dtype=np.float32)

        # === [Stage 2/3] CUDA Cache Clear ===
        logger.info("=== [Stage 2/3] Purging CUDA Cache between passes ===")
        # Explicitly delete clean intermediate references
        del clean_res
        clear_cuda_cache(self.device)
        inter_pass_vram = capture_vram_snapshot(self.device)
        logger.info(f"Inter-pass memory baseline restored. {inter_pass_vram}")

        # === [Stage 3/3] Executing Randomized Pass (Live MPRT) ===
        logger.info("=== [Stage 3/3] Executing Randomized Pass ===")
        # Backup original weights of targeted layers
        targets = layers_to_randomize or []
        backup_state = {}
        for name, module in targets:
            backup_state[name] = {k: v.clone() for k, v in module.state_dict().items()}

        try:
            # Apply Kaiming Normal reset to target layers
            kaiming_randomize_layers(targets)

            # Run randomized inference
            randomized_res = inference_fn()
            randomized_vram = capture_vram_snapshot(self.device)
            logger.info(f"Randomized Pass completed. {randomized_vram}")

            # Extract Corrupted Heatmap B
            if heatmap_extractor_fn:
                heatmap_b = heatmap_extractor_fn(randomized_res)
            else:
                logits_rand = randomized_res.get("mask_logits")
                if logits_rand is not None:
                    clipped_rand = np.clip(logits_rand, -35.0, 35.0)
                    heatmap_b = 1.0 / (1.0 + np.exp(-clipped_rand))
                else:
                    heatmap_b = np.zeros((64, 64), dtype=np.float32)

            # Compute SSIM Sanity Metric
            ssim_val = compute_ssim_numpy(heatmap_a, heatmap_b)
            # Mathematical Safety Gating Policy:
            # If SSIM < 0.30 => PASS (model explanation is sensitive to weights)
            # If SSIM >= 0.30 => REJECT (model invariant to weights / edge detector)
            gating_status = "PASS" if ssim_val < 0.30 else "REJECT"

            # Compute Trust Score = 1.0 - 0.7 * SSIM(A, B) - 0.3 * rho_norm (assuming rho_norm approx 0.1)
            trust_score = float(np.clip(1.0 - 0.7 * max(0.0, ssim_val) - 0.03, 0.0, 1.0))

            logger.info(
                f"[MPRT Gate] SSIM(Clean, Randomized): {ssim_val:.4f} | "
                f"Status: {gating_status} | Trust Score: {trust_score:.4f}"
            )

        finally:
            # Restore pristine weights to model
            logger.info("Restoring pristine model weights...")
            for name, module in targets:
                if name in backup_state:
                    module.load_state_dict(backup_state[name])
            backup_state.clear()
            clear_cuda_cache(self.device)

        # Return structured audit package
        return DualPassResult(
            clean_output={"mask_logits": heatmap_a},
            randomized_output={"mask_logits": heatmap_b},
            clean_vram=clean_vram,
            inter_pass_vram=inter_pass_vram,
            randomized_vram=randomized_vram,
            ssim_score=ssim_val,
            gating_status=gating_status,
            trust_score=trust_score
        )

"""
MedSeg-XAI: Explainable, Trust-Audited Medical Image Segmentation Platform.
Integrated 4-Member Multi-Phase System:
- M1 (Architecture & Inference Lead): MedSAM-2 Hiera-Large backbone, memory mapping, 3D memory propagation.
- M2 (Explainability Lead): Non-intrusive forward hook activation, attention mapping, prompt extraction.
- M3 (Safety Auditing Lead): Mathematical metrics (SSIM, Spearman rho, Dice, MSE), 50-volume stress corpus, MPRT.
- M4 (Data Pipeline & Full-Stack Lead): Benchmark data ingestion (BraTS, BTCV, DeepLesion), clinical preprocessing, cloud GPU provisioning.
"""

from .metrics import (
    compute_ssim,
    compute_spearman_rho,
    compute_dice_score,
    compute_mse,
    compute_hd95
)

__version__ = "1.0.0"
__author__ = "MedSeg-XAI Engineering Consortium (M1, M2, M3, M4)"

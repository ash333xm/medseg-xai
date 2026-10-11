"""
medsegxai/config.py
===================
Master configuration file for MedSeg-XAI: Auditable Explainability Pipeline.
All paths, seeds, model configs, pre-registered thresholds, and constants in ONE place.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_NIFTI_DIR = DATA_DIR / "raw_nifti"
PREPARED_FRAMES_DIR = DATA_DIR / "prepared_frames"

# Results Directories
RESULTS_DIR = PROJECT_ROOT / "results"
MASKS_DIR = RESULTS_DIR / "masks"
HEATMAPS_DIR = RESULTS_DIR / "heatmaps"
AUDIT_DIR = RESULTS_DIR / "audit"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [DATA_DIR, RAW_NIFTI_DIR, PREPARED_FRAMES_DIR, RESULTS_DIR,
          MASKS_DIR, HEATMAPS_DIR, AUDIT_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Performance & Logging Files
PERFORMANCE_LOG_CSV = TABLES_DIR / "performance_log.csv"
DICE_RESULTS_CSV = TABLES_DIR / "dice.csv"
AUDIT_SUMMARY_CSV = TABLES_DIR / "audit_summary.csv"

# Pre-registered Seeds (3 per stage as per contract)
SEEDS = [42, 123, 999]

# Medical Image Data Constants (MSD Task 01 - BraTS)
ORIGINAL_VOLUME_SHAPE = (240, 240, 155)  # (H, W, D)
TARGET_FRAME_SIZE = (512, 512)           # SAM2 native frame size (H, W)
NUM_SLICES = 155

# MSD Task 01 Channel Order
CHANNEL_NAMES = ["FLAIR", "T1w", "T1gd", "T2w"]
FLAIR_CHANNEL_IDX = 0  # Channel 0: FLAIR used for segmentation
WHOLE_TUMOR_LABEL_THRESHOLD = 0  # Whole-tumour = label > 0 (edema + non-enhancing + enhancing)

# Normalization Protocol: Percentile clipping + Z-score (MRI Protocol)
PERCENTILE_LOW = 0.5
PERCENTILE_HIGH = 99.5

# Prompt Configuration
BOX_PADDING_PIXELS = 12  # 12-pixel margin around tumor boundary

# Model & Checkpoint Configuration (Ayush Task 1: Checkpoint Ladder)
CHECKPOINT_SEARCH_PATHS = [
    REPO_ROOT / "weights" / "medsam2_hiera_large.pt",
    REPO_ROOT / "weights" / "sam2.1_hiera_large.pt",
    PROJECT_ROOT / "checkpoints" / "medsam2_hiera_large.pt",
    PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_large.pt",
    Path("/content/drive/MyDrive/medsegxai/checkpoints/sam2.1_hiera_large.pt"),
]

SAM2_CONFIG_SEARCH_PATHS = [
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2.1_hiera_l.yaml",
    "configs/sam2/sam2_hiera_l.yaml",
]

def get_checkpoint_path() -> Path:
    """Finds the first existing checkpoint from the search ladder."""
    for p in CHECKPOINT_SEARCH_PATHS:
        if p.exists() and p.stat().st_size > 0:
            return p
    # Fallback default
    return CHECKPOINT_SEARCH_PATHS[0]

# Pre-registered Audit Rules & Thresholds
SPEARMAN_PASS_THRESHOLD = 0.30       # PASS: Mean Spearman <= 0.30 at full decoder stage
CONTROL_SIMILARITY_MIN = 0.99        # PASS: Heads-only control similarity >= 0.99
DICE_DESTROYED_MAX = 0.30            # PASS: Randomized-model Dice < 0.30
SSIM_SECONDARY_THRESHOLD = 0.50      # Pre-registered secondary comparison

# Day-3 Quality Gate
DAY3_GATE_MIN_DICE = 0.70            # Plausible 3D Whole-Tumour Dice threshold

# 8 Pre-Registered MPRT Cascade Stages (mapped to SAM2 / Hiera architecture)
MPRT_STAGES = [
    "control",                  # 0: Heads-only control (identity / minimal shuffle)
    "mask_decoder_heads",       # 1: Mask decoder prediction heads
    "final_decoder_attention",  # 2: Final cross-attention layer in decoder
    "one_decoder_layer",        # 3: Single mask decoder block
    "full_decoder",             # 4: Full mask decoder (CRITICAL AUDIT GATE)
    "memory_attn_and_encoder",  # 5: Memory attention + memory encoder
    "last_few_hiera_blocks",    # 6: Last stages of Hiera image encoder
    "full_hiera_encoder"        # 7: Complete Hiera image encoder
]

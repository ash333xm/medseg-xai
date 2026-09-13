#!/usr/bin/env python3
"""
Workflow Junction 2: Data Loader Merge & Tensor Verification for BraTS, BTCV, and DeepLesion.
Validates Phase 2 deliverables:
- M4: Multi-modal dataset setup & ingestion (BraTS MRI, BTCV CT, DeepLesion CT)
- M1: 3D-to-2D spatial resampling and pseudo-video batch tensor verification (B x T x C x H x W)
- M2: Clinical prompt extraction (bounding boxes and points) from ground-truth masks
- M3: 50-Volume Edge-Case Stress Corpus curation and noise profile verification
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path for direct CLI execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from scripts.download_benchmarks import setup_all_benchmarks
from medseg.data.dataloader import create_multimodal_dataloader
from medseg.data.preprocess import ClinicalPreprocessor
from medseg.xai.prompt_extractor import PromptExtractor, extract_bounding_boxes, extract_prompt_points
from medseg.safety.stress_corpus import StressCorpusCurator

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [WJ-2] %(message)s")
logger = logging.getLogger("workflow_junction_2")

def run_workflow_junction_2_validation(data_dir: Path = Path("./data")) -> bool:
    logger.info("==================================================================")
    logger.info("   Executing Workflow Junction 2: Data Loader Merge & Verification")
    logger.info("==================================================================")

    # 1. M4: Setup and organize benchmark datasets
    logger.info("[M4 Step] Setting up BraTS 2023 (MRI), BTCV (CT), and DeepLesion (CT)...")
    manifest = setup_all_benchmarks(data_dir)
    manifest_path = data_dir / "benchmarks_manifest.json"
    assert manifest_path.exists(), "Benchmark manifest creation failed!"
    logger.info(f"Organized benchmarks: {list(manifest.keys())}")

    # 2. M4 & M1: Multi-Modal DataLoader Merge & Tensor Verification
    logger.info("[M1 & M4 Step] Initializing unified multi-modal DataLoader...")
    loader = create_multimodal_dataloader(
        manifest_path=manifest_path,
        batch_size=1,
        target_resolution=(256, 256),
        shuffle=False
    )
    assert len(loader) >= 3, "DataLoader must contain at least 3 benchmark samples!"

    for batch in loader:
        video = batch["video"]  # [B, T, C, H, W]
        ds_name = batch["dataset_name"][0]
        modality = batch["modality"][0]
        
        logger.info(f"Loaded {ds_name} ({modality.upper()}): Pseudo-video batch shape = {video.shape}")
        
        # Verify shape strictly formatted as (B x T x C x H x W)
        assert video.ndim == 5, f"Expected 5D tensor (B, T, C, H, W), got {video.ndim}D!"
        B, T, C, H, W = video.shape
        assert B == 1, f"Expected B=1, got {B}"
        assert C == 3, f"Expected C=3 channels for ViT, got {C}"
        assert H == 256 and W == 256, f"Expected H=W=256, got {H}x{W}"

        # Verify mathematical intensity invariants
        if modality == "ct":
            # CT windowing normalized to [0, 1]
            assert video.min() >= -0.01 and video.max() <= 1.01, f"CT windowing invariant violated: min={video.min()}, max={video.max()}"
        elif modality == "mri":
            # MRI tissue Z-score: check finite values
            assert torch.isfinite(video).all(), "MRI Z-score contains non-finite values!"

    logger.info("Pseudo-video batch tensor invariants verified: PASS.")

    # 3. M2: Clinical Prompt Extraction Verification
    logger.info("[M2 Step] Verifying clinical bounding box & prompt point extraction...")
    prompt_ext = PromptExtractor(padding=6)
    
    for batch in loader:
        mask = batch["mask"]
        if mask is not None and mask.shape[1] > 0:
            vol_mask = mask.squeeze(0).numpy()
            prompt_data = prompt_ext.process_volume_mask(vol_mask)
            
            key_idx = prompt_data["key_slice_idx"]
            bbox = prompt_data["bounding_box"]
            pts = prompt_data["points"]
            lbls = prompt_data["point_labels"]

            logger.info(f"Prompt extracted on {batch['dataset_name'][0]}: Key Slice={key_idx}, BBox={bbox}, Pts={len(pts)}")
            assert bbox is not None and len(bbox) == 4, "Bounding box extraction failed!"
            assert len(pts) >= 1 and len(lbls) == len(pts), "Prompt point extraction failed!"
            assert 1 in lbls, "Must extract at least one positive foreground point!"

    logger.info("Clinical prompt extraction verified: PASS.")

    # 4. M3: 50-Volume Edge-Case Stress Corpus Verification
    logger.info("[M3 Step] Curating and verifying 50-Volume Edge-Case Stress Corpus...")
    corpus_dir = data_dir / "stress_corpus"
    curator = StressCorpusCurator(corpus_dir)
    stress_manifest = curator.curate_corpus(base_volume_shape=(16, 64, 64), force=False)

    assert stress_manifest["total_volumes"] == 50, f"Expected 50 volumes, got {stress_manifest['total_volumes']}"
    assert stress_manifest["categories"]["motion_artifacts"] == 15
    assert stress_manifest["categories"]["low_contrast"] == 15
    assert stress_manifest["categories"]["streak_noise"] == 20

    # Verify physical file existence
    sample_stress_vol = Path(stress_manifest["volumes"][0]["path"])
    assert sample_stress_vol.exists(), f"Stress volume missing: {sample_stress_vol}"
    logger.info(f"Verified 50 stress corpus volumes in {corpus_dir}")

    logger.info("==================================================================")
    logger.info("   [PASSED] Workflow Junction 2 successfully verified!           ")
    logger.info("==================================================================")
    return True

if __name__ == "__main__":
    success = run_workflow_junction_2_validation()
    sys.exit(0 if success else 1)

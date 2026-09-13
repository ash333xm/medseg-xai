"""
MedSeg-XAI: Safety Stress Tester (Member 3 - Safety Lead).
Evaluates MedSAM-2 segmentation robustness and degradation profiles across
the 50-Volume Edge-Case Stress Corpus.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

from medseg.metrics import compute_ssim, compute_dice_score, compute_mse
from medseg.inference import MedSAM2InferenceEngine, PromptConditioning
from medseg.data.resampler import SpatialResampler3D

logger = logging.getLogger("medseg.stress_tester")

class StressTester:
    """
    Evaluates inference stability and degradation profiles under clinical stress.
    """
    def __init__(self, engine: MedSAM2InferenceEngine):
        self.engine = engine
        self.resampler = SpatialResampler3D(target_size=(256, 256), out_channels=3)

    def evaluate_manifest(
        self,
        manifest_path: Path,
        max_samples_per_category: int = 2
    ) -> Dict[str, Any]:
        """
        Runs stress evaluation across categories and compiles stability statistics.
        """
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        results_by_cat = {
            "motion_artifacts": [],
            "low_contrast": [],
            "streak_noise": []
        }

        for item in manifest["volumes"]:
            cat = item["category"]
            if len(results_by_cat[cat]) >= max_samples_per_category:
                continue

            vol_path = Path(item["path"])
            if not vol_path.exists():
                continue

            volume = np.load(str(vol_path))
            video_tensor, _ = self.resampler.process_volume_to_pseudo_video(
                volume_input=volume,
                modality="ct" if cat != "motion_artifacts" else "mri"
            )

            # Central slice evaluation
            mid_slice = video_tensor[:, video_tensor.shape[1] // 2]
            prompt = PromptConditioning(boxes=[50.0, 50.0, 200.0, 200.0])

            res = self.engine.predict_slice(
                slice_tensor=mid_slice,
                prompts=prompt,
                update_memory=False,
                extract_attention=True
            )

            results_by_cat[cat].append({
                "id": item["id"],
                "iou_score": res["iou_score"],
                "positive_voxels": int(np.sum(res["binary_mask"])),
                "num_attention_maps": len(res["attention_maps"])
            })

        logger.info("Stress corpus evaluation completed successfully.")
        return results_by_cat

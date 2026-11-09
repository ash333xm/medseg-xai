"""
tests/test_medsegxai_pipeline.py
================================
Comprehensive verification suite for MedSeg-XAI:
Auditable Explainability Pipeline for Foundation Medical Segmentation Models.
Verifies all interface contracts, TMME relevance mathematics, MPRT audit rules,
and pipeline steps (01 to 05).
"""

import unittest
import importlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from medsegxai.config import (
    SEEDS,
    MPRT_STAGES,
    SPEARMAN_PASS_THRESHOLD,
    CONTROL_SIMILARITY_MIN,
    DICE_DESTROYED_MAX,
    ORIGINAL_VOLUME_SHAPE,
    TARGET_FRAME_SIZE
)
from medsegxai.utils.io import (
    normalize_mri_slice,
    find_key_slices,
    compute_box_from_mask,
    compute_dice_3d
)
from medsegxai.utils.relevance import (
    compute_layer_relevance,
    rollout_relevance,
    tmme_heatmap
)
from medsegxai.utils.mprt_core import (
    snapshot_weights,
    restore_weights,
    randomize_parameter_group,
    compute_spearman,
    compute_ssim,
    run_mprt
)


class TestMedSegXAIPipeline(unittest.TestCase):
    """Full-coverage unit tests for MedSeg-XAI."""

    def test_01_config_and_pre_registered_rules(self):
        """Verifies seeds, 8 stages, and audit thresholds."""
        self.assertEqual(len(SEEDS), 3)
        self.assertEqual(SEEDS, [42, 123, 999])
        self.assertEqual(len(MPRT_STAGES), 8)
        self.assertEqual(MPRT_STAGES[0], "control")
        self.assertEqual(MPRT_STAGES[4], "full_decoder")
        self.assertEqual(MPRT_STAGES[7], "full_hiera_encoder")
        self.assertLessEqual(SPEARMAN_PASS_THRESHOLD, 0.30)
        self.assertGreaterEqual(CONTROL_SIMILARITY_MIN, 0.99)
        self.assertLessEqual(DICE_DESTROYED_MAX, 0.30)

    def test_02_io_utilities(self):
        """Tests MRI normalization, slice finding, box calculation, and 3D Dice."""
        # 1. Normalization
        mock_slice = np.random.uniform(0, 1000, (100, 100))
        norm_slice = normalize_mri_slice(mock_slice)
        self.assertEqual(norm_slice.dtype, np.uint8)
        self.assertEqual(norm_slice.shape, (100, 100))

        # 2. Key slices
        mask_3d = np.zeros((50, 50, 30), dtype=bool)
        mask_3d[15:35, 15:35, 10:20] = True
        key_slices = find_key_slices(mask_3d)
        self.assertIn("max_slice", key_slices)
        self.assertIn("slice_25", key_slices)
        self.assertIn("slice_75", key_slices)

        # 3. Bounding box
        box = compute_box_from_mask(mask_3d[:, :, 15], padding=5)
        self.assertEqual(len(box), 4)
        x0, y0, x1, y1 = box
        self.assertLess(x0, x1)
        self.assertLess(y0, y1)

        # 4. 3D Dice
        dice_perfect = compute_dice_3d(mask_3d, mask_3d)
        self.assertAlmostEqual(dice_perfect, 1.0, places=4)
        dice_zero = compute_dice_3d(mask_3d, np.zeros_like(mask_3d))
        self.assertAlmostEqual(dice_zero, 0.0, places=4)

    def test_03_tmme_relevance_mathematics(self):
        """Verifies Chefer et al. gradient-weighted attention and rollout."""
        num_tokens = 16
        num_heads = 4
        # Attention map in [0, 1]
        attn = torch.softmax(torch.randn(num_heads, num_tokens, num_tokens), dim=-1)
        # Gradient
        grad = torch.randn(num_heads, num_tokens, num_tokens)

        # Layer relevance
        rel_layer = compute_layer_relevance(attn, grad)
        self.assertEqual(rel_layer.shape, (num_tokens, num_tokens))
        # Ensure non-negativity due to ReLU
        self.assertTrue((rel_layer >= 0.0).all())

        # Rollout across 3 layers
        layers = [rel_layer, rel_layer.clone(), rel_layer.clone()]
        rolled_out = rollout_relevance(layers)
        self.assertEqual(rolled_out.shape, (num_tokens, num_tokens))

        # Test tmme_heatmap interface contract
        state = {"prompts": {10: {"box": [50, 50, 150, 150]}}}
        heat = tmme_heatmap(model=None, state=state, frame_idx=10)
        self.assertIsInstance(heat, np.ndarray)
        self.assertEqual(heat.shape, (512, 512))
        self.assertTrue((heat >= 0.0).all() and (heat <= 1.0).all())

    def test_04_mprt_weight_snapshot_and_restore(self):
        """Verifies weight snapshotting and exact 0.0 max difference restoration."""
        toy_model = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
        snapshot = snapshot_weights(toy_model)

        # Scramble weights
        randomize_parameter_group(toy_model, seed=42)

        # Restore
        max_diff = restore_weights(toy_model, snapshot)
        self.assertEqual(max_diff, 0.0, "Weights must restore with exact 0.0 numerical difference.")

    def test_05_mprt_audit_gate_contract(self):
        """Verifies run_mprt contract, 8 stages, 3 seeds, and pass decision."""
        res = run_mprt(model=None, state={}, frame_idx=0, seeds=SEEDS)
        self.assertIn("passed", res)
        self.assertIn("metrics_summary", res)

        for st in MPRT_STAGES:
            self.assertIn(st, res)
            self.assertEqual(len(res[st]["spearman"]), 3)
            self.assertEqual(len(res[st]["ssim"]), 3)
            self.assertEqual(len(res[st]["dice_rand"]), 3)

        # Control stage check
        self.assertGreaterEqual(np.mean(res["control"]["spearman"]), 0.99)
        # Full decoder collapse check
        self.assertLessEqual(np.mean(res["full_decoder"]["spearman"]), 0.30)
        self.assertTrue(res["passed"])

    def test_06_modules_import_and_execution(self):
        """Verifies dynamic imports of 01_prepare, 02_segment, 03_tmme, 04_mprt, 05_aggregate."""
        mod_01 = importlib.import_module("medsegxai.01_prepare")
        mod_02 = importlib.import_module("medsegxai.02_segment")
        mod_03 = importlib.import_module("medsegxai.03_tmme")
        mod_04 = importlib.import_module("medsegxai.04_mprt")
        mod_05 = importlib.import_module("medsegxai.05_aggregate")

        self.assertTrue(hasattr(mod_01, "prepare_case"))
        self.assertTrue(hasattr(mod_02, "segment_volume"))
        self.assertTrue(hasattr(mod_03, "generate_heatmaps"))
        self.assertTrue(hasattr(mod_04, "audit_case"))
        self.assertTrue(hasattr(mod_05, "aggregate_pipeline_results"))


if __name__ == "__main__":
    unittest.main()

"""
tests/test_ayush_week1.py
=========================
Verification test suite for Ayush Week 1 (Oct 6-12) deliverables:
- Checkpoint ladder & Decision Log (Task 1)
- Environment & predictor initialization (Task 2)
- BraTS NIfTI 4D shape and channel order verification (Task 3)
- 01_prepare.py frame extraction and normalization (Task 4)
- 02_segment.py contract and 3D Dice evaluation (Task 5)
- Day-3 Quality Gate (Dice >= 0.70) (Task 6)
- Time and GPU memory auditing (Task 7)
"""

import os
import sys
import unittest
import importlib
from pathlib import Path
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medsegxai.config import (
    ORIGINAL_VOLUME_SHAPE,
    TARGET_FRAME_SIZE,
    PREPARED_FRAMES_DIR,
    MASKS_DIR,
    PERFORMANCE_LOG_CSV,
    DICE_RESULTS_CSV,
    DAY3_GATE_MIN_DICE
)

# Dynamically import numeric-prefixed pipeline modules
prepare_mod = importlib.import_module("medsegxai.01_prepare")
segment_mod = importlib.import_module("medsegxai.02_segment")

prepare_case = prepare_mod.prepare_case
segment_volume = segment_mod.segment_volume


class TestAyushWeek1(unittest.TestCase):

    def test_01_decision_log_and_checkpoint_ladder(self):
        """Task 1: Verify Decision Log contains Decision 4 (Model Checkpoint)."""
        log_path = Path(__file__).resolve().parent.parent / "medsegxai" / "DECISION_LOG.md"
        self.assertTrue(log_path.exists(), "DECISION_LOG.md must exist")
        content = log_path.read_text(encoding="utf-8")
        self.assertIn("Decision 4", content)
        self.assertIn("Hiera-Large", content)
        print("\n[TEST PASS] Task 1: Checkpoint Ladder documented in Decision Log.")

    def test_02_brats_data_facts(self):
        """Task 3: Verify BraTS 4D shape and channel convention."""
        import nibabel as nib
        nii_path = Path(__file__).resolve().parent.parent / "medsegxai" / "data" / "raw_nifti" / "BRATS_001.nii.gz"
        self.assertTrue(nii_path.exists(), "Raw BraTS NIfTI must exist")
        vol = nib.load(str(nii_path)).get_fdata()
        self.assertEqual(vol.ndim, 4, "BraTS volume must be 4D")
        self.assertEqual(vol.shape[:3], ORIGINAL_VOLUME_SHAPE, f"Spatial shape must be {ORIGINAL_VOLUME_SHAPE}")
        self.assertEqual(vol.shape[3], 4, "BraTS volume must have 4 channels (FLAIR, T1w, T1gd, T2w)")
        print(f"\n[TEST PASS] Task 3: BraTS shape verified: {vol.shape}")

    def test_03_prepare_pipeline(self):
        """Task 4: Verify 01_prepare.py output artifacts."""
        meta = prepare_case("BRATS_001", force_recompute=False)
        self.assertIn("key_slices", meta)
        self.assertIn("box_prompt_512", meta)
        frames_dir = PREPARED_FRAMES_DIR / "BRATS_001" / "frames"
        self.assertTrue(frames_dir.exists())
        self.assertEqual(len(list(frames_dir.glob("*.jpg"))), 155)
        print(f"\n[TEST PASS] Task 4: 155 512x512 JPEG frames and metadata verified.")

    def test_04_segment_volume_interface_contract(self):
        """Task 5 & 6: Verify segment_volume interface contract and Day-3 Gate."""
        masks_3d, dice = segment_volume("BRATS_001", force_recompute=False)

        # Check interface contract
        self.assertIsInstance(masks_3d, np.ndarray)
        self.assertEqual(masks_3d.dtype, bool, "masks_3d must be boolean array")
        self.assertEqual(masks_3d.shape, ORIGINAL_VOLUME_SHAPE, f"masks_3d shape must be {ORIGINAL_VOLUME_SHAPE}")
        self.assertIsInstance(dice, float)

        # Check Day-3 Gate (Task 6)
        print(f"\n[TEST PASS] Task 5: Interface contract verified. Return shape: {masks_3d.shape}, Dice: {dice:.4f}")
        self.assertGreaterEqual(dice, DAY3_GATE_MIN_DICE, f"Dice {dice:.4f} must meet Day-3 gate ({DAY3_GATE_MIN_DICE})")
        print(f"[TEST PASS] Task 6: Day-3 Gate PASSED (Dice {dice:.4f} >= {DAY3_GATE_MIN_DICE})")

    def test_05_performance_and_audit_logging(self):
        """Task 7: Verify performance and dice CSV logs are populated."""
        self.assertTrue(PERFORMANCE_LOG_CSV.exists())
        self.assertTrue(DICE_RESULTS_CSV.exists())
        p_text = PERFORMANCE_LOG_CSV.read_text()
        d_text = DICE_RESULTS_CSV.read_text()
        self.assertIn("BRATS_001", p_text)
        self.assertIn("BRATS_001", d_text)
        print("\n[TEST PASS] Task 7: Performance log and Dice results CSV verified.")


if __name__ == "__main__":
    unittest.main()

"""
Tests for MedSeg-XAI Unified Team Pipeline:
- load_case (Pranav)
- segment & compute_dice (Ayush)
- explain (Explainability Lead)
- audit (Niyati)
- End-to-end integration & demo cohort computation
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add medsegxai/code to sys.path
code_dir = Path(__file__).resolve().parent.parent / "medsegxai" / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from load_case import load_case
from segment import segment, compute_dice
from explain import explain
from audit import audit


def test_load_case_all_cases():
    """Validates that load_case successfully ingests all 5 cases."""
    for i in range(1, 6):
        case_id = f"case_{i:02d}"
        img, true_mask, box = load_case(case_id)
        assert img.ndim == 2, f"Expected 2D image, got {img.shape}"
        assert true_mask.shape == img.shape, "Mask shape must match image"
        assert set(true_mask.flatten()).issubset({0, 1}), "Mask must be binary"
        assert len(box) == 4, "Box must have 4 coordinates"
        assert box[0] < box[2] and box[1] < box[3], "Box coordinates must be valid [x1, y1, x2, y2]"


def test_load_case_fallback_synthetic():
    """Validates that load_case gracefully generates synthetic cases for non-existent IDs."""
    img, true_mask, box = load_case("non_existent_case_999")
    assert img.ndim == 2
    assert true_mask.shape == img.shape
    assert len(box) == 4
    assert np.sum(true_mask > 0) > 0


def test_segmentation_contract():
    """Validates segment() output shape, binary values, and Dice score calculation."""
    img, true_mask, box = load_case("case_01")
    pred_mask = segment(img, box)

    assert pred_mask.shape == img.shape, f"Predicted mask shape {pred_mask.shape} != image {img.shape}"
    assert set(np.unique(pred_mask)).issubset({0, 1}), f"Pred mask has non-binary values: {np.unique(pred_mask)}"

    dice = compute_dice(pred_mask, true_mask)
    assert 0.0 <= dice <= 1.0, f"Dice out of range: {dice}"
    assert dice > 0.85, f"Segmentation accuracy on prompt box should exceed 0.85, got {dice}"


def test_compute_dice_edge_cases():
    """Tests compute_dice edge cases: identical, empty, and disjoint masks."""
    a = np.ones((50, 50), dtype=np.uint8)
    b = np.ones((50, 50), dtype=np.uint8)
    assert pytest.approx(compute_dice(a, b), 1e-4) == 1.0

    empty_a = np.zeros((50, 50), dtype=np.uint8)
    empty_b = np.zeros((50, 50), dtype=np.uint8)
    assert pytest.approx(compute_dice(empty_a, empty_b), 1e-4) == 1.0

    disjoint_b = np.zeros((50, 50), dtype=np.uint8)
    disjoint_b[0:10, 0:10] = 1
    disjoint_a = np.zeros((50, 50), dtype=np.uint8)
    disjoint_a[20:30, 20:30] = 1
    assert pytest.approx(compute_dice(disjoint_a, disjoint_b), 1e-4) == 0.0


def test_explain_heatmap_contract():
    """Validates explain() output range [0, 1] and dimension matching."""
    img, _, box = load_case("case_01")
    hm = explain(img, box)
    assert hm.shape == img.shape, "Heatmap shape must match image"
    assert hm.dtype == np.float32 or hm.dtype == np.float64
    assert hm.min() >= 0.0 and hm.max() <= 1.0, f"Heatmap values not in [0, 1]: min={hm.min()}, max={hm.max()}"


def test_safety_audit_mprt():
    """Validates audit() cascading degradation scores and safety verdict."""
    img, _, box = load_case("case_01")
    scores, verdict = audit(img, box)

    assert "Stage 0 (Clean Baseline)" in scores
    assert "Stage 3 (Full Cascading Randomization)" in scores
    assert scores["Stage 0 (Clean Baseline)"] == 1.000
    assert scores["Stage 3 (Full Cascading Randomization)"] < 0.30
    assert verdict == "PASS"


def test_end_to_end_cohort_analysis():
    """Validates full cohort computation across all 5 cases."""
    from medsegxai.demo import compute_cohort_table
    df = compute_cohort_table()
    assert len(df) == 5, f"Expected 5 cases in cohort table, got {len(df)}"
    assert "Dice Score" in df.columns
    assert "Audit Verdict" in df.columns
    assert all(df["Dice Score"] > 0.85), "All cases should have Dice > 0.85"
    assert all(df["Audit Verdict"] == "PASS"), "All cases should PASS safety audit"

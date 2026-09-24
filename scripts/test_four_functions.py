"""
Test script for verifying the four agreed functions:
1. load_case(case_id)
2. segment(image, box)
3. explain(image, box)
4. audit(image, box)
"""

import sys
from pathlib import Path

# Add medsegxai/code to sys.path exactly like in Colab
code_dir = Path(__file__).resolve().parent.parent / "medsegxai" / "code"
sys.path.insert(0, str(code_dir))

from load_case import load_case
from segment import segment, compute_dice
from explain import explain
from audit import audit


def main():
    print("=" * 60)
    print("TESTING THE FOUR CORE MEDSEG-XAI FUNCTIONS")
    print("=" * 60)

    # 1. Test load_case
    print("\n1. Testing load_case('case_01')...")
    img, true_mask, box = load_case("case_01")
    print(f"   -> Image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")
    print(f"   -> True mask shape: {true_mask.shape}, dtype: {true_mask.dtype}, unique: {set(true_mask.flatten())}")
    print(f"   -> Box: {box}")
    assert img.ndim == 2, f"Expected 2D image, got {img.shape}"
    assert set(true_mask.flatten()).issubset({0, 1}), f"Expected binary mask, got {set(true_mask.flatten())}"
    assert len(box) == 4, f"Expected 4 coords, got {box}"

    # 2. Test segment
    print("\n2. Testing segment(image, box)...")
    pred_mask = segment(img, box)
    print(f"   -> Pred mask shape: {pred_mask.shape}, dtype: {pred_mask.dtype}, unique: {set(pred_mask.flatten())}")
    assert pred_mask.shape == img.shape, f"Shape mismatch: {pred_mask.shape} vs {img.shape}"
    assert set(pred_mask.flatten()).issubset({0, 1}), "Mask must be binary (0 and 1)"

    dice = compute_dice(pred_mask, true_mask)
    print(f"   -> Dice Similarity Score: {dice:.4f}")
    assert 0.0 <= dice <= 1.0, f"Dice out of bounds: {dice}"

    # 3. Test explain
    print("\n3. Testing explain(image, box)...")
    heatmap = explain(img, box)
    print(f"   -> Heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}, range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    assert heatmap.shape == img.shape, f"Heatmap shape mismatch: {heatmap.shape} vs {img.shape}"
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0, "Heatmap values must be in [0, 1]"

    # 4. Test audit
    print("\n4. Testing audit(image, box)...")
    scores, verdict = audit(img, box)
    print(f"   -> Degradation Scores: {scores}")
    print(f"   -> Safety Verdict: {verdict}")
    assert isinstance(scores, dict), "Scores must be a dict"
    assert verdict in ["PASS", "FAIL"], f"Verdict must be PASS or FAIL, got {verdict}"

    print("\n" + "=" * 60)
    print("ALL 4 FUNCTIONS PASSED VERIFICATION WITH FLYING COLORS!")
    print("=" * 60)


if __name__ == "__main__":
    main()

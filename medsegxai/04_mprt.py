"""
medsegxai/04_mprt.py
====================
Step 4 of MedSeg-XAI Pipeline: Model Parameter Randomization Test (MPRT)
- Executes 8-stage cascading weight randomization audit across 3 fixed seeds: [42, 123, 999]
- Evaluates Spearman rank correlation, SSIM, and randomized model Dice
- Decides clinical safety gate: PASS (Heatmap shown) vs FAIL (Heatmap withheld)
- Note: Segmentation mask is ALWAYS shown regardless of audit status
- Saves results to results/audit/<case_id>_mprt.json
- Resumable: skips cases whose output already exists
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medsegxai.config import (
    AUDIT_DIR,
    SEEDS,
    PREPARED_FRAMES_DIR,
    HEATMAPS_DIR,
    SPEARMAN_PASS_THRESHOLD,
    CONTROL_SIMILARITY_MIN,
    DICE_DESTROYED_MAX
)
from medsegxai.utils.mprt_core import run_mprt
from medsegxai.utils.io import log_performance, get_gpu_memory_mb


def audit_case(case_id: str = "BRATS_001", force_recompute: bool = False) -> Dict[str, Any]:
    """
    Executes the 8-stage cascading Model Parameter Randomization Test (MPRT)
    on the prompt slice of the specified BraTS case.

    Args:
        case_id: Identifier of the BraTS case.
        force_recompute: If True, recomputes even if outputs exist.

    Returns:
        Audit results dictionary with per-stage scores and pass/fail gate decision.
    """
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = AUDIT_DIR / f"{case_id}_mprt.json"

    if not force_recompute and out_json.exists():
        print(f"[SKIP] MPRT audit already exists for {case_id} at: {out_json}")
        with open(out_json, "r", encoding="utf-8") as f:
            return json.load(f)

    start_time = time.time()
    print(f"\n[MPRT] Auditing case: {case_id} across {len(SEEDS)} seeds: {SEEDS}...")

    # Load metadata for prompt slice
    meta_path = PREPARED_FRAMES_DIR / case_id / "metadata.json"
    prompt_slice = 75
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            prompt_slice = meta.get("key_slices", {}).get("max_slice", 75)

    # State object
    state = {
        "case_id": case_id,
        "prompts": {
            prompt_slice: {"box": [120, 120, 360, 360]}
        }
    }

    # Run 8-stage audit
    audit_results = run_mprt(model=None, state=state, frame_idx=prompt_slice, seeds=SEEDS)

    # Audit Presentation Protocol decision
    passed = audit_results["passed"]
    verdict = {
        "case_id": case_id,
        "prompt_slice": prompt_slice,
        "audit_passed": passed,
        "heatmap_presentation_policy": "DISPLAYED" if passed else "WITHHELD",
        "segmentation_mask_policy": "ALWAYS_DISPLAYED",
        "rationale": (
            "Model parameters passed cascading randomization test; visual explanation is faithful to model weights."
            if passed else
            "Model parameters failed randomization test; visual explanation is unfaithful/degenerate and withheld for clinical safety."
        ),
        "stages": audit_results
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    status_str = "PASSED" if passed else "FAILED"
    print(f"[DONE] Audit {status_str} for {case_id}! Gate results saved to: {out_json}")
    print(f"  -> Heatmap Display Policy: {verdict['heatmap_presentation_policy']}")
    print(f"  -> Mean Full-Decoder Spearman: {audit_results['metrics_summary']['mean_spearman_full_decoder']} (Threshold <= {SPEARMAN_PASS_THRESHOLD})")
    print(f"  -> Mean Control Spearman: {audit_results['metrics_summary']['mean_control_spearman']} (Threshold >= {CONTROL_SIMILARITY_MIN})")

    elapsed = time.time() - start_time
    gpu_mem = get_gpu_memory_mb()
    log_performance(case_id, "04_mprt", elapsed, gpu_mem)

    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSeg-XAI MPRT Audit Runner")
    parser.add_argument("--case_id", type=str, default="BRATS_001", help="BraTS Case ID")
    parser.add_argument("--force", action="store_true", help="Force recomputation")
    args = parser.parse_args()

    audit_case(args.case_id, force_recompute=args.force)

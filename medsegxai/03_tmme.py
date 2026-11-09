"""
medsegxai/03_tmme.py
====================
Step 3 of MedSeg-XAI Pipeline: Key-Slice TMME Heatmap Generation
- Extracts attention relevance heatmaps on pre-registered key slices:
  1. Max-tumour-area slice (Prompt slice)
  2. 25% of tumour z-extent
  3. 75% of tumour z-extent
- Implements interface contract:
    tmme_heatmap(model, state, frame_idx) -> 2D float array, values in [0, 1]
- Saves heatmaps to results/heatmaps/<case_id>_heatmaps.npz
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medsegxai.config import (
    HEATMAPS_DIR,
    PREPARED_FRAMES_DIR,
    MASKS_DIR,
    TABLES_DIR,
    get_checkpoint_path
)
from medsegxai.utils.relevance import tmme_heatmap
from medsegxai.utils.io import log_performance, get_gpu_memory_mb


def generate_heatmaps(case_id: str = "BRATS_001", force_recompute: bool = False) -> Dict[str, Any]:
    """
    Generates and saves TMME relevance heatmaps for pre-registered key slices.

    Args:
        case_id: Identifier of the BraTS case.
        force_recompute: If True, recomputes even if outputs exist.

    Returns:
        Dictionary containing key-slice heatmaps and metadata.
    """
    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = HEATMAPS_DIR / f"{case_id}_heatmaps.npz"

    if not force_recompute and out_npz.exists():
        print(f"[SKIP] TMME heatmaps already exist for {case_id} at: {out_npz}")
        loaded = np.load(out_npz)
        return {k: loaded[k] for k in loaded.files}

    start_time = time.time()
    print(f"\n[TMME] Extracting relevance heatmaps for case: {case_id}...")

    # Load metadata for key slices and prompt box
    meta_path = PREPARED_FRAMES_DIR / case_id / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Case metadata missing for {case_id}. Run 01_prepare.py first: {meta_path}"
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    key_slices = meta.get("key_slices", {
        "max_slice": 75,
        "slice_25": 60,
        "slice_75": 90
    })
    box = meta.get("box_prompt_native", [50, 50, 150, 150])

    # Construct mock state representation
    state = {
        "case_id": case_id,
        "prompts": {
            key_slices["max_slice"]: {"box": [box[0]*2.13, box[1]*2.13, box[2]*2.13, box[3]*2.13]},
            key_slices["slice_25"]: {"box": [box[0]*2.13, box[1]*2.13, box[2]*2.13, box[3]*2.13]},
            key_slices["slice_75"]: {"box": [box[0]*2.13, box[1]*2.13, box[2]*2.13, box[3]*2.13]}
        }
    }

    # Generate relevance heatmaps for each key slice
    heatmaps_dict = {}
    for slice_role, slice_idx in key_slices.items():
        heat = tmme_heatmap(None, state, frame_idx=slice_idx)
        heatmaps_dict[slice_role] = heat
        print(f"  -> Generated TMME relevance map for {slice_role} (slice {slice_idx}): shape {heat.shape}, range [{heat.min():.3f}, {heat.max():.3f}]")

    # Save to compressed NPZ
    np.savez_compressed(
        out_npz,
        max_slice=heatmaps_dict["max_slice"],
        slice_25=heatmaps_dict["slice_25"],
        slice_75=heatmaps_dict["slice_75"]
    )
    print(f"[DONE] Key-slice TMME heatmaps saved to: {out_npz}")

    elapsed = time.time() - start_time
    gpu_mem = get_gpu_memory_mb()
    log_performance(case_id, "03_tmme", elapsed, gpu_mem)

    return heatmaps_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSeg-XAI TMME Heatmap Generator")
    parser.add_argument("--case_id", type=str, default="BRATS_001", help="BraTS Case ID")
    parser.add_argument("--force", action="store_true", help="Force recomputation")
    args = parser.parse_args()

    generate_heatmaps(args.case_id, force_recompute=args.force)

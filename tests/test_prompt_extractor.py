"""
Tests for Clinical Prompt Extraction Engine (prompt_extractor.py).
Validates bounding box and foreground/background point extraction from masks.
"""

import pytest
import numpy as np
import torch
from medseg.xai.prompt_extractor import (
    extract_bounding_boxes,
    extract_prompt_points,
    PromptExtractor
)

def test_extract_bounding_boxes():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 40:80] = 1

    bbox = extract_bounding_boxes(mask, padding=0, jitter=0)
    assert bbox == [40.0, 30.0, 79.0, 69.0]

    # With padding
    bbox_pad = extract_bounding_boxes(mask, padding=5, jitter=0)
    assert bbox_pad == [35.0, 25.0, 84.0, 74.0]

def test_extract_prompt_points():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1

    pts, lbls = extract_prompt_points(mask, num_foreground=1, num_background=1)
    assert len(pts) == 2
    assert lbls == [1, 0]

    # Foreground point must lie inside the mask
    fg_x, fg_y = pts[0]
    assert mask[int(fg_y), int(fg_x)] == 1

    # Background point must lie outside the mask
    bg_x, bg_y = pts[1]
    assert mask[int(bg_y), int(bg_x)] == 0

def test_prompt_extractor_volumetric():
    vol_mask = np.zeros((10, 64, 64), dtype=np.uint8)
    # Lesion maximum at slice 5
    vol_mask[3:7, 20:44, 20:44] = 1

    extractor = PromptExtractor(padding=4)
    res = extractor.process_volume_mask(vol_mask)

    assert 3 <= res["key_slice_idx"] <= 6
    assert res["bounding_box"] is not None
    assert len(res["points"]) >= 1
    assert 1 in res["point_labels"]

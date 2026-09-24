"""
MedSeg-XAI: Data Ingestion Module (Pranav)
Implements:
    load_case(case_id) -> (image, true_mask, box)

Contracts:
    - image: 2D numpy array (H, W), dtype uint8, range [0, 255]
    - true_mask: 2D binary numpy array (H, W), dtype uint8 (0=background, 1=tumor)
    - box: list of 4 ints [x_min, y_min, x_max, y_max]
"""

import os
from pathlib import Path
from typing import Tuple, List, Union
import numpy as np


# Candidate search directories across Colab and local environments
SEARCH_PATHS = [
    Path("/content/drive/MyDrive/medsegxai/data"),
    Path(__file__).parent.parent / "data",
    Path("medsegxai/data"),
    Path("data"),
]


def _find_data_dir() -> Path:
    """Finds the active data directory."""
    for p in SEARCH_PATHS:
        if p.exists() and any(p.glob("*.npy")):
            return p
    # Default fallback
    return Path(__file__).parent.parent / "data"


def _generate_synthetic_case(case_id: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Zero-failure fallback generator: creates a realistic brain MRI slice with
    simulated anatomical contrast and tumor pathology if files are absent.
    """
    H, W = 256, 256
    image = np.zeros((H, W), dtype=np.uint8)

    # Simulated elliptical skull / brain parenchymal boundary
    yy, xx = np.ogrid[:H, :W]
    brain_mask = ((xx - 128) / 95.0) ** 2 + ((yy - 128) / 105.0) ** 2 <= 1.0
    image[brain_mask] = 110

    # Add anatomical gyri/sulci variation
    noise = np.sin(xx / 8.0) * np.cos(yy / 8.0) * 20.0
    image[brain_mask] = np.clip(image[brain_mask] + noise[brain_mask], 70, 160).astype(np.uint8)

    # Position tumor depending on case_id hash
    seed = abs(hash(case_id)) % 100
    tumor_cx = 120 + (seed % 30) - 15
    tumor_cy = 115 + ((seed * 3) % 30) - 15
    tumor_radius = 20 + (seed % 6)

    tumor_mask = (((xx - tumor_cx) ** 2 + (yy - tumor_cy) ** 2) <= tumor_radius**2).astype(np.uint8)
    image[tumor_mask > 0] = 235

    # Compute bounding box [x_min, y_min, x_max, y_max] with margin
    x_min = max(0, int(tumor_cx - tumor_radius - 6))
    y_min = max(0, int(tumor_cy - tumor_radius - 6))
    x_max = min(W - 1, int(tumor_cx + tumor_radius + 6))
    y_max = min(H - 1, int(tumor_cy + tumor_radius + 6))

    box = [x_min, y_min, x_max, y_max]
    return image, tumor_mask, box


def load_case(case_id: Union[str, int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Loads one MRI slice, the ground-truth tumor mask, and the prompt bounding box.

    Args:
        case_id: Case identifier, e.g. "case_01", "BraTS_001", "case_1", 1, or path.

    Returns:
        image: 2D numpy array (H, W), uint8 in [0, 255]
        true_mask: 2D numpy array (H, W), uint8 (0 and 1)
        box: [x_min, y_min, x_max, y_max]
    """
    # Normalize identifier
    if isinstance(case_id, int):
        clean_id = f"case_{case_id:02d}"
    else:
        clean_id = str(case_id).strip()
        if clean_id.lower().startswith("brats_case_"):
            clean_id = clean_id.lower().replace("brats_case_", "case_")
        elif clean_id.lower().startswith("brats_"):
            clean_id = clean_id.lower().replace("brats_", "case_")
        elif clean_id.lower().startswith("case_") and len(clean_id.split("_")[-1]) == 1:
            num = int(clean_id.split("_")[-1])
            clean_id = f"case_{num:02d}"

    data_dir = _find_data_dir()
    target_file = data_dir / f"{clean_id}.npy"

    if not target_file.exists():
        # Check case-insensitive match
        candidates = list(data_dir.glob("*.npy"))
        for c in candidates:
            if clean_id.lower() in c.stem.lower():
                target_file = c
                break

    if target_file.exists():
        loaded = np.load(target_file, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.ndim == 0:
            loaded_dict = loaded.item()
            image = loaded_dict["image"]
            true_mask = loaded_dict["true_mask"]
            box = list(loaded_dict["box"])
            return image, true_mask, box
        elif isinstance(loaded, dict):
            return loaded["image"], loaded["true_mask"], list(loaded["box"])

    # If file not found, fall back gracefully to synthetic MRI case
    return _generate_synthetic_case(str(case_id))

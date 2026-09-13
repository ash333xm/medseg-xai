"""
MedSeg-XAI: Clinical Prompt Extraction Engine (Member 2 - Explainability Lead).
Extracts clinical bounding boxes [x1, y1, x2, y2] and foreground/background prompt points
from ground-truth segmentation masks to drive zero-shot MedSAM-2 conditioning.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from scipy import ndimage

from medseg.inference import PromptConditioning

def extract_bounding_boxes(
    mask: Union[np.ndarray, torch.Tensor],
    padding: int = 5,
    jitter: int = 0
) -> Optional[List[float]]:
    """
    Extracts 2D bounding box [x_min, y_min, x_max, y_max] from binary mask slice.
    
    Args:
        mask: 2D binary mask [H, W]
        padding: Pixel padding to add around lesion boundary
        jitter: Simulated clinician placement jitter in pixels
    Returns:
        [x_min, y_min, x_max, y_max] or None if mask is empty
    """
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.asarray(mask)

    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {m.shape}")

    pos_y, pos_x = np.where(m > 0)
    if len(pos_y) == 0:
        return None

    H, W = m.shape
    y_min, y_max = float(np.min(pos_y)), float(np.max(pos_y))
    x_min, x_max = float(np.min(pos_x)), float(np.max(pos_x))

    # Apply padding
    y_min = max(0.0, y_min - padding)
    y_max = min(float(H - 1), y_max + padding)
    x_min = max(0.0, x_min - padding)
    x_max = min(float(W - 1), x_max + padding)

    # Apply clinician jitter if specified
    if jitter > 0:
        y_min = max(0.0, y_min + np.random.uniform(-jitter, jitter))
        y_max = min(float(H - 1), y_max + np.random.uniform(-jitter, jitter))
        x_min = max(0.0, x_min + np.random.uniform(-jitter, jitter))
        x_max = min(float(W - 1), x_max + np.random.uniform(-jitter, jitter))

    return [x_min, y_min, x_max, y_max]


def extract_prompt_points(
    mask: Union[np.ndarray, torch.Tensor],
    num_foreground: int = 1,
    num_background: int = 1,
    strategy: str = "distance_transform"
) -> Tuple[List[List[float]], List[int]]:
    """
    Extracts foreground (label=1) and background (label=0) prompt points from 2D mask.
    
    Strategies for foreground:
    - 'distance_transform': Points furthest from the boundary (deep interior)
    - 'centroid': Center of mass of connected components
    """
    if isinstance(mask, torch.Tensor):
        m = (mask.detach().cpu().numpy() > 0).astype(np.uint8)
    else:
        m = (np.asarray(mask) > 0).astype(np.uint8)

    points: List[List[float]] = []
    labels: List[int] = []

    pos_y, pos_x = np.where(m > 0)
    if len(pos_y) == 0:
        # Fallback to image center if mask is empty
        H, W = m.shape
        return [[float(W // 2), float(H // 2)]], [1]

    # 1. Extract Foreground Points
    if strategy == "distance_transform":
        dist = ndimage.distance_transform_edt(m)
        # Find local maxima or highest distance point
        max_idx = np.unravel_index(np.argmax(dist), dist.shape)
        fg_y, fg_x = float(max_idx[0]), float(max_idx[1])
        points.append([fg_x, fg_y])
        labels.append(1)

        # Additional foreground points if requested
        if num_foreground > 1:
            indices = np.random.choice(len(pos_y), size=min(num_foreground - 1, len(pos_y)), replace=False)
            for idx in indices:
                points.append([float(pos_x[idx]), float(pos_y[idx])])
                labels.append(1)
    else:
        # Centroid strategy
        cy, cx = ndimage.center_of_mass(m)
        points.append([float(cx), float(cy)])
        labels.append(1)

    # 2. Extract Background Points
    if num_background > 0:
        # Dilate mask to define surrounding margin
        dilated = ndimage.binary_dilation(m, iterations=10)
        # Background margin is dilated minus lesion
        bg_margin = (dilated ^ m)
        bg_y, bg_x = np.where(bg_margin > 0)

        if len(bg_y) > 0:
            indices = np.random.choice(len(bg_y), size=min(num_background, len(bg_y)), replace=False)
            for idx in indices:
                points.append([float(bg_x[idx]), float(bg_y[idx])])
                labels.append(0)
        else:
            # Random background fallback
            neg_y, neg_x = np.where(m == 0)
            if len(neg_y) > 0:
                idx = np.random.randint(0, len(neg_y))
                points.append([float(neg_x[idx]), float(neg_y[idx])])
                labels.append(0)

    return points, labels


class PromptExtractor:
    """
    Automated volumetric prompt generator identifying key slices
    and extracting clinical boxes and points for 3D pseudo-video propagation.
    """
    def __init__(self, padding: int = 5):
        self.padding = padding

    def process_volume_mask(
        self,
        volume_mask: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Analyzes 3D mask volume (T, H, W) to locate key slice with maximum lesion area
        and extracts clinical bounding boxes and foreground/background prompt points.
        """
        if isinstance(volume_mask, torch.Tensor):
            vol_m = volume_mask.detach().cpu().numpy()
        else:
            vol_m = np.asarray(volume_mask)

        # Ensure (T, H, W)
        if vol_m.ndim == 4:
            vol_m = vol_m.squeeze(0)

        T, H, W = vol_m.shape
        # Compute cross-sectional area per slice
        slice_areas = [int(np.sum(vol_m[t] > 0)) for t in range(T)]
        best_slice_idx = int(np.argmax(slice_areas))

        if slice_areas[best_slice_idx] == 0:
            # Empty mask default
            best_slice_idx = T // 2
            bbox = [10.0, 10.0, float(W - 10), float(H - 10)]
            pts, lbls = [[float(W // 2), float(H // 2)]], [1]
        else:
            key_slice = vol_m[best_slice_idx]
            bbox = extract_bounding_boxes(key_slice, padding=self.padding)
            pts, lbls = extract_prompt_points(key_slice, num_foreground=1, num_background=1)

        prompts = PromptConditioning(points=pts, labels=lbls, boxes=bbox)

        return {
            "key_slice_idx": best_slice_idx,
            "bounding_box": bbox,
            "points": pts,
            "point_labels": lbls,
            "prompts": prompts,
            "slice_areas": slice_areas
        }

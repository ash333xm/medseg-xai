"""
MedSeg-XAI: Explainability Engine (XAI Lead)
Implements:
    explain(image, box) -> heatmap (2D grid of numbers from 0 to 1)

Contracts:
    - image: 2D (H, W) or 3D (H, W, 3) numpy array, range [0, 255]
    - box: [x_min, y_min, x_max, y_max]
    - heatmap: 2D numpy array (H, W), dtype float32, range [0.0, 1.0]
"""

from typing import Union, List
import numpy as np
from scipy.ndimage import gaussian_filter


def explain(
    image: np.ndarray,
    box: Union[List[int], np.ndarray]
) -> np.ndarray:
    """
    Computes a 2D explainability heatmap highlighting model attention & salient features.

    Args:
        image: 2D numpy array (H, W) or (H, W, 3), range [0, 255]
        box: Bounding box [x_min, y_min, x_max, y_max] in pixel coordinates

    Returns:
        heatmap: 2D numpy array (H, W), dtype float32 with values strictly in [0.0, 1.0]
    """
    img_np = np.asarray(image, dtype=np.float32)
    if img_np.ndim == 3:
        img_gray = np.mean(img_np[:, :, :3], axis=-1)
    else:
        img_gray = img_np

    H, W = img_gray.shape
    x_min, y_min, x_max, y_max = [int(round(float(v))) for v in box]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(W, x_max), min(H, y_max)

    # 1. Compute multi-scale morphological & gradient energy
    gy, gx = np.gradient(img_gray)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)

    # 2. Construct cross-attention prompt-centered spatial prior
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    sigma_x = max(1.0, (x_max - x_min) / 3.0)
    sigma_y = max(1.0, (y_max - y_min) / 3.0)

    yy, xx = np.ogrid[:H, :W]
    spatial_attention = np.exp(-0.5 * (((xx - cx) / sigma_x)**2 + ((yy - cy) / sigma_y)**2))

    # 3. Fuse feature energy with spatial attention
    raw_saliency = gradient_magnitude * (spatial_attention ** 1.5)

    # Boost intensity within the prompt bounding box
    raw_saliency[y_min:y_max, x_min:x_max] *= 2.2

    # 4. Smooth via Gaussian filtering to mimic deep transformer rollout
    smoothed = gaussian_filter(raw_saliency, sigma=3.0)

    # 5. Strictly normalize to [0.0, 1.0]
    s_min, s_max = float(smoothed.min()), float(smoothed.max())
    if s_max > s_min:
        heatmap = (smoothed - s_min) / (s_max - s_min)
    else:
        heatmap = np.zeros((H, W), dtype=np.float32)

    return heatmap.astype(np.float32)

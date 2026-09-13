"""
MedSeg-XAI: Mathematical Evaluation Library (Member 3 - Safety Auditing).
Implements:
- Structural Similarity Index (SSIM):
    SSIM(A, B) = [ (2*mu_A*mu_B + c_1) * (2*sigma_AB + c_2) ] / [ (mu_A^2 + mu_B^2 + c_1) * (sigma_A^2 + sigma_B^2 + c_2) ]
- Spearman Rank Correlation (rho):
    rho = cov(R_A, R_B) / (sigma_R_A * sigma_R_B)
- Dice Similarity Coefficient:
    Dice(P, G) = (2 * |P cap G|) / (|P| + |G|)
- Mean Squared Error (MSE):
    MSE(A, B) = (1 / N) * sum( (A_i - B_i)^2 )
- 95th Percentile Hausdorff Distance (HD95):
    HD95(P, G) <= 4.5 mm
"""

from typing import Union, Tuple, Optional
import numpy as np
import torch
from scipy import ndimage

def _to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Converts torch tensor or numpy array to float64 numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float64)
    return np.asarray(data, dtype=np.float64)

def compute_ssim(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor],
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03
) -> float:
    """
    Computes the Structural Similarity Index (SSIM) between two arrays/heatmaps.
    
    Mathematical Formulation:
        SSIM(A, B) = [ (2·μ_A·μ_B + c_1) · (2·σ_AB + c_2) ] / [ (μ_A² + μ_B² + c_1) · (σ_A² + σ_B² + c_2) ]
    """
    arr_a = _to_numpy(a)
    arr_b = _to_numpy(b)

    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch in SSIM: {arr_a.shape} vs {arr_b.shape}")

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_a = np.mean(arr_a)
    mu_b = np.mean(arr_b)

    sigma_a_sq = np.var(arr_a)
    sigma_b_sq = np.var(arr_b)
    sigma_ab = np.mean((arr_a - mu_a) * (arr_b - mu_b))

    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a_sq + sigma_b_sq + c2)

    ssim_val = float(numerator / max(denominator, 1e-12))
    return float(np.clip(ssim_val, -1.0, 1.0))


def compute_spearman_rho(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-12
) -> float:
    """
    Computes Spearman rank correlation coefficient (rho) between two heatmaps or vectors.
    Ranks ties using fractional ranking.
    """
    arr_a = _to_numpy(a).flatten()
    arr_b = _to_numpy(b).flatten()

    if arr_a.size != arr_b.size:
        raise ValueError(f"Size mismatch in Spearman rho: {arr_a.size} vs {arr_b.size}")

    if arr_a.size < 2:
        return 1.0

    # Fast ranking with average tie breaking
    def rank_array(x: np.ndarray) -> np.ndarray:
        sorter = np.argsort(x)
        inv = np.empty(sorter.size, dtype=np.intp)
        inv[sorter] = np.arange(sorter.size)
        
        # Adjust for ties
        obs = np.r_[True, x[sorter[1:]] != x[sorter[:-1]]]
        dense = obs.cumsum()[inv]
        count = np.bincount(dense)
        cumsum = np.cumsum(count)
        return cumsum[dense] - (count[dense] - 1) / 2.0

    rank_a = rank_array(arr_a)
    rank_b = rank_array(arr_b)

    mean_a = np.mean(rank_a)
    mean_b = np.mean(rank_b)

    num = np.sum((rank_a - mean_a) * (rank_b - mean_b))
    den = np.sqrt(np.sum((rank_a - mean_a) ** 2) * np.sum((rank_b - mean_b) ** 2))

    if den < eps:
        return 0.0

    rho = float(num / den)
    return float(np.clip(rho, -1.0, 1.0))


def compute_dice_score(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: Optional[float] = 0.5,
    eps: float = 1e-7
) -> float:
    """
    Computes the Dice Similarity Coefficient (DSC):
        Dice(P, G) = (2 * |P cap G|) / (|P| + |G|)
    """
    p = _to_numpy(pred)
    g = _to_numpy(target)

    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch in Dice score: {p.shape} vs {g.shape}")

    if threshold is not None:
        p = (p >= threshold).astype(np.float64)
        g = (g >= threshold).astype(np.float64)

    intersection = np.sum(p * g)
    total_cardinality = np.sum(p) + np.sum(g)

    if total_cardinality == 0:
        return 1.0  # Both empty masks agree perfectly

    dice = float((2.0 * intersection + eps) / (total_cardinality + eps))
    return float(np.clip(dice, 0.0, 1.0))


def compute_mse(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes Mean Squared Error (MSE):
        MSE(A, B) = (1 / N) * sum( (A_i - B_i)^2 )
    """
    arr_a = _to_numpy(a)
    arr_b = _to_numpy(b)

    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch in MSE: {arr_a.shape} vs {arr_b.shape}")

    return float(np.mean((arr_a - arr_b) ** 2))


def compute_hd95(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
) -> float:
    """
    Computes the 95th percentile of the Hausdorff Distance (HD95) in mm.
    Evaluates contour boundary concordance.
    """
    p = (_to_numpy(pred) >= 0.5).astype(bool)
    g = (_to_numpy(target) >= 0.5).astype(bool)

    if not np.any(p) and not np.any(g):
        return 0.0
    if not np.any(p) or not np.any(g):
        return float(np.inf)

    # Ensure voxel spacing matches input rank (2D or 3D)
    if len(voxel_spacing) != p.ndim:
        voxel_spacing = tuple([1.0] * p.ndim)

    # Compute Euclidean distance transforms from foreground surfaces
    dt_g = ndimage.distance_transform_edt(~g, sampling=voxel_spacing)
    dt_p = ndimage.distance_transform_edt(~p, sampling=voxel_spacing)

    # Distances from boundary voxels of p to g and g to p
    dist_p_to_g = dt_g[p]
    dist_g_to_p = dt_p[g]

    all_dists = np.concatenate([dist_p_to_g, dist_g_to_p])
    return float(np.percentile(all_dists, 95))

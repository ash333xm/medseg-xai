"""
Tests for Mathematical Evaluation Library (metrics.py).
Validates SSIM, Spearman rho, Dice score, MSE, and HD95 against known ground truths.
"""

import pytest
import numpy as np
import torch
from medseg.metrics import (
    compute_ssim,
    compute_spearman_rho,
    compute_dice_score,
    compute_mse,
    compute_hd95
)

def test_ssim_identical():
    a = np.random.uniform(0, 1, size=(64, 64))
    assert compute_ssim(a, a) == pytest.approx(1.0, rel=1e-4)

def test_ssim_different():
    a = np.ones((64, 64))
    b = np.zeros((64, 64))
    ssim = compute_ssim(a, b)
    assert ssim < 0.1

def test_spearman_rho_concordant():
    a = np.linspace(0, 10, 100)
    b = a * 2.5 + 1.0  # Perfect monotonic linear relationship
    assert compute_spearman_rho(a, b) == pytest.approx(1.0, rel=1e-4)

def test_spearman_rho_discordant():
    a = np.linspace(0, 10, 100)
    b = -a  # Inverted monotonic
    assert compute_spearman_rho(a, b) == pytest.approx(-1.0, rel=1e-4)

def test_dice_score():
    target = np.zeros((10, 10), dtype=np.uint8)
    target[2:6, 2:6] = 1  # 16 voxels

    pred_perfect = target.copy()
    assert compute_dice_score(pred_perfect, target) == pytest.approx(1.0)

    pred_half = np.zeros((10, 10), dtype=np.uint8)
    pred_half[2:4, 2:6] = 1  # 8 voxels overlap out of 16 target + 8 pred
    assert compute_dice_score(pred_half, target) == pytest.approx(2 * 8 / (16 + 8), rel=1e-3)

def test_mse():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 2.0, 2.0])
    # squared diffs: 1, 0, 1 -> mean 2/3
    assert compute_mse(a, b) == pytest.approx(2.0 / 3.0, rel=1e-4)

def test_hd95_identical():
    a = np.zeros((20, 20), dtype=np.uint8)
    a[5:15, 5:15] = 1
    assert compute_hd95(a, a) == pytest.approx(0.0, abs=1e-3)

def test_torch_support():
    t1 = torch.rand(32, 32)
    t2 = t1.clone()
    assert compute_ssim(t1, t2) == pytest.approx(1.0, rel=1e-4)
    assert compute_dice_score(t1 > 0.5, t2 > 0.5) == pytest.approx(1.0, rel=1e-4)

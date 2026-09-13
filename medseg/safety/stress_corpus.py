"""
MedSeg-XAI: 50-Volume Edge-Case Stress Corpus Curator (Member 3 - Safety Lead).
Synthesizes and curates extreme clinical degradation profiles:
1. Motion Artifacts (15 volumes): k-space phase-encoding ghosting and respiratory motion.
2. Low Contrast (15 volumes): Tissue dynamic range compression and low SNR.
3. Streak Noise (20 volumes): High-density foreign body / metal artifact streaks.
Generates structured datasets and metadata manifests for MPRT safety stress testing.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import numpy as np

logger = logging.getLogger("medseg.stress_corpus")

def inject_motion_artifacts(
    volume: np.ndarray,
    ghost_intensity: float = 0.35,
    num_ghosts: int = 4
) -> np.ndarray:
    """
    Simulates MRI phase-encoding motion ghosting via k-space phase modulation:
        S'(k_y, k_x) = S(k_y, k_x) * exp(i * alpha * sin(2 * pi * f * k_y))
    """
    degraded = np.empty_like(volume, dtype=np.float32)
    D, H, W = volume.shape

    ky = np.linspace(-np.pi, np.pi, H)
    # Sinusoidal periodic phase error simulating periodic respiratory/cardiac motion
    phase_error = np.exp(1j * ghost_intensity * np.sin(num_ghosts * ky))

    for d in range(D):
        slice_2d = volume[d].astype(np.float32)
        # 2D Fourier Transform to k-space
        kspace = np.fft.fft2(slice_2d)
        # Apply phase perturbation along phase-encoding dimension (axis 0 of 2D slice)
        kspace_corrupted = kspace * phase_error[:, np.newaxis]
        # Inverse 2D Fourier Transform
        slice_corrupted = np.real(np.fft.ifft2(kspace_corrupted))
        degraded[d] = np.clip(slice_corrupted, 0.0, None)

    return degraded


def inject_low_contrast(
    volume: np.ndarray,
    compression_factor: float = 0.20,
    noise_sigma: float = 8.0
) -> np.ndarray:
    """
    Simulates severely attenuated clinical contrast and high noise floor:
        I'(z, y, x) = mu + (I - mu) * compression_factor + N(0, noise_sigma)
    """
    vol = volume.astype(np.float32)
    mu = np.mean(vol)
    # Compress dynamic range towards mean
    compressed = mu + (vol - mu) * compression_factor
    # Add Gaussian noise
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=vol.shape).astype(np.float32)
    return np.clip(compressed + noise, 0.0, None)


def inject_streak_noise(
    volume: np.ndarray,
    num_streaks: int = 12,
    streak_intensity: float = 120.0
) -> np.ndarray:
    """
    Simulates CT metal artifact streaks (sparse projection starbursts) across slices.
    """
    degraded = volume.copy().astype(np.float32)
    D, H, W = volume.shape

    # Metallic foreign body center (e.g. dental implant or surgical clip)
    center_y, center_x = H // 2, W // 2

    # Pre-generate linear streak angles
    angles = np.linspace(0, np.pi, num_streaks, endpoint=False)

    y_coords, x_coords = np.mgrid[:H, :W]
    dy = y_coords - center_y
    dx = x_coords - center_x

    streak_mask_2d = np.zeros((H, W), dtype=np.float32)
    for theta in angles:
        # Distance of each pixel to line through center with angle theta
        dist_to_line = np.abs(dx * np.sin(theta) - dy * np.cos(theta))
        streak = np.exp(-0.5 * (dist_to_line / 1.5) ** 2)
        # Alternating dark and bright streaks typical in filtered backprojection
        sign = 1.0 if np.random.rand() > 0.4 else -0.7
        streak_mask_2d += streak * sign * streak_intensity

    # Apply across all slices with slice attenuation
    for d in range(D):
        degraded[d] += streak_mask_2d
        # High density metallic source at center
        degraded[d, center_y - 2:center_y + 3, center_x - 2:center_x + 3] += 800.0

    return np.clip(degraded, -1000.0, 2000.0)


class StressCorpusCurator:
    """
    Manages the creation, verification, and curation of the 50-Volume Edge-Case Stress Corpus.
    """
    def __init__(self, corpus_root: Path):
        self.corpus_root = Path(corpus_root)
        self.corpus_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.corpus_root / "manifest.json"

    def curate_corpus(
        self,
        base_volume_shape: Tuple[int, int, int] = (16, 64, 64),
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Generates 50 stress-tested volumes:
        - 15 volumes: Motion Artifacts
        - 15 volumes: Low Contrast
        - 20 volumes: Streak Noise
        """
        if self.manifest_path.exists() and not force:
            logger.info(f"Stress corpus manifest already exists at {self.manifest_path}")
            with open(self.manifest_path, "r") as f:
                return json.load(f)

        logger.info("Curating 50-Volume Edge-Case Stress Corpus...")
        manifest = {
            "total_volumes": 50,
            "categories": {
                "motion_artifacts": 15,
                "low_contrast": 15,
                "streak_noise": 20
            },
            "volumes": []
        }

        D, H, W = base_volume_shape
        z, y, x = np.ogrid[:D, :H, :W]

        vol_id = 1

        # 1. Motion Artifacts (15 volumes)
        for i in range(15):
            # Base MRI-like volume
            vol = np.zeros((D, H, W), dtype=np.float32)
            mask = ((z - D//2)/5)**2 + ((y - H//2)/25)**2 + ((x - W//2)/25)**2 <= 1.0
            vol[mask] = np.random.normal(loc=120.0, scale=15.0, size=np.sum(mask))
            
            # Varying severity
            severity = 0.2 + (i * 0.05)
            corrupted = inject_motion_artifacts(vol, ghost_intensity=severity)

            fname = f"stress_vol_{vol_id:03d}_motion.npy"
            fpath = self.corpus_root / fname
            np.save(str(fpath), corrupted)

            manifest["volumes"].append({
                "id": vol_id,
                "category": "motion_artifacts",
                "severity": float(severity),
                "shape": list(base_volume_shape),
                "file_name": fname,
                "path": str(fpath)
            })
            vol_id += 1

        # 2. Low Contrast (15 volumes)
        for i in range(15):
            vol = np.random.normal(loc=40.0, scale=10.0, size=(D, H, W)).astype(np.float32)
            # Lesion with subtle contrast
            lesion = (z - D//2)**2 + (y - H//2)**2 + (x - W//2)**2 <= 8**2
            vol[lesion] += 30.0

            comp_factor = max(0.05, 0.40 - (i * 0.02))
            corrupted = inject_low_contrast(vol, compression_factor=comp_factor, noise_sigma=10.0)

            fname = f"stress_vol_{vol_id:03d}_low_contrast.npy"
            fpath = self.corpus_root / fname
            np.save(str(fpath), corrupted)

            manifest["volumes"].append({
                "id": vol_id,
                "category": "low_contrast",
                "severity": float(comp_factor),
                "shape": list(base_volume_shape),
                "file_name": fname,
                "path": str(fpath)
            })
            vol_id += 1

        # 3. Streak Noise (20 volumes)
        for i in range(20):
            vol = np.random.normal(loc=35.0, scale=10.0, size=(D, H, W)).astype(np.float32)
            num_streaks = 6 + i
            corrupted = inject_streak_noise(vol, num_streaks=num_streaks)

            fname = f"stress_vol_{vol_id:03d}_streak.npy"
            fpath = self.corpus_root / fname
            np.save(str(fpath), corrupted)

            manifest["volumes"].append({
                "id": vol_id,
                "category": "streak_noise",
                "severity": int(num_streaks),
                "shape": list(base_volume_shape),
                "file_name": fname,
                "path": str(fpath)
            })
            vol_id += 1

        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"50-Volume Stress Corpus successfully generated at {self.corpus_root}")
        return manifest

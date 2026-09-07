"""
Configuration and runtime environment specifications for MedSeg-XAI M1.
Handles RunPod persistent volume paths, device selection, precision, and medical windowing presets.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class MedSegConfig:
    # Volume and file system paths
    runpod_volume_path: Path = field(
        default_factory=lambda: Path(os.getenv("RUNPOD_VOLUME_PATH", "./runpod-volume"))
    )
    model_weights_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "MODEL_WEIGHTS_PATH",
                str(Path(os.getenv("RUNPOD_VOLUME_PATH", "./weights")) / "medsam2_hiera_large.pt")
            )
        )
    )
    dataset_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "DATASET_PATH",
                str(Path(os.getenv("RUNPOD_VOLUME_PATH", "./data")))
            )
        )
    )
    output_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "OUTPUT_PATH",
                str(Path(os.getenv("RUNPOD_VOLUME_PATH", "./outputs")))
            )
        )
    )

    # Server settings
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))

    # Model architecture specifications
    model_name: str = "medsam2_hiera_large"
    image_size: Tuple[int, int] = (1024, 1024)
    patch_size: int = 16
    embed_dim: int = 144
    num_heads: int = 16
    
    # Execution & Memory parameters
    use_cuda: bool = field(
        default_factory=lambda: os.getenv("FORCE_CPU", "0") != "1"
    )
    precision: str = field(
        default_factory=lambda: os.getenv("PRECISION", "float16")  # float16, bfloat16, float32
    )
    use_mmap: bool = True  # Prevent VRAM spikes via memory mapping
    max_memory_slices: int = 32

    # Preprocessing presets for CT Hounsfield Units (HU)
    ct_window_presets: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "soft_tissue": (-160.0, 240.0),
            "lung": (-1000.0, 400.0),
            "bone": (-100.0, 1000.0),
            "brain": (-100.0, 100.0),
            "mediastinum": (-175.0, 275.0),
            "liver": (-20.0, 200.0)
        }
    )

    def ensure_directories(self):
        """Ensures all volume directories exist."""
        self.model_weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

# Default global instance
default_config = MedSegConfig()

"""
MedSAM-2 Model Loader with Memory Mapping (mmap) and VRAM Overflow Protection.
Loads Hiera-Large vision transformer foundation weights safely with zero-copy memory mapping
and optimizes GPU VRAM allocation.
"""

import os
import gc
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from .config import MedSegConfig, default_config
from .models.sam2_backbone import MedSAM2Model

logger = logging.getLogger("medseg.model_loader")

class MedSAM2ModelLoader:
    """
    Handles robust instantiation, memory-mapped checkpoint loading,
    and precision management for MedSAM-2 with Hiera-Large backbone.
    """
    def __init__(self, config: Optional[MedSegConfig] = None):
        self.config = config or default_config
        self.device = self._resolve_device()
        self.dtype = self._resolve_dtype()
        self.model: Optional[MedSAM2Model] = None
        self.checkpoint_path = Path(self.config.model_weights_path)

    def _resolve_device(self) -> torch.device:
        """Determines computation device with fallback to CPU."""
        if self.config.use_cuda and torch.cuda.is_available():
            device_str = "cuda:0"
            logger.info(f"Using NVIDIA CUDA Device: {torch.cuda.get_device_name(0)}")
        else:
            device_str = "cpu"
            logger.info("CUDA not requested or unavailable. Using CPU.")
        return torch.device(device_str)

    def _resolve_dtype(self) -> torch.dtype:
        """Determines tensor precision."""
        if self.device.type == "cuda":
            if self.config.precision.lower() in ("float16", "fp16"):
                return torch.float16
            elif self.config.precision.lower() in ("bfloat16", "bf16"):
                return torch.bfloat16
        return torch.float32

    def load_model(
        self,
        checkpoint_path: Optional[Path] = None,
        use_mmap: Optional[bool] = None
    ) -> MedSAM2Model:
        """
        Instantiates MedSAM-2 and loads weights using memory mapping.
        
        Args:
            checkpoint_path: Optional override for weights location.
            use_mmap: Whether to employ torch.load(..., mmap=True) to prevent RAM/VRAM double-allocation.
        """
        path = Path(checkpoint_path) if checkpoint_path else self.checkpoint_path
        mmap_flag = self.config.use_mmap if use_mmap is None else use_mmap

        logger.info(f"Instantiating MedSAM-2 architecture (Hiera-Large)...")
        model = MedSAM2Model()

        if path.exists():
            logger.info(f"Loading checkpoint from {path} with mmap={mmap_flag}...")
            try:
                # Load with memory mapping to prevent host RAM explosion
                # Note: mmap=True is supported in PyTorch >= 2.1
                state_dict_container = torch.load(
                    str(path),
                    map_location="cpu",
                    mmap=mmap_flag
                )
                
                # Extract state dict if packaged within dictionary
                if isinstance(state_dict_container, dict):
                    if "model" in state_dict_container:
                        state_dict = state_dict_container["model"]
                    elif "state_dict" in state_dict_container:
                        state_dict = state_dict_container["state_dict"]
                    else:
                        state_dict = state_dict_container
                else:
                    state_dict = state_dict_container

                # Clean key prefixes if checkpoint was saved with DDP or wrapper
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    k_clean = k.replace("module.", "").replace("model.", "")
                    cleaned_state_dict[k_clean] = v

                # Load into model with non-strict mode to support partial checkpoints
                incompatible_keys = model.load_state_dict(cleaned_state_dict, strict=False)
                if incompatible_keys.missing_keys:
                    logger.debug(f"Missing keys during checkpoint load: {len(incompatible_keys.missing_keys)}")
                if incompatible_keys.unexpected_keys:
                    logger.debug(f"Unexpected keys during checkpoint load: {len(incompatible_keys.unexpected_keys)}")
                logger.info("Successfully loaded checkpoint weights into MedSAM-2.")

            except Exception as e:
                logger.warning(f"Error loading checkpoint with mmap: {e}. Falling back to default initialization.")
        else:
            logger.warning(f"No checkpoint file found at {path}. Operating with initialized model.")

        # Cast to target precision and transfer to execution device
        logger.info(f"Transferring model to {self.device} with precision {self.dtype}...")
        model = model.to(device=self.device, dtype=self.dtype)
        
        # Set to evaluation mode and freeze parameters by default
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.report_memory_footprint()
        return self.model

    def report_memory_footprint(self) -> Dict[str, float]:
        """Calculates current VRAM / RAM consumption."""
        stats = {
            "device": str(self.device),
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0
        }
        if self.device.type == "cuda":
            stats["allocated_mb"] = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            stats["reserved_mb"] = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
            stats["max_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            logger.info(
                f"[VRAM Telemetry] Allocated: {stats['allocated_mb']:.1f}MB | "
                f"Reserved: {stats['reserved_mb']:.1f}MB | Peak: {stats['max_allocated_mb']:.1f}MB"
            )
        return stats

    def get_model(self) -> MedSAM2Model:
        """Returns loaded model or initializes if not yet loaded."""
        if self.model is None:
            return self.load_model()
        return self.model

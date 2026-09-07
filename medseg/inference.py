"""
MedSAM-2 Zero-Shot Inference Engine with 3D Memory Attention Propagation
and Deep Layer Attention Extraction for XAI & Live MPRT.
Supports point prompts (foreground/background) and bounding box conditioning.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MedSegConfig, default_config
from .model_loader import MedSAM2ModelLoader
from .models.sam2_backbone import MedSAM2Model

logger = logging.getLogger("medseg.inference")

class PromptConditioning:
    """
    Encapsulates sparse prompt inputs (points and bounding boxes).
    """
    def __init__(
        self,
        points: Optional[Union[List[List[float]], torch.Tensor]] = None,
        labels: Optional[Union[List[int], torch.Tensor]] = None,
        boxes: Optional[Union[List[float], List[List[float]], torch.Tensor]] = None
    ):
        self.points = points
        self.labels = labels
        self.boxes = boxes

    def to_tensors(self, device: torch.device) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """Converts prompts to properly shaped PyTorch tensors."""
        point_tuple = None
        box_tensor = None

        if self.points is not None and self.labels is not None:
            if not isinstance(self.points, torch.Tensor):
                pts = torch.tensor(self.points, dtype=torch.float32, device=device)
            else:
                pts = self.points.to(device=device, dtype=torch.float32)

            if not isinstance(self.labels, torch.Tensor):
                lbls = torch.tensor(self.labels, dtype=torch.int64, device=device)
            else:
                lbls = self.labels.to(device=device, dtype=torch.int64)

            # Ensure batch dimension [B, N, 2] and [B, N]
            if pts.ndim == 2:
                pts = pts.unsqueeze(0)
            if lbls.ndim == 1:
                lbls = lbls.unsqueeze(0)
            point_tuple = (pts, lbls)

        if self.boxes is not None:
            if not isinstance(self.boxes, torch.Tensor):
                bx = torch.tensor(self.boxes, dtype=torch.float32, device=device)
            else:
                bx = self.boxes.to(device=device, dtype=torch.float32)

            # Ensure shape [B, M, 4]
            if bx.ndim == 1:
                bx = bx.unsqueeze(0).unsqueeze(0)
            elif bx.ndim == 2:
                bx = bx.unsqueeze(0)
            box_tensor = bx

        return point_tuple, box_tensor


class MedSAM2InferenceEngine:
    """
    Inference engine orchestrating zero-shot prompts, 3D memory bank attention propagation,
    and deep layer attention map extraction from Hiera-Large transformer blocks.
    """
    def __init__(
        self,
        model_loader: Optional[MedSAM2ModelLoader] = None,
        config: Optional[MedSegConfig] = None
    ):
        self.config = config or default_config
        self.loader = model_loader or MedSAM2ModelLoader(self.config)
        self.model: MedSAM2Model = self.loader.get_model()
        self.device = self.loader.device
        self.dtype = self.loader.dtype

        # 3D Memory bank maintaining temporal context across slices
        self.memory_bank: List[torch.Tensor] = []
        self.max_memory_size = self.config.max_memory_slices

        # Cache for extracted deep attention maps (M1 -> M2 handoff)
        self.deep_attention_cache: Dict[str, torch.Tensor] = {}

    def clear_memory(self):
        """Clears the 3D memory bank and cached attention tensors."""
        self.memory_bank.clear()
        self.deep_attention_cache.clear()

    def predict_slice(
        self,
        slice_tensor: torch.Tensor,
        prompts: Optional[PromptConditioning] = None,
        update_memory: bool = True,
        extract_attention: bool = True
    ) -> Dict[str, Any]:
        """
        Executes single 2D slice inference with prompt conditioning and memory querying.
        Args:
            slice_tensor: [1, C, H, W] or [C, H, W]
            prompts: PromptConditioning containing points and/or boxes
            update_memory: Whether to encode output mask into the 3D memory bank
            extract_attention: Whether to capture deep Hiera attention maps
        Returns:
            Dictionary containing binary masks, mask logits, IoU scores, and attention maps
        """
        # Ensure 4D tensor [1, C, H, W]
        if slice_tensor.ndim == 3:
            slice_tensor = slice_tensor.unsqueeze(0)
        slice_tensor = slice_tensor.to(device=self.device, dtype=self.dtype)

        # Parse prompts to tensors
        point_tuple, box_tensor = None, None
        if prompts is not None:
            point_tuple, box_tensor = prompts.to_tensors(self.device)

        # Forward pass through MedSAM-2
        with torch.no_grad():
            outputs = self.model.forward_slice(
                slice_tensor=slice_tensor,
                points=point_tuple,
                boxes=box_tensor,
                memory_bank=self.memory_bank
            )

        mask_logits = outputs["masks"]  # [1, num_masks, H_feat, W_feat]
        # Select best mask based on highest predicted IoU
        iou_preds = outputs["iou_predictions"]  # [1, num_masks]
        best_mask_idx = torch.argmax(iou_preds, dim=-1)[0].item()
        selected_logits = mask_logits[:, best_mask_idx:best_mask_idx+1, :, :]

        # Upscale logits to slice resolution
        H_orig, W_orig = slice_tensor.shape[-2:]
        high_res_logits = F.interpolate(
            selected_logits, size=(H_orig, W_orig), mode="bilinear", align_corners=False
        )
        binary_mask = (torch.sigmoid(high_res_logits) >= 0.5).to(torch.uint8)

        # Update 3D Memory Bank if enabled
        if update_memory:
            with torch.no_grad():
                mem_token = self.model.memory_encoder(
                    outputs["deep_features"], high_res_logits
                )
                self.memory_bank.append(mem_token)
                if len(self.memory_bank) > self.max_memory_size:
                    self.memory_bank.pop(0)

        # Capture deep attention maps
        if extract_attention and outputs.get("attention_maps"):
            self.deep_attention_cache = {
                k: v.detach().cpu() for k, v in outputs["attention_maps"].items()
            }

        return {
            "binary_mask": binary_mask.squeeze(0).squeeze(0).cpu().numpy(),
            "mask_logits": high_res_logits.squeeze(0).squeeze(0).cpu().numpy(),
            "iou_score": float(iou_preds[0, best_mask_idx].item()),
            "attention_maps": self.deep_attention_cache
        }

    def propagate_volume(
        self,
        video_tensor: torch.Tensor,
        prompt_slice_idx: int = 0,
        prompts: Optional[PromptConditioning] = None,
        bidirectional: bool = True
    ) -> Dict[str, Any]:
        """
        Propagates segmentation across the entire 3D pseudo-video batch (1, T, C, H, W).
        1. Conditions on prompt_slice_idx with provided prompts.
        2. Propagates forward in time through memory attention: prompt_slice_idx -> T-1.
        3. If bidirectional=True, propagates backward in time: prompt_slice_idx-1 -> 0.
        Returns:
            Dictionary containing:
                "mask_volume": 3D binary volume (T, H, W)
                "logits_volume": 3D float logits (T, H, W)
                "slice_ious": List of slice IoU scores
                "attention_summary": Deep attention maps from prompted slice
        """
        # video_tensor shape: [1, T, C, H, W] or [T, C, H, W]
        if video_tensor.ndim == 4:
            video_tensor = video_tensor.unsqueeze(0)

        B, T, C, H, W = video_tensor.shape
        logger.info(f"Initiating 3D memory propagation across {T} slices (Prompted key slice: {prompt_slice_idx})...")

        self.clear_memory()
        predicted_masks = [None] * T
        predicted_logits = [None] * T
        slice_ious = [0.0] * T

        # Step 1: Condition on key slice
        key_slice = video_tensor[:, prompt_slice_idx]
        res_key = self.predict_slice(
            slice_tensor=key_slice,
            prompts=prompts,
            update_memory=True,
            extract_attention=True
        )
        predicted_masks[prompt_slice_idx] = res_key["binary_mask"]
        predicted_logits[prompt_slice_idx] = res_key["mask_logits"]
        slice_ious[prompt_slice_idx] = res_key["iou_score"]
        key_attention = res_key["attention_maps"]

        # Step 2: Forward propagation (t = prompt_slice_idx + 1 ... T - 1)
        for t in range(prompt_slice_idx + 1, T):
            slice_t = video_tensor[:, t]
            res_t = self.predict_slice(
                slice_tensor=slice_t,
                prompts=None,  # Zero-shot memory propagation
                update_memory=True,
                extract_attention=False
            )
            predicted_masks[t] = res_t["binary_mask"]
            predicted_logits[t] = res_t["mask_logits"]
            slice_ious[t] = res_t["iou_score"]

        # Step 3: Backward propagation (t = prompt_slice_idx - 1 ... 0)
        if bidirectional and prompt_slice_idx > 0:
            logger.info(f"Executing reverse temporal propagation for slices {prompt_slice_idx - 1} down to 0...")
            # We can re-seed memory with the key slice or preserve current memory bank
            for t in range(prompt_slice_idx - 1, -1, -1):
                slice_t = video_tensor[:, t]
                res_t = self.predict_slice(
                    slice_tensor=slice_t,
                    prompts=None,
                    update_memory=True,
                    extract_attention=False
                )
                predicted_masks[t] = res_t["binary_mask"]
                predicted_logits[t] = res_t["mask_logits"]
                slice_ious[t] = res_t["iou_score"]

        import numpy as np
        mask_vol = np.stack(predicted_masks, axis=0)  # [T, H, W]
        logits_vol = np.stack(predicted_logits, axis=0)  # [T, H, W]

        logger.info(f"Propagation complete. Mean predicted slice confidence IoU: {sum(slice_ious) / T:.4f}")

        return {
            "mask_volume": mask_vol,
            "logits_volume": logits_vol,
            "slice_ious": slice_ious,
            "key_attention_maps": key_attention
        }

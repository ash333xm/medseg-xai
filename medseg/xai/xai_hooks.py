"""
PyTorch Non-Intrusive Forward Hook Activation & Attention Extraction Engine (Member 2).
Maps multi-head self-attention and cross-attention blocks across MedSAM-2 Hiera-Large backbone
and captures raw attention matrices for Transformer Multimodal Explainability (TMME) & Rollout.
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
import torch
import torch.nn as nn
from medseg.models.hiera_large import MultiHeadSelfAttention, HieraBlock
from medseg.models.sam2_backbone import MemoryAttention, MedSAM2Model

logger = logging.getLogger("medseg.xai_hooks")

def map_attention_blocks(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Inspects MedSAM-2 architecture and maps out all multi-head self-attention
    and cross-attention blocks eligible for PyTorch forward hook insertion.
    
    Returns:
        Dictionary mapping layer identifier strings to metadata dictionaries:
        {
            "layer_name": {
                "module": nn.Module,
                "type": "self_attention" | "cross_attention",
                "stage": int or str,
                "num_heads": int,
                "dim": int,
                "is_deep_layer": bool
            }
        }
    """
    mapped_layers = {}

    for name, module in model.named_modules():
        if isinstance(module, MultiHeadSelfAttention):
            # Parse stage and block index from module name (e.g. image_encoder.stages.2.35.attn)
            parts = name.split(".")
            stage_idx = None
            block_idx = None
            for i, p in enumerate(parts):
                if p == "stages" and i + 2 < len(parts):
                    try:
                        stage_idx = int(parts[i + 1]) + 1
                        block_idx = int(parts[i + 2])
                    except ValueError:
                        pass

            is_deep = False
            if stage_idx is not None:
                # Stage 3 (blocks >= 32) and Stage 4 are semantically rich deep layers
                if stage_idx == 3 and block_idx is not None and block_idx >= 32:
                    is_deep = True
                elif stage_idx >= 4:
                    is_deep = True

            mapped_layers[name] = {
                "module": module,
                "type": "self_attention",
                "stage": stage_idx,
                "block": block_idx,
                "num_heads": getattr(module, "num_heads", 8),
                "dim": getattr(module, "dim", 256),
                "is_deep_layer": is_deep
            }
        elif isinstance(module, MemoryAttention):
            mapped_layers[name] = {
                "module": module,
                "type": "cross_attention",
                "stage": "memory_bank",
                "block": "cross_attn",
                "num_heads": getattr(module, "num_heads", 8),
                "dim": getattr(module, "feat_dim", 256),
                "is_deep_layer": True
            }

    logger.info(f"Mapped {len(mapped_layers)} attention blocks ({sum(1 for v in mapped_layers.values() if v['is_deep_layer'])} deep layers).")
    return mapped_layers


class AttentionHookManager:
    """
    Manages non-intrusive PyTorch forward hooks across multi-head and cross-attention blocks.
    Captures raw attention weight tensors without breaking the autograd computation graph.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.mapped_blocks = map_attention_blocks(model)
        self.registered_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.captured_attentions: Dict[str, torch.Tensor] = {}
        self.hook_count: int = 0

    def _create_hook(self, layer_name: str, block_type: str) -> Callable:
        def hook_fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
            # For MultiHeadSelfAttention, check if last_attn_weights is cached
            if hasattr(module, "last_attn_weights") and module.last_attn_weights is not None:
                attn = module.last_attn_weights
            elif isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], torch.Tensor):
                attn = output[1]
            elif hasattr(module, "last_cross_attn_weights") and module.last_cross_attn_weights is not None:
                attn = module.last_cross_attn_weights
            else:
                attn = None

            if attn is not None:
                # Capture attention tensor without breaking graph
                self.captured_attentions[layer_name] = attn
                self.hook_count += 1

        return hook_fn

    def register_hooks(
        self,
        target_layers: Optional[List[str]] = None,
        deep_only: bool = False
    ) -> int:
        """
        Registers forward hooks on specified or all attention blocks.
        
        Args:
            target_layers: Optional explicit list of layer names to hook.
            deep_only: If True, only hook semantically rich deep attention blocks (Stages 3 & 4).
        Returns:
            Number of hooks successfully attached.
        """
        self.remove_hooks()
        self.captured_attentions.clear()

        for layer_name, meta in self.mapped_blocks.items():
            if target_layers and layer_name not in target_layers:
                continue
            if deep_only and not meta["is_deep_layer"]:
                continue

            module = meta["module"]
            hook_fn = self._create_hook(layer_name, meta["type"])
            handle = module.register_forward_hook(hook_fn)
            self.registered_handles.append(handle)

        logger.info(f"Registered {len(self.registered_handles)} attention forward hooks.")
        return len(self.registered_handles)

    def remove_hooks(self):
        """Detaches all active hooks cleanly."""
        for handle in self.registered_handles:
            handle.remove()
        self.registered_handles.clear()
        self.hook_count = 0

    def clear(self):
        """Clears captured attention matrices."""
        self.captured_attentions.clear()
        self.hook_count = 0

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Returns currently captured attention tensors."""
        return self.captured_attentions

    def get_attention_entropy(self) -> Dict[str, float]:
        """
        Calculates Shannon entropy across attention distributions as an interpretability metric:
            H(A) = - sum( A * log(A + eps) )
        """
        entropy_dict = {}
        for name, attn in self.captured_attentions.items():
            with torch.no_grad():
                probs = torch.clamp(attn.float(), min=1e-12, max=1.0)
                entropy = -(probs * torch.log(probs)).sum(dim=-1).mean().item()
                entropy_dict[name] = float(entropy)
        return entropy_dict

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

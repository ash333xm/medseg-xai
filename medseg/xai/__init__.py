"""
MedSeg-XAI: Explainability (XAI) Subsystem (Member 2).
Provides PyTorch forward hook insertion, attention map extraction,
TMME algorithm preparation, and clinical prompt generation from ground truth masks.
"""

from .xai_hooks import AttentionHookManager, map_attention_blocks
from .prompt_extractor import extract_bounding_boxes, extract_prompt_points, PromptExtractor

__all__ = [
    "AttentionHookManager",
    "map_attention_blocks",
    "extract_bounding_boxes",
    "extract_prompt_points",
    "PromptExtractor"
]

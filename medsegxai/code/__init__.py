"""
MedSeg-XAI Unified Core API
Exports the four agreed team functions:
- load_case(case_id) -> (image, true_mask, box)
- segment(image, box) -> pred_mask
- explain(image, box) -> heatmap
- audit(image, box) -> (similarity_scores, verdict)
- compute_dice(pred_mask, true_mask) -> float
"""

from .load_case import load_case
from .segment import segment, compute_dice
from .explain import explain
from .audit import audit

__all__ = [
    "load_case",
    "segment",
    "compute_dice",
    "explain",
    "audit",
]

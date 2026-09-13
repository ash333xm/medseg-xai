"""
MedSeg-XAI: Safety Auditing Subsystem (Member 3).
Provides Live MPRT auditing, mathematical evaluation metrics,
and the 50-volume edge-case stress corpus for robust clinical evaluation.
"""

from .stress_corpus import (
    StressCorpusCurator,
    inject_motion_artifacts,
    inject_low_contrast,
    inject_streak_noise
)
from .stress_tester import StressTester

__all__ = [
    "StressCorpusCurator",
    "inject_motion_artifacts",
    "inject_low_contrast",
    "inject_streak_noise",
    "StressTester"
]

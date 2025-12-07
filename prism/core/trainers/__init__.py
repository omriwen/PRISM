"""Trainers module for training algorithms.

This module provides trainers for different training strategies.
"""

from prism.core.trainers.base import (
    AbstractTrainer,
    EpochResult,
    MetricsCollector,
    TrainingConfig,
    TrainingResult,
)

__all__ = [
    "AbstractTrainer",
    "EpochResult",
    "MetricsCollector",
    "TrainingConfig",
    "TrainingResult",
]

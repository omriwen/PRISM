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
from prism.core.trainers.progressive import PRISMTrainer, create_scheduler

__all__ = [
    "AbstractTrainer",
    "EpochResult",
    "MetricsCollector",
    "TrainingConfig",
    "TrainingResult",
    "PRISMTrainer",
    "create_scheduler",
]

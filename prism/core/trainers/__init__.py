"""Trainers module for training algorithms.

This module provides trainers for different training strategies:
- PRISMTrainer: Progressive sample-by-sample training for PRISM algorithm
- EpochalTrainer: Epochal training for Mo-PIE algorithm
"""

from prism.core.trainers.base import (
    AbstractTrainer,
    EpochResult,
    MetricsCollector,
    TrainingConfig,
    TrainingResult,
)
from prism.core.trainers.epochal import EpochalTrainer, EpochalTrainerConfig
from prism.core.trainers.progressive import PRISMTrainer, create_scheduler

__all__ = [
    # Base classes
    "AbstractTrainer",
    "EpochResult",
    "MetricsCollector",
    "TrainingConfig",
    "TrainingResult",
    # Progressive trainer (PRISM)
    "PRISMTrainer",
    "create_scheduler",
    # Epochal trainer (Mo-PIE)
    "EpochalTrainer",
    "EpochalTrainerConfig",
]

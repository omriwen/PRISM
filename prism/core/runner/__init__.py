"""Runner module for experiment execution.

This module provides runners for different algorithms:
- PRISMRunner: Experiment runner for PRISM deep learning algorithm
- MoPIERunner: Experiment runner for Mo-PIE iterative phase retrieval
- RunnerFactory: Factory for creating runners based on algorithm type
"""

from prism.core.runner.base import AbstractRunner, ExperimentResult
from prism.core.runner.factory import RunnerFactory
from prism.core.runner.mixins import DataLoadingMixin, LineSamplingMixin, SetupMixin
from prism.core.runner.mopie_runner import MoPIERunner
from prism.core.runner.prism_runner import PRISMRunner

__all__ = [
    # Base classes
    "AbstractRunner",
    "ExperimentResult",
    # Factory
    "RunnerFactory",
    # Runners
    "PRISMRunner",
    "MoPIERunner",
    # Mixins
    "SetupMixin",
    "DataLoadingMixin",
    "LineSamplingMixin",
]

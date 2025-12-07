"""Runner module for experiment execution.

This module provides runners for different algorithms (PRISM, MoPIE).
"""

from prism.core.runner.base import AbstractRunner, ExperimentResult
from prism.core.runner.prism_runner import PRISMRunner

__all__ = ["AbstractRunner", "ExperimentResult", "PRISMRunner"]

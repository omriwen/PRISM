"""
Module: spids.visualization.plotters
Purpose: Main plotter classes for SPIDS visualizations

Description:
    Provides publication-quality plotters for various visualization needs:
    - TrainingVisualizer: Real-time training display
    - ReconstructionComparisonPlotter: Side-by-side comparisons with metrics
    - SyntheticAperturePlotter: K-space coverage visualization
    - LearningCurvesPlotter: Training metrics curves
"""

from __future__ import annotations

from prism.visualization.plotters.kspace import SyntheticAperturePlotter
from prism.visualization.plotters.metrics import LearningCurvesPlotter
from prism.visualization.plotters.reconstruction import ReconstructionComparisonPlotter
from prism.visualization.plotters.training import TrainingVisualizer


__all__ = [
    "TrainingVisualizer",
    "ReconstructionComparisonPlotter",
    "SyntheticAperturePlotter",
    "LearningCurvesPlotter",
]

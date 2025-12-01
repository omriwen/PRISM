"""
Module: spids.visualization.protocols
Purpose: Protocol definitions and TypedDicts for visualization system
Dependencies: typing, torch, matplotlib

Description:
    Defines structural typing protocols for plotters and TypedDicts for
    data passing between visualization components.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from torch import Tensor

    from prism.visualization.config import VisualizationConfig


@runtime_checkable
class PlotterProtocol(Protocol):
    """Interface for all plotters.

    Any class implementing this protocol can be used as a plotter
    in the visualization system.
    """

    config: VisualizationConfig

    def plot(self, **kwargs: Any) -> Figure:
        """Create the plot and return figure handle."""
        ...

    def save(self, path: str | Path, **kwargs: Any) -> None:
        """Save plot to file."""
        ...

    def cleanup(self) -> None:
        """Clean up matplotlib resources."""
        ...


class PlotData(TypedDict, total=False):
    """Data structure for plot inputs."""

    tensor: Tensor
    reconstruction: Tensor
    ground_truth: Tensor
    static_measurement: Tensor
    telescope: Any  # TelescopeProtocol
    telescope_agg: Any  # TelescopeAggregatorProtocol
    centers: list[list[float]]
    sample_idx: int
    radius: float
    crop_size: int
    ref_radius: float


class MetricsData(TypedDict, total=False):
    """Metrics for overlay display."""

    ssim: float
    psnr: float
    loss: float
    rmse: float
    sample_number: int
    total_samples: int


class SaveFigureOptions(TypedDict, total=False):
    """Options for saving figures."""

    dpi: int
    format: str
    transparent: bool
    bbox_inches: str
    pad_inches: float


class LearningCurvesData(TypedDict):
    """Data for learning curves plotting."""

    losses: list[float]
    ssims: list[float]
    psnrs: list[float]

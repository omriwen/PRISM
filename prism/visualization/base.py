"""
Module: spids.visualization.base
Purpose: Abstract base class for all SPIDS plotters
Dependencies: abc, gc, matplotlib, numpy

Description:
    Provides common functionality for all plotters including configuration
    management, figure lifecycle management via context managers, memory
    cleanup, and style application.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from prism.visualization.config import VisualizationConfig


if TYPE_CHECKING:
    from numpy.typing import NDArray


class BasePlotter(ABC):
    """Abstract base class for all SPIDS plotters.

    Provides common functionality:
    - Configuration management
    - Figure lifecycle management (context manager)
    - Memory cleanup
    - Style application

    Parameters
    ----------
    config : VisualizationConfig, optional
        Visualization configuration. If None, uses defaults.

    Examples
    --------
    >>> with ReconstructionPlotter(config) as plotter:
    ...     fig = plotter.plot(reconstruction=rec, ground_truth=gt)
    ...     plotter.save("output.png")

    Notes
    -----
    Subclasses must implement:
    - plot(**kwargs) -> Figure
    - _create_figure() -> tuple[Figure, NDArray]
    """

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        """Initialize plotter with configuration.

        Parameters
        ----------
        config : VisualizationConfig, optional
            Visualization configuration. If None, uses defaults.
        """
        self.config = config or VisualizationConfig()
        self._fig: Figure | None = None
        self._axes: NDArray[np.object_] | None = None

    def __enter__(self) -> Self:
        """Enter context manager - apply styles."""
        self.config.style.apply()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager - cleanup."""
        self.cleanup()

    @abstractmethod
    def plot(self, **kwargs: Any) -> Figure:
        """Create the plot. Must be implemented by subclasses.

        Returns
        -------
        Figure
            Matplotlib figure handle
        """
        ...

    @abstractmethod
    def _create_figure(self) -> tuple[Figure, NDArray[np.object_]]:
        """Create figure and axes. Override in subclasses.

        Returns
        -------
        tuple[Figure, NDArray]
            Figure and flattened axes array
        """
        ...

    def save(
        self,
        path: str | Path,
        dpi: int | None = None,
        bbox_inches: str = "tight",
        **kwargs: Any,
    ) -> None:
        """Save current figure to file.

        Parameters
        ----------
        path : str or Path
            Output file path
        dpi : int, optional
            Override DPI for saving. If None, uses config.figure.dpi
        bbox_inches : str
            Bounding box setting (default: 'tight')
        **kwargs
            Additional arguments passed to savefig

        Raises
        ------
        RuntimeError
            If no figure exists (plot() not called)
        """
        if self._fig is None:
            raise RuntimeError("No figure to save. Call plot() first.")
        save_dpi = dpi or self.config.figure.dpi
        self._fig.savefig(path, dpi=save_dpi, bbox_inches=bbox_inches, **kwargs)

    def cleanup(self) -> None:
        """Clean up matplotlib resources and memory."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None
        if self.config.memory_cleanup:
            gc.collect()

    @contextmanager
    def figure_context(self) -> Iterator[tuple[Figure, NDArray[np.object_]]]:
        """Context manager for figure lifecycle.

        Yields
        ------
        tuple[Figure, NDArray]
            Figure and axes array

        Examples
        --------
        >>> with plotter.figure_context() as (fig, axes):
        ...     axes[0].plot(x, y)
        """
        fig, axes = self._create_figure()
        self._fig = fig
        self._axes = axes
        try:
            yield fig, axes
        finally:
            pass  # Cleanup handled by __exit__ or explicit cleanup()

    @property
    def figure(self) -> Figure | None:
        """Get current figure handle."""
        return self._fig

    @property
    def axes(self) -> NDArray[np.object_] | None:
        """Get current axes array."""
        return self._axes

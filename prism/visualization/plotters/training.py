"""
Module: spids.visualization.plotters.training
Purpose: Real-time training visualization
Dependencies: matplotlib, torch, numpy

Description:
    Provides real-time visualization during SPIDS training showing:
    - K-space with accumulated aperture mask
    - Current measurement
    - Current reconstruction
    - Ground truth target
    - Target through accumulated mask
    - Target with ideal aperture
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch import Tensor

from prism.visualization.base import BasePlotter
from prism.visualization.config import VisualizationConfig
from prism.visualization.helpers import (
    compute_kspace_display,
    create_aperture_overlay,
    create_roi_circle,
    ensure_4d_tensor,
    prepare_tensor_for_display,
)


if TYPE_CHECKING:
    pass


class TrainingVisualizer(BasePlotter):
    """Real-time training visualization with live updates.

    Replaces the legacy plot_meas_agg function with a stateful,
    configurable class that supports live updates during training.

    Parameters
    ----------
    config : VisualizationConfig, optional
        Visualization configuration
    update_interval : float
        Minimum seconds between display updates (default: 0.1)

    Examples
    --------
    >>> with TrainingVisualizer(INTERACTIVE) as viz:
    ...     for sample in samples:
    ...         # ... training code ...
    ...         fig, axes = viz.update(
    ...             tensor=image_gt,
    ...             telescope=telescope_agg,
    ...             reconstruction=model(),
    ...             target=ground_truth,
    ...             centers=centers,
    ...             ref_radius=roi_diameter / 2,
    ...         )
    """

    def __init__(
        self,
        config: VisualizationConfig | None = None,
        update_interval: float = 0.1,
    ) -> None:
        """Initialize training visualizer.

        Parameters
        ----------
        config : VisualizationConfig, optional
            Visualization configuration
        update_interval : float
            Minimum seconds between display updates
        """
        super().__init__(config)
        self.update_interval = update_interval
        self._last_update: float = 0.0

    def _create_figure(self) -> tuple[Figure, NDArray[np.object_]]:
        """Create 2x3 subplot layout."""
        fig, axes = plt.subplots(
            2,
            3,
            figsize=self.config.figure.figsize,
        )
        return fig, np.asarray(axes)

    def plot(  # type: ignore[override]
        self,
        tensor: Tensor,
        telescope: Any,
        reconstruction: Tensor,
        target: Tensor,
        centers: list[list[float]] | None = None,
        sample_idx: int | None = None,
        radius: float | None = None,
        crop_size: int | None = None,
        ref_radius: float | None = None,
    ) -> Figure:
        """Create training visualization figure.

        Parameters
        ----------
        tensor : Tensor
            Input object tensor
        telescope : TelescopeProtocol
            Telescope aggregator with cum_mask
        reconstruction : Tensor
            Current reconstruction from model
        target : Tensor
            Ground truth target image
        centers : list[list[float]], optional
            Sample center positions [[y, x], ...]
        sample_idx : int, optional
            Current sample index to highlight
        radius : float, optional
            Aperture radius (defaults to telescope.r)
        crop_size : int, optional
            Crop size for display
        ref_radius : float, optional
            Reference radius for ROI circle

        Returns
        -------
        Figure
            Matplotlib figure handle
        """
        if centers is None:
            centers = [[0, 0]]
        if radius is None:
            radius = float(telescope.r.item())
        if sample_idx is None:
            sample_idx = len(centers) // 2

        fig, axes = self._create_figure()
        self._fig = fig
        self._axes = axes

        # Row 0: New measurement, Current reconstruction, Target through mask
        self._plot_measurement(axes[0, 0], tensor, telescope, centers, radius)
        self._plot_reconstruction(axes[0, 1], reconstruction, crop_size)
        self._plot_target_masked(axes[0, 2], target, telescope, centers, radius, crop_size)

        # Row 1: K-space, Target, Target best
        self._plot_kspace(axes[1, 0], tensor, telescope, centers, radius, ref_radius)
        self._plot_target(axes[1, 1], target, crop_size)
        self._plot_target_best(axes[1, 2], target, telescope, centers, radius, crop_size)

        if self.config.figure.tight_layout:
            plt.tight_layout()

        return fig

    def update(self, **kwargs: Any) -> tuple[Figure | None, NDArray[np.object_] | None]:
        """Update live display with new data.

        Throttled by update_interval to prevent display lag.

        Returns
        -------
        tuple[Figure | None, NDArray | None]
            Figure and axes handles for continued use, or (None, None) if throttled
        """
        now = time.time()
        if now - self._last_update < self.update_interval:
            return self._fig, self._axes

        self._last_update = now

        # Clear and replot if figure exists
        if self._fig is not None and self._axes is not None:
            for ax in self._axes.flatten():
                ax.cla()

        fig = self.plot(**kwargs)
        plt.draw()
        plt.pause(0.01)

        return fig, self._axes

    def _plot_kspace(
        self,
        ax: Axes,
        tensor: Tensor,
        telescope: Any,
        centers: list[list[float]],
        radius: float,
        ref_radius: float | None,
    ) -> None:
        """Plot k-space with aperture overlay."""
        nx = int(telescope.n.item())

        # Compute k-space display
        kspace = compute_kspace_display(tensor, nx)
        ax.imshow(kspace, cmap=self.config.style.colormap)

        # Add accumulated mask overlay
        if self.config.kspace.show_aperture_mask and hasattr(telescope, "cum_mask"):
            overlay = create_aperture_overlay(
                telescope.cum_mask,
                self.config.kspace.mask_color,
            )
            ax.imshow(overlay)

        # Add current sample positions overlay
        if self.config.kspace.show_sample_centers:
            centers_map = (
                torch.stack([telescope.mask(center, radius) for center in centers], dim=0)
                .sum(dim=0)
                .squeeze()
                > 0
            )
            centers_overlay = create_aperture_overlay(
                centers_map,
                (1.0, 0.0, 0.0, 0.5),  # Red for current samples
            )
            ax.imshow(centers_overlay)

        # Add ROI circle
        if ref_radius is not None:
            roi = create_roi_circle(
                (nx // 2, nx // 2),
                ref_radius,
                color=self.config.kspace.roi_color,
                linestyle=self.config.kspace.roi_line_style,
                linewidth=self.config.kspace.roi_line_width,
            )
            ax.add_patch(roi)
            ax.set_xlim((nx // 2 - ref_radius, nx // 2 + ref_radius))
            ax.set_ylim((nx // 2 - ref_radius, nx // 2 + ref_radius))

        ax.set_title("K-space")
        ax.axis("off")

    def _plot_measurement(
        self,
        ax: Axes,
        tensor: Tensor,
        telescope: Any,
        centers: list[list[float]],
        radius: float,
    ) -> None:
        """Plot current measurement."""
        tensor_4d = ensure_4d_tensor(tensor)
        measurement = telescope(tensor_4d, centers, radius, add_noise=True)
        display = prepare_tensor_for_display(measurement[:, 0, :, :])

        if display.ndim == 3:
            display = display[min(1, display.shape[0] - 1)]

        ax.imshow(display, cmap=self.config.style.colormap)
        ax.set_title("New Measurement")
        ax.axis("off")

    def _plot_reconstruction(
        self,
        ax: Axes,
        reconstruction: Tensor,
        crop_size: int | None,
    ) -> None:
        """Plot current reconstruction."""
        display = prepare_tensor_for_display(reconstruction, crop_size)
        ax.imshow(display, cmap=self.config.style.colormap)
        ax.set_title("Current Reconstruction")
        ax.axis("off")

    def _plot_target(self, ax: Axes, target: Tensor, crop_size: int | None) -> None:
        """Plot target image."""
        display = prepare_tensor_for_display(target, crop_size)
        ax.imshow(display, cmap=self.config.style.colormap)
        ax.set_title("Target")
        ax.axis("off")

    def _plot_target_masked(
        self,
        ax: Axes,
        target: Tensor,
        telescope: Any,
        centers: list[list[float]],
        radius: float,
        crop_size: int | None,
    ) -> None:
        """Plot target through accumulated mask."""
        centers_map = (
            torch.stack([telescope.mask(center, radius) for center in centers], dim=0)
            .sum(dim=0)
            .squeeze()
            > 0
        )
        masked = telescope.measure_through_accumulated_mask(target, mask_to_add=centers_map)
        display = prepare_tensor_for_display(masked, crop_size)
        ax.imshow(display, cmap=self.config.style.colormap)
        ax.set_title("Target through Mask")
        ax.axis("off")

    def _plot_target_best(
        self,
        ax: Axes,
        target: Tensor,
        telescope: Any,
        centers: list[list[float]],
        radius: float,
        crop_size: int | None,
    ) -> None:
        """Plot target with maximum aperture."""
        r_max = max(torch.tensor(c).pow(2).sum().sqrt().item() for c in centers) + radius

        target_4d = ensure_4d_tensor(target)
        best = telescope(target_4d, [[0, 0]], r_max)
        display = prepare_tensor_for_display(best[:, 0, :, :], crop_size)

        if display.ndim == 3:
            display = display[min(1, display.shape[0] - 1)]

        ax.imshow(display, cmap=self.config.style.colormap)
        ax.set_title("Target Best")
        ax.axis("off")

"""
Module: spids.visualization.plotters.kspace
Purpose: K-space and synthetic aperture visualization
Dependencies: matplotlib, torch, numpy

Description:
    Provides visualization for k-space coverage, synthetic aperture
    demonstration, and aperture mask overlays.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch import Tensor

from prism.visualization.base import BasePlotter
from prism.visualization.helpers import (
    compute_kspace_display,
    create_aperture_overlay,
    create_roi_circle,
)


class SyntheticAperturePlotter(BasePlotter):
    """K-space coverage and synthetic aperture visualization.

    Shows accumulated k-space coverage with ROI overlay.

    Parameters
    ----------
    config : VisualizationConfig, optional
        Visualization configuration

    Examples
    --------
    >>> with SyntheticAperturePlotter(PUBLICATION) as plotter:
    ...     fig = plotter.plot(
    ...         tensor=image,
    ...         telescope_agg=telescope,
    ...         roi_diameter=200.0,
    ...     )
    ...     plotter.save("aperture.pdf")
    """

    def _create_figure(self) -> tuple[Figure, NDArray[np.object_]]:
        """Create single subplot."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        return fig, np.array([ax])

    def plot(  # type: ignore[override]
        self,
        tensor: Tensor,
        telescope_agg: Any,
        roi_diameter: float,
    ) -> Figure:
        """Create synthetic aperture demonstration figure.

        Parameters
        ----------
        tensor : Tensor
            Input object tensor
        telescope_agg : TelescopeAggregatorProtocol
            Telescope aggregator with accumulated mask
        roi_diameter : float
            Region of interest diameter

        Returns
        -------
        Figure
            Matplotlib figure handle
        """
        fig, axes = self._create_figure()
        self._fig = fig
        self._axes = axes
        ax = axes[0]

        nx = int(telescope_agg.n.item())
        roi_radius = roi_diameter / 2

        # K-space display
        kspace = compute_kspace_display(tensor, nx)
        ax.imshow(kspace, cmap=self.config.style.colormap)

        # Accumulated mask overlay
        if self.config.kspace.show_aperture_mask and hasattr(telescope_agg, "cum_mask"):
            overlay = create_aperture_overlay(
                telescope_agg.cum_mask,
                self.config.kspace.mask_color,
            )
            ax.imshow(overlay)

        # ROI circle
        roi = create_roi_circle(
            (nx // 2, nx // 2),
            roi_radius,
            color=self.config.kspace.roi_color,
            linestyle=self.config.kspace.roi_line_style,
            linewidth=self.config.kspace.roi_line_width,
            label="ROI",
        )
        ax.add_patch(roi)

        ax.set_title("Synthetic Aperture Demonstration")
        ax.set_xlim((nx // 2 - roi_radius, nx // 2 + roi_radius))
        ax.set_ylim((nx // 2 - roi_radius, nx // 2 + roi_radius))
        ax.legend(loc="upper right")
        ax.axis("off")

        if self.config.figure.tight_layout:
            plt.tight_layout()

        return fig

    def plot_coverage_progression(
        self,
        tensor: Tensor,
        masks: list[Tensor],
        roi_diameter: float,
        *,
        ncols: int = 4,
    ) -> Figure:
        """Create multi-panel coverage progression figure.

        Parameters
        ----------
        tensor : Tensor
            Input object tensor
        masks : list[Tensor]
            List of accumulated masks at different stages
        roi_diameter : float
            Region of interest diameter
        ncols : int
            Number of columns in subplot grid

        Returns
        -------
        Figure
            Matplotlib figure handle with progression panels
        """
        n_masks = len(masks)
        nrows = (n_masks + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        self._fig = fig
        self._axes = np.asarray(axes).flatten() if n_masks > 1 else np.array([axes])

        # Get k-space size from first mask
        nx = masks[0].shape[-1]
        roi_radius = roi_diameter / 2

        # Base k-space display
        kspace = compute_kspace_display(tensor, nx)

        for i, mask in enumerate(masks):
            ax = self._axes[i] if n_masks > 1 else self._axes[0]

            # K-space background
            ax.imshow(kspace, cmap=self.config.style.colormap)

            # Mask overlay
            overlay = create_aperture_overlay(mask, self.config.kspace.mask_color)
            ax.imshow(overlay)

            # ROI circle
            roi = create_roi_circle(
                (nx // 2, nx // 2),
                roi_radius,
                color=self.config.kspace.roi_color,
                linestyle=self.config.kspace.roi_line_style,
                linewidth=self.config.kspace.roi_line_width,
            )
            ax.add_patch(roi)

            # Compute coverage percentage
            coverage = (mask > 0).float().mean().item() * 100
            ax.set_title(f"Step {i + 1}: {coverage:.1f}% coverage")
            ax.set_xlim((nx // 2 - roi_radius, nx // 2 + roi_radius))
            ax.set_ylim((nx // 2 - roi_radius, nx // 2 + roi_radius))
            ax.axis("off")

        # Hide unused axes
        for i in range(n_masks, len(self._axes)):
            self._axes[i].axis("off")

        if self.config.figure.tight_layout:
            plt.tight_layout()

        return fig

"""
Module: spids.visualization.plotters.reconstruction
Purpose: Publication-quality reconstruction comparison figures
Dependencies: matplotlib, torch, numpy

Description:
    Provides publication-quality reconstruction comparison figures
    with side-by-side comparisons and metrics overlays.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch import Tensor

from prism.visualization.base import BasePlotter
from prism.visualization.helpers import prepare_tensor_for_display


class ReconstructionComparisonPlotter(BasePlotter):
    """Publication-quality reconstruction comparison figure.

    Creates side-by-side comparison of:
    - Ground truth
    - Final reconstruction
    - Static measurement (baseline)

    With SSIM/PSNR metrics overlay.

    Parameters
    ----------
    config : VisualizationConfig, optional
        Visualization configuration

    Examples
    --------
    >>> with ReconstructionComparisonPlotter(PUBLICATION) as plotter:
    ...     fig = plotter.plot(
    ...         ground_truth=gt,
    ...         reconstruction=rec,
    ...         static_measurement=meas,
    ...         obj_size=256,
    ...     )
    ...     plotter.save("results.pdf")
    """

    def _create_figure(self) -> tuple[Figure, NDArray[np.object_]]:
        """Create 1x3 subplot layout."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        return fig, np.asarray(axes)

    def plot(  # type: ignore[override]
        self,
        ground_truth: Tensor,
        reconstruction: Tensor,
        static_measurement: Tensor,
        obj_size: int,
    ) -> Figure:
        """Create comparison figure with metrics.

        Parameters
        ----------
        ground_truth : Tensor
            Ground truth image
        reconstruction : Tensor
            Final reconstruction from model
        static_measurement : Tensor
            Static measurement for baseline comparison
        obj_size : int
            Object size for cropping and metrics computation

        Returns
        -------
        Figure
            Matplotlib figure handle
        """
        from prism.utils.metrics import compute_ssim, psnr

        fig, axes = self._create_figure()
        self._fig = fig
        self._axes = axes

        # Compute metrics
        ssim_rec = compute_ssim(reconstruction, ground_truth, size=obj_size)
        psnr_rec = psnr(reconstruction, ground_truth, size=obj_size)
        ssim_meas = compute_ssim(static_measurement, ground_truth, size=obj_size)
        psnr_meas = psnr(static_measurement, ground_truth, size=obj_size)

        # Ground truth
        gt_display = prepare_tensor_for_display(ground_truth, obj_size)
        axes[0].imshow(gt_display, cmap=self.config.style.colormap)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        # Reconstruction with metrics
        rec_display = prepare_tensor_for_display(reconstruction, obj_size)
        axes[1].imshow(rec_display, cmap=self.config.style.colormap)
        axes[1].set_title(f"Final Reconstruction\nSSIM: {ssim_rec:.3f}, PSNR: {psnr_rec:.1f} dB")
        axes[1].axis("off")

        # Static measurement with metrics
        meas_display = prepare_tensor_for_display(static_measurement, obj_size)
        axes[2].imshow(meas_display, cmap=self.config.style.colormap)
        axes[2].set_title(f"Static Detector (0,0)\nSSIM: {ssim_meas:.3f}, PSNR: {psnr_meas:.1f} dB")
        axes[2].axis("off")

        if self.config.figure.tight_layout:
            plt.tight_layout()

        return fig

    def plot_with_difference(
        self,
        ground_truth: Tensor,
        reconstruction: Tensor,
        obj_size: int,
        *,
        show_colorbar: bool = True,
    ) -> Figure:
        """Create comparison with difference map.

        Parameters
        ----------
        ground_truth : Tensor
            Ground truth image
        reconstruction : Tensor
            Final reconstruction from model
        obj_size : int
            Object size for cropping and metrics computation
        show_colorbar : bool
            Whether to show colorbar on difference map

        Returns
        -------
        Figure
            Matplotlib figure handle
        """
        from prism.utils.metrics import compute_ssim, psnr

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self._fig = fig
        self._axes = np.asarray(axes)

        # Compute metrics
        ssim_val = compute_ssim(reconstruction, ground_truth, size=obj_size)
        psnr_val = psnr(reconstruction, ground_truth, size=obj_size)

        # Prepare displays
        gt_display = prepare_tensor_for_display(ground_truth, obj_size)
        rec_display = prepare_tensor_for_display(reconstruction, obj_size)

        # Ground truth
        axes[0].imshow(gt_display, cmap=self.config.style.colormap)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        # Reconstruction
        axes[1].imshow(rec_display, cmap=self.config.style.colormap)
        axes[1].set_title(f"Reconstruction\nSSIM: {ssim_val:.4f}, PSNR: {psnr_val:.1f} dB")
        axes[1].axis("off")

        # Difference map
        diff = np.abs(gt_display - rec_display)
        im = axes[2].imshow(diff, cmap="hot", vmin=0, vmax=diff.max())
        axes[2].set_title("Absolute Difference")
        axes[2].axis("off")

        if show_colorbar and self.config.show_colorbars:
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        if self.config.figure.tight_layout:
            plt.tight_layout()

        return fig

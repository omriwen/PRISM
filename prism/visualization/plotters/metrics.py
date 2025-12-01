"""
Module: spids.visualization.plotters.metrics
Purpose: Training metrics and learning curves visualization
Dependencies: matplotlib, numpy

Description:
    Provides visualization for training metrics including loss curves,
    SSIM progression, and PSNR tracking.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from prism.visualization.base import BasePlotter


class LearningCurvesPlotter(BasePlotter):
    """Training metrics visualization (loss, SSIM, PSNR).

    Creates three subplots showing training progression.

    Parameters
    ----------
    config : VisualizationConfig, optional
        Visualization configuration

    Examples
    --------
    >>> with LearningCurvesPlotter(PUBLICATION) as plotter:
    ...     fig = plotter.plot(
    ...         losses=[0.5, 0.3, 0.1],
    ...         ssims=[0.5, 0.7, 0.9],
    ...         psnrs=[20, 25, 30],
    ...     )
    ...     plotter.save("curves.pdf")
    """

    def _create_figure(self) -> tuple[Figure, NDArray[np.object_]]:
        """Create 1x3 subplot layout."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        return fig, np.asarray(axes)

    def plot(  # type: ignore[override]
        self,
        losses: Sequence[float],
        ssims: Sequence[float],
        psnrs: Sequence[float],
    ) -> Figure:
        """Create learning curves figure.

        Parameters
        ----------
        losses : Sequence[float]
            Loss values per sample
        ssims : Sequence[float]
            SSIM values per sample
        psnrs : Sequence[float]
            PSNR values per sample

        Returns
        -------
        Figure
            Matplotlib figure handle
        """
        fig, axes = self._create_figure()
        self._fig = fig
        self._axes = axes

        if len(losses) == 0:
            # Handle empty data
            for ax in axes:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return fig

        sample_numbers = range(1, len(losses) + 1)
        lw = self.config.style.line_width

        # Loss curve (log scale)
        axes[0].plot(sample_numbers, losses, "b-", linewidth=lw)
        axes[0].set_xlabel("Sample Number")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(True, alpha=self.config.style.grid_alpha)
        if len(losses) > 0 and min(losses) > 0:
            axes[0].set_yscale("log")

        # SSIM curve
        axes[1].plot(sample_numbers, ssims, "g-", linewidth=lw)
        axes[1].set_xlabel("Sample Number")
        axes[1].set_ylabel("SSIM")
        axes[1].set_title("Structural Similarity Index")
        axes[1].grid(True, alpha=self.config.style.grid_alpha)
        axes[1].set_ylim((0, 1))

        # PSNR curve
        axes[2].plot(sample_numbers, psnrs, "r-", linewidth=lw)
        axes[2].set_xlabel("Sample Number")
        axes[2].set_ylabel("PSNR [dB]")
        axes[2].set_title("Peak Signal-to-Noise Ratio")
        axes[2].grid(True, alpha=self.config.style.grid_alpha)

        if self.config.figure.tight_layout:
            plt.tight_layout()

        return fig

    def plot_combined(
        self,
        losses: Sequence[float],
        ssims: Sequence[float],
        psnrs: Sequence[float],
        *,
        show_convergence_markers: bool = True,
    ) -> Figure:
        """Create combined metrics figure with dual y-axes.

        Parameters
        ----------
        losses : Sequence[float]
            Loss values per sample
        ssims : Sequence[float]
            SSIM values per sample
        psnrs : Sequence[float]
            PSNR values per sample
        show_convergence_markers : bool
            Whether to show markers at convergence points

        Returns
        -------
        Figure
            Matplotlib figure handle
        """
        fig, ax1 = plt.subplots(figsize=self.config.figure.figsize)
        self._fig = fig
        self._axes = np.array([ax1])

        if len(losses) == 0:
            ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)
            return fig

        sample_numbers = list(range(1, len(losses) + 1))
        lw = self.config.style.line_width

        # Primary axis: Loss
        color1 = "tab:blue"
        ax1.set_xlabel("Sample Number")
        ax1.set_ylabel("Loss", color=color1)
        line1 = ax1.plot(sample_numbers, losses, color=color1, linewidth=lw, label="Loss")
        ax1.tick_params(axis="y", labelcolor=color1)
        if len(losses) > 0 and min(losses) > 0:
            ax1.set_yscale("log")
        ax1.grid(True, alpha=self.config.style.grid_alpha)

        # Secondary axis: SSIM
        ax2 = ax1.twinx()
        color2 = "tab:green"
        ax2.set_ylabel("SSIM", color=color2)
        line2 = ax2.plot(sample_numbers, ssims, color=color2, linewidth=lw, label="SSIM")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim((0, 1))

        # Third axis: PSNR (offset)
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        color3 = "tab:red"
        ax3.set_ylabel("PSNR [dB]", color=color3)
        line3 = ax3.plot(sample_numbers, psnrs, color=color3, linewidth=lw, label="PSNR")
        ax3.tick_params(axis="y", labelcolor=color3)

        # Add convergence markers
        if show_convergence_markers and len(ssims) > 1:
            # Find best SSIM point
            best_idx = int(np.argmax(ssims))
            ax2.axvline(x=best_idx + 1, color="gray", linestyle="--", alpha=0.5)
            ax2.annotate(
                f"Best: {ssims[best_idx]:.4f}",
                xy=(best_idx + 1, ssims[best_idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # Combined legend
        lines = line1 + line2 + line3
        labels: list[str] = [str(line.get_label()) for line in lines]
        ax1.legend(lines, labels, loc="upper right")

        ax1.set_title("Training Progress")

        if self.config.figure.tight_layout:
            fig.tight_layout()

        return fig

    def plot_per_sample_statistics(
        self,
        losses_per_sample: list[list[float]],
        *,
        show_variance: bool = True,
    ) -> Figure:
        """Plot per-sample training statistics.

        Parameters
        ----------
        losses_per_sample : list[list[float]]
            List of loss histories, one per sample
        show_variance : bool
            Whether to show variance bands

        Returns
        -------
        Figure
            Matplotlib figure handle
        """
        fig, ax = plt.subplots(figsize=self.config.figure.figsize)
        self._fig = fig
        self._axes = np.array([ax])

        if len(losses_per_sample) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return fig

        # Find max epoch count
        max_epochs = max(len(losses) for losses in losses_per_sample)
        lw = self.config.style.line_width * 0.5  # Thinner for individual lines

        # Plot each sample's loss curve
        for i, losses in enumerate(losses_per_sample):
            alpha = 0.3 if len(losses_per_sample) > 10 else 0.7
            ax.plot(range(1, len(losses) + 1), losses, alpha=alpha, linewidth=lw)

        # Compute and plot mean + variance
        if show_variance:
            # Pad shorter sequences
            padded = np.zeros((len(losses_per_sample), max_epochs))
            padded[:] = np.nan
            for i, losses in enumerate(losses_per_sample):
                padded[i, : len(losses)] = losses

            mean_loss = np.nanmean(padded, axis=0)
            std_loss = np.nanstd(padded, axis=0)

            epochs = np.arange(1, max_epochs + 1)
            ax.plot(epochs, mean_loss, "k-", linewidth=self.config.style.line_width, label="Mean")
            ax.fill_between(
                epochs,
                mean_loss - std_loss,
                mean_loss + std_loss,
                alpha=0.2,
                color="black",
                label="±1 std",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Per-Sample Training Curves (n={len(losses_per_sample)})")
        ax.grid(True, alpha=self.config.style.grid_alpha)
        if show_variance:
            ax.legend()

        # Use log scale if appropriate
        all_losses = [loss for losses in losses_per_sample for loss in losses]
        if len(all_losses) > 0 and min(all_losses) > 0:
            ax.set_yscale("log")

        if self.config.figure.tight_layout:
            plt.tight_layout()

        return fig

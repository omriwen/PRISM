"""Visualization helpers for pattern plotting."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from torch import Tensor


def plot_sample_positions(
    ax: Axes,
    sample_centers: Tensor,
    roi_diameter: float,
    title: str,
) -> None:
    """Plot sample positions on an axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    sample_centers : Tensor
        Sample center coordinates
    roi_diameter : float
        ROI diameter in pixels
    title : str
        Plot title
    """
    # Get points
    if sample_centers.shape[1] == 2:
        # Line sampling
        lines = sample_centers.cpu().numpy()
        for line in lines:
            ax.plot(line[:, 0], line[:, 1], "b-", alpha=0.5, linewidth=1)
        centers = lines.mean(axis=1)
        ax.scatter(centers[:, 0], centers[:, 1], c="red", s=20, alpha=0.7)
    else:
        # Point sampling
        points = sample_centers[:, 0, :].cpu().numpy()
        ax.scatter(points[:, 0], points[:, 1], c="blue", s=50, alpha=0.6)

    # Draw ROI circle
    circle = Circle((0, 0), roi_diameter / 2, fill=False, color="red", linestyle="--", linewidth=2)
    ax.add_patch(circle)

    # Format
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12, fontweight="bold")
    lim = roi_diameter / 2 * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def plot_coverage_heatmap(
    ax: Axes,
    sample_centers: Tensor,
    roi_diameter: float,
    title: str,
) -> None:
    """Plot coverage heatmap on an axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    sample_centers : Tensor
        Sample center coordinates
    roi_diameter : float
        ROI diameter in pixels
    title : str
        Plot title
    """
    # Get centers
    if sample_centers.shape[1] == 2:
        centers = sample_centers.mean(dim=1)
    else:
        centers = sample_centers[:, 0, :]

    # Create grid
    grid_size = 100
    lim = roi_diameter / 2 * 1.1
    x = np.linspace(-lim, lim, grid_size)
    y = np.linspace(-lim, lim, grid_size)
    xx, yy = np.meshgrid(x, y)

    # Compute coverage
    coverage_map = np.zeros_like(xx)
    centers_np = centers.cpu().numpy()
    for cx, cy in centers_np:
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        coverage_map += 1.0 / (dist + 10)

    # Plot
    ax.imshow(
        coverage_map,
        extent=(-lim, lim, -lim, lim),
        origin="lower",
        cmap="hot",
        interpolation="bilinear",
    )

    # Draw ROI circle
    circle = Circle((0, 0), roi_diameter / 2, fill=False, color="cyan", linestyle="--", linewidth=2)
    ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.set_title(f"{title} - Coverage", fontsize=12, fontweight="bold")

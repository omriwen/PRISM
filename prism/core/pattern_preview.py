"""
Pattern visualization and verification tools.

Provides utilities for previewing sampling patterns before running
expensive experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Circle


matplotlib.use("Agg")  # Non-interactive backend


def compute_pattern_statistics(
    sample_centers: torch.Tensor,
    roi_diameter: float,
) -> Dict[str, Any]:
    """
    Compute statistical properties of sampling pattern.

    Args:
        sample_centers: Pattern positions (n_samples, n_points, 2)
        roi_diameter: ROI diameter for coverage calculation

    Returns:
        Dictionary with statistics:
        - n_samples: Number of sampling positions
        - is_line_sampling: Whether using line sampling
        - radial_mean: Mean distance from center
        - radial_std: Std of distance from center
        - radial_min: Minimum distance from center
        - radial_max: Maximum distance from center
        - coverage_percentage: Percentage of ROI covered
        - inter_sample_distances: Statistics on distances between samples
    """
    # Get center points
    if sample_centers.shape[1] == 2:
        # Line sampling - use midpoints
        centers = sample_centers.mean(dim=1)
        is_line = True
    else:
        centers = sample_centers[:, 0, :]
        is_line = False

    # Compute radial distances
    radii = torch.sqrt((centers**2).sum(dim=-1))

    # Compute inter-sample distances
    n = centers.shape[0]
    dists = []
    for i in range(min(n, 100)):  # Sample for efficiency
        others = torch.cat([centers[:i], centers[i + 1 :]])
        if len(others) > 0:
            d = torch.sqrt(((centers[i : i + 1] - others) ** 2).sum(dim=-1))
            dists.append(d.min().item())

    # Estimate coverage (rough approximation)
    roi_area = np.pi * (roi_diameter / 2) ** 2
    sample_area = n * np.pi * (np.mean(dists) / 2) ** 2 if dists else 0
    coverage = min(100, (sample_area / roi_area) * 100)

    return {
        "n_samples": n,
        "is_line_sampling": is_line,
        "radial_mean": radii.mean().item(),
        "radial_std": radii.std().item(),
        "radial_min": radii.min().item(),
        "radial_max": radii.max().item(),
        "coverage_percentage": coverage,
        "inter_sample_dist_mean": np.mean(dists) if dists else 0,
        "inter_sample_dist_std": np.std(dists) if dists else 0,
    }


def visualize_pattern(
    sample_centers: torch.Tensor,
    roi_diameter: float,
    save_path: Optional[Path] = None,
    show_statistics: bool = True,
) -> Figure:
    """
    Create comprehensive visualization of sampling pattern.

    Args:
        sample_centers: Pattern positions (n_samples, n_points, 2)
        roi_diameter: ROI diameter
        save_path: Optional path to save figure
        show_statistics: Whether to include statistics panel

    Returns:
        Matplotlib figure
    """
    # Compute statistics
    stats = compute_pattern_statistics(sample_centers, roi_diameter)

    # Create figure with subplots
    if show_statistics:
        fig, axes_arr = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes_arr.flatten()
    else:
        fig, axes_arr = plt.subplots(1, 3, figsize=(15, 5))
        axes = list(axes_arr)

    # Subplot 1: Sample positions
    ax = axes[0]
    _plot_sample_positions(ax, sample_centers, roi_diameter)

    # Subplot 2: Coverage heatmap
    ax = axes[1]
    _plot_coverage_heatmap(ax, sample_centers, roi_diameter)

    # Subplot 3: Radial distribution
    ax = axes[2]
    _plot_radial_distribution(ax, sample_centers, roi_diameter, stats)

    # Subplot 4: Statistics (if enabled)
    if show_statistics:
        ax = axes[3]
        _plot_statistics_text(ax, stats)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _plot_sample_positions(ax: Any, sample_centers: torch.Tensor, roi_diameter: float) -> None:
    """Plot sample positions with ROI circle."""
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
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")
    ax.set_title("Sample Positions")
    lim = roi_diameter / 2 * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def _plot_coverage_heatmap(ax: Any, sample_centers: torch.Tensor, roi_diameter: float) -> None:
    """Plot coverage heatmap."""
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

    # Compute coverage (inverse distance weighting)
    coverage_map = np.zeros_like(xx)
    centers_np = centers.cpu().numpy()
    for cx, cy in centers_np:
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        coverage_map += 1.0 / (dist + 10)  # Avoid division by zero

    # Plot
    im = ax.imshow(
        coverage_map,
        extent=[-lim, lim, -lim, lim],
        origin="lower",
        cmap="hot",
        interpolation="bilinear",
    )

    # Draw ROI circle
    circle = Circle((0, 0), roi_diameter / 2, fill=False, color="cyan", linestyle="--", linewidth=2)
    ax.add_patch(circle)

    plt.colorbar(im, ax=ax, label="Coverage Density")
    ax.set_aspect("equal")
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")
    ax.set_title("K-Space Coverage Heatmap")


def _plot_radial_distribution(
    ax: Any, sample_centers: torch.Tensor, roi_diameter: float, stats: Dict[str, Any]
) -> None:
    """Plot radial distribution histogram."""
    # Get centers
    if sample_centers.shape[1] == 2:
        centers = sample_centers.mean(dim=1)
    else:
        centers = sample_centers[:, 0, :]

    # Compute radii
    radii = torch.sqrt((centers**2).sum(dim=-1)).cpu().numpy()

    # Plot histogram
    ax.hist(radii, bins=30, alpha=0.7, color="blue", edgecolor="black")

    # Add vertical line for mean
    ax.axvline(
        stats["radial_mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {stats['radial_mean']:.1f}",
    )

    # Add ROI boundary
    ax.axvline(
        roi_diameter / 2,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"ROI: {roi_diameter / 2:.1f}",
    )

    ax.set_xlabel("Distance from Center (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Radial Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_statistics_text(ax: Any, stats: Dict[str, Any]) -> None:
    """Plot statistics as text."""
    ax.axis("off")

    text = f"""
Pattern Statistics
{"=" * 40}

Samples: {stats["n_samples"]}
Sampling Type: {"Line" if stats["is_line_sampling"] else "Point"}

Radial Statistics:
  Mean: {stats["radial_mean"]:.2f} pixels
  Std:  {stats["radial_std"]:.2f} pixels
  Min:  {stats["radial_min"]:.2f} pixels
  Max:  {stats["radial_max"]:.2f} pixels

Coverage: {stats["coverage_percentage"]:.1f}%

Inter-Sample Distance:
  Mean: {stats["inter_sample_dist_mean"]:.2f} pixels
  Std:  {stats["inter_sample_dist_std"]:.2f} pixels
"""

    ax.text(
        0.1,
        0.5,
        text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        transform=ax.transAxes,
    )


def preview_pattern(
    pattern_spec: str,
    config: Any,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Preview a pattern from specification.

    Args:
        pattern_spec: Pattern specification (builtin:name or file path)
        config: Configuration object
        save_path: Optional path to save visualization

    Returns:
        Dictionary with:
        - 'sample_centers': Generated pattern tensor
        - 'statistics': Pattern statistics
        - 'metadata': Pattern metadata (source, hash, etc.)
        - 'figure': Matplotlib figure (if not saved and closed)
    """
    from prism.core.pattern_loader import load_and_generate_pattern

    # Generate pattern
    sample_centers, metadata = load_and_generate_pattern(pattern_spec, config)

    # Compute statistics
    stats = compute_pattern_statistics(sample_centers, config.roi_diameter)

    # Create visualization
    fig = visualize_pattern(
        sample_centers,
        config.roi_diameter,
        save_path=save_path,
        show_statistics=True,
    )

    return {
        "sample_centers": sample_centers,
        "statistics": stats,
        "metadata": metadata,
        "figure": fig if save_path is None else None,
    }

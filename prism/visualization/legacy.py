"""
Module: spids.visualization.legacy
Purpose: Backward-compatible wrappers for legacy visualization functions

Description:
    Provides compatibility wrappers for code that still uses the old
    visualization API. These functions wrap the new class-based plotters
    to maintain backward compatibility during migration.

Note:
    These functions are deprecated and will be removed in a future version.
    Please migrate to the new class-based plotters.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from prism.utils.image import crop_image, crop_pad
from prism.utils.transforms import fft


def plot_meas_agg(
    tensor: Any,
    telescope: Any,
    reconstruction: Any,
    target: Any,
    centers: Any = None,
    sample: Any = None,
    radius: Any = None,
    sum_pattern: Any = None,
    crop_size: Any = None,
    ref_radius: Any = None,
    fig: Any = None,
    ref_radius_1: Any = None,
) -> tuple[Figure, np.ndarray]:
    """
    Real-time aggregated training visualization.

    This is a backward-compatible wrapper that maintains the old function
    signature. For new code, use TrainingVisualizer instead.

    Parameters
    ----------
    tensor : Tensor
        Input object tensor
    telescope : TelescopeAggregatorProtocol
        Telescope aggregator with cum_mask
    reconstruction : Tensor
        Current reconstruction from model
    target : Tensor
        Ground truth target image
    centers : list, optional
        Sample center positions [[y, x], ...]
    sample : int, optional
        Current sample index (unused)
    radius : float, optional
        Aperture radius (defaults to telescope.r)
    sum_pattern : Any, optional
        Sum pattern (unused in new implementation)
    crop_size : int, optional
        Crop size for display
    ref_radius : float, optional
        Reference radius for ROI circle
    fig : tuple, optional
        Existing (figure, axes) tuple to reuse
    ref_radius_1 : float, optional
        Secondary reference radius (unused)

    Returns
    -------
    tuple[Figure, np.ndarray]
        Figure and axes handles for continued use
    """
    cpu = torch.device("cpu")
    if centers is None:
        centers = [[0, 0]]
    if sample is None:
        sample = len(centers) // 2
    if radius is None:
        radius = telescope.r

    # Create or reuse figure
    if fig is None:
        fig_obj, axs = plt.subplots(2, 3, figsize=(8, 5))
    else:
        fig_obj, axs = fig

    # Clear existing axes
    for ax in axs.flatten():
        ax.cla()

    # K-space plot (bottom left)
    log_spec = fft(crop_pad(tensor, telescope.n.item())).abs().log10().squeeze().to(cpu)
    axs[1][0].imshow(log_spec, cmap="gray")

    nx = len(telescope.x.flatten().to(cpu).numpy())

    # Accumulated mask overlay (green)
    axs[1][0].imshow(
        np.stack(
            [
                np.zeros((nx, nx)),
                telescope.cum_mask.to(cpu).numpy(),
                np.zeros((nx, nx)),
                telescope.cum_mask.to(cpu).numpy(),
            ],
            axis=2,
        )
    )

    # Current centers overlay (red)
    centers_map = (
        torch.stack([telescope.mask(center, radius) for center in centers], dim=0)
        .sum(dim=0)
        .squeeze()
        > 0
    )
    r_max = torch.stack([center.pow(2).sum().sqrt() for center in centers]).max() + radius
    axs[1][0].imshow(
        np.stack(
            [
                centers_map.to(cpu).numpy(),
                np.zeros((nx, nx)),
                np.zeros((nx, nx)),
                centers_map.to(cpu).numpy(),
            ],
            axis=2,
        )
    )

    # ROI circle
    if ref_radius is not None:
        axs[1][0].add_patch(
            plt.Circle(
                (nx // 2, nx // 2), ref_radius, color="r", fill=False, linewidth=5, linestyle="--"
            )
        )

    # Ensure tensor is 4D [B, C, H, W]
    tensor_4d = (
        tensor.unsqueeze(0).unsqueeze(0)
        if tensor.ndim == 2
        else tensor.unsqueeze(0)
        if tensor.ndim == 3
        else tensor
    )

    # New measurement (top left)
    axs[0][0].imshow(
        crop_image(telescope(tensor_4d, centers, radius, add_noise=True), crop_size)
        .abs()
        .squeeze(1)[1]
        .to(cpu)
        .numpy(),
        cmap="gray",
    )

    # Current reconstruction (top middle)
    axs[0][1].imshow(
        crop_image(reconstruction, crop_size).abs().squeeze().to(cpu).numpy(), cmap="gray"
    )

    # Target (bottom middle)
    axs[1][1].imshow(crop_image(target, crop_size).squeeze().abs().to(cpu).numpy(), cmap="gray")

    # Target through mask (top right)
    axs[0][2].imshow(
        crop_image(
            telescope.measure_through_accumulated_mask(target, mask_to_add=centers_map), crop_size
        )
        .abs()
        .squeeze()
        .to(cpu)
        .numpy(),
        cmap="gray",
    )

    # Ensure target is 4D [B, C, H, W]
    target_4d = (
        target.unsqueeze(0).unsqueeze(0)
        if target.ndim == 2
        else target.unsqueeze(0)
        if target.ndim == 3
        else target
    )

    # Target best (bottom right)
    axs[1][2].imshow(
        crop_image(telescope(target_4d, [[0, 0]], r_max), crop_size)
        .abs()
        .squeeze(1)[1]
        .to(cpu)
        .numpy(),
        cmap="gray",
    )

    # Set titles
    axs[1][0].set_title("K-space")
    axs[0][0].set_title("New Measurement")
    axs[0][1].set_title("Current reconstruction")
    axs[1][1].set_title("Target")
    axs[0][2].set_title("Target through mask")
    axs[1][2].set_title("Target best")

    # Set k-space view limits
    if ref_radius is not None:
        axs[1][0].set_xlim([nx // 2 - ref_radius, nx // 2 + ref_radius])
        axs[1][0].set_ylim([nx // 2 - ref_radius, nx // 2 + ref_radius])

    plt.tight_layout()
    plt.show()
    plt.draw()
    plt.pause(0.1)
    plt.show()

    return fig_obj, axs

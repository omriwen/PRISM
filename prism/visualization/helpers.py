"""
Module: spids.visualization.helpers
Purpose: Helper functions for visualization data preparation
Dependencies: numpy, torch, matplotlib

Description:
    Provides utility functions for preparing tensors for display,
    computing k-space representations, creating overlay masks,
    and other common visualization tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from numpy.typing import NDArray
from torch import Tensor


if TYPE_CHECKING:
    from prism.visualization.config import MetricsOverlayConfig
    from prism.visualization.protocols import MetricsData


def prepare_tensor_for_display(
    tensor: Tensor,
    crop_size: int | None = None,
    normalize: bool = True,
    log_scale: bool = False,
    take_abs: bool = True,
    percentile_clip: tuple[float, float] | None = None,
) -> NDArray[np.floating]:
    """Prepare tensor for matplotlib display.

    Parameters
    ----------
    tensor : Tensor
        Input tensor [B, C, H, W], [C, H, W], or [H, W]
    crop_size : int, optional
        Size to crop to center
    normalize : bool
        Normalize to [0, 1] range
    log_scale : bool
        Apply log10 transform (for k-space display)
    take_abs : bool
        Take absolute value (for complex tensors)
    percentile_clip : tuple[float, float], optional
        Clip to percentile range (low, high), e.g., (0.5, 99.5)

    Returns
    -------
    NDArray[np.floating]
        2D array ready for imshow
    """
    from prism.utils.image import crop_image

    # Detach and move to CPU
    arr = tensor.detach().cpu()

    # Crop if requested
    if crop_size is not None:
        arr = crop_image(arr, crop_size)

    # Take absolute value for complex
    if take_abs:
        arr = arr.abs()

    # Apply log scale
    if log_scale:
        arr = arr.add(1e-10).log10()

    # Squeeze to 2D
    arr = arr.squeeze()

    # Convert to numpy
    result: NDArray[np.floating] = arr.numpy()

    # Percentile clipping for dynamic range
    if percentile_clip is not None:
        vmin = np.percentile(result, percentile_clip[0])
        vmax = np.percentile(result, percentile_clip[1])
        result = np.clip(result, vmin, vmax)

    # Normalize
    if normalize:
        vmin_arr, vmax_arr = result.min(), result.max()
        if vmax_arr > vmin_arr:
            result = (result - vmin_arr) / (vmax_arr - vmin_arr)

    return result


def compute_kspace_display(
    tensor: Tensor,
    target_size: int | None = None,
    normalize: bool = True,
) -> NDArray[np.floating]:
    """Compute k-space representation for display.

    Parameters
    ----------
    tensor : Tensor
        Input image tensor
    target_size : int, optional
        Target size for k-space computation (pad/crop)
    normalize : bool
        Whether to normalize output

    Returns
    -------
    NDArray[np.floating]
        Log-scaled k-space magnitude for display
    """
    from prism.utils.image import crop_pad
    from prism.utils.transforms import fft

    if target_size is not None:
        tensor = crop_pad(tensor, target_size)

    kspace = fft(tensor)
    return prepare_tensor_for_display(
        kspace,
        log_scale=True,
        normalize=normalize,
        percentile_clip=(0.5, 99.5),
    )


def create_aperture_overlay(
    mask: Tensor,
    color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5),
) -> NDArray[np.floating]:
    """Create RGBA overlay for aperture visualization.

    Parameters
    ----------
    mask : Tensor
        Binary mask tensor [H, W]
    color : tuple
        RGBA color for the overlay (R, G, B, A)

    Returns
    -------
    NDArray[np.floating]
        RGBA array [H, W, 4] suitable for imshow
    """
    mask_np = mask.detach().cpu().numpy().squeeze()
    h, w = mask_np.shape

    overlay: NDArray[np.floating] = np.zeros((h, w, 4), dtype=np.float32)
    overlay[:, :, 0] = color[0]  # R
    overlay[:, :, 1] = color[1]  # G
    overlay[:, :, 2] = color[2]  # B
    overlay[:, :, 3] = mask_np.astype(np.float32) * color[3]  # A

    return overlay


def ensure_4d_tensor(tensor: Tensor) -> Tensor:
    """Ensure tensor is 4D [B, C, H, W].

    Parameters
    ----------
    tensor : Tensor
        Input tensor of any dimension (2D, 3D, or 4D)

    Returns
    -------
    Tensor
        4D tensor [B, C, H, W]
    """
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        return tensor.unsqueeze(0)
    return tensor


def add_metrics_text(
    ax: Axes,
    metrics: MetricsData,
    config: MetricsOverlayConfig,
) -> None:
    """Add metrics text overlay to axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to add text to
    metrics : MetricsData
        Metrics to display
    config : MetricsOverlayConfig
        Overlay configuration
    """
    lines: list[str] = []

    if config.show_ssim and "ssim" in metrics:
        lines.append(f"SSIM: {metrics['ssim']:.{config.decimal_places_ssim}f}")
    if config.show_psnr and "psnr" in metrics:
        lines.append(f"PSNR: {metrics['psnr']:.{config.decimal_places_psnr}f} dB")
    if config.show_loss and "loss" in metrics:
        lines.append(f"Loss: {metrics['loss']:.6f}")

    if not lines:
        return

    text = "\n".join(lines)

    # Determine position
    pos_map: dict[str, tuple[float, float, str, str]] = {
        "top-left": (0.02, 0.98, "left", "top"),
        "top-right": (0.98, 0.98, "right", "top"),
        "bottom-left": (0.02, 0.02, "left", "bottom"),
        "bottom-right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = pos_map[config.position]

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=config.font_size,
        fontfamily="monospace",
        bbox={
            "boxstyle": config.box_style,
            "facecolor": "white",
            "alpha": config.background_alpha,
            "edgecolor": "gray",
        },
    )


def create_roi_circle(
    center: tuple[float, float],
    radius: float,
    color: str = "red",
    linestyle: str = "--",
    linewidth: float = 3.0,
    fill: bool = False,
    label: str | None = None,
) -> Circle:
    """Create circle patch for ROI visualization.

    Parameters
    ----------
    center : tuple[float, float]
        (x, y) center coordinates
    radius : float
        Circle radius in pixels
    color : str
        Circle color
    linestyle : str
        Line style ('--', '-', ':', '-.')
    linewidth : float
        Line width
    fill : bool
        Whether to fill the circle
    label : str, optional
        Label for legend

    Returns
    -------
    Circle
        Matplotlib Circle patch
    """
    return Circle(
        center,
        radius,
        color=color,
        fill=fill,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )


def get_cpu_device() -> torch.device:
    """Get CPU device for tensor operations."""
    return torch.device("cpu")

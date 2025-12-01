"""
Module: spids.utils.metrics
Purpose: Image quality metrics for reconstruction evaluation
Dependencies: torch, skimage, numpy
Main Functions:
    - compute_ssim(img1, img2, size): Structural Similarity Index metric
    - compute_rmse(img1, img2, size): Normalized Root Mean Square Error
    - psnr(img1, img2, size, data_range): Peak Signal-to-Noise Ratio in dB

Description:
    This module provides quality metrics (SSIM, RMSE, PSNR) for evaluating
    reconstruction quality in SPIDS experiments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from torch import Tensor

# Import crop_image from image module
from .image import crop_image


def compute_ssim(img1: Tensor, img2: Tensor, size: Optional[int] = None) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    Args:
        img1: First image tensor
        img2: Second image tensor
        size: Optional size to crop both images before comparison

    Returns:
        SSIM value (higher is better, max 1.0)
    """
    if size is not None:
        img1 = crop_image(img1, size)
        img2 = crop_image(img2, size)
    # Convert the tensors to numpy arrays
    img1_np = img1.squeeze().detach().cpu().numpy()
    img2_np = img2.squeeze().detach().cpu().numpy()

    # Calculate the SSIM using skimage
    ssim_value = compare_ssim(
        img1_np,
        img2_np,
        data_range=1,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        multichannel=False,
    )

    return float(ssim_value)


def compute_rmse(img1: Tensor, img2: Tensor, size: Optional[int] = None) -> float:
    """
    Calculate normalized Root Mean Square Error (RMSE) between two images.

    Args:
        img1: First image tensor
        img2: Second image tensor (reference)
        size: Optional size to crop both images before comparison

    Returns:
        Normalized RMSE as percentage
    """
    if size is not None:
        img1 = crop_image(img1, size)
        img2 = crop_image(img2, size)
    # Convert the tensors to numpy arrays
    img1_np = img1.squeeze().detach().cpu().numpy()
    img2_np = img2.squeeze().detach().cpu().numpy()

    # Calculate the RMSE
    rmse = ((img1_np - img2_np) ** 2).mean() ** 0.5
    norm = (img2_np**2).mean() ** 0.5

    return float(rmse / norm * 100)


def psnr(img1: Tensor, img2: Tensor, size: Optional[int] = None, data_range: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1: First image tensor
        img2: Second image tensor (reference)
        size: Optional size to crop both images
        data_range: Data range of the images (default: 1.0)

    Returns:
        PSNR value in dB
    """
    if size is not None:
        img1 = crop_image(img1, size)
        img2 = crop_image(img2, size)

    # Convert to numpy arrays
    img1_np = img1.squeeze().detach().cpu().numpy()
    img2_np = img2.squeeze().detach().cpu().numpy()

    # Calculate MSE
    mse = ((img1_np - img2_np) ** 2).mean()

    # Avoid division by zero
    if mse == 0:
        return float("inf")

    # Calculate PSNR
    psnr_value = 20 * np.log10(data_range / np.sqrt(mse))

    return float(psnr_value)

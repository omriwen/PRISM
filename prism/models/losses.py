"""
Module: losses.py
Purpose: Loss functions for progressive SPIDS training
Dependencies: torch
Main Classes:
    - LossAggregator: **CRITICAL** - Aggregated loss for progressive training
        Combines loss from old measurements (accumulated mask) with new measurement
        Returns (loss_old, loss_new) for independent convergence checking

Architecture Details:
    LossAggregator (Progressive Loss):
    - Computes loss on both old (accumulated) and new measurements
    - Normalized by zero-loss to make threshold-based stopping robust
    - Supports L1 or L2 loss
    - Returns separate old/new losses for dual convergence criteria
    - Used with MeasurementSystem to generate measurements through cumulative mask

Usage Pattern:
    from prism.models.losses import LossAggregator
    from prism.core.instruments import Telescope, TelescopeConfig
    from prism.core.measurement_system import MeasurementSystem

    # Initialization
    criterion = LossAggregator(loss_type="l1")
    config = TelescopeConfig(n_pixels=1024, aperture_radius_pixels=50)
    telescope = Telescope(config)
    measurement_system = MeasurementSystem(telescope)

    # Progressive training loop
    for sample_center in sample_centers:
        # Generate measurement through telescope
        measurement = measurement_system.measure(image, reconstruction, sample_center)

        # Forward pass (decoder-only model)
        output = model()

        # Compute dual loss (old mask + new mask)
        loss_old, loss_new = criterion(output, measurement, measurement_system, sample_center)

        # Check convergence on both losses
        if loss_old < threshold and loss_new < threshold:
            break  # Both losses converged

        # Optimize
        loss = loss_old + loss_new
        loss.backward()
        optimizer.step()

        # Add new mask to accumulator
        measurement_system.add_measurement(sample_center)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ============================================================================
# SSIM Helper Functions
# ============================================================================


def _create_gaussian_window(window_size: int, sigma: float, channels: int = 1) -> Tensor:
    """
    Create a Gaussian window for SSIM computation.

    This function generates a 2D Gaussian kernel used for computing local statistics
    in SSIM calculation. The kernel is normalized and matches scikit-image's
    implementation for consistency with evaluation metrics.

    Args:
        window_size: Size of the window (should be odd, typically 11)
        sigma: Standard deviation of Gaussian (typically 1.5)
        channels: Number of channels (1 for grayscale)

    Returns:
        Gaussian window tensor of shape [channels, 1, window_size, window_size]
        Normalized so that all elements sum to 1.0

    Example:
        >>> window = _create_gaussian_window(11, 1.5, channels=1)
        >>> window.shape
        torch.Size([1, 1, 11, 11])
        >>> window.sum()
        tensor(1.0000)

    Notes:
        - Uses separable Gaussian (1D outer product) for efficiency
        - Window is expanded for multi-channel support (groups convolution)
        - Matches scikit-image's Gaussian weighting parameter
    """
    # Create 1D Gaussian kernel
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()  # Normalize

    # Create 2D Gaussian kernel by outer product
    g_2d = g.unsqueeze(1) * g.unsqueeze(0)  # [window_size, window_size]

    # Reshape for conv2d: [channels, 1, H, W]
    window = g_2d.unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, window_size, window_size)

    return window.contiguous()


def _gaussian_filter(img: Tensor, window: Tensor) -> Tensor:
    """
    Apply Gaussian filter to image using conv2d.

    This function applies a Gaussian weighted filter to compute local statistics
    for SSIM. Uses grouped convolution to handle multi-channel images efficiently.
    Uses reflection padding to match scikit-image's default behavior.

    Args:
        img: Input tensor [B, C, H, W]
        window: Gaussian window [C, 1, K, K] where K is window size

    Returns:
        Filtered tensor [B, C, H, W] with same spatial dimensions

    Example:
        >>> img = torch.rand(1, 1, 128, 128)
        >>> window = _create_gaussian_window(11, 1.5, channels=1)
        >>> filtered = _gaussian_filter(img, window)
        >>> filtered.shape
        torch.Size([1, 1, 128, 128])

    Notes:
        - Uses reflection padding to match scikit-image behavior
        - groups=channels ensures separate filtering per channel
        - Window should be on same device as img
    """
    channels = img.shape[1]
    padding = window.shape[-1] // 2

    # Apply reflection padding to match scikit-image
    # Padding format: (left, right, top, bottom)
    img_padded = F.pad(img, (padding, padding, padding, padding), mode="reflect")

    # Apply convolution without additional padding (already padded)
    # groups=channels → separate conv for each channel
    return F.conv2d(img_padded, window, padding=0, groups=channels)


def _ssim_single(
    img1: Tensor,
    img2: Tensor,
    window: Tensor,
    data_range: float = 1.0,
    use_sample_covariance: bool = False,
) -> Tensor:
    """
    Compute SSIM between two images (single scale).

    Computes the Structural Similarity Index using Gaussian-weighted local
    statistics. This implementation matches scikit-image's SSIM for consistency
    with evaluation metrics.

    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        window: Gaussian window [C, 1, K, K]
        data_range: Expected range of image values (default: 1.0 for [0,1] images)
        use_sample_covariance: Use sample (N-1) or population (N) covariance
                              (default: False to match scikit-image)

    Returns:
        SSIM scalar tensor (mean over all spatial locations and channels)

    Example:
        >>> img1 = torch.rand(1, 1, 128, 128)
        >>> img2 = img1 + 0.1 * torch.randn(1, 1, 128, 128)
        >>> window = _create_gaussian_window(11, 1.5, channels=1)
        >>> window = window.to(img1.device).type_as(img1)
        >>> ssim_val = _ssim_single(img1, img2, window)
        >>> print(f"SSIM: {ssim_val:.4f}")

    Notes:
        - Constants C1, C2 match scikit-image specification
        - Returns mean SSIM (scalar) for compatibility with loss computation
        - Differentiable for use in gradient-based optimization
        - SSIM formula: (2*μ1*μ2 + C1)(2*σ12 + C2) / ((μ1² + μ2² + C1)(σ1² + σ2² + C2))
    """
    # Constants (matching scikit-image)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # Compute local means
    mu1 = _gaussian_filter(img1, window)
    mu2 = _gaussian_filter(img2, window)

    # Compute local variances and covariance
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _gaussian_filter(img1**2, window) - mu1_sq
    sigma2_sq = _gaussian_filter(img2**2, window) - mu2_sq
    sigma12 = _gaussian_filter(img1 * img2, window) - mu1_mu2

    # Apply sample correction if requested (scikit-image uses population by default)
    # Note: We keep this parameter for API compatibility but don't implement it
    # as scikit-image uses population covariance (N denominator) by default
    if use_sample_covariance:
        # This would require knowing the effective window size
        # For now, we match scikit-image's default (population)
        pass

    # Compute SSIM components
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / denominator

    # Return mean SSIM (scalar)
    return ssim_map.mean()


def _compute_ssim(
    img1: Tensor,
    img2: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> Tensor:
    """
    High-level SSIM computation matching compute_ssim parameters.

    This is a convenience wrapper that creates the Gaussian window and calls
    _ssim_single(). It matches the parameter interface of the evaluation
    metric compute_ssim() for consistency.

    Args:
        img1: First image [B, C, H, W] or [C, H, W]
        img2: Second image [B, C, H, W] or [C, H, W]
        window_size: Gaussian window size (default: 11)
        sigma: Gaussian sigma (default: 1.5)
        data_range: Image value range (default: 1.0)

    Returns:
        SSIM scalar tensor

    Example:
        >>> img1 = torch.rand(1, 1, 256, 256)
        >>> img2 = torch.rand(1, 1, 256, 256)
        >>> ssim_val = _compute_ssim(img1, img2)
        >>> print(f"SSIM: {ssim_val:.4f}")
        >>>
        >>> # Also works with 3D tensors
        >>> img1_3d = torch.rand(1, 128, 128)
        >>> img2_3d = torch.rand(1, 128, 128)
        >>> ssim_val = _compute_ssim(img1_3d, img2_3d)

    Notes:
        - Automatically handles 3D inputs by adding batch dimension
        - Creates fresh Gaussian window each call (LossAggregator caches it)
        - Parameters match scikit-image's structural_similarity defaults
        - Differentiable for use as loss function
        - AMP-safe: disables autocast and uses float32 for numerical stability
    """
    # AMP-safe SSIM computation: disable autocast and use float32
    # SSIM requires float32 for numerical stability in convolution operations
    with torch.cuda.amp.autocast(enabled=False):
        # Ensure 4D tensors
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)

        # Convert to float32 for SSIM computation
        img1_fp32 = img1.float()
        img2_fp32 = img2.float()

        channels = img1_fp32.shape[1]

        # Create Gaussian window (always float32 for SSIM)
        window = _create_gaussian_window(window_size, sigma, channels)
        window = window.to(img1_fp32.device, dtype=torch.float32)

        # Compute SSIM
        result = _ssim_single(img1_fp32, img2_fp32, window, data_range, use_sample_covariance=False)

    # Return result (will be cast back to original dtype by caller if needed)
    return result


def _ms_ssim(
    img1: Tensor,
    img2: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    weights: Optional[List[float]] = None,
) -> Tensor:
    """
    Compute Multi-Scale SSIM.

    MS-SSIM computes SSIM at multiple scales by iteratively downsampling the images.
    At scales 1 to M-1, only contrast and structure components are computed.
    At the final scale M, all three components (luminance, contrast, structure) are computed.
    The final MS-SSIM is the weighted product of these components.

    Args:
        img1: First image [B, C, H, W] or [C, H, W]
        img2: Second image [B, C, H, W] or [C, H, W]
        window_size: Gaussian window size (default: 11)
        sigma: Gaussian sigma (default: 1.5)
        data_range: Image value range (default: 1.0)
        weights: Scale weights (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
                 Standard 5-scale weights from Wang et al. (2003)

    Returns:
        MS-SSIM scalar tensor

    Example:
        >>> img1 = torch.rand(1, 1, 256, 256)
        >>> img2 = torch.rand(1, 1, 256, 256)
        >>> ms_ssim_val = _ms_ssim(img1, img2)
        >>> print(f"MS-SSIM: {ms_ssim_val:.4f}")
        >>>
        >>> # Custom weights for 3 scales
        >>> ms_ssim_3 = _ms_ssim(img1, img2, weights=[0.33, 0.33, 0.34])

    Notes:
        - Requires minimum image size of 176×176 for 5 scales (after 4 downsamplings)
        - Uses 2×2 average pooling for downsampling between scales
        - Automatically adjusts to fewer scales if image becomes too small
        - Differentiable for use as loss function
        - Default weights sum to 1.0
        - MS-SSIM formula: ∏(component_j^weight_j) for j=1 to M

    References:
        Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003).
        "Multiscale structural similarity for image quality assessment."
    """
    # AMP-safe MS-SSIM computation: disable autocast and use float32
    # MS-SSIM requires float32 for numerical stability in convolution operations
    with torch.cuda.amp.autocast(enabled=False):
        # Default weights for 5 scales (standard configuration)
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

        # Ensure 4D tensors
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)

        # Convert to float32 for MS-SSIM computation
        img1_fp32 = img1.float()
        img2_fp32 = img2.float()

        # Convert weights to tensor (float32)
        weights_tensor = torch.tensor(weights, device=img1_fp32.device, dtype=torch.float32)

        levels = len(weights)
        mcs = []  # Multi-scale contrast-structure components

        channels = img1_fp32.shape[1]
        window = _create_gaussian_window(window_size, sigma, channels)
        window = window.to(img1_fp32.device, dtype=torch.float32)

        # Constants (matching scikit-image)
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        for i in range(levels):
            # Compute SSIM components at this scale
            mu1 = _gaussian_filter(img1_fp32, window)
            mu2 = _gaussian_filter(img2_fp32, window)

            mu1_sq = mu1**2
            mu2_sq = mu2**2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = _gaussian_filter(img1_fp32**2, window) - mu1_sq
            sigma2_sq = _gaussian_filter(img2_fp32**2, window) - mu2_sq
            sigma12 = _gaussian_filter(img1_fp32 * img2_fp32, window) - mu1_mu2

            # Contrast-structure component
            cs = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)

            if i == levels - 1:
                # Last scale: include luminance component
                l_component = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
                ssim_val = l_component * cs
                mcs.append(ssim_val.mean())
            else:
                # Intermediate scales: only contrast-structure
                mcs.append(cs.mean())

                # Downsample for next scale
                img1_fp32 = F.avg_pool2d(img1_fp32, kernel_size=2, stride=2)
                img2_fp32 = F.avg_pool2d(img2_fp32, kernel_size=2, stride=2)

                # Check minimum size - need at least window_size in each dimension
                if img1_fp32.shape[-1] < window_size or img1_fp32.shape[-2] < window_size:
                    # Image too small, break early and adjust weights
                    weights_tensor = weights_tensor[: i + 1]
                    weights_tensor = weights_tensor / weights_tensor.sum()
                    break

        # Combine scales with weighted product
        # MS-SSIM = ∏(component_j^weight_j)
        # Use log-space for numerical stability: exp(∑ weight_j * log(component_j))
        mcs_tensor = torch.stack(mcs)

        # Clamp to avoid log(0) and ensure positive values
        mcs_tensor = torch.clamp(mcs_tensor, min=1e-10, max=1.0)

        # Compute in log space for numerical stability
        ms_ssim = torch.exp(torch.sum(weights_tensor * torch.log(mcs_tensor)))

    return ms_ssim


# ============================================================================
# Loss Function Switching for Retries
# ============================================================================

# Order of loss functions to try during retry attempts
# This allows trying different loss landscapes when a sample fails to converge
LOSS_RETRY_ORDER: List[str] = ["l1", "ssim", "l2", "ms-ssim"]


def get_retry_loss_type(original_loss: str, retry_num: int) -> str:
    """Get alternative loss function for retry attempt.

    Cycles through LOSS_RETRY_ORDER to provide different loss landscapes
    when a sample fails to converge with the original loss function.

    Parameters
    ----------
    original_loss : str
        Original loss type used ("l1", "l2", "ssim", "ms-ssim")
    retry_num : int
        Current retry attempt number (1-based)

    Returns
    -------
    str
        Alternative loss type to use for this retry

    Examples
    --------
    >>> get_retry_loss_type("l1", 1)
    'ssim'
    >>> get_retry_loss_type("l1", 2)
    'l2'
    >>> get_retry_loss_type("ssim", 1)
    'l2'

    Notes
    -----
    The function cycles through LOSS_RETRY_ORDER starting from the
    original loss type. This ensures each retry tries a different
    loss landscape that might help the sample converge.
    """
    try:
        current_idx = LOSS_RETRY_ORDER.index(original_loss)
    except ValueError:
        # If original loss not in order, start from beginning
        current_idx = 0

    # Cycle through alternatives
    new_idx = (current_idx + retry_num) % len(LOSS_RETRY_ORDER)
    return LOSS_RETRY_ORDER[new_idx]


# ============================================================================
# Loss Strategy Pattern
# ============================================================================


class ProgressiveLossStrategy(ABC):
    """
    Abstract base class for progressive loss functions.

    All loss strategies implement a common interface for computing
    loss between predictions and targets. This enables flexible composition
    and extension of loss functions without modifying core training logic.

    Subclasses must implement:
        - __call__: Compute loss between prediction and target
        - name: Property returning loss function name for logging

    Example:
        >>> class CustomLoss(ProgressiveLossStrategy):
        ...     def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        ...         return torch.abs(pred - target).mean()
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "custom"
    """

    @abstractmethod
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute loss between prediction and target.

        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]

        Returns:
            Scalar loss value (differentiable)

        Notes:
            - Implementation should return a scalar tensor for backpropagation
            - Loss should be differentiable w.r.t. pred
            - Both inputs should be on the same device
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Loss function name for logging and identification."""
        pass


class L1LossStrategy(ProgressiveLossStrategy):
    """
    L1 (Mean Absolute Error) loss strategy.

    Computes the mean absolute difference between predictions and targets.
    Also known as MAE. Robust to outliers compared to L2.

    Formula: L1 = mean(|pred - target|)

    Example:
        >>> loss_fn = L1LossStrategy()
        >>> pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        >>> loss = loss_fn(pred, target)
        >>> print(f"L1 Loss: {loss:.4f}")
        L1 Loss: 0.5000

    Notes:
        - Suitable for general-purpose reconstruction tasks
        - Less sensitive to outliers than L2
        - Gradient magnitude is constant (not proportional to error)
    """

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute L1 loss."""
        return F.l1_loss(pred, target, reduction="mean")

    @property
    def name(self) -> str:
        return "l1"


class L2LossStrategy(ProgressiveLossStrategy):
    """
    L2 (Mean Squared Error) loss strategy.

    Computes the mean squared difference between predictions and targets.
    Also known as MSE. Penalizes large errors more than L1.

    Formula: L2 = mean((pred - target)²)

    Example:
        >>> loss_fn = L2LossStrategy()
        >>> pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        >>> loss = loss_fn(pred, target)
        >>> print(f"L2 Loss: {loss:.4f}")
        L2 Loss: 0.2500

    Notes:
        - Standard choice for many reconstruction tasks
        - More sensitive to outliers than L1 (quadratic penalty)
        - Gradient magnitude proportional to error (larger errors → larger gradients)
    """

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute L2 loss."""
        return F.mse_loss(pred, target, reduction="mean")

    @property
    def name(self) -> str:
        return "l2"


class SSIMLossStrategy(ProgressiveLossStrategy):
    """
    SSIM (Structural Similarity) loss strategy.

    Computes structural similarity between predictions and targets,
    considering luminance, contrast, and structure. Returns DSSIM as loss.

    Formula: Loss = (1 - SSIM) / 2, where SSIM ∈ [-1, 1]
    Range: [0, 0.5] where 0 = perfect match, 0.5 = maximum dissimilarity

    Args:
        window_size: Gaussian window size (default: 11)
        sigma: Gaussian sigma (default: 1.5)
        data_range: Expected range of values (default: 1.0 for [0,1] images)

    Example:
        >>> loss_fn = SSIMLossStrategy(window_size=11, sigma=1.5)
        >>> pred = torch.rand(1, 1, 128, 128)
        >>> target = torch.rand(1, 1, 128, 128)
        >>> loss = loss_fn(pred, target)
        >>> print(f"SSIM Loss: {loss:.4f}")

    Notes:
        - Better perceptual quality than L1/L2 for images
        - Considers structural information, not just pixel differences
        - Window is cached per device for efficiency
        - Requires images larger than window_size in each dimension
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        """
        Initialize SSIM loss strategy.

        Args:
            window_size: Gaussian window size (must be odd)
            sigma: Gaussian standard deviation
            data_range: Expected range of input values
        """
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self._window_cache: Dict[Tuple[int, float, Any, Any, int], Tensor] = {}

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute SSIM loss with device-aware window caching.

        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]

        Returns:
            DSSIM loss scalar in range [0, 0.5]
        """
        # Get or create Gaussian window on correct device
        device = pred.device
        dtype = pred.dtype
        channels = pred.shape[1]
        cache_key = (self.window_size, self.sigma, device, dtype, channels)

        if cache_key not in self._window_cache:
            window = _create_gaussian_window(self.window_size, self.sigma, channels)
            self._window_cache[cache_key] = window.to(device, dtype=dtype)

        window = self._window_cache[cache_key]

        # Compute SSIM
        ssim_val = _ssim_single(pred, target, window, self.data_range, use_sample_covariance=False)

        # Return DSSIM loss: (1 - SSIM) / 2
        return torch.as_tensor((1.0 - ssim_val) / 2.0)

    @property
    def name(self) -> str:
        return "ssim"


class MSSSIMLossStrategy(ProgressiveLossStrategy):
    """
    Multi-Scale SSIM (MS-SSIM) loss strategy.

    Computes SSIM at multiple scales for better perceptual quality assessment.
    More robust to scaling and viewing distance than single-scale SSIM.

    Args:
        window_size: Gaussian window size (default: 11)
        sigma: Gaussian sigma (default: 1.5)
        data_range: Expected range of values (default: 1.0)
        weights: Scale weights (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    Example:
        >>> loss_fn = MSSSIMLossStrategy()
        >>> pred = torch.rand(1, 1, 256, 256)
        >>> target = torch.rand(1, 1, 256, 256)
        >>> loss = loss_fn(pred, target)
        >>> print(f"MS-SSIM Loss: {loss:.4f}")

    Notes:
        - Requires minimum image size of 176×176 for 5 scales
        - Automatically adjusts to fewer scales if image is too small
        - Default weights from Wang et al. (2003) sum to 1.0
        - More computationally expensive than single-scale SSIM
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize MS-SSIM loss strategy.

        Args:
            window_size: Gaussian window size
            sigma: Gaussian standard deviation
            data_range: Expected range of input values
            weights: Per-scale weights (default: standard 5-scale weights)
        """
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.weights = weights or [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.num_scales = len(self.weights)

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute MS-SSIM loss with input validation.

        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]

        Returns:
            MS-DSSIM loss scalar in range [0, 0.5]

        Raises:
            ValueError: If input is too small for the number of scales
        """
        # Validate input size
        min_size = 2 ** (self.num_scales - 1) * self.window_size
        if pred.shape[-1] < min_size or pred.shape[-2] < min_size:
            raise ValueError(
                f"Input too small for MS-SSIM with {self.num_scales} scales. "
                f"Got {pred.shape[-2:]}, need at least ({min_size}, {min_size})"
            )

        # Compute multi-scale SSIM
        msssim_val = _ms_ssim(
            pred, target, self.window_size, self.sigma, self.data_range, self.weights
        )

        # Return MS-DSSIM loss: (1 - MS-SSIM) / 2
        return torch.as_tensor((1.0 - msssim_val) / 2.0)

    @property
    def name(self) -> str:
        return "msssim"


class FastSSIMLossStrategy(ProgressiveLossStrategy):
    """
    Optimized SSIM loss with automatic downsampling for large images.

    This strategy provides significant speedup (5-10x) for large images by
    downsampling before computing SSIM. Since SSIM measures structural similarity,
    downsampling preserves the essential quality metrics while dramatically
    reducing computation time.

    Args:
        window_size: Gaussian window size (default: 11)
        sigma: Gaussian sigma (default: 1.5)
        data_range: Expected range of values (default: 1.0)
        max_size: Maximum image size before downsampling (default: 256)
            Images larger than this will be downsampled to this size.
        downsample_mode: Interpolation mode for downsampling (default: 'bilinear')

    Example:
        >>> # Standard SSIM for 1024x1024: ~3ms
        >>> # Fast SSIM for 1024x1024: ~0.3ms (10x speedup)
        >>> loss_fn = FastSSIMLossStrategy(max_size=256)
        >>> pred = torch.rand(1, 1, 1024, 1024)
        >>> target = torch.rand(1, 1, 1024, 1024)
        >>> loss = loss_fn(pred, target)

    Notes:
        - Preserves SSIM quality metrics (structural similarity is scale-invariant)
        - 5-10x speedup for images > 512x512
        - Recommended for training where exact SSIM isn't critical
        - For evaluation, use standard SSIMLossStrategy for exact values
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        max_size: int = 256,
        downsample_mode: str = "bilinear",
    ):
        """
        Initialize fast SSIM loss strategy.

        Args:
            window_size: Gaussian window size (must be odd)
            sigma: Gaussian standard deviation
            data_range: Expected range of input values
            max_size: Maximum dimension before downsampling (default: 256)
            downsample_mode: Interpolation mode ('bilinear', 'bicubic', 'area')
        """
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.max_size = max_size
        self.downsample_mode = downsample_mode
        self._window_cache: Dict[Tuple[int, float, Any, Any, int], Tensor] = {}

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute SSIM loss with automatic downsampling for large images.

        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]

        Returns:
            DSSIM loss scalar in range [0, 0.5]
        """
        h, w = pred.shape[-2:]

        # Downsample if image is larger than max_size
        if h > self.max_size or w > self.max_size:
            scale = self.max_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            pred = F.interpolate(
                pred, size=(new_h, new_w), mode=self.downsample_mode, align_corners=False
            )
            target = F.interpolate(
                target, size=(new_h, new_w), mode=self.downsample_mode, align_corners=False
            )

        # Get or create Gaussian window on correct device
        device = pred.device
        dtype = pred.dtype
        channels = pred.shape[1]
        cache_key = (self.window_size, self.sigma, device, dtype, channels)

        if cache_key not in self._window_cache:
            window = _create_gaussian_window(self.window_size, self.sigma, channels)
            self._window_cache[cache_key] = window.to(device, dtype=dtype)

        window = self._window_cache[cache_key]

        # Compute SSIM
        ssim_val = _ssim_single(pred, target, window, self.data_range, use_sample_covariance=False)

        # Return DSSIM loss: (1 - SSIM) / 2
        return torch.as_tensor((1.0 - ssim_val) / 2.0)

    @property
    def name(self) -> str:
        return "fast_ssim"


class AMPLossWrapper(ProgressiveLossStrategy):
    """
    Automatic Mixed Precision (AMP) wrapper for loss strategies.

    This wrapper ensures loss computations are AMP-compatible by:
    1. Casting inputs to float32 for SSIM computation (numerical stability)
    2. Running the underlying loss in the appropriate precision
    3. Returning results in the original dtype

    Using mixed precision training (float16) can provide 1.5-2x speedup on
    compatible GPUs while maintaining accuracy.

    Args:
        strategy: The underlying loss strategy to wrap
        force_float32: Force float32 for numerically sensitive operations (default: True)

    Example:
        >>> from torch.cuda.amp import autocast, GradScaler
        >>> scaler = GradScaler()
        >>>
        >>> # Wrap SSIM for AMP compatibility
        >>> ssim_loss = AMPLossWrapper(SSIMLossStrategy())
        >>>
        >>> with autocast():
        ...     pred = model(input)  # float16
        ...     loss = ssim_loss(pred, target)  # Handled correctly
        >>>
        >>> scaler.scale(loss).backward()

    Notes:
        - SSIM and MS-SSIM require float32 for numerical stability
        - L1/L2 losses work fine with float16
        - Automatically detects and handles dtype conversion
    """

    def __init__(self, strategy: ProgressiveLossStrategy, force_float32: bool = True):
        """
        Initialize AMP loss wrapper.

        Args:
            strategy: The loss strategy to wrap
            force_float32: Force float32 precision (recommended for SSIM)
        """
        self.strategy = strategy
        self.force_float32 = force_float32
        self._needs_float32 = strategy.name in ("ssim", "msssim", "fast_ssim")

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute loss with AMP-compatible dtype handling.

        Args:
            pred: Predicted tensor [B, C, H, W] (may be float16)
            target: Target tensor [B, C, H, W] (may be float16)

        Returns:
            Loss value (cast back to original dtype)
        """
        original_dtype = pred.dtype

        # For SSIM-based losses, ensure float32 for numerical stability
        if self.force_float32 and self._needs_float32 and pred.dtype != torch.float32:
            pred = pred.float()
            target = target.float()

        # Compute loss
        loss = self.strategy(pred, target)

        # Cast back to original dtype if needed
        if loss.dtype != original_dtype:
            loss = loss.to(original_dtype)

        return loss

    @property
    def name(self) -> str:
        return f"amp_{self.strategy.name}"


class CompositeLossStrategy(ProgressiveLossStrategy):
    """
    Weighted combination of multiple loss strategies.

    Enables flexible composition of different loss functions with custom weights.
    Useful for combining pixel-wise losses (L1/L2) with perceptual losses (SSIM).

    Args:
        losses: Dict of {name: (strategy, weight)} pairs

    Example:
        >>> # Combine L1 (70%) and SSIM (30%)
        >>> composite = CompositeLossStrategy({
        ...     'l1': (L1LossStrategy(), 0.7),
        ...     'ssim': (SSIMLossStrategy(), 0.3)
        ... })
        >>> pred = torch.rand(1, 1, 128, 128)
        >>> target = torch.rand(1, 1, 128, 128)
        >>> loss = composite(pred, target)
        >>>
        >>> # Triple combination
        >>> composite3 = CompositeLossStrategy({
        ...     'l1': (L1LossStrategy(), 0.5),
        ...     'l2': (L2LossStrategy(), 0.3),
        ...     'ssim': (SSIMLossStrategy(), 0.2)
        ... })

    Notes:
        - Weights must sum to 1.0 (within tolerance of ±0.01)
        - All strategies are evaluated on the same pred/target pair
        - Loss name is constructed from weighted components (e.g., "0.7*l1+0.3*ssim")
        - Useful for balancing different aspects of reconstruction quality
    """

    def __init__(self, losses: Dict[str, Tuple[ProgressiveLossStrategy, float]]):
        """
        Initialize composite loss strategy.

        Args:
            losses: Dict of {name: (strategy, weight)} pairs

        Raises:
            ValueError: If weights don't sum to approximately 1.0

        Example:
            >>> losses = {
            ...     'pixel': (L1LossStrategy(), 0.6),
            ...     'structural': (SSIMLossStrategy(), 0.4)
            ... }
            >>> composite = CompositeLossStrategy(losses)
        """
        self.losses = losses
        total_weight = sum(w for _, w in losses.values())
        if not (0.99 < total_weight < 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight:.4f}")

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute weighted sum of all component losses.

        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]

        Returns:
            Weighted combination of all losses
        """
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for strategy, weight in self.losses.values():
            total_loss = total_loss + weight * strategy(pred, target)
        return total_loss

    @property
    def name(self) -> str:
        """Generate descriptive name from component losses."""
        names = [f"{w:.1f}*{s.name}" for s, w in self.losses.values()]
        return "+".join(names)


class LossAggregator(nn.Module):
    """
    Aggregated loss for progressive SPIDS training.

    This is a critical component of SPIDS. It computes loss on both:
    1. Old measurements (accumulated mask from previous samples)
    2. New measurement (current sample only)

    The dual loss allows independent convergence checking:
    - loss_old: Ensures model still fits all previous measurements
    - loss_new: Ensures model fits the new measurement

    For L1/L2 losses: Both losses are normalized by "zero loss" (loss of zeros vs target)
    to make threshold-based stopping robust to varying measurement intensities.

    For SSIM losses: Uses DSSIM = (1 - SSIM) / 2 formulation, operating in measurement space
    (same as L1/L2, but using structural similarity instead of pixel-wise difference).

    Args:
        loss_type: Type of loss function ("l1", "l2", "ssim", or "ms-ssim") (default: "l1")
        new_weight: Weight for new loss (unused, kept for compatibility)
        f_weight: Frequency-domain weight (unused, kept for compatibility)

    Attributes:
        running_loss1: Running average of loss_old (for monitoring)
        running_loss2: Running average of loss_new (for monitoring)

    Returns:
        Tuple[Tensor, Tensor]: (loss_old, loss_new)
            - loss_old: Normalized loss on accumulated measurements
            - loss_new: Normalized loss on new measurement

    Example:
        >>> # L1 loss in measurement space
        >>> criterion = LossAggregator(loss_type="l1")
        >>> measurement_system = MeasurementSystem(telescope)
        >>>
        >>> # First measurement (only new loss is meaningful)
        >>> measurement = measurement_system.measure(image, reconstruction, center=[0, 0])
        >>> output = model()
        >>> loss_old, loss_new = criterion(output, measurement, measurement_system, [0, 0])
        >>> print(f"Old: {loss_old:.4f}, New: {loss_new:.4f}")
        >>>
        >>> # Add to accumulator
        >>> measurement_system.add_measurement([0, 0])
        >>>
        >>> # Second measurement (both losses are meaningful)
        >>> measurement = measurement_system.measure(image, reconstruction, center=[10, 10])
        >>> output = model()
        >>> loss_old, loss_new = criterion(output, measurement, measurement_system, [10, 10])
        >>> # loss_old checks fit to first measurement
        >>> # loss_new checks fit to second measurement
        >>>
        >>> # SSIM loss in measurement space (same as L1, different metric)
        >>> criterion_ssim = LossAggregator(loss_type="ssim")
        >>> measurement = measurement_system.measure(image, reconstruction, center=[10, 10])
        >>> output = model()
        >>> loss_old, loss_new = criterion_ssim(output, measurement, measurement_system, [10, 10])

    Notes:
        - All losses operate in measurement space using MeasurementSystem
        - L1/L2: Pixel-wise differences, normalized by zero-loss for consistent thresholds
        - SSIM: Structural similarity in measurement space, uses DSSIM = (1 - SSIM) / 2, range [0, 0.5]
        - Returns tuple of two losses for dual convergence checking
    """

    def __init__(
        self,
        loss_type: Literal["l1", "l2", "ssim", "ms-ssim", "composite"] = "l1",
        new_weight: Optional[float] = None,
        f_weight: Optional[float] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        **strategy_kwargs: Any,
    ) -> None:
        """
        Initialize aggregated loss.

        Args:
            loss_type: One of ["l1", "l2", "ssim", "ms-ssim", "composite"]
            new_weight: Weight for new loss (unused, kept for compatibility)
            f_weight: Frequency-domain weight (unused, kept for compatibility)
            loss_weights: For composite losses, dict of {loss_name: weight}
                         Example: {'l1': 0.7, 'ssim': 0.3}
            **strategy_kwargs: Additional arguments for loss strategies
                              (e.g., window_size, sigma for SSIM)

        Raises:
            ValueError: If loss_type is not one of the supported types

        Example:
            >>> # Simple L1 loss
            >>> loss = LossAggregator(loss_type='l1')
            >>>
            >>> # SSIM with custom window
            >>> loss = LossAggregator(loss_type='ssim', window_size=7, sigma=1.0)
            >>>
            >>> # Composite: 70% L1 + 30% SSIM
            >>> loss = LossAggregator(
            ...     loss_type='composite',
            ...     loss_weights={'l1': 0.7, 'ssim': 0.3}
            ... )
        """
        super().__init__()
        self.loss_type = loss_type
        self.register_buffer("running_loss1", torch.tensor(1.0))
        self.register_buffer("running_loss2", torch.tensor(1.0))

        if loss_type == "l1":
            self.loss: nn.Module = nn.L1Loss(reduction="none")
            self.use_strategy = False
        elif loss_type == "l2":
            self.loss = nn.MSELoss(reduction="none")
            self.use_strategy = False
        elif loss_type in ["ssim", "ms-ssim"]:
            # SSIM-based losses
            # Create Gaussian window (will be moved to device on first forward pass)
            self.window_size = strategy_kwargs.get("window_size", 11)
            self.sigma = strategy_kwargs.get("sigma", 1.5)
            self.data_range = strategy_kwargs.get("data_range", 1.0)
            self.window: Optional[Tensor] = None  # Created on first forward pass
            self.use_strategy = False
        elif loss_type == "composite":
            # Composite loss using strategy pattern
            if loss_weights is None:
                raise ValueError("loss_weights must be provided for composite loss type")

            # Create individual strategies
            strategies = {}
            for name, weight in loss_weights.items():
                strategy = self._create_base_strategy(name, **strategy_kwargs)
                strategies[name] = (strategy, weight)

            self.loss_strategy = CompositeLossStrategy(strategies)
            self.use_strategy = True
        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}. "
                f"Must be one of ['l1', 'l2', 'ssim', 'ms-ssim', 'composite']"
            )

    @staticmethod
    def _create_base_strategy(name: str, **kwargs: Any) -> ProgressiveLossStrategy:
        """
        Create a base loss strategy by name.

        Args:
            name: Strategy name ('l1', 'l2', 'ssim', 'msssim')
            **kwargs: Strategy-specific parameters

        Returns:
            ProgressiveLossStrategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        if name == "l1":
            return L1LossStrategy()
        elif name == "l2":
            return L2LossStrategy()
        elif name == "ssim":
            return SSIMLossStrategy(**kwargs)
        elif name == "msssim" or name == "ms-ssim":
            return MSSSIMLossStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown strategy name: {name}")

    def forward(
        self,
        inputs: Tensor,  # [B, C, H, W]
        target: Tensor,  # [2, C, H, W]
        telescope: Optional[nn.Module] = None,
        center: Optional[List[float]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute aggregated loss on old and new measurements.

        Args:
            inputs: Model output (reconstructed image) [B, C, H, W]
            target: Target measurements [2, C, H, W]
                    target[0] = old accumulated measurements
                    target[1] = new measurement
            telescope: MeasurementSystem instance for generating measurements from reconstruction
            center: Center coordinates for new measurement

        Returns:
            Tuple[Tensor, Tensor]: (loss_old, loss_new)
                - L1/L2: Normalized loss on accumulated/new measurements
                - SSIM: DSSIM = (1 - SSIM) / 2, range [0, 0.5]

        Notes:
            - All losses operate in measurement space
            - Telescope transforms reconstruction into measurements (diffraction patterns)
            - L1/L2: Pixel-wise differences, normalized by zero-loss
            - SSIM: Structural similarity metric on measurements
            - If telescope is None, inputs are duplicated (for simple testing)
        """
        # Validate input shapes
        assert inputs.ndim == 4, f"Expected 4D input tensor [B, C, H, W], got {inputs.ndim}D"
        assert target.ndim == 4, f"Expected 4D target tensor [2, C, H, W], got {target.ndim}D"
        assert target.shape[0] == 2, f"Expected target with 2 measurements, got {target.shape[0]}"

        if self.loss_type in ["l1", "l2"]:
            # Original L1/L2 implementation (measurement space)
            if telescope is None:
                inputs = torch.cat([inputs, inputs], dim=0)
            else:
                inputs = telescope(inputs, center)

            # Compute loss per pixel
            loss = self.loss(inputs, target).view(target.size(0), -1).mean(dim=-1)

            # Normalize by zero-loss (loss of zeros vs target)
            norm = self.loss(torch.zeros_like(inputs), target).view(target.size(0), -1).mean(dim=-1)
            loss1, loss2 = loss / norm

            return loss1, loss2

        elif self.loss_type in ["ssim", "ms-ssim"]:
            # SSIM-based losses operate in MEASUREMENT space (same as L1/L2)
            # Generate measurements through telescope, then apply SSIM

            # Generate measurements through telescope
            if telescope is None:
                measurements = torch.cat([inputs, inputs], dim=0)
            else:
                measurements = telescope(inputs, center)

            # AMP-safe SSIM computation: disable autocast and use float32
            # SSIM requires float32 for numerical stability in convolution operations
            with torch.cuda.amp.autocast(enabled=False):
                # Convert to float32 for SSIM computation
                measurements_fp32 = measurements.float()
                target_fp32 = target.float()

                # Initialize window on first call
                if self.window is None:
                    channels = measurements_fp32.shape[1]
                    self.window = _create_gaussian_window(self.window_size, self.sigma, channels)

                # Move window to same device (always float32 for SSIM)
                self.window = self.window.to(measurements_fp32.device, dtype=torch.float32)

                # Split measurements into old and new
                measurement_old = measurements_fp32[0].unsqueeze(0)  # [1, C, H, W]
                measurement_new = measurements_fp32[1].unsqueeze(0)  # [1, C, H, W]

                # Ensure target tensors have batch dimension
                target_old = target_fp32[0].unsqueeze(0) if target_fp32[0].dim() == 3 else target_fp32[0]
                target_new = target_fp32[1].unsqueeze(0) if target_fp32[1].dim() == 3 else target_fp32[1]

                # Compute SSIM between measurements and targets
                if self.loss_type == "ssim":
                    # Single-scale SSIM
                    ssim_old = _ssim_single(
                        measurement_old,
                        target_old,
                        self.window,
                        self.data_range,
                        use_sample_covariance=False,
                    )
                    ssim_new = _ssim_single(
                        measurement_new,
                        target_new,
                        self.window,
                        self.data_range,
                        use_sample_covariance=False,
                    )
                else:  # ms-ssim
                    # Multi-scale SSIM
                    ssim_old = _ms_ssim(
                        measurement_old,
                        target_old,
                        self.window_size,
                        self.sigma,
                        self.data_range,
                    )
                    ssim_new = _ms_ssim(
                        measurement_new,
                        target_new,
                        self.window_size,
                        self.sigma,
                        self.data_range,
                    )

                # Convert SSIM to DSSIM loss: (1 - SSIM) / 2
                # Range: [0, 0.5] where 0 = perfect match, 0.5 = maximum dissimilarity
                loss_old = (1.0 - ssim_old) / 2.0
                loss_new = (1.0 - ssim_new) / 2.0

            # Note: No zero-loss normalization for SSIM
            # DSSIM is already bounded and in a reasonable range

            # Convert back to original dtype for compatibility with AMP
            return loss_old.to(inputs.dtype), loss_new.to(inputs.dtype)

        elif self.loss_type == "composite":
            # Composite loss operates in MEASUREMENT space using strategy pattern
            # Generate measurements through telescope
            if telescope is None:
                measurements = torch.cat([inputs, inputs], dim=0)
            else:
                measurements = telescope(inputs, center)

            # Split measurements into old and new
            measurement_old = measurements[0].unsqueeze(0)  # [1, C, H, W]
            measurement_new = measurements[1].unsqueeze(0)  # [1, C, H, W]

            # Ensure target tensors have batch dimension
            target_old = target[0].unsqueeze(0) if target[0].dim() == 3 else target[0]
            target_new = target[1].unsqueeze(0) if target[1].dim() == 3 else target[1]

            # Apply composite strategy to each measurement
            loss_old = self.loss_strategy(measurement_old, target_old)
            loss_new = self.loss_strategy(measurement_new, target_new)

            # Note: No zero-loss normalization for composite losses
            # Each component strategy should return normalized/bounded values

            return loss_old, loss_new

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

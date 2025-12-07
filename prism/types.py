"""Type definitions for PRISM.

This module provides type aliases and protocols used throughout the PRISM codebase
for better type safety and documentation.

Protocols:
    - TelescopeProtocol: Interface for telescope measurement systems
    - PropagatorProtocol: Interface for optical propagation
    - HasForward: Generic callable with forward method
    - Optimizer: PyTorch optimizer protocol
    - LRScheduler: Learning rate scheduler protocol

TypedDicts:
    - TrainingArgs: Configuration for training parameters
    - ExperimentData: Experiment checkpoint and metrics data
    - ConvergenceStats: Convergence monitoring statistics
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeAlias, Union, runtime_checkable

import numpy as np
import torch
from torch import Tensor
from typing_extensions import TypedDict


# ============================================================================
# Tensor Type Aliases
# ============================================================================

# Tensor type aliases with shape annotations
TensorImage: TypeAlias = Tensor  # [B, C, H, W] - Batch of images
ComplexTensor: TypeAlias = Tensor  # Complex-valued tensor
RealTensor: TypeAlias = Tensor  # Real-valued tensor
MaskTensor: TypeAlias = Tensor  # Binary mask tensor (bool or float)

# Spatial coordinates and patterns
Point2D: TypeAlias = Tuple[float, float]  # (y, x) coordinates
SamplePattern: TypeAlias = List[Point2D]  # List of sampling positions
GridCoordinates: TypeAlias = Tuple[Tensor, Tensor]  # (Y, X) coordinate grids

# Device types
Device: TypeAlias = Union[str, torch.device]

# Numpy-Torch interop
ArrayOrTensor: TypeAlias = Union[np.ndarray, Tensor]

# Configuration types
PathLike: TypeAlias = Union[str, "Path"]  # type: ignore[name-defined]  # noqa: F821


# ============================================================================
# Protocols for Core Components
# ============================================================================


@runtime_checkable
class TelescopeProtocol(Protocol):
    """Protocol for telescope measurement systems.

    This protocol defines the interface that any telescope-like measurement
    system must implement. It allows duck-typed usage of telescope objects
    in sampling, pattern generation, and visualization functions.

    Attributes:
        n: Image size in pixels (Tensor buffer)
        r: Aperture radius in pixels (Tensor buffer)
        x: Spatial x coordinates centered at image center
        y: Spatial y coordinates centered at image center

    Methods:
        mask: Generate aperture mask at given center
        forward/__call__: Complete measurement pipeline

    Example:
        >>> def compute_energy(obj: Tensor, telescope: TelescopeProtocol) -> Tensor:
        ...     measurement = telescope(obj, centers=[[0, 0]])
        ...     return measurement.norm()
    """

    n: Tensor
    r: Tensor

    @property
    def x(self) -> Tensor:
        """Spatial x coordinates centered at image center."""
        ...

    @property
    def y(self) -> Tensor:
        """Spatial y coordinates centered at image center."""
        ...

    def mask(self, center: Optional[List[float]] = None, r: Optional[float] = None) -> Tensor:
        """Generate aperture mask at given center.

        Args:
            center: Center position [y, x]. Defaults to [0, 0]
            r: Radius override. Defaults to self.r

        Returns:
            Boolean mask of shape (n, n)
        """
        ...

    def __call__(
        self,
        tensor: Tensor,
        centers: Union[Tensor, List[List[float]], None] = None,
        r: Optional[float] = None,
        is_sum: Optional[bool] = None,
        sum_pattern: Optional[List[List[int]]] = None,
        add_noise: bool = False,
    ) -> Tensor:
        """Complete measurement pipeline through telescope aperture(s).

        Args:
            tensor: Input object image [B, C, H, W]
            centers: Aperture centers [[y0, x0], [y1, x1], ...]
            r: Aperture radius override
            is_sum: Whether to sum measurements
            sum_pattern: Grouping pattern for measurements
            add_noise: Whether to add shot noise

        Returns:
            Measurement(s) [N, 1, H, W]
        """
        ...


@runtime_checkable
class TelescopeAggregatorProtocol(TelescopeProtocol, Protocol):
    """Protocol for telescope aggregators that accumulate measurements.

    This protocol extends TelescopeProtocol with methods for tracking
    accumulated k-space coverage and measuring through combined masks.
    Used by visualization components to display k-space coverage.

    Attributes:
        cum_mask: Cumulative measurement mask showing k-space coverage

    Methods:
        measure_through_accumulated_mask: Apply accumulated mask to measurement
        add_mask_to_cum_mask: Add new mask to accumulated coverage

    Example:
        >>> def visualize_coverage(tel: TelescopeAggregatorProtocol) -> None:
        ...     coverage = tel.cum_mask.cpu().numpy()
        ...     plt.imshow(coverage, cmap='viridis')
    """

    cum_mask: Tensor

    def measure_through_accumulated_mask(
        self,
        tensor: Tensor,
        mask_to_add: Optional[Tensor] = None,
    ) -> Tensor:
        """Measure tensor through accumulated aperture mask.

        Args:
            tensor: Input image tensor
            mask_to_add: Optional additional mask to include

        Returns:
            Measured tensor filtered by accumulated mask
        """
        ...

    def add_mask_to_cum_mask(self, mask: Tensor) -> None:
        """Add mask to cumulative coverage.

        Args:
            mask: Mask to add to accumulated coverage
        """
        ...


@runtime_checkable
class PropagatorProtocol(Protocol):
    """Protocol for optical propagation systems.

    This protocol defines the interface for optical propagators that
    transform optical fields between planes (e.g., object to image,
    near-field to far-field).

    Example:
        >>> def propagate(field: Tensor, prop: PropagatorProtocol) -> Tensor:
        ...     return prop(field, direction="forward")
    """

    def __call__(
        self,
        tensor: Tensor,
        direction: str = "forward",
        **kwargs: Any,
    ) -> Tensor:
        """Propagate optical field.

        Args:
            tensor: Input optical field
            direction: Propagation direction ("forward" or "inverse")
            **kwargs: Additional propagation parameters

        Returns:
            Propagated optical field
        """
        ...


class HasForward(Protocol):
    """Protocol for objects with a forward method (like nn.Module)."""

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Forward pass."""
        ...


class Optimizer(Protocol):
    """Protocol for PyTorch optimizers."""

    def zero_grad(self) -> None:
        """Clear gradients."""
        ...

    def step(self) -> None:
        """Update parameters."""
        ...


class LRScheduler(Protocol):
    """Protocol for learning rate schedulers."""

    def step(self) -> None:
        """Update learning rate."""
        ...

    def get_last_lr(self) -> List[float]:
        """Get current learning rate."""
        ...


# ============================================================================
# TypedDicts for Configuration and Data Structures
# ============================================================================


class TrainingArgs(TypedDict, total=False):
    """Configuration for training parameters.

    This TypedDict defines the expected fields for training configuration
    passed to trainers and runners. All fields are optional (total=False)
    to support partial configurations.

    Attributes:
        lr: Learning rate
        max_epochs: Maximum epochs per sample
        batch_size: Batch size for training
        loss_type: Loss function type ('l1', 'l2', 'ssim', etc.)
        scheduler_type: LR scheduler type ('plateau', 'cosine_warm_restarts')
        patience: Epochs without improvement before LR reduction
        obj_size: Object/reconstruction size in pixels
        n_samples: Number of sampling positions
        snr: Signal-to-noise ratio in dB
        debug: Enable debug mode
        name: Experiment name
        log_dir: Directory for logs and checkpoints
    """

    lr: float
    max_epochs: int
    batch_size: int
    loss_type: str
    scheduler_type: str
    patience: int
    obj_size: int
    n_samples: int
    snr: float
    debug: bool
    name: str
    log_dir: str


class ExperimentMetrics(TypedDict, total=False):
    """Metrics collected during training.

    Attributes:
        loss: List of loss values per epoch
        ssim: List of SSIM values per epoch
        psnr: List of PSNR values per epoch
        lr: List of learning rates per epoch
    """

    loss: List[float]
    ssim: List[float]
    psnr: List[float]
    lr: List[float]


class FinalMetrics(TypedDict, total=False):
    """Final metrics at end of training.

    Attributes:
        epochs: Total epochs trained
        loss: Final loss value
        ssim: Final SSIM value
        psnr: Final PSNR value
        rmse: Final RMSE value
        time: Training time in seconds
    """

    epochs: int
    loss: float
    ssim: float
    psnr: float
    rmse: float
    time: float


class ExperimentData(TypedDict, total=False):
    """Experiment checkpoint and metrics data.

    This TypedDict defines the structure for experiment data loaded from
    checkpoints or used in the dashboard/visualization.

    Attributes:
        config: Training configuration dictionary
        metrics: Per-epoch metrics (ExperimentMetrics)
        final_metrics: End-of-training metrics (FinalMetrics)
        reconstruction: Final reconstructed image tensor
        ground_truth: Ground truth image tensor
        checkpoint_path: Path to checkpoint file
    """

    config: Dict[str, Any]
    metrics: ExperimentMetrics
    final_metrics: FinalMetrics
    reconstruction: Tensor
    ground_truth: Tensor
    checkpoint_path: str


class ConvergenceStats(TypedDict, total=False):
    """Convergence monitoring statistics.

    This TypedDict defines the structure for convergence statistics
    returned by convergence monitors.

    Attributes:
        is_converged: Whether training has converged
        current_tier: Current optimization tier name
        best_loss: Best loss achieved
        epochs_trained: Number of epochs trained
        plateau_count: Number of plateaus detected
        improvement_rate: Rate of loss improvement
    """

    is_converged: bool
    current_tier: str
    best_loss: float
    epochs_trained: int
    plateau_count: int
    improvement_rate: float

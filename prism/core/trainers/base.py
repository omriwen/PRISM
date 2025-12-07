"""
Abstract base class for trainers.

This module provides the AbstractTrainer base class that defines the common
interface for all training algorithms in PRISM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import nn


@dataclass
class TrainingConfig:
    """Configuration for training.

    Parameters
    ----------
    n_epochs : int
        Number of epochs per outer loop iteration
    max_epochs : int
        Maximum number of outer loop iterations
    loss_threshold : float
        Loss threshold for convergence
    learning_rate : float
        Base learning rate
    """
    n_epochs: int = 100
    max_epochs: int = 100
    loss_threshold: float = 1e-4
    learning_rate: float = 1e-4


@dataclass
class EpochResult:
    """Result of a single epoch.

    Parameters
    ----------
    loss : float
        Total loss for the epoch
    loss_old : float
        Loss on previous measurements (cumulative)
    loss_new : float
        Loss on new measurement
    """
    loss: float
    loss_old: float = 0.0
    loss_new: float = 0.0


@dataclass
class TrainingResult:
    """Result of a complete training run.

    Parameters
    ----------
    losses : list[float]
        Loss values per sample
    ssims : list[float]
        SSIM values per sample
    psnrs : list[float]
        PSNR values per sample (dB)
    rmses : list[float]
        RMSE values per sample
    final_reconstruction : torch.Tensor | None
        Final reconstructed output
    failed_samples : list[int]
        Indices of samples that failed to converge
    wall_time_seconds : float
        Total training time
    epochs_per_sample : list[int]
        Number of epochs used for each sample
    """
    losses: list[float] = field(default_factory=list)
    ssims: list[float] = field(default_factory=list)
    psnrs: list[float] = field(default_factory=list)
    rmses: list[float] = field(default_factory=list)
    final_reconstruction: Optional[torch.Tensor] = None
    failed_samples: list[int] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    epochs_per_sample: list[int] = field(default_factory=list)


class MetricsCollector:
    """Collects and aggregates training metrics.

    This class tracks metrics across training and provides methods
    for recording and finalizing results.
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self.losses: list[float] = []
        self.ssims: list[float] = []
        self.psnrs: list[float] = []
        self.rmses: list[float] = []
        self.failed_samples: list[int] = []
        self.epochs_per_sample: list[int] = []
        self._start_time: float = 0.0

    def start(self) -> None:
        """Start timing the training."""
        import time
        self._start_time = time.time()

    def record_sample(
        self,
        loss: float,
        ssim: float,
        psnr: float,
        rmse: float,
        epochs: int,
    ) -> None:
        """Record metrics for a completed sample.

        Parameters
        ----------
        loss : float
            Final loss for the sample
        ssim : float
            SSIM score
        psnr : float
            PSNR in dB
        rmse : float
            Root mean squared error
        epochs : int
            Number of epochs used
        """
        self.losses.append(loss)
        self.ssims.append(ssim)
        self.psnrs.append(psnr)
        self.rmses.append(rmse)
        self.epochs_per_sample.append(epochs)

    def record_failure(self, sample_idx: int) -> None:
        """Record a failed sample.

        Parameters
        ----------
        sample_idx : int
            Index of the failed sample
        """
        self.failed_samples.append(sample_idx)

    def finalize(self, final_reconstruction: Optional[torch.Tensor] = None) -> TrainingResult:
        """Finalize and return the training result.

        Parameters
        ----------
        final_reconstruction : torch.Tensor | None
            The final reconstructed image

        Returns
        -------
        TrainingResult
            Complete training results
        """
        import time
        wall_time = time.time() - self._start_time if self._start_time > 0 else 0.0

        return TrainingResult(
            losses=self.losses,
            ssims=self.ssims,
            psnrs=self.psnrs,
            rmses=self.rmses,
            final_reconstruction=final_reconstruction,
            failed_samples=self.failed_samples,
            wall_time_seconds=wall_time,
            epochs_per_sample=self.epochs_per_sample,
        )


class AbstractTrainer(ABC):
    """Abstract base class for trainers.

    This class defines the common interface for all training algorithms.
    Subclasses implement specific training strategies (progressive, epochal, etc.)

    Parameters
    ----------
    model : nn.Module
        The model to train
    device : torch.device
        Device to run training on
    config : TrainingConfig | None
        Training configuration (optional, can use args instead)

    Attributes
    ----------
    model : nn.Module
        The model being trained
    device : torch.device
        Training device
    config : TrainingConfig | None
        Training configuration
    metrics : MetricsCollector
        Metrics collector instance
    current_reconstruction : torch.Tensor | None
        Current reconstructed output

    Examples
    --------
    >>> class MyTrainer(AbstractTrainer):
    ...     def train(self, data): ...
    ...     def train_epoch(self, epoch, data): ...
    ...     def compute_metrics(self, output, target): ...
    >>> trainer = MyTrainer(model, device)
    >>> result = trainer.train(data)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        """Initialize the trainer.

        Parameters
        ----------
        model : nn.Module
            The model to train
        device : torch.device
            Device to run training on
        config : TrainingConfig | None
            Training configuration
        """
        self.model = model
        self.device = device
        self.config = config
        self.metrics = MetricsCollector()
        self.current_reconstruction: Optional[torch.Tensor] = None

    @abstractmethod
    def train(self, **kwargs: Any) -> TrainingResult:
        """Execute the complete training procedure.

        This is the main entry point for training. Subclasses implement
        their specific training strategy.

        Parameters
        ----------
        **kwargs : Any
            Algorithm-specific training parameters

        Returns
        -------
        TrainingResult
            Complete training results
        """
        ...

    @abstractmethod
    def train_epoch(self, epoch: int, **kwargs: Any) -> EpochResult:
        """Execute a single training epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number
        **kwargs : Any
            Algorithm-specific parameters

        Returns
        -------
        EpochResult
            Results from this epoch
        """
        ...

    @abstractmethod
    def compute_metrics(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Parameters
        ----------
        output : torch.Tensor
            Model output / reconstruction
        target : torch.Tensor
            Ground truth target

        Returns
        -------
        dict[str, float]
            Dictionary of metric name to value
        """
        ...

    def should_stop(self, loss: float) -> bool:
        """Check if training should stop based on loss.

        Default implementation checks against config threshold.
        Subclasses can override for custom stopping criteria.

        Parameters
        ----------
        loss : float
            Current loss value

        Returns
        -------
        bool
            True if training should stop
        """
        if self.config is None:
            return False
        return loss < self.config.loss_threshold

    def save_checkpoint(self, path: str, **kwargs: Any) -> None:
        """Save a training checkpoint.

        Default implementation saves model state. Subclasses can
        override to save additional state.

        Parameters
        ----------
        path : str
            Path to save checkpoint
        **kwargs : Any
            Additional data to include in checkpoint
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "current_reconstruction": self.current_reconstruction,
            **kwargs,
        }
        torch.save(checkpoint, path)

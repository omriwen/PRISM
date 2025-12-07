"""
Epochal trainer for Mo-PIE algorithm.

This module provides the EpochalTrainer class that implements epochal training
(iterate over all samples repeatedly) for iterative phase retrieval algorithms.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from prism.core import patterns
from prism.core.algorithms.mopie import MoPIE
from prism.core.trainers.base import (
    AbstractTrainer,
    EpochResult,
    MetricsCollector,
    TrainingConfig,
    TrainingResult,
)
from prism.utils.progress import ETACalculator, TrainingProgress


class EpochalTrainerConfig(TrainingConfig):
    """Configuration for epochal training.

    Parameters
    ----------
    n_epochs : int
        Number of full passes through all samples
    rand_perm : bool
        Whether to randomly permute sample order each epoch
    n_samples : int
        Number of samples per epoch
    samples_per_line_meas : int
        Number of samples per line measurement
    samples_per_line_rec : int
        Number of samples per line reconstruction
    """

    n_epochs: int = 100
    rand_perm: bool = True
    n_samples: int = 100
    samples_per_line_meas: int = 1
    samples_per_line_rec: int = 1


class EpochalTrainer(AbstractTrainer):
    """Epochal trainer for Mo-PIE algorithm.

    Implements epochal training where all samples are processed once per epoch,
    and the process repeats for multiple epochs. This differs from progressive
    training where each sample is trained to convergence before moving on.

    Parameters
    ----------
    model : MoPIE
        MoPIE model to train
    device : torch.device
        Device to run training on
    config : EpochalTrainerConfig | None
        Training configuration
    sample_centers : torch.Tensor
        Sample center positions [N, n_points, 2]
    samples_per_line_meas : int
        Number of measurement samples per line
    samples_per_line_rec : int
        Number of reconstruction samples per line
    writer : SummaryWriter | None
        TensorBoard writer for logging
    log_dir : str | None
        Directory for saving logs

    Attributes
    ----------
    model : MoPIE
        The MoPIE model being trained
    sample_centers : torch.Tensor
        Sample center positions
    n_samples : int
        Number of samples
    n_epochs : int
        Number of training epochs
    rand_perm : bool
        Whether to randomly permute sample order

    Examples
    --------
    >>> model = MoPIE(n=256, r=25, ground_truth=image)
    >>> config = EpochalTrainerConfig(n_epochs=100)
    >>> trainer = EpochalTrainer(model, device, config, sample_centers)
    >>> result = trainer.train()
    """

    def __init__(
        self,
        model: MoPIE,
        device: torch.device,
        config: EpochalTrainerConfig | None = None,
        sample_centers: torch.Tensor | None = None,
        samples_per_line_meas: int = 1,
        samples_per_line_rec: int = 1,
        writer: SummaryWriter | None = None,
        log_dir: str | None = None,
        n_samples_0: int = 0,
        rand_perm: bool = True,
    ) -> None:
        """Initialize epochal trainer."""
        super().__init__(model, device, config)
        self.model: MoPIE = model  # Type narrow for mypy
        self.sample_centers = sample_centers
        self.samples_per_line_meas = samples_per_line_meas
        self.samples_per_line_rec = samples_per_line_rec
        self.writer = writer
        self.log_dir = log_dir
        self.n_samples_0 = n_samples_0
        self.rand_perm = rand_perm

        # Extract config values
        if config is not None:
            self.n_epochs = config.n_epochs
            self.n_samples = config.n_samples
        else:
            self.n_epochs = 100
            self.n_samples = len(sample_centers) if sample_centers is not None else 100

        # Initialize metrics
        self.metrics = MetricsCollector()
        self.rmses: list[float] = []
        self.ssims: list[float] = []
        self.psnrs: list[float] = []
        self.epoch_times: list[float] = []

        # Training state
        self.training_start_time = time.time()

    def train(self, **kwargs: Any) -> TrainingResult:
        """Execute the complete epochal training procedure.

        This method runs multiple epochs, where each epoch processes all
        samples once in (optionally randomized) order.

        Parameters
        ----------
        **kwargs : Any
            Optional overrides for training parameters:
            - n_epochs: Number of epochs to run
            - sample_centers: Override sample centers

        Returns
        -------
        TrainingResult
            Complete training results including metrics and reconstruction
        """
        # Allow overriding sample centers
        if "sample_centers" in kwargs:
            self.sample_centers = kwargs["sample_centers"]

        if self.sample_centers is None:
            raise ValueError("sample_centers must be provided")

        n_epochs = kwargs.get("n_epochs", self.n_epochs)
        self.n_samples = len(self.sample_centers)

        # Start timing
        self.metrics.start()
        self.training_start_time = time.time()

        # Setup progress tracking
        total_steps = n_epochs * self.n_samples
        epoch_eta = ETACalculator(n_epochs)
        sample_eta = ETACalculator(total_steps)
        steps_completed = 0

        logger.info(
            f"Starting Mo-PIE training: {n_epochs} epochs × {self.n_samples} samples = "
            f"{total_steps} iterations"
        )

        with TrainingProgress() as training_progress:
            epoch_task_id = training_progress.add_task("Epochs", total=n_epochs)
            sample_task_id = training_progress.add_task("Samples", total=total_steps)

            for epoch in range(n_epochs):
                epoch_result = self.train_epoch(epoch)

                # Record metrics
                self.rmses.append(epoch_result.loss)  # Using loss field for RMSE
                # SSIM and PSNR stored in loss_old and loss_new temporarily
                ssim_val = epoch_result.loss_old
                psnr_val = epoch_result.loss_new
                self.ssims.append(ssim_val)
                self.psnrs.append(psnr_val)

                # Update progress
                steps_completed += self.n_samples
                sample_eta_seconds = sample_eta.update(steps_completed)
                training_progress.advance(
                    sample_task_id,
                    advance=self.n_samples,
                    metrics={
                        "epoch": epoch + 1,
                        "sample": self.n_samples,
                    },
                    eta_seconds=sample_eta_seconds,
                    description=f"Epoch {epoch + 1}/{n_epochs}",
                )

                epoch_eta_seconds = epoch_eta.update(epoch + 1)
                training_progress.advance(
                    epoch_task_id,
                    metrics={
                        "epoch": epoch + 1,
                        "ssim": ssim_val,
                        "rmse": epoch_result.loss,
                        "psnr": psnr_val,
                    },
                    eta_seconds=epoch_eta_seconds,
                    description=f"Epochs {epoch + 1}/{n_epochs}",
                )

                # Log epoch results
                logger.debug(
                    f"Epoch {epoch + 1}/{n_epochs}: SSIM={ssim_val:.3f}, "
                    f"RMSE={epoch_result.loss:.2e}, PSNR={psnr_val:.1f} dB"
                )

                # Log to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar("SSIM", ssim_val, epoch)
                    self.writer.add_scalar("RMSE", epoch_result.loss, epoch)
                    self.writer.add_scalar("PSNR", psnr_val, epoch)
                    self.writer.add_scalar("Time/per_epoch", self.epoch_times[-1], epoch)

            training_progress.complete(epoch_task_id)
            training_progress.complete(sample_task_id)

        # Get final reconstruction
        with torch.no_grad():
            self.current_reconstruction = self.model.og.abs().detach().clone()

        # Build result
        wall_time = time.time() - self.training_start_time
        return TrainingResult(
            losses=[],  # Mo-PIE doesn't have per-sample losses
            ssims=self.ssims,
            psnrs=self.psnrs,
            rmses=self.rmses,
            final_reconstruction=self.current_reconstruction,
            failed_samples=[],  # Mo-PIE doesn't track failed samples
            wall_time_seconds=wall_time,
            epochs_per_sample=[self.n_epochs] * self.n_samples,  # All samples get same epochs
        )

    def train_epoch(self, epoch: int, **kwargs: Any) -> EpochResult:
        """Execute a single training epoch.

        Processes all samples once in (optionally randomized) order,
        updating the object and probe estimates.

        Parameters
        ----------
        epoch : int
            Current epoch number
        **kwargs : Any
            Not used, for interface compatibility

        Returns
        -------
        EpochResult
            Epoch results with RMSE in loss field, SSIM in loss_old, PSNR in loss_new
        """
        t0 = time.time()

        # Get sample order
        if self.rand_perm:
            indices = torch.randperm(self.n_samples)
        else:
            indices = torch.arange(self.n_samples)

        # Process each sample
        assert self.sample_centers is not None, "sample_centers must be set"
        for center_idx, sample_idx in enumerate(indices):
            center_ends = self.sample_centers[sample_idx]

            # Create patterns for this sample
            cntr, center_rec = patterns.create_patterns(
                center_ends,
                self.samples_per_line_meas,
                self.samples_per_line_rec,
            )

            # Update model with new center position
            self.model.update_cntr(cntr, center_rec, center_idx + self.n_samples_0)

            # Perform Mo-PIE update step (object and probe)
            self.model.update_step()

        # Compute metrics at end of epoch
        rmse, ssim_val, psnr_val = self.model.errors()

        # Track epoch time
        epoch_time = time.time() - t0
        self.epoch_times.append(epoch_time)

        # Return using EpochResult (repurposing fields for Mo-PIE metrics)
        return EpochResult(
            loss=rmse,  # RMSE stored in loss
            loss_old=ssim_val,  # SSIM stored in loss_old
            loss_new=psnr_val,  # PSNR stored in loss_new
        )

    def compute_metrics(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        For Mo-PIE, we use the model's built-in error computation which
        compares the reconstruction against ground truth.

        Parameters
        ----------
        output : torch.Tensor
            Model output / reconstruction (not used, model stores internally)
        target : torch.Tensor
            Ground truth target (not used, model stores internally)

        Returns
        -------
        dict[str, float]
            Dictionary with 'ssim', 'rmse', 'psnr' metrics
        """
        rmse, ssim_val, psnr_val = self.model.errors()
        return {
            "ssim": ssim_val,
            "rmse": rmse,
            "psnr": psnr_val,
        }

    def should_stop(self, loss: float) -> bool:
        """Check if training should stop.

        For epochal training, we typically run all epochs without early stopping.
        Override this method for custom stopping criteria.

        Parameters
        ----------
        loss : float
            Current loss value

        Returns
        -------
        bool
            Always False for epochal training (run all epochs)
        """
        return False

    def save_checkpoint(self, path: str, **kwargs: Any) -> None:
        """Save a training checkpoint.

        Parameters
        ----------
        path : str
            Path to save checkpoint
        **kwargs : Any
            Additional data to include in checkpoint
        """
        checkpoint = {
            "object": self.model.Og.detach(),
            "probe": self.model.Pg.detach(),
            "sample_centers": self.sample_centers,
            "ssims": torch.tensor(self.ssims),
            "rmses": torch.tensor(self.rmses),
            "psnrs": torch.tensor(self.psnrs),
            "epoch_times": torch.tensor(self.epoch_times),
            "current_reconstruction": self.current_reconstruction,
            **kwargs,
        }
        torch.save(checkpoint, path)

    def get_reconstruction(self) -> torch.Tensor:
        """Get current reconstruction.

        Returns
        -------
        torch.Tensor
            Current object estimate magnitude
        """
        return self.model.og.abs()

    def get_object_fourier(self) -> torch.Tensor:
        """Get object estimate in Fourier domain.

        Returns
        -------
        torch.Tensor
            Object estimate in k-space
        """
        return self.model.Og

    def get_probe(self) -> torch.Tensor:
        """Get probe estimate.

        Returns
        -------
        torch.Tensor
            Probe estimate in k-space
        """
        return self.model.Pg

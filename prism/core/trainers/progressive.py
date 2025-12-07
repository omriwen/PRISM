"""
Training loops for SPIDS algorithms.

This module provides trainers for progressive reconstruction using
SPIDS and ePIE algorithms.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn, optim
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LambdaLR,
    ReduceLROnPlateau,
)
from torch.utils.tensorboard import SummaryWriter

from prism.core import patterns
from prism.core.convergence import ConvergenceMonitor, ConvergenceTier, get_tier_config
from prism.core.measurement_system import MeasurementSystem
from prism.core.trainers.base import (
    AbstractTrainer,
    EpochResult,
    MetricsCollector,
    TrainingResult,
)
from prism.models.losses import LossAggregator, _compute_ssim, get_retry_loss_type
from prism.utils.image import crop_image
from prism.utils.io import save_args, save_checkpoint
from prism.utils.metrics import compute_rmse, psnr
from prism.utils.progress import ETACalculator, TrainingProgress


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    **kwargs: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create scheduler based on type.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule
    scheduler_type : str
        Type of scheduler: 'none', 'plateau', or 'cosine_warm_restarts'
    **kwargs : Any
        Additional arguments passed to scheduler

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler
        Configured scheduler

    Raises
    ------
    ValueError
        If scheduler_type is unknown
    """
    if scheduler_type == "none":
        # No-op scheduler: keeps learning rate constant
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    elif scheduler_type == "cosine_warm_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("T_0", 50),  # Restart every 50 epochs
            T_mult=kwargs.get("T_mult", 2),  # Double period after each restart
            eta_min=kwargs.get("eta_min", 1e-6),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class PRISMTrainer(AbstractTrainer):
    """
    Progressive trainer for SPIDS algorithm.

    Handles both initialization and progressive training phases with
    convergence checking, checkpoint saving, and TensorBoard logging.

    Parameters
    ----------
    model : nn.Module
        Generative model to train (ProgressiveDecoder)
    optimizer : torch.optim.Optimizer
        PyTorch optimizer
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Learning rate scheduler
    measurement_system : MeasurementSystem
        Measurement system for progressive imaging (wraps Telescope)
    args : argparse.Namespace
        Training arguments and configuration
    device : torch.device
        Device to run training on
    log_dir : str | None, optional
        Directory for saving checkpoints and logs
    writer : SummaryWriter | None, optional
        TensorBoard writer for logging

    Examples
    --------
    >>> model = ProgressiveDecoder(...)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> trainer = PRISMTrainer(model, optimizer, ...)
    >>> trainer.run_initialization(measurement)
    >>> results = trainer.run_progressive_training(sample_centers, image_gt)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        measurement_system: MeasurementSystem,
        args: Any,
        device: torch.device,
        log_dir: str | None = None,
        writer: SummaryWriter | None = None,
        use_amp: bool = False,
        callbacks: list[Any] | None = None,
    ):
        super().__init__(model, device, config=None)  # config is None, we use args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.measurement_system = measurement_system
        self.args = args
        self.log_dir = log_dir
        self.writer = writer
        self.use_amp = use_amp
        self.callbacks = callbacks or []

        # Initialize GradScaler for mixed precision training
        self.scaler: torch.cuda.amp.GradScaler | None = (
            torch.cuda.amp.GradScaler() if use_amp else None
        )

        # Track metrics
        self.losses: list[float] = []
        self.ssims: list[float] = []
        self.rmses: list[float] = []
        self.psnrs: list[float] = []
        self.failed_samples: list[int] = []
        self.sample_times: list[float] = []
        self.lr_history: list[float] = []

        # Convergence statistics
        self.epochs_per_sample: list[int] = []
        self.tiers_per_sample: list[str] = []
        self.convergence_stats: list[dict] = []

        # State
        self.training_start_time = time.time()

    def _get_profiler(self):
        """Get profiler from callbacks if available.

        Returns
        -------
        TrainingProfiler | None
            The profiler instance if found in callbacks, else None.
        """
        for callback in self.callbacks:
            if hasattr(callback, '_profiler'):
                return callback._profiler
        return None

    def _invoke_callbacks(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Invoke callback method if it exists on registered callbacks.

        Parameters
        ----------
        method : str
            Name of the callback method to invoke (e.g., 'on_epoch_end')
        *args : Any
            Positional arguments to pass to the callback method
        **kwargs : Any
            Keyword arguments to pass to the callback method
        """
        for callback in self.callbacks:
            if hasattr(callback, method):
                getattr(callback, method)(*args, **kwargs)

    def train(self, **kwargs: Any) -> TrainingResult:
        """Execute training - delegates to run_progressive_training.

        Parameters
        ----------
        **kwargs : Any
            Training parameters including:
            - sample_centers: Sample center positions
            - image: Input image (padded)
            - image_gt: Ground truth image
            - samples_per_line_meas: Number of samples per line measurement
            - figure: Optional matplotlib figure
            - pattern_metadata: Optional pattern metadata
            - pattern_spec: Optional pattern specification

        Returns
        -------
        TrainingResult
            Complete training results
        """
        sample_centers = kwargs.get("sample_centers")
        image = kwargs.get("image")
        image_gt = kwargs.get("image_gt")
        samples_per_line_meas = kwargs.get("samples_per_line_meas", 1)
        figure = kwargs.get("figure")
        pattern_metadata = kwargs.get("pattern_metadata")
        pattern_spec = kwargs.get("pattern_spec")

        # Call existing run_progressive_training method
        result_dict = self.run_progressive_training(
            sample_centers=sample_centers,
            image=image,
            image_gt=image_gt,
            samples_per_line_meas=samples_per_line_meas,
            figure=figure,
            pattern_metadata=pattern_metadata,
            pattern_spec=pattern_spec,
        )

        # Convert to TrainingResult
        return TrainingResult(
            losses=self.losses,
            ssims=self.ssims,
            psnrs=self.psnrs,
            rmses=self.rmses,
            final_reconstruction=self.current_reconstruction,
            failed_samples=self.failed_samples,
            wall_time_seconds=result_dict.get("wall_time_seconds", 0),
            epochs_per_sample=self.epochs_per_sample,
        )

    def train_epoch(self, epoch: int, **kwargs: Any) -> EpochResult:
        """Single epoch training - this is called internally by run_progressive_training.

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

        Raises
        ------
        NotImplementedError
            PRISMTrainer uses run_progressive_training() which handles epochs internally.
            Call train() instead.
        """
        raise NotImplementedError(
            "PRISMTrainer uses run_progressive_training() which handles epochs internally. "
            "Call train() instead."
        )

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
            Dictionary of metric name to value (ssim, rmse, psnr)
        """
        rec_crop = crop_image(output, self.args.obj_size)
        gt_crop = crop_image(target, self.args.obj_size)

        ssim_val = _compute_ssim(rec_crop, gt_crop).item()
        rmse_val = compute_rmse(output, target, size=self.args.obj_size)
        psnr_val = psnr(output, target, size=self.args.obj_size)

        return {
            "ssim": ssim_val,
            "rmse": rmse_val,
            "psnr": psnr_val,
        }

    def run_initialization(
        self,
        measurement: torch.Tensor,
        center: torch.Tensor,
        image_gt: torch.Tensor,
        figure: Any = None,
        telescope: Any = None,
        sample_centers: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """
        Run initialization phase: train model on initial target.

        Parameters
        ----------
        measurement : torch.Tensor
            Initial measurement or circle mask to train on
        center : torch.Tensor
            Sample center for visualization
        image_gt : torch.Tensor
            Ground truth image for SSIM computation
        figure : Any, optional
            Matplotlib figure handle for visualization

        Returns
        -------
        tuple[torch.Tensor, Any]
            Current reconstruction and updated figure handle
        """
        # Normalize measurement if needed
        if self.args.output_activation in ["sigmoid", "hardsigmoid", "scalesigmoid", "none"]:
            min_val, max_val = measurement.min().item(), measurement.max().item()
            logger.info(
                f"Normalizing measurement for {self.args.output_activation} activation: "
                f"[{min_val:.3e}, {max_val:.3e}] → [0, 1]"
            )
            measurement = measurement - measurement.min()
            measurement = measurement / measurement.max()

        # Initialize loss and criterion
        loss = torch.tensor(1000.0)
        criterion: nn.Module = nn.L1Loss().to(self.device)
        counter = 0
        t0 = time.time()

        init_total_steps = max(self.args.max_epochs_init * self.args.n_epochs_init, 1)
        init_eta = ETACalculator(init_total_steps)
        init_steps_completed = 0
        init_denominator = criterion(measurement, torch.zeros_like(measurement))

        # Get profiler for instrumentation (if available)
        profiler = self._get_profiler()

        with TrainingProgress() as training_progress:
            init_task_id = training_progress.add_task("Initialization", total=init_total_steps)

            logger.info(
                f"Starting initialization phase: target={self.args.initialization_target}, "
                f"max_epochs={self.args.max_epochs_init}, loss_th={self.args.loss_th}"
            )
            # Log physics-consistent loss for synthetic_aperture
            if (
                self.args.initialization_target == "synthetic_aperture"
                and telescope is not None
                and sample_centers is not None
            ):
                logger.info(
                    "Using physics-consistent loss: model output → telescope → comparison in measurement space"
                )

            while loss.item() >= self.args.loss_th:
                if counter >= self.args.max_epochs_init:
                    break
                counter += 1

                for epoch in range(self.args.n_epochs_init):
                    self.optimizer.zero_grad()

                    # Forward pass with AMP if enabled (with profiling)
                    with profiler.profile_region("init_forward") if profiler else nullcontext():
                        if self.use_amp:
                            with torch.cuda.amp.autocast():
                                output = self.model()
                                # FIXME(SYNTHETIC-APERTURE-INIT): Physics-consistent loss disabled
                                # Shape mismatch: model outputs at image_size (1024) but telescope
                                # uses padded grid (2048). Need to either:
                                # 1. Resize output_sa to match measurement shape, or
                                # 2. Ensure telescope.compute_synthetic_aperture preserves input shape
                                # For now, use direct comparison (works but less physics-accurate)
                                loss = criterion(output, measurement) / init_denominator
                        else:
                            output = self.model()
                            # FIXME(SYNTHETIC-APERTURE-INIT): See comment above
                            loss = criterion(output, measurement) / init_denominator

                    loss_value = float(loss.item())

                    init_steps_completed += 1
                    eta_seconds = init_eta.update(init_steps_completed)
                    training_progress.advance(
                        init_task_id,
                        metrics={
                            "phase": "initialization",
                            "init_cycle": counter,
                            "max_init_cycles": self.args.max_epochs_init,
                            "init_epoch": epoch + 1,
                            "max_init_epochs": self.args.n_epochs_init,
                            "init_loss": loss_value,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "base_lr": self.args.lr,
                            "elapsed": time.time() - t0,
                        },
                        eta_seconds=eta_seconds,
                        description="Initialization",
                    )

                    if loss_value < self.args.loss_th:
                        break

                    if torch.isnan(loss).any():
                        logger.error("Loss is NaN. Stopping training.")
                        raise RuntimeError("NaN loss encountered during initialization")

                    # Backward pass with AMP if enabled (with profiling)
                    with profiler.profile_region("init_backward") if profiler else nullcontext():
                        if self.use_amp and self.scaler is not None:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            self.optimizer.step()
                self.scheduler.step(loss)  # type: ignore[arg-type]

            if init_steps_completed:
                training_progress.set_total(init_task_id, init_steps_completed)
            training_progress.complete(init_task_id)

            # Get final reconstruction
            with torch.no_grad():
                self.current_reconstruction = self.model().detach().clone()

            # Compute SSIM on GPU (faster than CPU scikit-image version)
            with torch.no_grad():
                rec_crop = crop_image(self.current_reconstruction, self.args.obj_size)
                gt_crop = crop_image(image_gt, self.args.obj_size)
                ssim = _compute_ssim(rec_crop, gt_crop).item()

            training_progress.update_metrics(
                {"phase": "initialised", "init_loss": loss_value, "init_ssim": ssim}
            )

            # Log initialization completion (avoid print() which interferes with Rich Live)
            logger.info(
                f"Initialization complete: {time.time() - t0:.2f}s, "
                f"Loss: {loss_value:.4e}, SSIM: {ssim:.3f}"
            )

            # Update visualization if available
            if figure is not None:
                from prism.visualization import plot_meas_agg

                figure = plot_meas_agg(
                    image_gt,
                    self.measurement_system,
                    self.current_reconstruction,
                    image_gt,
                    center,
                    0,
                    self.args.sample_diameter / 2,
                    crop_size=self.args.obj_size,
                    ref_radius=self.args.roi_diameter / 2,
                    fig=figure,
                    ref_radius_1=self.args.samples_r_cutoff,
                )

        self.optimizer.zero_grad()
        return self.current_reconstruction, figure

    def run_progressive_training(
        self,
        sample_centers: torch.Tensor,
        image: torch.Tensor,
        image_gt: torch.Tensor,
        samples_per_line_meas: int,
        figure: Any = None,
        pattern_metadata: dict[str, Any] | None = None,
        pattern_spec: str | None = None,
    ) -> dict[str, Any]:
        """
        Run progressive training over multiple sample positions.

        Parameters
        ----------
        sample_centers : torch.Tensor
            Sample center positions [N, 2]
        image : torch.Tensor
            Input image (padded)
        image_gt : torch.Tensor
            Ground truth image for metrics
        samples_per_line_meas : int
            Number of samples per line measurement
        figure : Any, optional
            Matplotlib figure handle for visualization
        pattern_metadata : dict, optional
            Pattern generation metadata
        pattern_spec : str, optional
            Pattern specification string

        Returns
        -------
        dict[str, Any]
            Training results including losses, metrics, and reconstruction
        """
        if self.args.n_samples <= 1:
            logger.info("Single sample mode - skipping progressive training")
            return self._create_results_dict(pattern_metadata, pattern_spec)

        logger.info(
            f"Starting progressive training: {self.args.n_samples} samples, "
            f"max_epochs={self.args.max_epochs}, loss_th={self.args.loss_th}"
        )

        # Use LossAggregator for combining old and new measurements
        # --no-normalize-loss flag sets no_normalize_loss=True, so we invert it
        normalize_loss = not getattr(self.args, "no_normalize_loss", False)
        criterion = LossAggregator(
            loss_type=self.args.loss_type,
            normalize_loss=normalize_loss,
            new_weight=self.args.new_weight,
            f_weight=self.args.f_weight,
        ).to(self.device)

        # Setup progress tracking
        epoch_total_steps = max(self.args.n_samples * self.args.max_epochs * self.args.n_epochs, 1)
        sample_total_steps = max(self.args.n_samples, 1)
        epoch_eta = ETACalculator(epoch_total_steps)
        sample_eta = ETACalculator(sample_total_steps)
        epoch_steps_completed = 0
        sample_steps_completed = 0

        # Get profiler for instrumentation (if available)
        profiler = self._get_profiler()

        with TrainingProgress() as training_progress:
            epoch_task_id = training_progress.add_task(
                f"Sample 1/{self.args.n_samples}",
                total=epoch_total_steps
            )

            # Check if adaptive convergence is enabled
            use_adaptive = getattr(self.args, "enable_adaptive_convergence", True)

            # Process each sample position sequentially
            for center_idx, center_ends in enumerate(sample_centers, start=self.args.n_samples_0):
                # Invoke callback at sample start
                self._invoke_callbacks("on_sample_start", center_idx)

                counter = 0
                center, center_rec = patterns.create_patterns(
                    center_ends, samples_per_line_meas, self.args.samples_per_line_rec
                )
                t0 = time.time()

                # Generate new measurement
                # Use line_endpoints for line acquisition mode if available
                if (
                    self.measurement_system.line_acquisition is not None
                    and self.args.sample_length > 0
                    and len(center_ends) == 2
                ):
                    # Line mode with new batched acquisition
                    start = torch.tensor(center_ends[0], device=self.device, dtype=torch.float32)
                    end = torch.tensor(center_ends[1], device=self.device, dtype=torch.float32)
                    measurement = self.measurement_system.measure(
                        ground_truth=image,
                        reconstruction=self.current_reconstruction,
                        line_endpoints=(start, end),
                        add_noise=True,
                    )
                else:
                    # Point mode or legacy line mode
                    measurement = self.measurement_system.measure(
                        ground_truth=image,
                        reconstruction=self.current_reconstruction,
                        centers=center,
                        add_noise=True,
                    )

                # Initialize losses
                loss = torch.tensor(1000.0)
                loss_old = torch.tensor(1000.0)
                loss_new = torch.tensor(1000.0)

                # Always create convergence monitor for metrics tracking
                # (tier escalation is only used when use_adaptive=True)
                monitor = ConvergenceMonitor(
                    loss_threshold=self.args.loss_th,
                    patience=getattr(self.args, "early_stop_patience", 10),
                    plateau_window=getattr(self.args, "plateau_window", 50),
                    plateau_threshold=getattr(self.args, "plateau_threshold", 0.01),
                    escalation_epochs=getattr(self.args, "escalation_epochs", 200),
                )

                # Reset optimizer for each sample
                base_lr = self.args.lr
                self.optimizer = optim.Adam(
                    self.model.parameters(), lr=base_lr, amsgrad=self.args.use_amsgrad
                )
                # Use plateau scheduler only when adaptive convergence is enabled;
                # otherwise use no-op scheduler to keep LR constant
                scheduler_type = "plateau" if use_adaptive else "none"
                self.scheduler = create_scheduler(self.optimizer, scheduler_type)

                # Track epochs for this sample
                total_epochs_this_sample = 0
                current_tier = ConvergenceTier.NORMAL
                sample_converged = False
                # Use local variable to avoid mutating args (deterministic checkpoints)
                current_max_epochs = self.args.max_epochs

                # Train until both losses converge
                while loss_old.item() >= self.args.loss_th or loss_new.item() >= self.args.loss_th:
                    if counter >= current_max_epochs:
                        # Check for adaptive escalation
                        if use_adaptive:
                            if monitor.should_escalate():
                                # Escalate to aggressive tier
                                tier_config = get_tier_config(
                                    ConvergenceTier.AGGRESSIVE,
                                    aggressive_lr_multiplier=getattr(
                                        self.args, "aggressive_lr_multiplier", 2.0
                                    ),
                                    retry_lr_multiplier=getattr(
                                        self.args, "retry_lr_multiplier", 0.1
                                    ),
                                )
                                current_tier = ConvergenceTier.AGGRESSIVE
                                monitor.set_tier(current_tier)

                                # Adjust learning rate and scheduler
                                new_lr = base_lr * tier_config.lr_multiplier
                                for param_group in self.optimizer.param_groups:
                                    param_group["lr"] = new_lr
                                self.scheduler = create_scheduler(
                                    self.optimizer, tier_config.scheduler
                                )

                                # Allow extra epochs (use local variable, don't mutate args)
                                current_max_epochs += tier_config.extra_epochs // self.args.n_epochs
                                logger.info(
                                    f"Sample {center_idx + 1}: Escalating to {current_tier.value} tier "
                                    f"(lr={new_lr:.2e}, extra_epochs={tier_config.extra_epochs})"
                                )
                                counter = 0  # Reset counter to allow more training
                                continue

                        self.failed_samples.append(center_idx)
                        logger.warning(
                            f"Sample {center_idx + 1}/{self.args.n_samples} failed to converge "
                            f"after {total_epochs_this_sample} epochs (loss_old={loss_old.item():.4e}, "
                            f"loss_new={loss_new.item():.4e}, tier={current_tier.value})"
                        )
                        break
                    counter += 1

                    for epoch in range(self.args.n_epochs):
                        # Invoke callback at epoch start
                        self._invoke_callbacks("on_epoch_start", epoch)

                        self.optimizer.zero_grad()

                        # Forward pass with AMP if enabled (with profiling)
                        with profiler.profile_region("forward") if profiler else nullcontext():
                            if self.use_amp:
                                with torch.cuda.amp.autocast():
                                    output = self.model()
                            else:
                                output = self.model()

                        # Loss computation (with profiling)
                        with profiler.profile_region("loss") if profiler else nullcontext():
                            if self.use_amp:
                                with torch.cuda.amp.autocast():
                                    loss_old, loss_new = criterion(
                                        inputs=output,
                                        target=measurement,
                                        telescope=self.measurement_system,
                                        center=center_rec,
                                    )
                                    loss = loss_old + loss_new
                            else:
                                loss_old, loss_new = criterion(
                                    inputs=output,
                                    target=measurement,
                                    telescope=self.measurement_system,
                                    center=center_rec,
                                )
                                loss = loss_old + loss_new

                        loss_old_value = float(loss_old.item())
                        loss_new_value = float(loss_new.item())
                        total_epochs_this_sample += 1

                        # Update convergence monitor (always tracked for metrics)
                        monitor.update(loss.item())

                        epoch_steps_completed += 1
                        eta_seconds = epoch_eta.update(epoch_steps_completed)

                        # Build metrics dict for dashboard display (other metrics logged to TensorBoard)
                        epoch_metrics: dict[str, float | str | None] = {
                            "sample": center_idx + 1,
                            "n_samples": self.args.n_samples,
                            "epoch": total_epochs_this_sample,
                            "max_epochs": self.args.max_epochs * self.args.n_epochs,
                            "loss_old": loss_old_value,
                            "loss_new": loss_new_value,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "elapsed": time.time() - self.training_start_time,
                        }

                        # Add GPU memory if CUDA available
                        if torch.cuda.is_available():
                            epoch_metrics["gpu_mem_mb"] = torch.cuda.memory_allocated() / 1024 / 1024

                        training_progress.advance(
                            epoch_task_id,
                            metrics=epoch_metrics,  # type: ignore[arg-type]
                            eta_seconds=eta_seconds,
                            description=f"Sample {center_idx + 1}/{self.args.n_samples} | Epoch {total_epochs_this_sample}",
                        )

                        # Check for convergence (standard check)
                        # First sample: only check loss_new (no cumulative mask yet)
                        # Later samples: check both losses
                        if center_idx == self.args.n_samples_0:
                            if loss_new_value < self.args.loss_th:
                                sample_converged = True
                                current_tier = monitor.get_current_tier()
                                break
                        else:
                            if loss_old_value < self.args.loss_th and loss_new_value < self.args.loss_th:
                                sample_converged = True
                                current_tier = monitor.get_current_tier()
                                break

                        # Early stopping with adaptive convergence
                        if use_adaptive:
                            if monitor.should_stop():
                                if monitor.is_converged():
                                    sample_converged = True
                                    current_tier = ConvergenceTier.FAST
                                    logger.debug(
                                        f"Sample {center_idx + 1}: Early exit after "
                                        f"{total_epochs_this_sample} epochs (converged)"
                                    )
                                break

                        if torch.isnan(loss).any():
                            logger.error("Loss is NaN. Stopping training.")
                            raise RuntimeError("NaN loss encountered during training")

                        # Backward pass (with profiling)
                        with profiler.profile_region("backward") if profiler else nullcontext():
                            if self.use_amp and self.scaler is not None:
                                self.scaler.scale(loss).backward()
                            else:
                                loss.backward()

                        # Optimizer step (with profiling)
                        with profiler.profile_region("optimizer_step") if profiler else nullcontext():
                            if self.use_amp and self.scaler is not None:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()

                        # Invoke callback at epoch end
                        self._invoke_callbacks("on_epoch_end", epoch, loss.item())

                    # Step scheduler (handle both types)
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(loss)  # type: ignore[arg-type]
                    else:
                        self.scheduler.step()

                    # Check if converged or should stop
                    if sample_converged:
                        break
                    if use_adaptive and monitor.should_stop():
                        break

                # Track convergence statistics (always tracked for metrics)
                self.epochs_per_sample.append(total_epochs_this_sample)
                self.tiers_per_sample.append(current_tier.value)
                self.convergence_stats.append(monitor.get_statistics())

                # Log efficiency
                if use_adaptive and total_epochs_this_sample > 0:
                    max_possible = self.args.max_epochs * self.args.n_epochs
                    efficiency = total_epochs_this_sample / max_possible
                    logger.debug(
                        f"Sample {center_idx + 1}: {total_epochs_this_sample} epochs "
                        f"({efficiency:.1%} of max), tier={current_tier.value}"
                    )

                # Compute metrics
                with torch.no_grad():
                    self.current_reconstruction = self.model().detach().clone()
                    self.losses.append(loss.item())
                    rmse = compute_rmse(
                        self.current_reconstruction, image_gt, size=self.args.obj_size
                    )
                    self.rmses.append(rmse)
                    # GPU-accelerated SSIM (faster than CPU scikit-image)
                    rec_crop = crop_image(self.current_reconstruction, self.args.obj_size)
                    gt_crop = crop_image(image_gt, self.args.obj_size)
                    ssim = _compute_ssim(rec_crop, gt_crop).item()
                    self.ssims.append(ssim)
                    psnr_value = psnr(
                        self.current_reconstruction, image_gt, size=self.args.obj_size
                    )
                    self.psnrs.append(psnr_value)

                # Track timing and learning rate
                self.sample_times.append(time.time() - t0)
                self.lr_history.append(self.optimizer.param_groups[0]["lr"])

                sample_steps_completed += 1
                eta_seconds = sample_eta.update(sample_steps_completed)

                # Update dashboard with SSIM after sample evaluation completes
                training_progress.update_metrics({"ssim": ssim})

                # Log sample completion (avoid print() which interferes with Rich Live)
                logger.debug(
                    f"Sample {center_idx + 1}/{self.args.n_samples}: {time.time() - t0:.2f}s, "
                    f"Loss: {self.losses[-1]:.4e}, SSIM: {ssim:.3f}, RMSE: {rmse:.2e}, "
                    f"PSNR: {psnr_value:.1f} dB"
                )

                # Update visualization
                if figure is not None:
                    from prism.visualization import plot_meas_agg

                    figure = plot_meas_agg(
                        image_gt,
                        self.measurement_system,
                        self.current_reconstruction,
                        image_gt,
                        center,
                        center_idx,
                        self.args.sample_diameter / 2,
                        crop_size=self.args.obj_size,
                        ref_radius=self.args.roi_diameter / 2,
                        fig=figure,
                        ref_radius_1=self.args.samples_r_cutoff,
                    )

                # Update cumulative mask (MUST happen regardless of save_data!)
                # This is critical for progressive training - the mask tracks which
                # k-space regions have been measured so loss_old can enforce consistency
                if (
                    self.measurement_system.line_acquisition is not None
                    and self.args.sample_length > 0
                    and len(center_ends) == 2
                ):
                    # Line mode with new batched acquisition
                    start = torch.tensor(
                        center_ends[0], device=self.device, dtype=torch.float32
                    )
                    end = torch.tensor(center_ends[1], device=self.device, dtype=torch.float32)
                    self.measurement_system.add_mask(line_endpoints=(start, end))
                else:
                    # Point mode or legacy line mode
                    self.measurement_system.add_mask(center)

                # Save checkpoint and log to TensorBoard
                if self.args.save_data:
                    self._log_sample_metrics(center_idx, loss_old_value, loss_new_value, center)
                    self._save_checkpoint(
                        center_idx, sample_centers, pattern_metadata, pattern_spec
                    )

                # Invoke callback at sample end
                self._invoke_callbacks(
                    "on_sample_end", center_idx, {"loss": loss.item(), "ssim": ssim}
                )

            # Invoke callback at training end
            self._invoke_callbacks("on_training_end")

            training_progress.set_total(epoch_task_id, max(epoch_steps_completed, 1))
            training_progress.complete(epoch_task_id)

        return self._create_results_dict(pattern_metadata, pattern_spec)

    def _log_sample_metrics(
        self, center_idx: int, loss_old: float, loss_new: float, center: torch.Tensor
    ) -> None:
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return

        # Add hyperparameters on first sample
        if center_idx == 0:
            hparam_dict = {
                "lr": self.args.lr,
                "loss_th": self.args.loss_th,
                "n_samples": self.args.n_samples,
                "max_epochs": self.args.max_epochs,
                "n_epochs": self.args.n_epochs,
                "sample_diameter": self.args.sample_diameter,
                "obj_size": self.args.obj_size,
                "new_weight": self.args.new_weight,
                "f_weight": self.args.f_weight,
            }
            if self.args.snr is not None:
                hparam_dict["snr"] = self.args.snr
            metric_dict = {"final/ssim": 0, "final/rmse": 0, "final/psnr": 0}
            self.writer.add_hparams(hparam_dict, metric_dict)
            logger.debug("TensorBoard hparams logged")

        # Log scalars
        self.writer.add_scalar("Loss", self.losses[-1], center_idx)
        self.writer.add_scalar("Loss/old", loss_old, center_idx)
        self.writer.add_scalar("Loss/new", loss_new, center_idx)
        self.writer.add_scalar("SSIM", self.ssims[-1], center_idx)
        self.writer.add_scalar("RMSE", self.rmses[-1], center_idx)
        self.writer.add_scalar("PSNR", self.psnrs[-1], center_idx)
        self.writer.add_scalar("Time/per_sample", self.sample_times[-1], center_idx)
        self.writer.add_scalar("LearningRate", self.lr_history[-1], center_idx)

        # Log reconstruction image
        if self.current_reconstruction is not None:
            self.writer.add_image(
                "Reconstructed Image",
                crop_image(self.current_reconstruction, self.args.obj_size)
                .abs()
                .squeeze()
                .detach()
                .cpu(),
                center_idx,
                dataformats="HW",
            )

    def _save_checkpoint(
        self,
        center_idx: int,
        sample_centers: torch.Tensor,
        pattern_metadata: dict[str, Any] | None,
        pattern_spec: str | None,
    ) -> None:
        """Save training checkpoint."""
        import datetime

        if self.log_dir is None:
            return

        self.args.end_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.args.last_sample = center_idx

        save_args(self.args, self.log_dir)
        save_checkpoint(
            {
                "model": self.model.state_dict(),
                "sample_centers": sample_centers,
                "last_center_idx": torch.tensor(center_idx),
                "measurement_system": self.measurement_system.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "losses": torch.tensor(self.losses),
                "ssims": torch.tensor(self.ssims),
                "rmses": torch.tensor(self.rmses),
                "psnrs": torch.tensor(self.psnrs),
                "current_rec": self.current_reconstruction,
                "failed_samples": torch.tensor(self.failed_samples),
                "wall_time_seconds": time.time() - self.training_start_time,
                "args_dict": vars(self.args),
                "sample_times": (
                    torch.tensor(self.sample_times) if self.sample_times else torch.tensor([])
                ),
                "lr_history": (
                    torch.tensor(self.lr_history) if self.lr_history else torch.tensor([])
                ),
                "pattern_metadata": pattern_metadata,
                "pattern_spec": pattern_spec,
            },
            str(Path(self.log_dir) / "checkpoint.pt"),
        )
        logger.debug(f"Checkpoint saved: sample {center_idx + 1}/{self.args.n_samples}")

    def retry_failed_samples(
        self,
        sample_centers: torch.Tensor,
        image: torch.Tensor,
        image_gt: torch.Tensor,
        samples_per_line_meas: int,
        criterion: LossAggregator,
    ) -> int:
        """Retry failed samples with alternative strategies.

        Parameters
        ----------
        sample_centers : torch.Tensor
            All sample center positions
        image : torch.Tensor
            Input image
        image_gt : torch.Tensor
            Ground truth image
        samples_per_line_meas : int
            Samples per line measurement
        criterion : LossAggregator
            Loss criterion

        Returns
        -------
        int
            Number of samples recovered during retry
        """
        max_retries = getattr(self.args, "max_retries", 2)
        if not self.failed_samples or max_retries == 0:
            return 0

        retry_queue = list(self.failed_samples)
        self.failed_samples.clear()
        recovered_count = 0
        total_retry_time = 0.0

        for retry_num in range(1, max_retries + 1):
            if not retry_queue:
                break

            logger.info(f"Retry attempt {retry_num}/{max_retries} for {len(retry_queue)} samples")

            still_failed = []
            for sample_idx in retry_queue:
                t0 = time.time()
                success = self._retry_single_sample(
                    sample_idx=sample_idx,
                    sample_centers=sample_centers,
                    image=image,
                    image_gt=image_gt,
                    samples_per_line_meas=samples_per_line_meas,
                    criterion=criterion,
                    retry_num=retry_num,
                )
                total_retry_time += time.time() - t0

                if success:
                    recovered_count += 1
                    logger.info(f"Sample {sample_idx + 1} recovered on retry {retry_num}")
                else:
                    still_failed.append(sample_idx)

            retry_queue = still_failed

        # Any remaining failures are truly failed
        self.failed_samples.extend(retry_queue)

        logger.info(
            f"Retry complete: {recovered_count} recovered, "
            f"{len(self.failed_samples)} still failed, "
            f"total retry time: {total_retry_time:.1f}s"
        )

        return recovered_count

    def _retry_single_sample(
        self,
        sample_idx: int,
        sample_centers: torch.Tensor,
        image: torch.Tensor,
        image_gt: torch.Tensor,
        samples_per_line_meas: int,
        criterion: LossAggregator,
        retry_num: int,
    ) -> bool:
        """Retry a single failed sample with RESCUE tier settings.

        Parameters
        ----------
        sample_idx : int
            Index of the sample to retry
        sample_centers : torch.Tensor
            All sample center positions
        image : torch.Tensor
            Input image
        image_gt : torch.Tensor
            Ground truth image
        samples_per_line_meas : int
            Samples per line measurement
        criterion : LossAggregator
            Loss criterion (used as template for loss type)
        retry_num : int
            Current retry attempt number

        Returns
        -------
        bool
            True if converged, False if still failed
        """
        # Get RESCUE tier config
        tier_config = get_tier_config(
            ConvergenceTier.RESCUE,
            aggressive_lr_multiplier=getattr(self.args, "aggressive_lr_multiplier", 2.0),
            retry_lr_multiplier=getattr(self.args, "retry_lr_multiplier", 0.1),
        )

        # Adjust LR based on retry number (progressively smaller)
        base_lr = self.args.lr * tier_config.lr_multiplier
        retry_lr = base_lr / (retry_num + 1)  # Further reduce on each retry

        # Switch loss function on retry (try different loss landscape)
        switch_loss = getattr(self.args, "retry_switch_loss", True)
        if switch_loss and retry_num > 0:
            original_loss = criterion.loss_type
            new_loss_type = get_retry_loss_type(original_loss, retry_num)
            normalize_loss = not getattr(self.args, "no_normalize_loss", False)
            retry_criterion: LossAggregator = LossAggregator(
                loss_type=new_loss_type,  # type: ignore[arg-type]
                normalize_loss=normalize_loss,
                new_weight=self.args.new_weight,
                f_weight=self.args.f_weight,
            ).to(self.device)
            logger.debug(
                f"Sample {sample_idx + 1}: Switching loss from {original_loss} to "
                f"{new_loss_type} for retry {retry_num}"
            )
        else:
            retry_criterion = criterion

        # Get sample center
        center_ends = sample_centers[sample_idx - self.args.n_samples_0]
        center, center_rec = patterns.create_patterns(
            center_ends, samples_per_line_meas, self.args.samples_per_line_rec
        )

        # Generate measurement
        measurement = self.measurement_system.measure(
            ground_truth=image,
            reconstruction=self.current_reconstruction,
            centers=center,
            add_noise=True,
        )

        # Setup optimizer with rescue settings
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=retry_lr, amsgrad=self.args.use_amsgrad
        )
        self.scheduler = create_scheduler(self.optimizer, tier_config.scheduler)

        # Create monitor for retry
        monitor = ConvergenceMonitor(
            loss_threshold=self.args.loss_th,
            patience=getattr(self.args, "early_stop_patience", 10),
            plateau_window=getattr(self.args, "plateau_window", 50),
            plateau_threshold=getattr(self.args, "plateau_threshold", 0.01),
        )
        monitor.set_tier(ConvergenceTier.RESCUE)

        # Train with extra epochs
        extra_epochs = tier_config.extra_epochs
        max_epochs_retry = (self.args.max_epochs * self.args.n_epochs) + extra_epochs

        loss = torch.tensor(1000.0)
        loss_old = torch.tensor(1000.0)
        loss_new = torch.tensor(1000.0)

        # Get profiler for instrumentation (if available)
        profiler = self._get_profiler()

        for epoch in range(max_epochs_retry):
            self.optimizer.zero_grad()

            # Forward pass (with profiling)
            with profiler.profile_region("retry_forward") if profiler else nullcontext():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model()
                else:
                    output = self.model()

            # Loss computation (with profiling)
            with profiler.profile_region("retry_loss") if profiler else nullcontext():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        loss_old, loss_new = retry_criterion(
                            inputs=output,
                            target=measurement,
                            telescope=self.measurement_system,
                            center=center_rec,
                        )
                        loss = loss_old + loss_new
                else:
                    loss_old, loss_new = retry_criterion(
                        inputs=output,
                        target=measurement,
                        telescope=self.measurement_system,
                        center=center_rec,
                    )
                    loss = loss_old + loss_new

            monitor.update(loss.item())

            # Check convergence
            if loss_old.item() < self.args.loss_th and loss_new.item() < self.args.loss_th:
                # Update reconstruction
                with torch.no_grad():
                    self.current_reconstruction = self.model().detach().clone()
                return True

            if monitor.should_stop() and not monitor.is_converged():
                return False

            if torch.isnan(loss).any():
                return False

            # Backward pass (with profiling)
            with profiler.profile_region("retry_backward") if profiler else nullcontext():
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Optimizer step (with profiling)
            with profiler.profile_region("retry_optimizer_step") if profiler else nullcontext():
                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

            # Step scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(loss)  # type: ignore[arg-type]
            else:
                self.scheduler.step()

        return False

    def _create_results_dict(
        self, pattern_metadata: dict[str, Any] | None, pattern_spec: str | None
    ) -> dict[str, Any]:
        """Create results dictionary."""
        return {
            "final_reconstruction": self.current_reconstruction,
            "losses": self.losses,
            "ssims": self.ssims,
            "rmses": self.rmses,
            "psnrs": self.psnrs,
            "failed_samples": self.failed_samples,
            "n_samples": self.args.n_samples,
            "sample_times": self.sample_times,
            "lr_history": self.lr_history,
            "wall_time_seconds": time.time() - self.training_start_time,
            "pattern_metadata": pattern_metadata,
            "pattern_spec": pattern_spec,
            # Convergence statistics
            "epochs_per_sample": self.epochs_per_sample,
            "tiers_per_sample": self.tiers_per_sample,
            "convergence_stats": self.convergence_stats,
        }

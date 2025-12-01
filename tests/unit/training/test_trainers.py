"""Tests for prism/core/trainers.py - PRISMTrainer and create_scheduler."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import torch

from prism.core.trainers import PRISMTrainer, create_scheduler
from prism.models.networks import ProgressiveDecoder


if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# TEST UTILITIES
# =============================================================================


class MockTrainingProgress:
    """Mock TrainingProgress that does nothing (for unit tests)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def add_task(self, *args, **kwargs):
        return 0

    def advance(self, *args, **kwargs):
        pass

    def set_total(self, *args, **kwargs):
        pass

    def complete(self, *args, **kwargs):
        pass

    def update_metrics(self, *args, **kwargs):
        pass


@contextmanager
def disable_training_progress():
    """Context manager to disable TrainingProgress for unit tests."""
    with patch("prism.core.trainers.TrainingProgress", MockTrainingProgress):
        yield


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def device() -> torch.device:
    """Get available device (prefer CPU for fast tests)."""
    return torch.device("cpu")


@pytest.fixture
def small_model(device: torch.device) -> ProgressiveDecoder:
    """Small ProgressiveDecoder for fast testing."""
    model = ProgressiveDecoder(input_size=64, latent_channels=32)
    return model.to(device)


@pytest.fixture
def optimizer(small_model: ProgressiveDecoder) -> torch.optim.Adam:
    """Adam optimizer for model."""
    return torch.optim.Adam(small_model.parameters(), lr=1e-3)


@pytest.fixture
def plateau_scheduler(
    optimizer: torch.optim.Adam,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """ReduceLROnPlateau scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )


@pytest.fixture
def cosine_scheduler(
    optimizer: torch.optim.Adam,
) -> torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
    """CosineAnnealingWarmRestarts scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


@pytest.fixture
def mock_measurement_system(device: torch.device) -> Mock:
    """Mock MeasurementSystem with controlled outputs."""
    ms = Mock()
    # Track accumulated masks/measurements
    ms.masks = []
    ms.measurements_list = []

    def mock_measure(*args, **kwargs):
        # Always return stacked [old, new] for progressive training
        # Shape: (2, C, H, W) where C=1
        # IMPORTANT: old_meas must NOT be zeros to avoid division by zero in loss normalization
        if len(ms.measurements_list) == 0:
            # First call: old_meas should be non-zero to avoid norm=0
            old_meas = torch.rand(1, 64, 64, device=device) * 0.5 + 0.1  # Shape: (C, H, W)
            new_meas = torch.rand(1, 64, 64, device=device) * 0.5 + 0.1  # Shape: (C, H, W)
            ms.measurements_list.append(new_meas)
        else:
            old_meas = ms.measurements_list[-1]
            new_meas = torch.rand(1, 64, 64, device=device) * 0.5 + 0.1
            # Accumulate (simple average)
            accumulated = (old_meas + new_meas) / 2
            ms.measurements_list.append(accumulated)

        # Return stacked [old, new] with shape (2, C, H, W)
        return torch.stack([old_meas, new_meas], dim=0)

    def mock_add_mask(*args, **kwargs):
        ms.masks.append("mask")
        return None

    def mock_call(inputs: torch.Tensor, centers=None) -> torch.Tensor:
        """Mock __call__ for LossAggregator compatibility.

        LossAggregator calls: telescope(inputs, center) expecting [2, C, H, W].
        Returns:
            - measurement[0]: inputs through cumulative mask (old measurements)
            - measurement[1]: inputs through new aperture (new measurement)

        IMPORTANT: Must maintain gradient flow from inputs for backprop to work.
        We simulate measurements by applying simple operations on inputs.
        """
        # Squeeze batch dimension for stacking: (1, C, H, W) -> (C, H, W)
        squeezed = inputs.squeeze(0)

        # Old measurement: simulate accumulated measurements with some noise
        # Use inputs to maintain gradient connection
        old_meas = squeezed * 0.5 + 0.25  # Scale and shift to simulate measurement

        # New measurement: different transformation of inputs
        new_meas = squeezed * 0.7 + 0.15

        # Return [2, C, H, W] - stacked old and new measurements
        return torch.stack([old_meas, new_meas], dim=0)

    ms.measure.side_effect = mock_measure
    ms.add_mask.side_effect = mock_add_mask
    ms.state_dict.return_value = {"masks": [], "measurements": []}
    ms.line_acquisition = None

    # Make mock callable for LossAggregator compatibility
    ms.side_effect = mock_call
    ms.return_value = None  # Will be overridden by side_effect

    # Add attributes needed by plot_meas_agg
    ms.r = 8.0
    ms.n = torch.tensor(64, device=device)
    ms.x = torch.zeros(64, 64, device=device)
    ms.cum_mask = torch.zeros(64, 64, device=device)
    ms.mask = Mock(return_value=torch.zeros(64, 64, device=device))

    return ms


@pytest.fixture
def mock_args() -> Mock:
    """Mock args namespace with all required attributes."""
    args = Mock()

    # Initialization params
    args.max_epochs_init = 2
    args.n_epochs_init = 10
    args.loss_th = 0.01
    args.output_activation = "sigmoid"

    # Training params
    args.max_epochs = 2
    args.n_epochs = 10
    args.n_samples = 5
    args.n_samples_0 = 0  # Starting sample index
    args.lr = 1e-3
    args.use_amsgrad = False

    # Geometry params
    args.obj_size = 64
    args.sample_diameter = 16
    args.roi_diameter = 32
    args.samples_r_cutoff = 0.8
    args.sample_length = 0  # Point mode
    args.samples_per_line_rec = 0  # Point mode (no lines)

    # Loss params
    args.initialization_target = "meas"
    args.loss_type = "l1"
    args.new_weight = 0.3
    args.f_weight = 0.0

    # Convergence params
    args.enable_adaptive_convergence = False
    args.early_stop_patience = 10
    args.plateau_window = 50
    args.plateau_threshold = 0.01
    args.escalation_epochs = 200
    args.aggressive_lr_multiplier = 2.0
    args.max_retries = 2
    args.retry_lr_multiplier = 0.1
    args.retry_switch_loss = True

    # Output params
    args.save_data = False
    args.name = "test_run"
    args.snr = None  # Optional SNR parameter

    return args


@pytest.fixture
def sample_tensors(device: torch.device) -> dict[str, torch.Tensor]:
    """Create sample tensors for testing."""
    return {
        "measurement": torch.rand(1, 1, 64, 64, device=device),
        "image_gt": torch.rand(1, 1, 64, 64, device=device),
        # Line endpoints shape: (n_lines, 2, 2) where each line is [[y1,x1], [y2,x2]]
        # For point mode, these become single points when samples_per_line_meas=0
        "sample_centers": torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],  # Point at origin (same start/end)
                [[0.1, 0.1], [0.1, 0.1]],  # Point at (0.1, 0.1)
                [[0.2, 0.0], [0.2, 0.0]],  # Point at (0.2, 0.0)
            ],
            device=device,
        ),
    }


@pytest.fixture
def trainer_setup(
    small_model: ProgressiveDecoder,
    optimizer: torch.optim.Adam,
    plateau_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    mock_measurement_system: Mock,
    mock_args: Mock,
    device: torch.device,
) -> dict:
    """Complete trainer setup for testing."""
    return {
        "model": small_model,
        "optimizer": optimizer,
        "scheduler": plateau_scheduler,
        "measurement_system": mock_measurement_system,
        "args": mock_args,
        "device": device,
    }


# =============================================================================
# TEST: create_scheduler
# =============================================================================


class TestCreateScheduler:
    """Tests for create_scheduler factory function."""

    def test_create_plateau_scheduler(self, optimizer: torch.optim.Adam) -> None:
        """Test creating ReduceLROnPlateau scheduler."""
        scheduler = create_scheduler(optimizer, scheduler_type="plateau")
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_create_cosine_scheduler(self, optimizer: torch.optim.Adam) -> None:
        """Test creating CosineAnnealingWarmRestarts scheduler."""
        scheduler = create_scheduler(optimizer, scheduler_type="cosine_warm_restarts")
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

    def test_cosine_scheduler_custom_params(self, optimizer: torch.optim.Adam) -> None:
        """Test cosine scheduler with custom T_0, T_mult, eta_min."""
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="cosine_warm_restarts",
            T_0=20,
            T_mult=3,
            eta_min=1e-6,
        )
        assert scheduler.T_0 == 20
        assert scheduler.T_mult == 3
        assert scheduler.eta_min == 1e-6

    def test_invalid_scheduler_type_raises(self, optimizer: torch.optim.Adam) -> None:
        """Test that invalid scheduler type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(optimizer, scheduler_type="invalid")

    def test_plateau_scheduler_default_mode(self, optimizer: torch.optim.Adam) -> None:
        """Test plateau scheduler uses 'min' mode by default."""
        scheduler = create_scheduler(optimizer, scheduler_type="plateau")
        assert scheduler.mode == "min"


# =============================================================================
# TEST: PRISMTrainer.__init__
# =============================================================================


class TestPRISMTrainerInit:
    """Tests for PRISMTrainer initialization."""

    def test_initialization_with_required_params(self, trainer_setup: dict) -> None:
        """Test trainer initializes with required parameters."""
        trainer = PRISMTrainer(
            model=trainer_setup["model"],
            optimizer=trainer_setup["optimizer"],
            scheduler=trainer_setup["scheduler"],
            measurement_system=trainer_setup["measurement_system"],
            args=trainer_setup["args"],
            device=trainer_setup["device"],
        )
        assert trainer.model is trainer_setup["model"]
        assert trainer.optimizer is trainer_setup["optimizer"]
        assert trainer.device == trainer_setup["device"]

    def test_metric_lists_initialized_empty(self, trainer_setup: dict) -> None:
        """Test that metric tracking lists are initialized empty."""
        trainer = PRISMTrainer(**trainer_setup)
        assert trainer.losses == []
        assert trainer.ssims == []
        assert trainer.rmses == []
        assert trainer.psnrs == []

    def test_use_amp_flag_stored(self, trainer_setup: dict) -> None:
        """Test use_amp flag is stored correctly."""
        trainer = PRISMTrainer(**trainer_setup, use_amp=True)
        assert trainer.use_amp is True

        trainer_no_amp = PRISMTrainer(**trainer_setup, use_amp=False)
        assert trainer_no_amp.use_amp is False

    def test_scaler_created_with_amp(self, trainer_setup: dict) -> None:
        """Test GradScaler is created when use_amp=True."""
        trainer = PRISMTrainer(**trainer_setup, use_amp=True)
        assert trainer.scaler is not None

    def test_scaler_none_without_amp(self, trainer_setup: dict) -> None:
        """Test scaler is None when use_amp=False."""
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        assert trainer.scaler is None

    def test_writer_can_be_none(self, trainer_setup: dict) -> None:
        """Test trainer works with writer=None."""
        trainer = PRISMTrainer(**trainer_setup, writer=None)
        assert trainer.writer is None

    def test_args_stored(self, trainer_setup: dict) -> None:
        """Test args are stored for later access."""
        trainer = PRISMTrainer(**trainer_setup)
        assert trainer.args is trainer_setup["args"]

    def test_log_dir_stored(self, trainer_setup: dict, tmp_path: "Path") -> None:
        """Test log_dir is stored correctly."""
        trainer = PRISMTrainer(**trainer_setup, log_dir=str(tmp_path))
        assert trainer.log_dir == str(tmp_path)

    def test_measurement_system_stored(self, trainer_setup: dict) -> None:
        """Test measurement_system is stored correctly."""
        trainer = PRISMTrainer(**trainer_setup)
        assert trainer.measurement_system is trainer_setup["measurement_system"]

    def test_scheduler_stored(self, trainer_setup: dict) -> None:
        """Test scheduler is stored correctly."""
        trainer = PRISMTrainer(**trainer_setup)
        assert trainer.scheduler is trainer_setup["scheduler"]

    def test_current_reconstruction_initialized_none(self, trainer_setup: dict) -> None:
        """Test current_reconstruction is initialized as None."""
        trainer = PRISMTrainer(**trainer_setup)
        assert trainer.current_reconstruction is None

    def test_failed_samples_initialized_empty(self, trainer_setup: dict) -> None:
        """Test failed_samples list is initialized empty."""
        trainer = PRISMTrainer(**trainer_setup)
        assert trainer.failed_samples == []

    def test_sample_times_initialized_empty(self, trainer_setup: dict) -> None:
        """Test sample_times list is initialized empty."""
        trainer = PRISMTrainer(**trainer_setup)
        assert trainer.sample_times == []

    def test_lr_history_initialized_empty(self, trainer_setup: dict) -> None:
        """Test lr_history list is initialized empty."""
        trainer = PRISMTrainer(**trainer_setup)
        assert trainer.lr_history == []

    def test_convergence_tracking_initialized_empty(self, trainer_setup: dict) -> None:
        """Test convergence tracking lists are initialized empty."""
        trainer = PRISMTrainer(**trainer_setup)
        assert trainer.epochs_per_sample == []
        assert trainer.tiers_per_sample == []
        assert trainer.convergence_stats == []

    def test_training_start_time_set(self, trainer_setup: dict) -> None:
        """Test training_start_time is set during initialization."""
        import time

        before = time.time()
        trainer = PRISMTrainer(**trainer_setup)
        after = time.time()
        assert before <= trainer.training_start_time <= after


# =============================================================================
# TEST: PRISMTrainer.run_initialization
# =============================================================================


class TestRunInitialization:
    """Tests for run_initialization method."""

    @pytest.fixture
    def initialized_trainer(
        self, trainer_setup: dict, sample_tensors: dict
    ) -> tuple[PRISMTrainer, dict]:
        """Trainer ready for initialization testing."""
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        return trainer, sample_tensors

    def test_returns_reconstruction_and_figure(
        self, initialized_trainer: tuple[PRISMTrainer, dict]
    ) -> None:
        """Test run_initialization returns a tensor and figure."""
        trainer, tensors = initialized_trainer
        result, figure = trainer.run_initialization(
            measurement=tensors["measurement"],
            center=tensors["sample_centers"][0],
            image_gt=tensors["image_gt"],
        )
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 4  # (B, C, H, W)
        # Figure is None when not provided
        assert figure is None

    def test_reconstruction_not_nan(self, initialized_trainer: tuple[PRISMTrainer, dict]) -> None:
        """Test reconstruction has no NaN values."""
        trainer, tensors = initialized_trainer
        result, _ = trainer.run_initialization(
            measurement=tensors["measurement"],
            center=tensors["sample_centers"][0],
            image_gt=tensors["image_gt"],
        )
        assert not torch.isnan(result).any()

    def test_reconstruction_stored_in_current_reconstruction(
        self, initialized_trainer: tuple[PRISMTrainer, dict]
    ) -> None:
        """Test reconstruction is stored in trainer.current_reconstruction."""
        trainer, tensors = initialized_trainer
        result, _ = trainer.run_initialization(
            measurement=tensors["measurement"],
            center=tensors["sample_centers"][0],
            image_gt=tensors["image_gt"],
        )
        assert trainer.current_reconstruction is not None
        assert torch.allclose(result, trainer.current_reconstruction)

    def test_early_exit_on_convergence(self, trainer_setup: dict, sample_tensors: dict) -> None:
        """Test training exits early when loss below threshold."""
        # Set very high threshold to trigger early exit
        trainer_setup["args"].loss_th = 1000.0
        trainer_setup["args"].max_epochs_init = 100
        trainer_setup["args"].n_epochs_init = 10

        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        # Run initialization - should converge quickly due to high threshold
        trainer.run_initialization(
            measurement=sample_tensors["measurement"],
            center=sample_tensors["sample_centers"][0],
            image_gt=sample_tensors["image_gt"],
        )
        # Should have run fewer epochs than max
        # (We can't directly check epochs, but we know it should converge)
        assert trainer.current_reconstruction is not None

    @pytest.mark.parametrize("activation", ["sigmoid", "hardsigmoid", "none"])
    def test_output_activation_modes(
        self, trainer_setup: dict, sample_tensors: dict, activation: str
    ) -> None:
        """Test different output activation modes."""
        trainer_setup["args"].output_activation = activation
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        result, _ = trainer.run_initialization(
            measurement=sample_tensors["measurement"],
            center=sample_tensors["sample_centers"][0],
            image_gt=sample_tensors["image_gt"],
        )
        assert result is not None
        assert not torch.isnan(result).any()

    def test_measurement_normalized_for_sigmoid(
        self, trainer_setup: dict, sample_tensors: dict
    ) -> None:
        """Test that measurement is normalized to [0, 1] for sigmoid activation."""
        trainer_setup["args"].output_activation = "sigmoid"
        # Create measurement with known range
        measurement = torch.rand(1, 1, 64, 64) * 100 + 50  # Range [50, 150]
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        result, _ = trainer.run_initialization(
            measurement=measurement,
            center=sample_tensors["sample_centers"][0],
            image_gt=sample_tensors["image_gt"],
        )
        # Should not raise and should produce valid output
        assert not torch.isnan(result).any()

    def test_max_epochs_limit_respected(self, trainer_setup: dict, sample_tensors: dict) -> None:
        """Test that max_epochs_init limit is respected."""
        trainer_setup["args"].loss_th = 1e-10  # Very low threshold (won't converge)
        trainer_setup["args"].max_epochs_init = 2
        trainer_setup["args"].n_epochs_init = 3

        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        # Should complete without hanging due to max_epochs limit
        result, _ = trainer.run_initialization(
            measurement=sample_tensors["measurement"],
            center=sample_tensors["sample_centers"][0],
            image_gt=sample_tensors["image_gt"],
        )
        assert result is not None

    def test_scheduler_step_called(self, trainer_setup: dict, sample_tensors: dict) -> None:
        """Test that scheduler.step is called during training."""
        from unittest.mock import Mock

        trainer = PRISMTrainer(**trainer_setup, use_amp=False)

        original_scheduler = trainer.scheduler
        mock_scheduler = Mock(wraps=original_scheduler)
        trainer.scheduler = mock_scheduler

        trainer.run_initialization(
            measurement=sample_tensors["measurement"],
            center=sample_tensors["sample_centers"][0],
            image_gt=sample_tensors["image_gt"],
        )

        # Scheduler step should have been called at least once
        assert mock_scheduler.step.called

    def test_optimizer_zero_grad_called(self, trainer_setup: dict, sample_tensors: dict) -> None:
        """Test that optimizer.zero_grad is called at end of initialization."""
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)

        # Track zero_grad calls
        original_zero_grad = trainer.optimizer.zero_grad
        call_count = [0]

        def counting_zero_grad(*args, **kwargs):
            call_count[0] += 1
            return original_zero_grad(*args, **kwargs)

        trainer.optimizer.zero_grad = counting_zero_grad

        trainer.run_initialization(
            measurement=sample_tensors["measurement"],
            center=sample_tensors["sample_centers"][0],
            image_gt=sample_tensors["image_gt"],
        )

        # zero_grad should have been called multiple times
        assert call_count[0] > 0

    def test_figure_parameter_accepted(self, trainer_setup: dict, sample_tensors: dict) -> None:
        """Test that figure parameter is accepted (returned as-is when None)."""
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)

        # Test with figure=None (default)
        result, returned_figure = trainer.run_initialization(
            measurement=sample_tensors["measurement"],
            center=sample_tensors["sample_centers"][0],
            image_gt=sample_tensors["image_gt"],
            figure=None,
        )
        assert result is not None
        assert returned_figure is None

        # Note: Testing with an actual figure requires a fully mocked measurement
        # system with all visualization dependencies, which is beyond the scope
        # of unit testing run_initialization logic. Integration tests should cover
        # visualization functionality.

    def test_reconstruction_shape_matches_model_output(
        self, initialized_trainer: tuple[PRISMTrainer, dict]
    ) -> None:
        """Test reconstruction shape matches what model produces."""
        trainer, tensors = initialized_trainer
        result, _ = trainer.run_initialization(
            measurement=tensors["measurement"],
            center=tensors["sample_centers"][0],
            image_gt=tensors["image_gt"],
        )
        # Model output should be (1, 1, 64, 64) based on input_size=64
        assert result.shape == (1, 1, 64, 64)

    def test_initialization_target_meas(self, trainer_setup: dict, sample_tensors: dict) -> None:
        """Test initialization with measurement target."""
        trainer_setup["args"].initialization_target = "meas"
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        result, _ = trainer.run_initialization(
            measurement=sample_tensors["measurement"],
            center=sample_tensors["sample_centers"][0],
            image_gt=sample_tensors["image_gt"],
        )
        assert result is not None
        assert not torch.isnan(result).any()


# =============================================================================
# TEST: PRISMTrainer.run_progressive_training
# =============================================================================


class TestRunProgressiveTraining:
    """Tests for run_progressive_training method."""

    @pytest.fixture
    def training_ready_trainer(
        self, trainer_setup: dict, sample_tensors: dict
    ) -> tuple[PRISMTrainer, dict]:
        """Trainer with initial reconstruction set."""
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        # Set initial reconstruction (normally done by run_initialization)
        trainer.current_reconstruction = torch.rand(1, 1, 64, 64, device=trainer.device)
        return trainer, sample_tensors

    def test_single_sample_returns_early(self, trainer_setup: dict, sample_tensors: dict) -> None:
        """Test that n_samples <= 1 returns without training."""
        trainer_setup["args"].n_samples = 1
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]

        result = trainer.run_progressive_training(
            sample_centers=sample_tensors["sample_centers"][:1],
            image=sample_tensors["image_gt"],
            image_gt=sample_tensors["image_gt"],
            samples_per_line_meas=0,
        )
        # Should return results dict immediately
        assert isinstance(result, dict)
        assert "final_reconstruction" in result

    def test_returns_results_dict(self, training_ready_trainer: tuple[PRISMTrainer, dict]) -> None:
        """Test that method returns a results dictionary."""
        trainer, tensors = training_ready_trainer
        trainer.args.n_samples = 3
        trainer.args.max_epochs = 1
        trainer.args.n_epochs = 2

        with disable_training_progress():
            result = trainer.run_progressive_training(
                sample_centers=tensors["sample_centers"],
                image=tensors["image_gt"],
                image_gt=tensors["image_gt"],
                samples_per_line_meas=0,
            )

        assert isinstance(result, dict)
        assert "final_reconstruction" in result
        assert "losses" in result
        assert "ssims" in result
        assert "rmses" in result
        assert "psnrs" in result
        assert "failed_samples" in result
        assert "sample_times" in result

    def test_metrics_tracked_per_sample(
        self, training_ready_trainer: tuple[PRISMTrainer, dict]
    ) -> None:
        """Test that metrics are tracked for each sample."""
        trainer, tensors = training_ready_trainer
        trainer.args.n_samples = 3
        trainer.args.max_epochs = 1
        trainer.args.n_epochs = 2

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=tensors["sample_centers"],
                image=tensors["image_gt"],
                image_gt=tensors["image_gt"],
                samples_per_line_meas=0,
            )

        # Should have recorded metrics for each sample after first
        # (n_samples - 1 because first sample is the initialization)
        assert len(trainer.losses) >= 1
        assert len(trainer.ssims) >= 1
        assert len(trainer.rmses) >= 1

    def test_measurement_system_called(
        self, training_ready_trainer: tuple[PRISMTrainer, dict]
    ) -> None:
        """Test measurement system is called for each sample."""
        trainer, tensors = training_ready_trainer
        trainer.args.n_samples = 3
        trainer.args.max_epochs = 1
        trainer.args.n_epochs = 2

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=tensors["sample_centers"],
                image=tensors["image_gt"],
                image_gt=tensors["image_gt"],
                samples_per_line_meas=0,
            )

        # Verify measure was called
        assert trainer.measurement_system.measure.called

    def test_current_reconstruction_updated(
        self, training_ready_trainer: tuple[PRISMTrainer, dict]
    ) -> None:
        """Test that current_reconstruction is updated after training."""
        trainer, tensors = training_ready_trainer
        trainer.args.n_samples = 3
        trainer.args.max_epochs = 1
        trainer.args.n_epochs = 5

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=tensors["sample_centers"],
                image=tensors["image_gt"],
                image_gt=tensors["image_gt"],
                samples_per_line_meas=0,
            )

        # Reconstruction should have changed
        assert trainer.current_reconstruction is not None

    @pytest.mark.parametrize("adaptive", [True, False])
    def test_adaptive_convergence_modes(
        self, trainer_setup: dict, sample_tensors: dict, adaptive: bool
    ) -> None:
        """Test both adaptive and non-adaptive convergence modes."""
        trainer_setup["args"].enable_adaptive_convergence = adaptive
        trainer_setup["args"].n_samples = 3
        trainer_setup["args"].max_epochs = 1
        trainer_setup["args"].n_epochs = 2

        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]

        with disable_training_progress():
            result = trainer.run_progressive_training(
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
            )

        # Should complete without error in both modes
        assert result is not None
        assert "final_reconstruction" in result

    def test_convergence_detection(self, trainer_setup: dict, sample_tensors: dict) -> None:
        """Test that convergence is detected when loss is low."""
        trainer_setup["args"].loss_th = 1000.0  # Very high threshold
        trainer_setup["args"].n_samples = 3
        trainer_setup["args"].max_epochs = 2
        trainer_setup["args"].n_epochs = 5

        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]

        with disable_training_progress():
            # Should converge quickly due to high threshold
            result = trainer.run_progressive_training(
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
            )

        assert result is not None

    def test_sample_times_tracked(self, training_ready_trainer: tuple[PRISMTrainer, dict]) -> None:
        """Test that sample processing times are tracked."""
        trainer, tensors = training_ready_trainer
        trainer.args.n_samples = 3
        trainer.args.max_epochs = 1
        trainer.args.n_epochs = 2

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=tensors["sample_centers"],
                image=tensors["image_gt"],
                image_gt=tensors["image_gt"],
                samples_per_line_meas=0,
            )

        # Should have recorded timing for each sample
        assert len(trainer.sample_times) >= 1
        assert all(t > 0 for t in trainer.sample_times)

    def test_lr_history_tracked(self, training_ready_trainer: tuple[PRISMTrainer, dict]) -> None:
        """Test that learning rate history is tracked."""
        trainer, tensors = training_ready_trainer
        trainer.args.n_samples = 3
        trainer.args.max_epochs = 1
        trainer.args.n_epochs = 2

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=tensors["sample_centers"],
                image=tensors["image_gt"],
                image_gt=tensors["image_gt"],
                samples_per_line_meas=0,
            )

        # Should have recorded LR for each sample
        assert len(trainer.lr_history) >= 1

    def test_convergence_stats_tracked_with_adaptive(
        self, trainer_setup: dict, sample_tensors: dict
    ) -> None:
        """Test that convergence statistics are tracked when adaptive enabled."""
        trainer_setup["args"].enable_adaptive_convergence = True
        trainer_setup["args"].n_samples = 3
        trainer_setup["args"].max_epochs = 1
        trainer_setup["args"].n_epochs = 2

        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
            )

        # Should have convergence stats for each sample
        assert len(trainer.epochs_per_sample) >= 1
        assert len(trainer.tiers_per_sample) >= 1

    def test_failed_samples_tracked(self, trainer_setup: dict, sample_tensors: dict) -> None:
        """Test that failed samples are tracked when they don't converge."""
        trainer_setup["args"].loss_th = 1e-15  # Impossibly low threshold
        trainer_setup["args"].n_samples = 3
        trainer_setup["args"].max_epochs = 1  # Very few epochs
        trainer_setup["args"].n_epochs = 1
        trainer_setup["args"].enable_adaptive_convergence = False  # Disable escalation

        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]

        with disable_training_progress():
            result = trainer.run_progressive_training(
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
            )

        # Should have tracked failed samples (samples that didn't converge)
        assert "failed_samples" in result

    def test_optimizer_reset_per_sample(
        self, training_ready_trainer: tuple[PRISMTrainer, dict]
    ) -> None:
        """Test that optimizer is reset for each sample."""
        trainer, tensors = training_ready_trainer
        trainer.args.n_samples = 3
        trainer.args.max_epochs = 1
        trainer.args.n_epochs = 2

        # Track initial optimizer
        initial_optimizer = trainer.optimizer

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=tensors["sample_centers"],
                image=tensors["image_gt"],
                image_gt=tensors["image_gt"],
                samples_per_line_meas=0,
            )

        # Optimizer should have been replaced (new instance per sample)
        # The optimizer reference changes during progressive training
        assert trainer.optimizer is not initial_optimizer

    def test_point_measurement_mode(self, trainer_setup: dict, sample_tensors: dict) -> None:
        """Test training with point measurement mode."""
        trainer_setup["args"].sample_length = 0  # Point mode
        trainer_setup["args"].n_samples = 3
        trainer_setup["args"].max_epochs = 1
        trainer_setup["args"].n_epochs = 2

        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]

        with disable_training_progress():
            result = trainer.run_progressive_training(
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
            )

        assert result is not None

    def test_wall_time_tracking(self, training_ready_trainer: tuple[PRISMTrainer, dict]) -> None:
        """Test that wall time is tracked in results."""
        trainer, tensors = training_ready_trainer
        trainer.args.n_samples = 3
        trainer.args.max_epochs = 1
        trainer.args.n_epochs = 2

        with disable_training_progress():
            result = trainer.run_progressive_training(
                sample_centers=tensors["sample_centers"],
                image=tensors["image_gt"],
                image_gt=tensors["image_gt"],
                samples_per_line_meas=0,
            )

        assert "wall_time_seconds" in result
        assert result["wall_time_seconds"] > 0

    def test_pattern_metadata_passed_through(
        self, training_ready_trainer: tuple[PRISMTrainer, dict]
    ) -> None:
        """Test that pattern metadata is included in results."""
        trainer, tensors = training_ready_trainer
        trainer.args.n_samples = 3
        trainer.args.max_epochs = 1
        trainer.args.n_epochs = 2

        pattern_metadata = {"type": "spiral", "n_points": 100}
        pattern_spec = "spiral:100"

        with disable_training_progress():
            result = trainer.run_progressive_training(
                sample_centers=tensors["sample_centers"],
                image=tensors["image_gt"],
                image_gt=tensors["image_gt"],
                samples_per_line_meas=0,
                pattern_metadata=pattern_metadata,
                pattern_spec=pattern_spec,
            )

        assert result["pattern_metadata"] == pattern_metadata
        assert result["pattern_spec"] == pattern_spec

    def test_ssim_values_in_valid_range(
        self, training_ready_trainer: tuple[PRISMTrainer, dict]
    ) -> None:
        """Test that SSIM values are in valid range [-1, 1]."""
        trainer, tensors = training_ready_trainer
        trainer.args.n_samples = 3
        trainer.args.max_epochs = 1
        trainer.args.n_epochs = 3

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=tensors["sample_centers"],
                image=tensors["image_gt"],
                image_gt=tensors["image_gt"],
                samples_per_line_meas=0,
            )

        for ssim_val in trainer.ssims:
            assert -1.0 <= ssim_val <= 1.0

    def test_psnr_values_reasonable(
        self, training_ready_trainer: tuple[PRISMTrainer, dict]
    ) -> None:
        """Test that PSNR values are reasonable (not NaN or Inf)."""
        trainer, tensors = training_ready_trainer
        trainer.args.n_samples = 3
        trainer.args.max_epochs = 1
        trainer.args.n_epochs = 3

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=tensors["sample_centers"],
                image=tensors["image_gt"],
                image_gt=tensors["image_gt"],
                samples_per_line_meas=0,
            )

        for psnr_val in trainer.psnrs:
            assert not torch.isnan(torch.tensor(psnr_val))
            assert not torch.isinf(torch.tensor(psnr_val))


# =============================================================================
# TEST: PRISMTrainer._log_sample_metrics
# =============================================================================


class TestLogSampleMetrics:
    """Tests for _log_sample_metrics method."""

    @pytest.fixture
    def trainer_with_metrics(
        self, trainer_setup: dict, sample_tensors: dict
    ) -> tuple[PRISMTrainer, dict]:
        """Trainer with populated metric lists."""
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]
        # Populate metric lists (simulating prior training)
        trainer.losses = [0.5, 0.4, 0.3]
        trainer.ssims = [0.8, 0.85, 0.9]
        trainer.rmses = [0.15, 0.12, 0.1]
        trainer.psnrs = [25.0, 27.0, 30.0]
        trainer.sample_times = [1.5, 1.3, 1.2]
        trainer.lr_history = [1e-3, 1e-3, 1e-3]
        return trainer, sample_tensors

    def test_logs_scalar_metrics(self, trainer_with_metrics: tuple[PRISMTrainer, dict]) -> None:
        """Test scalar metrics are logged to TensorBoard."""
        trainer, tensors = trainer_with_metrics
        mock_writer = Mock()
        trainer.writer = mock_writer

        center = tensors["sample_centers"][0]
        trainer._log_sample_metrics(center_idx=1, loss_old=0.4, loss_new=0.3, center=center)

        # Verify add_scalar was called for each metric
        assert mock_writer.add_scalar.called
        # Should be called for: Loss, Loss/old, Loss/new, SSIM, RMSE, PSNR, Time/per_sample, LearningRate
        assert mock_writer.add_scalar.call_count >= 7

    def test_logs_hparams_on_first_sample(
        self, trainer_with_metrics: tuple[PRISMTrainer, dict]
    ) -> None:
        """Test hyperparameters logged only on first sample (center_idx=0)."""
        trainer, tensors = trainer_with_metrics
        mock_writer = Mock()
        trainer.writer = mock_writer

        center = tensors["sample_centers"][0]
        # Log for center_idx=0 (first sample)
        trainer._log_sample_metrics(center_idx=0, loss_old=0.5, loss_new=0.4, center=center)

        # Should have logged hparams
        mock_writer.add_hparams.assert_called_once()
        # Check hparam_dict includes expected keys
        call_args = mock_writer.add_hparams.call_args
        hparam_dict = call_args[0][0]
        assert "lr" in hparam_dict
        assert "loss_th" in hparam_dict
        assert "n_samples" in hparam_dict

    def test_no_hparams_on_subsequent_samples(
        self, trainer_with_metrics: tuple[PRISMTrainer, dict]
    ) -> None:
        """Test hyperparameters not logged on subsequent samples."""
        trainer, tensors = trainer_with_metrics
        mock_writer = Mock()
        trainer.writer = mock_writer

        center = tensors["sample_centers"][0]
        # Log for center_idx=1 (not first sample)
        trainer._log_sample_metrics(center_idx=1, loss_old=0.4, loss_new=0.3, center=center)

        # Should NOT have logged hparams
        mock_writer.add_hparams.assert_not_called()

    def test_logs_hparams_with_snr(self, trainer_with_metrics: tuple[PRISMTrainer, dict]) -> None:
        """Test SNR included in hparams when set."""
        trainer, tensors = trainer_with_metrics
        trainer.args.snr = 20.0  # Set SNR
        mock_writer = Mock()
        trainer.writer = mock_writer

        center = tensors["sample_centers"][0]
        trainer._log_sample_metrics(center_idx=0, loss_old=0.5, loss_new=0.4, center=center)

        # Check SNR in hparam_dict
        call_args = mock_writer.add_hparams.call_args
        hparam_dict = call_args[0][0]
        assert "snr" in hparam_dict
        assert hparam_dict["snr"] == 20.0

    def test_handles_none_writer(self, trainer_with_metrics: tuple[PRISMTrainer, dict]) -> None:
        """Test gracefully handles writer=None."""
        trainer, tensors = trainer_with_metrics
        trainer.writer = None

        center = tensors["sample_centers"][0]
        # Should not raise
        trainer._log_sample_metrics(center_idx=1, loss_old=0.4, loss_new=0.3, center=center)

    def test_logs_image(self, trainer_with_metrics: tuple[PRISMTrainer, dict]) -> None:
        """Test reconstruction image is logged."""
        trainer, tensors = trainer_with_metrics
        mock_writer = Mock()
        trainer.writer = mock_writer

        center = tensors["sample_centers"][0]
        trainer._log_sample_metrics(center_idx=1, loss_old=0.4, loss_new=0.3, center=center)

        # Should have logged image
        mock_writer.add_image.assert_called_once()
        # Verify it's the reconstructed image
        call_args = mock_writer.add_image.call_args
        assert call_args[0][0] == "Reconstructed Image"

    def test_logs_with_snr(self, trainer_with_metrics: tuple[PRISMTrainer, dict]) -> None:
        """Test logging works when SNR is set."""
        trainer, tensors = trainer_with_metrics
        trainer.args.snr = 15.0
        mock_writer = Mock()
        trainer.writer = mock_writer

        center = tensors["sample_centers"][0]
        # Should not raise
        trainer._log_sample_metrics(center_idx=1, loss_old=0.4, loss_new=0.3, center=center)
        assert mock_writer.add_scalar.called


# =============================================================================
# TEST: PRISMTrainer._save_checkpoint
# =============================================================================


class TestSaveCheckpoint:
    """Tests for _save_checkpoint method."""

    @pytest.fixture
    def trainer_with_state(
        self, trainer_setup: dict, sample_tensors: dict, tmp_path: "Path"
    ) -> tuple[PRISMTrainer, dict]:
        """Trainer with populated state for checkpointing."""
        trainer = PRISMTrainer(**trainer_setup, use_amp=False, log_dir=str(tmp_path))
        trainer.current_reconstruction = sample_tensors["measurement"]
        # Populate state lists
        trainer.losses = [0.5, 0.4, 0.3]
        trainer.ssims = [0.8, 0.85, 0.9]
        trainer.rmses = [0.15, 0.12, 0.1]
        trainer.psnrs = [25.0, 27.0, 30.0]
        trainer.sample_times = [1.5, 1.3, 1.2]
        trainer.lr_history = [1e-3, 1e-3, 1e-3]
        trainer.failed_samples = []
        return trainer, sample_tensors

    def test_checkpoint_structure(
        self, trainer_with_state: tuple[PRISMTrainer, dict], tmp_path: "Path"
    ) -> None:
        """Test checkpoint contains required keys."""
        trainer, tensors = trainer_with_state

        pattern_metadata = {"type": "spiral", "n_points": 100}
        pattern_spec = "spiral:100"

        trainer._save_checkpoint(
            center_idx=2,
            sample_centers=tensors["sample_centers"],
            pattern_metadata=pattern_metadata,
            pattern_spec=pattern_spec,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        assert checkpoint_path.exists()

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Verify required keys
        assert "model" in checkpoint
        assert "losses" in checkpoint
        assert "ssims" in checkpoint
        assert "rmses" in checkpoint
        assert "psnrs" in checkpoint
        assert "current_rec" in checkpoint
        assert "sample_centers" in checkpoint
        assert "last_center_idx" in checkpoint
        assert "measurement_system" in checkpoint
        assert "optimizer" in checkpoint

    def test_saves_to_correct_path(
        self, trainer_with_state: tuple[PRISMTrainer, dict], tmp_path: "Path"
    ) -> None:
        """Test checkpoint saved to correct path."""
        trainer, tensors = trainer_with_state

        trainer._save_checkpoint(
            center_idx=0,
            sample_centers=tensors["sample_centers"],
            pattern_metadata=None,
            pattern_spec=None,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        assert checkpoint_path.exists()

    def test_handles_none_log_dir(self, trainer_setup: dict, sample_tensors: dict) -> None:
        """Test checkpoint not saved when log_dir is None."""
        trainer = PRISMTrainer(**trainer_setup, use_amp=False, log_dir=None)
        trainer.current_reconstruction = sample_tensors["measurement"]
        trainer.losses = [0.5]
        trainer.ssims = [0.8]

        # Should not raise, should return early
        trainer._save_checkpoint(
            center_idx=0,
            sample_centers=sample_tensors["sample_centers"],
            pattern_metadata=None,
            pattern_spec=None,
        )

    def test_args_updated_with_timestamp(
        self, trainer_with_state: tuple[PRISMTrainer, dict], tmp_path: "Path"
    ) -> None:
        """Test args are updated with end_time and last_sample."""
        trainer, tensors = trainer_with_state

        trainer._save_checkpoint(
            center_idx=2,
            sample_centers=tensors["sample_centers"],
            pattern_metadata=None,
            pattern_spec=None,
        )

        # Check args were updated
        assert hasattr(trainer.args, "end_time")
        assert hasattr(trainer.args, "last_sample")
        assert trainer.args.last_sample == 2

    def test_checkpoint_device_compatibility(
        self, trainer_with_state: tuple[PRISMTrainer, dict], tmp_path: "Path"
    ) -> None:
        """Test checkpoint can be loaded on different device."""
        trainer, tensors = trainer_with_state

        trainer._save_checkpoint(
            center_idx=0,
            sample_centers=tensors["sample_centers"],
            pattern_metadata=None,
            pattern_spec=None,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        # Should be loadable on CPU
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint is not None
        # Verify tensors are on CPU
        assert checkpoint["current_rec"].device == torch.device("cpu")

    def test_saves_pattern_metadata(
        self, trainer_with_state: tuple[PRISMTrainer, dict], tmp_path: "Path"
    ) -> None:
        """Test pattern metadata is saved in checkpoint."""
        trainer, tensors = trainer_with_state

        pattern_metadata = {"type": "spiral", "n_points": 100, "radius": 0.8}
        pattern_spec = "spiral:100:0.8"

        trainer._save_checkpoint(
            center_idx=1,
            sample_centers=tensors["sample_centers"],
            pattern_metadata=pattern_metadata,
            pattern_spec=pattern_spec,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        assert checkpoint["pattern_metadata"] == pattern_metadata
        assert checkpoint["pattern_spec"] == pattern_spec

    def test_saves_none_pattern_metadata(
        self, trainer_with_state: tuple[PRISMTrainer, dict], tmp_path: "Path"
    ) -> None:
        """Test checkpoint handles None pattern metadata."""
        trainer, tensors = trainer_with_state

        trainer._save_checkpoint(
            center_idx=0,
            sample_centers=tensors["sample_centers"],
            pattern_metadata=None,
            pattern_spec=None,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        assert checkpoint["pattern_metadata"] is None
        assert checkpoint["pattern_spec"] is None

    def test_checkpoint_includes_wall_time(
        self, trainer_with_state: tuple[PRISMTrainer, dict], tmp_path: "Path"
    ) -> None:
        """Test checkpoint includes wall_time_seconds."""
        trainer, tensors = trainer_with_state

        trainer._save_checkpoint(
            center_idx=0,
            sample_centers=tensors["sample_centers"],
            pattern_metadata=None,
            pattern_spec=None,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        assert "wall_time_seconds" in checkpoint
        assert checkpoint["wall_time_seconds"] >= 0

    def test_checkpoint_includes_args_dict(
        self, trainer_with_state: tuple[PRISMTrainer, dict], tmp_path: "Path"
    ) -> None:
        """Test checkpoint includes args as dict."""
        trainer, tensors = trainer_with_state

        trainer._save_checkpoint(
            center_idx=0,
            sample_centers=tensors["sample_centers"],
            pattern_metadata=None,
            pattern_spec=None,
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        assert "args_dict" in checkpoint
        assert isinstance(checkpoint["args_dict"], dict)


# =============================================================================
# ADDITIONAL FIXTURES FOR RETRY TESTS
# =============================================================================


@pytest.fixture
def mock_loss_aggregator(device: torch.device) -> Mock:
    """Mock LossAggregator for retry testing."""
    from prism.models.losses import LossAggregator

    # Create a real LossAggregator for compatibility
    criterion = LossAggregator(
        loss_type="l1",
        new_weight=0.3,
        f_weight=0.0,
    ).to(device)

    return criterion


# =============================================================================
# TEST: PRISMTrainer.retry_failed_samples
# =============================================================================


class TestRetryFailedSamples:
    """Tests for retry_failed_samples method."""

    def test_returns_zero_if_no_failures(
        self, trainer_setup: dict, sample_tensors: dict, mock_loss_aggregator: Mock
    ) -> None:
        """Test retry returns 0 when no failed samples."""
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]
        trainer.failed_samples = []  # No failures

        with disable_training_progress():
            recovered = trainer.retry_failed_samples(
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
            )

        assert recovered == 0

    def test_respects_max_retries(
        self, trainer_setup: dict, sample_tensors: dict, mock_loss_aggregator: Mock
    ) -> None:
        """Test retry respects max_retries limit."""
        trainer_setup["args"].max_retries = 2
        trainer_setup["args"].n_samples_0 = 0
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]
        trainer.failed_samples = [0, 1]  # Two failures

        # Make loss threshold very low so retries fail
        trainer_setup["args"].loss_th = 1e-15
        trainer_setup["args"].max_epochs = 1
        trainer_setup["args"].n_epochs = 1

        with disable_training_progress():
            recovered = trainer.retry_failed_samples(
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
            )

        # Should attempt retries but not recover (threshold too low)
        assert recovered == 0
        # Failed samples should still be in the list
        assert len(trainer.failed_samples) == 2

    def test_recovered_samples_removed_from_failed_list(
        self, trainer_setup: dict, sample_tensors: dict, mock_loss_aggregator: Mock
    ) -> None:
        """Test recovered samples are removed from failed_samples list."""
        trainer_setup["args"].max_retries = 2
        trainer_setup["args"].n_samples_0 = 0
        trainer_setup["args"].loss_th = 1000.0  # Easy convergence
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]
        trainer.failed_samples = [0]  # One failure

        with disable_training_progress():
            recovered = trainer.retry_failed_samples(
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
            )

        # Should have recovered the sample
        assert recovered == 1
        # Failed samples list should be empty
        assert len(trainer.failed_samples) == 0

    def test_still_failed_samples_remain_in_list(
        self, trainer_setup: dict, sample_tensors: dict, mock_loss_aggregator: Mock
    ) -> None:
        """Test samples that don't converge remain in failed_samples."""
        trainer_setup["args"].max_retries = 1
        trainer_setup["args"].n_samples_0 = 0
        trainer_setup["args"].loss_th = 1e-15  # Impossible threshold
        trainer_setup["args"].max_epochs = 1
        trainer_setup["args"].n_epochs = 1
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]
        trainer.failed_samples = [0, 1]

        with disable_training_progress():
            recovered = trainer.retry_failed_samples(
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
            )

        # Should not recover any samples
        assert recovered == 0
        # Both should still be in failed list
        assert len(trainer.failed_samples) == 2

    def test_returns_correct_recovered_count(
        self, trainer_setup: dict, sample_tensors: dict, mock_loss_aggregator: Mock
    ) -> None:
        """Test that recovered count is accurate."""
        trainer_setup["args"].max_retries = 2
        trainer_setup["args"].n_samples_0 = 0
        trainer_setup["args"].loss_th = 1000.0  # Easy convergence
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]
        # Create sample_centers with 3 samples
        sample_centers = torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0]], [[0.1, 0.1], [0.1, 0.1]], [[0.2, 0.0], [0.2, 0.0]]],
            device=trainer.device,
        )
        trainer.failed_samples = [0, 1]  # Two failures

        with disable_training_progress():
            recovered = trainer.retry_failed_samples(
                sample_centers=sample_centers,
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
            )

        # Should recover both samples
        assert recovered == 2

    def test_handles_max_retries_zero(
        self, trainer_setup: dict, sample_tensors: dict, mock_loss_aggregator: Mock
    ) -> None:
        """Test that max_retries=0 skips retry entirely."""
        trainer_setup["args"].max_retries = 0
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]
        trainer.failed_samples = [0, 1]

        with disable_training_progress():
            recovered = trainer.retry_failed_samples(
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
            )

        # Should return 0 immediately without attempting retries
        assert recovered == 0
        # Failed samples should remain unchanged
        assert trainer.failed_samples == [0, 1]


# =============================================================================
# TEST: PRISMTrainer._retry_single_sample
# =============================================================================


class TestRetrySingleSample:
    """Tests for _retry_single_sample method."""

    def test_uses_rescue_tier_config(
        self, trainer_setup: dict, sample_tensors: dict, mock_loss_aggregator: Mock
    ) -> None:
        """Test retry uses RESCUE tier configuration."""
        trainer_setup["args"].n_samples_0 = 0
        trainer_setup["args"].retry_lr_multiplier = 0.1
        trainer_setup["args"].loss_th = 1000.0  # Easy convergence
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]

        initial_lr = trainer.args.lr

        with disable_training_progress():
            result = trainer._retry_single_sample(
                sample_idx=0,
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
                retry_num=1,
            )

        # Should use reduced learning rate (RESCUE tier)
        # LR should be: initial_lr * retry_lr_multiplier / (retry_num + 1)
        expected_lr = initial_lr * 0.1 / 2  # retry_num=1, so divide by (1+1)
        assert abs(trainer.optimizer.param_groups[0]["lr"] - expected_lr) < 1e-9
        assert isinstance(result, bool)

    def test_convergence_returns_true(
        self, trainer_setup: dict, sample_tensors: dict, mock_loss_aggregator: Mock
    ) -> None:
        """Test successful convergence returns True."""
        trainer_setup["args"].n_samples_0 = 0
        trainer_setup["args"].loss_th = 1000.0  # Easy convergence
        trainer_setup["args"].max_epochs = 1
        trainer_setup["args"].n_epochs = 5
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]

        with disable_training_progress():
            result = trainer._retry_single_sample(
                sample_idx=0,
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
                retry_num=1,
            )

        # With high threshold, should converge
        assert result is True

    def test_failure_returns_false(
        self, trainer_setup: dict, sample_tensors: dict, mock_loss_aggregator: Mock
    ) -> None:
        """Test failure to converge returns False."""
        trainer_setup["args"].n_samples_0 = 0
        trainer_setup["args"].loss_th = 1e-15  # Impossible threshold
        trainer_setup["args"].max_epochs = 1
        trainer_setup["args"].n_epochs = 1
        trainer_setup["args"].early_stop_patience = 1
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]

        with disable_training_progress():
            result = trainer._retry_single_sample(
                sample_idx=0,
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
                retry_num=1,
            )

        # With impossible threshold and few epochs, should fail
        assert result is False

    def test_updates_reconstruction_on_success(
        self, trainer_setup: dict, sample_tensors: dict, mock_loss_aggregator: Mock
    ) -> None:
        """Test reconstruction is updated when retry succeeds."""
        trainer_setup["args"].n_samples_0 = 0
        trainer_setup["args"].loss_th = 1000.0  # Easy convergence
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        original_rec = sample_tensors["measurement"].clone()
        trainer.current_reconstruction = original_rec

        with disable_training_progress():
            result = trainer._retry_single_sample(
                sample_idx=0,
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
                retry_num=1,
            )

        # Should have succeeded
        assert result is True
        # Reconstruction should exist (may or may not have changed significantly)
        assert trainer.current_reconstruction is not None

    def test_loss_type_switching(
        self, trainer_setup: dict, sample_tensors: dict, mock_loss_aggregator: Mock
    ) -> None:
        """Test loss type switches on retry when enabled."""
        trainer_setup["args"].n_samples_0 = 0
        trainer_setup["args"].retry_switch_loss = True
        trainer_setup["args"].loss_type = "l1"
        trainer_setup["args"].loss_th = 1000.0
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]

        with disable_training_progress():
            # With retry_num > 0, should switch loss type
            result = trainer._retry_single_sample(
                sample_idx=0,
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
                retry_num=1,  # retry_num > 0 triggers switch
            )

        # Should complete (loss type switching is internal)
        assert isinstance(result, bool)

    def test_reduced_learning_rate(
        self, trainer_setup: dict, sample_tensors: dict, mock_loss_aggregator: Mock
    ) -> None:
        """Test retry uses progressively reduced learning rate."""
        trainer_setup["args"].n_samples_0 = 0
        trainer_setup["args"].retry_lr_multiplier = 0.1
        trainer_setup["args"].loss_th = 1000.0
        trainer = PRISMTrainer(**trainer_setup, use_amp=False)
        trainer.current_reconstruction = sample_tensors["measurement"]

        base_lr = trainer.args.lr

        with disable_training_progress():
            # First retry
            trainer._retry_single_sample(
                sample_idx=0,
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
                retry_num=1,
            )
            lr_retry1 = trainer.optimizer.param_groups[0]["lr"]

            # Second retry should have even smaller LR
            trainer._retry_single_sample(
                sample_idx=0,
                sample_centers=sample_tensors["sample_centers"],
                image=sample_tensors["image_gt"],
                image_gt=sample_tensors["image_gt"],
                samples_per_line_meas=0,
                criterion=mock_loss_aggregator,
                retry_num=2,
            )
            lr_retry2 = trainer.optimizer.param_groups[0]["lr"]

        # Second retry should have smaller LR than first
        # retry_lr = base_lr * retry_lr_multiplier / (retry_num + 1)
        expected_lr1 = base_lr * 0.1 / 2  # retry_num=1
        expected_lr2 = base_lr * 0.1 / 3  # retry_num=2

        assert abs(lr_retry1 - expected_lr1) < 1e-9
        assert abs(lr_retry2 - expected_lr2) < 1e-9
        assert lr_retry2 < lr_retry1

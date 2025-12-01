"""Integration tests for PRISMTrainer end-to-end workflows.

These tests verify complete training workflows using real components:
- Real ProgressiveDecoder models
- Mock MeasurementSystem (real one has separate integration tests)
- Real optimizer and scheduler
- Actual training loops (with minimal iterations)

The goal is to verify that all PRISMTrainer components work together correctly,
not to test MeasurementSystem integration (covered by test_measurement_system_*.py).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest
import torch

from prism.core.trainers import PRISMTrainer, create_scheduler
from prism.models.networks import ProgressiveDecoder


if TYPE_CHECKING:
    from pathlib import Path

    from prism.core.measurement_system import MeasurementSystem


# =============================================================================
# TEST UTILITIES
# =============================================================================


class MockTrainingProgress:
    """Mock TrainingProgress that does nothing (for integration tests)."""

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
    """Context manager to disable TrainingProgress for integration tests."""
    with patch("prism.core.trainers.TrainingProgress", MockTrainingProgress):
        yield


def create_mock_measurement_system(device: torch.device) -> Mock:
    """Create a mock MeasurementSystem with controlled outputs.

    This mock is designed to work with LossAggregator and the training loop.
    It returns properly shaped tensors that maintain gradient flow.
    """
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


def create_minimal_args() -> Mock:
    """Create minimal args for integration testing."""
    args = Mock()

    # Initialization params
    args.max_epochs_init = 1
    args.n_epochs_init = 3
    args.loss_th = 0.001
    args.output_activation = "sigmoid"

    # Training params
    args.max_epochs = 1
    args.n_epochs = 3
    args.n_samples = 3
    args.n_samples_0 = 0
    args.lr = 1e-3
    args.use_amsgrad = False

    # Geometry params
    args.obj_size = 64
    args.sample_diameter = 16
    args.roi_diameter = 32
    args.samples_r_cutoff = 0.8
    args.sample_length = 0  # Point mode
    args.samples_per_line_rec = 0

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
    args.name = "integration_test"
    args.snr = None

    return args


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def small_model(device: torch.device) -> ProgressiveDecoder:
    """Create a small ProgressiveDecoder for integration testing."""
    model = ProgressiveDecoder(input_size=64, latent_channels=32)
    return model.to(device)


@pytest.fixture
def measurement_system(device: torch.device) -> Mock:
    """Create a mock MeasurementSystem for integration testing.

    Using a mock here because MeasurementSystem integration is tested
    separately in test_measurement_system_*.py files. This allows us
    to focus on PRISMTrainer integration.
    """
    return create_mock_measurement_system(device)


@pytest.fixture
def integration_setup(
    small_model: ProgressiveDecoder,
    measurement_system: Mock,
    device: torch.device,
) -> dict[str, Any]:
    """Complete integration test setup with real model and mock measurement."""
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
    scheduler = create_scheduler(optimizer, scheduler_type="plateau")
    args = create_minimal_args()

    return {
        "model": small_model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "measurement_system": measurement_system,
        "args": args,
        "device": device,
    }


@pytest.fixture
def sample_data(device: torch.device) -> dict[str, torch.Tensor]:
    """Create sample data for integration testing."""
    # Create reproducible random data
    torch.manual_seed(42)

    return {
        "measurement": torch.rand(1, 1, 64, 64, device=device),
        "image": torch.rand(1, 1, 64, 64, device=device),
        "image_gt": torch.rand(1, 1, 64, 64, device=device),
        # Line endpoints shape: (n_lines, 2, 2) where each line is [[y1,x1], [y2,x2]]
        # For point mode, these become single points when samples_per_line_meas=0
        "sample_centers": torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],  # Point at origin
                [[0.1, 0.1], [0.1, 0.1]],  # Point at (0.1, 0.1)
                [[0.2, 0.0], [0.2, 0.0]],  # Point at (0.2, 0.0)
            ],
            device=device,
        ),
    }


# =============================================================================
# INTEGRATION TESTS: Full Workflows
# =============================================================================


@pytest.mark.integration
class TestTrainerFullWorkflow:
    """Integration tests for complete training workflows."""

    def test_initialization_completes_without_error(
        self, integration_setup: dict[str, Any], sample_data: dict[str, torch.Tensor]
    ) -> None:
        """Test that initialization phase completes without errors."""
        trainer = PRISMTrainer(**integration_setup, use_amp=False)

        result, figure = trainer.run_initialization(
            measurement=sample_data["measurement"],
            center=sample_data["sample_centers"][0],
            image_gt=sample_data["image_gt"],
        )

        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 4
        assert not torch.isnan(result).any()
        assert figure is None  # No figure provided

    def test_initialization_then_progressive(
        self, integration_setup: dict[str, Any], sample_data: dict[str, torch.Tensor]
    ) -> None:
        """Test complete workflow: initialization → progressive training."""
        trainer = PRISMTrainer(**integration_setup, use_amp=False)

        # Run initialization
        rec, _ = trainer.run_initialization(
            measurement=sample_data["measurement"],
            center=sample_data["sample_centers"][0],
            image_gt=sample_data["image_gt"],
        )
        assert rec is not None
        assert not torch.isnan(rec).any()

        # Run progressive training
        with disable_training_progress():
            results = trainer.run_progressive_training(
                sample_centers=sample_data["sample_centers"],
                image=sample_data["image"],
                image_gt=sample_data["image_gt"],
                samples_per_line_meas=0,
            )

        # Verify results structure
        assert isinstance(results, dict)
        assert "final_reconstruction" in results
        assert "losses" in results
        assert "ssims" in results
        assert "rmses" in results
        assert "psnrs" in results
        assert "failed_samples" in results
        assert "sample_times" in results
        assert "wall_time_seconds" in results

        # Verify metrics collected
        assert len(trainer.losses) > 0
        assert len(trainer.ssims) > 0

    def test_progressive_training_with_adaptive_convergence(
        self, integration_setup: dict[str, Any], sample_data: dict[str, torch.Tensor]
    ) -> None:
        """Test progressive training with adaptive convergence enabled."""
        integration_setup["args"].enable_adaptive_convergence = True
        trainer = PRISMTrainer(**integration_setup, use_amp=False)

        # Run initialization first
        trainer.run_initialization(
            measurement=sample_data["measurement"],
            center=sample_data["sample_centers"][0],
            image_gt=sample_data["image_gt"],
        )

        # Run progressive training with adaptive convergence
        with disable_training_progress():
            results = trainer.run_progressive_training(
                sample_centers=sample_data["sample_centers"],
                image=sample_data["image"],
                image_gt=sample_data["image_gt"],
                samples_per_line_meas=0,
            )

        # Verify convergence statistics are tracked
        assert "epochs_per_sample" in results
        assert "tiers_per_sample" in results
        assert "convergence_stats" in results

    def test_training_with_pattern_metadata(
        self, integration_setup: dict[str, Any], sample_data: dict[str, torch.Tensor]
    ) -> None:
        """Test that pattern metadata is passed through correctly."""
        trainer = PRISMTrainer(**integration_setup, use_amp=False)

        # Run initialization
        trainer.run_initialization(
            measurement=sample_data["measurement"],
            center=sample_data["sample_centers"][0],
            image_gt=sample_data["image_gt"],
        )

        pattern_metadata = {"type": "spiral", "n_points": 100, "radius": 0.8}
        pattern_spec = "spiral:100:0.8"

        with disable_training_progress():
            results = trainer.run_progressive_training(
                sample_centers=sample_data["sample_centers"],
                image=sample_data["image"],
                image_gt=sample_data["image_gt"],
                samples_per_line_meas=0,
                pattern_metadata=pattern_metadata,
                pattern_spec=pattern_spec,
            )

        assert results["pattern_metadata"] == pattern_metadata
        assert results["pattern_spec"] == pattern_spec


@pytest.mark.integration
class TestTrainerCheckpointing:
    """Integration tests for checkpoint saving and loading."""

    def test_full_training_with_checkpointing(
        self,
        integration_setup: dict[str, Any],
        sample_data: dict[str, torch.Tensor],
        tmp_path: "Path",
    ) -> None:
        """Test training with checkpoint saving."""
        integration_setup["args"].save_data = True
        trainer = PRISMTrainer(**integration_setup, use_amp=False, log_dir=str(tmp_path))

        # Run initialization
        trainer.run_initialization(
            measurement=sample_data["measurement"],
            center=sample_data["sample_centers"][0],
            image_gt=sample_data["image_gt"],
        )

        # Run progressive training (will save checkpoints)
        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=sample_data["sample_centers"],
                image=sample_data["image"],
                image_gt=sample_data["image_gt"],
                samples_per_line_meas=0,
            )

        # Verify checkpoint was created
        checkpoint_path = tmp_path / "checkpoint.pt"
        assert checkpoint_path.exists()

        # Verify checkpoint contents
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert "model" in checkpoint
        assert "losses" in checkpoint
        assert "ssims" in checkpoint
        assert "current_rec" in checkpoint

    def test_checkpoint_device_compatibility(
        self,
        integration_setup: dict[str, Any],
        sample_data: dict[str, torch.Tensor],
        tmp_path: "Path",
    ) -> None:
        """Test that checkpoint can be loaded on CPU regardless of source device."""
        integration_setup["args"].save_data = True
        trainer = PRISMTrainer(**integration_setup, use_amp=False, log_dir=str(tmp_path))

        # Run minimal training to generate checkpoint
        trainer.run_initialization(
            measurement=sample_data["measurement"],
            center=sample_data["sample_centers"][0],
            image_gt=sample_data["image_gt"],
        )

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=sample_data["sample_centers"],
                image=sample_data["image"],
                image_gt=sample_data["image_gt"],
                samples_per_line_meas=0,
            )

        # Load checkpoint on CPU
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Verify tensors are on CPU
        assert checkpoint["current_rec"].device == torch.device("cpu")
        assert checkpoint["losses"].device == torch.device("cpu")


@pytest.mark.integration
class TestTrainerRetryMechanism:
    """Integration tests for retry mechanism."""

    def test_retry_failed_samples_workflow(
        self, integration_setup: dict[str, Any], sample_data: dict[str, torch.Tensor]
    ) -> None:
        """Test retry mechanism in a complete workflow."""
        # Set impossibly low threshold to force failures
        integration_setup["args"].loss_th = 1e-15
        integration_setup["args"].max_epochs = 1
        integration_setup["args"].n_epochs = 1

        trainer = PRISMTrainer(**integration_setup, use_amp=False)

        # Run initialization
        trainer.run_initialization(
            measurement=sample_data["measurement"],
            center=sample_data["sample_centers"][0],
            image_gt=sample_data["image_gt"],
        )

        # Run progressive training - should fail to converge
        with disable_training_progress():
            results = trainer.run_progressive_training(
                sample_centers=sample_data["sample_centers"],
                image=sample_data["image"],
                image_gt=sample_data["image_gt"],
                samples_per_line_meas=0,
            )

        # Should have tracked failed samples
        assert "failed_samples" in results
        # Note: Actual failure count depends on training dynamics


@pytest.mark.integration
class TestTrainerMetrics:
    """Integration tests for metric tracking."""

    def test_all_metrics_tracked(
        self, integration_setup: dict[str, Any], sample_data: dict[str, torch.Tensor]
    ) -> None:
        """Test that all metrics are properly tracked during training."""
        trainer = PRISMTrainer(**integration_setup, use_amp=False)

        # Run initialization
        trainer.run_initialization(
            measurement=sample_data["measurement"],
            center=sample_data["sample_centers"][0],
            image_gt=sample_data["image_gt"],
        )

        # Run progressive training
        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=sample_data["sample_centers"],
                image=sample_data["image"],
                image_gt=sample_data["image_gt"],
                samples_per_line_meas=0,
            )

        # Verify all metrics are tracked
        assert len(trainer.losses) >= 1
        assert len(trainer.ssims) >= 1
        assert len(trainer.rmses) >= 1
        assert len(trainer.psnrs) >= 1
        assert len(trainer.sample_times) >= 1
        assert len(trainer.lr_history) >= 1

    def test_ssim_values_in_valid_range(
        self, integration_setup: dict[str, Any], sample_data: dict[str, torch.Tensor]
    ) -> None:
        """Test that SSIM values are within valid range."""
        trainer = PRISMTrainer(**integration_setup, use_amp=False)

        trainer.run_initialization(
            measurement=sample_data["measurement"],
            center=sample_data["sample_centers"][0],
            image_gt=sample_data["image_gt"],
        )

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=sample_data["sample_centers"],
                image=sample_data["image"],
                image_gt=sample_data["image_gt"],
                samples_per_line_meas=0,
            )

        for ssim_val in trainer.ssims:
            assert -1.0 <= ssim_val <= 1.0

    def test_psnr_values_reasonable(
        self, integration_setup: dict[str, Any], sample_data: dict[str, torch.Tensor]
    ) -> None:
        """Test that PSNR values are not NaN or Inf."""
        trainer = PRISMTrainer(**integration_setup, use_amp=False)

        trainer.run_initialization(
            measurement=sample_data["measurement"],
            center=sample_data["sample_centers"][0],
            image_gt=sample_data["image_gt"],
        )

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=sample_data["sample_centers"],
                image=sample_data["image"],
                image_gt=sample_data["image_gt"],
                samples_per_line_meas=0,
            )

        for psnr_val in trainer.psnrs:
            assert not torch.isnan(torch.tensor(psnr_val))
            assert not torch.isinf(torch.tensor(psnr_val))


@pytest.mark.integration
@pytest.mark.gpu
class TestTrainerCUDA:
    """Integration tests for CUDA-specific functionality."""

    def test_training_on_cuda(self) -> None:
        """Test training on CUDA device."""
        device = torch.device("cuda")

        # Create model on CUDA
        model = ProgressiveDecoder(input_size=64, latent_channels=32).to(device)

        # Create mock measurement system for CUDA
        measurement_system = create_mock_measurement_system(device)

        # Create optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = create_scheduler(optimizer, scheduler_type="plateau")
        args = create_minimal_args()

        trainer = PRISMTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            measurement_system=measurement_system,
            args=args,
            device=device,
            use_amp=False,
        )

        # Create CUDA tensors
        measurement = torch.rand(1, 1, 64, 64, device=device)
        image = torch.rand(1, 1, 64, 64, device=device)
        image_gt = torch.rand(1, 1, 64, 64, device=device)
        sample_centers = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.1, 0.1], [0.1, 0.1]],
            ],
            device=device,
        )

        # Run training
        trainer.run_initialization(
            measurement=measurement,
            center=sample_centers[0],
            image_gt=image_gt,
        )

        with disable_training_progress():
            results = trainer.run_progressive_training(
                sample_centers=sample_centers,
                image=image,
                image_gt=image_gt,
                samples_per_line_meas=0,
            )

        assert results is not None
        assert results["final_reconstruction"] is not None

    def test_training_with_amp_on_cuda(self) -> None:
        """Test training on CUDA with automatic mixed precision.

        Tests that AMP (Automatic Mixed Precision) works correctly with SSIM loss.
        The SSIM computation is now properly wrapped with autocast(enabled=False)
        and uses float32 for numerical stability.
        """
        device = torch.device("cuda")

        # Create model on CUDA
        model = ProgressiveDecoder(input_size=64, latent_channels=32, use_amp=True).to(device)

        # Create mock measurement system for CUDA
        measurement_system = create_mock_measurement_system(device)

        # Create optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = create_scheduler(optimizer, scheduler_type="plateau")
        args = create_minimal_args()

        trainer = PRISMTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            measurement_system=measurement_system,
            args=args,
            device=device,
            use_amp=True,  # Enable AMP
        )

        # Create CUDA tensors
        measurement = torch.rand(1, 1, 64, 64, device=device)
        image = torch.rand(1, 1, 64, 64, device=device)
        image_gt = torch.rand(1, 1, 64, 64, device=device)
        sample_centers = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.1, 0.1], [0.1, 0.1]],
            ],
            device=device,
        )

        # Run training with AMP
        trainer.run_initialization(
            measurement=measurement,
            center=sample_centers[0],
            image_gt=image_gt,
        )

        with disable_training_progress():
            results = trainer.run_progressive_training(
                sample_centers=sample_centers,
                image=image,
                image_gt=image_gt,
                samples_per_line_meas=0,
            )

        assert results is not None
        # Verify no NaN values (AMP stability check)
        final_rec = results["final_reconstruction"]
        assert not torch.isnan(final_rec).any()


@pytest.mark.integration
class TestSchedulerIntegration:
    """Integration tests for scheduler functionality."""

    def test_plateau_scheduler_integration(
        self, integration_setup: dict[str, Any], sample_data: dict[str, torch.Tensor]
    ) -> None:
        """Test ReduceLROnPlateau scheduler integration."""
        trainer = PRISMTrainer(**integration_setup, use_amp=False)

        # Run initialization
        trainer.run_initialization(
            measurement=sample_data["measurement"],
            center=sample_data["sample_centers"][0],
            image_gt=sample_data["image_gt"],
        )

        _ = trainer.optimizer.param_groups[0]["lr"]  # noqa: F841 - verify lr exists

        with disable_training_progress():
            trainer.run_progressive_training(
                sample_centers=sample_data["sample_centers"],
                image=sample_data["image"],
                image_gt=sample_data["image_gt"],
                samples_per_line_meas=0,
            )

        # LR history should be tracked
        assert len(trainer.lr_history) > 0

    def test_cosine_scheduler_integration(
        self,
        small_model: ProgressiveDecoder,
        measurement_system: MeasurementSystem,
        sample_data: dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        """Test CosineAnnealingWarmRestarts scheduler integration."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        scheduler = create_scheduler(optimizer, scheduler_type="cosine_warm_restarts")
        args = create_minimal_args()

        trainer = PRISMTrainer(
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler,
            measurement_system=measurement_system,
            args=args,
            device=device,
            use_amp=False,
        )

        # Run initialization
        trainer.run_initialization(
            measurement=sample_data["measurement"],
            center=sample_data["sample_centers"][0],
            image_gt=sample_data["image_gt"],
        )

        with disable_training_progress():
            results = trainer.run_progressive_training(
                sample_centers=sample_data["sample_centers"],
                image=sample_data["image"],
                image_gt=sample_data["image_gt"],
                samples_per_line_meas=0,
            )

        # Should complete without error
        assert results is not None

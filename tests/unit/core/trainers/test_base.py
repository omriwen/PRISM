"""Unit tests for AbstractTrainer base class.

This module tests the AbstractTrainer interface, TrainingConfig,
EpochResult, TrainingResult, and MetricsCollector classes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from prism.core.trainers.base import (
    AbstractTrainer,
    EpochResult,
    MetricsCollector,
    TrainingConfig,
    TrainingResult,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x):
        return self.linear(x)


class ConcreteTrainer(AbstractTrainer):
    """Concrete implementation of AbstractTrainer for testing."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: TrainingConfig | None = None,
    ) -> None:
        super().__init__(model, device, config)
        self.train_called = False
        self.train_epoch_called = False
        self.epochs_trained = 0

    def train(self, **kwargs: Any) -> TrainingResult:
        self.train_called = True
        self.metrics.start()

        n_samples = kwargs.get("n_samples", 4)
        for i in range(n_samples):
            # Simulate training a sample
            output = torch.randn(1, 1, 64, 64)
            target = torch.randn(1, 1, 64, 64)
            metrics = self.compute_metrics(output, target)
            self.metrics.record_sample(
                loss=0.1 - i * 0.02,
                ssim=metrics["ssim"],
                psnr=metrics["psnr"],
                rmse=metrics["rmse"],
                epochs=10,
            )

        self.current_reconstruction = torch.randn(1, 1, 64, 64)
        return self.metrics.finalize(self.current_reconstruction)

    def train_epoch(self, epoch: int, **kwargs: Any) -> EpochResult:
        self.train_epoch_called = True
        self.epochs_trained += 1
        return EpochResult(loss=0.1 / (epoch + 1), loss_old=0.05, loss_new=0.05)

    def compute_metrics(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        # Simple mock metrics
        return {
            "ssim": 0.9,
            "psnr": 30.0,
            "rmse": 0.1,
        }


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        assert config.n_epochs == 100
        assert config.max_epochs == 100
        assert config.loss_threshold == 1e-4
        assert config.learning_rate == 1e-4

    def test_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            n_epochs=200,
            max_epochs=50,
            loss_threshold=1e-3,
            learning_rate=1e-3,
        )
        assert config.n_epochs == 200
        assert config.max_epochs == 50
        assert config.loss_threshold == 1e-3
        assert config.learning_rate == 1e-3


class TestEpochResult:
    """Tests for EpochResult dataclass."""

    def test_default_values(self):
        """Test EpochResult default values."""
        result = EpochResult(loss=0.1)
        assert result.loss == 0.1
        assert result.loss_old == 0.0
        assert result.loss_new == 0.0

    def test_with_values(self):
        """Test EpochResult with custom values."""
        result = EpochResult(loss=0.1, loss_old=0.03, loss_new=0.07)
        assert result.loss == 0.1
        assert result.loss_old == 0.03
        assert result.loss_new == 0.07


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_default_values(self):
        """Test TrainingResult default values."""
        result = TrainingResult()
        assert result.losses == []
        assert result.ssims == []
        assert result.psnrs == []
        assert result.rmses == []
        assert result.final_reconstruction is None
        assert result.failed_samples == []
        assert result.wall_time_seconds == 0.0
        assert result.epochs_per_sample == []

    def test_with_values(self):
        """Test TrainingResult with custom values."""
        recon = torch.randn(1, 1, 64, 64)
        result = TrainingResult(
            losses=[0.1, 0.05],
            ssims=[0.8, 0.9],
            psnrs=[25.0, 30.0],
            rmses=[0.2, 0.1],
            final_reconstruction=recon,
            failed_samples=[1],
            wall_time_seconds=60.0,
            epochs_per_sample=[100, 50],
        )
        assert result.losses == [0.1, 0.05]
        assert result.ssims == [0.8, 0.9]
        assert result.psnrs == [25.0, 30.0]
        assert result.rmses == [0.2, 0.1]
        assert torch.equal(result.final_reconstruction, recon)
        assert result.failed_samples == [1]
        assert result.wall_time_seconds == 60.0
        assert result.epochs_per_sample == [100, 50]

    def test_empty_lists_not_shared(self):
        """Test that default empty lists are not shared between instances."""
        result1 = TrainingResult()
        result2 = TrainingResult()
        result1.losses.append(0.5)
        assert result2.losses == []


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture
    def collector(self):
        """Create a MetricsCollector instance."""
        return MetricsCollector()

    def test_initialization(self, collector):
        """Test MetricsCollector initialization."""
        assert collector.losses == []
        assert collector.ssims == []
        assert collector.psnrs == []
        assert collector.rmses == []
        assert collector.failed_samples == []
        assert collector.epochs_per_sample == []

    def test_start_records_time(self, collector):
        """Test that start() records the start time."""
        collector.start()
        assert collector._start_time > 0

    def test_record_sample(self, collector):
        """Test recording a sample."""
        collector.record_sample(loss=0.1, ssim=0.9, psnr=30.0, rmse=0.1, epochs=50)

        assert collector.losses == [0.1]
        assert collector.ssims == [0.9]
        assert collector.psnrs == [30.0]
        assert collector.rmses == [0.1]
        assert collector.epochs_per_sample == [50]

    def test_record_multiple_samples(self, collector):
        """Test recording multiple samples."""
        collector.record_sample(loss=0.1, ssim=0.8, psnr=25.0, rmse=0.2, epochs=100)
        collector.record_sample(loss=0.05, ssim=0.9, psnr=30.0, rmse=0.1, epochs=50)

        assert len(collector.losses) == 2
        assert collector.ssims == [0.8, 0.9]

    def test_record_failure(self, collector):
        """Test recording a failed sample."""
        collector.record_failure(sample_idx=5)
        collector.record_failure(sample_idx=10)

        assert collector.failed_samples == [5, 10]

    def test_finalize_returns_training_result(self, collector):
        """Test that finalize() returns a TrainingResult."""
        collector.start()
        collector.record_sample(loss=0.1, ssim=0.9, psnr=30.0, rmse=0.1, epochs=50)
        collector.record_failure(sample_idx=2)

        recon = torch.randn(1, 1, 64, 64)
        result = collector.finalize(recon)

        assert isinstance(result, TrainingResult)
        assert result.losses == [0.1]
        assert result.ssims == [0.9]
        assert result.psnrs == [30.0]
        assert result.rmses == [0.1]
        assert result.failed_samples == [2]
        assert result.epochs_per_sample == [50]
        assert torch.equal(result.final_reconstruction, recon)

    def test_finalize_computes_wall_time(self, collector):
        """Test that finalize() computes wall time."""
        collector.start()
        time.sleep(0.1)  # Small delay
        result = collector.finalize()

        assert result.wall_time_seconds >= 0.1

    def test_finalize_without_start(self, collector):
        """Test finalize() without calling start()."""
        result = collector.finalize()
        assert result.wall_time_seconds == 0.0

    def test_finalize_without_reconstruction(self, collector):
        """Test finalize() without providing reconstruction."""
        collector.start()
        result = collector.finalize()
        assert result.final_reconstruction is None


class TestAbstractTrainer:
    """Tests for AbstractTrainer base class."""

    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel()

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def config(self):
        """Create a training config."""
        return TrainingConfig(n_epochs=10, max_epochs=5, loss_threshold=1e-3)

    @pytest.fixture
    def trainer(self, model, device, config):
        """Create a concrete trainer instance."""
        return ConcreteTrainer(model, device, config)

    def test_initialization(self, trainer, model, device, config):
        """Test trainer initialization."""
        assert trainer.model == model
        assert trainer.device == device
        assert trainer.config == config
        assert isinstance(trainer.metrics, MetricsCollector)
        assert trainer.current_reconstruction is None

    def test_initialization_without_config(self, model, device):
        """Test trainer initialization without config."""
        trainer = ConcreteTrainer(model, device)
        assert trainer.config is None

    def test_train_returns_training_result(self, trainer):
        """Test that train() returns TrainingResult."""
        result = trainer.train(n_samples=4)

        assert isinstance(result, TrainingResult)
        assert trainer.train_called
        assert len(result.ssims) == 4

    def test_train_epoch_returns_epoch_result(self, trainer):
        """Test that train_epoch() returns EpochResult."""
        result = trainer.train_epoch(epoch=0)

        assert isinstance(result, EpochResult)
        assert trainer.train_epoch_called

    def test_compute_metrics_returns_dict(self, trainer):
        """Test that compute_metrics() returns a dict."""
        output = torch.randn(1, 1, 64, 64)
        target = torch.randn(1, 1, 64, 64)

        metrics = trainer.compute_metrics(output, target)

        assert isinstance(metrics, dict)
        assert "ssim" in metrics
        assert "psnr" in metrics
        assert "rmse" in metrics

    def test_should_stop_with_config(self, trainer):
        """Test should_stop() with config threshold."""
        # Loss below threshold
        assert trainer.should_stop(1e-4) is True
        assert trainer.should_stop(1e-5) is True

        # Loss above threshold
        assert trainer.should_stop(1e-2) is False
        assert trainer.should_stop(0.1) is False

    def test_should_stop_without_config(self, model, device):
        """Test should_stop() without config."""
        trainer = ConcreteTrainer(model, device, config=None)

        # Always returns False without config
        assert trainer.should_stop(1e-10) is False
        assert trainer.should_stop(0.1) is False

    def test_save_checkpoint(self, trainer, tmp_path):
        """Test save_checkpoint() saves model state."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        trainer.current_reconstruction = torch.randn(1, 1, 64, 64)
        trainer.save_checkpoint(str(checkpoint_path), extra_data="test")

        # Verify checkpoint was saved
        assert checkpoint_path.exists()

        # Load and verify contents
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "model" in checkpoint
        assert "current_reconstruction" in checkpoint
        assert "extra_data" in checkpoint
        assert checkpoint["extra_data"] == "test"

    def test_save_checkpoint_without_reconstruction(self, trainer, tmp_path):
        """Test save_checkpoint() with None reconstruction."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        trainer.current_reconstruction = None
        trainer.save_checkpoint(str(checkpoint_path))

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert checkpoint["current_reconstruction"] is None


class TestAbstractTrainerAbstractMethods:
    """Tests verifying abstract methods must be implemented."""

    def test_cannot_instantiate_abstract_trainer(self):
        """Test that AbstractTrainer cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AbstractTrainer(SimpleModel(), torch.device("cpu"))

    def test_missing_train_raises(self):
        """Test that missing train() raises TypeError."""

        class IncompleteTrainer(AbstractTrainer):
            def train_epoch(self, epoch, **kwargs):
                return EpochResult(loss=0.1)

            def compute_metrics(self, output, target):
                return {"ssim": 0.9, "psnr": 30.0, "rmse": 0.1}

        with pytest.raises(TypeError):
            IncompleteTrainer(SimpleModel(), torch.device("cpu"))

    def test_missing_train_epoch_raises(self):
        """Test that missing train_epoch() raises TypeError."""

        class IncompleteTrainer(AbstractTrainer):
            def train(self, **kwargs):
                return TrainingResult()

            def compute_metrics(self, output, target):
                return {"ssim": 0.9, "psnr": 30.0, "rmse": 0.1}

        with pytest.raises(TypeError):
            IncompleteTrainer(SimpleModel(), torch.device("cpu"))

    def test_missing_compute_metrics_raises(self):
        """Test that missing compute_metrics() raises TypeError."""

        class IncompleteTrainer(AbstractTrainer):
            def train(self, **kwargs):
                return TrainingResult()

            def train_epoch(self, epoch, **kwargs):
                return EpochResult(loss=0.1)

        with pytest.raises(TypeError):
            IncompleteTrainer(SimpleModel(), torch.device("cpu"))


class TestAbstractTrainerIntegration:
    """Integration tests for AbstractTrainer."""

    @pytest.fixture
    def training_scenario(self):
        """Create a complete training scenario."""
        model = SimpleModel()
        device = torch.device("cpu")
        config = TrainingConfig(
            n_epochs=100,
            max_epochs=10,
            loss_threshold=0.05,
            learning_rate=1e-3,
        )
        return model, device, config

    def test_full_training_workflow(self, training_scenario):
        """Test complete training workflow."""
        model, device, config = training_scenario
        trainer = ConcreteTrainer(model, device, config)

        result = trainer.train(n_samples=10)

        assert isinstance(result, TrainingResult)
        assert len(result.ssims) == 10
        # wall_time is 0 when test runs very fast and start() is called
        assert result.wall_time_seconds >= 0
        assert result.final_reconstruction is not None

    def test_metrics_collection_during_training(self, training_scenario):
        """Test that metrics are properly collected during training."""
        model, device, config = training_scenario
        trainer = ConcreteTrainer(model, device, config)

        result = trainer.train(n_samples=5)

        # All samples should have metrics
        assert len(result.losses) == 5
        assert len(result.ssims) == 5
        assert len(result.psnrs) == 5
        assert len(result.rmses) == 5
        assert len(result.epochs_per_sample) == 5

    def test_checkpoint_roundtrip(self, training_scenario, tmp_path):
        """Test that checkpoints can be saved and loaded."""
        model, device, config = training_scenario
        trainer = ConcreteTrainer(model, device, config)

        # Train and save
        trainer.train(n_samples=3)
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        # Create new model and load
        new_model = SimpleModel()
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        new_model.load_state_dict(checkpoint["model"])

        # Verify model weights match
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.equal(param1, param2)

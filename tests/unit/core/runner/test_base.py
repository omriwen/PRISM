"""Unit tests for AbstractRunner base class.

This module tests the AbstractRunner template method pattern and the
ExperimentResult dataclass.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from prism.core.runner.base import AbstractRunner, ExperimentResult


class ConcreteRunner(AbstractRunner):
    """Concrete implementation of AbstractRunner for testing."""

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        self.setup_called = False
        self.load_data_called = False
        self.create_components_called = False
        self.run_experiment_called = False
        self.save_results_called = False
        self.cleanup_called = False
        self.should_fail_on_setup = False
        self.should_fail_on_experiment = False

    def setup(self) -> None:
        if self.should_fail_on_setup:
            raise RuntimeError("Setup failed")
        self.setup_called = True
        self.device = torch.device("cpu")
        self.log_dir = Path("/tmp/test")

    def load_data(self) -> None:
        self.load_data_called = True

    def create_components(self) -> None:
        self.create_components_called = True

    def run_experiment(self) -> ExperimentResult:
        if self.should_fail_on_experiment:
            raise RuntimeError("Experiment failed")
        self.run_experiment_called = True
        return ExperimentResult(
            ssims=[0.9, 0.95],
            psnrs=[30.0, 32.0],
            rmses=[0.1, 0.05],
            final_reconstruction=torch.randn(1, 1, 64, 64),
            log_dir=self.log_dir,
            elapsed_time=10.0,
        )

    def save_results(self, result: ExperimentResult) -> None:
        self.save_results_called = True

    def cleanup(self) -> None:
        self.cleanup_called = True
        super().cleanup()


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_default_values(self):
        """Test ExperimentResult default values."""
        result = ExperimentResult()
        assert result.ssims == []
        assert result.psnrs == []
        assert result.rmses == []
        assert result.final_reconstruction is None
        assert result.log_dir is None
        assert result.elapsed_time == 0.0
        assert result.failed_samples == []

    def test_with_values(self):
        """Test ExperimentResult with custom values."""
        recon = torch.randn(1, 1, 64, 64)
        log_dir = Path("/tmp/test")
        result = ExperimentResult(
            ssims=[0.9, 0.95],
            psnrs=[30.0, 32.0],
            rmses=[0.1, 0.05],
            final_reconstruction=recon,
            log_dir=log_dir,
            elapsed_time=10.5,
            failed_samples=[2, 5],
        )
        assert result.ssims == [0.9, 0.95]
        assert result.psnrs == [30.0, 32.0]
        assert result.rmses == [0.1, 0.05]
        assert torch.equal(result.final_reconstruction, recon)
        assert result.log_dir == log_dir
        assert result.elapsed_time == 10.5
        assert result.failed_samples == [2, 5]

    def test_empty_lists_not_shared(self):
        """Test that default empty lists are not shared between instances."""
        result1 = ExperimentResult()
        result2 = ExperimentResult()
        result1.ssims.append(0.5)
        assert result2.ssims == []

    def test_failed_samples_tracking(self):
        """Test that failed samples can be tracked."""
        result = ExperimentResult(failed_samples=[0, 3, 7])
        assert len(result.failed_samples) == 3
        assert 3 in result.failed_samples


class TestAbstractRunner:
    """Tests for AbstractRunner base class."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for testing."""
        args = Namespace()
        args.name = "test_run"
        args.log_dir = "runs"
        args.save_data = True
        args.use_cuda = False
        args.device_num = 0
        return args

    @pytest.fixture
    def runner(self, mock_args):
        """Create a concrete runner instance."""
        return ConcreteRunner(mock_args)

    def test_initialization(self, runner, mock_args):
        """Test runner initialization stores args."""
        assert runner.args == mock_args
        assert runner.config is None
        assert runner.device is None
        assert runner.log_dir is None
        assert runner.writer is None

    def test_run_calls_methods_in_order(self, runner):
        """Test that run() calls methods in the correct order."""
        result = runner.run()

        assert runner.setup_called
        assert runner.load_data_called
        assert runner.create_components_called
        assert runner.run_experiment_called
        assert runner.save_results_called
        assert runner.cleanup_called
        assert isinstance(result, ExperimentResult)

    def test_run_returns_experiment_result(self, runner):
        """Test that run() returns ExperimentResult."""
        result = runner.run()

        assert isinstance(result, ExperimentResult)
        assert result.ssims == [0.9, 0.95]
        assert result.psnrs == [30.0, 32.0]

    def test_cleanup_called_on_success(self, runner):
        """Test that cleanup is called on successful completion."""
        runner.run()
        assert runner.cleanup_called

    def test_cleanup_called_on_failure(self, runner):
        """Test that cleanup is called even when experiment fails."""
        runner.should_fail_on_experiment = True

        with pytest.raises(RuntimeError, match="Experiment failed"):
            runner.run()

        assert runner.cleanup_called

    def test_cleanup_called_on_setup_failure(self, runner):
        """Test that cleanup is called even when setup fails."""
        runner.should_fail_on_setup = True

        with pytest.raises(RuntimeError, match="Setup failed"):
            runner.run()

        assert runner.cleanup_called

    def test_cleanup_closes_writer(self, runner):
        """Test that cleanup closes TensorBoard writer."""
        mock_writer = MagicMock()
        runner.writer = mock_writer

        runner.cleanup()

        mock_writer.close.assert_called_once()

    def test_cleanup_handles_none_writer(self, runner):
        """Test that cleanup handles None writer gracefully."""
        runner.writer = None
        runner.cleanup()  # Should not raise

    def test_default_cleanup_behavior(self, mock_args):
        """Test default cleanup implementation from base class."""
        runner = ConcreteRunner(mock_args)
        mock_writer = MagicMock()
        runner.writer = mock_writer

        # Call base class cleanup via super()
        runner.cleanup()

        mock_writer.close.assert_called_once()


class TestAbstractRunnerAbstractMethods:
    """Tests verifying abstract methods must be implemented."""

    def test_cannot_instantiate_abstract_runner(self):
        """Test that AbstractRunner cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AbstractRunner(Namespace())

    def test_missing_setup_raises(self):
        """Test that missing setup() raises TypeError."""

        class IncompleteRunner(AbstractRunner):
            def load_data(self) -> None:
                pass

            def create_components(self) -> None:
                pass

            def run_experiment(self) -> ExperimentResult:
                return ExperimentResult()

            def save_results(self, result: ExperimentResult) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteRunner(Namespace())

    def test_missing_load_data_raises(self):
        """Test that missing load_data() raises TypeError."""

        class IncompleteRunner(AbstractRunner):
            def setup(self) -> None:
                pass

            def create_components(self) -> None:
                pass

            def run_experiment(self) -> ExperimentResult:
                return ExperimentResult()

            def save_results(self, result: ExperimentResult) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteRunner(Namespace())

    def test_missing_create_components_raises(self):
        """Test that missing create_components() raises TypeError."""

        class IncompleteRunner(AbstractRunner):
            def setup(self) -> None:
                pass

            def load_data(self) -> None:
                pass

            def run_experiment(self) -> ExperimentResult:
                return ExperimentResult()

            def save_results(self, result: ExperimentResult) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteRunner(Namespace())

    def test_missing_run_experiment_raises(self):
        """Test that missing run_experiment() raises TypeError."""

        class IncompleteRunner(AbstractRunner):
            def setup(self) -> None:
                pass

            def load_data(self) -> None:
                pass

            def create_components(self) -> None:
                pass

            def save_results(self, result: ExperimentResult) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteRunner(Namespace())

    def test_missing_save_results_raises(self):
        """Test that missing save_results() raises TypeError."""

        class IncompleteRunner(AbstractRunner):
            def setup(self) -> None:
                pass

            def load_data(self) -> None:
                pass

            def create_components(self) -> None:
                pass

            def run_experiment(self) -> ExperimentResult:
                return ExperimentResult()

        with pytest.raises(TypeError):
            IncompleteRunner(Namespace())


class TestAbstractRunnerTemplateMethod:
    """Tests for the template method pattern in AbstractRunner."""

    @pytest.fixture
    def tracking_runner(self):
        """Create a runner that tracks method call order."""

        class TrackingRunner(AbstractRunner):
            def __init__(self, args):
                super().__init__(args)
                self.call_order = []

            def setup(self) -> None:
                self.call_order.append("setup")
                self.device = torch.device("cpu")

            def load_data(self) -> None:
                self.call_order.append("load_data")

            def create_components(self) -> None:
                self.call_order.append("create_components")

            def run_experiment(self) -> ExperimentResult:
                self.call_order.append("run_experiment")
                return ExperimentResult(ssims=[0.9], psnrs=[30.0], rmses=[0.1])

            def save_results(self, result: ExperimentResult) -> None:
                self.call_order.append("save_results")

            def cleanup(self) -> None:
                self.call_order.append("cleanup")
                super().cleanup()

        return TrackingRunner(Namespace())

    def test_methods_called_in_order(self, tracking_runner):
        """Test that template method calls hooks in correct order."""
        tracking_runner.run()

        assert tracking_runner.call_order == [
            "setup",
            "load_data",
            "create_components",
            "run_experiment",
            "save_results",
            "cleanup",
        ]

    def test_cleanup_always_last_on_exception(self, tracking_runner):
        """Test that cleanup is always called last, even on exception."""

        def failing_experiment(self) -> ExperimentResult:
            self.call_order.append("run_experiment")
            raise ValueError("Test failure")

        tracking_runner.run_experiment = lambda: failing_experiment(tracking_runner)

        with pytest.raises(ValueError):
            tracking_runner.run()

        assert tracking_runner.call_order[-1] == "cleanup"

    def test_early_failure_still_calls_cleanup(self, tracking_runner):
        """Test cleanup is called even if early method fails."""

        def failing_load_data(self) -> None:
            self.call_order.append("load_data")
            raise RuntimeError("Load failed")

        tracking_runner.load_data = lambda: failing_load_data(tracking_runner)

        with pytest.raises(RuntimeError):
            tracking_runner.run()

        assert "cleanup" in tracking_runner.call_order
        assert "create_components" not in tracking_runner.call_order


class TestAbstractRunnerIntegration:
    """Integration tests for AbstractRunner with real components."""

    @pytest.fixture
    def full_mock_args(self):
        """Create comprehensive mock arguments."""
        args = Namespace()
        args.name = "integration_test"
        args.log_dir = "runs"
        args.save_data = False
        args.use_cuda = False
        args.device_num = 0
        args.log_level = "INFO"
        args.image_size = 64
        args.n_samples = 4
        return args

    def test_full_workflow_with_mock_components(self, full_mock_args, tmp_path):
        """Test complete workflow with mocked components."""
        full_mock_args.log_dir = str(tmp_path)

        class IntegrationRunner(AbstractRunner):
            def setup(self) -> None:
                self.device = torch.device("cpu")
                self.log_dir = Path(self.args.log_dir)

            def load_data(self) -> None:
                self.image = torch.randn(1, 1, 64, 64)
                self.sample_centers = [[0, 0], [10, 10], [20, 20], [30, 30]]

            def create_components(self) -> None:
                self.model = MagicMock()
                self.trainer = MagicMock()

            def run_experiment(self) -> ExperimentResult:
                return ExperimentResult(
                    ssims=[0.8, 0.85, 0.9, 0.95],
                    psnrs=[25.0, 27.0, 29.0, 31.0],
                    rmses=[0.2, 0.15, 0.1, 0.05],
                    final_reconstruction=torch.randn(1, 1, 64, 64),
                    log_dir=self.log_dir,
                    elapsed_time=5.0,
                )

            def save_results(self, result: ExperimentResult) -> None:
                pass

        runner = IntegrationRunner(full_mock_args)
        result = runner.run()

        assert len(result.ssims) == 4
        assert result.ssims[-1] == 0.95
        assert result.elapsed_time == 5.0

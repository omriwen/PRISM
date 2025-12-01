"""
Unit tests for experiment comparison functionality.

Tests the ExperimentComparator class and related functions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from prism.analysis.comparison import ExperimentComparator, ExperimentResult


class TestExperimentResult:
    """Test ExperimentResult dataclass."""

    def test_creation(self):
        """Test creating an ExperimentResult."""
        result = ExperimentResult(
            name="test_exp",
            path=Path("/fake/path"),
            config={"lr": 0.001},
            final_metrics={"loss": 0.01, "ssim": 0.95},
            training_history={"losses": [0.1, 0.05, 0.01]},
            checkpoint={"model": "fake_model"},
        )

        assert result.name == "test_exp"
        assert result.final_metrics["loss"] == 0.01
        assert len(result.training_history["losses"]) == 3


class TestExperimentComparator:
    """Test ExperimentComparator class."""

    @pytest.fixture
    def comparator(self):
        """Create a comparator instance."""
        return ExperimentComparator()

    @pytest.fixture
    def mock_experiments(self):
        """Create mock experiment results for testing."""
        exp1 = ExperimentResult(
            name="exp1",
            path=Path("/fake/exp1"),
            config={"lr": 0.001, "n_samples": 100, "pattern": "fermat"},
            final_metrics={"loss": 0.01, "ssim": 0.95, "psnr": 30.0, "epochs": 100},
            training_history={
                "losses": [0.1, 0.05, 0.01],
                "ssims": [0.8, 0.9, 0.95],
                "psnrs": [20.0, 25.0, 30.0],
                "rmses": [10.0, 5.0, 2.0],
            },
            checkpoint={},
        )

        exp2 = ExperimentResult(
            name="exp2",
            path=Path("/fake/exp2"),
            config={"lr": 0.005, "n_samples": 100, "pattern": "random"},
            final_metrics={"loss": 0.02, "ssim": 0.90, "psnr": 28.0, "epochs": 150},
            training_history={
                "losses": [0.15, 0.08, 0.02],
                "ssims": [0.75, 0.85, 0.90],
                "psnrs": [18.0, 23.0, 28.0],
                "rmses": [12.0, 7.0, 3.0],
            },
            checkpoint={},
        )

        return [exp1, exp2]

    def test_compare_metrics(self, comparator, mock_experiments):
        """Test comparing metrics across experiments."""
        comparison = comparator.compare_metrics(mock_experiments)

        # Check that all metrics are present
        assert "loss" in comparison
        assert "ssim" in comparison
        assert "psnr" in comparison
        assert "epochs" in comparison

        # Check loss comparison (lower is better)
        assert comparison["loss"]["lower_is_better"] is True
        assert comparison["loss"]["best_experiment"] == "exp1"
        assert comparison["loss"]["best_value"] == 0.01

        # Check SSIM comparison (higher is better)
        assert comparison["ssim"]["lower_is_better"] is False
        assert comparison["ssim"]["best_experiment"] == "exp1"
        assert comparison["ssim"]["best_value"] == 0.95

        # Check statistics
        assert comparison["loss"]["mean"] == pytest.approx(0.015)
        assert comparison["ssim"]["mean"] == pytest.approx(0.925)

    def test_compare_configs(self, comparator, mock_experiments):
        """Test finding configuration differences."""
        differences = comparator.compare_configs(mock_experiments)

        # Learning rate differs
        assert "lr" in differences
        assert "0.001" in differences["lr"]
        assert "0.005" in differences["lr"]

        # Pattern differs
        assert "pattern" in differences
        assert "fermat" in differences["pattern"]
        assert "random" in differences["pattern"]

        # n_samples is the same, should not be in differences
        assert "n_samples" not in differences

    def test_compare_empty_experiments(self, comparator):
        """Test comparing with no experiments."""
        comparison = comparator.compare_metrics([])
        assert comparison == {}

        diff = comparator.compare_configs([])
        assert diff == {}

    def test_load_experiments_missing_path(self, comparator):
        """Test loading from non-existent path."""
        experiments = comparator.load_experiments([Path("/fake/nonexistent")])
        assert len(experiments) == 0

    def test_load_experiments_real_checkpoint(self, tmp_path):
        """Test loading from a real checkpoint file."""
        # Create a temporary experiment directory
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()

        # Create mock checkpoint
        checkpoint = {
            "losses": torch.tensor([0.1, 0.05, 0.01]),
            "ssims": torch.tensor([0.8, 0.9, 0.95]),
            "psnrs": torch.tensor([20.0, 25.0, 30.0]),
            "rmses": torch.tensor([10.0, 5.0, 2.0]),
            "model": {"fake": "model"},
        }
        torch.save(checkpoint, exp_dir / "checkpoint.pt")

        # Create mock args
        args = {
            "lr": 0.001,
            "n_samples": 100,
            "pattern": "fermat",
        }
        torch.save(args, exp_dir / "args.pt")

        # Load experiment
        comparator = ExperimentComparator()
        experiments = comparator.load_experiments([exp_dir])

        assert len(experiments) == 1
        assert experiments[0].name == "test_experiment"
        assert experiments[0].final_metrics["loss"] == pytest.approx(0.01)
        assert experiments[0].final_metrics["ssim"] == pytest.approx(0.95)
        assert experiments[0].final_metrics["psnr"] == pytest.approx(30.0)
        assert experiments[0].config["lr"] == 0.001

    def test_plot_comparison(self, comparator, mock_experiments):
        """Test generating comparison plot."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        fig = comparator.plot_comparison(mock_experiments)

        # Check that figure was created
        assert fig is not None

        # Check that figure has multiple axes
        assert len(fig.axes) >= 3

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_comparison_with_save(self, comparator, mock_experiments, tmp_path):
        """Test saving comparison plot."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        output_path = tmp_path / "comparison.png"

        fig = comparator.plot_comparison(
            mock_experiments,
            output_path=output_path,
        )

        # Check that file was created
        assert output_path.exists()

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_comparison_no_experiments(self, comparator):
        """Test plotting with no experiments raises error."""
        with pytest.raises(ValueError, match="No experiments to plot"):
            comparator.plot_comparison([])

    def test_print_metrics_table(self, comparator, mock_experiments):
        """Test printing metrics table."""
        comparator.print_metrics_table(mock_experiments)
        # Just verify no errors occurred

    def test_print_config_diff(self, comparator, mock_experiments):
        """Test printing configuration differences."""
        comparator.print_config_diff(mock_experiments)
        # Just verify no errors occurred

    def test_print_config_diff_identical(self, comparator):
        """Test printing config diff when configs are identical."""
        exp1 = ExperimentResult(
            name="exp1",
            path=Path("/fake/exp1"),
            config={"lr": 0.001, "n_samples": 100},
            final_metrics={},
            training_history={},
            checkpoint={},
        )

        exp2 = ExperimentResult(
            name="exp2",
            path=Path("/fake/exp2"),
            config={"lr": 0.001, "n_samples": 100},
            final_metrics={},
            training_history={},
            checkpoint={},
        )

        # Should indicate configs are identical
        comparator.print_config_diff([exp1, exp2])

    def test_metrics_with_missing_data(self, comparator):
        """Test metrics comparison when some experiments are missing data."""
        exp1 = ExperimentResult(
            name="exp1",
            path=Path("/fake/exp1"),
            config={},
            final_metrics={"loss": 0.01, "ssim": 0.95},
            training_history={},
            checkpoint={},
        )

        exp2 = ExperimentResult(
            name="exp2",
            path=Path("/fake/exp2"),
            config={},
            final_metrics={"loss": 0.02},  # Missing SSIM
            training_history={},
            checkpoint={},
        )

        comparison = comparator.compare_metrics([exp1, exp2])

        # Loss should have both experiments
        assert len(comparison["loss"]["experiments"]) == 2

        # SSIM should only have exp1
        assert len(comparison["ssim"]["experiments"]) == 1
        assert comparison["ssim"]["experiments"][0] == "exp1"


class TestMetricFiltering:
    """Test metric filtering functionality."""

    def test_filter_specific_metrics(self):
        """Test displaying only specific metrics."""
        comparator = ExperimentComparator()

        exp = ExperimentResult(
            name="exp",
            path=Path("/fake"),
            config={},
            final_metrics={"loss": 0.01, "ssim": 0.95, "psnr": 30.0, "rmse": 2.0},
            training_history={},
            checkpoint={},
        )

        # This should not raise an error
        comparator.print_metrics_table([exp], metrics=["loss", "ssim"])

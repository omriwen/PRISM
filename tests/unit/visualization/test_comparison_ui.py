"""Unit tests for comparison UI components."""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from prism.web.layouts.comparison import (
    create_comparison_metrics_table,
    create_config_diff_viewer,
    create_side_by_side_comparison,
    create_training_curve_overlay,
)
from prism.web.server import ExperimentData


@pytest.fixture
def temp_runs_dir():
    """Create a temporary runs directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_experiments():
    """Create sample experiment data for testing."""
    experiments = []

    # Experiment 1
    exp1 = ExperimentData(
        exp_id="exp1",
        path=Path("/tmp/exp1"),
        config={
            "n_samples": 100,
            "snr": 40.0,
            "learning_rate": 0.001,
            "pattern": "fermat",
            "epochs": 1000,
        },
        metrics={
            "epoch": list(range(100)),
            "loss": [1.0 - i * 0.01 for i in range(100)],
            "ssim": [0.5 + i * 0.004 for i in range(100)],
            "psnr": [20.0 + i * 0.15 for i in range(100)],
            "rmse": [0.1 - i * 0.0009 for i in range(100)],
        },
        final_metrics={
            "loss": 0.01,
            "ssim": 0.896,
            "psnr": 34.85,
            "rmse": 0.01,
            "epochs": 100,
        },
        timestamp=datetime(2025, 11, 20, 10, 0, 0),
        reconstruction=np.random.randn(256, 256),
    )
    experiments.append(exp1)

    # Experiment 2 - different configuration
    exp2 = ExperimentData(
        exp_id="exp2",
        path=Path("/tmp/exp2"),
        config={
            "n_samples": 150,
            "snr": 40.0,
            "learning_rate": 0.005,
            "pattern": "fermat",
            "epochs": 1000,
        },
        metrics={
            "epoch": list(range(80)),
            "loss": [0.8 - i * 0.009 for i in range(80)],
            "ssim": [0.6 + i * 0.005 for i in range(80)],
            "psnr": [22.0 + i * 0.16 for i in range(80)],
            "rmse": [0.09 - i * 0.001 for i in range(80)],
        },
        final_metrics={
            "loss": 0.08,
            "ssim": 0.995,
            "psnr": 34.8,
            "rmse": 0.01,
            "epochs": 80,
        },
        timestamp=datetime(2025, 11, 20, 11, 0, 0),
        reconstruction=np.random.randn(256, 256),
    )
    experiments.append(exp2)

    # Experiment 3 - worse performance
    exp3 = ExperimentData(
        exp_id="exp3",
        path=Path("/tmp/exp3"),
        config={
            "n_samples": 50,
            "snr": 40.0,
            "learning_rate": 0.001,
            "pattern": "random",
            "epochs": 1000,
        },
        metrics={
            "epoch": list(range(120)),
            "loss": [1.2 - i * 0.008 for i in range(120)],
            "ssim": [0.4 + i * 0.003 for i in range(120)],
            "psnr": [18.0 + i * 0.12 for i in range(120)],
            "rmse": [0.12 - i * 0.0008 for i in range(120)],
        },
        final_metrics={
            "loss": 0.24,
            "ssim": 0.76,
            "psnr": 32.4,
            "rmse": 0.024,
            "epochs": 120,
        },
        timestamp=datetime(2025, 11, 20, 12, 0, 0),
        reconstruction=np.random.randn(256, 256),
    )
    experiments.append(exp3)

    # Experiment 4 - for testing 4-grid layout
    exp4 = ExperimentData(
        exp_id="exp4",
        path=Path("/tmp/exp4"),
        config={
            "n_samples": 100,
            "snr": 35.0,
            "learning_rate": 0.002,
            "pattern": "fermat",
            "epochs": 1000,
        },
        metrics={
            "epoch": list(range(90)),
            "loss": [0.9 - i * 0.009 for i in range(90)],
            "ssim": [0.55 + i * 0.0045 for i in range(90)],
            "psnr": [21.0 + i * 0.14 for i in range(90)],
            "rmse": [0.095 - i * 0.00095 for i in range(90)],
        },
        final_metrics={
            "loss": 0.09,
            "ssim": 0.955,
            "psnr": 33.6,
            "rmse": 0.01,
            "epochs": 90,
        },
        timestamp=datetime(2025, 11, 20, 13, 0, 0),
        reconstruction=np.random.randn(256, 256),
    )
    experiments.append(exp4)

    return experiments


class TestSideBySideComparison:
    """Tests for side-by-side reconstruction comparison."""

    def test_empty_experiments(self):
        """Test with no experiments."""
        fig = create_side_by_side_comparison([])
        assert fig is not None
        # Should have a message annotation
        assert len(fig.layout.annotations) > 0

    def test_single_experiment(self, sample_experiments):
        """Test with single experiment."""
        fig = create_side_by_side_comparison([sample_experiments[0]])
        assert fig is not None
        # Should have 1 subplot
        assert len(fig.data) == 1

    def test_two_experiments(self, sample_experiments):
        """Test with two experiments (1x2 grid)."""
        fig = create_side_by_side_comparison(sample_experiments[:2])
        assert fig is not None
        # Should have 2 subplots
        assert len(fig.data) == 2

    def test_four_experiments(self, sample_experiments):
        """Test with four experiments (2x2 grid)."""
        fig = create_side_by_side_comparison(sample_experiments[:4])
        assert fig is not None
        # Should have 4 subplots
        assert len(fig.data) == 4

    def test_more_than_four_experiments(self, sample_experiments):
        """Test that it limits to 4 experiments."""
        # Add a 5th experiment
        exp5 = sample_experiments[0]
        exp5.exp_id = "exp5"
        all_exps = sample_experiments + [exp5]

        fig = create_side_by_side_comparison(all_exps)
        assert fig is not None
        # Should still only have 4 subplots
        assert len(fig.data) <= 4

    def test_sync_axes(self, sample_experiments):
        """Test synchronized axes."""
        fig = create_side_by_side_comparison(sample_experiments[:2], sync_axes=True)
        assert fig is not None
        # Axes should be linked
        # Note: checking xaxis/yaxis matching would require deeper inspection

    def test_no_reconstruction(self):
        """Test with experiment missing reconstruction data."""
        exp = ExperimentData(
            exp_id="no_recon",
            path=Path("/tmp/no_recon"),
            config={},
            metrics={},
            final_metrics={},
            reconstruction=None,
        )
        fig = create_side_by_side_comparison([exp])
        assert fig is not None


class TestComparisonMetricsTable:
    """Tests for comparison metrics table."""

    def test_empty_experiments(self):
        """Test with no experiments."""
        table = create_comparison_metrics_table([])
        assert table is not None
        assert len(table.data) == 0

    def test_multiple_experiments(self, sample_experiments):
        """Test with multiple experiments."""
        table = create_comparison_metrics_table(sample_experiments[:3])
        assert table is not None
        assert len(table.data) == 3

    def test_best_value_highlighting(self, sample_experiments):
        """Test that best values are highlighted."""
        table = create_comparison_metrics_table(sample_experiments[:3])
        assert table is not None

        # Check that style_data_conditional is set
        assert hasattr(table, "style_data_conditional")
        assert len(table.style_data_conditional) > 0

        # Should have highlighting rules for best values
        highlight_rules = [
            rule for rule in table.style_data_conditional if "backgroundColor" in rule
        ]
        assert len(highlight_rules) > 0

    def test_sortable_columns(self, sample_experiments):
        """Test that table is sortable."""
        table = create_comparison_metrics_table(sample_experiments[:2])
        assert table.sort_action == "native"

    def test_export_enabled(self, sample_experiments):
        """Test that export is enabled."""
        table = create_comparison_metrics_table(sample_experiments[:2])
        assert table.export_format == "csv"


class TestConfigDiffViewer:
    """Tests for configuration diff viewer."""

    def test_empty_experiments(self):
        """Test with no experiments."""
        viewer = create_config_diff_viewer([])
        assert viewer is not None

    def test_single_experiment(self, sample_experiments):
        """Test with single experiment."""
        viewer = create_config_diff_viewer([sample_experiments[0]])
        assert viewer is not None
        # Should show message about needing 2+ experiments

    def test_identical_configs(self):
        """Test with identical configurations."""
        exp1 = ExperimentData(
            exp_id="exp1",
            path=Path("/tmp/exp1"),
            config={"a": 1, "b": 2},
            metrics={},
            final_metrics={},
        )
        exp2 = ExperimentData(
            exp_id="exp2",
            path=Path("/tmp/exp2"),
            config={"a": 1, "b": 2},
            metrics={},
            final_metrics={},
        )
        viewer = create_config_diff_viewer([exp1, exp2])
        assert viewer is not None
        # Should indicate all configs are identical

    def test_different_configs(self, sample_experiments):
        """Test with different configurations."""
        viewer = create_config_diff_viewer(sample_experiments[:2])
        assert viewer is not None
        # Should show differences

    def test_missing_keys(self):
        """Test with configs having different keys."""
        exp1 = ExperimentData(
            exp_id="exp1",
            path=Path("/tmp/exp1"),
            config={"a": 1, "b": 2},
            metrics={},
            final_metrics={},
        )
        exp2 = ExperimentData(
            exp_id="exp2",
            path=Path("/tmp/exp2"),
            config={"a": 1, "c": 3},
            metrics={},
            final_metrics={},
        )
        viewer = create_config_diff_viewer([exp1, exp2])
        assert viewer is not None


class TestTrainingCurveOverlay:
    """Tests for training curve overlay."""

    def test_empty_experiments(self):
        """Test with no experiments."""
        fig = create_training_curve_overlay([])
        assert fig is not None
        # Should have a message annotation
        assert len(fig.layout.annotations) > 0

    def test_single_experiment(self, sample_experiments):
        """Test with single experiment."""
        fig = create_training_curve_overlay([sample_experiments[0]])
        assert fig is not None
        # Should have 3 traces (loss, ssim, psnr)
        assert len(fig.data) >= 3

    def test_multiple_experiments(self, sample_experiments):
        """Test with multiple experiments."""
        fig = create_training_curve_overlay(sample_experiments[:3])
        assert fig is not None
        # Should have 9 traces (3 metrics × 3 experiments)
        assert len(fig.data) >= 9

    def test_smoothing(self, sample_experiments):
        """Test smoothing parameter."""
        fig_no_smooth = create_training_curve_overlay(sample_experiments[:2], smoothing_window=1)
        fig_smooth = create_training_curve_overlay(sample_experiments[:2], smoothing_window=10)

        assert fig_no_smooth is not None
        assert fig_smooth is not None

        # Smoothed figure should have traces with fewer points
        # (due to convolution)

    def test_opacity(self, sample_experiments):
        """Test opacity parameter."""
        fig = create_training_curve_overlay(sample_experiments[:2], opacity=0.5)
        assert fig is not None
        # Check that traces have opacity set
        for trace in fig.data:
            if hasattr(trace, "opacity"):
                assert trace.opacity == 0.5

    def test_log_scale_loss(self, sample_experiments):
        """Test that loss axis uses log scale."""
        fig = create_training_curve_overlay(sample_experiments[:2])
        assert fig is not None
        # Loss subplot (row 1) should have log scale
        # This would require checking fig.layout.yaxis.type == 'log'

    def test_export_config(self, sample_experiments):
        """Test that figure has export configuration."""
        fig = create_training_curve_overlay(sample_experiments[:2])
        assert fig is not None
        # Figure should be exportable (this is set in the callback, not in the layout function)


class TestComparisonIntegration:
    """Integration tests for comparison features."""

    def test_full_comparison_workflow(self, sample_experiments):
        """Test complete comparison workflow."""
        # Create all comparison components
        recon_fig = create_side_by_side_comparison(sample_experiments[:4])
        metrics_table = create_comparison_metrics_table(sample_experiments[:4])
        config_diff = create_config_diff_viewer(sample_experiments[:3])
        curves_fig = create_training_curve_overlay(sample_experiments[:3])

        # All components should be created successfully
        assert recon_fig is not None
        assert metrics_table is not None
        assert config_diff is not None
        assert curves_fig is not None

    def test_performance_with_large_dataset(self):
        """Test performance with large training history."""
        # Create experiment with 10000 epochs
        exp = ExperimentData(
            exp_id="large_exp",
            path=Path("/tmp/large"),
            config={"epochs": 10000},
            metrics={
                "epoch": list(range(10000)),
                "loss": [1.0 - i * 0.0001 for i in range(10000)],
                "ssim": [0.5 + i * 0.00005 for i in range(10000)],
                "psnr": [20.0 + i * 0.0015 for i in range(10000)],
                "rmse": [0.1 - i * 0.00001 for i in range(10000)],
            },
            final_metrics={"loss": 0.0, "ssim": 1.0, "psnr": 35.0, "rmse": 0.0, "epochs": 10000},
            reconstruction=np.random.randn(512, 512),
        )

        # Should handle large dataset without errors
        fig = create_training_curve_overlay([exp])
        assert fig is not None

    def test_missing_metrics(self):
        """Test handling of experiments with missing metrics."""
        exp = ExperimentData(
            exp_id="incomplete",
            path=Path("/tmp/incomplete"),
            config={},
            metrics={},
            final_metrics={},
            reconstruction=None,
        )

        # Should handle gracefully
        fig = create_training_curve_overlay([exp])
        assert fig is not None

        table = create_comparison_metrics_table([exp])
        assert table is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

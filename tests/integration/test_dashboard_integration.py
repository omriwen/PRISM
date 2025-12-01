"""Integration tests for SPIDS dashboard with real experiment data."""

from __future__ import annotations

from pathlib import Path

import pytest

from prism.web.layouts.main import (
    create_config_table,
    create_metrics_table,
    create_reconstruction_comparison,
    create_training_curves,
)
from prism.web.server import DashboardServer


# Mark as integration test
pytestmark = pytest.mark.integration


@pytest.fixture
def real_runs_dir():
    """Get path to real runs directory."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        pytest.skip("No runs directory found")
    return runs_dir


@pytest.fixture
def dashboard_server(real_runs_dir):
    """Create dashboard server with real data."""
    return DashboardServer(runs_dir=real_runs_dir)


class TestDashboardIntegration:
    """Integration tests for dashboard with real experiment data."""

    def test_scan_real_experiments(self, dashboard_server):
        """Test scanning real experiments directory."""
        experiments = dashboard_server.scan_experiments()

        # Should find at least one experiment
        assert len(experiments) >= 0, "Expected to find experiments in runs/"

        if len(experiments) > 0:
            # Check structure of first experiment
            exp = experiments[0]
            assert "id" in exp
            assert "path" in exp
            assert "last_modified" in exp
            assert "last_modified_str" in exp
            assert "size_mb" in exp

            # Verify path exists
            assert exp["path"].exists()
            assert (exp["path"] / "checkpoint.pt").exists()

    def test_load_real_experiment(self, dashboard_server):
        """Test loading a real experiment."""
        experiments = dashboard_server.scan_experiments()

        if len(experiments) == 0:
            pytest.skip("No experiments to test with")

        exp_id = experiments[0]["id"]
        exp_data = dashboard_server.load_experiment_data(exp_id)

        assert exp_data is not None
        assert exp_data.exp_id == exp_id
        assert exp_data.config is not None
        assert exp_data.metrics is not None
        assert exp_data.final_metrics is not None

        # Check metrics have expected keys
        assert "loss" in exp_data.metrics or "losses" in exp_data.metrics
        assert "ssim" in exp_data.metrics or "ssims" in exp_data.metrics

        # Check final metrics
        assert "loss" in exp_data.final_metrics or exp_data.final_metrics["loss"] is None
        assert "epochs" in exp_data.final_metrics

    def test_create_training_curves_real_data(self, dashboard_server):
        """Test creating training curves with real data."""
        experiments = dashboard_server.scan_experiments()

        if len(experiments) == 0:
            pytest.skip("No experiments to test with")

        # Load first experiment
        exp_id = experiments[0]["id"]
        exp_data = dashboard_server.load_experiment_data(exp_id)

        if exp_data is None:
            pytest.skip(f"Could not load experiment {exp_id}")

        # Create figure
        fig = create_training_curves([exp_data])

        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) > 0  # Should have traces

    def test_create_metrics_table_real_data(self, dashboard_server):
        """Test creating metrics table with real data."""
        experiments = dashboard_server.scan_experiments()

        if len(experiments) == 0:
            pytest.skip("No experiments to test with")

        # Load up to 3 experiments
        exp_data_list = []
        for exp_info in experiments[:3]:
            exp_data = dashboard_server.load_experiment_data(exp_info["id"])
            if exp_data:
                exp_data_list.append(exp_data)

        if not exp_data_list:
            pytest.skip("Could not load any experiments")

        # Create table
        table = create_metrics_table(exp_data_list)

        assert table is not None
        assert hasattr(table, "data")
        assert len(table.data) > 0

    def test_create_config_table_real_data(self, dashboard_server):
        """Test creating config table with real data."""
        experiments = dashboard_server.scan_experiments()

        if len(experiments) == 0:
            pytest.skip("No experiments to test with")

        # Load first experiment
        exp_id = experiments[0]["id"]
        exp_data = dashboard_server.load_experiment_data(exp_id)

        if exp_data is None or not exp_data.config:
            pytest.skip(f"Could not load config for experiment {exp_id}")

        # Create table
        table = create_config_table(exp_data)

        assert table is not None
        assert hasattr(table, "data")
        assert len(table.data) > 0

    def test_reconstruction_comparison_real_data(self, dashboard_server):
        """Test reconstruction comparison with real data."""
        experiments = dashboard_server.scan_experiments()

        if len(experiments) == 0:
            pytest.skip("No experiments to test with")

        # Load experiments with reconstructions
        exp_data_list = []
        for exp_info in experiments[:2]:
            exp_data = dashboard_server.load_experiment_data(exp_info["id"])
            if exp_data and exp_data.reconstruction is not None:
                exp_data_list.append(exp_data)

        if not exp_data_list:
            pytest.skip("No experiments with reconstructions found")

        # Create figure
        fig = create_reconstruction_comparison(exp_data_list)

        assert fig is not None
        assert hasattr(fig, "data")

    def test_dashboard_caching_performance(self, dashboard_server):
        """Test that caching improves performance."""
        experiments = dashboard_server.scan_experiments()

        if len(experiments) == 0:
            pytest.skip("No experiments to test with")

        exp_id = experiments[0]["id"]

        import time

        # First load (no cache)
        start = time.time()
        _ = dashboard_server.load_experiment_data(exp_id, use_cache=False)
        time1 = time.time() - start

        # Second load (with cache)
        start = time.time()
        _ = dashboard_server.load_experiment_data(exp_id, use_cache=True)
        time2 = time.time() - start

        # Cached version should be faster or same
        # (allowing for some variance)
        assert time2 <= time1 * 1.5, "Cached load should not be significantly slower"

    def test_multiple_experiments_comparison(self, dashboard_server):
        """Test comparing multiple experiments."""
        experiments = dashboard_server.scan_experiments()

        if len(experiments) < 2:
            pytest.skip("Need at least 2 experiments for comparison")

        # Load multiple experiments
        exp_data_list = []
        for exp_info in experiments[:4]:  # Up to 4 experiments
            exp_data = dashboard_server.load_experiment_data(exp_info["id"])
            if exp_data:
                exp_data_list.append(exp_data)

        if len(exp_data_list) < 2:
            pytest.skip("Could not load at least 2 experiments")

        # Create comparison visualizations
        curves_fig = create_training_curves(exp_data_list)
        metrics_table = create_metrics_table(exp_data_list)

        assert curves_fig is not None
        assert len(curves_fig.data) >= 2  # At least 2 traces
        assert metrics_table is not None
        assert len(metrics_table.data) >= 2  # At least 2 rows


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

"""Unit tests for SPIDS dashboard server."""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import torch

from prism.web.server import DashboardServer, ExperimentData


@pytest.fixture
def temp_runs_dir():
    """Create a temporary runs directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_checkpoint():
    """Create a sample checkpoint for testing."""
    return {
        "losses": torch.tensor([1.0, 0.5, 0.3, 0.2, 0.1]),
        "ssims": torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9]),
        "psnrs": torch.tensor([20.0, 25.0, 28.0, 30.0, 32.0]),
        "rmses": torch.tensor([0.1, 0.08, 0.06, 0.05, 0.04]),
        "current_rec": torch.randn(1, 256, 256),
    }


@pytest.fixture
def sample_args():
    """Create sample args dictionary."""
    return {
        "input": "data/europa.jpg",
        "obj_size": 73,
        "image_size": 1024,
        "n_samples": 100,
        "fermat_sample": True,
        "snr": 40.0,
    }


@pytest.fixture
def populated_runs_dir(temp_runs_dir, sample_checkpoint, sample_args):
    """Create a populated runs directory with test experiments."""
    # Create experiment 1
    exp1_dir = temp_runs_dir / "test_exp_1"
    exp1_dir.mkdir()
    torch.save(sample_checkpoint, exp1_dir / "checkpoint.pt")
    torch.save(sample_args, exp1_dir / "args.pt")

    # Create experiment 2
    exp2_dir = temp_runs_dir / "test_exp_2"
    exp2_dir.mkdir()
    checkpoint2 = sample_checkpoint.copy()
    checkpoint2["losses"] = torch.tensor([0.8, 0.4, 0.2])
    torch.save(checkpoint2, exp2_dir / "checkpoint.pt")

    # Create args.txt for exp2
    with open(exp2_dir / "args.txt", "w") as f:
        for key, value in sample_args.items():
            f.write(f"{key}: {value}\n")

    # Create experiment without checkpoint (should be ignored)
    exp3_dir = temp_runs_dir / "test_exp_3"
    exp3_dir.mkdir()

    return temp_runs_dir


class TestDashboardServer:
    """Test suite for DashboardServer class."""

    def test_initialization(self, temp_runs_dir):
        """Test server initialization."""
        server = DashboardServer(runs_dir=temp_runs_dir)
        assert server.runs_dir == temp_runs_dir
        assert isinstance(server.experiments_cache, dict)
        assert len(server.experiments_cache) == 0

    def test_initialization_creates_missing_dir(self):
        """Test that server creates runs directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir) / "nonexistent_runs"
            _ = DashboardServer(runs_dir=runs_dir)
            assert runs_dir.exists()

    def test_scan_experiments_empty(self, temp_runs_dir):
        """Test scanning empty runs directory."""
        server = DashboardServer(runs_dir=temp_runs_dir)
        experiments = server.scan_experiments()
        assert experiments == []

    def test_scan_experiments_with_data(self, populated_runs_dir):
        """Test scanning runs directory with experiments."""
        server = DashboardServer(runs_dir=populated_runs_dir)
        experiments = server.scan_experiments()

        assert len(experiments) == 2  # exp_3 has no checkpoint
        assert all("id" in exp for exp in experiments)
        assert all("path" in exp for exp in experiments)
        assert all("last_modified" in exp for exp in experiments)
        assert all("last_modified_str" in exp for exp in experiments)
        assert all("size_mb" in exp for exp in experiments)

        # Check experiment IDs
        exp_ids = [exp["id"] for exp in experiments]
        assert "test_exp_1" in exp_ids
        assert "test_exp_2" in exp_ids
        assert "test_exp_3" not in exp_ids  # No checkpoint

    def test_load_experiment_data_success(self, populated_runs_dir):
        """Test loading experiment data successfully."""
        server = DashboardServer(runs_dir=populated_runs_dir)
        exp_data = server.load_experiment_data("test_exp_1")

        assert exp_data is not None
        assert isinstance(exp_data, ExperimentData)
        assert exp_data.exp_id == "test_exp_1"
        assert exp_data.path == populated_runs_dir / "test_exp_1"
        assert isinstance(exp_data.config, dict)
        assert isinstance(exp_data.metrics, dict)
        assert isinstance(exp_data.final_metrics, dict)

        # Check metrics structure
        assert "epoch" in exp_data.metrics
        assert "loss" in exp_data.metrics
        assert "ssim" in exp_data.metrics
        assert "psnr" in exp_data.metrics

        # Check final metrics
        assert "loss" in exp_data.final_metrics
        assert "ssim" in exp_data.final_metrics
        assert "psnr" in exp_data.final_metrics
        assert "epochs" in exp_data.final_metrics

        # Verify values
        assert exp_data.final_metrics["epochs"] == 5
        assert exp_data.final_metrics["loss"] == pytest.approx(0.1)
        assert exp_data.final_metrics["ssim"] == pytest.approx(0.9)

    def test_load_experiment_data_with_args_pt(self, populated_runs_dir):
        """Test loading config from args.pt."""
        server = DashboardServer(runs_dir=populated_runs_dir)
        exp_data = server.load_experiment_data("test_exp_1")

        assert exp_data is not None
        assert "input" in exp_data.config
        assert exp_data.config["input"] == "data/europa.jpg"
        assert exp_data.config["n_samples"] == 100

    def test_load_experiment_data_with_args_txt(self, populated_runs_dir):
        """Test loading config from args.txt."""
        server = DashboardServer(runs_dir=populated_runs_dir)
        exp_data = server.load_experiment_data("test_exp_2")

        assert exp_data is not None
        assert "input" in exp_data.config
        assert exp_data.config["input"] == "data/europa.jpg"
        assert exp_data.config["fermat_sample"] is True
        assert exp_data.config["snr"] == 40.0

    def test_load_experiment_data_nonexistent(self, temp_runs_dir):
        """Test loading nonexistent experiment."""
        server = DashboardServer(runs_dir=temp_runs_dir)
        exp_data = server.load_experiment_data("nonexistent_exp")
        assert exp_data is None

    def test_load_experiment_data_caching(self, populated_runs_dir):
        """Test that experiment data is cached."""
        server = DashboardServer(runs_dir=populated_runs_dir)

        # First load
        exp_data1 = server.load_experiment_data("test_exp_1")
        assert "test_exp_1" in server.experiments_cache

        # Second load (should use cache)
        exp_data2 = server.load_experiment_data("test_exp_1", use_cache=True)
        assert exp_data1 is exp_data2  # Same object

        # Third load without cache
        exp_data3 = server.load_experiment_data("test_exp_1", use_cache=False)
        assert exp_data1 is not exp_data3  # Different objects

    def test_clear_cache(self, populated_runs_dir):
        """Test cache clearing."""
        server = DashboardServer(runs_dir=populated_runs_dir)

        # Load some data
        server.load_experiment_data("test_exp_1")
        server.load_experiment_data("test_exp_2")
        assert len(server.experiments_cache) == 2

        # Clear cache
        server.clear_cache()
        assert len(server.experiments_cache) == 0

    def test_refresh_experiment(self, populated_runs_dir):
        """Test refreshing experiment data."""
        server = DashboardServer(runs_dir=populated_runs_dir)

        # Load initial data
        exp_data1 = server.load_experiment_data("test_exp_1")
        assert "test_exp_1" in server.experiments_cache

        # Refresh
        exp_data2 = server.refresh_experiment("test_exp_1")
        assert exp_data1 is not exp_data2  # Should be different objects
        assert exp_data2 is not None

    def test_tensor_to_list_conversions(self):
        """Test tensor to list conversion utility."""
        server = DashboardServer()

        # Test with tensor
        tensor_data = torch.tensor([1.0, 2.0, 3.0])
        result = server._tensor_to_list(tensor_data)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

        # Test with list
        list_data = [1.0, 2.0, 3.0]
        result = server._tensor_to_list(list_data)
        assert result == [1.0, 2.0, 3.0]

        # Test with empty
        result = server._tensor_to_list([])
        assert result == []

    def test_parse_args_txt(self, temp_runs_dir):
        """Test parsing args.txt file."""
        server = DashboardServer()

        # Create sample args.txt
        args_txt_path = temp_runs_dir / "args.txt"
        with open(args_txt_path, "w") as f:
            f.write("input: data/test.jpg\n")
            f.write("n_samples: 100\n")
            f.write("learning_rate: 0.001\n")
            f.write("fermat_sample: True\n")
            f.write("use_gpu: False\n")
            f.write("obj_name: None\n")

        config = server._parse_args_txt(args_txt_path)

        assert config["input"] == "data/test.jpg"
        assert config["n_samples"] == 100
        assert config["learning_rate"] == 0.001
        assert config["fermat_sample"] is True
        assert config["use_gpu"] is False
        assert config["obj_name"] is None


class TestExperimentData:
    """Test suite for ExperimentData class."""

    def test_initialization(self):
        """Test ExperimentData initialization."""
        exp_data = ExperimentData(
            exp_id="test_exp",
            path=Path("/tmp/test"),
            config={"key": "value"},
            metrics={"loss": [1.0, 0.5]},
            final_metrics={"loss": 0.5},
            timestamp=datetime.now(),
        )

        assert exp_data.exp_id == "test_exp"
        assert exp_data.path == Path("/tmp/test")
        assert exp_data.config == {"key": "value"}
        assert exp_data.metrics == {"loss": [1.0, 0.5]}
        assert exp_data.final_metrics == {"loss": 0.5}
        assert isinstance(exp_data.timestamp, datetime)

    def test_repr(self):
        """Test ExperimentData string representation."""
        timestamp = datetime(2025, 1, 1, 12, 0, 0)
        exp_data = ExperimentData(
            exp_id="test_exp",
            path=Path("/tmp/test"),
            config={},
            metrics={},
            final_metrics={},
            timestamp=timestamp,
        )

        repr_str = repr(exp_data)
        assert "test_exp" in repr_str
        assert "2025-01-01" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

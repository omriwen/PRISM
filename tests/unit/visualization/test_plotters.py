"""Unit tests for spids.visualization.plotters module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from matplotlib.figure import Figure

from prism.visualization import (
    DASHBOARD,
    LearningCurvesPlotter,
    ReconstructionComparisonPlotter,
    SyntheticAperturePlotter,
    TrainingVisualizer,
)


@pytest.fixture
def mock_telescope() -> MagicMock:
    """Create a mock telescope object."""
    telescope = MagicMock()
    telescope.n = torch.tensor(128)
    telescope.r = torch.tensor(10.0)
    telescope.cum_mask = torch.zeros(128, 128)
    telescope.x = torch.linspace(-64, 63, 128).unsqueeze(1).expand(128, 128)
    telescope.mask = MagicMock(return_value=torch.zeros(128, 128))
    telescope.measure_through_accumulated_mask = MagicMock(return_value=torch.randn(1, 1, 64, 64))
    telescope.__call__ = MagicMock(return_value=torch.randn(2, 1, 64, 64))
    return telescope


@pytest.fixture
def sample_tensors() -> dict[str, torch.Tensor]:
    """Create sample tensors for testing."""
    return {
        "ground_truth": torch.randn(1, 1, 128, 128),
        "reconstruction": torch.randn(1, 1, 128, 128),
        "static_measurement": torch.randn(1, 1, 128, 128),
    }


class TestReconstructionComparisonPlotter:
    """Tests for ReconstructionComparisonPlotter."""

    def test_creates_figure(self, sample_tensors: dict[str, torch.Tensor]) -> None:
        """Test that plot() creates a figure."""
        with ReconstructionComparisonPlotter(DASHBOARD) as plotter:
            fig = plotter.plot(
                ground_truth=sample_tensors["ground_truth"],
                reconstruction=sample_tensors["reconstruction"],
                static_measurement=sample_tensors["static_measurement"],
                obj_size=64,
            )
            assert isinstance(fig, Figure)

    def test_context_manager_cleanup(self, sample_tensors: dict[str, torch.Tensor]) -> None:
        """Test that context manager cleans up resources."""
        plotter = ReconstructionComparisonPlotter(DASHBOARD)
        with plotter:
            plotter.plot(
                ground_truth=sample_tensors["ground_truth"],
                reconstruction=sample_tensors["reconstruction"],
                static_measurement=sample_tensors["static_measurement"],
                obj_size=64,
            )
        # After context, figure should be closed
        # We can't directly test if figure is closed, but we verify no exceptions

    def test_save_creates_file(
        self, sample_tensors: dict[str, torch.Tensor], tmp_path: pytest.TempPathFactory
    ) -> None:
        """Test that save() creates an image file."""
        output_path = tmp_path / "test_reconstruction.png"  # type: ignore
        with ReconstructionComparisonPlotter(DASHBOARD) as plotter:
            plotter.plot(
                ground_truth=sample_tensors["ground_truth"],
                reconstruction=sample_tensors["reconstruction"],
                static_measurement=sample_tensors["static_measurement"],
                obj_size=64,
            )
            plotter.save(str(output_path))
        assert output_path.exists()

    def test_plot_with_difference(self, sample_tensors: dict[str, torch.Tensor]) -> None:
        """Test plot_with_difference method."""
        with ReconstructionComparisonPlotter(DASHBOARD) as plotter:
            fig = plotter.plot_with_difference(
                ground_truth=sample_tensors["ground_truth"],
                reconstruction=sample_tensors["reconstruction"],
                obj_size=64,
            )
            assert isinstance(fig, Figure)


class TestLearningCurvesPlotter:
    """Tests for LearningCurvesPlotter."""

    @pytest.fixture
    def sample_metrics(self) -> dict[str, list[float]]:
        """Create sample training metrics."""
        n_samples = 20
        return {
            "losses": [1.0 / (i + 1) for i in range(n_samples)],
            "ssims": [0.5 + 0.5 * i / n_samples for i in range(n_samples)],
            "psnrs": [15.0 + i for i in range(n_samples)],
        }

    def test_creates_figure(self, sample_metrics: dict[str, list[float]]) -> None:
        """Test that plot() creates a figure."""
        with LearningCurvesPlotter(DASHBOARD) as plotter:
            fig = plotter.plot(
                losses=sample_metrics["losses"],
                ssims=sample_metrics["ssims"],
                psnrs=sample_metrics["psnrs"],
            )
            assert isinstance(fig, Figure)

    def test_empty_data_handled(self) -> None:
        """Test that empty data is handled gracefully."""
        with LearningCurvesPlotter(DASHBOARD) as plotter:
            fig = plotter.plot(losses=[], ssims=[], psnrs=[])
            assert isinstance(fig, Figure)

    def test_plot_combined(self, sample_metrics: dict[str, list[float]]) -> None:
        """Test plot_combined method."""
        with LearningCurvesPlotter(DASHBOARD) as plotter:
            fig = plotter.plot_combined(
                losses=sample_metrics["losses"],
                ssims=sample_metrics["ssims"],
                psnrs=sample_metrics["psnrs"],
            )
            assert isinstance(fig, Figure)


class TestSyntheticAperturePlotter:
    """Tests for SyntheticAperturePlotter."""

    def test_creates_figure(self, mock_telescope: MagicMock) -> None:
        """Test that plot() creates a figure."""
        tensor = torch.randn(1, 1, 128, 128)
        with SyntheticAperturePlotter(DASHBOARD) as plotter:
            fig = plotter.plot(
                tensor=tensor,
                telescope_agg=mock_telescope,
                roi_diameter=100.0,
            )
            assert isinstance(fig, Figure)


class TestTrainingVisualizer:
    """Tests for TrainingVisualizer."""

    @pytest.fixture
    def full_mock_telescope(self) -> MagicMock:
        """Create a more complete mock telescope for training visualizer."""
        telescope = MagicMock()
        telescope.n = torch.tensor(128)
        telescope.r = torch.tensor(10.0)
        telescope.cum_mask = torch.zeros(128, 128)
        telescope.x = torch.linspace(-64, 63, 128).unsqueeze(1).expand(128, 128)

        # Return proper tensor for all methods
        telescope.mask.return_value = torch.zeros(128, 128)
        telescope.measure_through_accumulated_mask.return_value = torch.randn(64, 64)
        telescope.return_value = torch.randn(2, 1, 64, 64)

        return telescope

    def test_initializes_correctly(self) -> None:
        """Test that TrainingVisualizer initializes with config."""
        with TrainingVisualizer(DASHBOARD) as viz:
            assert viz.config == DASHBOARD
            assert viz.update_interval == 0.1  # default

    def test_custom_update_interval(self) -> None:
        """Test custom update interval."""
        with TrainingVisualizer(DASHBOARD, update_interval=0.5) as viz:
            assert viz.update_interval == 0.5

    def test_context_manager(self) -> None:
        """Test context manager behavior."""
        with TrainingVisualizer(DASHBOARD) as viz:
            assert viz is not None
        # Should not raise after exit

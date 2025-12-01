"""
Unit tests for animation generation functionality.

Tests the TrainingAnimator and MultiExperimentAnimator classes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prism.visualization.animation import MultiExperimentAnimator, TrainingAnimator


# Define MockTelescopeAgg at module level to avoid pickle issues
class MockTelescopeAgg:
    """Mock telescope aggregator for testing."""

    def __init__(self, init_im):
        if isinstance(init_im, np.ndarray):
            self.init_im = torch.from_numpy(init_im)
        else:
            self.init_im = init_im


class TestTrainingAnimator:
    """Test TrainingAnimator class."""

    @pytest.fixture
    def mock_checkpoint_path(self, tmp_path):
        """Create a mock checkpoint file for testing."""
        # Create mock checkpoint directory
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()

        # Create mock ground truth
        ground_truth = np.random.rand(256, 256).astype(np.float32)

        # Create mock telescope_agg with init_im
        telescope_agg = MockTelescopeAgg(ground_truth)

        # Create mock checkpoint with training history
        checkpoint = {
            "current_rec": torch.from_numpy(ground_truth + np.random.randn(256, 256) * 0.1),
            "telescope_agg": telescope_agg,
            "losses": [0.5, 0.3, 0.2, 0.1, 0.05, 0.01],
            "ssims": [0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
            "psnrs": [15.0, 18.0, 22.0, 25.0, 28.0, 30.0],
            "rmses": [20.0, 15.0, 10.0, 7.0, 5.0, 2.0],
        }

        checkpoint_path = exp_dir / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        return exp_dir

    @pytest.fixture
    def animator(self, mock_checkpoint_path):
        """Create an animator instance with mock checkpoint."""
        return TrainingAnimator(mock_checkpoint_path)

    def test_initialization(self, animator):
        """Test animator initialization."""
        assert animator.exp_path.exists()
        assert animator.checkpoint is not None
        assert animator.ground_truth is not None
        assert animator.ground_truth.shape == (256, 256)

    def test_initialization_missing_path(self, tmp_path):
        """Test initialization with non-existent path raises error."""
        fake_path = tmp_path / "nonexistent_experiment"
        with pytest.raises(FileNotFoundError):
            TrainingAnimator(fake_path)

    def test_initialization_missing_checkpoint(self, tmp_path):
        """Test initialization with missing checkpoint file raises error."""
        exp_dir = tmp_path / "experiment_no_checkpoint"
        exp_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            TrainingAnimator(exp_dir)

    def test_extract_ground_truth(self, animator):
        """Test ground truth extraction."""
        assert animator.ground_truth is not None
        assert isinstance(animator.ground_truth, np.ndarray)
        assert animator.ground_truth.ndim == 2

    def test_get_reconstruction_history(self, animator):
        """Test reconstruction history generation."""
        n_frames = 10
        history = animator._get_reconstruction_history(n_frames)

        assert len(history) == n_frames
        assert all(isinstance(frame, np.ndarray) for frame in history)
        assert all(frame.shape == (256, 256) for frame in history)

    def test_get_reconstruction_history_default_frames(self, animator):
        """Test reconstruction history with default number of frames."""
        history = animator._get_reconstruction_history()

        # Should use min(history_length, 100), where history_length = 6
        assert len(history) == 6
        assert all(isinstance(frame, np.ndarray) for frame in history)

    def test_ease_in_out(self):
        """Test easing function."""
        # Test boundary conditions
        assert TrainingAnimator._ease_in_out(0.0) == 0.0
        assert TrainingAnimator._ease_in_out(1.0) == 1.0

        # Test smoothness (derivative at boundaries should be 0)
        # For t=0: 3t^2 - 2t^3 => derivative = 6t - 6t^2 => at t=0, derivative = 0
        # For t=1: derivative = 6 - 6 = 0

        # Test mid-point
        mid_value = TrainingAnimator._ease_in_out(0.5)
        assert 0.4 < mid_value < 0.6  # Should be roughly in the middle

    def test_get_metrics_at_frame(self, animator):
        """Test metrics extraction for specific frame."""
        total_frames = 6
        metrics = animator._get_metrics_at_frame(5, total_frames)

        assert "loss" in metrics
        assert "ssim" in metrics
        assert "psnr" in metrics
        assert "rmse" in metrics
        assert "sample" in metrics
        assert "total_samples" in metrics

        # Last frame should have final metrics
        assert metrics["loss"] == 0.01
        assert metrics["ssim"] == 0.95
        assert metrics["psnr"] == 30.0
        assert metrics["total_samples"] == 6

    def test_create_frame(self, animator):
        """Test frame creation."""
        reconstruction = np.random.rand(256, 256)
        metrics = {"loss": 0.1, "ssim": 0.9, "psnr": 25.0, "sample": 50, "total_samples": 100}

        frame = animator.create_frame(
            frame_idx=0,
            reconstruction=reconstruction,
            metrics=metrics,
            show_metrics=True,
            show_difference=True,
        )

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3  # RGB image
        assert frame.shape[2] == 3  # RGB channels
        assert frame.dtype == np.uint8

    def test_create_frame_no_ground_truth(self, tmp_path):
        """Test frame creation when no ground truth is available."""
        # Create checkpoint without ground truth
        exp_dir = tmp_path / "no_gt_experiment"
        exp_dir.mkdir()

        checkpoint = {
            "current_rec": torch.randn(1, 1, 256, 256),
            "losses": [0.1],
            "ssims": [0.9],
            "psnrs": [25.0],
        }

        checkpoint_path = exp_dir / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        animator = TrainingAnimator(exp_dir)
        assert animator.ground_truth is None

        reconstruction = np.random.rand(256, 256)
        metrics = {"loss": 0.1}

        frame = animator.create_frame(
            frame_idx=0, reconstruction=reconstruction, metrics=metrics, show_difference=False
        )

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3

    def test_generate_video_requires_opencv(self, animator, tmp_path):
        """Test that video generation fails gracefully without opencv."""
        output_path = tmp_path / "test.mp4"

        # This test will pass if opencv is installed, or raise ImportError if not
        # We just want to verify the function exists and handles the dependency
        try:
            animator.generate_video(output_path, fps=2, n_frames=2)
            # If successful, check output exists
            assert output_path.exists()
        except ImportError as e:
            # Expected if opencv not installed
            assert "opencv" in str(e).lower()

    def test_generate_gif(self, animator, tmp_path):
        """Test GIF generation."""
        output_path = tmp_path / "test.gif"

        # Generate small GIF for testing
        animator.generate_gif(output_path, duration=100, n_frames=3)

        # Check output exists
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_generate_gif_custom_settings(self, animator, tmp_path):
        """Test GIF generation with custom settings."""
        output_path = tmp_path / "custom.gif"

        animator.generate_gif(
            output_path, duration=50, n_frames=5, show_metrics=False, show_difference=False, loop=1
        )

        assert output_path.exists()


class TestMultiExperimentAnimator:
    """Test MultiExperimentAnimator class."""

    @pytest.fixture
    def mock_experiments(self, tmp_path):
        """Create multiple mock experiment directories."""
        experiments = []

        for i in range(3):
            exp_dir = tmp_path / f"experiment_{i}"
            exp_dir.mkdir()

            # Create varied checkpoints
            ground_truth = np.random.rand(128, 128).astype(np.float32)

            telescope_agg = MockTelescopeAgg(ground_truth)

            # Each experiment has different metrics
            checkpoint = {
                "current_rec": torch.from_numpy(ground_truth + np.random.randn(128, 128) * 0.1),
                "telescope_agg": telescope_agg,
                "losses": [0.5 - i * 0.1, 0.3 - i * 0.05, 0.1 - i * 0.02],
                "ssims": [0.6 + i * 0.05, 0.8 + i * 0.03, 0.9 + i * 0.02],
                "psnrs": [15.0 + i * 2, 20.0 + i * 2, 25.0 + i * 2],
            }

            checkpoint_path = exp_dir / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            experiments.append(exp_dir)

        return experiments

    @pytest.fixture
    def multi_animator(self, mock_experiments):
        """Create multi-experiment animator."""
        return TrainingAnimator.from_multiple(mock_experiments)

    def test_initialization(self, multi_animator):
        """Test multi-animator initialization."""
        assert isinstance(multi_animator, MultiExperimentAnimator)
        assert multi_animator.n_experiments == 3
        assert len(multi_animator.animators) == 3

    def test_initialization_too_few_experiments(self, tmp_path):
        """Test initialization with too few experiments raises error."""
        # Create single experiment
        exp_dir = tmp_path / "single_exp"
        exp_dir.mkdir()

        checkpoint = {"current_rec": torch.randn(128, 128), "losses": [0.1]}
        torch.save(checkpoint, exp_dir / "checkpoint.pt")

        with pytest.raises(ValueError, match="at least 2 experiments"):
            TrainingAnimator.from_multiple([exp_dir])

    def test_initialization_too_many_experiments(self, mock_experiments, tmp_path):
        """Test initialization with too many experiments raises error."""
        # Create 5 experiments
        for i in range(5):
            exp_dir = tmp_path / f"extra_exp_{i}"
            exp_dir.mkdir()
            checkpoint = {"current_rec": torch.randn(128, 128), "losses": [0.1]}
            torch.save(checkpoint, exp_dir / "checkpoint.pt")
            mock_experiments.append(exp_dir)

        with pytest.raises(ValueError, match="Maximum 4 experiments"):
            TrainingAnimator.from_multiple(mock_experiments)

    def test_create_frame(self, multi_animator):
        """Test comparison frame creation."""
        reconstructions = [np.random.rand(128, 128) for _ in range(3)]
        metrics_list = [
            {"loss": 0.1 + i * 0.05, "ssim": 0.9 - i * 0.05, "sample": 50, "total_samples": 100}
            for i in range(3)
        ]

        frame = multi_animator.create_frame(
            frame_idx=0, reconstructions=reconstructions, metrics_list=metrics_list, layout="grid"
        )

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3

    def test_create_frame_horizontal_layout(self, multi_animator):
        """Test horizontal layout for comparison."""
        reconstructions = [np.random.rand(128, 128) for _ in range(3)]
        metrics_list = [{"loss": 0.1} for _ in range(3)]

        frame = multi_animator.create_frame(
            frame_idx=0,
            reconstructions=reconstructions,
            metrics_list=metrics_list,
            layout="horizontal",
        )

        assert isinstance(frame, np.ndarray)
        # Horizontal layout should be wider
        assert frame.shape[1] > frame.shape[0]

    def test_generate_gif(self, multi_animator, tmp_path):
        """Test comparison GIF generation."""
        output_path = tmp_path / "comparison.gif"

        multi_animator.generate_gif(output_path, duration=100, n_frames=2, layout="grid")

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_generate_video_requires_opencv(self, multi_animator, tmp_path):
        """Test that comparison video generation handles opencv dependency."""
        output_path = tmp_path / "comparison.mp4"

        try:
            multi_animator.generate_video(output_path, fps=2, n_frames=2, layout="horizontal")
            # If successful, check output exists
            assert output_path.exists()
        except ImportError as e:
            # Expected if opencv not installed
            assert "opencv" in str(e).lower()


class TestAnimationEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_reconstruction_field(self, tmp_path):
        """Test handling of checkpoint without current_rec."""
        exp_dir = tmp_path / "bad_checkpoint"
        exp_dir.mkdir()

        # Checkpoint missing required field
        checkpoint = {"losses": [0.1], "ssims": [0.9]}

        checkpoint_path = exp_dir / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(ValueError, match="missing reconstruction data"):
            TrainingAnimator(exp_dir)

    def test_empty_metrics_history(self, tmp_path):
        """Test handling of checkpoint with empty metric histories."""
        exp_dir = tmp_path / "empty_metrics"
        exp_dir.mkdir()

        checkpoint = {
            "current_rec": torch.randn(128, 128),
            "losses": [],  # Empty
        }

        checkpoint_path = exp_dir / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        animator = TrainingAnimator(exp_dir)
        metrics = animator._get_metrics_at_frame(0, 10)

        # Should return empty dict or default values
        assert isinstance(metrics, dict)

    def test_malformed_reconstruction_shape(self, tmp_path):
        """Test handling of unexpected reconstruction shape."""
        exp_dir = tmp_path / "bad_shape"
        exp_dir.mkdir()

        # 1D reconstruction (invalid)
        checkpoint = {
            "current_rec": torch.randn(128),  # 1D instead of 2D
            "losses": [0.1],
        }

        checkpoint_path = exp_dir / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        animator = TrainingAnimator(exp_dir)

        # Should handle gracefully during frame creation
        history = animator._get_reconstruction_history(n_frames=1)
        # The history will be created from the squeezed/reshaped data
        assert len(history) == 1

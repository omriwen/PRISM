"""
Coverage tests for runner.py.

These tests focus on initialization and basic attribute access
in the PRISMRunner class.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from prism.core.runner import PRISMRunner


@pytest.fixture
def mock_args():
    """Create mock args with required attributes."""
    args = Mock()

    # Device
    args.device = "cpu"
    args.cuda_device = 0

    # Image parameters
    args.image_size = 64
    args.dx = 1e-5
    args.dxf = 1.0 / (64 * 1e-5)
    args.obj_size = 64
    args.crop_obj = True

    # Physics
    args.wavelength = 520e-9
    args.obj_distance = 1.0
    args.sample_diameter = 0.01
    args.snr = None
    args.blur_image = False

    # Training
    args.n_epochs_init = 10
    args.max_epochs_init = 5
    args.batch_size = 1
    args.lr = 1e-3

    # Model
    args.model_type = "ProgressiveDecoder"
    args.mixed_precision = False
    args.output_activation = "tanh"

    # Logging
    args.log_dir = None
    args.save_interval = 100
    args.log_interval = 10

    # Pattern
    args.pattern_type = "double_fft"
    args.n_samples = 10

    return args


def test_runner_initialization(mock_args):
    """Test basic PRISMRunner initialization."""
    runner = PRISMRunner(args=mock_args)

    assert runner.args is mock_args
    assert runner.device is None  # Not set until setup()
    assert runner.log_dir is None
    assert runner.writer is None
    assert runner.config is None


def test_runner_initialization_state(mock_args):
    """Test that runner initializes with None state."""
    runner = PRISMRunner(args=mock_args)

    # All components should be None initially
    assert runner.telescope is None
    assert runner.measurement_system is None  # Changed from telescope_agg
    assert runner.model is None
    assert runner.optimizer is None
    assert runner.scheduler is None
    assert runner.trainer is None
    assert runner.image is None
    assert runner.image_gt is None
    assert runner.sample_centers is None
    assert runner.pattern_metadata is None
    assert runner.pattern_spec is None


def test_runner_args_access(mock_args):
    """Test that runner properly stores args."""
    runner = PRISMRunner(args=mock_args)

    assert runner.args is mock_args
    assert runner.args.image_size == 64
    assert runner.args.dx == 1e-5
    assert runner.args.wavelength == 520e-9


def test_runner_different_image_sizes(mock_args):
    """Test runner with different image sizes."""
    image_sizes = [64, 128, 256]

    for size in image_sizes:
        mock_args.image_size = size
        runner = PRISMRunner(args=mock_args)
        assert runner.args.image_size == size


def test_runner_different_wavelengths(mock_args):
    """Test runner with different wavelengths."""
    wavelengths = [400e-9, 520e-9, 700e-9]

    for wavelength in wavelengths:
        mock_args.wavelength = wavelength
        runner = PRISMRunner(args=mock_args)
        assert runner.args.wavelength == wavelength


def test_runner_with_mixed_precision(mock_args):
    """Test runner with mixed precision flag."""
    mock_args.mixed_precision = True
    runner = PRISMRunner(args=mock_args)
    assert runner.args.mixed_precision is True


def test_runner_with_snr(mock_args):
    """Test runner with SNR values."""
    snr_values = [None, 10.0, 20.0, 30.0]

    for snr in snr_values:
        mock_args.snr = snr
        runner = PRISMRunner(args=mock_args)
        assert runner.args.snr == snr


def test_runner_with_blur(mock_args):
    """Test runner with blur flag."""
    mock_args.blur_image = True
    runner = PRISMRunner(args=mock_args)
    assert runner.args.blur_image is True


def test_runner_different_pattern_types(mock_args):
    """Test runner with different pattern types."""
    pattern_types = ["double_fft", "hadamard", "random"]

    for pattern_type in pattern_types:
        mock_args.pattern_type = pattern_type
        runner = PRISMRunner(args=mock_args)
        assert runner.args.pattern_type == pattern_type


def test_runner_different_learning_rates(mock_args):
    """Test runner with different learning rates."""
    learning_rates = [1e-4, 1e-3, 1e-2]

    for lr in learning_rates:
        mock_args.lr = lr
        runner = PRISMRunner(args=mock_args)
        assert runner.args.lr == lr


def test_runner_different_batch_sizes(mock_args):
    """Test runner with different batch sizes."""
    batch_sizes = [1, 2, 4, 8]

    for batch_size in batch_sizes:
        mock_args.batch_size = batch_size
        runner = PRISMRunner(args=mock_args)
        assert runner.args.batch_size == batch_size


def test_runner_crop_obj_flag(mock_args):
    """Test runner with crop_obj flag."""
    for crop_obj in [True, False]:
        mock_args.crop_obj = crop_obj
        runner = PRISMRunner(args=mock_args)
        assert runner.args.crop_obj == crop_obj

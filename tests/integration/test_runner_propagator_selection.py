"""
Integration tests for PRISMRunner propagator selection.

Tests that the runner correctly integrates propagator selection from config.
"""

from __future__ import annotations

import pytest
import torch

from prism.config.base import PhysicsConfig, PRISMConfig
from prism.core.propagators import (
    AngularSpectrumPropagator,
    FraunhoferPropagator,
    FresnelPropagator,
)
from prism.core.runner import PRISMRunner


@pytest.fixture
def base_config():
    """Create a base config for testing."""
    return PRISMConfig(
        name="test_propagator_selection",
        save_data=False,
        physics=PhysicsConfig(
            wavelength=550e-9,  # 550 nm
            obj_distance=1e6,  # 1 Mm (far field)
            obj_diameter=10000,  # 10 km
            dxf=100.0,  # Frequency pixel size [1/m] - gives dx~10µm, FOV~5mm
        ),
    )


@pytest.fixture
def mock_args(base_config):
    """Create mock args from config."""
    import argparse

    args = argparse.Namespace()

    # Copy config values to args
    args.name = base_config.name
    args.save_data = base_config.save_data
    args.wavelength = base_config.physics.wavelength
    args.obj_distance = base_config.physics.obj_distance
    args.obj_diameter = base_config.physics.obj_diameter
    args.dxf = base_config.physics.dxf
    args.obj_name = "europa"

    # Image settings
    args.image_size = 512
    args.obj_size = 256
    args.crop_obj = False
    args.invert_image = False
    args.input = None

    # Telescope settings
    args.sample_diameter = 20.0
    args.sample_length = 0
    args.n_samples = 10
    args.snr = None
    args.blur_image = False
    args.roi_diameter = 512
    args.samples_r_cutoff = 256
    args.pattern_fn = "builtin:random"

    # Point source
    args.is_point_source = False

    # Training settings
    args.n_epochs = 10
    args.max_epochs = 1
    args.n_epochs_init = 10
    args.max_epochs_init = 1
    args.initialization_target = "circle"
    args.lr = 1e-3
    args.loss_th = 1e-3
    args.loss_type = "l1"
    args.use_amsgrad = False
    args.new_weight = 1.0
    args.f_weight = 1e-4

    # Model settings
    args.use_bn = True
    args.output_activation = "none"
    args.use_leaky = True
    args.middle_activation = "sigmoid"
    args.complex_data = False

    # Device
    args.use_cuda = False
    args.device_num = 0

    # Logging
    args.log_dir = "runs"
    args.log_level = "INFO"
    args.checkpoint = None

    # Line sampling
    args.samples_per_line_meas = None
    args.samples_per_line_rec = None

    # Calculate dx (normally done in load_image_and_pattern)
    args.dx = args.wavelength * args.obj_distance / (args.dxf * args.image_size)

    return args


def test_runner_auto_propagator_selection_fraunhofer(base_config, mock_args):
    """Test PRISMRunner with auto propagator selection (should select Fraunhofer)."""
    # Configure for far-field (Fraunhofer)
    base_config.telescope.propagator_method = "auto"
    base_config.physics.obj_distance = 1e6  # 1 Mm - very far
    mock_args.obj_distance = 1e6
    mock_args.dx = (
        mock_args.wavelength * mock_args.obj_distance / (mock_args.dxf * mock_args.image_size)
    )

    # Create runner
    runner = PRISMRunner(mock_args)
    runner.config = base_config
    runner.args = mock_args
    runner.device = torch.device("cpu")

    # Create model and telescope
    runner.create_model_and_telescope()

    # Verify propagator was selected and is Fraunhofer
    assert runner.telescope is not None
    assert runner.telescope.propagator is not None
    assert isinstance(runner.telescope.propagator, FraunhoferPropagator)


def test_runner_auto_propagator_selection_angular_spectrum(base_config, mock_args):
    """Test PRISMRunner with auto propagator selection (should select Angular Spectrum)."""
    # Configure for near-field (Angular Spectrum)
    # For F > 10, we need large FOV or small distance
    # With distance=1m, wavelength=550e-9, need FOV > 2.35mm for F > 10
    # Use dxf=1e-4 to get dx~10µm, FOV~5mm, F~48
    base_config.telescope.propagator_method = "auto"
    base_config.physics.obj_distance = 1.0  # 1 m
    base_config.physics.dxf = 1e-4  # Gives dx~10µm, FOV~5mm
    mock_args.obj_distance = 1.0
    mock_args.dxf = 1e-4
    mock_args.dx = (
        mock_args.wavelength * mock_args.obj_distance / (mock_args.dxf * mock_args.image_size)
    )

    # Create runner
    runner = PRISMRunner(mock_args)
    runner.config = base_config
    runner.args = mock_args
    runner.device = torch.device("cpu")

    # Create model and telescope
    runner.create_model_and_telescope()

    # Verify propagator was selected and is Angular Spectrum
    assert runner.telescope is not None
    assert runner.telescope.propagator is not None
    assert isinstance(runner.telescope.propagator, AngularSpectrumPropagator)


def test_runner_manual_propagator_selection_fraunhofer(base_config, mock_args):
    """Test PRISMRunner with manual Fraunhofer selection."""
    base_config.telescope.propagator_method = "fraunhofer"

    # Create runner
    runner = PRISMRunner(mock_args)
    runner.config = base_config
    runner.args = mock_args
    runner.device = torch.device("cpu")

    # Create model and telescope
    runner.create_model_and_telescope()

    # Verify Fraunhofer propagator
    assert runner.telescope is not None
    assert runner.telescope.propagator is not None
    assert isinstance(runner.telescope.propagator, FraunhoferPropagator)


def test_runner_manual_propagator_selection_fresnel(base_config, mock_args):
    """Test PRISMRunner with manual Fresnel selection."""
    base_config.telescope.propagator_method = "fresnel"
    base_config.physics.obj_distance = 1.0  # 1 m
    mock_args.obj_distance = 1.0
    mock_args.dx = (
        mock_args.wavelength * mock_args.obj_distance / (mock_args.dxf * mock_args.image_size)
    )

    # Create runner
    runner = PRISMRunner(mock_args)
    runner.config = base_config
    runner.args = mock_args
    runner.device = torch.device("cpu")

    # Create model and telescope
    runner.create_model_and_telescope()

    # Verify Fresnel propagator
    assert runner.telescope is not None
    assert runner.telescope.propagator is not None
    assert isinstance(runner.telescope.propagator, FresnelPropagator)


def test_runner_manual_propagator_selection_angular_spectrum(base_config, mock_args):
    """Test PRISMRunner with manual Angular Spectrum selection."""
    base_config.telescope.propagator_method = "angular_spectrum"

    # Create runner
    runner = PRISMRunner(mock_args)
    runner.config = base_config
    runner.args = mock_args
    runner.device = torch.device("cpu")

    # Create model and telescope
    runner.create_model_and_telescope()

    # Verify Angular Spectrum propagator
    assert runner.telescope is not None
    assert runner.telescope.propagator is not None
    assert isinstance(runner.telescope.propagator, AngularSpectrumPropagator)


def test_runner_no_propagator_method_backward_compatibility(base_config, mock_args):
    """Test PRISMRunner with no propagator_method (backward compatibility)."""
    # Don't set propagator_method (None by default)
    assert base_config.telescope.propagator_method is None

    # Create runner
    runner = PRISMRunner(mock_args)
    runner.config = base_config
    runner.args = mock_args
    runner.device = torch.device("cpu")

    # Create model and telescope
    runner.create_model_and_telescope()

    # Verify propagator was created (FreeSpacePropagator/FresnelPropagator)
    assert runner.telescope is not None
    assert runner.telescope.propagator is not None
    assert isinstance(runner.telescope.propagator, FresnelPropagator)


def test_runner_europa_scenario(base_config, mock_args):
    """Test Europa observation scenario - should auto-select Fraunhofer."""
    # Europa parameters
    # At distance=6.28e11m, wavelength=550nm, need FOV < 186m for F < 0.1
    # Use dxf=2000 to get FOV~173m, F~0.085
    base_config.telescope.propagator_method = "auto"
    base_config.physics.wavelength = 550e-9  # 550 nm
    base_config.physics.obj_distance = 6.28e11  # ~628 million km
    base_config.physics.obj_diameter = 3121e3  # Europa diameter ~3121 km
    base_config.physics.dxf = 2000.0  # Gives FOV~173m for far-field

    mock_args.wavelength = 550e-9
    mock_args.obj_distance = 6.28e11
    mock_args.obj_diameter = 3121e3
    mock_args.dxf = 2000.0
    mock_args.dx = (
        mock_args.wavelength * mock_args.obj_distance / (mock_args.dxf * mock_args.image_size)
    )

    # Create runner
    runner = PRISMRunner(mock_args)
    runner.config = base_config
    runner.args = mock_args
    runner.device = torch.device("cpu")

    # Create model and telescope
    runner.create_model_and_telescope()

    # Verify Fraunhofer selected (far field)
    assert runner.telescope is not None
    assert runner.telescope.propagator is not None
    assert isinstance(runner.telescope.propagator, FraunhoferPropagator)

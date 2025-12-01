"""Test CLI propagator method argument."""

from __future__ import annotations

import pytest

from prism.cli.parser import create_main_parser
from prism.config import args_to_config


def test_propagator_method_cli_argument():
    """Test that --propagator-method CLI argument is properly parsed."""
    parser = create_main_parser()

    # Test with auto
    args = parser.parse_args(["--propagator-method", "auto", "--name", "test"])
    assert args.propagator_method == "auto"

    # Test with fraunhofer
    args = parser.parse_args(["--propagator-method", "fraunhofer", "--name", "test"])
    assert args.propagator_method == "fraunhofer"

    # Test with fresnel
    args = parser.parse_args(["--propagator-method", "fresnel", "--name", "test"])
    assert args.propagator_method == "fresnel"

    # Test with angular_spectrum
    args = parser.parse_args(["--propagator-method", "angular_spectrum", "--name", "test"])
    assert args.propagator_method == "angular_spectrum"

    # Test default (None)
    args = parser.parse_args(["--name", "test"])
    assert args.propagator_method is None


def test_propagator_method_invalid_choice():
    """Test that invalid propagator method raises error."""
    parser = create_main_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--propagator-method", "invalid", "--name", "test"])


def test_propagator_method_maps_to_config():
    """Test that propagator method is properly mapped to config."""
    parser = create_main_parser()

    # Test with auto
    args = parser.parse_args(["--propagator-method", "auto", "--name", "test"])
    config = args_to_config(args)
    assert config.telescope.propagator_method == "auto"

    # Test with fraunhofer
    args = parser.parse_args(["--propagator-method", "fraunhofer", "--name", "test"])
    config = args_to_config(args)
    assert config.telescope.propagator_method == "fraunhofer"

    # Test default (None)
    args = parser.parse_args(["--name", "test"])
    config = args_to_config(args)
    assert config.telescope.propagator_method is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Unit tests for illumination source models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prism.core.illumination import (
    ContrastMode,
    IlluminationConfig,
    LaserSource,
    LEDSource,
    SolarSource,
    SourceGeometry,
    create_illumination_source,
    validate_bf_df_configuration,
)


class TestIlluminationConfig:
    """Test IlluminationConfig dataclass."""

    def test_valid_brightfield_config(self) -> None:
        """Test valid brightfield configuration."""
        config = IlluminationConfig(
            na_objective=0.9,
            na_condenser=0.7,
            wavelength=532e-9,
            mode="brightfield",
        )
        assert config.na_objective == 0.9
        assert config.na_condenser == 0.7
        assert config.mode == "brightfield"

    def test_valid_darkfield_config(self) -> None:
        """Test valid darkfield configuration."""
        config = IlluminationConfig(
            na_objective=0.9,
            na_condenser=1.2,
            wavelength=532e-9,
            mode="darkfield",
        )
        assert config.na_objective == 0.9
        assert config.na_condenser == 1.2
        assert config.mode == "darkfield"

    def test_brightfield_na_constraint(self) -> None:
        """Test brightfield requires NA_cond <= NA_obj."""
        with pytest.raises(ValueError, match="Brightfield requires"):
            IlluminationConfig(
                na_objective=0.9,
                na_condenser=1.0,  # > NA_obj, invalid for BF
                wavelength=532e-9,
                mode="brightfield",
            )

    def test_darkfield_na_constraint(self) -> None:
        """Test darkfield requires NA_cond > NA_obj."""
        with pytest.raises(ValueError, match="Darkfield requires"):
            IlluminationConfig(
                na_objective=0.9,
                na_condenser=0.8,  # <= NA_obj, invalid for DF
                wavelength=532e-9,
                mode="darkfield",
            )

    def test_na_exceeds_medium_index(self) -> None:
        """Test NA cannot exceed medium index."""
        with pytest.raises(ValueError, match="cannot exceed"):
            IlluminationConfig(
                na_objective=1.2,
                na_condenser=1.0,
                wavelength=532e-9,
                medium_index=1.0,  # NA > medium_index
                mode="brightfield",
            )

    def test_negative_values_rejected(self) -> None:
        """Test negative values are rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            IlluminationConfig(
                na_objective=-0.9,
                na_condenser=0.7,
                wavelength=532e-9,
            )

    def test_cutoff_frequency(self) -> None:
        """Test cutoff frequency calculation."""
        config = IlluminationConfig(
            na_objective=0.9,
            na_condenser=0.7,
            wavelength=500e-9,
            mode="brightfield",
        )
        expected = 0.9 / 500e-9  # NA / wavelength
        assert np.isclose(config.cutoff_frequency, expected)


class TestSourceGeometry:
    """Test SourceGeometry static methods."""

    def test_disk_grid_shape(self) -> None:
        """Test disk grid produces correct output shapes."""
        positions, weights = SourceGeometry.disk_grid(
            na=0.7,
            wavelength=532e-9,
            n_points=100,
        )
        assert positions.dim() == 2
        assert positions.shape[1] == 2  # (N, 2)
        assert weights.dim() == 1
        assert weights.shape[0] == positions.shape[0]

    def test_disk_grid_weights_normalized(self) -> None:
        """Test disk grid weights sum to 1."""
        positions, weights = SourceGeometry.disk_grid(
            na=0.7,
            wavelength=532e-9,
            n_points=100,
        )
        assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

    def test_disk_grid_within_na(self) -> None:
        """Test all disk grid points are within NA limit."""
        na = 0.7
        wavelength = 532e-9
        positions, _ = SourceGeometry.disk_grid(
            na=na,
            wavelength=wavelength,
            n_points=100,
        )
        radii = torch.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
        max_freq = na / wavelength
        assert (radii <= max_freq * 1.01).all()  # Small tolerance

    def test_annular_grid_shape(self) -> None:
        """Test annular grid produces correct output shapes."""
        positions, weights = SourceGeometry.annular_grid(
            na_inner=0.9,
            na_outer=1.2,
            wavelength=532e-9,
            n_points=100,
        )
        assert positions.dim() == 2
        assert positions.shape[1] == 2
        assert weights.dim() == 1
        assert weights.shape[0] == positions.shape[0]

    def test_annular_grid_within_bounds(self) -> None:
        """Test annular grid points are within NA bounds."""
        na_inner = 0.9
        na_outer = 1.2
        wavelength = 532e-9
        positions, _ = SourceGeometry.annular_grid(
            na_inner=na_inner,
            na_outer=na_outer,
            wavelength=wavelength,
            n_points=100,
        )
        radii = torch.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
        freq_inner = na_inner / wavelength
        freq_outer = na_outer / wavelength

        # All points should be in the annular region
        assert (radii >= freq_inner * 0.99).all()
        assert (radii <= freq_outer * 1.01).all()

    def test_annular_grid_invalid_na(self) -> None:
        """Test annular grid rejects invalid NA ordering."""
        with pytest.raises(ValueError, match="na_inner must be less than na_outer"):
            SourceGeometry.annular_grid(
                na_inner=1.2,
                na_outer=0.9,  # inner > outer
                wavelength=532e-9,
                n_points=100,
            )

    def test_single_point(self) -> None:
        """Test single point source."""
        positions, weights = SourceGeometry.single_point()
        assert positions.shape == (1, 2)
        assert weights.shape == (1,)
        assert torch.allclose(positions, torch.zeros(1, 2))
        assert torch.allclose(weights, torch.ones(1))

    def test_oblique_point(self) -> None:
        """Test oblique point source."""
        na = 1.0
        wavelength = 532e-9
        positions, weights = SourceGeometry.oblique_point(
            na=na,
            wavelength=wavelength,
            azimuth=0.0,  # Along x-axis
        )
        assert positions.shape == (1, 2)
        assert weights.shape == (1,)

        # Should be at (NA/wavelength, 0) for azimuth=0
        expected_x = na / wavelength
        assert torch.isclose(positions[0, 0], torch.tensor(expected_x, dtype=torch.float32))
        assert torch.isclose(positions[0, 1], torch.tensor(0.0))

    def test_oblique_point_azimuth(self) -> None:
        """Test oblique point with different azimuth angles."""
        na = 1.0
        wavelength = 532e-9

        # Test 90 degrees (along y-axis)
        positions, _ = SourceGeometry.oblique_point(
            na=na,
            wavelength=wavelength,
            azimuth=np.pi / 2,
        )
        expected_y = na / wavelength
        assert torch.isclose(positions[0, 0], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(
            positions[0, 1], torch.tensor(expected_y, dtype=torch.float32), atol=1e-6
        )

    def test_device_placement(self) -> None:
        """Test tensors are placed on correct device."""
        device = torch.device("cpu")
        positions, weights = SourceGeometry.disk_grid(
            na=0.7,
            wavelength=532e-9,
            n_points=50,
            device=device,
        )
        assert positions.device == device
        assert weights.device == device


class TestLEDSource:
    """Test LED source model."""

    def test_led_brightfield(self) -> None:
        """Test LED source in brightfield mode."""
        config = IlluminationConfig(
            na_objective=0.9,
            na_condenser=0.7,
            wavelength=532e-9,
            mode="brightfield",
        )
        led = LEDSource(config)
        positions, weights = led.get_source_grid(n_points=50)

        assert positions.shape[0] > 0
        assert weights.sum().item() == pytest.approx(1.0, abs=1e-6)

        # Check positions are within condenser NA
        radii = torch.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
        max_freq = config.na_condenser / config.wavelength
        assert (radii <= max_freq * 1.01).all()

    def test_led_darkfield(self) -> None:
        """Test LED source in darkfield mode."""
        config = IlluminationConfig(
            na_objective=0.9,
            na_condenser=1.2,
            wavelength=532e-9,
            mode="darkfield",
        )
        led = LEDSource(config)
        positions, weights = led.get_source_grid(n_points=50)

        assert positions.shape[0] > 0

        # Check positions are OUTSIDE objective NA
        radii = torch.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
        freq_obj = config.na_objective / config.wavelength
        assert (radii >= freq_obj).all()


class TestLaserSource:
    """Test laser source model."""

    def test_laser_brightfield(self) -> None:
        """Test laser source in brightfield mode (on-axis)."""
        config = IlluminationConfig(
            na_objective=0.9,
            na_condenser=0.7,
            wavelength=532e-9,
            mode="brightfield",
        )
        laser = LaserSource(config)
        positions, weights = laser.get_source_grid()

        assert positions.shape == (1, 2)
        assert torch.allclose(positions, torch.zeros(1, 2))

    def test_laser_darkfield(self) -> None:
        """Test laser source in darkfield mode (off-axis)."""
        config = IlluminationConfig(
            na_objective=0.9,
            na_condenser=1.2,
            wavelength=532e-9,
            mode="darkfield",
        )
        laser = LaserSource(config, azimuth=0.0)
        positions, weights = laser.get_source_grid()

        assert positions.shape == (1, 2)
        # Position should be off-axis (not at origin)
        assert not torch.allclose(positions, torch.zeros(1, 2))

        # Should be tilted beyond objective NA
        radius = torch.sqrt(positions[0, 0] ** 2 + positions[0, 1] ** 2)
        freq_obj = config.na_objective / config.wavelength
        assert radius > freq_obj

    def test_laser_ignores_n_points(self) -> None:
        """Test laser always returns single point regardless of n_points."""
        config = IlluminationConfig(
            na_objective=0.9,
            na_condenser=0.7,
            wavelength=532e-9,
            mode="brightfield",
        )
        laser = LaserSource(config)
        positions, weights = laser.get_source_grid(n_points=1000)
        assert positions.shape == (1, 2)


class TestSolarSource:
    """Test solar source model."""

    def test_solar_direct_brightfield(self) -> None:
        """Test direct solar in brightfield (coherent-like)."""
        config = IlluminationConfig(
            na_objective=0.4,
            na_condenser=0.3,
            wavelength=550e-9,
            mode="brightfield",
        )
        solar = SolarSource(config, diffused=False)
        positions, weights = solar.get_source_grid()

        assert positions.shape == (1, 2)
        assert torch.allclose(positions, torch.zeros(1, 2))

    def test_solar_direct_darkfield_invalid(self) -> None:
        """Test direct solar in darkfield is invalid."""
        config = IlluminationConfig(
            na_objective=0.4,
            na_condenser=0.5,
            wavelength=550e-9,
            mode="darkfield",
        )
        solar = SolarSource(config, diffused=False)

        with pytest.raises(ValueError, match="not compatible with darkfield"):
            solar.get_source_grid()

    def test_solar_diffused_brightfield(self) -> None:
        """Test diffused solar in brightfield (like LED)."""
        config = IlluminationConfig(
            na_objective=0.4,
            na_condenser=0.3,
            wavelength=550e-9,
            mode="brightfield",
        )
        solar = SolarSource(config, diffused=True)
        positions, weights = solar.get_source_grid(n_points=50)

        assert positions.shape[0] > 1  # Multiple points

    def test_solar_diffused_darkfield(self) -> None:
        """Test diffused solar in darkfield."""
        config = IlluminationConfig(
            na_objective=0.4,
            na_condenser=0.6,
            wavelength=550e-9,
            mode="darkfield",
        )
        solar = SolarSource(config, diffused=True)
        positions, weights = solar.get_source_grid(n_points=50)

        assert positions.shape[0] > 1

        # Check positions are outside objective NA
        radii = torch.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
        freq_obj = config.na_objective / config.wavelength
        assert (radii >= freq_obj).all()


class TestCreateIlluminationSource:
    """Test factory function."""

    def test_create_led(self) -> None:
        """Test creating LED source via factory."""
        config = IlluminationConfig(
            na_objective=0.9,
            na_condenser=0.7,
            wavelength=532e-9,
            mode="brightfield",
        )
        source = create_illumination_source("led", config)
        assert isinstance(source, LEDSource)

    def test_create_laser(self) -> None:
        """Test creating laser source via factory."""
        config = IlluminationConfig(
            na_objective=0.9,
            na_condenser=0.7,
            wavelength=532e-9,
            mode="brightfield",
        )
        source = create_illumination_source("laser", config, azimuth=np.pi / 4)
        assert isinstance(source, LaserSource)
        assert source.azimuth == np.pi / 4

    def test_create_solar(self) -> None:
        """Test creating solar source via factory."""
        config = IlluminationConfig(
            na_objective=0.4,
            na_condenser=0.3,
            wavelength=550e-9,
            mode="brightfield",
        )
        source = create_illumination_source("solar", config, diffused=True)
        assert isinstance(source, SolarSource)
        assert source.diffused is True

    def test_create_unknown_type(self) -> None:
        """Test factory rejects unknown source type."""
        config = IlluminationConfig(
            na_objective=0.9,
            na_condenser=0.7,
            wavelength=532e-9,
            mode="brightfield",
        )
        with pytest.raises(ValueError, match="Unknown source type"):
            create_illumination_source("halogen", config)

    def test_case_insensitive(self) -> None:
        """Test factory is case-insensitive."""
        config = IlluminationConfig(
            na_objective=0.9,
            na_condenser=0.7,
            wavelength=532e-9,
            mode="brightfield",
        )
        source = create_illumination_source("LED", config)
        assert isinstance(source, LEDSource)


class TestValidateBFDFConfiguration:
    """Test configuration validation helper."""

    def test_valid_brightfield(self) -> None:
        """Test valid brightfield passes validation."""
        assert validate_bf_df_configuration(
            na_objective=0.9,
            na_condenser=0.7,
            mode="brightfield",
        )

    def test_valid_darkfield(self) -> None:
        """Test valid darkfield passes validation."""
        assert validate_bf_df_configuration(
            na_objective=0.9,
            na_condenser=1.2,
            mode="darkfield",
        )

    def test_invalid_brightfield(self) -> None:
        """Test invalid brightfield raises error."""
        with pytest.raises(ValueError, match="Brightfield requires"):
            validate_bf_df_configuration(
                na_objective=0.9,
                na_condenser=1.0,
                mode="brightfield",
            )

    def test_invalid_darkfield(self) -> None:
        """Test invalid darkfield raises error."""
        with pytest.raises(ValueError, match="Darkfield requires"):
            validate_bf_df_configuration(
                na_objective=0.9,
                na_condenser=0.8,
                mode="darkfield",
            )

    def test_unknown_mode(self) -> None:
        """Test unknown mode raises error."""
        with pytest.raises(ValueError, match="Unknown mode"):
            validate_bf_df_configuration(
                na_objective=0.9,
                na_condenser=0.7,
                mode="fluorescence",
            )


class TestContrastMode:
    """Test ContrastMode enum."""

    def test_enum_values(self) -> None:
        """Test enum has expected values."""
        assert ContrastMode.BRIGHTFIELD.value == "brightfield"
        assert ContrastMode.DARKFIELD.value == "darkfield"

    def test_enum_string_comparison(self) -> None:
        """Test enum can be compared to strings."""
        assert ContrastMode.BRIGHTFIELD == "brightfield"
        assert ContrastMode.DARKFIELD == "darkfield"

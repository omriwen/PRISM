"""Unit tests for unified Telescope implementation.

Tests the new Telescope class from prism.core.instruments.telescope which
implements pure optical physics without SPIDS-specific measurement logic.
"""

from __future__ import annotations

import pytest
import torch

from prism.core.apertures import CircularAperture
from prism.core.instruments import Telescope, TelescopeConfig, create_instrument
from prism.core.instruments.base import Instrument


class TestTelescopeConfig:
    """Test TelescopeConfig validation and initialization."""

    def test_config_creation_defaults(self) -> None:
        """Test config with default values."""
        config = TelescopeConfig()
        assert config.wavelength == 550e-9
        assert config.n_pixels == 1024
        assert config.aperture_radius_pixels == 10.0
        assert config.aperture_type == "circular"
        assert config.snr is None

    def test_config_creation_custom(self) -> None:
        """Test config with custom values."""
        config = TelescopeConfig(
            n_pixels=512,
            aperture_radius_pixels=25.0,
            aperture_diameter=8.2,  # VLT
            wavelength=550e-9,
            snr=40.0,
            focal_length=120.0,
        )
        assert config.n_pixels == 512
        assert config.aperture_radius_pixels == 25.0
        assert config.aperture_diameter == 8.2
        assert config.snr == 40.0
        assert config.focal_length == 120.0

    def test_config_validation_negative_radius(self) -> None:
        """Test validation fails for negative aperture radius."""
        with pytest.raises(ValueError, match="Aperture radius must be positive"):
            TelescopeConfig(aperture_radius_pixels=-5.0)

    def test_config_validation_negative_diameter(self) -> None:
        """Test validation fails for negative aperture diameter."""
        with pytest.raises(ValueError, match="Aperture diameter must be positive"):
            TelescopeConfig(aperture_diameter=-8.0)

    def test_config_validation_negative_focal_length(self) -> None:
        """Test validation fails for negative focal length."""
        with pytest.raises(ValueError, match="Focal length must be positive"):
            TelescopeConfig(focal_length=-10.0)

    def test_config_validation_negative_snr(self) -> None:
        """Test validation fails for negative SNR."""
        with pytest.raises(ValueError, match="SNR must be positive"):
            TelescopeConfig(snr=-10.0)

    def test_config_validation_invalid_aperture_type(self) -> None:
        """Test validation fails for invalid aperture type."""
        with pytest.raises(ValueError, match="Unknown aperture type"):
            TelescopeConfig(aperture_type="invalid")

    def test_config_aperture_kwargs_initialization(self) -> None:
        """Test aperture_kwargs is properly initialized."""
        config = TelescopeConfig()
        assert config.aperture_kwargs == {}

        config2 = TelescopeConfig(aperture_kwargs={"radius": 15.0})
        assert config2.aperture_kwargs == {"radius": 15.0}


class TestTelescopeCreation:
    """Test Telescope instantiation and initialization."""

    def test_telescope_creation(self) -> None:
        """Test basic telescope creation."""
        config = TelescopeConfig(n_pixels=256, aperture_radius_pixels=20.0)
        telescope = Telescope(config)
        assert telescope is not None
        assert isinstance(telescope, Instrument)

    def test_telescope_inherits_instrument(self) -> None:
        """Test that Telescope inherits from Instrument."""
        config = TelescopeConfig()
        telescope = Telescope(config)
        assert isinstance(telescope, Instrument)

    def test_telescope_does_not_inherit_nn_module(self) -> None:
        """Test that new Telescope does NOT inherit from nn.Module."""
        config = TelescopeConfig()
        telescope = Telescope(config)
        # Should not have nn.Module methods
        assert not hasattr(telescope, "parameters") or callable(telescope.parameters) is False

    def test_telescope_type(self) -> None:
        """Test get_instrument_type returns 'telescope'."""
        config = TelescopeConfig()
        telescope = Telescope(config)
        assert telescope.get_instrument_type() == "telescope"

    def test_circular_aperture_created(self) -> None:
        """Test circular aperture is created by default."""
        config = TelescopeConfig(aperture_radius_pixels=15.0)
        telescope = Telescope(config)
        assert isinstance(telescope.aperture, CircularAperture)
        assert telescope.aperture.radius == 15.0

    def test_noise_model_created_with_snr(self) -> None:
        """Test noise model is created when SNR is specified."""
        config = TelescopeConfig(snr=40.0)
        telescope = Telescope(config)
        assert telescope.noise_model is not None

    def test_no_noise_model_without_snr(self) -> None:
        """Test noise model is None when SNR is not specified."""
        config = TelescopeConfig()
        telescope = Telescope(config)
        assert telescope.noise_model is None


class TestTelescopeInstrumentInterface:
    """Test Telescope implements Instrument interface correctly."""

    def test_has_compute_psf(self) -> None:
        """Test compute_psf method exists."""
        config = TelescopeConfig()
        telescope = Telescope(config)
        assert hasattr(telescope, "compute_psf")
        assert callable(telescope.compute_psf)

    def test_has_forward(self) -> None:
        """Test forward method exists with correct signature."""
        config = TelescopeConfig()
        telescope = Telescope(config)
        assert hasattr(telescope, "forward")
        assert callable(telescope.forward)

    def test_has_resolution_limit(self) -> None:
        """Test resolution_limit property exists."""
        config = TelescopeConfig()
        telescope = Telescope(config)
        assert hasattr(telescope, "resolution_limit")

    def test_forward_signature_matches_instrument(self) -> None:
        """Test forward has standard Instrument signature."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)

        # Create test field
        field = torch.ones(128, 128, dtype=torch.complex64)

        # Should accept standard Instrument parameters
        output = telescope.forward(
            field,
            illumination_mode=None,
            illumination_params=None,
            aperture_center=None,
            add_noise=False,
        )
        assert output is not None


class TestTelescopePSF:
    """Test PSF computation."""

    def test_psf_shape(self) -> None:
        """Test PSF has correct shape."""
        config = TelescopeConfig(n_pixels=256)
        telescope = Telescope(config)
        psf = telescope.compute_psf()

        assert psf.shape == (256, 256)

    def test_psf_normalized(self) -> None:
        """Test PSF is normalized to peak value of 1."""
        config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=10.0)
        telescope = Telescope(config)
        psf = telescope.compute_psf()

        assert psf.max() <= 1.0 + 1e-6  # Allow small numerical error
        assert psf.max() >= 0.99  # Should be close to 1

    def test_psf_non_negative(self) -> None:
        """Test PSF is non-negative (intensity)."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)
        psf = telescope.compute_psf()

        assert psf.min() >= 0

    def test_psf_with_center(self) -> None:
        """Test PSF computation with custom aperture center."""
        config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=10.0)
        telescope = Telescope(config)
        psf = telescope.compute_psf(center=[0, 0])

        assert psf.shape == (128, 128)
        assert psf.max() <= 1.0 + 1e-6

    def test_psf_airy_disk_pattern(self) -> None:
        """Test PSF resembles Airy disk (central peak)."""
        config = TelescopeConfig(n_pixels=256, aperture_radius_pixels=20.0)
        telescope = Telescope(config)
        psf = telescope.compute_psf()

        # Check central peak is brightest
        center = psf.shape[0] // 2
        center_value = psf[center, center]

        # Central region should be bright
        assert center_value > 0.8


class TestTelescopeResolution:
    """Test resolution limit calculations."""

    def test_resolution_with_diameter(self) -> None:
        """Test resolution calculation with explicit diameter."""
        config = TelescopeConfig(
            aperture_diameter=8.2,  # VLT 8.2m
            wavelength=550e-9,
        )
        telescope = Telescope(config)

        # Rayleigh criterion: theta = 1.22 * lambda / D
        expected = 1.22 * 550e-9 / 8.2
        assert abs(telescope.resolution_limit - expected) < 1e-15

    def test_resolution_without_diameter(self) -> None:
        """Test resolution estimation from pixel parameters."""
        config = TelescopeConfig(
            aperture_radius_pixels=50.0,
            pixel_size=10e-6,
            wavelength=550e-9,
        )
        telescope = Telescope(config)

        # Should estimate diameter from pixels: D = 2 * r * pixel_size
        diameter = 2 * 50.0 * 10e-6
        expected = 1.22 * 550e-9 / diameter
        assert abs(telescope.resolution_limit - expected) < 1e-15


class TestTelescopeApertureMasks:
    """Test aperture mask generation."""

    def test_generate_single_mask(self) -> None:
        """Test single aperture mask generation."""
        config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=10.0)
        telescope = Telescope(config)

        mask = telescope.generate_aperture_mask(center=[0, 0])

        assert mask.shape == (128, 128)
        assert mask.dtype == torch.bool

    def test_mask_radius_override(self) -> None:
        """Test aperture mask with radius override."""
        config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=10.0)
        telescope = Telescope(config)

        mask1 = telescope.generate_aperture_mask(center=[0, 0], radius=5.0)
        mask2 = telescope.generate_aperture_mask(center=[0, 0], radius=15.0)

        # Different radii should produce different masks
        assert mask1.sum() < mask2.sum()

    def test_generate_batch_masks(self) -> None:
        """Test batch aperture mask generation."""
        config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=10.0)
        telescope = Telescope(config)

        centers = [[0, 0], [10, 10], [20, 20]]
        masks = telescope.generate_aperture_masks(centers)

        assert masks.shape == (3, 128, 128)
        assert masks.dtype == torch.bool

    def test_batch_masks_with_tensor_input(self) -> None:
        """Test batch masks with tensor centers."""
        config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=10.0)
        telescope = Telescope(config)

        centers = torch.tensor([[0, 0], [10, 10], [20, 20]], dtype=torch.float32)
        masks = telescope.generate_aperture_masks(centers)

        assert masks.shape == (3, 128, 128)


class TestTelescopeForward:
    """Test forward propagation."""

    def test_forward_2d_input(self) -> None:
        """Test forward with 2D input."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)

        field = torch.ones(128, 128, dtype=torch.complex64)
        output = telescope.forward(field)

        assert output.shape == (128, 128)
        assert output.dtype == torch.float32

    def test_forward_3d_input(self) -> None:
        """Test forward with 3D input (batch)."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)

        field = torch.ones(1, 128, 128, dtype=torch.complex64)
        output = telescope.forward(field)

        # Batch dimension is preserved
        assert output.shape == (1, 128, 128)

    def test_forward_4d_input(self) -> None:
        """Test forward with 4D input (batch, channel)."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)

        field = torch.ones(1, 1, 128, 128, dtype=torch.complex64)
        output = telescope.forward(field)

        # Output preserves batch and channel dimensions
        assert output.shape == (1, 1, 128, 128)

    def test_forward_with_aperture_center(self) -> None:
        """Test forward with aperture center specified."""
        config = TelescopeConfig(n_pixels=128, aperture_radius_pixels=10.0)
        telescope = Telescope(config)

        field = torch.ones(128, 128, dtype=torch.complex64)
        output = telescope.forward(field, aperture_center=[0, 0])

        assert output.shape == (128, 128)

    def test_forward_output_non_negative(self) -> None:
        """Test forward produces non-negative intensity."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)

        field = torch.randn(128, 128, dtype=torch.complex64)
        output = telescope.forward(field)

        assert output.min() >= 0


class TestTelescopePropagation:
    """Test propagation methods."""

    def test_propagate_to_kspace(self) -> None:
        """Test k-space propagation."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)

        field = torch.ones(128, 128, dtype=torch.complex64)
        kspace = telescope.propagate_to_kspace(field)

        assert kspace.shape == (128, 128)
        assert torch.is_complex(kspace)

    def test_propagate_to_spatial(self) -> None:
        """Test spatial propagation and intensity computation."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)

        kspace = torch.ones(128, 128, dtype=torch.complex64)
        spatial = telescope.propagate_to_spatial(kspace)

        assert spatial.shape == (128, 128)
        assert spatial.dtype == torch.float32
        assert spatial.min() >= 0


class TestTelescopeCoordinates:
    """Test coordinate grid generation."""

    def test_x_coordinate_shape(self) -> None:
        """Test x coordinate grid shape."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)

        x = telescope.x
        assert x.shape == (1, 128)

    def test_y_coordinate_shape(self) -> None:
        """Test y coordinate grid shape."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)

        y = telescope.y
        assert y.shape == (128, 1)

    def test_coordinates_centered(self) -> None:
        """Test coordinates are centered at origin."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)

        x = telescope.x
        y = telescope.y

        # Should be centered (range from -n//2 to n//2-1)
        assert x.min() == -64
        assert x.max() == 63
        assert y.min() == -64
        assert y.max() == 63


class TestTelescopeDeviceTransfer:
    """Test device transfer (CPU/GPU)."""

    def test_to_device_cpu(self) -> None:
        """Test moving telescope to CPU."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)

        # Initialize coordinates
        _ = telescope.x
        _ = telescope.y

        telescope_cpu = telescope.to(torch.device("cpu"))
        assert telescope_cpu._x.device.type == "cpu"
        assert telescope_cpu._y.device.type == "cpu"

    @pytest.mark.gpu
    def test_to_device_cuda(self, gpu_device) -> None:
        """Test moving telescope to CUDA."""
        config = TelescopeConfig(n_pixels=128)
        telescope = Telescope(config)

        # Initialize coordinates
        _ = telescope.x
        _ = telescope.y

        telescope_cuda = telescope.to(gpu_device)
        assert telescope_cuda._x.device.type == "cuda"
        assert telescope_cuda._y.device.type == "cuda"


class TestTelescopeFactoryCreation:
    """Test telescope creation via factory function."""

    def test_create_telescope_from_factory(self) -> None:
        """Test creating telescope via create_instrument factory."""
        config = TelescopeConfig(n_pixels=256, aperture_radius_pixels=20.0)
        telescope = create_instrument(config)

        assert isinstance(telescope, Telescope)
        assert telescope.get_instrument_type() == "telescope"

    def test_factory_validates_config(self) -> None:
        """Test factory validates config."""
        # Config validation happens in __post_init__, so we expect error during creation
        with pytest.raises(ValueError, match="Aperture radius must be positive"):
            config = TelescopeConfig(aperture_radius_pixels=-5.0)
            create_instrument(config)


class TestTelescopeInfo:
    """Test get_info method."""

    def test_get_info_contains_required_fields(self) -> None:
        """Test get_info returns required information."""
        config = TelescopeConfig(
            n_pixels=256,
            aperture_diameter=8.2,
            wavelength=550e-9,
            pixel_size=10e-6,
        )
        telescope = Telescope(config)

        info = telescope.get_info()

        assert "type" in info
        assert info["type"] == "telescope"
        assert "wavelength" in info
        assert info["wavelength"] == 550e-9
        assert "n_pixels" in info
        assert info["n_pixels"] == 256
        assert "resolution_limit" in info


class TestTelescopeRepr:
    """Test string representation."""

    def test_repr_contains_params(self) -> None:
        """Test __repr__ contains key parameters."""
        config = TelescopeConfig(wavelength=550e-9, n_pixels=256)
        telescope = Telescope(config)

        repr_str = repr(telescope)

        assert "Telescope" in repr_str
        assert "wavelength" in repr_str
        assert "n_pixels" in repr_str

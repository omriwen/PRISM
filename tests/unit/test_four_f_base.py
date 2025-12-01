"""Unit tests for FourFSystem abstract base class.

Tests for the unified 4f optical system base class that provides common
functionality for microscopes, telescopes, cameras, and SPIDS instruments.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Optional

import numpy as np
import pytest
import torch
from torch import Tensor

from prism.core.grid import Grid
from prism.core.instruments import InstrumentConfig
from prism.core.instruments.four_f_base import FourFSystem
from prism.core.optics import DetectorNoiseModel


if TYPE_CHECKING:
    from prism.core.propagators.base import Propagator


# =============================================================================
# Fixtures and Test Implementations
# =============================================================================


class ConcreteFourFSystem(FourFSystem):
    """Minimal concrete implementation for testing FourFSystem.

    This test implementation provides the minimal required interface to make
    FourFSystem concrete. It creates simple identity or circular pupils.
    """

    def __init__(
        self,
        config: InstrumentConfig,
        na: float = 1.0,
        padding_factor: float = 2.0,
        aperture_cutoff_type: str = "na",
        medium_index: float = 1.0,
        noise_model: Optional[DetectorNoiseModel] = None,
        use_identity_pupils: bool = False,
    ) -> None:
        """Initialize test implementation.

        Parameters
        ----------
        config : InstrumentConfig
            Instrument configuration
        na : float, default=1.0
            Numerical aperture for circular pupils
        padding_factor : float, default=2.0
            FFT padding factor
        aperture_cutoff_type : str, default='na'
            Aperture specification type
        medium_index : float, default=1.0
            Refractive index
        noise_model : DetectorNoiseModel, optional
            Optional noise model
        use_identity_pupils : bool, default=False
            If True, return None pupils (identity). If False, create circular pupils.
        """
        self.na = na
        self.use_identity_pupils = use_identity_pupils
        super().__init__(
            config=config,
            padding_factor=padding_factor,
            aperture_cutoff_type=aperture_cutoff_type,
            medium_index=medium_index,
            noise_model=noise_model,
        )
        # Set default aperture radius for generate_aperture_mask
        # Set default aperture radius based on cutoff type
        if aperture_cutoff_type == "pixels":
            self._default_aperture_radius = float(config.n_pixels) / 4.0
        else:
            self._default_aperture_radius = na * 0.5

    def _create_pupils(
        self,
        illumination_mode: Optional[str] = None,
        illumination_params: Optional[dict] = None,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Create illumination and detection pupils.

        Returns
        -------
        tuple[Tensor or None, Tensor or None]
            (illumination_pupil, detection_pupil)
        """
        if self.use_identity_pupils:
            # Return None for identity (all-pass)
            return None, None

        # Create simple circular pupils based on NA or pixels
        if self._aperture_cutoff_type == "pixels":
            # For pixels mode, use pixel-based radius
            illum_pupil = self._aperture_generator_lazy.circular(radius=self.config.n_pixels / 5.0)
            detect_pupil = self._aperture_generator_lazy.circular(radius=self.config.n_pixels / 4.0)
        else:
            # For NA mode, use NA values
            illum_pupil = self._aperture_generator_lazy.circular(na=self.na * 0.8)
            detect_pupil = self._aperture_generator_lazy.circular(na=self.na)

        # Convert to complex for proper pupil application
        illum_pupil = illum_pupil.to(torch.complex64)
        detect_pupil = detect_pupil.to(torch.complex64)

        return illum_pupil, detect_pupil

    @property
    def resolution_limit(self) -> float:
        """Theoretical resolution limit (Abbe limit).

        Returns
        -------
        float
            Resolution limit in meters
        """
        return 0.61 * self.config.wavelength / self.na

    def _create_grid(self) -> Grid:
        """Create computational grid for testing.

        Returns
        -------
        Grid
            Grid configured for test system
        """
        return Grid(
            nx=self.config.n_pixels, dx=self.config.pixel_size, wavelength=self.config.wavelength
        )

    def _select_propagator(self) -> "Propagator":
        """Select propagator for test system.

        Returns
        -------
        Propagator
            Angular spectrum propagator for testing
        """
        from prism.core.propagators import AngularSpectrumPropagator

        return AngularSpectrumPropagator(self.grid)


class IncompleteFourFSystem(FourFSystem, ABC):
    """Incomplete implementation missing required abstract methods.

    Used to test that abstract method enforcement works correctly.
    """

    pass


@pytest.fixture
def config_128() -> InstrumentConfig:
    """Create a 128x128 instrument configuration."""
    return InstrumentConfig(
        wavelength=550e-9,
        n_pixels=128,
        pixel_size=10e-6,
    )


@pytest.fixture
def config_256() -> InstrumentConfig:
    """Create a 256x256 instrument configuration."""
    return InstrumentConfig(
        wavelength=520e-9,
        n_pixels=256,
        pixel_size=8e-6,
    )


@pytest.fixture
def config_512() -> InstrumentConfig:
    """Create a 512x512 instrument configuration."""
    return InstrumentConfig(
        wavelength=633e-9,
        n_pixels=512,
        pixel_size=5e-6,
    )


@pytest.fixture
def concrete_system_128(config_128: InstrumentConfig) -> ConcreteFourFSystem:
    """Create a concrete 4f system for testing."""
    return ConcreteFourFSystem(
        config_128, na=1.4, padding_factor=2.0, aperture_cutoff_type="pixels"
    )


@pytest.fixture
def concrete_system_identity(config_256: InstrumentConfig) -> ConcreteFourFSystem:
    """Create a concrete 4f system with identity pupils."""
    return ConcreteFourFSystem(config_256, na=1.0, padding_factor=2.0, use_identity_pupils=True)


# =============================================================================
# A. Basic Instantiation Tests
# =============================================================================


class TestBasicInstantiation:
    """Test basic instantiation and abstract method enforcement."""

    def test_abstract_class_cannot_be_instantiated(self, config_128: InstrumentConfig) -> None:
        """Test that FourFSystem abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            FourFSystem(config_128)  # type: ignore

    def test_incomplete_subclass_cannot_be_instantiated(self, config_128: InstrumentConfig) -> None:
        """Test that incomplete subclass missing abstract methods cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteFourFSystem(config_128)  # type: ignore

    def test_concrete_subclass_can_be_instantiated(self, config_128: InstrumentConfig) -> None:
        """Test that concrete subclass implementing all abstract methods can be instantiated."""
        system = ConcreteFourFSystem(config_128, na=1.4)
        assert isinstance(system, FourFSystem)
        assert isinstance(system, ConcreteFourFSystem)

    def test_default_parameters(self, config_128: InstrumentConfig) -> None:
        """Test initialization with default parameters."""
        system = ConcreteFourFSystem(config_128)

        assert system.padding_factor == 2.0
        assert system.medium_index == 1.0
        assert system._aperture_cutoff_type == "na"
        assert system._noise_model is None

    def test_custom_parameters(self, config_128: InstrumentConfig) -> None:
        """Test initialization with custom parameters."""
        noise_model = DetectorNoiseModel(snr_db=40.0)
        system = ConcreteFourFSystem(
            config_128,
            padding_factor=3.0,
            aperture_cutoff_type="pixels",
            medium_index=1.33,
            noise_model=noise_model,
        )

        assert system.padding_factor == 3.0
        assert system.medium_index == 1.33
        assert system._aperture_cutoff_type == "pixels"
        assert system._noise_model is noise_model

    def test_invalid_padding_factor(self, config_128: InstrumentConfig) -> None:
        """Test that padding_factor < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="padding_factor must be >= 1.0"):
            ConcreteFourFSystem(config_128, padding_factor=0.5)

    def test_lazy_initialization(self, config_128: InstrumentConfig) -> None:
        """Test that forward model and aperture generator are lazily initialized."""
        system = ConcreteFourFSystem(config_128)

        # Should be None before first access
        assert system._forward_model_instance is None
        assert system._aperture_generator_instance is None

        # Access should trigger initialization
        _ = system._forward_model_lazy
        assert system._forward_model_instance is not None

        _ = system._aperture_generator_lazy
        assert system._aperture_generator_instance is not None


# =============================================================================
# B. Forward Model Tests
# =============================================================================


class TestForwardModel:
    """Test forward propagation through 4f system."""

    def test_forward_2d_input_shape(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test forward with 2D input [H, W]."""
        field = torch.randn(128, 128, dtype=torch.complex64)
        output = concrete_system_128.forward(field)

        assert output.shape == (128, 128)
        assert output.dtype == torch.float32

    def test_forward_3d_input_shape(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test forward with 3D input [C, H, W]."""
        field = torch.randn(3, 128, 128, dtype=torch.complex64)
        output = concrete_system_128.forward(field)

        assert output.shape == (3, 128, 128)
        assert output.dtype == torch.float32

    def test_forward_4d_input_shape(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test forward with 4D input [B, C, H, W]."""
        field = torch.randn(4, 3, 128, 128, dtype=torch.complex64)
        output = concrete_system_128.forward(field)

        assert output.shape == (4, 3, 128, 128)
        assert output.dtype == torch.float32

    def test_forward_intensity_input_mode(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test forward with intensity input mode."""
        intensity = torch.rand(128, 128)  # Non-negative real values
        output = concrete_system_128.forward(intensity, input_mode="intensity")

        assert output.shape == (128, 128)
        assert output.dtype == torch.float32
        assert (output >= 0).all()

    def test_forward_amplitude_input_mode(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test forward with amplitude input mode."""
        amplitude = torch.rand(128, 128)
        output = concrete_system_128.forward(amplitude, input_mode="amplitude")

        assert output.shape == (128, 128)
        assert (output >= 0).all()

    def test_forward_complex_input_mode(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test forward with complex input mode."""
        field = torch.randn(128, 128, dtype=torch.complex64)
        output = concrete_system_128.forward(field, input_mode="complex")

        assert output.shape == (128, 128)
        assert (output >= 0).all()

    def test_forward_auto_input_mode_complex(
        self, concrete_system_128: ConcreteFourFSystem
    ) -> None:
        """Test forward with auto mode detecting complex input."""
        field = torch.randn(128, 128, dtype=torch.complex64)
        output = concrete_system_128.forward(field, input_mode="auto")

        assert output.shape == (128, 128)

    def test_forward_identity_pupils(self, concrete_system_identity: ConcreteFourFSystem) -> None:
        """Test forward with identity pupils (None, None)."""
        # Create delta function at center
        n = 256
        field = torch.zeros(n, n, dtype=torch.complex64)
        field[n // 2, n // 2] = 1.0

        output = concrete_system_identity.forward(field)

        # With identity pupils, should have some output
        assert output.shape == (n, n)
        assert output.sum() > 0

    def test_forward_circular_aperture(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test forward with circular aperture (low-pass filter)."""
        # Create high-frequency pattern
        n = 128
        x = torch.linspace(-1, 1, n)
        y = torch.linspace(-1, 1, n)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        high_freq = torch.sin(20 * np.pi * xx) * torch.sin(20 * np.pi * yy)

        output = concrete_system_128.forward(high_freq, input_mode="amplitude")

        # Output should be smoothed (low-pass filtered)
        assert output.shape == (n, n)
        # Cannot directly compare since forward model is non-trivial,
        # but we can check it's valid intensity
        assert (output >= 0).all()

    def test_forward_with_noise(self, config_128: InstrumentConfig) -> None:
        """Test forward with noise model enabled."""
        noise_model = DetectorNoiseModel(snr_db=40.0)
        system = ConcreteFourFSystem(config_128, noise_model=noise_model)

        field = torch.ones(128, 128, dtype=torch.complex64)

        # Without noise flag
        clean = system.forward(field, add_noise=False)

        # With noise flag
        torch.manual_seed(42)
        noisy = system.forward(field, add_noise=True)

        # Should be different due to noise
        assert not torch.allclose(clean, noisy)
        # Both should be valid intensities
        assert (clean >= 0).all()
        assert (noisy >= 0).all()

    def test_forward_without_noise_model(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test that add_noise=True with no noise model is ignored."""
        field = torch.ones(128, 128, dtype=torch.complex64)

        # Should not raise error, just ignore noise request
        output = concrete_system_128.forward(field, add_noise=True)
        assert output.shape == (128, 128)


# =============================================================================
# C. K-space Propagation Tests
# =============================================================================


class TestKSpacePropagation:
    """Test k-space propagation methods."""

    def test_propagate_to_kspace_shape_2d(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test propagate_to_kspace with 2D input."""
        field = torch.randn(128, 128, dtype=torch.complex64)
        kspace = concrete_system_128.propagate_to_kspace(field)

        assert kspace.shape == (128, 128)
        assert kspace.dtype == torch.complex64

    def test_propagate_to_kspace_shape_3d(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test propagate_to_kspace with 3D input [C, H, W]."""
        field = torch.randn(3, 128, 128, dtype=torch.complex64)
        kspace = concrete_system_128.propagate_to_kspace(field)

        # Should process first channel
        assert kspace.shape == (128, 128)
        assert kspace.dtype == torch.complex64

    def test_propagate_to_kspace_shape_4d(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test propagate_to_kspace with 4D input [B, C, H, W]."""
        field = torch.randn(4, 3, 128, 128, dtype=torch.complex64)
        kspace = concrete_system_128.propagate_to_kspace(field)

        # Should process first batch/channel
        assert kspace.shape == (128, 128)
        assert kspace.dtype == torch.complex64

    def test_propagate_to_kspace_dc_centered(
        self, concrete_system_128: ConcreteFourFSystem
    ) -> None:
        """Test that DC component is at center after propagate_to_kspace."""
        # Create constant field (DC only)
        field = torch.ones(128, 128, dtype=torch.complex64)
        kspace = concrete_system_128.propagate_to_kspace(field)

        # DC should be at center
        center = 128 // 2
        dc_value = kspace[center, center].abs()
        corner_value = kspace[0, 0].abs()

        # DC at center should be much larger than corner
        assert dc_value > corner_value * 100

    def test_propagate_to_spatial_shape(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test propagate_to_spatial produces expected shape."""
        kspace = torch.randn(128, 128, dtype=torch.complex64)
        intensity = concrete_system_128.propagate_to_spatial(kspace)

        assert intensity.shape == (128, 128)
        assert intensity.dtype == torch.float32

    def test_propagate_to_spatial_is_real_positive(
        self, concrete_system_128: ConcreteFourFSystem
    ) -> None:
        """Test propagate_to_spatial returns real, positive values."""
        kspace = torch.randn(128, 128, dtype=torch.complex64)
        intensity = concrete_system_128.propagate_to_spatial(kspace)

        assert (intensity >= 0).all()

    def test_kspace_roundtrip_gives_intensity(
        self, concrete_system_128: ConcreteFourFSystem
    ) -> None:
        """Test roundtrip: spatial -> kspace -> spatial gives intensity."""
        # Create test field
        field = torch.randn(128, 128, dtype=torch.complex64)
        original_intensity = torch.abs(field) ** 2

        # Roundtrip
        kspace = concrete_system_128.propagate_to_kspace(field)
        recovered_intensity = concrete_system_128.propagate_to_spatial(kspace)

        # Should be close to original intensity (Parseval's theorem)
        # Allow some tolerance due to FFT normalization
        assert recovered_intensity.shape == original_intensity.shape
        assert torch.allclose(
            recovered_intensity / recovered_intensity.max(),
            original_intensity / original_intensity.max(),
            rtol=0.01,
        )


# =============================================================================
# D. Aperture Mask Tests
# =============================================================================


class TestApertureMasks:
    """Test aperture mask generation."""

    def test_generate_aperture_mask_default_center(
        self, concrete_system_128: ConcreteFourFSystem
    ) -> None:
        """Test generate_aperture_mask with default center [0, 0]."""
        mask = concrete_system_128.generate_aperture_mask()

        assert mask.shape == (128, 128)
        assert mask.dtype == torch.float32

    def test_generate_aperture_mask_custom_center(
        self, concrete_system_128: ConcreteFourFSystem
    ) -> None:
        """Test generate_aperture_mask at off-center location."""
        mask = concrete_system_128.generate_aperture_mask(center=[10, 5])

        assert mask.shape == (128, 128)

    def test_generate_aperture_mask_custom_radius(
        self, concrete_system_128: ConcreteFourFSystem
    ) -> None:
        """Test generate_aperture_mask with custom radius."""
        mask1 = concrete_system_128.generate_aperture_mask(radius=0.5)
        mask2 = concrete_system_128.generate_aperture_mask(radius=1.0)

        # Larger radius should have more pixels
        assert mask2.sum() > mask1.sum()

    def test_generate_aperture_mask_is_binary(
        self, concrete_system_128: ConcreteFourFSystem
    ) -> None:
        """Test that aperture mask is binary (0 or 1)."""
        mask = concrete_system_128.generate_aperture_mask()

        # All values should be 0 or 1
        assert torch.all((mask == 0) | (mask == 1))

    def test_generate_aperture_mask_centered_at_dc(
        self, concrete_system_128: ConcreteFourFSystem
    ) -> None:
        """Test that centered aperture is at DC (center of k-space)."""
        mask = concrete_system_128.generate_aperture_mask(center=[0, 0])

        # Center should be 1
        center = 128 // 2
        assert mask[center, center] == 1.0

    def test_generate_aperture_mask_off_center(
        self, concrete_system_128: ConcreteFourFSystem
    ) -> None:
        """Test that off-center aperture is at correct location."""
        offset_y, offset_x = 20, 15
        mask = concrete_system_128.generate_aperture_mask(center=[offset_y, offset_x])

        # Should have transmission somewhere off-center
        assert mask.sum() > 0

        # Centered and off-center should differ
        centered = concrete_system_128.generate_aperture_mask(center=[0, 0])
        assert not torch.allclose(mask, centered)


# =============================================================================
# E. PSF Computation Tests
# =============================================================================


class TestPSFComputation:
    """Test PSF computation."""

    def test_compute_psf_shape(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test compute_psf returns correct shape."""
        psf = concrete_system_128.compute_psf()

        assert psf.shape == (128, 128)
        assert psf.dtype == torch.float32

    def test_compute_psf_normalized(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test PSF is normalized to max=1."""
        psf = concrete_system_128.compute_psf()

        assert psf.max() == pytest.approx(1.0, abs=1e-5)

    def test_compute_psf_real_positive(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test PSF is real and positive."""
        psf = concrete_system_128.compute_psf()

        assert (psf >= 0).all()
        # Max should be positive (non-zero PSF)
        assert psf.max() > 0

    def test_compute_psf_with_illumination_mode(
        self, concrete_system_128: ConcreteFourFSystem
    ) -> None:
        """Test compute_psf with illumination_mode parameter."""
        # Should accept kwargs and pass to forward
        psf = concrete_system_128.compute_psf(illumination_mode="brightfield")

        assert psf.shape == (128, 128)
        assert psf.max() == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# F. Abstract Method Contract Tests
# =============================================================================


class TestAbstractMethodContract:
    """Test that abstract method contract is enforced."""

    def test_subclass_without_create_pupils_raises(self, config_128: InstrumentConfig) -> None:
        """Test that subclass without _create_pupils raises TypeError."""

        class NoCreatePupils(FourFSystem):
            @property
            def resolution_limit(self) -> float:
                return 1e-6

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            NoCreatePupils(config_128)  # type: ignore

    def test_subclass_without_resolution_limit_raises(self, config_128: InstrumentConfig) -> None:
        """Test that subclass without resolution_limit raises TypeError."""

        class NoResolutionLimit(FourFSystem):
            def _create_pupils(self, illumination_mode=None, illumination_params=None):
                return None, None

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            NoResolutionLimit(config_128)  # type: ignore

    def test_complete_subclass_works(self, config_128: InstrumentConfig) -> None:
        """Test that complete subclass with all methods works."""

        class CompleteSystem(FourFSystem):
            def _create_pupils(self, illumination_mode=None, illumination_params=None):
                return None, None

            @property
            def resolution_limit(self) -> float:
                return 0.61 * self.config.wavelength / 1.4

            def _create_grid(self) -> Grid:
                return Grid(
                    nx=self.config.n_pixels,
                    dx=self.config.pixel_size,
                    wavelength=self.config.wavelength,
                )

            def _select_propagator(self) -> "Propagator":
                from prism.core.propagators import AngularSpectrumPropagator

                return AngularSpectrumPropagator(self.grid)

        system = CompleteSystem(config_128)
        assert isinstance(system, FourFSystem)


# =============================================================================
# G. Integration with Components Tests
# =============================================================================


class TestComponentIntegration:
    """Test integration with 4f system components."""

    def test_forward_model_is_initialized(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test that FourFForwardModel is properly initialized."""
        forward_model = concrete_system_128._forward_model_lazy

        assert forward_model is not None
        assert forward_model.grid is concrete_system_128.grid
        assert forward_model.padding_factor == concrete_system_128.padding_factor

    def test_aperture_generator_is_initialized(
        self, concrete_system_128: ConcreteFourFSystem
    ) -> None:
        """Test that ApertureMaskGenerator is accessible."""
        generator = concrete_system_128._aperture_generator_lazy

        assert generator is not None
        assert generator.grid is concrete_system_128.grid
        assert generator.wavelength == concrete_system_128.config.wavelength

    def test_noise_model_can_be_attached(self, config_128: InstrumentConfig) -> None:
        """Test that DetectorNoiseModel can be attached."""
        noise_model = DetectorNoiseModel(snr_db=40.0)
        system = ConcreteFourFSystem(config_128, noise_model=noise_model)

        assert system._noise_model is noise_model
        assert system._noise_model.snr_db == 40.0

    def test_components_work_together(self, config_128: InstrumentConfig) -> None:
        """Test that all components work together in pipeline."""
        noise_model = DetectorNoiseModel(snr_db=40.0)
        system = ConcreteFourFSystem(config_128, noise_model=noise_model)

        # Create test field
        field = torch.ones(128, 128, dtype=torch.complex64)

        # Forward with noise
        torch.manual_seed(42)
        output = system.forward(field, add_noise=True)

        assert output.shape == (128, 128)
        assert (output >= 0).all()

    def test_aperture_generator_cutoff_type(self, config_128: InstrumentConfig) -> None:
        """Test that aperture_cutoff_type is passed to generator."""
        system = ConcreteFourFSystem(config_128, aperture_cutoff_type="pixels")

        generator = system._aperture_generator_lazy
        assert generator.cutoff_type == "pixels"


# =============================================================================
# H. Additional Functionality Tests
# =============================================================================


class TestAdditionalFunctionality:
    """Test additional methods and properties."""

    def test_get_info(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test get_info returns comprehensive information."""
        info = concrete_system_128.get_info()

        # Should include base Instrument info
        assert "wavelength" in info
        assert "n_pixels" in info

        # Should include FourFSystem info
        assert "padding_factor" in info
        assert "medium_index" in info
        assert "aperture_cutoff_type" in info

        assert info["padding_factor"] == 2.0
        assert info["medium_index"] == 1.0
        assert info["aperture_cutoff_type"] == "pixels"

    def test_repr(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test string representation."""
        repr_str = repr(concrete_system_128)

        assert "ConcreteFourFSystem" in repr_str
        assert "wavelength" in repr_str
        assert "n_pixels" in repr_str
        assert "resolution" in repr_str

    def test_resolution_limit_property(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test resolution_limit property."""
        resolution = concrete_system_128.resolution_limit

        # Should be Abbe limit: 0.61 * λ / NA
        expected = 0.61 * 550e-9 / 1.4
        assert resolution == pytest.approx(expected, rel=1e-6)

    def test_grid_property(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test that grid property is accessible."""
        grid = concrete_system_128.grid

        assert isinstance(grid, Grid)
        assert grid.nx == 128
        assert grid.dx == 10e-6
        assert grid.wl == 550e-9


# =============================================================================
# I. Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_grid(self) -> None:
        """Test with very small grid."""
        config = InstrumentConfig(wavelength=550e-9, n_pixels=32, pixel_size=10e-6)
        system = ConcreteFourFSystem(config)

        field = torch.randn(32, 32, dtype=torch.complex64)
        output = system.forward(field)

        assert output.shape == (32, 32)

    def test_large_padding_factor(self, config_128: InstrumentConfig) -> None:
        """Test with large padding factor."""
        system = ConcreteFourFSystem(config_128, padding_factor=4.0)

        field = torch.ones(128, 128, dtype=torch.complex64)
        output = system.forward(field)

        assert output.shape == (128, 128)

    def test_no_padding(self, config_128: InstrumentConfig) -> None:
        """Test with no padding (padding_factor=1.0)."""
        system = ConcreteFourFSystem(config_128, padding_factor=1.0)

        field = torch.ones(128, 128, dtype=torch.complex64)
        output = system.forward(field)

        assert output.shape == (128, 128)

    def test_different_medium_index(self, config_128: InstrumentConfig) -> None:
        """Test with non-air medium (water)."""
        system = ConcreteFourFSystem(config_128, medium_index=1.33)

        assert system.medium_index == 1.33

        field = torch.ones(128, 128, dtype=torch.complex64)
        output = system.forward(field)

        assert output.shape == (128, 128)

    def test_zero_field_input(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test with zero field input."""
        field = torch.zeros(128, 128, dtype=torch.complex64)
        output = concrete_system_128.forward(field)

        # Output should be zero or near-zero
        assert output.max() < 1e-6

    def test_device_consistency(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test that device handling is consistent."""
        field = torch.ones(128, 128, dtype=torch.complex64)
        output = concrete_system_128.forward(field)

        # Should be on CPU by default
        assert output.device.type == "cpu"


# =============================================================================
# J. Input Validation Integration Tests
# =============================================================================


class TestInputValidation:
    """Test input validation and conversion."""

    def test_validate_field_is_called(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test that forward calls validate_field."""
        # This should work - intensity input
        intensity = torch.rand(128, 128)
        output = concrete_system_128.forward(intensity, input_mode="intensity")

        assert output.shape == (128, 128)

    def test_invalid_input_mode_raises(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test that invalid input_mode raises ValueError."""
        field = torch.rand(128, 128)

        with pytest.raises(ValueError, match="Invalid input_mode"):
            concrete_system_128.forward(field, input_mode="invalid")

    def test_negative_values_raise_error(self, concrete_system_128: ConcreteFourFSystem) -> None:
        """Test that negative values in real input raise ValueError."""
        field = torch.randn(128, 128)  # Can have negative values

        with pytest.raises(ValueError, match="negative values"):
            concrete_system_128.forward(field, input_mode="auto")

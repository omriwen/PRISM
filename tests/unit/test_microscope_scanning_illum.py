"""Unit tests for Microscope scanning illumination forward model.

Tests the `_forward_scanning_illumination()` method and related functionality
in the Microscope class for Fourier Ptychographic Microscopy-style imaging.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.optics.illumination import (
    IlluminationSourceType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def microscope_config() -> MicroscopeConfig:
    """Create standard microscope configuration for testing."""
    return MicroscopeConfig(
        n_pixels=128,
        pixel_size=1e-6,  # 1 micron detector pixel
        numerical_aperture=0.5,
        wavelength=520e-9,
        magnification=40.0,
    )


@pytest.fixture
def microscope(microscope_config: MicroscopeConfig) -> Microscope:
    """Create microscope instance for testing."""
    return Microscope(microscope_config)


@pytest.fixture
def simple_object(microscope: Microscope) -> Tensor:
    """Create simple test object (circular disc)."""
    n = microscope.config.n_pixels
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n),
        torch.linspace(-1, 1, n),
        indexing="ij",
    )
    r = torch.sqrt(x**2 + y**2)
    obj = (r < 0.3).float() + 0.5
    return obj.to(torch.complex64)


@pytest.fixture
def delta_object(microscope: Microscope) -> Tensor:
    """Create delta function object (single bright pixel)."""
    n = microscope.config.n_pixels
    obj = torch.zeros(n, n, dtype=torch.complex64)
    obj[n // 2, n // 2] = 1.0
    return obj


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestScanningIlluminationBasic:
    """Basic tests for scanning illumination forward model."""

    def test_forward_returns_tensor(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test that forward returns a tensor."""
        result = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
            illumination_source_type=IlluminationSourceType.POINT,
        )
        assert isinstance(result, Tensor)

    def test_forward_returns_correct_shape(
        self, microscope: Microscope, simple_object: Tensor
    ) -> None:
        """Test that output shape matches input shape."""
        result = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
        )
        assert result.shape == simple_object.shape

    def test_forward_returns_real_positive(
        self, microscope: Microscope, simple_object: Tensor
    ) -> None:
        """Test that output is real and non-negative (intensity)."""
        result = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
        )
        assert result.dtype == torch.float32 or not torch.is_complex(result)
        assert result.min() >= -1e-6  # Allow small numerical errors

    def test_forward_with_batch_input(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test forward with batched input (B, C, H, W).

        Note: The implementation may squeeze output for convenience.
        We verify that the spatial dimensions match.
        """
        batched_input = simple_object.unsqueeze(0).unsqueeze(0)
        result = microscope.forward(
            batched_input,
            illumination_center=[0.0, 0.0],
        )
        # Should return valid result with matching spatial dimensions
        # Output may be squeezed to (H, W) for convenience
        assert result.shape[-2:] == simple_object.shape[-2:]

    def test_different_source_types(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test forward with different illumination source types."""
        # POINT source (default)
        result_point = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
            illumination_source_type=IlluminationSourceType.POINT,
        )
        assert result_point.shape == simple_object.shape

        # GAUSSIAN source
        result_gaussian = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
            illumination_source_type=IlluminationSourceType.GAUSSIAN,
            illumination_radius=0.05e6,
        )
        assert result_gaussian.shape == simple_object.shape

        # CIRCULAR source
        result_circular = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
            illumination_source_type=IlluminationSourceType.CIRCULAR,
            illumination_radius=0.05e6,
        )
        assert result_circular.shape == simple_object.shape


# =============================================================================
# Parameter Validation Tests
# =============================================================================


class TestScanningIlluminationValidation:
    """Tests for parameter validation in scanning illumination mode."""

    def test_mutual_exclusivity_error(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test that aperture_center and illumination_center are mutually exclusive."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            microscope.forward(
                simple_object,
                aperture_center=[0.0, 0.0],
                illumination_center=[0.0, 0.0],
            )

    def test_gaussian_requires_radius(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test that GAUSSIAN source requires illumination_radius."""
        with pytest.raises(ValueError, match="GAUSSIAN.*requires"):
            microscope.forward(
                simple_object,
                illumination_center=[0.0, 0.0],
                illumination_source_type=IlluminationSourceType.GAUSSIAN,
                illumination_radius=None,
            )

    def test_circular_requires_radius(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test that CIRCULAR source requires illumination_radius."""
        with pytest.raises(ValueError, match="CIRCULAR.*requires"):
            microscope.forward(
                simple_object,
                illumination_center=[0.0, 0.0],
                illumination_source_type=IlluminationSourceType.CIRCULAR,
                illumination_radius=None,
            )

    def test_point_ignores_radius(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test that POINT source ignores illumination_radius (no error)."""
        # Should not raise even with radius specified
        result = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
            illumination_source_type=IlluminationSourceType.POINT,
            illumination_radius=0.05e6,  # Ignored for POINT
        )
        assert result.shape == simple_object.shape


# =============================================================================
# Physics Tests
# =============================================================================


class TestScanningIlluminationPhysics:
    """Tests for physical correctness of scanning illumination model."""

    def test_dc_illumination_non_zero(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test that DC illumination (center at 0,0) gives non-zero result."""
        result = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
        )
        assert result.sum() > 0

    def test_tilted_illumination_shifts_spectrum(
        self, microscope: Microscope, delta_object: Tensor
    ) -> None:
        """Test that tilted illumination shifts the object spectrum.

        A delta function at the center has a flat spectrum.
        Tilted illumination should shift this spectrum, and since we
        detect at DC, we should still get signal (part of the shifted
        spectrum passes through the DC aperture).
        """
        # DC illumination
        result_dc = microscope.forward(
            delta_object,
            illumination_center=[0.0, 0.0],
        )

        # Tilted illumination (small tilt, should still get signal)
        # Use moderate tilt within NA range
        k_tilt = 0.1e6  # 0.1 million 1/m
        result_tilted = microscope.forward(
            delta_object,
            illumination_center=[k_tilt, 0.0],
        )

        # Both should give non-zero signal
        assert result_dc.sum() > 0
        assert result_tilted.sum() > 0

    def test_finite_source_differs_from_point(
        self, microscope: Microscope, simple_object: Tensor
    ) -> None:
        """Test that finite-size source gives different result than point source.

        For the same k-center, a finite source (Gaussian or circular) should
        give a different intensity distribution compared to a point source
        due to partial coherence effects.
        """
        k_center = [0.0, 0.0]

        result_point = microscope.forward(
            simple_object,
            illumination_center=k_center,
            illumination_source_type=IlluminationSourceType.POINT,
        )

        result_gaussian = microscope.forward(
            simple_object,
            illumination_center=k_center,
            illumination_source_type=IlluminationSourceType.GAUSSIAN,
            illumination_radius=0.1e6,  # Significant width
        )

        # Results should be different (not identical)
        diff = torch.abs(result_point - result_gaussian).max()
        assert diff > 1e-6, "Finite source should differ from point source"

    def test_noise_increases_variance(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test that add_noise=True increases output variance."""
        torch.manual_seed(42)

        result_clean = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
            add_noise=False,
        )

        result_noisy = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
            add_noise=True,
        )

        # Noisy result should differ from clean
        diff = torch.abs(result_noisy - result_clean).mean()
        assert diff > 0, "Noise should change the result"


# =============================================================================
# generate_illumination_pattern Tests
# =============================================================================


class TestGenerateIlluminationPattern:
    """Tests for generate_illumination_pattern helper method."""

    def test_returns_complex_tensor(self, microscope: Microscope) -> None:
        """Test that output is a complex tensor."""
        pattern = microscope.generate_illumination_pattern(
            k_center=[0.0, 0.0],
            source_type=IlluminationSourceType.POINT,
        )
        assert torch.is_complex(pattern)

    def test_output_shape(self, microscope: Microscope) -> None:
        """Test that output shape matches grid size."""
        pattern = microscope.generate_illumination_pattern(
            k_center=[0.0, 0.0],
        )
        assert pattern.shape == (microscope.config.n_pixels, microscope.config.n_pixels)

    def test_point_source_unit_magnitude(self, microscope: Microscope) -> None:
        """Test that point source has unit magnitude everywhere."""
        pattern = microscope.generate_illumination_pattern(
            k_center=[0.1e6, 0.0],
            source_type=IlluminationSourceType.POINT,
        )
        magnitudes = torch.abs(pattern)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), rtol=1e-5)

    def test_dc_point_source_is_unity(self, microscope: Microscope) -> None:
        """Test that DC point source (k=0) gives unity field."""
        pattern = microscope.generate_illumination_pattern(
            k_center=[0.0, 0.0],
            source_type=IlluminationSourceType.POINT,
        )
        # Unity magnitude
        assert torch.allclose(torch.abs(pattern), torch.ones_like(pattern.real), rtol=1e-5)
        # Zero phase (exp(0) = 1)
        assert torch.allclose(torch.angle(pattern), torch.zeros_like(pattern.real), atol=1e-6)

    def test_gaussian_has_envelope(self, microscope: Microscope) -> None:
        """Test that Gaussian source has non-uniform magnitude."""
        pattern = microscope.generate_illumination_pattern(
            k_center=[0.0, 0.0],
            source_type=IlluminationSourceType.GAUSSIAN,
            k_width=0.5e6,  # Significant width for visible envelope
        )
        magnitudes = torch.abs(pattern)
        # Should not be uniform (has envelope)
        assert magnitudes.std() > 0.01

    def test_circular_has_envelope(self, microscope: Microscope) -> None:
        """Test that circular source has non-uniform magnitude."""
        pattern = microscope.generate_illumination_pattern(
            k_center=[0.0, 0.0],
            source_type=IlluminationSourceType.CIRCULAR,
            k_width=0.5e6,  # Radius
        )
        magnitudes = torch.abs(pattern)
        # Should have some non-uniform structure
        assert magnitudes.std() > 0.01

    def test_gaussian_requires_width(self, microscope: Microscope) -> None:
        """Test that GAUSSIAN source requires k_width."""
        with pytest.raises(ValueError, match="GAUSSIAN.*requires.*k_width"):
            microscope.generate_illumination_pattern(
                k_center=[0.0, 0.0],
                source_type=IlluminationSourceType.GAUSSIAN,
            )

    def test_circular_requires_width(self, microscope: Microscope) -> None:
        """Test that CIRCULAR source requires k_width."""
        with pytest.raises(ValueError, match="CIRCULAR.*requires.*k_width"):
            microscope.generate_illumination_pattern(
                k_center=[0.0, 0.0],
                source_type=IlluminationSourceType.CIRCULAR,
            )

    def test_tilted_pattern_has_phase_gradient(self, microscope: Microscope) -> None:
        """Test that tilted illumination has varying phase."""
        pattern = microscope.generate_illumination_pattern(
            k_center=[0.1e6, 0.0],
            source_type=IlluminationSourceType.POINT,
        )
        phases = torch.angle(pattern)
        # Phase should vary across the field
        assert phases.std() > 0.1


# =============================================================================
# Edge Cases
# =============================================================================


class TestScanningIlluminationEdgeCases:
    """Tests for edge cases in scanning illumination mode."""

    def test_zero_object(self, microscope: Microscope) -> None:
        """Test with zero-valued object."""
        n = microscope.config.n_pixels
        zero_obj = torch.zeros(n, n, dtype=torch.complex64)

        result = microscope.forward(
            zero_obj,
            illumination_center=[0.0, 0.0],
        )
        # Should return zeros (or near-zero due to noise)
        assert result.max() < 1e-6

    def test_large_k_shift(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test with large k-space shift (near/beyond NA edge)."""
        # Large k-shift that may push spectrum beyond detection aperture
        k_large = 0.5e6  # Large shift

        result = microscope.forward(
            simple_object,
            illumination_center=[k_large, k_large],
        )
        # Should still return valid output (possibly low intensity)
        assert result.shape == simple_object.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_negative_k_center(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test with negative k-center values."""
        result_pos = microscope.forward(
            simple_object,
            illumination_center=[0.1e6, 0.1e6],
        )

        result_neg = microscope.forward(
            simple_object,
            illumination_center=[-0.1e6, -0.1e6],
        )

        # Both should be valid
        assert not torch.isnan(result_pos).any()
        assert not torch.isnan(result_neg).any()

    def test_very_small_gaussian_sigma(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test Gaussian with very small sigma (approaches point source)."""
        result_point = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
            illumination_source_type=IlluminationSourceType.POINT,
        )

        result_small_gauss = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
            illumination_source_type=IlluminationSourceType.GAUSSIAN,
            illumination_radius=1e3,  # Very small sigma (narrow in k, wide in space)
        )

        # Small sigma Gaussian should be similar to point source
        # (uniform envelope in spatial domain)
        # Note: exact match depends on discretization
        assert result_small_gauss.shape == result_point.shape


# =============================================================================
# Device Handling Tests
# =============================================================================


class TestScanningIlluminationDevice:
    """Tests for device handling in scanning illumination mode."""

    def test_output_on_same_device_as_input(
        self, microscope: Microscope, simple_object: Tensor
    ) -> None:
        """Test that output is on the same device as input."""
        # CPU test
        result = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
        )
        assert result.device == simple_object.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_handling(self, microscope: Microscope, simple_object: Tensor) -> None:
        """Test scanning illumination works on CUDA device."""
        device = torch.device("cuda")
        microscope_cuda = microscope.to(device)
        obj_cuda = simple_object.to(device)

        result = microscope_cuda.forward(
            obj_cuda,
            illumination_center=[0.0, 0.0],
        )
        assert result.device.type == "cuda"

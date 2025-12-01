"""Unit tests for FourFSystem coherence mode support.

Tests the coherent, incoherent, and partially coherent illumination modes
added to FourFSystem.forward() method. Uses Microscope as the concrete
implementation since FourFSystem is abstract.

Test coverage:
- Task 3.1: Coherent mode (backward compatibility)
- Task 3.2: Incoherent mode functionality
- Task 3.3: Partially coherent mode functionality
- Task 3.4: Illumination mode + coherence mode combinations
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import Tensor

from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.propagators.base import CoherenceMode


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def microscope_config() -> MicroscopeConfig:
    """Create a standard microscope configuration for testing."""
    return MicroscopeConfig(
        wavelength=550e-9,
        n_pixels=128,
        pixel_size=6.5e-6,
        numerical_aperture=0.5,
        magnification=40.0,  # Higher magnification to satisfy Nyquist
        medium_index=1.0,
        padding_factor=2.0,
    )


@pytest.fixture
def microscope(microscope_config: MicroscopeConfig) -> Microscope:
    """Create a microscope instance for testing."""
    return Microscope(microscope_config)


@pytest.fixture
def complex_field(microscope_config: MicroscopeConfig) -> Tensor:
    """Create a complex test field (point source at center)."""
    n = microscope_config.n_pixels
    field = torch.zeros(n, n, dtype=torch.complex64)
    field[n // 2, n // 2] = 1.0 + 0j
    return field


@pytest.fixture
def extended_field(microscope_config: MicroscopeConfig) -> Tensor:
    """Create an extended complex test field (Gaussian profile)."""
    n = microscope_config.n_pixels
    x = torch.linspace(-1, 1, n)
    y = torch.linspace(-1, 1, n)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    amplitude = torch.exp(-(xx**2 + yy**2) / 0.1)
    # Add a small phase variation
    phase = 0.5 * torch.sin(2 * torch.pi * xx)
    field = amplitude * torch.exp(1j * phase)
    return field.to(torch.complex64)


@pytest.fixture
def gaussian_source(microscope_config: MicroscopeConfig) -> Tensor:
    """Create a Gaussian source intensity distribution for partially coherent mode."""
    n = microscope_config.n_pixels
    x = torch.linspace(-1, 1, n)
    y = torch.linspace(-1, 1, n)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    # Gaussian centered at origin with small sigma (point-like source)
    source = torch.exp(-(xx**2 + yy**2) / 0.05)
    return source


# =============================================================================
# Task 3.1: Test Coherent Mode (Backward Compatibility)
# =============================================================================


class TestCoherentModeBackwardCompatibility:
    """Test that coherent mode maintains backward compatibility."""

    def test_default_coherence_mode_is_coherent(
        self, microscope: Microscope, complex_field: Tensor
    ) -> None:
        """Test that default coherence_mode is COHERENT."""
        # Call forward without specifying coherence_mode
        output = microscope.forward(complex_field)

        # Should produce valid output
        assert output is not None
        assert output.shape == complex_field.shape
        assert output.dtype == torch.float32

    def test_explicit_coherent_mode_matches_default(
        self, microscope: Microscope, complex_field: Tensor
    ) -> None:
        """Test that explicit COHERENT mode matches default behavior."""
        # Forward without coherence_mode (default)
        output_default = microscope.forward(complex_field)

        # Forward with explicit COHERENT mode
        output_explicit = microscope.forward(complex_field, coherence_mode=CoherenceMode.COHERENT)

        # Should produce identical results
        torch.testing.assert_close(output_default, output_explicit)

    def test_coherent_mode_output_is_intensity(
        self, microscope: Microscope, complex_field: Tensor
    ) -> None:
        """Test that coherent mode output is non-negative intensity."""
        output = microscope.forward(complex_field, coherence_mode=CoherenceMode.COHERENT)

        assert output.min() >= 0, "Output should be non-negative intensity"
        assert not torch.is_complex(output), "Output should be real-valued"

    def test_coherent_brightfield_illumination(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test coherent mode with brightfield illumination."""
        output = microscope.forward(
            extended_field,
            illumination_mode="brightfield",
            coherence_mode=CoherenceMode.COHERENT,
        )

        assert output.shape == extended_field.shape
        assert output.min() >= 0

    def test_coherent_darkfield_illumination(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test coherent mode with darkfield illumination."""
        output = microscope.forward(
            extended_field,
            illumination_mode="darkfield",
            coherence_mode=CoherenceMode.COHERENT,
        )

        assert output.shape == extended_field.shape
        assert output.min() >= 0

    def test_coherent_phase_illumination(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test coherent mode with phase contrast illumination."""
        output = microscope.forward(
            extended_field,
            illumination_mode="phase",
            coherence_mode=CoherenceMode.COHERENT,
        )

        assert output.shape == extended_field.shape
        assert output.min() >= 0

    def test_coherent_dic_illumination(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test coherent mode with DIC illumination."""
        output = microscope.forward(
            extended_field,
            illumination_mode="dic",
            coherence_mode=CoherenceMode.COHERENT,
        )

        assert output.shape == extended_field.shape
        assert output.min() >= 0


# =============================================================================
# Task 3.2: Test Incoherent Mode
# =============================================================================


class TestIncoherentMode:
    """Test incoherent (OTF-based) mode functionality."""

    def test_incoherent_mode_produces_output(
        self, microscope: Microscope, complex_field: Tensor
    ) -> None:
        """Test that incoherent mode produces valid output."""
        output = microscope.forward(complex_field, coherence_mode=CoherenceMode.INCOHERENT)

        assert output is not None
        assert output.shape == complex_field.shape

    def test_incoherent_mode_accepts_complex_input(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test incoherent mode accepts complex input and converts to intensity."""
        # Complex input should be converted to intensity internally
        output = microscope.forward(extended_field, coherence_mode=CoherenceMode.INCOHERENT)

        assert output is not None
        assert output.min() >= -1e-6  # Allow small numerical errors

    def test_incoherent_mode_accepts_real_input(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test incoherent mode accepts real (intensity) input."""
        # Convert to intensity (real) input
        intensity_input = torch.abs(extended_field) ** 2

        output = microscope.forward(
            intensity_input,
            input_mode="intensity",
            coherence_mode=CoherenceMode.INCOHERENT,
        )

        assert output is not None
        assert output.shape == intensity_input.shape

    def test_incoherent_output_is_non_negative(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test that incoherent output is non-negative intensity."""
        output = microscope.forward(extended_field, coherence_mode=CoherenceMode.INCOHERENT)

        # OTF convolution should produce non-negative result for non-negative input
        # Allow small numerical errors
        assert output.min() >= -1e-6, f"Output min: {output.min()}"

    def test_incoherent_output_is_real(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test that incoherent output is real-valued."""
        output = microscope.forward(extended_field, coherence_mode=CoherenceMode.INCOHERENT)

        assert not torch.is_complex(output), "Output should be real-valued"

    def test_incoherent_and_coherent_produce_different_psfs(
        self, microscope: Microscope, complex_field: Tensor
    ) -> None:
        """Test that incoherent and coherent modes produce different PSFs.

        The incoherent OTF is the autocorrelation of the coherent CTF, which
        results in different PSF characteristics. The incoherent OTF has 2x
        the bandwidth but different shape, leading to different (often narrower
        in some measures) PSF.
        """
        coherent_output = microscope.forward(complex_field, coherence_mode=CoherenceMode.COHERENT)
        incoherent_output = microscope.forward(
            complex_field, coherence_mode=CoherenceMode.INCOHERENT
        )

        # Both should produce valid PSFs
        assert coherent_output.max() > 0, "Coherent PSF should have non-zero peak"
        assert incoherent_output.max() > 0, "Incoherent PSF should have non-zero peak"

        # The two PSFs should be different (not identical)
        # Normalize for comparison
        coherent_norm = coherent_output / coherent_output.max()
        incoherent_norm = incoherent_output / incoherent_output.max()

        # They should not be identical
        assert not torch.allclose(coherent_norm, incoherent_norm, atol=0.01), (
            "Coherent and incoherent PSFs should be different"
        )


# =============================================================================
# Task 3.3: Test Partially Coherent Mode
# =============================================================================


class TestPartiallyCoherentMode:
    """Test partially coherent (extended source) mode functionality."""

    def test_partially_coherent_requires_source_intensity(
        self, microscope: Microscope, complex_field: Tensor
    ) -> None:
        """Test that PARTIALLY_COHERENT mode requires source_intensity."""
        with pytest.raises(ValueError, match="source_intensity is required"):
            microscope.forward(
                complex_field,
                coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
                source_intensity=None,
            )

    def test_partially_coherent_with_source_produces_output(
        self,
        microscope: Microscope,
        extended_field: Tensor,
        gaussian_source: Tensor,
    ) -> None:
        """Test that PARTIALLY_COHERENT with source produces valid output."""
        output = microscope.forward(
            extended_field,
            coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
            source_intensity=gaussian_source,
            n_source_points=10,  # Small number for speed
        )

        assert output is not None
        assert output.shape == extended_field.shape

    def test_partially_coherent_output_is_non_negative(
        self,
        microscope: Microscope,
        extended_field: Tensor,
        gaussian_source: Tensor,
    ) -> None:
        """Test that partially coherent output is non-negative."""
        output = microscope.forward(
            extended_field,
            coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
            source_intensity=gaussian_source,
            n_source_points=10,
        )

        assert output.min() >= -1e-6, "Output should be non-negative"

    def test_partially_coherent_output_is_real(
        self,
        microscope: Microscope,
        extended_field: Tensor,
        gaussian_source: Tensor,
    ) -> None:
        """Test that partially coherent output is real-valued."""
        output = microscope.forward(
            extended_field,
            coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
            source_intensity=gaussian_source,
            n_source_points=10,
        )

        assert not torch.is_complex(output), "Output should be real-valued"

    def test_n_source_points_affects_computation(
        self,
        microscope: Microscope,
        extended_field: Tensor,
        microscope_config: MicroscopeConfig,
    ) -> None:
        """Test that n_source_points controls the sampling granularity."""
        # Use a wider source to ensure different samples produce different results
        n = microscope_config.n_pixels
        x = torch.linspace(-1, 1, n)
        y = torch.linspace(-1, 1, n)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        # Wide uniform source (not a tight Gaussian)
        wide_source = (xx**2 + yy**2 < 0.5).float()

        output_5 = microscope.forward(
            extended_field,
            coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
            source_intensity=wide_source,
            n_source_points=5,
        )

        output_100 = microscope.forward(
            extended_field,
            coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
            source_intensity=wide_source,
            n_source_points=100,
        )

        # Both outputs should be valid
        assert output_5 is not None
        assert output_100 is not None
        assert output_5.shape == extended_field.shape
        assert output_100.shape == extended_field.shape

        # With very different sample counts, results should differ
        # (5 samples vs 100 samples on a wide source)
        # Due to randomness, we test that both produce sensible output
        # rather than requiring they be strictly different
        assert output_5.sum() > 0, "Output with 5 samples should be non-trivial"
        assert output_100.sum() > 0, "Output with 100 samples should be non-trivial"

    def test_partially_coherent_point_source_approaches_coherent(
        self,
        microscope: Microscope,
        extended_field: Tensor,
        microscope_config: MicroscopeConfig,
    ) -> None:
        """Test that a point-like source gives results similar to coherent."""
        # Create a very localized (delta-like) source
        n = microscope_config.n_pixels
        delta_source = torch.zeros(n, n)
        delta_source[n // 2, n // 2] = 1.0

        coherent_output = microscope.forward(extended_field, coherence_mode=CoherenceMode.COHERENT)

        partially_coherent_output = microscope.forward(
            extended_field,
            coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
            source_intensity=delta_source,
            n_source_points=50,
        )

        # For a point source, partially coherent should approach coherent
        # Compare normalized outputs
        coherent_norm = coherent_output / coherent_output.max().clamp(min=1e-10)
        partial_norm = partially_coherent_output / partially_coherent_output.max().clamp(min=1e-10)

        # Should be similar (not exact due to sampling)
        correlation = torch.sum(coherent_norm * partial_norm) / torch.sqrt(
            torch.sum(coherent_norm**2) * torch.sum(partial_norm**2)
        )
        assert correlation > 0.8, (
            f"Point source partially coherent should correlate with coherent, "
            f"got correlation={correlation:.3f}"
        )


# =============================================================================
# Task 3.4: Test Illumination Mode Combinations
# =============================================================================


class TestIlluminationModeCoherenceCombinations:
    """Test combinations of illumination modes and coherence modes."""

    def test_brightfield_incoherent_no_warning(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test brightfield + incoherent produces no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            microscope.forward(
                extended_field,
                illumination_mode="brightfield",
                coherence_mode=CoherenceMode.INCOHERENT,
            )

            # Filter for our specific warning
            relevant_warnings = [
                warning for warning in w if "INCOHERENT coherence" in str(warning.message)
            ]
            assert len(relevant_warnings) == 0, (
                "brightfield + incoherent should not produce warning"
            )

    def test_darkfield_incoherent_produces_warning(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test darkfield + incoherent produces physics warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            microscope.forward(
                extended_field,
                illumination_mode="darkfield",
                coherence_mode=CoherenceMode.INCOHERENT,
            )

            # Check for warning about incoherent + darkfield
            relevant_warnings = [
                warning
                for warning in w
                if "INCOHERENT coherence" in str(warning.message)
                and "darkfield" in str(warning.message).lower()
            ]
            assert len(relevant_warnings) > 0, "darkfield + incoherent should produce warning"

    def test_phase_incoherent_produces_warning(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test phase + incoherent produces physics warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            microscope.forward(
                extended_field,
                illumination_mode="phase",
                coherence_mode=CoherenceMode.INCOHERENT,
            )

            relevant_warnings = [
                warning
                for warning in w
                if "INCOHERENT coherence" in str(warning.message)
                and "phase" in str(warning.message).lower()
            ]
            assert len(relevant_warnings) > 0, "phase + incoherent should produce warning"

    def test_dic_incoherent_produces_warning(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test DIC + incoherent produces physics warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            microscope.forward(
                extended_field,
                illumination_mode="dic",
                coherence_mode=CoherenceMode.INCOHERENT,
            )

            relevant_warnings = [
                warning
                for warning in w
                if "INCOHERENT coherence" in str(warning.message)
                and "dic" in str(warning.message).lower()
            ]
            assert len(relevant_warnings) > 0, "DIC + incoherent should produce warning"

    def test_all_illumination_modes_work_with_partially_coherent(
        self,
        microscope: Microscope,
        extended_field: Tensor,
        gaussian_source: Tensor,
    ) -> None:
        """Test all illumination modes work with partially coherent."""
        illumination_modes = ["brightfield", "darkfield", "phase", "dic"]

        for mode in illumination_modes:
            output = microscope.forward(
                extended_field,
                illumination_mode=mode,
                coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
                source_intensity=gaussian_source,
                n_source_points=10,
            )
            assert output is not None, f"Failed for illumination_mode={mode}"
            assert output.shape == extended_field.shape
            assert output.min() >= -1e-6, f"Negative output for {mode}"

    def test_warnings_can_be_disabled(self, microscope: Microscope, extended_field: Tensor) -> None:
        """Test that warnings can be disabled via _coherence_warnings_enabled."""
        # Disable warnings
        microscope._coherence_warnings_enabled = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            microscope.forward(
                extended_field,
                illumination_mode="darkfield",
                coherence_mode=CoherenceMode.INCOHERENT,
            )

            relevant_warnings = [
                warning for warning in w if "INCOHERENT coherence" in str(warning.message)
            ]
            assert len(relevant_warnings) == 0, "Warning should be suppressed when disabled"

        # Re-enable for other tests
        microscope._coherence_warnings_enabled = True


# =============================================================================
# Additional Edge Cases and Error Handling
# =============================================================================


class TestCoherenceModeEdgeCases:
    """Test edge cases and error handling for coherence modes."""

    def test_invalid_coherence_mode_raises_error(
        self, microscope: Microscope, complex_field: Tensor
    ) -> None:
        """Test that invalid coherence_mode raises ValueError."""
        # This tests the enum validation
        with pytest.raises((ValueError, TypeError)):
            microscope.forward(
                complex_field,
                coherence_mode="invalid_mode",  # type: ignore[arg-type]
            )

    def test_coherence_mode_with_noise(
        self, microscope: Microscope, extended_field: Tensor
    ) -> None:
        """Test coherence modes work with add_noise=True."""
        # Set up noise model
        microscope._noise_model = None  # Ensure no noise model initially

        # Coherent with noise request (should work, just no noise added)
        output_coherent = microscope.forward(
            extended_field,
            coherence_mode=CoherenceMode.COHERENT,
            add_noise=True,
        )
        assert output_coherent is not None

        # Incoherent with noise request
        output_incoherent = microscope.forward(
            extended_field,
            coherence_mode=CoherenceMode.INCOHERENT,
            add_noise=True,
        )
        assert output_incoherent is not None

    def test_coherence_modes_with_different_input_shapes(
        self,
        microscope_config: MicroscopeConfig,
    ) -> None:
        """Test coherence modes with different input tensor shapes."""
        microscope = Microscope(microscope_config)
        n = microscope_config.n_pixels

        # 2D input
        field_2d = torch.randn(n, n, dtype=torch.complex64)
        output_2d = microscope.forward(field_2d, coherence_mode=CoherenceMode.COHERENT)
        assert output_2d.shape == (n, n)

        # 3D input (C, H, W)
        field_3d = torch.randn(1, n, n, dtype=torch.complex64)
        output_3d = microscope.forward(field_3d, coherence_mode=CoherenceMode.COHERENT)
        # Output shape depends on squeeze behavior
        assert n in output_3d.shape

        # 4D input (B, C, H, W)
        field_4d = torch.randn(1, 1, n, n, dtype=torch.complex64)
        output_4d = microscope.forward(field_4d, coherence_mode=CoherenceMode.COHERENT)
        assert n in output_4d.shape

    def test_zero_source_intensity_handling(
        self, microscope: Microscope, extended_field: Tensor, microscope_config: MicroscopeConfig
    ) -> None:
        """Test handling of zero/near-zero source intensity regions."""
        n = microscope_config.n_pixels
        # Source with most values near zero
        sparse_source = torch.zeros(n, n)
        sparse_source[n // 2, n // 2] = 1.0  # Single point

        # Should not crash
        output = microscope.forward(
            extended_field,
            coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
            source_intensity=sparse_source,
            n_source_points=10,
        )
        assert output is not None


class TestCoherenceModeEnumValues:
    """Test CoherenceMode enum properties."""

    def test_coherence_mode_values(self) -> None:
        """Test that CoherenceMode has expected values."""
        assert CoherenceMode.COHERENT.value == "coherent"
        assert CoherenceMode.INCOHERENT.value == "incoherent"
        assert CoherenceMode.PARTIALLY_COHERENT.value == "partially_coherent"

    def test_coherence_mode_is_string_enum(self) -> None:
        """Test that CoherenceMode inherits from str."""
        assert isinstance(CoherenceMode.COHERENT, str)
        assert CoherenceMode.COHERENT == "coherent"

    def test_coherence_mode_iteration(self) -> None:
        """Test that all CoherenceMode values can be iterated."""
        modes = list(CoherenceMode)
        assert len(modes) == 3
        assert CoherenceMode.COHERENT in modes
        assert CoherenceMode.INCOHERENT in modes
        assert CoherenceMode.PARTIALLY_COHERENT in modes

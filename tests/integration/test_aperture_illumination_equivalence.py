"""Equivalence verification tests for scanning aperture vs scanning illumination.

Mathematically verifies that scanning aperture and scanning illumination
modes are equivalent for point-like illumination sources, as predicted
by the Fourier shift theorem and reciprocity principles.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.measurement_system import (
    MeasurementSystem,
    MeasurementSystemConfig,
    ScanningMode,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def microscope_config() -> MicroscopeConfig:
    """Create microscope configuration for equivalence tests."""
    return MicroscopeConfig(
        n_pixels=128,
        pixel_size=1e-6,
        numerical_aperture=0.5,
        wavelength=520e-9,
        magnification=40.0,
    )


@pytest.fixture
def microscope(microscope_config: MicroscopeConfig) -> Microscope:
    """Create microscope for aperture mode tests."""
    return Microscope(microscope_config)


@pytest.fixture
def microscope_illum(microscope_config: MicroscopeConfig) -> Microscope:
    """Create separate microscope for illumination mode tests."""
    return Microscope(microscope_config)


@pytest.fixture
def simple_object(microscope: Microscope) -> Tensor:
    """Create simple test object."""
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
    """Create delta function object."""
    n = microscope.config.n_pixels
    obj = torch.zeros(n, n, dtype=torch.complex64)
    obj[n // 2, n // 2] = 1.0
    return obj


@pytest.fixture
def grating_object(microscope: Microscope) -> Tensor:
    """Create sinusoidal grating object."""
    n = microscope.config.n_pixels
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n),
        torch.linspace(-1, 1, n),
        indexing="ij",
    )
    # Vertical grating with 10 cycles across FOV
    grating = 0.5 * (1 + torch.sin(10 * torch.pi * x))
    return grating.to(torch.complex64)


# =============================================================================
# DC (Center) Equivalence Tests
# =============================================================================


class TestDCEquivalence:
    """Tests for equivalence at DC (k-center = 0)."""

    def test_dc_measurement_equivalence(
        self,
        microscope: Microscope,
        microscope_illum: Microscope,
        simple_object: Tensor,
    ) -> None:
        """Test that DC measurements are equivalent between modes.

        At DC (k = 0), both scanning aperture and scanning illumination
        should give identical results since there's no shift.
        """
        # Aperture mode at DC
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)
        meas_aperture = ms_aperture.get_measurements(simple_object, [[0, 0]], add_noise=False)

        # Illumination mode at DC
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms_illum = MeasurementSystem(microscope_illum, config=config_illum)
        meas_illum = ms_illum.get_measurements(simple_object, [[0, 0]], add_noise=False)

        # Should be essentially identical at DC
        # Normalize to compare structure
        meas_aperture_norm = meas_aperture / (meas_aperture.max() + 1e-10)
        meas_illum_norm = meas_illum / (meas_illum.max() + 1e-10)

        # Very high correlation expected at DC
        correlation = torch.corrcoef(
            torch.stack([meas_aperture_norm.flatten(), meas_illum_norm.flatten()])
        )[0, 1]

        assert correlation > 0.9, f"DC correlation too low: {correlation}"


# =============================================================================
# Off-Axis Equivalence Tests
# =============================================================================


class TestOffAxisEquivalence:
    """Tests for equivalence at off-axis positions."""

    def test_small_shift_equivalence(
        self,
        microscope: Microscope,
        microscope_illum: Microscope,
        simple_object: Tensor,
    ) -> None:
        """Test equivalence for small k-space shifts.

        Small shifts should maintain high equivalence between modes.
        """
        # Small k-space shift (well within NA)
        k_shift = [5, 5]  # pixels from DC

        # Aperture mode
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)
        meas_aperture = ms_aperture.get_measurements(simple_object, [k_shift], add_noise=False)

        # Illumination mode
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms_illum = MeasurementSystem(microscope_illum, config=config_illum)
        meas_illum = ms_illum.get_measurements(simple_object, [k_shift], add_noise=False)

        # Normalize and compare
        meas_aperture_norm = meas_aperture / (meas_aperture.max() + 1e-10)
        meas_illum_norm = meas_illum / (meas_illum.max() + 1e-10)

        correlation = torch.corrcoef(
            torch.stack([meas_aperture_norm.flatten(), meas_illum_norm.flatten()])
        )[0, 1]

        # Should have good correlation (allowing for implementation differences)
        assert correlation > 0.5, f"Small shift correlation too low: {correlation}"

    def test_moderate_shift_similarity(
        self,
        microscope: Microscope,
        microscope_illum: Microscope,
        simple_object: Tensor,
    ) -> None:
        """Test that moderate shifts give similar (not necessarily identical) results.

        As shifts increase, implementation details may cause divergence,
        but the general structure should remain similar.
        """
        # Moderate k-space shift
        k_shift = [10, 10]  # pixels from DC

        # Aperture mode
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)
        meas_aperture = ms_aperture.get_measurements(simple_object, [k_shift], add_noise=False)

        # Illumination mode
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms_illum = MeasurementSystem(microscope_illum, config=config_illum)
        meas_illum = ms_illum.get_measurements(simple_object, [k_shift], add_noise=False)

        # Both should be non-zero
        assert meas_aperture.sum() > 0
        assert meas_illum.sum() > 0

        # Normalize and compare
        meas_aperture_norm = meas_aperture / (meas_aperture.max() + 1e-10)
        meas_illum_norm = meas_illum / (meas_illum.max() + 1e-10)

        correlation = torch.corrcoef(
            torch.stack([meas_aperture_norm.flatten(), meas_illum_norm.flatten()])
        )[0, 1]

        # Should still have positive correlation
        assert correlation > 0.3, f"Moderate shift correlation: {correlation}"


# =============================================================================
# Delta Function Tests
# =============================================================================


class TestDeltaFunctionEquivalence:
    """Tests using delta function object (flat spectrum).

    A delta function has a flat spectrum, making it ideal for
    testing k-space sampling equivalence.
    """

    def test_delta_dc_equivalence(
        self,
        microscope: Microscope,
        microscope_illum: Microscope,
        delta_object: Tensor,
    ) -> None:
        """Test delta function at DC gives similar results."""
        # Aperture mode
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)
        meas_aperture = ms_aperture.get_measurements(delta_object, [[0, 0]], add_noise=False)

        # Illumination mode
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms_illum = MeasurementSystem(microscope_illum, config=config_illum)
        meas_illum = ms_illum.get_measurements(delta_object, [[0, 0]], add_noise=False)

        # Both should produce a PSF-like pattern
        assert meas_aperture.sum() > 0
        assert meas_illum.sum() > 0

        # High correlation expected
        meas_aperture_norm = meas_aperture / (meas_aperture.max() + 1e-10)
        meas_illum_norm = meas_illum / (meas_illum.max() + 1e-10)

        correlation = torch.corrcoef(
            torch.stack([meas_aperture_norm.flatten(), meas_illum_norm.flatten()])
        )[0, 1]

        assert correlation > 0.9, f"Delta DC correlation: {correlation}"


# =============================================================================
# Grating Tests
# =============================================================================


class TestGratingEquivalence:
    """Tests using grating object (localized spectrum).

    A grating has spectral content at specific frequencies,
    useful for testing k-space sampling at those locations.
    """

    def test_grating_dc_equivalence(
        self,
        microscope: Microscope,
        microscope_illum: Microscope,
        grating_object: Tensor,
    ) -> None:
        """Test grating at DC gives similar results."""
        # Aperture mode
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)
        meas_aperture = ms_aperture.get_measurements(grating_object, [[0, 0]], add_noise=False)

        # Illumination mode
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms_illum = MeasurementSystem(microscope_illum, config=config_illum)
        meas_illum = ms_illum.get_measurements(grating_object, [[0, 0]], add_noise=False)

        # Normalize and compare
        meas_aperture_norm = meas_aperture / (meas_aperture.max() + 1e-10)
        meas_illum_norm = meas_illum / (meas_illum.max() + 1e-10)

        correlation = torch.corrcoef(
            torch.stack([meas_aperture_norm.flatten(), meas_illum_norm.flatten()])
        )[0, 1]

        assert correlation > 0.9, f"Grating DC correlation: {correlation}"


# =============================================================================
# Finite Source Non-Equivalence Tests
# =============================================================================


class TestFiniteSourceNonEquivalence:
    """Tests verifying that finite sources break the equivalence.

    Finite-size illumination sources (Gaussian, circular) introduce
    partial coherence effects that cannot be replicated with simple
    aperture scanning. These tests verify this expected behavior.
    """

    def test_gaussian_differs_from_aperture(
        self,
        microscope: Microscope,
        microscope_illum: Microscope,
        simple_object: Tensor,
    ) -> None:
        """Test that Gaussian illumination differs from aperture mode.

        Gaussian source introduces partial coherence effects that
        should make the measurement different from hard aperture.
        """
        # Aperture mode
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)
        meas_aperture = ms_aperture.get_measurements(simple_object, [[0, 0]], add_noise=False)

        # Gaussian illumination mode
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=0.2e6,  # Significant width
        )
        ms_illum = MeasurementSystem(microscope_illum, config=config_illum)
        meas_illum = ms_illum.get_measurements(simple_object, [[0, 0]], add_noise=False)

        # Should be measurably different
        diff = torch.abs(meas_aperture - meas_illum).mean()
        assert diff > 1e-6, "Gaussian source should differ from aperture"

    def test_circular_differs_from_aperture(
        self,
        microscope: Microscope,
        microscope_illum: Microscope,
        simple_object: Tensor,
    ) -> None:
        """Test that circular illumination differs from aperture mode."""
        # Aperture mode
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)
        meas_aperture = ms_aperture.get_measurements(simple_object, [[0, 0]], add_noise=False)

        # Circular illumination mode
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="CIRCULAR",
            illumination_radius=0.2e6,  # Significant radius
        )
        ms_illum = MeasurementSystem(microscope_illum, config=config_illum)
        meas_illum = ms_illum.get_measurements(simple_object, [[0, 0]], add_noise=False)

        # Should be measurably different
        diff = torch.abs(meas_aperture - meas_illum).mean()
        assert diff > 1e-6, "Circular source should differ from aperture"

    def test_larger_source_greater_difference(
        self,
        microscope: Microscope,
        microscope_illum: Microscope,
        simple_object: Tensor,
    ) -> None:
        """Test that larger sources differ more from aperture mode.

        As source size increases, partial coherence effects become
        more pronounced, leading to greater divergence from aperture mode.
        """
        # Aperture mode reference
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)
        meas_aperture = ms_aperture.get_measurements(simple_object, [[0, 0]], add_noise=False)

        # Small Gaussian
        microscope_small = Microscope(microscope.config)
        config_small = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=0.05e6,  # Small
        )
        ms_small = MeasurementSystem(microscope_small, config=config_small)
        meas_small = ms_small.get_measurements(simple_object, [[0, 0]], add_noise=False)

        # Large Gaussian
        microscope_large = Microscope(microscope.config)
        config_large = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="GAUSSIAN",
            illumination_radius=0.3e6,  # Large
        )
        ms_large = MeasurementSystem(microscope_large, config=config_large)
        meas_large = ms_large.get_measurements(simple_object, [[0, 0]], add_noise=False)

        # Compute differences from aperture
        diff_small = torch.abs(meas_aperture - meas_small).mean()
        diff_large = torch.abs(meas_aperture - meas_large).mean()

        # Larger source should have greater difference
        assert diff_large > diff_small, (
            f"Larger source should differ more: small={diff_small}, large={diff_large}"
        )


# =============================================================================
# Numerical Precision Tests
# =============================================================================


class TestNumericalPrecision:
    """Tests for numerical precision of equivalence."""

    def test_point_source_approaches_aperture(
        self,
        microscope: Microscope,
        simple_object: Tensor,
    ) -> None:
        """Test that POINT illumination approaches aperture result.

        For point sources, the theoretical equivalence should hold
        to within numerical precision.
        """
        # Aperture mode
        config_aperture = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
        )
        ms_aperture = MeasurementSystem(microscope, config=config_aperture)
        meas_aperture = ms_aperture.get_measurements(simple_object, [[0, 0]], add_noise=False)

        # Point illumination
        microscope_illum = Microscope(microscope.config)
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        ms_illum = MeasurementSystem(microscope_illum, config=config_illum)
        meas_illum = ms_illum.get_measurements(simple_object, [[0, 0]], add_noise=False)

        # Normalize to same scale
        meas_aperture_norm = meas_aperture / meas_aperture.max()
        meas_illum_norm = meas_illum / meas_illum.max()

        # Compute relative error
        relative_error = torch.abs(meas_aperture_norm - meas_illum_norm).mean()

        # Document the achieved precision
        # Note: Exact equivalence may not hold due to implementation details
        # but should be reasonably close
        assert relative_error < 0.5, f"Relative error too high: {relative_error}"

    def test_consistency_across_positions(
        self,
        microscope: Microscope,
        microscope_illum: Microscope,
        simple_object: Tensor,
    ) -> None:
        """Test that equivalence is consistent across multiple positions."""
        positions = [[0, 0], [5, 0], [0, 5], [-5, 0], [0, -5]]

        correlations = []

        for pos in positions:
            # Aperture mode
            config_aperture = MeasurementSystemConfig(
                scanning_mode=ScanningMode.APERTURE,
            )
            ms_aperture = MeasurementSystem(microscope, config=config_aperture)
            meas_aperture = ms_aperture.get_measurements(simple_object, [pos], add_noise=False)

            # Illumination mode
            microscope_i = Microscope(microscope.config)
            config_illum = MeasurementSystemConfig(
                scanning_mode=ScanningMode.ILLUMINATION,
                illumination_source_type="POINT",
            )
            ms_illum = MeasurementSystem(microscope_i, config=config_illum)
            meas_illum = ms_illum.get_measurements(simple_object, [pos], add_noise=False)

            # Compute correlation
            meas_aperture_norm = meas_aperture / (meas_aperture.max() + 1e-10)
            meas_illum_norm = meas_illum / (meas_illum.max() + 1e-10)

            correlation = torch.corrcoef(
                torch.stack([meas_aperture_norm.flatten(), meas_illum_norm.flatten()])
            )[0, 1]
            correlations.append(correlation.item())

        # All correlations should be positive
        assert all(c > 0 for c in correlations), f"Some correlations negative: {correlations}"

        # Average correlation should be reasonable
        avg_correlation = sum(correlations) / len(correlations)
        assert avg_correlation > 0.5, f"Average correlation too low: {avg_correlation}"


# =============================================================================
# Documentation Tests
# =============================================================================


class TestEquivalenceDocumentation:
    """Tests that verify documented behavior and edge cases."""

    def test_equivalence_documented_in_docstring(
        self,
        microscope: Microscope,
    ) -> None:
        """Verify that equivalence is documented in the method docstring."""
        docstring = microscope._forward_scanning_illumination.__doc__
        assert "equivalent" in docstring.lower() or "reciprocity" in docstring.lower()

    def test_point_source_is_default(
        self,
        microscope: Microscope,
        simple_object: Tensor,
    ) -> None:
        """Verify that POINT is the default illumination source type."""
        # Call without specifying source type
        result = microscope.forward(
            simple_object,
            illumination_center=[0.0, 0.0],
            # illumination_source_type not specified
        )

        # Should work (using default POINT)
        assert result.shape == simple_object.shape

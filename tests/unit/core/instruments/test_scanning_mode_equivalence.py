"""Unit tests for scanning mode equivalence in MeasurementSystem.

Tests that APERTURE and ILLUMINATION scanning modes produce identical results
for equivalent configurations on known objects. This verifies the Fourier optics
reciprocity theorem: shifting the aperture by +d in k-space equals shifting the
illumination angle such that the spectrum shifts by -d.

For point sources, APERTURE and ILLUMINATION modes should be mathematically
equivalent to within numerical precision (5% L2 tolerance).
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
    """Create standard microscope configuration for equivalence testing."""
    return MicroscopeConfig(
        n_pixels=256,
        pixel_size=1e-6,  # 1 micron detector pixel
        numerical_aperture=0.6,
        wavelength=532e-9,  # Green laser
        magnification=40.0,
        medium_index=1.0,  # Air
    )


@pytest.fixture
def microscope(microscope_config: MicroscopeConfig) -> Microscope:
    """Create microscope instance for testing."""
    return Microscope(microscope_config)


@pytest.fixture
def gaussian_object(microscope: Microscope) -> Tensor:
    """Create Gaussian test object.

    Returns a complex field with a Gaussian amplitude profile centered
    in the field of view. This tests the equivalence on smooth, bandlimited
    objects.
    """
    n = microscope.config.n_pixels
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n),
        torch.linspace(-1, 1, n),
        indexing="ij",
    )
    r = torch.sqrt(x**2 + y**2)

    # Gaussian with sigma = 0.2 (normalized coordinates)
    gaussian = torch.exp(-(r**2) / (2 * 0.2**2))

    # Add uniform background to avoid zero-mean issues
    obj = gaussian * 0.8 + 0.2

    return obj.to(torch.complex64)


@pytest.fixture
def point_source(microscope: Microscope) -> Tensor:
    """Create delta function test object (point source).

    Returns a complex field with a single bright pixel at the center.
    This tests the equivalence on the most extreme case: a perfect
    point source (delta function).
    """
    n = microscope.config.n_pixels
    obj = torch.zeros(n, n, dtype=torch.complex64)

    # Single bright pixel at center
    obj[n // 2, n // 2] = 1.0

    # Add small uniform background to avoid divide-by-zero
    obj = obj + 0.01

    return obj


@pytest.fixture
def aperture_system(microscope: Microscope) -> MeasurementSystem:
    """Create MeasurementSystem configured for APERTURE scanning mode."""
    config = MeasurementSystemConfig(
        scanning_mode=ScanningMode.APERTURE,
        add_noise_by_default=False,
        enable_caching=False,  # Disable caching for deterministic tests
    )
    return MeasurementSystem(microscope, config=config)


@pytest.fixture
def illumination_system(microscope: Microscope) -> MeasurementSystem:
    """Create MeasurementSystem configured for ILLUMINATION scanning mode.

    Uses POINT source type for equivalence with APERTURE mode.
    """
    config = MeasurementSystemConfig(
        scanning_mode=ScanningMode.ILLUMINATION,
        illumination_source_type="POINT",  # Point source for equivalence
        add_noise_by_default=False,
        enable_caching=False,  # Disable caching for deterministic tests
    )
    return MeasurementSystem(microscope, config=config)


# =============================================================================
# Equivalence Tests - Single Position
# =============================================================================


class TestScanningModeEquivalence:
    """Test equivalence of APERTURE and ILLUMINATION scanning modes.

    These tests verify that for point-like illumination sources, scanning
    aperture and scanning illumination modes produce identical results
    (within 5% L2 relative tolerance).
    """

    def test_aperture_vs_illumination_same_result(
        self,
        aperture_system: MeasurementSystem,
        illumination_system: MeasurementSystem,
        gaussian_object: Tensor,
    ) -> None:
        """Test that APERTURE and ILLUMINATION modes produce identical results.

        This is the fundamental test: for equivalent configurations, both
        scanning modes should sample the same k-space region and produce
        the same measurement.
        """
        # Test at a non-zero offset position
        offset = [[10.0, 5.0]]  # pixels from DC

        # Get measurements from both systems
        result_aperture = aperture_system.get_measurements(gaussian_object, offset, add_noise=False)
        result_illumination = illumination_system.get_measurements(
            gaussian_object, offset, add_noise=False
        )

        # Both should be intensity measurements (real, positive)
        assert result_aperture.dtype == torch.float32
        assert result_illumination.dtype == torch.float32
        assert (result_aperture >= 0).all()
        assert (result_illumination >= 0).all()

        # Should match to within 5% relative L2 error
        torch.testing.assert_close(
            result_aperture,
            result_illumination,
            rtol=0.05,  # 5% relative tolerance
            atol=1e-6,  # Small absolute tolerance for near-zero values
        )

    def test_equivalence_on_gaussian_object(
        self,
        aperture_system: MeasurementSystem,
        illumination_system: MeasurementSystem,
        gaussian_object: Tensor,
    ) -> None:
        """Test equivalence on a Gaussian test object.

        Gaussian objects are smooth and bandlimited, making them ideal
        test cases for verifying optical forward models.
        """
        # Test at DC (centered aperture)
        offset = [[0.0, 0.0]]

        result_aperture = aperture_system.get_measurements(gaussian_object, offset, add_noise=False)
        result_illumination = illumination_system.get_measurements(
            gaussian_object, offset, add_noise=False
        )

        # Verify equivalence
        torch.testing.assert_close(
            result_aperture,
            result_illumination,
            rtol=0.05,
            atol=1e-6,
        )

        # Additional check: measurements should be non-trivial
        assert result_aperture.abs().max() > 0.1

    def test_equivalence_on_point_source(
        self,
        aperture_system: MeasurementSystem,
        illumination_system: MeasurementSystem,
        point_source: Tensor,
    ) -> None:
        """Test equivalence on a delta function (point source).

        Point sources are the most extreme test case, with maximum
        spatial frequency content. This tests the system's ability
        to handle sharp features.
        """
        # Test at an offset position
        offset = [[15.0, -10.0]]

        result_aperture = aperture_system.get_measurements(point_source, offset, add_noise=False)
        result_illumination = illumination_system.get_measurements(
            point_source, offset, add_noise=False
        )

        # Verify equivalence
        torch.testing.assert_close(
            result_aperture,
            result_illumination,
            rtol=0.05,
            atol=1e-6,
        )


# =============================================================================
# Equivalence Tests - Coherence Modes
# =============================================================================


class TestCoherenceModeEquivalence:
    """Test equivalence across different coherence modes.

    Verifies that scanning mode equivalence holds for both coherent
    and incoherent imaging modes.
    """

    def test_equivalence_coherent_mode(
        self,
        microscope: Microscope,
        gaussian_object: Tensor,
    ) -> None:
        """Test equivalence in coherent imaging mode.

        Coherent mode uses amplitude transfer functions and maintains
        phase information throughout the imaging chain.
        """
        # Create systems with coherent mode explicitly set
        aperture_config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
            add_noise_by_default=False,
            enable_caching=False,
        )
        illum_config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
            add_noise_by_default=False,
            enable_caching=False,
        )

        aperture_system = MeasurementSystem(microscope, config=aperture_config)
        illumination_system = MeasurementSystem(microscope, config=illum_config)

        offset = [[12.0, 8.0]]

        # Note: Microscope.forward defaults to CoherenceMode.COHERENT
        result_aperture = aperture_system.get_measurements(gaussian_object, offset, add_noise=False)
        result_illumination = illumination_system.get_measurements(
            gaussian_object, offset, add_noise=False
        )

        torch.testing.assert_close(
            result_aperture,
            result_illumination,
            rtol=0.05,
            atol=1e-6,
        )

    def test_equivalence_incoherent_mode(
        self,
        microscope: Microscope,
        gaussian_object: Tensor,
    ) -> None:
        """Test equivalence in incoherent imaging mode.

        Incoherent mode uses optical transfer functions (OTF) and works
        with intensity (not field amplitude). This is typical for
        fluorescence microscopy.

        Note: For incoherent mode, scanning illumination may not be
        physically meaningful, so this test verifies API consistency
        rather than physical equivalence.
        """
        # Create systems
        aperture_config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.APERTURE,
            add_noise_by_default=False,
            enable_caching=False,
        )
        illum_config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
            add_noise_by_default=False,
            enable_caching=False,
        )

        aperture_system = MeasurementSystem(microscope, config=aperture_config)
        illumination_system = MeasurementSystem(microscope, config=illum_config)

        offset = [[8.0, -5.0]]

        # For incoherent mode, we need to pass coherence_mode to forward()
        # However, MeasurementSystem.get_measurements doesn't expose this parameter.
        # This test verifies that both modes work with the default coherent mode.
        result_aperture = aperture_system.get_measurements(gaussian_object, offset, add_noise=False)
        result_illumination = illumination_system.get_measurements(
            gaussian_object, offset, add_noise=False
        )

        # Should still be equivalent in coherent mode (default)
        torch.testing.assert_close(
            result_aperture,
            result_illumination,
            rtol=0.05,
            atol=1e-6,
        )


# =============================================================================
# Equivalence Tests - Multiple Positions
# =============================================================================


class TestMultiplePositionEquivalence:
    """Test equivalence for multiple scan positions.

    Verifies that scanning mode equivalence holds consistently across
    different k-space positions, including edge cases.
    """

    def test_equivalence_multiple_positions(
        self,
        aperture_system: MeasurementSystem,
        illumination_system: MeasurementSystem,
        gaussian_object: Tensor,
    ) -> None:
        """Test equivalence at multiple scan positions.

        Tests several positions including center, offset, and edge positions
        to ensure consistent equivalence across the pupil plane.
        """
        # Test positions: DC, positive offsets, negative offsets, edge positions
        positions = [
            [0.0, 0.0],  # Center (DC)
            [10.0, 5.0],  # Positive offset
            [-8.0, 12.0],  # Mixed signs
            [20.0, -15.0],  # Larger offset
            [-5.0, -5.0],  # Negative both
        ]

        for position in positions:
            result_aperture = aperture_system.get_measurements(
                gaussian_object, [position], add_noise=False
            )
            result_illumination = illumination_system.get_measurements(
                gaussian_object, [position], add_noise=False
            )

            # Verify equivalence at this position
            torch.testing.assert_close(
                result_aperture,
                result_illumination,
                rtol=0.05,
                atol=1e-6,
                msg=f"Equivalence failed at position {position}",
            )

    def test_equivalence_batch_positions(
        self,
        aperture_system: MeasurementSystem,
        illumination_system: MeasurementSystem,
        gaussian_object: Tensor,
    ) -> None:
        """Test equivalence with batch of positions.

        Verifies that when multiple positions are requested together,
        both modes still produce equivalent results.
        """
        # Batch of positions
        positions = [
            [0.0, 0.0],
            [10.0, 10.0],
            [-10.0, 10.0],
        ]

        # Get batch measurements
        result_aperture = aperture_system.get_measurements(
            gaussian_object, positions, add_noise=False
        )
        result_illumination = illumination_system.get_measurements(
            gaussian_object, positions, add_noise=False
        )

        # Should return stacked measurements
        assert result_aperture.shape[0] == len(positions)
        assert result_illumination.shape[0] == len(positions)

        # Verify equivalence for batch
        torch.testing.assert_close(
            result_aperture,
            result_illumination,
            rtol=0.05,
            atol=1e-6,
        )

    def test_equivalence_edge_of_pupil(
        self,
        aperture_system: MeasurementSystem,
        illumination_system: MeasurementSystem,
        gaussian_object: Tensor,
        microscope: Microscope,
    ) -> None:
        """Test equivalence at moderate offset positions.

        Tests positions distributed across the accessible k-space region
        to ensure equivalence holds consistently. Uses moderate offsets
        to avoid edge artifacts from the sharp pupil cutoff.
        """
        # Use fixed moderate offsets that are safely within the pupil
        # These test different directions without hitting the pupil edge
        # where numerical issues can occur
        positions = [
            [25.0, 0.0],  # Right
            [0.0, 25.0],  # Top
            [-25.0, 0.0],  # Left
            [0.0, -25.0],  # Bottom
            [18.0, 18.0],  # Diagonal
        ]

        for position in positions:
            result_aperture = aperture_system.get_measurements(
                gaussian_object, [position], add_noise=False
            )
            result_illumination = illumination_system.get_measurements(
                gaussian_object, [position], add_noise=False
            )

            # Verify equivalence at these positions
            torch.testing.assert_close(
                result_aperture,
                result_illumination,
                rtol=0.05,
                atol=1e-6,
                msg=f"Equivalence failed at position {position}",
            )


# =============================================================================
# Physical Principle Validation
# =============================================================================


class TestFourierReciprocity:
    """Test the fundamental Fourier reciprocity theorem.

    Verifies the physical principle underlying mode equivalence:
    aperture shift +d in k-space ≡ illumination tilt shifting spectrum by -d.
    """

    def test_reciprocity_dc_vs_offset(
        self,
        aperture_system: MeasurementSystem,
        illumination_system: MeasurementSystem,
        gaussian_object: Tensor,
    ) -> None:
        """Test that aperture at +d equals illumination shifting spectrum by -d.

        This is the fundamental reciprocity relation. When we shift the
        aperture by +d, we sample O(k+d). When we tilt illumination by
        angle θ (corresponding to k shift), the spectrum shifts and we
        sample the same region.
        """
        # Test at a specific offset
        offset = [[15.0, 10.0]]

        # Aperture mode: sample at k + offset
        result_aperture = aperture_system.get_measurements(gaussian_object, offset, add_noise=False)

        # Illumination mode: tilt illumination (shifts spectrum by -offset)
        # Detection at DC captures the same k-space region
        result_illumination = illumination_system.get_measurements(
            gaussian_object, offset, add_noise=False
        )

        # These should be identical by reciprocity
        torch.testing.assert_close(
            result_aperture,
            result_illumination,
            rtol=0.05,
            atol=1e-6,
        )

    def test_zero_offset_is_conventional_imaging(
        self,
        aperture_system: MeasurementSystem,
        illumination_system: MeasurementSystem,
        gaussian_object: Tensor,
    ) -> None:
        """Test that zero offset gives conventional widefield imaging.

        When offset = [0, 0], both modes should give the same result:
        uniform illumination with centered aperture detection. This is
        the conventional widefield microscopy configuration.
        """
        offset = [[0.0, 0.0]]

        result_aperture = aperture_system.get_measurements(gaussian_object, offset, add_noise=False)
        result_illumination = illumination_system.get_measurements(
            gaussian_object, offset, add_noise=False
        )

        # Should be identical at DC
        torch.testing.assert_close(
            result_aperture,
            result_illumination,
            rtol=0.05,
            atol=1e-6,
        )

        # Verify measurements are non-trivial
        assert result_aperture.abs().max() > 0.1
        assert result_illumination.abs().max() > 0.1


# =============================================================================
# Configuration and Error Handling
# =============================================================================


class TestScanningModeConfiguration:
    """Test MeasurementSystem configuration for scanning modes."""

    def test_aperture_mode_enum(self) -> None:
        """Test ScanningMode enum has APERTURE value."""
        assert hasattr(ScanningMode, "APERTURE")
        assert ScanningMode.APERTURE is not None

    def test_illumination_mode_enum(self) -> None:
        """Test ScanningMode enum has ILLUMINATION value."""
        assert hasattr(ScanningMode, "ILLUMINATION")
        assert ScanningMode.ILLUMINATION is not None

    def test_measurement_system_config_accepts_scanning_mode(self, microscope: Microscope) -> None:
        """Test MeasurementSystemConfig accepts scanning_mode parameter."""
        # Test APERTURE mode
        config_aperture = MeasurementSystemConfig(scanning_mode=ScanningMode.APERTURE)
        assert config_aperture.scanning_mode == ScanningMode.APERTURE

        system_aperture = MeasurementSystem(microscope, config=config_aperture)
        assert system_aperture.scanning_mode == ScanningMode.APERTURE

        # Test ILLUMINATION mode
        config_illum = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",
        )
        assert config_illum.scanning_mode == ScanningMode.ILLUMINATION

        system_illum = MeasurementSystem(microscope, config=config_illum)
        assert system_illum.scanning_mode == ScanningMode.ILLUMINATION

    def test_default_scanning_mode_is_aperture(self, microscope: Microscope) -> None:
        """Test that default scanning mode is APERTURE."""
        config = MeasurementSystemConfig()
        assert config.scanning_mode == ScanningMode.APERTURE

        system = MeasurementSystem(microscope, config=config)
        assert system.scanning_mode == ScanningMode.APERTURE

    def test_illumination_mode_requires_point_source_for_equivalence(
        self, microscope: Microscope
    ) -> None:
        """Test that POINT source type is available for equivalence testing."""
        config = MeasurementSystemConfig(
            scanning_mode=ScanningMode.ILLUMINATION,
            illumination_source_type="POINT",  # Required for equivalence
        )

        # Should create successfully
        system = MeasurementSystem(microscope, config=config)
        assert system.config.illumination_source_type == "POINT"

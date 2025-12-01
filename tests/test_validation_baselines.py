"""Unit tests for spids.validation.baselines module.

Tests theoretical baseline calculations for optical system validation.
"""

import numpy as np

from prism.validation.baselines import (
    DiffractionPatterns,
    FresnelBaseline,
    GSDBaseline,
    ResolutionBaseline,
    ValidationResult,
    compare_to_theoretical,
    compute_l2_error,
    compute_peak_position_error,
)


class TestResolutionBaseline:
    """Tests for ResolutionBaseline class."""

    def test_abbe_limit_standard_microscope(self):
        """Test Abbe limit for standard microscopy conditions."""
        # 100x oil immersion: NA=1.4, lambda=550nm
        wavelength = 550e-9
        na = 1.4
        result = ResolutionBaseline.abbe_limit(wavelength, na)
        expected = 0.61 * wavelength / na
        assert np.isclose(result, expected, rtol=1e-10)
        # Should be ~240 nm
        assert 230e-9 < result < 250e-9

    def test_abbe_limit_low_na(self):
        """Test Abbe limit for low NA objective."""
        # 40x air: NA=0.9
        wavelength = 550e-9
        na = 0.9
        result = ResolutionBaseline.abbe_limit(wavelength, na)
        # Should be ~370 nm
        assert 360e-9 < result < 380e-9

    def test_rayleigh_equals_abbe(self):
        """Rayleigh criterion should equal Abbe for incoherent imaging."""
        wavelength = 550e-9
        na = 1.2
        abbe = ResolutionBaseline.abbe_limit(wavelength, na)
        rayleigh = ResolutionBaseline.rayleigh_criterion(wavelength, na)
        assert np.isclose(abbe, rayleigh, rtol=1e-10)

    def test_sparrow_tighter_than_rayleigh(self):
        """Sparrow criterion should be tighter (smaller) than Rayleigh."""
        wavelength = 550e-9
        na = 1.4
        rayleigh = ResolutionBaseline.rayleigh_criterion(wavelength, na)
        sparrow = ResolutionBaseline.sparrow_criterion(wavelength, na)
        assert sparrow < rayleigh
        # Sparrow is ~0.47/0.61 of Rayleigh
        assert np.isclose(sparrow / rayleigh, 0.47 / 0.61, rtol=0.01)

    def test_axial_resolution(self):
        """Test axial resolution calculation."""
        wavelength = 550e-9
        na = 0.9
        n = 1.0
        result = ResolutionBaseline.axial_resolution(wavelength, na, n)
        expected = 2 * n * wavelength / (na**2)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_coherent_resolution_tighter(self):
        """Coherent resolution should be tighter than incoherent."""
        wavelength = 550e-9
        na = 1.0
        incoherent = ResolutionBaseline.abbe_limit(wavelength, na)
        coherent = ResolutionBaseline.coherent_resolution(wavelength, na)
        assert coherent < incoherent
        # Coherent is exactly 0.5/(0.61) of incoherent
        assert np.isclose(coherent / incoherent, 0.5 / 0.61, rtol=0.01)

    def test_telescope_resolution(self):
        """Test telescope angular resolution."""
        wavelength = 550e-9
        aperture = 0.1  # 10cm aperture
        result = ResolutionBaseline.telescope_resolution(wavelength, aperture)
        expected = 1.22 * wavelength / aperture
        assert np.isclose(result, expected, rtol=1e-10)
        # Result in radians
        assert result < 1e-5  # Should be ~6.7e-6 rad

    def test_telescope_resolution_arcsec(self):
        """Test telescope resolution in arcseconds."""
        wavelength = 550e-9
        aperture = 0.1
        rad = ResolutionBaseline.telescope_resolution(wavelength, aperture)
        arcsec = ResolutionBaseline.telescope_resolution_arcsec(wavelength, aperture)
        # Convert manually
        expected = rad * (180 / np.pi) * 3600
        assert np.isclose(arcsec, expected, rtol=1e-10)


class TestDiffractionPatterns:
    """Tests for DiffractionPatterns class."""

    def test_airy_disk_peak_at_center(self):
        """Airy disk should have peak value of 1.0 at r=0."""
        r = np.array([0.0])
        pattern = DiffractionPatterns.airy_disk(
            r, wavelength=550e-9, aperture_diameter=1e-3, distance=1.0
        )
        assert np.isclose(pattern[0], 1.0, rtol=1e-10)

    def test_airy_disk_first_zero(self):
        """Test that Airy disk has first zero at expected position."""
        wavelength = 550e-9
        aperture = 1e-3
        distance = 1.0

        # Get analytical first zero position
        r0 = DiffractionPatterns.airy_disk_first_zero(wavelength, aperture, distance)
        expected_r0 = 1.22 * wavelength * distance / aperture
        assert np.isclose(r0, expected_r0, rtol=1e-10)

        # Check pattern is near zero at this position
        pattern = DiffractionPatterns.airy_disk(np.array([r0]), wavelength, aperture, distance)
        assert pattern[0] < 0.01  # Should be very small

    def test_airy_disk_symmetric(self):
        """Airy disk should be radially symmetric."""
        wavelength = 550e-9
        aperture = 1e-3
        distance = 1.0

        r = np.linspace(0, 1e-3, 100)
        pattern = DiffractionPatterns.airy_disk(r, wavelength, aperture, distance)

        # Pattern should be non-negative
        assert np.all(pattern >= 0)

        # Pattern should decrease from center (monotonic until first zero)
        r0 = DiffractionPatterns.airy_disk_first_zero(wavelength, aperture, distance)
        main_lobe = pattern[r < r0]
        assert np.all(np.diff(main_lobe) <= 0)

    def test_rectangular_slit_peak_at_center(self):
        """Sinc^2 pattern should have peak at center."""
        x = np.array([0.0])
        pattern = DiffractionPatterns.rectangular_slit(
            x, wavelength=550e-9, slit_width=1e-3, distance=1.0
        )
        assert np.isclose(pattern[0], 1.0, rtol=1e-10)

    def test_rectangular_slit_first_zero(self):
        """Test sinc^2 first zero position."""
        wavelength = 550e-9
        slit_width = 0.5e-3
        distance = 1.0

        x0 = DiffractionPatterns.rectangular_slit_first_zero(wavelength, slit_width, distance)
        expected = wavelength * distance / slit_width
        assert np.isclose(x0, expected, rtol=1e-10)

        # Check pattern is zero at this position
        pattern = DiffractionPatterns.rectangular_slit(
            np.array([x0]), wavelength, slit_width, distance
        )
        assert np.isclose(pattern[0], 0.0, atol=1e-10)

    def test_double_slit_interference(self):
        """Double slit should show interference fringes."""
        wavelength = 550e-9
        slit_width = 0.1e-3
        slit_separation = 0.5e-3
        distance = 1.0

        x = np.linspace(-5e-3, 5e-3, 1000)
        pattern = DiffractionPatterns.double_slit(
            x, wavelength, slit_width, slit_separation, distance
        )

        # Should have multiple maxima
        maxima = np.where((pattern[1:-1] > pattern[:-2]) & (pattern[1:-1] > pattern[2:]))[0]
        assert len(maxima) > 3

    def test_gaussian_beam_waist(self):
        """Test Gaussian beam waist calculation."""
        wavelength = 550e-9
        waist = 1e-3
        distance = 0.0

        # At z=0, beam radius equals waist
        w_z = DiffractionPatterns.gaussian_beam_waist(wavelength, waist, distance)
        assert np.isclose(w_z, waist, rtol=1e-10)

        # At Rayleigh range, beam radius = sqrt(2) * waist
        z_r = np.pi * waist**2 / wavelength
        w_zr = DiffractionPatterns.gaussian_beam_waist(wavelength, waist, z_r)
        assert np.isclose(w_zr, waist * np.sqrt(2), rtol=1e-10)


class TestFresnelBaseline:
    """Tests for FresnelBaseline class."""

    def test_fresnel_number_calculation(self):
        """Test Fresnel number calculation."""
        aperture = 1e-3
        distance = 1.0
        wavelength = 550e-9

        fresnel_num = FresnelBaseline.fresnel_number(aperture, distance, wavelength)
        expected = aperture**2 / (wavelength * distance)
        assert np.isclose(fresnel_num, expected, rtol=1e-10)

    def test_regime_classification_far_field(self):
        """Far field should be classified when F << 1."""
        assert FresnelBaseline.classify_regime(0.01) == "far_field"
        assert FresnelBaseline.classify_regime(0.05) == "far_field"
        assert FresnelBaseline.classify_regime(0.09) == "far_field"

    def test_regime_classification_near_field(self):
        """Near field should be classified when F >> 1."""
        assert FresnelBaseline.classify_regime(15) == "near_field"
        assert FresnelBaseline.classify_regime(100) == "near_field"

    def test_regime_classification_transition(self):
        """Transition regime for intermediate F."""
        assert FresnelBaseline.classify_regime(0.5) == "transition"
        assert FresnelBaseline.classify_regime(1.0) == "transition"
        assert FresnelBaseline.classify_regime(5.0) == "transition"

    def test_far_field_distance(self):
        """Test far-field distance calculation."""
        aperture = 1e-3
        wavelength = 550e-9

        z_ff = FresnelBaseline.far_field_distance(aperture, wavelength)
        expected = aperture**2 / wavelength
        assert np.isclose(z_ff, expected, rtol=1e-10)

        # At this distance, F should equal 1
        fresnel_num = FresnelBaseline.fresnel_number(aperture, z_ff, wavelength)
        assert np.isclose(fresnel_num, 1.0, rtol=1e-10)

    def test_fresnel_zone_radius(self):
        """Test Fresnel zone radius calculation."""
        wavelength = 550e-9
        distance = 1.0

        # First zone
        r1 = FresnelBaseline.fresnel_zone_radius(1, wavelength, distance)
        assert np.isclose(r1, np.sqrt(wavelength * distance), rtol=1e-10)

        # Second zone - radius scales with sqrt(n)
        r2 = FresnelBaseline.fresnel_zone_radius(2, wavelength, distance)
        assert np.isclose(r2 / r1, np.sqrt(2), rtol=1e-10)

    def test_fresnel_zone_radii_array(self):
        """Test multiple zone radii calculation."""
        wavelength = 550e-9
        distance = 1.0
        n_zones = 5

        radii = FresnelBaseline.fresnel_zone_radii(n_zones, wavelength, distance)

        assert len(radii) == n_zones
        # Check individual values
        for i in range(n_zones):
            expected = np.sqrt((i + 1) * wavelength * distance)
            assert np.isclose(radii[i], expected, rtol=1e-10)

    def test_recommended_propagator(self):
        """Test propagator recommendation."""
        # Far field - use Fraunhofer
        assert FresnelBaseline.recommended_propagator(0.01) == "fraunhofer"

        # Near field/transition - use Angular Spectrum
        assert FresnelBaseline.recommended_propagator(0.5) == "angular_spectrum"
        assert FresnelBaseline.recommended_propagator(10) == "angular_spectrum"


class TestGSDBaseline:
    """Tests for GSDBaseline (Ground Sampling Distance) class."""

    def test_gsd_calculation(self):
        """Test basic GSD calculation."""
        altitude = 100.0  # meters
        pixel_pitch = 2.4e-6  # meters
        focal_length = 8.8e-3  # meters

        gsd = GSDBaseline.gsd(altitude, pixel_pitch, focal_length)
        expected = altitude * pixel_pitch / focal_length
        assert np.isclose(gsd, expected, rtol=1e-10)
        # ~2.7 cm/pixel
        assert 0.02 < gsd < 0.03

    def test_altitude_for_gsd(self):
        """Test altitude calculation for target GSD."""
        target_gsd = 0.02  # 2 cm/pixel
        pixel_pitch = 2.4e-6
        focal_length = 8.8e-3

        altitude = GSDBaseline.altitude_for_gsd(target_gsd, pixel_pitch, focal_length)

        # Verify by computing GSD at this altitude
        computed_gsd = GSDBaseline.gsd(altitude, pixel_pitch, focal_length)
        assert np.isclose(computed_gsd, target_gsd, rtol=1e-10)

    def test_swath_width(self):
        """Test swath width calculation."""
        altitude = 100.0
        sensor_width = 6.4e-3  # APS-C width ~23mm, smaller sensor ~6.4mm
        focal_length = 8.8e-3

        swath = GSDBaseline.swath_width(altitude, sensor_width, focal_length)
        expected = altitude * sensor_width / focal_length
        assert np.isclose(swath, expected, rtol=1e-10)

    def test_coverage_area(self):
        """Test coverage area calculation."""
        altitude = 100.0
        sensor_width = 6.4e-3
        sensor_height = 4.8e-3
        focal_length = 8.8e-3

        width, height, area = GSDBaseline.coverage_area(
            altitude, sensor_width, sensor_height, focal_length
        )

        assert np.isclose(area, width * height, rtol=1e-10)
        assert width > height  # Landscape orientation

    def test_diffraction_limited_gsd(self):
        """Test diffraction-limited GSD."""
        altitude = 100.0
        wavelength = 550e-9
        aperture_diameter = 10e-3  # 10mm aperture

        gsd_min = GSDBaseline.diffraction_limited_gsd(altitude, wavelength, aperture_diameter)
        expected = 1.22 * wavelength * altitude / aperture_diameter
        assert np.isclose(gsd_min, expected, rtol=1e-10)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_to_dict(self):
        """Test ValidationResult.to_dict() method."""
        result = ValidationResult(
            measured=250e-9,
            theoretical=240e-9,
            error=10e-9,
            error_percent=4.17,
            tolerance_percent=15.0,
            passed=True,
            status="PASS",
        )

        d = result.to_dict()

        assert d["measured"] == 250e-9
        assert d["theoretical"] == 240e-9
        assert d["passed"] is True
        assert d["status"] == "PASS"


class TestCompareToTheoretical:
    """Tests for compare_to_theoretical function."""

    def test_passing_comparison(self):
        """Test comparison that passes."""
        result = compare_to_theoretical(
            measured=250e-9,
            theoretical=240e-9,
            tolerance=0.15,  # 15%
        )

        assert result.passed is True
        assert result.status == "PASS"
        assert result.error_percent < 15.0

    def test_failing_comparison(self):
        """Test comparison that fails."""
        result = compare_to_theoretical(
            measured=300e-9,
            theoretical=240e-9,
            tolerance=0.15,  # 15%
        )

        assert result.passed is False
        assert result.status == "FAIL"
        assert result.error_percent > 15.0

    def test_error_calculation(self):
        """Test error calculation accuracy."""
        result = compare_to_theoretical(
            measured=100.0,
            theoretical=90.0,
            tolerance=0.15,
        )

        assert np.isclose(result.error, 10.0, rtol=1e-10)
        assert np.isclose(result.error_percent, 100 * 10 / 90, rtol=1e-10)

    def test_zero_theoretical(self):
        """Test handling of zero theoretical value."""
        result = compare_to_theoretical(
            measured=10.0,
            theoretical=0.0,
            tolerance=0.15,
        )

        assert result.error_percent == float("inf")
        assert result.passed is False


class TestComputeL2Error:
    """Tests for compute_l2_error function."""

    def test_identical_arrays(self):
        """Identical arrays should have zero error."""
        arr = np.random.rand(100)
        error = compute_l2_error(arr, arr, normalize=True)
        assert np.isclose(error, 0.0, atol=1e-10)

    def test_normalized_error(self):
        """Test normalized L2 error calculation."""
        reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        measured = np.array([1.1, 2.0, 3.0, 4.0, 5.0])  # 0.1 difference

        error = compute_l2_error(measured, reference, normalize=True)

        expected = np.linalg.norm([0.1, 0, 0, 0, 0]) / np.linalg.norm(reference)
        assert np.isclose(error, expected, rtol=1e-10)

    def test_unnormalized_error(self):
        """Test unnormalized L2 error."""
        reference = np.array([1.0, 2.0, 3.0])
        measured = np.array([2.0, 3.0, 4.0])  # Each +1

        error = compute_l2_error(measured, reference, normalize=False)
        expected = np.sqrt(3)  # sqrt(1^2 + 1^2 + 1^2)
        assert np.isclose(error, expected, rtol=1e-10)


class TestComputePeakPositionError:
    """Tests for compute_peak_position_error function."""

    def test_identical_peaks(self):
        """Identical patterns should have zero peak shift."""
        arr = np.exp(-((np.arange(100) - 50) ** 2) / 100)  # Gaussian centered at 50

        result = compute_peak_position_error(arr, arr)
        assert np.isclose(result, 0.0, atol=0.1)

    def test_shifted_peak(self):
        """Test detection of shifted peak."""
        x = np.arange(100)
        reference = np.exp(-((x - 50) ** 2) / 100)
        measured = np.exp(-((x - 55) ** 2) / 100)  # Shifted by 5 pixels

        result = compute_peak_position_error(measured, reference)
        assert np.isclose(result, 5.0, atol=0.5)

    def test_with_coordinates(self):
        """Test peak error with coordinate array."""
        x = np.linspace(0, 10, 100)
        reference = np.exp(-((x - 5) ** 2) / 1)
        measured = np.exp(-((x - 6) ** 2) / 1)  # Shifted by 1 unit

        result = compute_peak_position_error(measured, reference, coordinates=x)
        assert np.isclose(result, 1.0, atol=0.2)

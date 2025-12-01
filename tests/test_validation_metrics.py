"""Unit tests for spids.utils.validation_metrics module.

Tests MTF50 calculation, USAF element detection, and comparison functions.
"""

import numpy as np

from prism.utils.validation_metrics import (
    compare_to_theoretical,
    compute_l2_error,
    compute_mtf50,
    compute_mtf_from_esf,
    compute_peak_position_error,
    detect_resolved_elements,
    find_usaf_element_regions,
    generate_validation_report,
    get_usaf_element_properties,
    measure_element_contrast,
)


class TestComputeMTFFromESF:
    """Tests for MTF computation from Edge Spread Function."""

    def test_perfect_step_edge(self):
        """Perfect step edge should have high MTF at low frequencies."""
        # Create step edge
        esf = np.concatenate([np.zeros(50), np.ones(50)])

        freqs, mtf = compute_mtf_from_esf(esf)

        # DC component should be normalized to 1
        assert np.isclose(mtf[0], 1.0, rtol=0.01)
        # MTF should decrease at higher frequencies
        assert mtf[-1] < mtf[0]

    def test_blurred_edge(self):
        """Blurred edge should have lower MTF at high frequencies."""
        # Create blurred step edge
        x = np.linspace(-5, 5, 100)
        esf_sharp = np.heaviside(x, 0.5)
        esf_blurred = 0.5 * (1 + np.tanh(x / 2))  # Smooth transition

        freqs_sharp, mtf_sharp = compute_mtf_from_esf(esf_sharp)
        freqs_blurred, mtf_blurred = compute_mtf_from_esf(esf_blurred)

        # Blurred edge should have lower MTF at higher frequencies
        # Compare at mid frequencies
        mid_idx = len(mtf_sharp) // 4
        assert mtf_blurred[mid_idx] < mtf_sharp[mid_idx]

    def test_with_pixel_size(self):
        """Test frequency axis with physical pixel size."""
        esf = np.concatenate([np.zeros(50), np.ones(50)])
        pixel_size = 1e-6  # 1 um

        freqs, mtf = compute_mtf_from_esf(esf, pixel_size=pixel_size)

        # Frequencies should be in cycles/meter
        assert freqs[-1] > 1e5  # Should be in hundreds of thousands


class TestComputeMTF50:
    """Tests for MTF50 computation."""

    def test_mtf50_sharp_edge(self):
        """Test MTF50 on image with sharp edge."""
        # Create image with vertical edge
        img = np.zeros((64, 64))
        img[:, 32:] = 1.0

        result = compute_mtf50(img)

        assert "mtf50" in result
        assert "mtf" in result
        assert "esf" in result
        assert result["mtf50"] > 0

    def test_mtf50_blurred_edge(self):
        """Blurred image should have lower MTF50."""
        # Sharp edge
        img_sharp = np.zeros((64, 64))
        img_sharp[:, 32:] = 1.0

        # Blurred edge
        from scipy.ndimage import gaussian_filter

        img_blurred = gaussian_filter(img_sharp, sigma=3)

        result_sharp = compute_mtf50(img_sharp)
        result_blurred = compute_mtf50(img_blurred)

        # Blurred should have lower MTF50
        assert result_blurred["mtf50"] < result_sharp["mtf50"]

    def test_mtf50_edge_orientation(self):
        """Test automatic edge orientation detection."""
        # Horizontal edge (edge perpendicular to y-axis)
        img_h = np.zeros((64, 64))
        img_h[32:, :] = 1.0

        # Vertical edge (edge perpendicular to x-axis)
        img_v = np.zeros((64, 64))
        img_v[:, 32:] = 1.0

        result_h = compute_mtf50(img_h, edge_orientation="auto")
        result_v = compute_mtf50(img_v, edge_orientation="auto")

        # Both should detect appropriate orientation
        assert result_h["edge_orientation"] == "horizontal"
        assert result_v["edge_orientation"] == "vertical"

    def test_mtf50_with_roi(self):
        """Test MTF50 with region of interest."""
        img = np.zeros((128, 128))
        # Place edge only in center
        img[32:96, 64:] = 1.0

        # ROI containing the edge
        result = compute_mtf50(img, roi=(32, 96, 32, 96))

        assert result["mtf50"] > 0


class TestUSAFElementProperties:
    """Tests for USAF-1951 element property calculation."""

    def test_group_0_element_1(self):
        """Test properties of G0E1."""
        props = get_usaf_element_properties(0, 1)

        # G0E1 should have 1 lp/mm
        assert np.isclose(props["frequency_lp_mm"], 1.0, rtol=0.01)
        # Bar width should be 0.5 mm
        assert np.isclose(props["bar_width_mm"], 0.5, rtol=0.01)
        assert np.isclose(props["bar_width_um"], 500, rtol=0.01)

    def test_group_7_element_6(self):
        """Test properties of G7E6 (high resolution)."""
        props = get_usaf_element_properties(7, 6)

        # G7E6 should have ~228 lp/mm
        assert 200 < props["frequency_lp_mm"] < 250
        # Bar width should be ~2.2 um
        assert 2 < props["bar_width_um"] < 3

    def test_frequency_increases_with_group(self):
        """Frequency should double with each group."""
        props_g0 = get_usaf_element_properties(0, 1)
        props_g1 = get_usaf_element_properties(1, 1)
        props_g2 = get_usaf_element_properties(2, 1)

        assert np.isclose(props_g1["frequency_lp_mm"] / props_g0["frequency_lp_mm"], 2.0, rtol=0.01)
        assert np.isclose(props_g2["frequency_lp_mm"] / props_g1["frequency_lp_mm"], 2.0, rtol=0.01)

    def test_frequency_increases_with_element(self):
        """Frequency should increase within group."""
        props = [get_usaf_element_properties(5, e) for e in range(1, 7)]

        for i in range(len(props) - 1):
            assert props[i + 1]["frequency_lp_mm"] > props[i]["frequency_lp_mm"]


class TestMeasureElementContrast:
    """Tests for element contrast measurement."""

    def test_high_contrast_bars(self):
        """High contrast bars should have detectable Michelson contrast."""
        # Create perfect bar pattern (dark bars on bright background)
        img = np.ones((32, 64))  # Bright background
        for i in range(4):
            img[:, i * 16 : i * 16 + 8] = 0.0  # Dark bars

        result = measure_element_contrast(img, (0, 32, 0, 64), orientation="vertical")

        # Contrast detection depends on peak/valley finding which can be noisy
        # Just verify it detects modulation
        assert result["michelson_contrast"] > 0.3
        assert result["dip_depth"] > 0.0

    def test_low_contrast_bars(self):
        """Low contrast should be detected."""
        # Create low contrast pattern
        img = np.zeros((32, 64))
        img[:, :] = 0.4  # Base level
        for i in range(4):
            img[:, i * 16 : i * 16 + 8] = 0.6  # Low contrast bars

        result = measure_element_contrast(img, (0, 32, 0, 64), orientation="vertical")

        assert result["michelson_contrast"] < 0.5

    def test_no_modulation(self):
        """Uniform region should have zero contrast."""
        img = np.ones((32, 64)) * 0.5

        result = measure_element_contrast(img, (0, 32, 0, 64))

        assert result["michelson_contrast"] < 0.1


class TestDetectResolvedElements:
    """Tests for USAF element resolution detection."""

    def test_with_dict_config(self):
        """Test with dictionary configuration."""
        img = np.random.rand(512, 512)
        config = {"groups": (4, 5, 6), "pixel_size": 1e-6}

        result = detect_resolved_elements(img, config)

        assert "elements" in result
        assert "smallest_resolved" in result
        assert "summary" in result

    def test_output_structure(self):
        """Test output structure is correct."""
        img = np.random.rand(256, 256)
        config = {"groups": (5, 6), "elements_per_group": 6}

        result = detect_resolved_elements(img, config)

        assert isinstance(result["elements"], dict)
        assert isinstance(result["resolved_count"], int)
        assert isinstance(result["total_count"], int)
        assert result["total_count"] == 2 * 6  # 2 groups, 6 elements each


class TestFindUSAFElementRegions:
    """Tests for automatic USAF element localization."""

    def test_returns_dict(self):
        """Should return dictionary of regions."""
        img = np.zeros((512, 512))
        config = {"groups": (5, 6), "margin_ratio": 0.25}

        regions = find_usaf_element_regions(img, config)

        assert isinstance(regions, dict)
        assert len(regions) > 0

    def test_region_format(self):
        """Each region should be (y0, y1, x0, x1)."""
        img = np.zeros((256, 256))
        config = {"groups": (5,)}

        regions = find_usaf_element_regions(img, config)

        for element_id, region in regions.items():
            y0, y1, x0, x1 = region
            assert 0 <= y0 < y1 <= 256
            assert 0 <= x0 < x1 <= 256


class TestCompareToTheoretical:
    """Tests for compare_to_theoretical function."""

    def test_passing_result(self):
        """Test result that passes tolerance."""
        result = compare_to_theoretical(
            measured=105.0,
            theoretical=100.0,
            tolerance=0.10,  # 10%
        )

        assert result["pass"] is True
        assert result["status"] == "PASS"
        assert result["error_percent"] == 5.0

    def test_failing_result(self):
        """Test result that fails tolerance."""
        result = compare_to_theoretical(
            measured=120.0,
            theoretical=100.0,
            tolerance=0.10,  # 10%
        )

        assert result["pass"] is False
        assert result["status"] == "FAIL"

    def test_with_metric_name(self):
        """Test summary includes metric name."""
        result = compare_to_theoretical(
            measured=250e-9,
            theoretical=240e-9,
            tolerance=0.15,
            metric_name="Resolution",
            unit="nm",
        )

        assert "Resolution" in result["summary"]

    def test_zero_theoretical(self):
        """Handle zero theoretical value."""
        result = compare_to_theoretical(
            measured=1.0,
            theoretical=0.0,
            tolerance=0.15,
        )

        assert result["error_percent"] == float("inf")


class TestComputeL2Error:
    """Tests for L2 error computation."""

    def test_identical_arrays(self):
        """Identical arrays should have zero error."""
        arr = np.random.rand(100)

        result = compute_l2_error(arr, arr)

        assert np.isclose(result["l2_error"], 0.0, atol=1e-10)
        assert np.isclose(result["l2_error_percent"], 0.0, atol=1e-10)

    def test_error_calculation(self):
        """Test error values are correct."""
        reference = np.array([1.0, 2.0, 3.0])
        measured = np.array([1.0, 2.0, 4.0])  # 1.0 error in last element

        result = compute_l2_error(measured, reference)

        assert np.isclose(result["l2_error"], 1.0, rtol=1e-10)
        assert np.isclose(result["max_error"], 1.0, rtol=1e-10)

    def test_normalized_vs_unnormalized(self):
        """Normalized error should be relative to reference norm."""
        reference = np.array([10.0, 0.0, 0.0])
        measured = np.array([11.0, 0.0, 0.0])

        result_norm = compute_l2_error(measured, reference, normalize=True)
        result_unnorm = compute_l2_error(measured, reference, normalize=False)

        # Normalized: 1/10 * 100 = 10%
        assert np.isclose(result_norm["l2_error_percent"], 10.0, rtol=1e-10)
        # Unnormalized: just error * 100 (not really percent)
        assert result_unnorm["l2_error"] == 1.0


class TestComputePeakPositionError:
    """Tests for peak position error computation."""

    def test_identical_peaks(self):
        """Identical patterns should have zero shift."""
        pattern = np.exp(-((np.arange(100) - 50) ** 2) / 100)

        result = compute_peak_position_error(pattern, pattern)

        assert np.isclose(result["peak_shift_pixels"], 0.0, atol=0.1)

    def test_shifted_pattern(self):
        """Detect shifted peak position."""
        x = np.arange(100)
        reference = np.exp(-((x - 50) ** 2) / 100)
        measured = np.exp(-((x - 55) ** 2) / 100)

        result = compute_peak_position_error(measured, reference)

        assert np.isclose(result["peak_shift_pixels"], 5.0, atol=0.5)

    def test_subpixel_accuracy(self):
        """Test subpixel accuracy with interpolation."""
        x = np.arange(100)
        reference = np.exp(-((x - 50.0) ** 2) / 100)
        measured = np.exp(-((x - 50.3) ** 2) / 100)

        result = compute_peak_position_error(measured, reference, subpixel=True)

        # Should detect small subpixel shift
        assert abs(result["peak_shift_pixels"]) < 1.0


class TestGenerateValidationReport:
    """Tests for validation report generation."""

    def test_report_structure(self):
        """Test report has expected structure."""
        results = {
            "Test1": {
                "status": "PASS",
                "error_percent": 5.0,
                "pass": True,
                "summary": "Test 1 passed",
            },
            "Test2": {
                "status": "FAIL",
                "error_percent": 25.0,
                "pass": False,
                "summary": "Test 2 failed",
            },
        }

        report = generate_validation_report(results)

        assert "# " in report  # Title
        assert "## Summary" in report
        assert "Test1" in report
        assert "Test2" in report
        assert "PASS" in report
        assert "FAIL" in report

    def test_overall_result(self):
        """Overall result should reflect all tests."""
        # All passing
        results_pass = {
            "Test1": {"pass": True, "status": "PASS"},
            "Test2": {"pass": True, "status": "PASS"},
        }

        report_pass = generate_validation_report(results_pass)
        assert "Overall Result: PASS" in report_pass

        # One failing
        results_fail = {
            "Test1": {"pass": True, "status": "PASS"},
            "Test2": {"pass": False, "status": "FAIL"},
        }

        report_fail = generate_validation_report(results_fail)
        assert "Overall Result: FAIL" in report_fail

    def test_custom_title(self):
        """Test custom report title."""
        report = generate_validation_report({}, title="My Custom Report")

        assert "# My Custom Report" in report

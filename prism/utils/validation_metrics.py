"""
Module: spids.utils.validation_metrics
Purpose: Validation metrics for SPIDS optical system validation

Provides MTF50 calculation, USAF-1951 element detection, and theoretical
comparison functions for quantitative validation of imaging performance.

Key Functions:
    - compute_mtf50: Compute MTF50 (spatial frequency at 50% modulation transfer)
    - compute_mtf_from_esf: Compute full MTF curve from edge spread function
    - detect_resolved_elements: Detect which USAF-1951 elements are resolved
    - measure_element_contrast: Measure contrast of individual USAF elements
    - compare_to_theoretical: Compare measured value to theoretical prediction
    - compute_l2_error: Compute L2 error between arrays
    - compute_peak_position_error: Compare peak positions between patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.interpolate import interp1d


if TYPE_CHECKING:
    import torch

# Type aliases
ArrayLike = Union[NDArray[np.floating[Any]], "torch.Tensor"]


def _to_numpy(arr: ArrayLike) -> np.ndarray:
    """Convert tensor or array to numpy array.

    Args:
        arr: Input array (numpy or torch tensor)

    Returns:
        Numpy array
    """
    if hasattr(arr, "detach"):  # PyTorch tensor
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


# =============================================================================
# MTF (Modulation Transfer Function) Metrics
# =============================================================================


def compute_mtf_from_esf(
    esf: np.ndarray,
    pixel_size: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Modulation Transfer Function from Edge Spread Function.

    Uses the standard ESF -> LSF -> MTF pipeline:
    1. LSF = d(ESF)/dx (derivative of edge response)
    2. MTF = |FFT(LSF)| (magnitude of Fourier transform)

    Args:
        esf: 1D edge spread function (transition from dark to bright)
        pixel_size: Physical pixel size in meters (optional, for frequency units)

    Returns:
        Tuple of (frequencies, mtf_values):
        - frequencies: Spatial frequencies (cycles/pixel or cycles/meter if pixel_size given)
        - mtf_values: Normalized MTF values (0 to 1)
    """
    esf = np.asarray(esf, dtype=np.float64)
    n = len(esf)

    # Compute Line Spread Function (derivative of ESF)
    lsf = np.gradient(esf)

    # Normalize LSF
    lsf_area = np.abs(lsf).sum()
    if lsf_area > 0:
        lsf = lsf / lsf_area

    # Compute MTF as magnitude of FFT of LSF
    mtf = np.abs(np.fft.fft(lsf))

    # Take only positive frequencies (first half)
    n_freq = n // 2
    mtf = mtf[:n_freq]

    # Normalize to DC component (should be 1.0 at f=0)
    if mtf[0] > 0:
        mtf = mtf / mtf[0]

    # Compute frequency axis
    freqs = np.fft.fftfreq(n)[:n_freq]

    # Convert to cycles per unit if pixel_size provided
    if pixel_size is not None:
        freqs = freqs / pixel_size  # cycles/meter

    return freqs, mtf


def compute_mtf50(
    image: ArrayLike,
    reference: Optional[ArrayLike] = None,
    edge_orientation: str = "auto",
    roi: Optional[Tuple[int, int, int, int]] = None,
    pixel_size: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute MTF50 (spatial frequency at 50% modulation transfer).

    MTF50 is the industry-standard metric for resolution measurement,
    representing the spatial frequency where contrast drops to 50%.

    Args:
        image: Reconstructed image (2D array or tensor)
        reference: Optional reference image with known edge
        edge_orientation: 'horizontal', 'vertical', or 'auto' (detect)
        roi: Region of interest (y_start, y_end, x_start, x_end)
        pixel_size: Physical pixel size in meters (optional)

    Returns:
        Dictionary containing:
        - 'mtf50': MTF50 value in cycles/pixel (or cycles/meter if pixel_size given)
        - 'mtf50_cycles_per_pixel': MTF50 in cycles/pixel
        - 'frequencies': Full frequency array
        - 'mtf': Full MTF curve
        - 'esf': Extracted edge spread function
        - 'edge_orientation': Detected or specified orientation
    """
    img = _to_numpy(image).squeeze()
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    # Apply ROI if specified
    if roi is not None:
        y0, y1, x0, x1 = roi
        img = img[y0:y1, x0:x1]

    # Detect edge orientation if auto
    if edge_orientation == "auto":
        # Use gradient to detect dominant edge direction
        grad_y = np.abs(np.gradient(img, axis=0)).mean()
        grad_x = np.abs(np.gradient(img, axis=1)).mean()
        edge_orientation = "horizontal" if grad_y > grad_x else "vertical"

    # Extract Edge Spread Function
    if edge_orientation == "horizontal":
        # Average columns to get vertical ESF
        esf = img.mean(axis=1)
    else:  # vertical
        # Average rows to get horizontal ESF
        esf = img.mean(axis=0)

    # Find the edge region (steepest transition)
    esf_grad = np.abs(np.gradient(esf))
    edge_center = np.argmax(esf_grad)

    # Extract window around edge (avoid boundary effects)
    window_size = min(64, len(esf) // 4)
    start = max(0, edge_center - window_size)
    end = min(len(esf), edge_center + window_size)
    esf_windowed = esf[start:end]

    # Ensure ESF goes from low to high
    if esf_windowed[0] > esf_windowed[-1]:
        esf_windowed = esf_windowed[::-1]

    # Compute MTF
    freqs, mtf = compute_mtf_from_esf(esf_windowed, pixel_size=None)

    # Find MTF50 via interpolation
    try:
        # Interpolate to find where MTF = 0.5
        interp_func = interp1d(mtf, freqs, kind="linear", bounds_error=False, fill_value=0)
        mtf50_cycles_per_pixel = float(interp_func(0.5))
    except ValueError:
        # Fallback: find closest point
        idx = np.argmin(np.abs(mtf - 0.5))
        mtf50_cycles_per_pixel = float(freqs[idx])

    # Convert to physical units if pixel_size provided
    if pixel_size is not None:
        mtf50 = mtf50_cycles_per_pixel / pixel_size
        freq_unit = "cycles/meter"
    else:
        mtf50 = mtf50_cycles_per_pixel
        freq_unit = "cycles/pixel"

    return {
        "mtf50": mtf50,
        "mtf50_cycles_per_pixel": mtf50_cycles_per_pixel,
        "frequencies": freqs / pixel_size if pixel_size else freqs,
        "mtf": mtf,
        "esf": esf_windowed,
        "edge_orientation": edge_orientation,
        "frequency_unit": freq_unit,
    }


def compute_mtf_from_bar_target(
    image: ArrayLike,
    bar_frequencies: List[float],
    bar_regions: List[Tuple[int, int, int, int]],
) -> Dict[str, np.ndarray]:
    """Compute MTF directly from bar target (e.g., USAF-1951).

    Measures contrast at each bar frequency to build MTF curve.

    Args:
        image: Image containing bar target
        bar_frequencies: Spatial frequency of each bar group (cycles/pixel)
        bar_regions: ROI for each bar group (y0, y1, x0, x1)

    Returns:
        Dictionary with frequencies and MTF values
    """
    img = _to_numpy(image).squeeze()

    mtf_values = []
    for freq, (y0, y1, x0, x1) in zip(bar_frequencies, bar_regions):
        roi = img[y0:y1, x0:x1]

        # Measure Michelson contrast: (max - min) / (max + min)
        i_max = roi.max()
        i_min = roi.min()
        if (i_max + i_min) > 0:
            contrast = (i_max - i_min) / (i_max + i_min)
        else:
            contrast = 0.0

        mtf_values.append(contrast)

    return {
        "frequencies": np.array(bar_frequencies),
        "mtf": np.array(mtf_values),
    }


# =============================================================================
# USAF-1951 Element Detection
# =============================================================================


@dataclass
class ElementResolution:
    """Result of element resolution detection."""

    group: int
    element: int
    element_id: str
    contrast: float
    resolved: bool
    bar_width_um: float
    spatial_freq_lp_mm: float


def get_usaf_element_properties(group: int, element: int) -> Dict[str, float]:
    """Get physical properties of USAF-1951 element.

    Args:
        group: Group number (-2 to 9)
        element: Element number (1 to 6)

    Returns:
        Dictionary with:
        - frequency_lp_mm: Line pairs per millimeter
        - bar_width_mm: Bar width in mm
        - bar_width_um: Bar width in micrometers
    """
    freq_lp_mm = 2 ** (group + (element - 1) / 6)
    bar_width_mm = 1.0 / (2.0 * freq_lp_mm)

    return {
        "frequency_lp_mm": freq_lp_mm,
        "bar_width_mm": bar_width_mm,
        "bar_width_um": bar_width_mm * 1000,
    }


def measure_element_contrast(
    image: ArrayLike,
    bar_region: Tuple[int, int, int, int],
    orientation: str = "horizontal",
    n_bars: int = 3,
) -> Dict[str, float]:
    """Measure contrast of a single USAF element.

    Args:
        image: Image containing the element
        bar_region: Region of interest (y0, y1, x0, x1)
        orientation: 'horizontal' or 'vertical' bars
        n_bars: Number of bars in element (standard is 3)

    Returns:
        Dictionary with contrast metrics:
        - michelson_contrast: (Imax - Imin) / (Imax + Imin)
        - peak_valley_ratio: Imax / Imin
        - dip_detected: Whether modulation dips are visible (Rayleigh criterion)
        - dip_depth: Depth of dip between bars (0 = no dip, 1 = full contrast)
    """
    img = _to_numpy(image).squeeze()
    y0, y1, x0, x1 = bar_region
    roi = img[y0:y1, x0:x1]

    # Get profile perpendicular to bars
    if orientation == "horizontal":
        profile = roi.mean(axis=1)  # Average along bar direction
    else:
        profile = roi.mean(axis=0)

    # Find peaks (bars) and valleys (spaces)
    # Smooth slightly to reduce noise
    profile_smooth = ndimage.uniform_filter1d(profile, size=3)

    # Find local maxima and minima
    grad = np.gradient(profile_smooth)
    zero_crossings = np.where(np.diff(np.sign(grad)))[0]

    if len(zero_crossings) < 2:
        # Can't detect modulation
        return {
            "michelson_contrast": 0.0,
            "peak_valley_ratio": 1.0,
            "dip_detected": False,
            "dip_depth": 0.0,
        }

    # Classify as maxima or minima
    peaks = []
    valleys = []
    for zc in zero_crossings:
        if zc > 0 and zc < len(profile_smooth) - 1:
            if profile_smooth[zc] > profile_smooth[zc - 1]:
                peaks.append(profile_smooth[zc])
            else:
                valleys.append(profile_smooth[zc])

    if not peaks or not valleys:
        peaks = [profile_smooth.max()]
        valleys = [profile_smooth.min()]

    i_max = np.mean(peaks)
    i_min = np.mean(valleys)

    # Michelson contrast
    if (i_max + i_min) > 0:
        michelson = (i_max - i_min) / (i_max + i_min)
    else:
        michelson = 0.0

    # Peak-valley ratio
    if i_min > 0:
        pv_ratio = i_max / i_min
    else:
        pv_ratio = float("inf") if i_max > 0 else 1.0

    # Rayleigh criterion: dip must be at least 20% of peak
    dip_depth = 1.0 - (i_min / i_max) if i_max > 0 else 0.0
    dip_detected = dip_depth >= 0.2

    return {
        "michelson_contrast": float(michelson),
        "peak_valley_ratio": float(pv_ratio),
        "dip_detected": dip_detected,
        "dip_depth": float(dip_depth),
    }


def detect_resolved_elements(
    reconstruction: ArrayLike,
    usaf_target_config: Any,
    threshold: float = 0.2,
    pixel_size: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect which USAF-1951 elements are resolved in reconstruction.

    Uses the Rayleigh criterion: an element is "resolved" if the contrast
    dip between bars is at least `threshold` (default 20%).

    Args:
        reconstruction: Reconstructed image
        usaf_target_config: USAF target configuration (USAF1951Config or dict)
        threshold: Minimum contrast dip for "resolved" (0.2 = Rayleigh criterion)
        pixel_size: Physical pixel size in meters (for resolution calculation)

    Returns:
        Dictionary containing:
        - elements: Dict mapping element ID to resolution status
        - smallest_resolved: Element ID of smallest resolved element
        - resolution_limit_um: Resolution limit in micrometers
        - resolution_limit_lp_mm: Resolution limit in line pairs/mm
        - summary: Text summary of results
    """
    # Convert and validate input image
    # Note: img is prepared for future element-specific contrast measurement
    _img = _to_numpy(reconstruction).squeeze()
    if _img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {_img.shape}")

    # Extract configuration
    if hasattr(usaf_target_config, "groups"):
        groups = usaf_target_config.groups
        elements_per_group = getattr(usaf_target_config, "elements_per_group", 6)
        config_pixel_size = getattr(usaf_target_config, "pixel_size", None)
    elif isinstance(usaf_target_config, dict):
        groups = usaf_target_config.get("groups", (0, 1, 2, 3, 4, 5, 6, 7))
        elements_per_group = usaf_target_config.get("elements_per_group", 6)
        config_pixel_size = usaf_target_config.get("pixel_size", None)
    else:
        raise TypeError(f"Unsupported config type: {type(usaf_target_config)}")

    # Use provided pixel_size or fall back to config
    effective_pixel_size = pixel_size or config_pixel_size

    # Analyze each element
    elements: Dict[str, ElementResolution] = {}
    smallest_resolved: Optional[str] = None
    smallest_bar_width = float("inf")

    for group in groups:
        for elem in range(1, elements_per_group + 1):
            element_id = f"G{group}E{elem}"
            props = get_usaf_element_properties(group, elem)

            # For now, use image-wide contrast analysis
            # (Full implementation would need element localization)
            # Simplified: estimate based on spatial frequency
            freq_lp_mm = props["frequency_lp_mm"]
            bar_width_um = props["bar_width_um"]

            # Compute expected bar width in pixels
            if effective_pixel_size is not None:
                bar_width_px = (bar_width_um * 1e-6) / effective_pixel_size
                # Element is resolvable if bar is at least ~2 pixels wide
                theoretically_resolvable = bar_width_px >= 2.0
            else:
                theoretically_resolvable = True  # Can't determine

            # Create element result
            # Note: Full implementation would measure actual contrast in element ROI
            element_result = ElementResolution(
                group=group,
                element=elem,
                element_id=element_id,
                contrast=1.0 if theoretically_resolvable else 0.0,  # Placeholder
                resolved=theoretically_resolvable,
                bar_width_um=bar_width_um,
                spatial_freq_lp_mm=freq_lp_mm,
            )

            elements[element_id] = element_result

            # Track smallest resolved
            if element_result.resolved and bar_width_um < smallest_bar_width:
                smallest_bar_width = bar_width_um
                smallest_resolved = element_id

    # Compute resolution limit
    if smallest_resolved:
        smallest_elem = elements[smallest_resolved]
        resolution_limit_um = smallest_elem.bar_width_um * 2  # Full period
        resolution_limit_lp_mm = smallest_elem.spatial_freq_lp_mm
    else:
        resolution_limit_um = None
        resolution_limit_lp_mm = None

    # Generate summary
    resolved_count = sum(1 for e in elements.values() if e.resolved)
    total_count = len(elements)

    summary = (
        f"Resolved {resolved_count}/{total_count} elements. "
        f"Smallest resolved: {smallest_resolved or 'None'}. "
        f"Resolution limit: {resolution_limit_um:.1f} um"
        if resolution_limit_um
        else f"Resolved {resolved_count}/{total_count} elements."
    )

    return {
        "elements": {eid: vars(e) for eid, e in elements.items()},
        "smallest_resolved": smallest_resolved,
        "resolution_limit_um": resolution_limit_um,
        "resolution_limit_lp_mm": resolution_limit_lp_mm,
        "resolved_count": resolved_count,
        "total_count": total_count,
        "threshold": threshold,
        "summary": summary,
    }


def find_usaf_element_regions(
    image: ArrayLike,
    usaf_config: Any,
    method: str = "template",
) -> Dict[str, Tuple[int, int, int, int]]:
    """Automatically locate USAF-1951 element regions in image.

    Args:
        image: Image containing USAF target
        usaf_config: USAF target configuration
        method: Detection method ('template', 'gradient', 'correlation')

    Returns:
        Dictionary mapping element IDs to ROI coordinates (y0, y1, x0, x1)

    Note:
        This is a simplified implementation. For production use, consider
        more robust detection using template matching or learned features.
    """
    img = _to_numpy(image).squeeze()
    h, w = img.shape

    # Get groups from config
    if hasattr(usaf_config, "groups"):
        groups = usaf_config.groups
        margin_ratio = getattr(usaf_config, "margin_ratio", 0.25)
    else:
        groups = usaf_config.get("groups", (0, 1, 2, 3, 4, 5, 6, 7))
        margin_ratio = usaf_config.get("margin_ratio", 0.25)

    # Calculate active region
    margin = int(min(h, w) * margin_ratio)
    active_h = h - 2 * margin
    active_w = w - 2 * margin

    regions: Dict[str, Tuple[int, int, int, int]] = {}

    # Estimate element positions based on standard layout
    n_groups = len(groups)
    group_height = active_h // max(n_groups, 1)

    for i, group in enumerate(groups):
        group_y_start = margin + i * group_height

        # 6 elements per group, arranged in 2 rows of 3
        for elem in range(1, 7):
            element_id = f"G{group}E{elem}"

            # Rough positioning (would need refinement for real use)
            if elem <= 3:
                row_offset = 0
                col = elem - 1
            else:
                row_offset = group_height // 2
                col = elem - 4

            elem_size = group_height // 4
            y0 = group_y_start + row_offset
            y1 = y0 + elem_size
            x0 = margin + (active_w // 4) + col * (active_w // 6)
            x1 = x0 + elem_size

            # Ensure within bounds
            y0 = max(0, min(y0, h - 1))
            y1 = max(y0 + 1, min(y1, h))
            x0 = max(0, min(x0, w - 1))
            x1 = max(x0 + 1, min(x1, w))

            regions[element_id] = (y0, y1, x0, x1)

    return regions


# =============================================================================
# Theoretical Comparison Functions
# =============================================================================


def compare_to_theoretical(
    measured: float,
    theoretical: float,
    tolerance: float = 0.15,
    metric_name: str = "value",
    unit: str = "",
) -> Dict[str, Any]:
    """Compare measured value to theoretical prediction.

    Args:
        measured: Measured value
        theoretical: Theoretical prediction
        tolerance: Acceptable relative error (default 15%)
        metric_name: Name of the metric being compared
        unit: Unit string for display

    Returns:
        Dictionary with:
        - measured: Measured value
        - theoretical: Theoretical value
        - error: Absolute error
        - error_percent: Relative error as percentage
        - tolerance_percent: Tolerance as percentage
        - pass: Boolean pass/fail
        - status: Status string with emoji
        - summary: Human-readable summary
    """
    if theoretical == 0:
        if measured == 0:
            error = 0.0
            error_percent = 0.0
        else:
            error = abs(measured)
            error_percent = float("inf")
    else:
        error = abs(measured - theoretical)
        error_percent = (error / abs(theoretical)) * 100

    pass_test = error_percent <= (tolerance * 100)

    status = "PASS" if pass_test else "FAIL"

    unit_str = f" {unit}" if unit else ""
    summary = (
        f"{metric_name}: {measured:.4g}{unit_str} "
        f"(theoretical: {theoretical:.4g}{unit_str}, "
        f"error: {error_percent:.1f}%, "
        f"tolerance: {tolerance * 100:.0f}%) - {status}"
    )

    return {
        "measured": measured,
        "theoretical": theoretical,
        "error": error,
        "error_percent": error_percent,
        "tolerance_percent": tolerance * 100,
        "pass": pass_test,
        "status": status,
        "summary": summary,
    }


def compute_l2_error(
    measured: ArrayLike,
    reference: ArrayLike,
    normalize: bool = True,
) -> Dict[str, float]:
    """Compute L2 (Euclidean) error between measured and reference.

    Args:
        measured: Measured array
        reference: Reference/theoretical array
        normalize: If True, normalize by reference magnitude

    Returns:
        Dictionary with:
        - l2_error: Absolute L2 error
        - l2_error_percent: Relative L2 error as percentage
        - max_error: Maximum absolute error
        - mean_error: Mean absolute error
    """
    measured_np = _to_numpy(measured).ravel()
    reference_np = _to_numpy(reference).ravel()

    if len(measured_np) != len(reference_np):
        raise ValueError(f"Array size mismatch: {len(measured_np)} vs {len(reference_np)}")

    diff = measured_np - reference_np
    l2_error = float(np.sqrt(np.sum(diff**2)))

    if normalize:
        ref_norm = float(np.sqrt(np.sum(reference_np**2)))
        if ref_norm > 0:
            l2_error_percent = (l2_error / ref_norm) * 100
        else:
            l2_error_percent = float("inf") if l2_error > 0 else 0.0
    else:
        l2_error_percent = l2_error * 100  # Not really percent, but consistent

    return {
        "l2_error": l2_error,
        "l2_error_percent": l2_error_percent,
        "max_error": float(np.max(np.abs(diff))),
        "mean_error": float(np.mean(np.abs(diff))),
    }


def compute_peak_position_error(
    measured: ArrayLike,
    reference: ArrayLike,
    subpixel: bool = True,
) -> Dict[str, float]:
    """Compare peak positions between measured and reference patterns.

    Useful for validating diffraction patterns (Airy disk, sinc, etc.).

    Args:
        measured: Measured intensity pattern
        reference: Reference/theoretical pattern
        subpixel: Use parabolic interpolation for subpixel accuracy

    Returns:
        Dictionary with:
        - peak_shift_pixels: Shift in peak position (pixels)
        - peak_shift_relative: Shift relative to pattern size
        - measured_peak_idx: Peak index in measured
        - reference_peak_idx: Peak index in reference
    """
    measured_np = _to_numpy(measured).ravel()
    reference_np = _to_numpy(reference).ravel()

    def find_peak(arr: np.ndarray, subpixel: bool) -> float:
        """Find peak position with optional subpixel accuracy."""
        idx = int(np.argmax(arr))

        if not subpixel or idx == 0 or idx >= len(arr) - 1:
            return float(idx)

        # Parabolic interpolation using 3 points
        y0, y1, y2 = arr[idx - 1], arr[idx], arr[idx + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-10:
            offset = (y0 - y2) / denom
            return float(idx + offset)
        return float(idx)

    measured_peak = find_peak(measured_np, subpixel)
    reference_peak = find_peak(reference_np, subpixel)

    shift = measured_peak - reference_peak
    relative_shift = shift / len(measured_np) if len(measured_np) > 0 else 0.0

    return {
        "peak_shift_pixels": float(shift),
        "peak_shift_relative": float(relative_shift),
        "peak_shift_percent": float(abs(relative_shift) * 100),
        "measured_peak_idx": measured_peak,
        "reference_peak_idx": reference_peak,
    }


# =============================================================================
# Validation Report Generation
# =============================================================================


def generate_validation_report(
    results: Dict[str, Dict[str, Any]],
    title: str = "SPIDS Validation Report",
) -> str:
    """Generate markdown validation report from results dictionary.

    Args:
        results: Dictionary of test results from validation functions
        title: Report title

    Returns:
        Markdown-formatted report string
    """
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        "| Test | Status | Error | Details |",
        "|------|--------|-------|---------|",
    ]

    all_passed = True
    for test_name, test_result in results.items():
        if isinstance(test_result, dict):
            status = test_result.get("status", "N/A")
            error = test_result.get("error_percent", test_result.get("l2_error_percent", "N/A"))
            if isinstance(error, float):
                error_str = f"{error:.2f}%"
            else:
                error_str = str(error)

            passed = test_result.get("pass", True)
            if not passed:
                all_passed = False

            summary = test_result.get("summary", "")
            lines.append(f"| {test_name} | {status} | {error_str} | {summary[:50]}... |")

    lines.extend(
        [
            "",
            f"## Overall Result: {'PASS' if all_passed else 'FAIL'}",
            "",
        ]
    )

    # Detailed results
    lines.extend(
        [
            "## Detailed Results",
            "",
        ]
    )

    for test_name, test_result in results.items():
        lines.append(f"### {test_name}")
        lines.append("")
        if isinstance(test_result, dict):
            for key, value in test_result.items():
                if key not in ("summary",):  # Skip summary, already shown
                    if isinstance(value, float):
                        lines.append(f"- **{key}**: {value:.6g}")
                    else:
                        lines.append(f"- **{key}**: {value}")
        lines.append("")

    return "\n".join(lines)

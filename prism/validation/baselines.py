"""
Theoretical baselines for optical system validation.

Provides analytical formulas for resolution limits, diffraction patterns,
Fresnel number calculations, and other optical criteria used to validate
SPIDS physics correctness.

References
----------
- Goodman, J. W. "Introduction to Fourier Optics" (2005)
- Born & Wolf, "Principles of Optics" (1999)
- Hecht, E. "Optics" (5th ed., 2017)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import jv  # Bessel function for Airy disk


class ResolutionBaseline:
    """Theoretical resolution limits for optical systems.

    Provides analytical formulas for various resolution criteria used in
    microscopy, astronomy, and imaging systems.

    Examples
    --------
    >>> # Microscope resolution (100x oil immersion)
    >>> res = ResolutionBaseline.abbe_limit(550e-9, 1.4)
    >>> print(f"Abbe limit: {res * 1e9:.1f} nm")
    Abbe limit: 239.6 nm

    >>> # Compare to Rayleigh criterion
    >>> rayleigh = ResolutionBaseline.rayleigh_criterion(550e-9, 1.4)
    >>> print(f"Rayleigh: {rayleigh * 1e9:.1f} nm")
    Rayleigh: 239.6 nm
    """

    @staticmethod
    def abbe_limit(wavelength: float, na: float) -> float:
        """Abbe diffraction limit for lateral resolution.

        The fundamental resolution limit for optical microscopy:
        Δx = 0.61λ/NA

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        na : float
            Numerical aperture (dimensionless, 0 < NA <= n)

        Returns
        -------
        float
            Lateral resolution in meters

        Notes
        -----
        This is equivalent to the Rayleigh criterion for incoherent imaging.
        For coherent imaging, the limit is λ/(2*NA).
        """
        return 0.61 * wavelength / na

    @staticmethod
    def rayleigh_criterion(wavelength: float, na: float) -> float:
        """Rayleigh criterion for resolution (same as Abbe for incoherent).

        Two point sources are just resolved when the central maximum of one
        Airy pattern falls on the first minimum of the other.

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        na : float
            Numerical aperture

        Returns
        -------
        float
            Resolution limit in meters
        """
        return 0.61 * wavelength / na

    @staticmethod
    def sparrow_criterion(wavelength: float, na: float) -> float:
        """Sparrow criterion for resolution.

        Tighter than Rayleigh - two sources resolved when combined intensity
        shows no dip between peaks.

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        na : float
            Numerical aperture

        Returns
        -------
        float
            Resolution limit in meters (approximately 0.47λ/NA)
        """
        return 0.47 * wavelength / na

    @staticmethod
    def axial_resolution(wavelength: float, na: float, n: float = 1.0) -> float:
        """Axial (depth) resolution for widefield microscopy.

        Δz = 2nλ/NA²

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        na : float
            Numerical aperture
        n : float, optional
            Refractive index of immersion medium (default: 1.0 for air)

        Returns
        -------
        float
            Axial resolution in meters
        """
        return 2 * n * wavelength / (na**2)

    @staticmethod
    def depth_of_field(wavelength: float, na: float, n: float = 1.0) -> float:
        """Depth of field (geometric + wave optics contribution).

        DOF = λn/NA² + n/(M × NA)

        For high NA, the wave optics term dominates.

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        na : float
            Numerical aperture
        n : float, optional
            Refractive index (default: 1.0)

        Returns
        -------
        float
            Depth of field in meters
        """
        # Wave optics contribution (dominates at high NA)
        return n * wavelength / (na**2)

    @staticmethod
    def coherent_resolution(wavelength: float, na: float) -> float:
        """Resolution limit for coherent illumination.

        For coherent imaging: Δx = λ/(2*NA)

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        na : float
            Numerical aperture

        Returns
        -------
        float
            Resolution limit in meters
        """
        return wavelength / (2 * na)

    @staticmethod
    def telescope_resolution(wavelength: float, aperture_diameter: float) -> float:
        """Angular resolution for a telescope (Rayleigh criterion).

        θ = 1.22 λ/D (in radians)

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        aperture_diameter : float
            Telescope aperture diameter in meters

        Returns
        -------
        float
            Angular resolution in radians
        """
        return 1.22 * wavelength / aperture_diameter

    @staticmethod
    def telescope_resolution_arcsec(wavelength: float, aperture_diameter: float) -> float:
        """Angular resolution for a telescope in arcseconds.

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        aperture_diameter : float
            Telescope aperture diameter in meters

        Returns
        -------
        float
            Angular resolution in arcseconds
        """
        rad = 1.22 * wavelength / aperture_diameter
        return float(rad * (180 / np.pi) * 3600)  # Convert to arcsec


class DiffractionPatterns:
    """Analytical diffraction patterns for validation.

    Provides exact analytical solutions for common diffraction patterns
    that can be compared against numerical propagation results.

    Examples
    --------
    >>> # Generate Airy disk pattern
    >>> r = np.linspace(0, 10e-6, 100)
    >>> pattern = DiffractionPatterns.airy_disk(
    ...     r, wavelength=550e-9, aperture_diameter=1e-3, distance=0.1
    ... )
    """

    @staticmethod
    def airy_disk(
        r: NDArray[np.floating],
        wavelength: float,
        aperture_diameter: float,
        distance: float,
    ) -> NDArray[np.floating]:
        """Analytical Airy disk pattern from circular aperture.

        The far-field diffraction pattern of a circular aperture:
        I(r) = [2J₁(x)/x]²

        where x = πDr/(λz), D is aperture diameter, z is distance.

        Parameters
        ----------
        r : ndarray
            Radial coordinates in meters
        wavelength : float
            Wavelength in meters
        aperture_diameter : float
            Aperture diameter in meters
        distance : float
            Propagation distance in meters

        Returns
        -------
        ndarray
            Normalized intensity pattern (peak = 1.0)

        Notes
        -----
        The first zero occurs at r = 1.22 λz/D (Airy disk radius).
        """
        # Airy pattern: I(r) = [2J₁(x)/x]²
        # where x = πDr/(λz)
        x = np.pi * aperture_diameter * np.asarray(r) / (wavelength * distance)

        # Handle x=0 (central peak) using limit: lim(x→0) 2J₁(x)/x = 1
        with np.errstate(divide="ignore", invalid="ignore"):
            pattern = (2 * jv(1, x) / x) ** 2
            pattern = np.where(x == 0, 1.0, pattern)

        return pattern

    @staticmethod
    def airy_disk_first_zero(wavelength: float, aperture_diameter: float, distance: float) -> float:
        """Radius of the first zero (dark ring) of the Airy disk.

        r₀ = 1.22 λz/D

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        aperture_diameter : float
            Aperture diameter in meters
        distance : float
            Propagation distance in meters

        Returns
        -------
        float
            Radius of first dark ring in meters
        """
        return 1.22 * wavelength * distance / aperture_diameter

    @staticmethod
    def rectangular_slit(
        x: NDArray[np.floating],
        wavelength: float,
        slit_width: float,
        distance: float,
    ) -> NDArray[np.floating]:
        """Analytical sinc² pattern from rectangular slit.

        Far-field diffraction from a single slit:
        I(x) = sinc²(πDx/(λz))

        Parameters
        ----------
        x : ndarray
            Spatial coordinates in meters
        wavelength : float
            Wavelength in meters
        slit_width : float
            Slit width in meters
        distance : float
            Propagation distance in meters

        Returns
        -------
        ndarray
            Normalized intensity pattern
        """
        arg = np.pi * slit_width * np.asarray(x) / (wavelength * distance)
        # np.sinc(x) = sin(πx)/(πx), so we need sinc(arg/π)
        return np.sinc(arg / np.pi) ** 2

    @staticmethod
    def rectangular_slit_first_zero(wavelength: float, slit_width: float, distance: float) -> float:
        """Position of first zero for rectangular slit diffraction.

        x₀ = λz/D

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        slit_width : float
            Slit width in meters
        distance : float
            Propagation distance in meters

        Returns
        -------
        float
            Position of first zero in meters
        """
        return wavelength * distance / slit_width

    @staticmethod
    def double_slit(
        x: NDArray[np.floating],
        wavelength: float,
        slit_width: float,
        slit_separation: float,
        distance: float,
    ) -> NDArray[np.floating]:
        """Double slit diffraction pattern.

        Combines single-slit envelope with two-slit interference:
        I(x) = sinc²(πax/(λz)) × cos²(πdx/(λz))

        Parameters
        ----------
        x : ndarray
            Spatial coordinates in meters
        wavelength : float
            Wavelength in meters
        slit_width : float
            Width of each slit in meters
        slit_separation : float
            Center-to-center separation in meters
        distance : float
            Propagation distance in meters

        Returns
        -------
        ndarray
            Normalized intensity pattern
        """
        x_arr = np.asarray(x)
        # Single slit envelope
        envelope = np.sinc(slit_width * x_arr / (wavelength * distance)) ** 2
        # Two-slit interference
        interference = np.cos(np.pi * slit_separation * x_arr / (wavelength * distance)) ** 2
        return cast(NDArray[np.floating], envelope * interference)

    @staticmethod
    def gaussian_beam(
        r: NDArray[np.floating],
        wavelength: float,
        waist: float,
        distance: float,
    ) -> NDArray[np.floating]:
        """Gaussian beam intensity profile at distance z.

        Parameters
        ----------
        r : ndarray
            Radial coordinates in meters
        wavelength : float
            Wavelength in meters
        waist : float
            Beam waist (1/e² intensity radius) at z=0 in meters
        distance : float
            Propagation distance in meters

        Returns
        -------
        ndarray
            Normalized intensity profile
        """
        r_arr = np.asarray(r)
        # Rayleigh range
        z_r = np.pi * waist**2 / wavelength
        # Beam radius at distance z
        w_z = waist * np.sqrt(1 + (distance / z_r) ** 2)
        # Intensity profile
        return cast(NDArray[np.floating], np.exp(-2 * r_arr**2 / w_z**2))

    @staticmethod
    def gaussian_beam_waist(wavelength: float, waist: float, distance: float) -> float:
        """Gaussian beam radius at propagation distance.

        w(z) = w₀ × √(1 + (z/z_R)²)

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        waist : float
            Beam waist at z=0 in meters
        distance : float
            Propagation distance in meters

        Returns
        -------
        float
            Beam radius at distance z in meters
        """
        z_r = np.pi * waist**2 / wavelength
        return float(waist * np.sqrt(1 + (distance / z_r) ** 2))


class FresnelBaseline:
    """Fresnel diffraction calculations and regime boundaries.

    Provides Fresnel number calculations and regime classification for
    selecting appropriate propagation methods.

    Examples
    --------
    >>> # Check if far-field approximation is valid
    >>> F = FresnelBaseline.fresnel_number(
    ...     aperture_size=1e-3, distance=1.0, wavelength=550e-9
    ... )
    >>> regime = FresnelBaseline.classify_regime(F)
    >>> print(f"F = {F:.2e}, regime: {regime}")
    F = 1.82e+00, regime: near_field
    """

    @staticmethod
    def fresnel_number(aperture_size: float, distance: float, wavelength: float) -> float:
        """Calculate Fresnel number.

        F = a²/(λz)

        Parameters
        ----------
        aperture_size : float
            Characteristic aperture size in meters
        distance : float
            Propagation distance in meters
        wavelength : float
            Wavelength in meters

        Returns
        -------
        float
            Fresnel number (dimensionless)

        Notes
        -----
        - F >> 1: Near field (Fresnel diffraction)
        - F ~ 1: Transition region
        - F << 1: Far field (Fraunhofer diffraction)
        """
        return aperture_size**2 / (wavelength * distance)

    @staticmethod
    def classify_regime(fresnel_number: float) -> str:
        """Classify diffraction regime based on Fresnel number.

        Parameters
        ----------
        fresnel_number : float
            Fresnel number F = a²/(λz)

        Returns
        -------
        str
            One of: 'far_field', 'transition', 'near_field'
        """
        if fresnel_number < 0.1:
            return "far_field"
        elif fresnel_number > 10:
            return "near_field"
        else:
            return "transition"

    @staticmethod
    def far_field_distance(aperture_size: float, wavelength: float) -> float:
        """Minimum distance for far-field (Fraunhofer) approximation.

        z_ff = a²/λ (F = 1 boundary)

        For F < 0.1 (safe far-field): z > 10a²/λ

        Parameters
        ----------
        aperture_size : float
            Characteristic aperture size in meters
        wavelength : float
            Wavelength in meters

        Returns
        -------
        float
            Far-field distance in meters (F = 1 boundary)
        """
        return aperture_size**2 / wavelength

    @staticmethod
    def fresnel_zone_radius(n: int, wavelength: float, distance: float) -> float:
        """Radius of the nth Fresnel zone.

        r_n = √(nλz)

        Parameters
        ----------
        n : int
            Zone number (1, 2, 3, ...)
        wavelength : float
            Wavelength in meters
        distance : float
            Distance from aperture to observation in meters

        Returns
        -------
        float
            Radius of nth Fresnel zone in meters
        """
        return float(np.sqrt(n * wavelength * distance))

    @staticmethod
    def fresnel_zone_radii(
        n_zones: int, wavelength: float, distance: float
    ) -> NDArray[np.floating]:
        """Calculate radii for multiple Fresnel zones.

        Parameters
        ----------
        n_zones : int
            Number of zones to calculate
        wavelength : float
            Wavelength in meters
        distance : float
            Distance in meters

        Returns
        -------
        ndarray
            Array of zone radii in meters
        """
        n = np.arange(1, n_zones + 1)
        return cast(NDArray[np.floating], np.sqrt(n * wavelength * distance))

    @staticmethod
    def recommended_propagator(fresnel_number: float) -> str:
        """Recommend propagator based on Fresnel number.

        Parameters
        ----------
        fresnel_number : float
            Fresnel number F = a²/(λz)

        Returns
        -------
        str
            Recommended propagator: 'fraunhofer' or 'angular_spectrum'
        """
        if fresnel_number < 0.1:
            return "fraunhofer"
        else:
            return "angular_spectrum"


class GSDBaseline:
    """Ground Sampling Distance calculations for aerial/drone imaging.

    Provides GSD calculations for drone and aerial camera systems.

    Examples
    --------
    >>> # DJI Phantom at 100m altitude
    >>> gsd = GSDBaseline.gsd(
    ...     altitude=100, pixel_pitch=2.4e-6, focal_length=8.8e-3
    ... )
    >>> print(f"GSD: {gsd * 100:.2f} cm/pixel")
    GSD: 2.73 cm/pixel
    """

    @staticmethod
    def gsd(altitude: float, pixel_pitch: float, focal_length: float) -> float:
        """Ground sampling distance calculation.

        GSD = H × p / f

        Parameters
        ----------
        altitude : float
            Height above ground in meters
        pixel_pitch : float
            Sensor pixel size in meters
        focal_length : float
            Lens focal length in meters

        Returns
        -------
        float
            GSD in meters per pixel
        """
        return altitude * pixel_pitch / focal_length

    @staticmethod
    def altitude_for_gsd(target_gsd: float, pixel_pitch: float, focal_length: float) -> float:
        """Calculate required altitude for target GSD.

        H = GSD × f / p

        Parameters
        ----------
        target_gsd : float
            Desired GSD in meters per pixel
        pixel_pitch : float
            Sensor pixel size in meters
        focal_length : float
            Lens focal length in meters

        Returns
        -------
        float
            Required altitude in meters
        """
        return target_gsd * focal_length / pixel_pitch

    @staticmethod
    def swath_width(
        altitude: float,
        sensor_width: float,
        focal_length: float,
    ) -> float:
        """Calculate ground swath width.

        Swath = H × W_sensor / f

        Parameters
        ----------
        altitude : float
            Height above ground in meters
        sensor_width : float
            Sensor width in meters
        focal_length : float
            Lens focal length in meters

        Returns
        -------
        float
            Swath width in meters
        """
        return altitude * sensor_width / focal_length

    @staticmethod
    def coverage_area(
        altitude: float,
        sensor_width: float,
        sensor_height: float,
        focal_length: float,
    ) -> Tuple[float, float, float]:
        """Calculate ground coverage area.

        Parameters
        ----------
        altitude : float
            Height above ground in meters
        sensor_width : float
            Sensor width in meters
        sensor_height : float
            Sensor height in meters
        focal_length : float
            Lens focal length in meters

        Returns
        -------
        tuple
            (width, height, area) in meters and square meters
        """
        width = altitude * sensor_width / focal_length
        height = altitude * sensor_height / focal_length
        return width, height, width * height

    @staticmethod
    def diffraction_limited_gsd(
        altitude: float,
        wavelength: float,
        aperture_diameter: float,
    ) -> float:
        """Diffraction-limited GSD (theoretical minimum).

        GSD_min = 1.22 × λ × H / D

        Parameters
        ----------
        altitude : float
            Height above ground in meters
        wavelength : float
            Wavelength in meters
        aperture_diameter : float
            Lens aperture diameter in meters

        Returns
        -------
        float
            Diffraction-limited GSD in meters
        """
        return 1.22 * wavelength * altitude / aperture_diameter


@dataclass
class ValidationResult:
    """Result of comparing measured value to theoretical baseline.

    Attributes
    ----------
    measured : float
        Measured value
    theoretical : float
        Theoretical prediction
    error : float
        Absolute error
    error_percent : float
        Relative error as percentage
    tolerance_percent : float
        Acceptance threshold as percentage
    passed : bool
        Whether test passed
    status : str
        Human-readable status string
    """

    measured: float
    theoretical: float
    error: float
    error_percent: float
    tolerance_percent: float
    passed: bool
    status: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "measured": self.measured,
            "theoretical": self.theoretical,
            "error": self.error,
            "error_percent": self.error_percent,
            "tolerance_percent": self.tolerance_percent,
            "passed": self.passed,
            "status": self.status,
        }


def compare_to_theoretical(
    measured: float,
    theoretical: float,
    tolerance: float = 0.15,
) -> ValidationResult:
    """Compare measured value to theoretical limit.

    Parameters
    ----------
    measured : float
        Measured value
    theoretical : float
        Theoretical prediction
    tolerance : float, optional
        Acceptable relative error (default 0.15 = 15%)

    Returns
    -------
    ValidationResult
        Validation result with error analysis

    Examples
    --------
    >>> result = compare_to_theoretical(254e-9, 240e-9, tolerance=0.15)
    >>> print(result.status)
    PASS
    >>> print(f"Error: {result.error_percent:.1f}%")
    Error: 5.8%
    """
    error = abs(measured - theoretical)
    error_percent = (error / theoretical) * 100 if theoretical != 0 else float("inf")
    tolerance_percent = tolerance * 100
    passed = error_percent <= tolerance_percent

    return ValidationResult(
        measured=measured,
        theoretical=theoretical,
        error=error,
        error_percent=error_percent,
        tolerance_percent=tolerance_percent,
        passed=passed,
        status="PASS" if passed else "FAIL",
    )


def compute_l2_error(
    measured: NDArray[np.floating],
    theoretical: NDArray[np.floating],
    normalize: bool = True,
) -> float:
    """Compute L2 (Euclidean) error between arrays.

    Parameters
    ----------
    measured : ndarray
        Measured/simulated pattern
    theoretical : ndarray
        Analytical/theoretical pattern
    normalize : bool, optional
        If True, return relative error (default: True)

    Returns
    -------
    float
        L2 error (relative if normalize=True)
    """
    measured = np.asarray(measured).flatten()
    theoretical = np.asarray(theoretical).flatten()

    l2_error = np.linalg.norm(measured - theoretical)

    if normalize and np.linalg.norm(theoretical) > 0:
        return float(l2_error / np.linalg.norm(theoretical))
    return float(l2_error)


def compute_peak_position_error(
    measured: NDArray[np.floating],
    theoretical: NDArray[np.floating],
    coordinates: Optional[NDArray[np.floating]] = None,
) -> float:
    """Compute error in peak position between two patterns.

    Parameters
    ----------
    measured : ndarray
        Measured pattern (1D or 2D)
    theoretical : ndarray
        Theoretical pattern
    coordinates : ndarray, optional
        Coordinate array for computing physical position error.
        If None, returns error in pixels.

    Returns
    -------
    float
        Peak position error (in coordinate units or pixels)
    """
    measured = np.asarray(measured)
    theoretical = np.asarray(theoretical)

    # Find peak positions (indices)
    measured_peak_idx = np.unravel_index(np.argmax(measured), measured.shape)
    theoretical_peak_idx = np.unravel_index(np.argmax(theoretical), theoretical.shape)

    if coordinates is not None:
        coordinates = np.asarray(coordinates)
        if measured.ndim == 1:
            return float(
                abs(coordinates[measured_peak_idx[0]] - coordinates[theoretical_peak_idx[0]])
            )
        else:
            # For 2D, compute Euclidean distance
            measured_pos = np.array([coordinates[i] for i in measured_peak_idx])
            theoretical_pos = np.array([coordinates[i] for i in theoretical_peak_idx])
            return float(np.linalg.norm(measured_pos - theoretical_pos))
    else:
        # Return pixel distance
        return float(np.linalg.norm(np.array(measured_peak_idx) - np.array(theoretical_peak_idx)))

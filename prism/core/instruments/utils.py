"""
Utility functions for instrument propagator selection and validation.

This module provides helper functions for:
- Fresnel number calculations
- Automatic propagator selection based on instrument type
- Nyquist sampling validation for different optical systems

These utilities complement the propagator selection logic built into
individual instrument classes.
"""

from __future__ import annotations

from typing import Optional

from prism.config.constants import fresnel_number as calculate_fresnel_number
from prism.core.grid import Grid
from prism.core.propagators import (
    AngularSpectrumPropagator,
    FraunhoferPropagator,
    Propagator,
)


def select_propagator(
    instrument_type: str,
    grid: Grid,
    distance: Optional[float] = None,
    aperture_size: Optional[float] = None,
    wavelength: Optional[float] = None,
) -> Propagator:
    """
    Automatically select optimal propagator for an instrument.

    This function provides instrument-specific defaults for propagator
    selection, using the Fresnel number to determine the appropriate
    diffraction regime.

    Parameters
    ----------
    instrument_type : str
        Type of instrument: 'telescope', 'microscope', or 'camera'
    grid : Grid
        Computational grid for the instrument
    distance : float, optional
        Propagation distance in meters (required for cameras)
    aperture_size : float, optional
        Aperture diameter in meters (required for Fresnel number calculation)
    wavelength : float, optional
        Wavelength in meters (required for Fresnel number calculation)

    Returns
    -------
    Propagator
        Appropriate Propagator instance

    Raises
    ------
    ValueError
        If instrument_type is unknown or required parameters are missing

    Notes
    -----
    Selection Logic:
        - Telescope: Always FraunhoferPropagator (far-field, F ~ 10^-12)
        - Microscope: Always AngularSpectrumPropagator (near-field, high accuracy)
        - Camera: Based on Fresnel number:
            * F < 0.1: FraunhoferPropagator (far-field)
            * 0.1 ≤ F ≤ 10: FresnelPropagator (intermediate)
            * F > 10: AngularSpectrumPropagator (near-field)

    Where Fresnel number F = a²/(λz), with:
        - a = aperture radius
        - λ = wavelength
        - z = propagation distance

    Examples
    --------
    >>> from prism.core.grid import Grid
    >>> from prism.core.instruments.utils import select_propagator
    >>>
    >>> # Telescope propagator (always far-field)
    >>> grid = Grid(nx=512, dx=10e-6, wavelength=550e-9)
    >>> prop = select_propagator('telescope', grid)
    >>> print(type(prop).__name__)
    'FraunhoferPropagator'
    >>>
    >>> # Camera with automatic selection
    >>> prop = select_propagator(
    ...     'camera',
    ...     grid,
    ...     distance=2.0,
    ...     aperture_size=0.05,
    ...     wavelength=550e-9
    ... )

    See Also
    --------
    spids.core.propagators.select_propagator : General propagator selection
    calculate_fresnel_number : Fresnel number calculation
    """
    # Instrument-specific defaults
    if instrument_type == "telescope":
        # Always use Fraunhofer for astronomical imaging (far-field)
        return FraunhoferPropagator(normalize=True)

    elif instrument_type == "microscope":
        # Always use Angular Spectrum for accuracy in near-field
        return AngularSpectrumPropagator(grid=grid)

    elif instrument_type == "camera":
        # Select based on Fresnel number
        if distance is not None and aperture_size is not None and wavelength is not None:
            fresnel_num = calculate_fresnel_number(aperture_size, distance, wavelength)

            if fresnel_num < 0.1:
                # Far-field regime
                return FraunhoferPropagator(normalize=True)
            elif fresnel_num > 10:
                # Near-field regime
                return AngularSpectrumPropagator(grid=grid)
            else:
                # Fresnel (intermediate) regime
                # Note: FresnelPropagator requires additional parameters
                # For safety, default to AngularSpectrumPropagator which works for all fresnel_num
                return AngularSpectrumPropagator(grid=grid)
        else:
            # Default to Angular Spectrum for general use (works for all distances)
            return AngularSpectrumPropagator(grid=grid)

    else:
        raise ValueError(
            f"Unknown instrument type: '{instrument_type}'. "
            f"Valid options: 'telescope', 'microscope', 'camera'"
        )


def validate_sampling(
    instrument_type: str,
    pixel_size: float,
    wavelength: float,
    numerical_aperture: Optional[float] = None,
    f_number: Optional[float] = None,
    magnification: Optional[float] = None,
) -> bool:
    """
    Validate that sampling meets Nyquist criteria for an instrument.

    This function checks whether the pixel sampling is adequate to resolve
    the diffraction-limited features of the optical system. Undersampling
    leads to aliasing and loss of resolution.

    Parameters
    ----------
    instrument_type : str
        Type of instrument: 'microscope', 'camera', or 'telescope'
    pixel_size : float
        Pixel size in meters (for microscope, this is object-space pixel size)
    wavelength : float
        Wavelength in meters
    numerical_aperture : float, optional
        Numerical aperture (required for microscopes)
    f_number : float, optional
        F-number (f/#) of lens (required for cameras)
    magnification : float, optional
        Magnification factor (for converting sensor pixel to object pixel)

    Returns
    -------
    bool
        True if sampling is adequate (meets Nyquist), False otherwise

    Raises
    ------
    ValueError
        If required parameters for the instrument type are missing

    Notes
    -----
    Nyquist Sampling Requirements:
        - Microscope: pixel_size ≤ λ/(4×NA)
            Requires at least 2 samples per resolution element (0.61λ/NA)
        - Camera: pixel_size ≤ 1.22×λ×f/#
            Requires at least 2 pixels per Airy disk diameter (2.44λf/#)
        - Telescope: Similar to camera, based on angular resolution

    The factor of 2 comes from the Nyquist-Shannon sampling theorem,
    which requires at least 2 samples per period of the highest spatial
    frequency component.

    Examples
    --------
    >>> # Validate microscope sampling
    >>> is_valid = validate_sampling(
    ...     'microscope',
    ...     pixel_size=50e-9,  # 50 nm object pixel
    ...     wavelength=532e-9,
    ...     numerical_aperture=1.4
    ... )
    >>> print(is_valid)
    True
    >>>
    >>> # Check camera sampling
    >>> is_valid = validate_sampling(
    ...     'camera',
    ...     pixel_size=4.3e-6,  # 4.3 μm sensor pixel
    ...     wavelength=550e-9,
    ...     f_number=2.8
    ... )

    See Also
    --------
    spids.core.instruments.Microscope : Microscope implementation with validation
    spids.core.instruments.Camera : Camera implementation
    """
    if instrument_type == "microscope":
        if numerical_aperture is None:
            raise ValueError("Numerical aperture (NA) required for microscope validation")

        # For microscopes, need at least 2 samples per Abbe resolution element
        # Abbe limit: 0.61 λ/NA
        # Nyquist: pixel_size ≤ (0.61 λ/NA) / 2 = 0.305 λ/NA ≈ λ/(4×NA)
        min_sampling = wavelength / (4 * numerical_aperture)

        # If magnification provided, convert sensor pixel to object pixel
        if magnification is not None:
            object_pixel = pixel_size / magnification
        else:
            object_pixel = pixel_size

        return object_pixel <= min_sampling

    elif instrument_type == "camera":
        if f_number is None:
            raise ValueError("F-number (f/#) required for camera validation")

        # Airy disk diameter: 2.44 × λ × f/#
        # Need at least 2 pixels per Airy disk
        airy_diameter = 2.44 * wavelength * f_number
        return pixel_size <= airy_diameter / 2

    elif instrument_type == "telescope":
        # Telescope validation similar to camera
        # Based on angular resolution at focal plane
        if f_number is None:
            raise ValueError("F-number (f/#) required for telescope validation")

        # Same criterion as camera: 2 pixels per Airy disk
        airy_diameter = 2.44 * wavelength * f_number
        return pixel_size <= airy_diameter / 2

    else:
        raise ValueError(
            f"Unknown instrument type: '{instrument_type}'. "
            f"Valid options: 'microscope', 'camera', 'telescope'"
        )


def get_resolution_limit(
    instrument_type: str,
    wavelength: float,
    numerical_aperture: Optional[float] = None,
    aperture_diameter: Optional[float] = None,
    f_number: Optional[float] = None,
) -> float:
    """
    Calculate theoretical resolution limit for an instrument.

    Computes the diffraction-limited resolution based on the optical
    configuration and instrument type.

    Parameters
    ----------
    instrument_type : str
        Type of instrument: 'microscope', 'camera', or 'telescope'
    wavelength : float
        Wavelength in meters
    numerical_aperture : float, optional
        Numerical aperture (required for microscopes)
    aperture_diameter : float, optional
        Aperture diameter in meters (required for telescopes)
    f_number : float, optional
        F-number (required for cameras)

    Returns
    -------
    float
        Resolution limit in meters (for microscope/camera) or radians (for telescope)

    Raises
    ------
    ValueError
        If required parameters are missing

    Notes
    -----
    Resolution Criteria:
        - Microscope: Abbe diffraction limit = 0.61 λ/NA (lateral resolution)
        - Telescope: Rayleigh criterion = 1.22 λ/D (angular resolution in radians)
        - Camera: Airy disk size = 2.44 λ × f/# (at sensor plane)

    Examples
    --------
    >>> # Microscope resolution
    >>> res = get_resolution_limit(
    ...     'microscope',
    ...     wavelength=532e-9,
    ...     numerical_aperture=1.4
    ... )
    >>> print(f"Resolution: {res*1e9:.1f} nm")
    Resolution: 231.7 nm
    >>>
    >>> # Telescope angular resolution
    >>> res = get_resolution_limit(
    ...     'telescope',
    ...     wavelength=550e-9,
    ...     aperture_diameter=8.2
    ... )
    >>> print(f"Angular resolution: {res*1e9:.2f} nanoradians")
    Angular resolution: 81.83 nanoradians

    See Also
    --------
    validate_sampling : Check if sampling meets Nyquist criterion
    """
    if instrument_type == "microscope":
        if numerical_aperture is None:
            raise ValueError("Numerical aperture required for microscope resolution")
        # Abbe diffraction limit
        return 0.61 * wavelength / numerical_aperture

    elif instrument_type == "telescope":
        if aperture_diameter is None:
            raise ValueError("Aperture diameter required for telescope resolution")
        # Rayleigh criterion (angular resolution in radians)
        return 1.22 * wavelength / aperture_diameter

    elif instrument_type == "camera":
        if f_number is None:
            raise ValueError("F-number required for camera resolution")
        # Diffraction-limited spot size at sensor
        return 2.44 * wavelength * f_number

    else:
        raise ValueError(
            f"Unknown instrument type: '{instrument_type}'. "
            f"Valid options: 'microscope', 'camera', 'telescope'"
        )


__all__ = [
    "select_propagator",
    "validate_sampling",
    "get_resolution_limit",
]

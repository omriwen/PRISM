"""
Module: spids.config.constants
Purpose: Physical constants and optical criteria for astronomical imaging
Dependencies: numpy
Main Functions:
    - fresnel_number(width, distance, wavelength): Fresnel number calculation
    - is_fraunhofer(width, distance, wavelength): Check Fraunhofer diffraction regime validity
    - is_fresnel(width, distance, wavelength): Check Fresnel diffraction regime validity
    - r_coh(width, distance, wavelength): Coherence radius calculation

Description:
    This module defines fundamental physical constants (speed of light, astronomical units),
    and optical criteria functions (Fresnel number, coherence radius) used in SPIDS for
    realistic astronomical imaging simulations.
"""

from __future__ import annotations

from numpy import pi


# %% Physical Constants

# Length scales
ly = 9.461e15  # light year in meters
pc = 3.086e16  # parsec in meters
au = 1.496e11  # astronomical unit in meters
nm = 1e-9  # nanometer in meters
um = 1e-6  # micrometer in meters
mm = 1e-3  # millimeter in meters
cm = 1e-2  # centimeter in meters
km = 1e3  # kilometer in meters

# Physical constants
c = 3e8  # speed of light in m/s
solar_radius = 6.957e8  # solar radius in meters


# %% Optical Criteria


def fresnel_number(width: float, distance: float, wavelength: float) -> float:
    """
    Calculate Fresnel number.

    Args:
        width: Aperture width (m)
        distance: Distance from aperture to observation plane (m)
        wavelength: Wavelength (m)

    Returns:
        Fresnel number (dimensionless)
    """
    return width**2 / (wavelength * distance)


def fresnel_number_critical(width: float, distance: float, wavelength: float) -> float:
    """Critical Fresnel number threshold.

    Parameters
    ----------
    width : float
        Aperture width in meters
    distance : float
        Propagation distance in meters
    wavelength : float
        Wavelength in meters

    Returns
    -------
    float
        Fresnel number
    """
    return fresnel_number(width, distance, wavelength)


def is_fraunhofer(width: float, distance: float, wavelength: float) -> bool:
    """
    Check if Fraunhofer diffraction approximation is valid.

    Args:
        width: Aperture width (m)
        distance: Distance from aperture to observation plane (m)
        wavelength: Wavelength (m)

    Returns:
        True if Fraunhofer approximation is valid (F < 0.1)
    """
    return fresnel_number(width, distance, wavelength) < 0.1


def is_fresnel(width: float, distance: float, wavelength: float) -> bool:
    """
    Check if Fresnel diffraction approximation is valid.

    Args:
        width: Aperture width (m)
        distance: Distance from aperture to observation plane (m)
        wavelength: Wavelength (m)

    Returns:
        True if Fresnel approximation is valid (fresnel_number_critical < 0.1)
    """
    return fresnel_number_critical(width, distance, wavelength) < 0.1


def r_coh(width: float, distance: float, wavelength: float) -> float:
    """
    Calculate coherence radius.

    Args:
        width: Source width (m)
        distance: Distance from source to observation plane (m)
        wavelength: Wavelength (m)

    Returns:
        Coherence radius (m)
    """
    return float(wavelength * distance / (pi * width))

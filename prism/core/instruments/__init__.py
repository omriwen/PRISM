"""Optical instruments module for SPIDS.

This module provides abstractions for different optical instruments:
- Telescope: Far-field astronomical imaging
- Microscope: Near-field high-NA imaging with flexible illumination modes
- Camera: General purpose imaging

Key Features:
- Dynamic illumination modes (brightfield, darkfield, phase, DIC)
- Scanning illumination mode for Fourier Ptychographic Microscopy (FPM)
- Support for all instruments to switch between illumination modes
- Physically accurate pupil functions for different imaging modes
- Factory function for automatic instrument creation
- Unified interface across all optical systems
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

# Re-export illumination source types for convenience
# (primary exports are in spids.core.optics)
from ..optics.illumination import IlluminationSource, IlluminationSourceType
from .base import Instrument, InstrumentConfig
from .camera import Camera, CameraConfig
from .four_f_base import FourFSystem
from .microscope import Microscope, MicroscopeConfig
from .telescope import Telescope, TelescopeConfig


# TYPE_CHECKING is kept for backward compatibility with old telescope imports
if TYPE_CHECKING:
    pass  # Legacy imports removed, using new unified telescope


def create_instrument(
    config: Union[InstrumentConfig, TelescopeConfig, MicroscopeConfig, CameraConfig],
) -> Instrument:
    """Factory function to create appropriate instrument from configuration.

    This function automatically selects the correct instrument type based on
    the configuration class provided.

    Parameters
    ----------
    config : TelescopeConfig, MicroscopeConfig, or CameraConfig
        Configuration for the instrument. The type of configuration determines
        which instrument class will be instantiated.

    Returns
    -------
    Instrument
        Instantiated instrument (Telescope, Microscope, or Camera)

    Raises
    ------
    TypeError
        If config is not a recognized instrument configuration type
    ValueError
        If configuration validation fails

    Examples
    --------
    Create a telescope:

    >>> from prism.core.instruments import TelescopeConfig, create_instrument
    >>> config = TelescopeConfig(
    ...     n_pixels=512,
    ...     aperture_radius_pixels=50,
    ...     aperture_diameter=8.2,  # 8.2m VLT
    ...     wavelength=550e-9
    ... )
    >>> telescope = create_instrument(config)
    >>> print(telescope.get_instrument_type())
    telescope

    Create a microscope:

    >>> from prism.core.instruments import MicroscopeConfig, create_instrument
    >>> config = MicroscopeConfig(
    ...     numerical_aperture=1.4,
    ...     magnification=100,
    ...     wavelength=532e-9
    ... )
    >>> microscope = create_instrument(config)
    >>> print(microscope.get_instrument_type())
    microscope

    Create a camera:

    >>> from prism.core.instruments import CameraConfig, create_instrument
    >>> config = CameraConfig(
    ...     focal_length=50e-3,
    ...     f_number=1.4,
    ...     object_distance=2.0
    ... )
    >>> camera = create_instrument(config)
    >>> print(camera.get_instrument_type())
    camera

    Notes
    -----
    The factory function provides a unified interface for instrument creation,
    making it easier to switch between different instrument types in experiments.
    """
    # Validate configuration first
    config.validate()

    # Select instrument based on config type
    if isinstance(config, TelescopeConfig):
        return Telescope(config=config)
    elif isinstance(config, MicroscopeConfig):
        return Microscope(config)
    elif isinstance(config, CameraConfig):
        return Camera(config)
    elif isinstance(config, InstrumentConfig):
        # Base config without specific instrument type
        raise TypeError(
            "Cannot create instrument from base InstrumentConfig. "
            "Please use a specific configuration: TelescopeConfig, MicroscopeConfig, or CameraConfig"
        )
    else:
        raise TypeError(
            f"Unknown instrument configuration type: {type(config).__name__}. "
            "Expected TelescopeConfig, MicroscopeConfig, or CameraConfig"
        )


__all__ = [
    "Instrument",
    "InstrumentConfig",
    "FourFSystem",
    "Telescope",
    "TelescopeConfig",
    "Microscope",
    "MicroscopeConfig",
    "Camera",
    "CameraConfig",
    "create_instrument",
    # Illumination source types (scanning illumination mode)
    "IlluminationSource",
    "IlluminationSourceType",
]

# Note: The factory function create_instrument() supports all three instrument
# types (Telescope, Microscope, Camera).

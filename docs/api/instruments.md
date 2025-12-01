# prism.core.instruments

Optical instruments module for PRISM.

This module provides abstractions for different optical instruments:
- Telescope: Far-field astronomical imaging
- Microscope: Near-field high-NA imaging with flexible illumination modes
- Camera: General purpose imaging

Key Features:
- Dynamic illumination modes (brightfield, darkfield, phase, DIC)
- Support for all instruments to switch between illumination modes
- Physically accurate pupil functions for different imaging modes
- Factory function for automatic instrument creation
- Unified interface across all optical systems

## Classes

## Functions

### create_instrument

```python
create_instrument(config: Union[prism.core.instruments.base.InstrumentConfig, ForwardRef('TelescopeConfig'), prism.core.instruments.microscope.MicroscopeConfig, prism.core.instruments.camera.CameraConfig]) -> prism.core.instruments.base.Instrument
```

Factory function to create appropriate instrument from configuration.

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

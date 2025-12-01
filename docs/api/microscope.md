# prism.core.instruments.microscope

Microscope implementation for PRISM.

This module provides microscope simulation with support for various illumination
modes including brightfield, darkfield, phase contrast, and DIC.

## Classes

### Microscope

```python
Microscope(config: prism.core.instruments.microscope.MicroscopeConfig)
```

Microscope implementation for near-field imaging.

Supports various illumination modes:
- Brightfield: Direct transmitted/reflected light
- Darkfield: Only scattered light (direct light blocked)
- Phase contrast: Phase shifts converted to intensity
- DIC: Differential interference contrast
- Custom: User-defined illumination/detection pupils

#### Methods

##### `__init__`

Initialize microscope with configuration.

Args:
    config: Microscope configuration

##### `compute_psf`

Compute 2D or 3D PSF for microscope.

Args:
    z_slices: Number of z-slices for 3D PSF (None for 2D)
    illumination_mode: Type of illumination
    illumination_params: Additional illumination parameters

Returns:
    2D or 3D PSF tensor, normalized to max=1

##### `forward`

Forward imaging through microscope.

Args:
    field: Complex field or intensity at object plane (sample)
    illumination_mode: Type of illumination (None uses 'brightfield')
    illumination_params: Parameters for illumination
    add_noise: Add realistic detector noise
    **kwargs: Additional parameters

Returns:
    Image at detector plane (intensity)

##### `get_info`

Get microscope information summary.

Returns:
    Dictionary with microscope parameters

##### `get_instrument_type`

Return instrument type identifier.

Returns:
    Lowercase instrument class name

##### `validate_field`

Validate and prepare input field.

Args:
    field: Input field tensor

Returns:
    Validated field tensor

Raises:
    ValueError: If field shape doesn't match grid

### MicroscopeConfig

```python
MicroscopeConfig(wavelength: float = 5.5e-07, n_pixels: int = 1024, pixel_size: float = 6.5e-06, grid_size: Optional[float] = None, numerical_aperture: float = 0.9, magnification: float = 40.0, medium_index: float = 1.0, tube_lens_focal: float = 0.2, working_distance: Optional[float] = None, default_illumination_na: Optional[float] = None) -> None
```

Configuration for microscope systems.

Attributes:
    numerical_aperture: Objective NA
    magnification: Total magnification
    medium_index: Immersion medium refractive index (1.0 for air, 1.33 for water, 1.515 for oil)
    tube_lens_focal: Tube lens focal length in meters
    working_distance: Working distance in meters
    default_illumination_na: Default NA for illumination (if None, uses detection NA)

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

##### `validate`

Validate configuration parameters.

# prism.core.instruments.camera

Camera instrument implementation for PRISM.

This module implements general camera systems with:
- Automatic propagation regime selection (Fraunhofer vs Angular Spectrum)
- Depth of field calculations
- Sensor noise modeling
- Defocus aberration support

## Classes

### Camera

```python
Camera(config: prism.core.instruments.camera.CameraConfig)
```

General camera implementation.

Supports both near-field and far-field imaging with automatic
propagation regime selection based on Fresnel number.

#### Methods

##### `__init__`

Initialize camera.

Args:
    config: Camera configuration

##### `calculate_depth_of_field`

Calculate depth of field.

Args:
    coc_limit: Circle of confusion limit in meters (default 30μm)

Returns:
    Tuple of (near_distance, far_distance) in meters

##### `calculate_image_distance`

Calculate image distance using thin lens equation.

Uses 1/f = 1/do + 1/di

Returns:
    Image distance in meters

##### `calculate_magnification`

Calculate lateral magnification.

Returns:
    Magnification (negative for inverted image, 0 for infinity focus)

##### `compute_psf`

Compute camera PSF with optional defocus.

Args:
    defocus: Defocus distance in meters
    **kwargs: Additional parameters (unused)

Returns:
    2D PSF tensor, normalized to peak intensity of 1

##### `forward`

Image formation through camera.

Args:
    field: Input scene (can be at infinity or finite distance)
    illumination_mode: Illumination type (unused for basic camera)
    illumination_params: Illumination parameters (unused for basic camera)
    **kwargs: Additional parameters:
        add_noise (bool): Add realistic sensor noise (default: False)

Returns:
    Image at sensor plane

##### `get_info`

Get camera information summary.

Returns:
    Dictionary with camera parameters and characteristics

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

### CameraConfig

```python
CameraConfig(wavelength: float = 5.5e-07, n_pixels: int = 1024, pixel_size: float = 6.5e-06, grid_size: Optional[float] = None, focal_length: float = 0.05, f_number: float = 2.8, sensor_size: Tuple[float, float] = (0.036, 0.024), object_distance: float = inf, focus_distance: Optional[float] = None, lens_type: str = 'thin') -> None
```

Configuration for camera systems.

Attributes:
    focal_length: Lens focal length in meters
    f_number: f-number (focal ratio) of the lens
    sensor_size: (width, height) of sensor in meters
    object_distance: Distance to object in meters (inf for far field)
    focus_distance: Focus distance in meters (None = same as object_distance)
    lens_type: Type of lens model ('thin', 'thick', 'compound')

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

##### `validate`

Validate configuration parameters.

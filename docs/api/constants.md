# prism.config.constants

Module: prism.config.constants
Purpose: Physical constants and optical criteria for astronomical imaging
Dependencies: numpy
Main Functions:
    - fresnel_number(width, distance, wavelength): Fresnel number calculation
    - fresnel_number_critical(width, distance, wavelength): Critical Fresnel number threshold
    - is_fraunhofer(width, distance, wavelength): Check Fraunhofer diffraction regime validity
    - is_fresnel(width, distance, wavelength): Check Fresnel diffraction regime validity
    - r_coh(width, distance, wavelength): Coherence radius calculation

Description:
    This module defines fundamental physical constants (speed of light, astronomical units),
    and optical criteria functions (Fresnel number, coherence radius) used in PRISM for
    realistic astronomical imaging simulations.

## Functions

### fresnel_number_critical

```python
fresnel_number_critical(width: float, distance: float, wavelength: float) -> float
```

Critical Fresnel number threshold.

Parameters:
    width: Aperture width in meters
    distance: Propagation distance in meters
    wavelength: Wavelength in meters

Returns:
    Fresnel number

### fresnel_number

```python
fresnel_number(width: float, distance: float, wavelength: float) -> float
```

Calculate Fresnel number.

Args:
    width: Aperture width (m)
    distance: Distance from aperture to observation plane (m)
    wavelength: Wavelength (m)

Returns:
    Fresnel number (dimensionless)

### is_fraunhofer

```python
is_fraunhofer(width: float, distance: float, wavelength: float) -> bool
```

Check if Fraunhofer diffraction approximation is valid.

Args:
    width: Aperture width (m)
    distance: Distance from aperture to observation plane (m)
    wavelength: Wavelength (m)

Returns:
    True if Fraunhofer approximation is valid (F < 0.1)

### is_fresnel

```python
is_fresnel(width: float, distance: float, wavelength: float) -> bool
```

Check if Fresnel diffraction approximation is valid.

Args:
    width: Aperture width (m)
    distance: Distance from aperture to observation plane (m)
    wavelength: Wavelength (m)

Returns:
    True if Fresnel approximation is valid (fresnel_number_critical < 0.1)

### r_coh

```python
r_coh(width: float, distance: float, wavelength: float) -> float
```

Calculate coherence radius.

Args:
    width: Source width (m)
    distance: Distance from source to observation plane (m)
    wavelength: Wavelength (m)

Returns:
    Coherence radius (m)

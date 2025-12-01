# prism.config.objects

Module: prism.config.objects
Purpose: Predefined astronomical object parameters
Dependencies: prism.config.constants
Main Functions:
    - get_obj_params(args): Retrieve physical parameters for predefined astronomical objects

Description:
    This module provides parameters for predefined celestial objects (Europa, Titan,
    Betelgeuse, Neptune) including realistic physical parameters such as diameter,
    distance from observer, wavelength, and automatically calculated aperture sizes
    based on coherence requirements.

## Classes

## Functions

### get_obj_params

```python
get_obj_params(args: Any) -> Any
```

Retrieve and apply parameters for predefined astronomical objects.

Args:
    args: Argument namespace containing obj_name and dxf attributes

Returns:
    Updated args with object-specific parameters applied

Raises:
    ValueError: If obj_name is not recognized

Notes:
    - Sets obj_diameter, obj_distance, input, wavelength from predefined values
    - Calculates sample_diameter based on coherence radius
    - Only updates args attributes that are None

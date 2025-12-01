"""
Module: spids.config.objects
Purpose: Predefined astronomical object parameters
Dependencies: spids.config.constants
Main Functions:
    - get_obj_params(args): Retrieve physical parameters for predefined astronomical objects

Description:
    This module provides parameters for predefined celestial objects (Europa, Titan,
    Betelgeuse, Neptune) including realistic physical parameters such as diameter,
    distance from observer, wavelength, and automatically calculated aperture sizes
    based on coherence requirements.
"""

from __future__ import annotations

from typing import Any

from .constants import au, km, ly, nm, r_coh, um


# %% Predefined Objects

PREDEFINED_OBJECTS = {
    "titan": {
        "obj_diameter": 5150 * km,
        "obj_distance": 1.2e9 * km,
        "input": "data/titan_temp.jpg",
        "wavelength": 2.0 * um,
    },
    "betelgeuse": {
        "obj_diameter": 2 * 617e6 * km,
        "obj_distance": 700 * ly,
        "input": "data/titan_temp.jpg",
        "wavelength": 500 * nm,  # Default wavelength
    },
    "neptune": {
        "obj_diameter": 50e3 * km,
        "obj_distance": 29.1 * au,
        "input": "data/titan_temp.jpg",
        "wavelength": 500 * nm,  # Default wavelength
    },
    "europa": {
        "obj_diameter": 3138 * km,
        "obj_distance": 628.3e6 * km,
        "input": "data/europa.jpg",
        "wavelength": 698.9 * nm,
    },
}


def get_obj_params(args: Any) -> Any:
    """
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
    """
    if args.obj_name is None:
        return args

    if args.obj_name not in PREDEFINED_OBJECTS:
        raise ValueError(
            f"Invalid object name: {args.obj_name}. "
            f"Valid options: {list(PREDEFINED_OBJECTS.keys())}"
        )

    params = PREDEFINED_OBJECTS[args.obj_name].copy()

    # Calculate sample diameter based on coherence radius
    W: float = params["obj_diameter"]  # type: ignore[assignment]  # noqa: N806
    L: float = params["obj_distance"]  # type: ignore[assignment]  # noqa: N806
    wavelength: float = params["wavelength"]  # type: ignore[assignment]
    sample_radius = r_coh(W / 2, L, wavelength)
    params["sample_diameter"] = int(2 * sample_radius / args.dxf)

    # Apply parameters to args (only if not already set)
    for key, value in params.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    return args

"""Configuration management for SPIDS."""

# Physical constants and units
# Configuration dataclasses

from __future__ import annotations

from .base import (
    ImageConfig,
    ModelConfig,
    MoPIEConfig,
    PhysicsConfig,
    PointSourceConfig,
    PRISMConfig,
    TelescopeConfig,
    TrainingConfig,
)
from .constants import (
    au,
    c,
    cm,
    fresnel_number,
    fresnel_number_critical,
    is_fraunhofer,
    is_fresnel,
    km,
    ly,
    mm,
    nm,
    pc,
    r_coh,
    solar_radius,
    um,
)

# Configuration loader/saver
from .loader import (
    args_to_config,
    config_to_args,
    load_config,
    merge_config_with_args,
    save_config,
)

# Astronomical objects
from .objects import (
    PREDEFINED_OBJECTS,
    get_obj_params,
)


__all__ = [
    # Length scales
    "ly",
    "pc",
    "au",
    "nm",
    "um",
    "mm",
    "cm",
    "km",
    # Physical constants
    "c",
    "solar_radius",
    # Optical criteria
    "fresnel_number",
    "fresnel_number_critical",
    "is_fraunhofer",
    "is_fresnel",
    "r_coh",
    # Astronomical objects
    "PREDEFINED_OBJECTS",
    "get_obj_params",
    # Configuration dataclasses
    "PRISMConfig",
    "ImageConfig",
    "TelescopeConfig",
    "ModelConfig",
    "TrainingConfig",
    "PhysicsConfig",
    "PointSourceConfig",
    "MoPIEConfig",
    # Configuration loader/saver
    "load_config",
    "save_config",
    "config_to_args",
    "args_to_config",
    "merge_config_with_args",
]

"""Core functionality for PRISM.

This module exports the primary API for PRISM core functionality.

Unified Instruments API
-----------------------
The unified instruments API provides clean, config-based instrument classes:

>>> from prism.core import Telescope, TelescopeConfig, MeasurementSystem
>>> config = TelescopeConfig(n_pixels=512, aperture_radius_pixels=25)
>>> telescope = Telescope(config)
>>> measurement_system = MeasurementSystem(telescope)

For progressive measurements (SPIDS algorithm), wrap any Instrument in
a MeasurementSystem which handles cumulative aperture tracking.
"""

from __future__ import annotations

# Core infrastructure
from prism.core.grid import Grid

# Illumination sources
from prism.core.illumination import (
    ContrastMode,
    IlluminationConfig,
    IlluminationSource,
    LaserSource,
    LEDSource,
    SolarSource,
    SourceGeometry,
    create_illumination_source,
    validate_bf_df_configuration,
)

# Unified Instruments API (Primary - Recommended for new code)
from prism.core.instruments import Telescope, TelescopeConfig
from prism.core.measurement_system import MeasurementSystem, MeasurementSystemConfig

# Pattern generation
from prism.core.patterns import (
    create_pattern,
    create_patterns,
    generate_fermat_spiral,
    generate_samples,
    generate_star_pattern,
    points_energy,
    sort_by_energy,
    sort_by_radius,
)
from prism.core.propagators import FreeSpacePropagator

# Test targets
from prism.core.targets import (
    CheckerboardConfig,
    CheckerboardTarget,
    PointSourceConfig,
    PointSourceTarget,
    Target,
    TargetConfig,
    USAF1951Config,
    USAF1951Target,
    create_checkerboard_target,
    create_target,
    create_usaf_target,
)


__all__ = [
    # === Unified Instruments API ===
    "Telescope",
    "TelescopeConfig",
    "MeasurementSystem",
    "MeasurementSystemConfig",
    # === Core Infrastructure ===
    "Grid",
    "FreeSpacePropagator",
    # === Pattern Generation ===
    "create_pattern",
    "create_patterns",
    "generate_fermat_spiral",
    "generate_samples",
    "generate_star_pattern",
    "points_energy",
    "sort_by_energy",
    "sort_by_radius",
    # === Test Targets ===
    "Target",
    "TargetConfig",
    "USAF1951Target",
    "USAF1951Config",
    "CheckerboardTarget",
    "CheckerboardConfig",
    "PointSourceTarget",
    "PointSourceConfig",
    "create_target",
    "create_usaf_target",
    "create_checkerboard_target",
    # === Illumination Sources ===
    "ContrastMode",
    "IlluminationConfig",
    "IlluminationSource",
    "LEDSource",
    "LaserSource",
    "SolarSource",
    "SourceGeometry",
    "create_illumination_source",
    "validate_bf_df_configuration",
]

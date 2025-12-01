"""Scenario configuration system for SPIDS optical simulations.

This module provides user-friendly configuration for real-world optical systems:
- Microscopy (brightfield, darkfield, phase, DIC)
- Drone cameras (finite distance, consumer lenses, GSD)
- Satellite cameras (Earth observation, diffraction limits)

Quick Start
-----------

Using presets::

    from prism.scenarios import get_scenario_preset

    scenario = get_scenario_preset("microscope_100x_oil")
    print(f"Resolution: {scenario.lateral_resolution_nm:.0f} nm")

    # Convert to SPIDS instrument config
    instrument_config = scenario.to_instrument_config()

Using builders::

    from prism.scenarios import MicroscopeBuilder

    scenario = (MicroscopeBuilder()
        .objective("100x_1.4NA_oil")
        .illumination("phase")
        .wavelength_nm(532)
        .build())

Direct configuration::

    from prism.scenarios import MicroscopeScenarioConfig

    scenario = MicroscopeScenarioConfig(
        objective_spec="40x_0.9NA_air",
        illumination_mode="brightfield"
    )

Listing available presets::

    from prism.scenarios import list_scenario_presets, get_preset_description

    # List all presets
    all_presets = list_scenario_presets()

    # List by category
    microscope_presets = list_scenario_presets("microscope")
    drone_presets = list_scenario_presets("drone")

    # Get preset description
    desc = get_preset_description("microscope_100x_oil")
"""

from __future__ import annotations

from .base import ScenarioConfig
from .drone_camera import DroneBuilder, DroneScenarioConfig, LensSpec, SensorSpec
from .microscopy import MicroscopeBuilder, MicroscopeScenarioConfig, ObjectiveSpec
from .presets import (
    get_preset_description,
    get_presets_by_category,
    get_scenario_preset,
    list_scenario_presets,
    print_all_presets,
)


__version__ = "0.1.0"

__all__ = [
    # Base
    "ScenarioConfig",
    # Microscopy
    "MicroscopeScenarioConfig",
    "MicroscopeBuilder",
    "ObjectiveSpec",
    # Drone cameras
    "DroneScenarioConfig",
    "DroneBuilder",
    "LensSpec",
    "SensorSpec",
    # Presets
    "get_scenario_preset",
    "list_scenario_presets",
    "get_preset_description",
    "get_presets_by_category",
    "print_all_presets",
]

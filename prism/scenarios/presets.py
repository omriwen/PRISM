"""Preset configurations for common optical scenarios.

This module provides ready-to-use scenario presets for microscopy, drone cameras,
and satellite systems.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import ScenarioConfig
from .drone_camera import DroneScenarioConfig
from .microscopy import MicroscopeScenarioConfig


# Microscopy presets
MICROSCOPE_PRESETS: Dict[str, dict] = {
    "microscope_100x_oil": {
        "objective_spec": "100x_1.4NA_oil",
        "illumination_mode": "brightfield",
        "wavelength": 550e-9,
        "description": "High-resolution oil immersion for cellular detail (~200nm resolution)",
    },
    "microscope_100x_oil_phase": {
        "objective_spec": "100x_1.4NA_oil",
        "illumination_mode": "phase",
        "wavelength": 550e-9,
        "description": "Oil immersion phase contrast for unstained live cells",
    },
    "microscope_60x_water": {
        "objective_spec": "60x_1.2NA_water",
        "illumination_mode": "brightfield",
        "wavelength": 550e-9,
        "description": "Water immersion for live tissue imaging (~240nm resolution)",
    },
    "microscope_40x_air": {
        "objective_spec": "40x_0.9NA_air",
        "illumination_mode": "brightfield",
        "wavelength": 550e-9,
        "description": "High-NA air objective for detailed observation (~370nm resolution)",
    },
    "microscope_40x_phase": {
        "objective_spec": "40x_0.9NA_air",
        "illumination_mode": "phase",
        "wavelength": 550e-9,
        "description": "Phase contrast for unstained samples (~370nm resolution)",
    },
    "microscope_40x_dic": {
        "objective_spec": "40x_0.9NA_air",
        "illumination_mode": "dic",
        "wavelength": 550e-9,
        "description": "Differential interference contrast for 3D-like relief",
    },
    "microscope_20x_air": {
        "objective_spec": "20x_0.75NA_air",
        "illumination_mode": "brightfield",
        "wavelength": 550e-9,
        "description": "Medium magnification for tissue sections (~450nm resolution)",
    },
    "microscope_10x_air": {
        "objective_spec": "10x_0.3NA_air",
        "illumination_mode": "brightfield",
        "wavelength": 550e-9,
        "description": "Low magnification for sample overview (~1.1µm resolution)",
    },
    "microscope_fluorescence_100x": {
        "objective_spec": "100x_1.4NA_oil",
        "illumination_mode": "brightfield",  # Note: SPIDS doesn't have fluorescence mode yet
        "wavelength": 488e-9,  # GFP excitation
        "description": "Fluorescence microscopy with GFP filter (488nm)",
    },
}

# Drone camera presets
DRONE_PRESETS: Dict[str, dict] = {
    "drone_10m_inspection": {
        "lens_spec": "35mm_f2.8",
        "sensor_spec": "aps_c",
        "altitude_m": 10.0,
        "ground_speed_mps": 2.0,
        "description": "Close-range inspection (GSD ~3mm)",
    },
    "drone_20m_detail": {
        "lens_spec": "50mm_f4.0",
        "sensor_spec": "aps_c",
        "altitude_m": 20.0,
        "ground_speed_mps": 5.0,
        "description": "Detailed surveying (GSD ~2cm)",
    },
    "drone_50m_survey": {
        "lens_spec": "50mm_f4.0",
        "sensor_spec": "full_frame",
        "altitude_m": 50.0,
        "ground_speed_mps": 10.0,
        "description": "Site survey with full-frame camera (GSD ~6.5cm)",
    },
    "drone_100m_mapping": {
        "lens_spec": "35mm_f4.0",
        "sensor_spec": "full_frame",
        "altitude_m": 100.0,
        "ground_speed_mps": 15.0,
        "description": "Large area mapping (GSD ~18.5cm)",
    },
    "drone_phantom_120m": {
        "lens_spec": "24mm_f2.8",
        "sensor_spec": "1_2.3_inch",
        "altitude_m": 120.0,
        "ground_speed_mps": 12.0,
        "description": "DJI Phantom 4 equivalent at max legal altitude (GSD ~5cm)",
    },
    "drone_hover_50m": {
        "lens_spec": "50mm_f4.0",
        "sensor_spec": "full_frame",
        "altitude_m": 50.0,
        "ground_speed_mps": 0.0,
        "description": "Hover mode for sharp imaging (no motion blur)",
    },
    "drone_agriculture_50m": {
        "lens_spec": "35mm_f4.0",
        "sensor_spec": "aps_c",
        "altitude_m": 50.0,
        "ground_speed_mps": 8.0,
        "description": "Agricultural monitoring (GSD ~5.5cm)",
    },
    "drone_infrastructure_30m": {
        "lens_spec": "50mm_f2.8",
        "sensor_spec": "full_frame",
        "altitude_m": 30.0,
        "ground_speed_mps": 3.0,
        "description": "Infrastructure inspection (GSD ~4cm)",
    },
}


def get_scenario_preset(name: str) -> ScenarioConfig:
    """Get scenario preset by name.

    Args:
        name: Preset name (e.g., "microscope_100x_oil", "drone_50m_survey")

    Returns:
        Configured ScenarioConfig instance

    Raises:
        ValueError: If preset name not found

    Example:
        >>> scenario = get_scenario_preset("microscope_100x_oil")
        >>> print(f"Resolution: {scenario.lateral_resolution_nm:.0f} nm")
        >>> instrument_config = scenario.to_instrument_config()
    """
    scenario: ScenarioConfig

    if name in MICROSCOPE_PRESETS:
        config_dict = MICROSCOPE_PRESETS[name].copy()
        description = config_dict.pop("description", "")
        scenario = MicroscopeScenarioConfig(**config_dict)
        # Set description after creation
        object.__setattr__(scenario, "description", description)
        return scenario

    if name in DRONE_PRESETS:
        config_dict = DRONE_PRESETS[name].copy()
        description = config_dict.pop("description", "")
        scenario = DroneScenarioConfig(**config_dict)
        # Set description after creation
        object.__setattr__(scenario, "description", description)
        return scenario

    available = list_scenario_presets()
    raise ValueError(f"Unknown preset: '{name}'. Available presets: {available}")


def list_scenario_presets(category: Optional[str] = None) -> List[str]:
    """List available scenario presets.

    Args:
        category: Filter by category ('microscope', 'drone', 'satellite')
                 If None, returns all presets

    Returns:
        List of preset names

    Example:
        >>> all_presets = list_scenario_presets()
        >>> microscope_presets = list_scenario_presets("microscope")
        >>> drone_presets = list_scenario_presets("drone")
    """
    if category == "microscope":
        return sorted(list(MICROSCOPE_PRESETS.keys()))
    elif category == "drone":
        return sorted(list(DRONE_PRESETS.keys()))
    elif category is None:
        return sorted(list(MICROSCOPE_PRESETS.keys()) + list(DRONE_PRESETS.keys()))
    else:
        raise ValueError(f"Unknown category '{category}'. Use 'microscope', 'drone', or None")


def get_preset_description(name: str) -> str:
    """Get description for a preset.

    Args:
        name: Preset name

    Returns:
        Description string, or empty string if not found

    Example:
        >>> desc = get_preset_description("microscope_100x_oil")
        >>> print(desc)
    """
    all_presets = {**MICROSCOPE_PRESETS, **DRONE_PRESETS}
    preset = all_presets.get(name, {})
    description = preset.get("description", "")
    return str(description)


def get_presets_by_category() -> Dict[str, List[str]]:
    """Get all presets organized by category.

    Returns:
        Dictionary mapping category names to lists of preset names

    Example:
        >>> presets = get_presets_by_category()
        >>> for category, names in presets.items():
        ...     print(f"{category}: {len(names)} presets")
    """
    return {
        "microscope": list_scenario_presets("microscope"),
        "drone": list_scenario_presets("drone"),
    }


def print_all_presets() -> None:
    """Print all available presets with descriptions.

    This is a convenience function for interactive use.
    """
    print("\n" + "=" * 80)
    print("Available Scenario Presets")
    print("=" * 80)

    print("\n🔬 Microscopy:")
    print("-" * 80)
    for name in list_scenario_presets("microscope"):
        desc = get_preset_description(name)
        print(f"  {name:30s} - {desc}")

    print("\n🚁 Drone Cameras:")
    print("-" * 80)
    for name in list_scenario_presets("drone"):
        desc = get_preset_description(name)
        print(f"  {name:30s} - {desc}")

    print("\n" + "=" * 80)
    print(f"Total: {len(list_scenario_presets())} presets")
    print("=" * 80 + "\n")

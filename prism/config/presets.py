"""
Module: presets.py
Purpose: Built-in configuration presets for common SPIDS experiment patterns

This module provides pre-configured experiment templates that combine common
parameter patterns, reducing the need for verbose CLI commands or config files.

Usage:
    # Via CLI
    python main.py --preset quick_test --obj europa
    python main_mopie.py --preset mopie_baseline --obj titan

    # Via Python API
    from prism.config.presets import get_preset, list_presets
    preset_config = get_preset("production")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Type alias for preset dictionaries
PresetDict = Dict[str, Any]

# =============================================================================
# SPIDS Presets (for main.py - deep learning)
# =============================================================================

QUICK_TEST: PresetDict = {
    "name": None,
    "comment": "Quick test preset for fast iteration and debugging",
    "save_data": False,
    "telescope": {
        "n_samples": 64,
        "sample_length": 64,
        "samples_per_line_meas": 9,
        "fermat_sample": True,
    },
    "training": {
        "max_epochs": 1,
        "n_epochs": 1000,
        "loss_threshold": 0.001,
    },
}

PRODUCTION: PresetDict = {
    "name": None,
    "comment": "Production preset for high-quality reconstruction",
    "save_data": True,
    "telescope": {
        "n_samples": 200,
        "sample_length": 0,  # Point sampling
        "fermat_sample": True,
        "snr": 40,
    },
    "training": {
        "max_epochs": 25,
        "n_epochs": 1000,
        "n_epochs_init": 100,
        "max_epochs_init": 100,
        "loss_threshold": 0.001,
        "lr": 0.001,
    },
}

HIGH_QUALITY: PresetDict = {
    "name": None,
    "comment": "High-quality preset for maximum reconstruction fidelity",
    "save_data": True,
    "telescope": {
        "n_samples": 240,
        "sample_length": 0,
        "fermat_sample": True,
        "snr": 50,
    },
    "training": {
        "max_epochs": 50,
        "n_epochs": 2000,
        "n_epochs_init": 200,
        "max_epochs_init": 200,
        "loss_threshold": 0.0005,
        "lr": 0.001,
    },
}

DEBUG: PresetDict = {
    "name": None,
    "comment": "Debug preset for minimal testing without saving",
    "save_data": False,
    "telescope": {
        "n_samples": 16,
        "sample_length": 0,
        "fermat_sample": True,
    },
    "training": {
        "max_epochs": 1,
        "n_epochs": 100,
        "n_epochs_init": 10,
        "max_epochs_init": 1,
        "loss_threshold": 0.01,
    },
}

LINE_SAMPLING: PresetDict = {
    "name": None,
    "comment": "Line sampling preset for efficient measurement acquisition",
    "save_data": True,
    "telescope": {
        "n_samples": 100,
        "sample_length": 128,
        "samples_per_line_meas": 17,
        "fermat_sample": True,
    },
    "training": {
        "max_epochs": 25,
        "n_epochs": 1000,
        "loss_threshold": 0.001,
    },
}

# =============================================================================
# Mo-PIE Presets (for main_mopie.py - traditional phase retrieval)
# =============================================================================

MOPIE_BASELINE: PresetDict = {
    "name": None,
    "comment": "Mo-PIE baseline preset for traditional phase retrieval comparison",
    "save_data": True,
    "telescope": {
        "n_samples": 200,
        "sample_length": 0,
        "fermat_sample": True,
    },
    "training": {
        "n_epochs": 500,
    },
    "mopie": {
        "lr_obj": 1.0,
        "lr_probe": 1.0,
        "fix_probe": True,
        "parallel_update": True,
        "plot_every": 10,
    },
}

MOPIE_FAST: PresetDict = {
    "name": None,
    "comment": "Fast Mo-PIE preset for quick testing",
    "save_data": False,
    "telescope": {
        "n_samples": 64,
        "sample_length": 0,
        "fermat_sample": True,
    },
    "training": {
        "n_epochs": 100,
    },
    "mopie": {
        "lr_obj": 1.0,
        "lr_probe": 1.0,
        "fix_probe": True,
        "parallel_update": True,
        "plot_every": 5,
    },
}

MOPIE_HIGH_QUALITY: PresetDict = {
    "name": None,
    "comment": "High-quality Mo-PIE preset for maximum fidelity",
    "save_data": True,
    "telescope": {
        "n_samples": 300,
        "sample_length": 0,
        "fermat_sample": True,
        "snr": 50,
    },
    "training": {
        "n_epochs": 1000,
    },
    "mopie": {
        "lr_obj": 0.8,
        "lr_probe": 0.8,
        "fix_probe": True,
        "parallel_update": True,
        "plot_every": 20,
        "rand_perm": True,  # Randomize sample order each epoch
    },
}

# =============================================================================
# Object-Specific Presets
# =============================================================================

EUROPA_PRODUCTION: PresetDict = {
    **PRODUCTION,
    "comment": "Europa production preset",
    "physics": {
        "obj_name": "europa",
    },
}

TITAN_PRODUCTION: PresetDict = {
    **PRODUCTION,
    "comment": "Titan production preset",
    "physics": {
        "obj_name": "titan",
    },
}

BETELGEUSE_PRODUCTION: PresetDict = {
    **PRODUCTION,
    "comment": "Betelgeuse production preset",
    "physics": {
        "obj_name": "betelgeuse",
    },
}

NEPTUNE_PRODUCTION: PresetDict = {
    **PRODUCTION,
    "comment": "Neptune production preset",
    "physics": {
        "obj_name": "neptune",
    },
}

# =============================================================================
# Preset Registry
# =============================================================================

# Main SPIDS presets (for main.py)
PRISM_PRESETS: Dict[str, PresetDict] = {
    "quick_test": QUICK_TEST,
    "production": PRODUCTION,
    "high_quality": HIGH_QUALITY,
    "debug": DEBUG,
    "line_sampling": LINE_SAMPLING,
    # Object-specific
    "europa": EUROPA_PRODUCTION,
    "titan": TITAN_PRODUCTION,
    "betelgeuse": BETELGEUSE_PRODUCTION,
    "neptune": NEPTUNE_PRODUCTION,
}

# Mo-PIE presets (for main_mopie.py)
MOPIE_PRESETS: Dict[str, PresetDict] = {
    "mopie_baseline": MOPIE_BASELINE,
    "mopie_fast": MOPIE_FAST,
    "mopie_high_quality": MOPIE_HIGH_QUALITY,
    # Can also use SPIDS presets with Mo-PIE
    "quick_test": QUICK_TEST,
    "production": PRODUCTION,
    "debug": DEBUG,
}

# All presets combined
ALL_PRESETS: Dict[str, PresetDict] = {
    **PRISM_PRESETS,
    **MOPIE_PRESETS,
}

# =============================================================================
# Public API
# =============================================================================


def get_preset(preset_name: str, mode: str = "prism") -> Optional[PresetDict]:
    """
    Get a preset configuration by name.

    Args:
        preset_name: Name of the preset to retrieve
        mode: Either "prism" (for main.py) or "mopie" (for main_mopie.py)

    Returns:
        Preset dictionary if found, None otherwise

    Examples:
        >>> preset = get_preset("quick_test", mode="prism")
        >>> preset = get_preset("mopie_baseline", mode="mopie")
    """
    if mode == "prism":
        return PRISM_PRESETS.get(preset_name)
    elif mode == "mopie":
        return MOPIE_PRESETS.get(preset_name)
    else:
        return ALL_PRESETS.get(preset_name)


def list_presets(mode: str = "prism") -> List[str]:
    """
    List all available preset names for a given mode.

    Args:
        mode: Either "prism", "mopie", or "all"

    Returns:
        List of preset names

    Examples:
        >>> spids_presets = list_presets("prism")
        >>> mopie_presets = list_presets("mopie")
        >>> all_presets = list_presets("all")
    """
    if mode == "prism":
        return sorted(PRISM_PRESETS.keys())
    elif mode == "mopie":
        return sorted(MOPIE_PRESETS.keys())
    else:
        return sorted(ALL_PRESETS.keys())


def get_preset_description(preset_name: str) -> str:
    """
    Get the description/comment for a preset.

    Args:
        preset_name: Name of the preset

    Returns:
        Description string, or empty string if preset not found
    """
    preset = ALL_PRESETS.get(preset_name)
    if preset:
        return str(preset.get("comment", ""))
    return ""


def merge_preset_with_overrides(preset: PresetDict, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge a preset with user overrides.

    User overrides take precedence over preset values.

    Args:
        preset: Preset configuration dictionary
        overrides: User-provided overrides

    Returns:
        Merged configuration dictionary

    Examples:
        >>> base = get_preset("quick_test")
        >>> custom = {"telescope": {"n_samples": 128}}
        >>> merged = merge_preset_with_overrides(base, custom)
    """
    import copy

    result = copy.deepcopy(preset)

    def deep_merge(base_dict: dict, override_dict: dict) -> dict:
        """Recursively merge override_dict into base_dict."""
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                base_dict[key] = deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict

    return deep_merge(result, overrides)


# =============================================================================
# Preset Validation
# =============================================================================


def validate_preset_name(preset_name: str, mode: str = "prism") -> bool:
    """
    Check if a preset name is valid for the given mode.

    Args:
        preset_name: Preset name to validate
        mode: Either "prism" or "mopie"

    Returns:
        True if preset exists, False otherwise
    """
    return preset_name in list_presets(mode)


def get_preset_summary(preset_name: str) -> str:
    """
    Get a human-readable summary of a preset's key parameters.

    Args:
        preset_name: Name of the preset

    Returns:
        Formatted summary string

    Examples:
        >>> print(get_preset_summary("quick_test"))
        Quick test preset for fast iteration and debugging
        - Samples: 64
        - Max epochs: 1
        - Save data: False
        - Sampling: Fermat spiral
    """
    preset = ALL_PRESETS.get(preset_name)
    if not preset:
        return f"Preset '{preset_name}' not found"

    lines = [preset.get("comment", "No description")]

    # Extract key parameters
    telescope = preset.get("telescope", {})
    training = preset.get("training", {})
    mopie = preset.get("mopie", {})

    if "n_samples" in telescope:
        lines.append(f"- Samples: {telescope['n_samples']}")

    if "sample_length" in telescope:
        sample_type = "Line" if telescope["sample_length"] > 0 else "Point"
        lines.append(f"- Sample type: {sample_type}")

    if telescope.get("fermat_sample"):
        lines.append("- Sampling: Fermat spiral")
    elif telescope.get("star_sample"):
        lines.append("- Sampling: Star pattern")

    if "max_epochs" in training:
        lines.append(f"- Max epochs: {training['max_epochs']}")

    if "n_epochs" in training:
        lines.append(f"- Epochs: {training['n_epochs']}")

    if "snr" in telescope:
        lines.append(f"- SNR: {telescope['snr']} dB")

    if "save_data" in preset:
        lines.append(f"- Save data: {preset['save_data']}")

    # Mo-PIE specific
    if mopie:
        lines.append(f"- Mo-PIE lr_obj: {mopie.get('lr_obj', 'N/A')}")
        lines.append(f"- Fix probe: {mopie.get('fix_probe', 'N/A')}")

    return "\n".join(lines)

# prism.config.presets

Module: presets.py
Purpose: Built-in configuration presets for common PRISM experiment patterns

This module provides pre-configured experiment templates that combine common
parameter patterns, reducing the need for verbose CLI commands or config files.

Usage:
    # Via CLI
    python main.py --preset quick_test --obj europa
    python main_epie.py --preset epie_baseline --obj titan

    # Via Python API
    from prism.config.presets import get_preset, list_presets
    preset_config = get_preset("production")

## Classes

## Functions

### get_preset

```python
get_preset(preset_name: str, mode: str = 'prism') -> Optional[Dict[str, Any]]
```

Get a preset configuration by name.

Args:
    preset_name: Name of the preset to retrieve
    mode: Either "prism" (for main.py) or "epie" (for main_epie.py)

Returns:
    Preset dictionary if found, None otherwise

Examples:
    >>> preset = get_preset("quick_test", mode="prism")
    >>> preset = get_preset("epie_baseline", mode="epie")

### get_preset_description

```python
get_preset_description(preset_name: str) -> str
```

Get the description/comment for a preset.

Args:
    preset_name: Name of the preset

Returns:
    Description string, or empty string if preset not found

### get_preset_summary

```python
get_preset_summary(preset_name: str) -> str
```

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

### list_presets

```python
list_presets(mode: str = 'prism') -> List[str]
```

List all available preset names for a given mode.

Args:
    mode: Either "prism", "epie", or "all"

Returns:
    List of preset names

Examples:
    >>> prism_presets = list_presets("prism")
    >>> epie_presets = list_presets("epie")
    >>> all_presets = list_presets("all")

### merge_preset_with_overrides

```python
merge_preset_with_overrides(preset: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]
```

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

### validate_preset_name

```python
validate_preset_name(preset_name: str, mode: str = 'prism') -> bool
```

Check if a preset name is valid for the given mode.

Args:
    preset_name: Preset name to validate
    mode: Either "prism" or "epie"

Returns:
    True if preset exists, False otherwise

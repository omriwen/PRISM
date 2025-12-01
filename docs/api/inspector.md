# prism.config.inspector

Module: inspector.py
Purpose: Configuration inspection and visualization utilities

Provides tools to explore, validate, and display PRISM configuration options:
- List available presets
- Show preset details
- Show predefined object parameters
- Display effective configuration
- Pretty-print config with rich formatting

Usage:
    # Via CLI
    python main.py --list-presets
    python main.py --show-preset quick_test
    python main.py --show-object europa
    python main.py --preset production --show-config

    # Via Python API
    from prism.config.inspector import show_effective_config
    show_effective_config(args)

## Classes

## Functions

### compare_configs

```python
compare_configs(config1_path: str, config2_path: str) -> None
```

Compare two configuration files and show differences.

Args:
    config1_path: Path to first config file
    config2_path: Path to second config file

### handle_inspection_flags

```python
handle_inspection_flags(args: Any, mode: str = 'prism') -> None
```

Handle inspection flags (--list-presets, --show-preset, --show-object).

Args:
    args: Parsed command-line arguments
    mode: Either "prism" or "epie"

### list_all_presets

```python
list_all_presets(mode: str = 'prism') -> None
```

Display all available presets in a formatted table.

Args:
    mode: Either "prism" (for main.py) or "epie" (for main_epie.py)

### print_config_summary

```python
print_config_summary(args: Any, show_all: bool = False) -> None
```

Print a compact summary of the current configuration.

Args:
    args: Parsed arguments
    show_all: If True, show all parameters. If False, show only key parameters

### show_effective_config

```python
show_effective_config(args: Any) -> None
```

Display the effective configuration after all merging and overrides.

Args:
    args: Parsed and processed command-line arguments

### show_object_parameters

```python
show_object_parameters(obj_name: str) -> None
```

Display parameters for a predefined astronomical object.

Args:
    obj_name: Name of the object (europa, titan, betelgeuse, neptune)

### show_preset_details

```python
show_preset_details(preset_name: str, mode: str = 'prism') -> None
```

Display detailed information about a specific preset.

Args:
    preset_name: Name of the preset to display
    mode: Either "prism" or "epie"

### validate_and_report

```python
validate_and_report(args: Any) -> bool
```

Validate configuration and print detailed error report if invalid.

Args:
    args: Parsed arguments or PRISMConfig object

Returns:
    True if valid, False otherwise

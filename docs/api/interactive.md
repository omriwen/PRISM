# prism.config.interactive

Module: prism.config.interactive
Purpose: Interactive CLI wizard for PRISM experiment configuration

This module provides an interactive mode for users who prefer guided parameter
selection over command-line flags or YAML configuration files.

Usage:
    # From main.py or main_epie.py
    from prism.config.interactive import run_interactive_setup
    args = run_interactive_setup(mode="prism")

Features:
    - Preset selection with descriptions
    - Astronomical object selection
    - Interactive parameter configuration
    - Configuration summary and validation
    - Optional config file saving
    - Clean cancellation with Ctrl+C

## Classes

## Functions

### configure_telescope_params

```python
configure_telescope_params(preset: Dict[str, Any]) -> Dict[str, Any]
```

Interactively configure telescope parameters.

Shows preset defaults and allows overrides for commonly-changed parameters.

Args:
    preset: Preset configuration dictionary

Returns:
    Dictionary of parameter overrides

Examples:
    >>> preset = get_preset("production")
    >>> overrides = configure_telescope_params(preset)

### configure_training_params

```python
configure_training_params(preset: Dict[str, Any], mode: str) -> Dict[str, Any]
```

Interactively configure training parameters.

Args:
    preset: Preset configuration dictionary
    mode: Either "prism" or "epie"

Returns:
    Dictionary of parameter overrides

### confirm_and_save

```python
confirm_and_save(config: Dict[str, Any], preset_name: str, obj_name: str, mode: str) -> Tuple[bool, Optional[str], str]
```

Display summary, optionally save config, and confirm execution.

Args:
    config: Complete configuration dictionary
    preset_name: Name of the selected preset
    obj_name: Name of the selected object
    mode: Either "prism" or "epie"

Returns:
    Tuple of (should_run, config_path, experiment_name)

### run_interactive_setup

```python
run_interactive_setup(mode: str = 'prism') -> Optional[argparse.Namespace]
```

Run the interactive configuration wizard.

This is the main entry point for interactive mode. It guides the user
through:
1. Preset selection
2. Object selection
3. Telescope parameter configuration
4. Training parameter configuration
5. Configuration review and saving

Args:
    mode: Either "prism" (for main.py) or "epie" (for main_epie.py)

Returns:
    argparse.Namespace with all configured parameters, or None if cancelled

Examples:
    >>> args = run_interactive_setup(mode="prism")
    >>> if args:
    ...     # Proceed with experiment using args
    ...     pass

### select_object

```python
select_object() -> str
```

Display astronomical object menu and get user selection.

Returns:
    Selected object name

Examples:
    >>> obj = select_object()
    >>> print(obj)
    'europa'

### select_preset

```python
select_preset(mode: str) -> str
```

Display preset menu and get user selection.

Args:
    mode: Either "prism" or "epie"

Returns:
    Selected preset name

Examples:
    >>> preset = select_preset("prism")
    >>> print(preset)
    'quick_test'

### show_config_summary

```python
show_config_summary(config: Dict[str, Any], preset_name: str, obj_name: str) -> None
```

Display configuration summary in a formatted panel.

Args:
    config: Complete configuration dictionary
    preset_name: Name of the selected preset
    obj_name: Name of the selected object

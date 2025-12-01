"""
Module: inspector.py
Purpose: Configuration inspection and visualization utilities

Provides tools to explore, validate, and display SPIDS configuration options:
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
"""

from __future__ import annotations

from typing import Any, Dict

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# Initialize rich console
console = Console()


def handle_inspection_flags(args: Any, mode: str = "prism") -> None:
    """
    Handle inspection flags (--list-presets, --show-preset, --show-object).

    Args:
        args: Parsed command-line arguments
        mode: Either "prism" or "epie"
    """
    if args.list_presets:
        list_all_presets(mode)
    elif args.show_preset:
        show_preset_details(args.show_preset, mode)
    elif args.show_object:
        show_object_parameters(args.show_object)


def list_all_presets(mode: str = "prism") -> None:
    """
    Display all available presets in a formatted table.

    Args:
        mode: Either "prism" (for main.py) or "epie" (for main_epie.py)
    """
    from prism.config.presets import get_preset_description, list_presets

    presets = list_presets(mode)

    # Create table
    table = Table(
        title=f"Available {'ePIE' if mode == 'epie' else 'SPIDS'} Configuration Presets",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Preset Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    # Add rows
    for preset_name in presets:
        description = get_preset_description(preset_name)
        table.add_row(preset_name, description)

    console.print()
    console.print(table)
    console.print()

    # Show usage example
    example_preset = presets[0] if presets else "quick_test"
    script_name = "main_epie.py" if mode == "epie" else "main.py"

    console.print(
        Panel(
            f"[bold]Usage Examples:[/bold]\n\n"
            f"  # Use a preset\n"
            f"  [cyan]uv run python {script_name} --preset {example_preset} --obj europa --name my_test[/cyan]\n\n"
            f"  # Show preset details\n"
            f"  [cyan]uv run python {script_name} --show-preset {example_preset}[/cyan]\n\n"
            f"  # Override preset parameters\n"
            f"  [cyan]uv run python {script_name} --preset {example_preset} --n_samples 128 --name custom[/cyan]",
            title="Quick Start",
            border_style="green",
        )
    )


def show_preset_details(preset_name: str, mode: str = "prism") -> None:
    """
    Display detailed information about a specific preset.

    Args:
        preset_name: Name of the preset to display
        mode: Either "prism" or "epie"
    """
    from prism.config.presets import get_preset, get_preset_summary, validate_preset_name

    if not validate_preset_name(preset_name, mode):
        console.print(f"[red]Error:[/red] Unknown preset '{preset_name}'")
        console.print("Use [cyan]--list-presets[/cyan] to see available presets")
        return

    # Get preset and summary
    preset = get_preset(preset_name, mode)
    if preset is None:
        console.print(f"[red]Error:[/red] Could not load preset '{preset_name}'")
        return

    summary = get_preset_summary(preset_name)

    # Display summary
    console.print()
    console.print(
        Panel(
            summary,
            title=f"Preset: {preset_name}",
            border_style="cyan",
        )
    )

    # Display full configuration
    console.print()
    console.print("[bold]Full Configuration:[/bold]")
    _print_config_dict(preset, indent=0)

    console.print()


def show_object_parameters(obj_name: str) -> None:
    """
    Display parameters for a predefined astronomical object.

    Args:
        obj_name: Name of the object (europa, titan, betelgeuse, neptune)
    """
    from prism.config.objects import PREDEFINED_OBJECTS

    if obj_name not in PREDEFINED_OBJECTS:
        console.print(f"[red]Error:[/red] Unknown object '{obj_name}'")
        console.print(f"Available objects: {', '.join(PREDEFINED_OBJECTS.keys())}")
        return

    obj_params = PREDEFINED_OBJECTS[obj_name]

    # Create table
    table = Table(
        title=f"Predefined Parameters for: {obj_name.title()}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_column("Description", style="white")

    # Add rows
    descriptions = {
        "wavelength": "Observation wavelength [m]",
        "obj_diameter": "Real diameter of object [m]",
        "obj_distance": "Distance from telescope [m]",
        "input": "Path to image file",
        "sample_diameter": "Telescope aperture size [pixels]",
    }

    for key, value in obj_params.items():
        desc = descriptions.get(key, "")
        if isinstance(value, float):
            if value < 1e-3:
                value_str = f"{value:.2e}"
            else:
                value_str = f"{value:.6f}"
        else:
            value_str = str(value)

        table.add_row(key, value_str, desc)

    console.print()
    console.print(table)

    # Show usage example
    console.print()
    console.print(
        Panel(
            f"[bold]Usage:[/bold]\n\n"
            f"  [cyan]uv run python main.py --obj {obj_name} --preset quick_test --name test[/cyan]",
            title="Example",
            border_style="green",
        )
    )
    console.print()


def show_effective_config(args: Any) -> None:
    """
    Display the effective configuration after all merging and overrides.

    Args:
        args: Parsed and processed command-line arguments
    """
    from prism.config import args_to_config

    # Convert to config
    config = args_to_config(args)

    # Display title
    console.print()
    console.print(
        Panel(
            "[bold]Effective Configuration[/bold]\n"
            "This shows the final configuration after merging:\n"
            "  1. Preset (if specified)\n"
            "  2. Config file (if specified)\n"
            "  3. Command-line arguments (highest priority)",
            border_style="cyan",
        )
    )

    # Display each section
    sections = [
        (
            "General",
            {
                "name": config.name,
                "comment": config.comment,
                "log_dir": config.log_dir,
                "save_data": config.save_data,
                "checkpoint": config.checkpoint,
            },
        ),
        ("Image", vars(config.image)),
        ("Telescope", vars(config.telescope)),
        ("Training", vars(config.training)),
        ("Model", vars(config.model)),
        ("Physics", vars(config.physics)),
        ("Point Source", vars(config.point_source)),
    ]

    # Add ePIE section if present
    if hasattr(config, "epie") and config.epie is not None:
        sections.append(("ePIE", vars(config.epie)))

    for section_name, section_dict in sections:
        console.print()
        console.print(f"[bold magenta]{section_name}:[/bold magenta]")
        _print_config_dict(section_dict, indent=2)

    console.print()

    # Validate and show result
    try:
        config.validate()
        console.print(
            Panel(
                "[bold green]✓ Configuration is valid[/bold green]",
                border_style="green",
            )
        )
    except ValueError as e:
        console.print(
            Panel(
                f"[bold red]✗ Configuration error:[/bold red]\n\n{str(e)}",
                border_style="red",
            )
        )

    console.print()


def _print_config_dict(config_dict: Dict[str, Any], indent: int = 0) -> None:
    """
    Pretty-print a configuration dictionary with colors.

    Args:
        config_dict: Dictionary to print
        indent: Indentation level (spaces)
    """
    indent_str = " " * indent

    for key, value in config_dict.items():
        # Skip private/internal keys
        if key.startswith("_"):
            continue

        # Format value
        if isinstance(value, dict):
            console.print(f"{indent_str}[cyan]{key}:[/cyan]")
            _print_config_dict(value, indent + 2)
        elif isinstance(value, bool):
            value_color = "green" if value else "red"
            console.print(f"{indent_str}[cyan]{key}:[/cyan] [{value_color}]{value}[/{value_color}]")
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and value < 1e-3:
                value_str = f"{value:.2e}"
            else:
                value_str = str(value)
            console.print(f"{indent_str}[cyan]{key}:[/cyan] [yellow]{value_str}[/yellow]")
        elif value is None:
            console.print(f"{indent_str}[cyan]{key}:[/cyan] [dim]None[/dim]")
        else:
            console.print(f"{indent_str}[cyan]{key}:[/cyan] [white]{value}[/white]")


def validate_and_report(args: Any) -> bool:
    """
    Validate configuration and print detailed error report if invalid.

    Args:
        args: Parsed arguments or PRISMConfig object

    Returns:
        True if valid, False otherwise
    """
    from prism.config import args_to_config

    # Convert to config if needed
    if not hasattr(args, "validate"):
        config = args_to_config(args)
    else:
        config = args

    try:
        config.validate()
        console.print("[bold green]✓ Configuration is valid[/bold green]")
        return True
    except ValueError as e:
        console.print()
        console.print(
            Panel(
                f"[bold red]Configuration Error:[/bold red]\n\n{str(e)}",
                title="Validation Failed",
                border_style="red",
            )
        )
        console.print()
        return False


def compare_configs(config1_path: str, config2_path: str) -> None:
    """
    Compare two configuration files and show differences.

    Args:
        config1_path: Path to first config file
        config2_path: Path to second config file
    """
    from prism.config.loader import load_config

    # Load both configs
    config1 = load_config(config1_path)
    config2 = load_config(config2_path)

    # Convert to dicts
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()

    # Find differences
    differences = _find_dict_differences(dict1, dict2)

    if not differences:
        console.print("[green]✓ Configurations are identical[/green]")
        return

    # Display differences
    table = Table(
        title="Configuration Differences",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Parameter", style="cyan")
    table.add_column(config1_path, style="yellow")
    table.add_column(config2_path, style="green")

    for key, (val1, val2) in differences.items():
        table.add_row(key, str(val1), str(val2))

    console.print()
    console.print(table)
    console.print()


def _find_dict_differences(dict1: Dict, dict2: Dict, prefix: str = "") -> Dict:
    """
    Recursively find differences between two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary
        prefix: Key prefix for nested dicts

    Returns:
        Dictionary of differences {key: (value1, value2)}
    """
    differences = {}

    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        full_key = f"{prefix}.{key}" if prefix else key

        val1 = dict1.get(key)
        val2 = dict2.get(key)

        if isinstance(val1, dict) and isinstance(val2, dict):
            # Recurse into nested dicts
            nested_diffs = _find_dict_differences(val1, val2, full_key)
            differences.update(nested_diffs)
        elif val1 != val2:
            differences[full_key] = (val1, val2)

    return differences


def print_config_summary(args: Any, show_all: bool = False) -> None:
    """
    Print a compact summary of the current configuration.

    Args:
        args: Parsed arguments
        show_all: If True, show all parameters. If False, show only key parameters
    """
    console.print()
    console.print("[bold]Experiment Configuration Summary[/bold]")
    console.print()

    # Key parameters
    key_params = {
        "Name": getattr(args, "name", "N/A"),
        "Object": getattr(args, "obj_name", "N/A"),
        "Samples": getattr(args, "n_samples", "N/A"),
        "Sampling": (
            "Fermat"
            if getattr(args, "fermat_sample", False)
            else "Star"
            if getattr(args, "star_sample", False)
            else "Random"
        ),
        "Max Epochs": getattr(args, "max_epochs", "N/A"),
        "Learning Rate": getattr(args, "lr", "N/A") if hasattr(args, "lr") else "N/A",
        "Save Data": getattr(args, "save_data", True),
    }

    for key, value in key_params.items():
        console.print(f"  [cyan]{key}:[/cyan] [yellow]{value}[/yellow]")

    if show_all:
        console.print()
        show_effective_config(args)

    console.print()

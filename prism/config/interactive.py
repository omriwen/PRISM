"""
Module: spids.config.interactive
Purpose: Interactive CLI wizard for SPIDS experiment configuration

This module provides an interactive mode for users who prefer guided parameter
selection over command-line flags or YAML configuration files.

Usage:
    # From main.py or main_mopie.py
    from prism.config.interactive import run_interactive_setup
    args = run_interactive_setup(mode="prism")

Features:
    - Preset selection with descriptions
    - Astronomical object selection
    - Interactive parameter configuration
    - Configuration summary and validation
    - Optional config file saving
    - Clean cancellation with Ctrl+C
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from .loader import args_to_config, save_config
from .objects import PREDEFINED_OBJECTS
from .presets import get_preset, get_preset_description, list_presets


# Initialize rich console
console = Console()


# =============================================================================
# Object Selection
# =============================================================================


def _get_object_description(obj_name: str) -> str:
    """Get human-readable object description."""
    descriptions = {
        "europa": "Jupiter's moon",
        "titan": "Saturn's moon",
        "betelgeuse": "Red supergiant star",
        "neptune": "Ice giant planet",
    }
    return descriptions.get(obj_name, "Custom astronomical object")


def select_object() -> str:
    """
    Display astronomical object menu and get user selection.

    Returns:
        Selected object name

    Examples:
        >>> obj = select_object()
        >>> print(obj)
        'europa'
    """
    objects = sorted(PREDEFINED_OBJECTS.keys())

    console.print("\n[bold cyan]═══ Step 2/5: Astronomical Object ═══[/bold cyan]\n")
    console.print("[bold]Available objects:[/bold]")

    for i, obj_name in enumerate(objects, 1):
        desc = _get_object_description(obj_name)
        marker = "→" if obj_name == "europa" else " "
        console.print(f"  {marker} {i}. [cyan]{obj_name:<15}[/cyan] {desc}")

    console.print()

    # Get choice
    choices = [str(i) for i in range(1, len(objects) + 1)] + objects
    choice = Prompt.ask(
        "Select object (number or name)",
        choices=choices,
        default="europa",
        show_choices=False,
    )

    # Convert number to object name if needed
    if choice.isdigit():
        return str(objects[int(choice) - 1])
    return str(choice)


# =============================================================================
# Preset Selection
# =============================================================================


def select_preset(mode: str) -> str:
    """
    Display preset menu and get user selection.

    Args:
        mode: Either "prism" or "mopie"

    Returns:
        Selected preset name

    Examples:
        >>> preset = select_preset("prism")
        >>> print(preset)
        'quick_test'
    """
    presets = list_presets(mode)

    console.print("\n[bold cyan]═══ Step 1/5: Experiment Preset ═══[/bold cyan]\n")
    console.print("[bold]Available presets:[/bold]\n")

    # Display preset menu with descriptions
    for i, preset_name in enumerate(presets, 1):
        desc = get_preset_description(preset_name)
        # Get first line of description
        first_line = desc.split("\n")[0] if desc else "No description"

        # Highlight recommended presets
        marker = "→" if preset_name in ["quick_test", "production"] else " "
        console.print(f"  {marker} {i}. [cyan]{preset_name:<20}[/cyan] {first_line}")

    console.print()

    # Get choice
    choices = [str(i) for i in range(1, len(presets) + 1)] + presets
    choice = Prompt.ask(
        "Select preset (number or name)",
        choices=choices,
        default="production",
        show_choices=False,
    )

    # Convert number to preset name if needed
    if choice.isdigit():
        return str(presets[int(choice) - 1])
    return str(choice)


# =============================================================================
# Parameter Configuration
# =============================================================================


def configure_telescope_params(preset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interactively configure telescope parameters.

    Shows preset defaults and allows overrides for commonly-changed parameters.

    Args:
        preset: Preset configuration dictionary

    Returns:
        Dictionary of parameter overrides

    Examples:
        >>> preset = get_preset("production")
        >>> overrides = configure_telescope_params(preset)
    """
    console.print("\n[bold cyan]═══ Step 3/5: Telescope Parameters ═══[/bold cyan]\n")

    telescope = preset.get("telescope", {})
    overrides: Dict[str, Any] = {}

    # Number of samples
    default_samples = telescope.get("n_samples", 200)
    console.print("[bold]Number of telescope samples[/bold]")
    console.print(f"  Current default: [yellow]{default_samples}[/yellow]")
    console.print("  Typical range: 64 (quick test) to 300 (high quality)")

    n_samples = IntPrompt.ask(
        "  Enter new value, or press Enter to keep default",
        default=default_samples,
        show_default=False,
    )

    if n_samples != default_samples:
        overrides["n_samples"] = n_samples

    # SNR (if defined in preset)
    if "snr" in telescope:
        default_snr = telescope["snr"]
        console.print("\n[bold]Signal-to-noise ratio (dB)[/bold]")
        console.print(f"  Current default: [yellow]{default_snr}[/yellow] dB")
        console.print("  Typical range: 20 (noisy) to 60 (clean)")

        snr = FloatPrompt.ask(
            "  Enter new value, or press Enter to keep default",
            default=float(default_snr),
            show_default=False,
        )

        if snr != default_snr:
            overrides["snr"] = snr

    # Fermat spiral sampling (if defined)
    if "fermat_sample" in telescope:
        default_fermat = telescope.get("fermat_sample", True)
        console.print("\n[bold]Sampling pattern[/bold]")
        console.print(
            f"  Current: [yellow]{'Fermat spiral' if default_fermat else 'Random'}[/yellow]"
        )
        console.print("  Fermat spiral provides better k-space coverage")

        use_fermat = Confirm.ask(
            "  Use Fermat spiral sampling?",
            default=default_fermat,
        )

        if use_fermat != default_fermat:
            overrides["fermat_sample"] = use_fermat

    return overrides


def configure_training_params(preset: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """
    Interactively configure training parameters.

    Args:
        preset: Preset configuration dictionary
        mode: Either "prism" or "mopie"

    Returns:
        Dictionary of parameter overrides
    """
    console.print("\n[bold cyan]═══ Step 4/5: Training Parameters ═══[/bold cyan]\n")

    training = preset.get("training", {})
    overrides: Dict[str, Any] = {}

    if mode == "prism":
        # Max epochs (SPIDS specific)
        default_max_epochs = training.get("max_epochs", 25)
        console.print("[bold]Maximum epochs per sample[/bold]")
        console.print(f"  Current default: [yellow]{default_max_epochs}[/yellow]")
        console.print("  Typical: 1 (quick test) to 50 (high quality)")

        max_epochs = IntPrompt.ask(
            "  Enter new value, or press Enter to keep default",
            default=default_max_epochs,
            show_default=False,
        )

        if max_epochs != default_max_epochs:
            overrides["max_epochs"] = max_epochs

    # Ask about advanced parameters
    console.print("\n[bold]Advanced training settings[/bold]")
    configure_advanced = Confirm.ask(
        "  Configure advanced parameters? (learning rate, loss threshold, etc.)",
        default=False,
    )

    if configure_advanced:
        # Learning rate
        default_lr = training.get("lr", 0.001)
        console.print("\n  [bold]Learning rate[/bold]")
        console.print(f"    Current default: [yellow]{default_lr}[/yellow]")

        lr = FloatPrompt.ask(
            "    Enter new value, or press Enter to keep default",
            default=float(default_lr),
            show_default=False,
        )

        if lr != default_lr:
            overrides["lr"] = lr

        # Loss threshold
        default_loss_th = training.get("loss_threshold", 0.001)
        console.print("\n  [bold]Loss threshold for convergence[/bold]")
        console.print(f"    Current default: [yellow]{default_loss_th}[/yellow]")

        loss_th = FloatPrompt.ask(
            "    Enter new value, or press Enter to keep default",
            default=float(default_loss_th),
            show_default=False,
        )

        if loss_th != default_loss_th:
            overrides["loss_threshold"] = loss_th

    return overrides


# =============================================================================
# Configuration Summary and Saving
# =============================================================================


def show_config_summary(config: Dict[str, Any], preset_name: str, obj_name: str) -> None:
    """
    Display configuration summary in a formatted panel.

    Args:
        config: Complete configuration dictionary
        preset_name: Name of the selected preset
        obj_name: Name of the selected object
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="yellow")

    # Extract key parameters
    telescope = config.get("telescope", {})
    training = config.get("training", {})

    # Add rows
    table.add_row("Preset", preset_name)
    table.add_row("Object", obj_name)
    table.add_row("Samples", str(telescope.get("n_samples", "N/A")))

    # Sampling pattern
    if telescope.get("fermat_sample"):
        sampling = "Fermat spiral"
    elif telescope.get("star_sample"):
        sampling = "Star pattern"
    else:
        sampling = "Random"
    table.add_row("Sampling", sampling)

    # SNR
    if "snr" in telescope:
        table.add_row("SNR", f"{telescope['snr']} dB")

    # Training parameters
    if "max_epochs" in training:
        table.add_row("Max Epochs", str(training["max_epochs"]))

    if "n_epochs" in training:
        table.add_row("Epochs/Sample", str(training["n_epochs"]))

    if "lr" in training:
        table.add_row("Learning Rate", str(training["lr"]))

    # Save data
    table.add_row("Save Data", str(config.get("save_data", True)))

    console.print("\n")
    console.print(
        Panel(
            table,
            title="[bold green]Configuration Summary[/bold green]",
            border_style="green",
        )
    )


def confirm_and_save(
    config: Dict[str, Any],
    preset_name: str,
    obj_name: str,
    mode: str,
) -> Tuple[bool, Optional[str], str]:
    """
    Display summary, optionally save config, and confirm execution.

    Args:
        config: Complete configuration dictionary
        preset_name: Name of the selected preset
        obj_name: Name of the selected object
        mode: Either "prism" or "mopie"

    Returns:
        Tuple of (should_run, config_path, experiment_name)
    """
    console.print("\n[bold cyan]═══ Step 5/5: Review and Run ═══[/bold cyan]")

    # Show summary
    show_config_summary(config, preset_name, obj_name)

    # Ask for experiment name
    default_name = f"{obj_name}_{preset_name}"
    name = Prompt.ask(
        "\n[bold]Experiment name[/bold]",
        default=default_name,
    )

    # Ask to save config
    console.print()
    should_save = Confirm.ask(
        "[bold]Save configuration to file?[/bold]",
        default=True,
    )

    config_path = None
    if should_save:
        default_path = f"configs/{name}.yaml"
        config_path = Prompt.ask(
            "Config file path",
            default=default_path,
        )

    # Ask to run
    console.print()
    should_run = Confirm.ask(
        "[bold green]Run experiment now?[/bold green]",
        default=True,
    )

    return should_run, config_path, name


# =============================================================================
# Main Interactive Setup Function
# =============================================================================


def run_interactive_setup(mode: str = "prism") -> Optional[argparse.Namespace]:
    """
    Run the interactive configuration wizard.

    This is the main entry point for interactive mode. It guides the user
    through:
    1. Preset selection
    2. Object selection
    3. Telescope parameter configuration
    4. Training parameter configuration
    5. Configuration review and saving

    Args:
        mode: Either "prism" (for main.py) or "mopie" (for main_mopie.py)

    Returns:
        argparse.Namespace with all configured parameters, or None if cancelled

    Examples:
        >>> args = run_interactive_setup(mode="prism")
        >>> if args:
        ...     # Proceed with experiment using args
        ...     pass
    """
    try:
        # Display welcome banner
        console.print()
        console.print(
            Panel(
                "[bold]SPIDS Interactive Configuration Wizard[/bold]\n\n"
                "This wizard will guide you through experiment setup.\n"
                "Press [cyan]Ctrl+C[/cyan] at any time to cancel.",
                border_style="blue",
                padding=(1, 2),
            )
        )

        # Step 1: Select preset
        preset_name = select_preset(mode)
        preset = get_preset(preset_name, mode)

        if preset is None:
            console.print(f"[red]Error:[/red] Preset '{preset_name}' not found")
            return None

        # Step 2: Select object
        obj_name = select_object()

        # Step 3: Configure telescope parameters
        telescope_overrides = configure_telescope_params(preset)

        # Step 4: Configure training parameters
        training_overrides = configure_training_params(preset, mode)

        # Merge overrides into preset
        config = preset.copy()
        if telescope_overrides:
            config.setdefault("telescope", {}).update(telescope_overrides)
        if training_overrides:
            config.setdefault("training", {}).update(training_overrides)

        # Add physics config
        config.setdefault("physics", {})["obj_name"] = obj_name

        # Step 5: Confirm and save
        should_run, config_path, name = confirm_and_save(config, preset_name, obj_name, mode)

        # Convert config dict to argparse.Namespace
        args = _config_dict_to_namespace(config, name, config_path, preset_name, obj_name)

        # Save config file if requested
        if config_path:
            try:
                # Convert namespace to PRISMConfig
                prism_config = args_to_config(args)

                # Ensure directory exists
                config_file = Path(config_path)
                config_file.parent.mkdir(parents=True, exist_ok=True)

                # Save config
                save_config(prism_config, config_path, minimal=True)
                console.print(f"[green]✓[/green] Configuration saved to: {config_path}")
            except Exception as e:  # noqa: BLE001 - Config save failure is non-fatal
                console.print(f"[yellow]Warning:[/yellow] Failed to save config: {e}")
                console.print("[dim]Experiment will continue without saving config file[/dim]")

        if not should_run:
            console.print("\n[yellow]Setup complete. Experiment not started.[/yellow]")
            return None

        console.print(
            f"\n[bold green]✓[/bold green] Starting experiment '[cyan]{name}[/cyan]'...\n"
        )

        return args

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interactive setup cancelled[/yellow]")
        return None


def _config_dict_to_namespace(
    config: Dict[str, Any],
    name: str,
    config_path: Optional[str],
    preset_name: str,
    obj_name: str,
) -> argparse.Namespace:
    """
    Convert configuration dictionary to argparse.Namespace.

    This creates a namespace compatible with main.py and main_mopie.py
    argument parsing.

    Args:
        config: Configuration dictionary
        name: Experiment name
        config_path: Path to save config (or None)
        preset_name: Name of the preset
        obj_name: Name of the object

    Returns:
        argparse.Namespace with all parameters
    """
    args = argparse.Namespace()

    # Experiment metadata
    args.name = name
    args.config = config_path
    args.preset = preset_name
    args.obj_name = obj_name
    args.interactive = True
    args.save_data = config.get("save_data", True)
    args.comment = config.get("comment", "")

    # Disable inspection flags
    args.list_presets = False
    args.show_preset = None
    args.show_object = None
    args.validate_only = False

    # Telescope parameters
    telescope = config.get("telescope", {})
    args.n_samples = telescope.get("n_samples", 200)
    args.sample_diameter = telescope.get("sample_diameter")
    args.sample_shape = telescope.get("sample_shape", "circle")
    args.sample_length = telescope.get("sample_length", 0)
    args.samples_per_line_meas = telescope.get("samples_per_line_meas")
    args.samples_per_line_rec = telescope.get("samples_per_line_rec")
    args.line_angle = telescope.get("line_angle")
    args.roi_diameter = telescope.get("roi_diameter")
    args.samples_r_cutoff = telescope.get("samples_r_cutoff")
    args.roi_shape = telescope.get("roi_shape", "circle")
    args.sample_sort = telescope.get("sample_sort", "center")
    args.n_angs = telescope.get("n_angs", 4)
    args.star_sample = telescope.get("star_sample", False)
    args.fermat_sample = telescope.get("fermat_sample", False)
    args.snr = telescope.get("snr")
    args.blur = telescope.get("blur", False)

    # Physics parameters
    physics = config.get("physics", {})
    args.obj_diameter = physics.get("obj_diameter")
    args.obj_distance = physics.get("obj_distance")
    args.wavelength = physics.get("wavelength")
    args.dxf = physics.get("dxf", 1e-5)  # Default to 1e-5

    # Image parameters
    image = config.get("image", {})
    args.image_size = image.get("image_size", 1024)
    args.obj_size = image.get("obj_size")
    args.crop = image.get("crop", False)
    args.invert = image.get("invert", False)
    args.input = image.get("input")

    # Training parameters
    training = config.get("training", {})
    args.max_epochs = training.get("max_epochs", 25)
    args.n_epochs = training.get("n_epochs", 1000)
    args.n_epochs_init = training.get("n_epochs_init", 100)
    args.max_epochs_init = training.get("max_epochs_init", 100)
    args.loss_threshold = training.get("loss_threshold", 0.001)
    args.lr = training.get("lr", 0.001)

    # Model parameters
    model = config.get("model", {})
    args.use_bn = model.get("use_bn", True)
    args.output_activation = model.get("output_activation", "none")
    args.use_leaky = model.get("use_leaky", True)
    args.middle_activation = model.get("middle_activation", "sigmoid")
    args.complex_data = model.get("complex_data", False)

    # moPIE parameters (if present)
    mopie = config.get("mopie", {})
    args.lr_obj = mopie.get("lr_obj", 1.0)
    args.lr_probe = mopie.get("lr_probe", 1.0)
    args.fix_probe = mopie.get("fix_probe", True)
    args.parallel_update = mopie.get("parallel_update", True)
    args.plot_every = mopie.get("plot_every", 10)
    args.rand_perm = mopie.get("rand_perm", False)

    # Checkpoint and debugging
    args.checkpoint = None  # Interactive mode doesn't resume
    args.debug = False

    return args

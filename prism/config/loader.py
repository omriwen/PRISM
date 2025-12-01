"""Configuration loading and saving utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import yaml

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


def load_config(config_path: str, _visited: Optional[set] = None) -> PRISMConfig:
    """Load SPIDS configuration from YAML file with support for inheritance.

    Supports the 'extends' keyword for config inheritance:
    - extends: path/to/parent.yaml (relative path)
    - extends: presets/quick_test.yaml (preset template)
    - extends: quick_test (preset name - will be resolved)

    Child configs override parent values (deep merge for nested dicts).

    Args:
        config_path: Path to YAML configuration file
        _visited: Internal parameter to track visited configs (prevents circular refs)

    Returns:
        PRISMConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid or has circular inheritance
    """
    # Initialize visited set for circular reference detection
    if _visited is None:
        _visited = set()

    config_file = Path(config_path).resolve()

    # Check for circular references
    if str(config_file) in _visited:
        raise ValueError(f"Circular config inheritance detected: {config_path}")

    _visited.add(str(config_file))

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    # Handle config inheritance via 'extends' keyword
    if "extends" in data:
        parent_path = data.pop("extends")  # Remove extends from data

        # Resolve parent path
        parent_path = _resolve_config_path(parent_path, config_file.parent)

        # Load parent config recursively
        parent_data = _load_config_data(parent_path, _visited)

        # Deep merge: parent data + child data (child overrides parent)
        data = _deep_merge_dicts(parent_data, data)

    # Extract top-level experiment parameters
    name = data.get("name")
    comment = data.get("comment", "")
    log_dir = data.get("log_dir", "runs")
    save_data = data.get("save_data", True)
    checkpoint = data.get("checkpoint")

    # Parse nested configurations
    image_config = ImageConfig(**data.get("image", {}))
    telescope_config = TelescopeConfig(**data.get("telescope", {}))
    model_config = ModelConfig(**data.get("model", {}))
    training_config = TrainingConfig(**data.get("training", {}))
    physics_config = PhysicsConfig(**data.get("physics", {}))
    point_source_config = PointSourceConfig(**data.get("point_source", {}))

    # Parse MoPIE config if present
    mopie_config = None
    if "mopie" in data and data["mopie"] is not None:
        mopie_config = MoPIEConfig(**data["mopie"])

    # Create master config
    config = PRISMConfig(
        name=name,
        comment=comment,
        log_dir=log_dir,
        save_data=save_data,
        checkpoint=checkpoint,
        image=image_config,
        telescope=telescope_config,
        model=model_config,
        training=training_config,
        physics=physics_config,
        point_source=point_source_config,
        mopie=mopie_config,
    )

    # Validate configuration
    config.validate()

    return config


def _resolve_config_path(extends_path: str, current_dir: Path) -> Path:
    """
    Resolve a config path from the 'extends' field.

    Supports:
    - Relative paths: "presets/quick_test.yaml", "../other.yaml"
    - Absolute paths: "/full/path/to/config.yaml"
    - Preset shortcuts: "quick_test" → "configs/presets/quick_test.yaml"

    Args:
        extends_path: Path or preset name from 'extends' field
        current_dir: Directory containing the current config file

    Returns:
        Resolved absolute Path object
    """
    extends_path_obj = Path(extends_path)

    # Case 1: Absolute path
    if extends_path_obj.is_absolute():
        return extends_path_obj.resolve()

    # Case 2: Relative path (contains / or \ or has .yaml extension)
    if "/" in extends_path or "\\" in extends_path or extends_path.endswith(".yaml"):
        # Resolve relative to current config directory
        resolved = (current_dir / extends_path_obj).resolve()
        if resolved.exists():
            return resolved

    # Case 3: Preset name (no path separators, no extension)
    # Try configs/presets/{name}.yaml
    preset_path = Path("configs/presets") / f"{extends_path}.yaml"
    if preset_path.exists():
        return preset_path.resolve()

    # Case 4: Try as relative path one more time
    resolved = (current_dir / extends_path_obj).resolve()
    if resolved.exists():
        return resolved

    # Not found anywhere
    raise FileNotFoundError(
        f"Config inheritance failed: cannot resolve '{extends_path}'\n"
        f"  Searched:\n"
        f"    - Relative to {current_dir}\n"
        f"    - As preset: configs/presets/{extends_path}.yaml\n"
        f"  Make sure the parent config file exists"
    )


def _load_config_data(config_path: Path, visited: set) -> dict[Any, Any]:
    """
    Load config file as raw dictionary (supports recursive inheritance).

    Args:
        config_path: Path to config file
        visited: Set of already visited config paths

    Returns:
        Config dictionary with inheritance resolved
    """
    # Check for circular references
    if str(config_path) in visited:
        raise ValueError(f"Circular config inheritance detected: {config_path}")

    visited.add(str(config_path))

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data: dict[Any, Any] = yaml.safe_load(f) or {}

    if not data:
        data = {}

    # Handle recursive inheritance
    if "extends" in data:
        parent_path = data.pop("extends")
        parent_path_resolved = _resolve_config_path(parent_path, config_path.parent)
        parent_data = _load_config_data(parent_path_resolved, visited)
        data = _deep_merge_dicts(parent_data, data)

    return data


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries (override values take precedence).

    Recursively merges nested dictionaries. For non-dict values,
    override always wins.

    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)

    Returns:
        Merged dictionary

    Examples:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> override = {"b": {"c": 99}, "e": 4}
        >>> _deep_merge_dicts(base, override)
        {"a": 1, "b": {"c": 99, "d": 3}, "e": 4}
    """
    import copy

    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


def save_config(config: PRISMConfig, output_path: str, minimal: bool = False) -> None:
    """Save SPIDS configuration to YAML file.

    Args:
        config: PRISMConfig object to save
        output_path: Path to output YAML file
        minimal: If True, save only non-default values (cleaner, more readable)
                If False, save all values (complete, verbose)
    """
    # Convert config to dictionary
    if minimal:
        config_dict = _config_to_minimal_dict(config)
    else:
        config_dict = config.to_dict()

    # Write to YAML file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def _config_to_minimal_dict(config: PRISMConfig) -> dict[str, Any]:
    """
    Convert config to dictionary with only non-default values.

    Compares config values against defaults from dataclass definitions.
    Only includes values that differ from defaults.

    Args:
        config: PRISMConfig object

    Returns:
        Minimal dictionary representation
    """
    from dataclasses import fields

    result: dict[str, Any] = {}

    # Top-level parameters (always include name, comment)
    if config.name is not None:
        result["name"] = config.name
    if config.comment:
        result["comment"] = config.comment
    if config.log_dir != "runs":
        result["log_dir"] = config.log_dir
    if not config.save_data:
        result["save_data"] = config.save_data
    if config.checkpoint is not None:
        result["checkpoint"] = config.checkpoint

    # Helper function to extract non-default fields
    def get_non_defaults(obj: Any, default_obj: Any) -> dict[str, Any]:
        non_defaults: dict[str, Any] = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            default_value = getattr(default_obj, field.name)
            if value != default_value:
                non_defaults[field.name] = value
        return non_defaults

    # Process each config section
    image_defaults = ImageConfig()
    telescope_defaults = TelescopeConfig()
    model_defaults = ModelConfig()
    training_defaults = TrainingConfig()
    physics_defaults = PhysicsConfig()
    point_source_defaults = PointSourceConfig()

    image_non_defaults = get_non_defaults(config.image, image_defaults)
    if image_non_defaults:
        result["image"] = image_non_defaults

    telescope_non_defaults = get_non_defaults(config.telescope, telescope_defaults)
    if telescope_non_defaults:
        result["telescope"] = telescope_non_defaults

    model_non_defaults = get_non_defaults(config.model, model_defaults)
    if model_non_defaults:
        result["model"] = model_non_defaults

    training_non_defaults = get_non_defaults(config.training, training_defaults)
    if training_non_defaults:
        result["training"] = training_non_defaults

    physics_non_defaults = get_non_defaults(config.physics, physics_defaults)
    if physics_non_defaults:
        result["physics"] = physics_non_defaults

    point_source_non_defaults = get_non_defaults(config.point_source, point_source_defaults)
    if point_source_non_defaults:
        result["point_source"] = point_source_non_defaults

    # MoPIE config (only if present and non-default)
    if config.mopie is not None:
        mopie_defaults = MoPIEConfig()
        mopie_non_defaults = get_non_defaults(config.mopie, mopie_defaults)
        if mopie_non_defaults:
            result["mopie"] = mopie_non_defaults

    return result


def config_to_args(config: PRISMConfig) -> argparse.Namespace:
    """Convert PRISMConfig to argparse.Namespace for backward compatibility.

    Args:
        config: PRISMConfig object

    Returns:
        argparse.Namespace with all parameters
    """
    args_dict: dict[str, Any] = {}

    # Top-level experiment parameters
    args_dict["name"] = config.name
    args_dict["comment"] = config.comment
    args_dict["log_dir"] = config.log_dir
    args_dict["save_data"] = config.save_data
    args_dict["checkpoint"] = config.checkpoint

    # Image parameters
    args_dict["input"] = config.image.input
    args_dict["obj_size"] = config.image.obj_size
    args_dict["image_size"] = config.image.image_size
    args_dict["invert_image"] = config.image.invert
    args_dict["crop_obj"] = config.image.crop

    # Telescope parameters
    args_dict["sample_diameter"] = config.telescope.sample_diameter
    args_dict["sample_shape"] = config.telescope.sample_shape
    args_dict["sample_length"] = config.telescope.sample_length
    args_dict["samples_per_line_meas"] = config.telescope.samples_per_line_meas
    args_dict["samples_per_line_rec"] = config.telescope.samples_per_line_rec
    args_dict["line_angle"] = config.telescope.line_angle
    args_dict["roi_diameter"] = config.telescope.roi_diameter
    args_dict["samples_r_cutoff"] = config.telescope.samples_r_cutoff
    args_dict["roi_shape"] = config.telescope.roi_shape
    args_dict["sample_sort"] = config.telescope.sample_sort
    args_dict["n_samples"] = config.telescope.n_samples
    args_dict["n_angs"] = config.telescope.n_angs
    args_dict["star_sample"] = config.telescope.star_sample
    args_dict["fermat_sample"] = config.telescope.fermat_sample
    args_dict["snr"] = config.telescope.snr
    args_dict["blur_image"] = config.telescope.blur
    args_dict["propagator_method"] = config.telescope.propagator_method

    # Model parameters
    args_dict["use_bn"] = config.model.use_bn
    args_dict["output_activation"] = config.model.output_activation
    args_dict["use_leaky"] = config.model.use_leaky
    args_dict["middle_activation"] = config.model.middle_activation
    args_dict["complex_data"] = config.model.complex_data

    # Training parameters
    args_dict["n_epochs"] = config.training.n_epochs
    args_dict["max_epochs"] = config.training.max_epochs
    args_dict["n_epochs_init"] = config.training.n_epochs_init
    args_dict["max_epochs_init"] = config.training.max_epochs_init
    args_dict["initialization_target"] = config.training.initialization_target
    args_dict["loss_type"] = config.training.loss_type
    args_dict["new_weight"] = config.training.new_weight
    args_dict["f_weight"] = config.training.f_weight
    args_dict["lr"] = config.training.lr
    args_dict["loss_th"] = config.training.loss_threshold
    args_dict["use_amsgrad"] = config.training.use_amsgrad
    args_dict["device_num"] = config.training.device_num
    args_dict["use_cuda"] = config.training.use_cuda

    # Physics parameters
    args_dict["wavelength"] = config.physics.wavelength
    args_dict["dxf"] = config.physics.dxf
    args_dict["obj_diameter"] = config.physics.obj_diameter
    args_dict["obj_distance"] = config.physics.obj_distance
    args_dict["obj_name"] = config.physics.obj_name

    # Point source parameters
    args_dict["is_point_source"] = config.point_source.is_point_source
    args_dict["point_source_diameter"] = config.point_source.diameter
    args_dict["point_source_spacing"] = config.point_source.spacing
    args_dict["point_source_number"] = config.point_source.number

    # MoPIE parameters (if present)
    if config.mopie is not None:
        args_dict["lr_obj"] = config.mopie.lr_obj
        args_dict["lr_probe"] = config.mopie.lr_probe
        args_dict["grad_mopie"] = config.mopie.grad_mopie
        args_dict["fix_probe"] = config.mopie.fix_probe
        args_dict["parallel_update"] = config.mopie.parallel_update
        args_dict["plot_every"] = config.mopie.plot_every
        args_dict["single_sample"] = config.mopie.single_sample
        args_dict["rand_perm"] = config.mopie.rand_perm
        args_dict["load_config_only"] = config.mopie.load_config_only

    return argparse.Namespace(**args_dict)


def args_to_config(args: argparse.Namespace) -> PRISMConfig:
    """Convert argparse.Namespace to PRISMConfig.

    Args:
        args: argparse.Namespace from argument parser

    Returns:
        PRISMConfig object
    """
    # Image config
    image = ImageConfig(
        input=getattr(args, "input", None),
        obj_size=getattr(args, "obj_size", None),
        image_size=getattr(args, "image_size", 1024),
        invert=getattr(args, "invert_image", False),
        crop=getattr(args, "crop_obj", False),
    )

    # Telescope config
    telescope = TelescopeConfig(
        sample_diameter=getattr(args, "sample_diameter", None),
        sample_shape=getattr(args, "sample_shape", "circle"),
        sample_length=getattr(args, "sample_length", 0),
        samples_per_line_meas=getattr(args, "samples_per_line_meas", None),
        samples_per_line_rec=getattr(args, "samples_per_line_rec", None),
        line_angle=getattr(args, "line_angle", None),
        roi_diameter=getattr(args, "roi_diameter", None),
        samples_r_cutoff=getattr(args, "samples_r_cutoff", None),
        roi_shape=getattr(args, "roi_shape", "circle"),
        sample_sort=getattr(args, "sample_sort", "center"),
        n_samples=getattr(args, "n_samples", 200),
        n_angs=getattr(args, "n_angs", 4),
        star_sample=getattr(args, "star_sample", False),
        fermat_sample=getattr(args, "fermat_sample", False),
        snr=getattr(args, "snr", None),
        blur=getattr(args, "blur_image", False),
        propagator_method=getattr(args, "propagator_method", None),
    )

    # Model config
    model = ModelConfig(
        use_bn=getattr(args, "use_bn", True),
        output_activation=getattr(args, "output_activation", "none"),
        use_leaky=getattr(args, "use_leaky", True),
        middle_activation=getattr(args, "middle_activation", "sigmoid"),
        complex_data=getattr(args, "complex_data", False),
    )

    # Training config
    training = TrainingConfig(
        n_epochs=getattr(args, "n_epochs", 1000),
        max_epochs=getattr(args, "max_epochs", 1),
        n_epochs_init=getattr(args, "n_epochs_init", 100),
        max_epochs_init=getattr(args, "max_epochs_init", 100),
        initialization_target=getattr(args, "initialization_target", "circle"),
        loss_type=getattr(args, "loss_type", "l1"),
        new_weight=getattr(args, "new_weight", 1.0),
        f_weight=getattr(args, "f_weight", 1e-4),
        lr=getattr(args, "lr", 1e-3),
        loss_threshold=getattr(args, "loss_th", 1e-3),
        use_amsgrad=getattr(args, "use_amsgrad", False),
        device_num=getattr(args, "device_num", 0),
        use_cuda=getattr(args, "use_cuda", True),
    )

    # Physics config
    physics = PhysicsConfig(
        wavelength=getattr(args, "wavelength", None),
        dxf=getattr(args, "dxf", 1e-2),
        obj_diameter=getattr(args, "obj_diameter", None),
        obj_distance=getattr(args, "obj_distance", None),
        obj_name=getattr(args, "obj_name", "europa"),
    )

    # Point source config
    point_source = PointSourceConfig(
        is_point_source=getattr(args, "is_point_source", False),
        diameter=getattr(args, "point_source_diameter", 3.0),
        spacing=getattr(args, "point_source_spacing", 5.0),
        number=getattr(args, "point_source_number", 4),
    )

    # MoPIE config (optional)
    mopie = None
    if hasattr(args, "lr_obj"):
        mopie = MoPIEConfig(
            lr_obj=getattr(args, "lr_obj", 1.0),
            lr_probe=getattr(args, "lr_probe", 1.0),
            grad_mopie=getattr(args, "grad_mopie", False),
            fix_probe=getattr(args, "fix_probe", True),
            parallel_update=getattr(args, "parallel_update", True),
            plot_every=getattr(args, "plot_every", 1),
            single_sample=getattr(args, "single_sample", False),
            rand_perm=getattr(args, "rand_perm", False),
            load_config_only=getattr(args, "load_config_only", False),
        )

    # Create master config
    config = PRISMConfig(
        name=getattr(args, "name", None),
        comment=getattr(args, "comment", ""),
        log_dir=getattr(args, "log_dir", "runs"),
        save_data=getattr(args, "save_data", True),
        checkpoint=getattr(args, "checkpoint", None),
        image=image,
        telescope=telescope,
        model=model,
        training=training,
        physics=physics,
        point_source=point_source,
        mopie=mopie,
    )

    return config


def merge_config_with_args(
    config: PRISMConfig, cli_args: argparse.Namespace, cli_provided_args: Optional[set] = None
) -> argparse.Namespace:
    """Merge config with CLI arguments (CLI args take precedence).

    Args:
        config: Base configuration from YAML
        cli_args: Command-line arguments from argparse
        cli_provided_args: Set of argument names that were explicitly provided on CLI.
                          If None, will attempt to detect from sys.argv

    Returns:
        Merged argparse.Namespace
    """
    import sys

    # Convert config to args namespace
    merged_args = config_to_args(config)

    # Detect which args were explicitly provided on CLI
    if cli_provided_args is None:
        cli_provided_args = set()
        # Parse sys.argv to find which arguments were actually provided
        for arg in sys.argv[1:]:
            if arg.startswith("--"):
                # Remove leading dashes and get the argument name
                arg_name = arg.lstrip("-").split("=")[0]
                # Convert kebab-case to snake_case (e.g., --n-samples -> n_samples)
                arg_name = arg_name.replace("-", "_")
                cli_provided_args.add(arg_name)
            elif arg.startswith("-") and len(arg) == 2:
                # Short form argument (we don't use these much, but handle them)
                pass

    # Override with CLI args (only if explicitly provided)
    cli_dict = vars(cli_args)
    merged_dict = vars(merged_args)

    for key, value in cli_dict.items():
        # Skip the config parameter itself
        if key == "config":
            continue

        # Only override if this argument was explicitly provided on command line
        if key in cli_provided_args:
            merged_dict[key] = value

    return argparse.Namespace(**merged_dict)

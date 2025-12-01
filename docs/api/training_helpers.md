# prism.utils.training_helpers

Training helper utilities for PRISM main scripts.

This module provides shared utility functions used by both main.py and main_epie.py
to reduce code duplication and improve maintainability.

Functions:
    load_config_with_checkpoint_fallback: Load config with automatic checkpoint fallback
    setup_device: Configure PyTorch device (CPU/CUDA)
    configure_line_sampling: Calculate line sampling parameters
    validate_training_args: Validate training arguments

Created: Session 5 - Extract Shared Utilities

## Classes

## Functions

### configure_line_sampling

```python
configure_line_sampling(args: argparse.Namespace, sample_length: int, sample_diameter: int) -> Tuple[int, float]
```

Calculate line sampling parameters for measurement and reconstruction.

For point samples (sample_length=0), no subsampling is needed.
For line samples, determines how many points to measure along each line
and calculates the SNR penalty from subsampling.

Args:
    args: Argument namespace with samples_per_line_meas, samples_per_line_rec attributes
    sample_length: Length of line samples in pixels (0 for point samples)
    sample_diameter: Diameter of telescope aperture in pixels

Returns:
    Tuple of (samples_per_line_meas, dsnr):
        - samples_per_line_meas: Number of measurement points along each line
        - dsnr: SNR penalty in dB from line subsampling

Example:
    >>> samples_per_line_meas, dsnr = configure_line_sampling(args, 64, 32)
    >>> logger.info(f"Line sampling: {samples_per_line_meas} points, dSNR: {dsnr:.2f} dB")

Notes:
    - Updates args.samples_per_line_meas, args.samples_per_line_rec, and args.dsnr in place
    - Ensures odd number of samples for symmetric patterns
    - Auto-calculates spacing by half aperture diameter if samples_per_line_meas < 1

### load_config_with_checkpoint_fallback

```python
load_config_with_checkpoint_fallback(args: argparse.Namespace, load_config_func: Callable[[str], Dict[str, Any]], merge_config_func: Callable[[Dict[str, Any], argparse.Namespace], argparse.Namespace]) -> argparse.Namespace
```

Load configuration file with automatic checkpoint fallback.

If resuming from a checkpoint and no explicit config is provided, automatically
loads the config.yaml from the checkpoint directory. Otherwise loads the specified
config file.

Args:
    args: Argument namespace with checkpoint, config, and log_dir attributes
    load_config_func: Function to load config file (e.g., load_config from config_utils)
    merge_config_func: Function to merge config with args (e.g., merge_config_with_args)

Returns:
    Updated argument namespace with config values merged

Example:
    >>> args = load_config_with_checkpoint_fallback(
    ...     args, load_config, merge_config_with_args
    ... )

### setup_device

```python
setup_device(args: argparse.Namespace) -> torch.device
```

Configure PyTorch device (CPU or CUDA GPU).

Checks if CUDA is available and requested, enables cudnn benchmark if using GPU,
and returns the appropriate torch.device.

Args:
    args: Argument namespace with use_cuda and device_num attributes

Returns:
    torch.device configured for CPU or CUDA

Example:
    >>> device = setup_device(args)
    >>> model = model.to(device)

### validate_training_args

```python
validate_training_args(args: argparse.Namespace) -> None
```

Validate training arguments for common issues.

Checks for invalid parameter values and raises ValueError with helpful messages
if validation fails.

Args:
    args: Argument namespace to validate

Raises:
    ValueError: If any argument fails validation

Example:
    >>> validate_training_args(args)  # Raises ValueError if args invalid

Validation checks:
    - Learning rate must be positive
    - Loss threshold must be positive
    - Number of samples must be positive
    - Max epochs must be positive
    - Sample diameter must be positive
    - ROI diameter must be >= sample diameter

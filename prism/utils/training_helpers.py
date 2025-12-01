"""
Training helper utilities for SPIDS main scripts.

This module provides shared utility functions used by both main.py and main_epie.py
to reduce code duplication and improve maintainability.

Functions:
    load_config_with_checkpoint_fallback: Load config with automatic checkpoint fallback
    setup_device: Configure PyTorch device (CPU/CUDA)
    configure_line_sampling: Calculate line sampling parameters
    validate_training_args: Validate training arguments

Created: Session 5 - Extract Shared Utilities
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, Tuple

import torch
import torch.backends.cudnn as cudnn
from loguru import logger


def load_config_with_checkpoint_fallback(
    args: argparse.Namespace,
    load_config_func: Callable[[str], Dict[str, Any]],
    merge_config_func: Callable[[Dict[str, Any], argparse.Namespace], argparse.Namespace],
) -> argparse.Namespace:
    """
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
    """
    # Auto-load config from checkpoint directory if resuming and no explicit config provided
    if args.checkpoint is not None and args.config is None:
        checkpoint_config_path = os.path.join(args.log_dir, args.checkpoint, "config.yaml")
        if os.path.exists(checkpoint_config_path):
            logger.info(f"Auto-loading configuration from checkpoint: {checkpoint_config_path}")
            args.config = checkpoint_config_path

    # Load config file if provided and merge with CLI args
    if args.config is not None:
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config_func(args.config)
        args = merge_config_func(config, args)
        logger.debug("Configuration loaded and merged with CLI arguments")

    return args


def setup_device(args: argparse.Namespace) -> torch.device:
    """
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
    """
    # Check if CUDA is available and requested
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    if args.use_cuda:
        device = torch.device(f"cuda:{args.device_num}")
        cudnn.benchmark = True  # Enable cudnn autotuner for performance
        logger.info(f"Device: CUDA (GPU {args.device_num})")
    else:
        device = torch.device("cpu")
        logger.info("Device: CPU")

    return device


def configure_line_sampling(
    args: argparse.Namespace, sample_length: int, sample_diameter: int
) -> Tuple[int, float]:
    """
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
    """
    if sample_length == 0:
        # Point sampling: no line subsampling needed
        samples_per_line_meas = 0
        args.samples_per_line_rec = 0
        dsnr = 0.0
    else:
        # Line sampling: determine how many points to measure along each line
        if args.samples_per_line_meas is None:
            samples_per_line_meas = sample_length + 1  # Full line measurement
        elif args.samples_per_line_meas < 1:
            # Auto-calculate: space samples by half the aperture diameter (Nyquist-like)
            samples_per_line_meas = int(sample_length // (sample_diameter / 2)) + 1
        else:
            samples_per_line_meas = args.samples_per_line_meas

        # Ensure odd number of samples for symmetric patterns
        if samples_per_line_meas % 2 == 0:
            samples_per_line_meas += 1

        # Calculate down-sampling SNR penalty (dsnr)
        # Subsampling reduces effective SNR - track this for noise modeling
        dsnr = 10 * torch.log10(torch.tensor((sample_length + 1) / samples_per_line_meas)).item()

    # Update args in place
    args.samples_per_line_meas = int(samples_per_line_meas)
    args.dsnr = dsnr

    # Configure reconstruction subsampling (can differ from measurement)
    if sample_length > 0:
        if args.samples_per_line_rec is None:
            args.samples_per_line_rec = sample_length // (sample_diameter / 2) + 1
        elif args.samples_per_line_rec < 1:
            args.samples_per_line_rec = samples_per_line_meas
        if args.samples_per_line_rec % 2 == 0 and args.samples_per_line_rec > 0:
            args.samples_per_line_rec += 1

    return samples_per_line_meas, dsnr


def validate_training_args(args: argparse.Namespace) -> None:
    """
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
    """
    if hasattr(args, "lr") and args.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got: {args.lr}")

    if hasattr(args, "loss_th") and args.loss_th <= 0:
        raise ValueError(f"Loss threshold must be positive, got: {args.loss_th}")

    if hasattr(args, "n_samples") and args.n_samples <= 0:
        raise ValueError(f"Number of samples must be positive, got: {args.n_samples}")

    if hasattr(args, "max_epochs") and args.max_epochs <= 0:
        raise ValueError(f"Max epochs must be positive, got: {args.max_epochs}")

    if hasattr(args, "n_epochs") and args.n_epochs <= 0:
        raise ValueError(f"Number of epochs must be positive, got: {args.n_epochs}")

    if hasattr(args, "sample_diameter") and args.sample_diameter <= 0:
        raise ValueError(f"Sample diameter must be positive, got: {args.sample_diameter}")

    if hasattr(args, "roi_diameter") and hasattr(args, "sample_diameter"):
        if args.roi_diameter < args.sample_diameter:
            raise ValueError(
                f"ROI diameter ({args.roi_diameter}) must be >= sample diameter ({args.sample_diameter})"
            )

    logger.debug("Training arguments validated successfully")

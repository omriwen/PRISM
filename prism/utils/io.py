"""
Module: spids.utils.io
Purpose: Data I/O and experiment configuration management
Dependencies: os, torch
Main Functions:
    - save_args(args, path): Save experiment arguments to disk in both PyTorch and text formats
    - save_checkpoint(state, path, is_best): Save training checkpoint
    - load_checkpoint(path): Load training checkpoint
    - export_results(checkpoint, output_dir): Export experiment results

Description:
    This module handles data I/O operations for SPIDS experiments.
    It provides utilities to save and load experiment configurations,
    checkpoints, and results. The module ensures reproducibility by
    storing complete experiment parameters in both machine-readable
    (.pt) and human-readable (.txt) formats.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Union, cast

import torch

from prism.types import PathLike


def save_args(args: Union[object, Dict[str, Any]], path: PathLike) -> None:
    """
    Save experiment arguments to disk.

    Args:
        args: Argument namespace or dict containing experiment parameters
        path: Directory path to save arguments

    Saves:
        - args.pt: PyTorch serialized arguments
        - args.txt: Human-readable text format
    """
    args_dict: Dict[str, Any] = (
        vars(args) if hasattr(args, "__dict__") else cast(Dict[str, Any], args)
    )
    torch.save(args_dict, os.path.join(path, "args.pt"))
    with open(os.path.join(path, "args.txt"), "w") as f:
        for key, value in args_dict.items():
            f.write("{}: {}\n".format(key, value))


def save_checkpoint(state: Dict[str, Any], path: PathLike, is_best: bool = False) -> None:
    """
    Save training checkpoint.

    Args:
        state: Dictionary containing model state, optimizer state, metrics, etc.
        path: Path to save checkpoint
        is_best: Whether this is the best checkpoint so far

    Note:
        This is a placeholder for future implementation.
        Currently, checkpoints are saved directly in main.py.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

    if is_best:
        best_path = path.parent / "best_checkpoint.pt"
        torch.save(state, best_path)


def load_checkpoint(path: PathLike) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint file

    Returns:
        Dictionary containing checkpoint data

    Note:
        This is a placeholder for future implementation.
        Currently, checkpoints are loaded directly in main.py.
    """
    result = torch.load(path)
    return cast(Dict[str, Any], result)


def export_results(checkpoint: Dict[str, Any], output_dir: PathLike, format: str = "numpy") -> None:
    """
    Export experiment results to various formats.

    Args:
        checkpoint: Checkpoint dictionary containing results
        output_dir: Directory to save exported results
        format: Export format ('numpy', 'image', 'mat')

    Note:
        This is a placeholder for future implementation.
        Will support exporting reconstructions, metrics, and visualizations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder for future implementation
    # - Export reconstruction as numpy array
    # - Export metrics as CSV
    # - Export visualizations as images
    pass

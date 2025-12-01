# prism.utils.io

Module: prism.utils.io
Purpose: Data I/O and experiment configuration management
Dependencies: os, torch
Main Functions:
    - save_args(args, path): Save experiment arguments to disk in both PyTorch and text formats
    - save_checkpoint(state, path, is_best): Save training checkpoint
    - load_checkpoint(path): Load training checkpoint
    - export_results(checkpoint, output_dir): Export experiment results

Description:
    This module handles data I/O operations for PRISM experiments.
    It provides utilities to save and load experiment configurations,
    checkpoints, and results. The module ensures reproducibility by
    storing complete experiment parameters in both machine-readable
    (.pt) and human-readable (.txt) formats.

## Classes

## Functions

### export_results

```python
export_results(checkpoint: Dict[str, Any], output_dir: Union[str, ForwardRef('Path')], format: str = 'numpy') -> None
```

Export experiment results to various formats.

Args:
    checkpoint: Checkpoint dictionary containing results
    output_dir: Directory to save exported results
    format: Export format ('numpy', 'image', 'mat')

Note:
    This is a placeholder for future implementation.
    Will support exporting reconstructions, metrics, and visualizations.

### load_checkpoint

```python
load_checkpoint(path: Union[str, ForwardRef('Path')]) -> Dict[str, Any]
```

Load training checkpoint.

Args:
    path: Path to checkpoint file

Returns:
    Dictionary containing checkpoint data

Note:
    This is a placeholder for future implementation.
    Currently, checkpoints are loaded directly in main.py.

### save_args

```python
save_args(args: Union[object, Dict[str, Any]], path: Union[str, ForwardRef('Path')]) -> None
```

Save experiment arguments to disk.

Args:
    args: Argument namespace or dict containing experiment parameters
    path: Directory path to save arguments

Saves:
    - args.pt: PyTorch serialized arguments
    - args.txt: Human-readable text format

### save_checkpoint

```python
save_checkpoint(state: Dict[str, Any], path: Union[str, ForwardRef('Path')], is_best: bool = False) -> None
```

Save training checkpoint.

Args:
    state: Dictionary containing model state, optimizer state, metrics, etc.
    path: Path to save checkpoint
    is_best: Whether this is the best checkpoint so far

Note:
    This is a placeholder for future implementation.
    Currently, checkpoints are saved directly in main.py.

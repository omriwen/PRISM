# prism.utils.checkpointing

Efficient checkpoint management for training.

This module provides utilities for managing model checkpoints during training,
including automatic cleanup of old checkpoints and tracking of best models.

## Classes

### CheckpointManager

```python
CheckpointManager(checkpoint_dir: Union[str, pathlib.Path], max_checkpoints: int = 3, max_best_checkpoints: int = 3, mode: str = 'min') -> None
```

Manage model checkpoints with automatic cleanup.

This class handles saving, loading, and managing checkpoints during
training, ensuring that only a limited number of checkpoints are kept
to save disk space while preserving the best models.

Parameters
----------
checkpoint_dir : Union[str, Path]
    Directory to store checkpoints
max_checkpoints : int, optional
    Maximum number of regular checkpoints to keep (default: 3)
max_best_checkpoints : int, optional
    Maximum number of best checkpoints to keep (default: 3)
mode : str, optional
    'min' or 'max' for determining best checkpoints (default: 'min')

Attributes
----------
checkpoint_dir : Path
    Directory for checkpoints
max_checkpoints : int
    Maximum regular checkpoints to keep
max_best_checkpoints : int
    Maximum best checkpoints to keep
mode : str
    Mode for best checkpoint selection
checkpoints : List[Dict[str, Any]]
    List of checkpoint metadata
best_checkpoints : List[Dict[str, Any]]
    List of best checkpoint metadata

Examples
--------
>>> manager = CheckpointManager('runs/experiment/checkpoints', max_checkpoints=3)
>>> # During training
>>> for epoch in range(num_epochs):
...     # Training code
...     state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
...     manager.save(state, epoch=epoch, loss=loss.item())
>>> # Load best checkpoint
>>> best_state = manager.load_best()

#### Methods

##### `__init__`

Initialize checkpoint manager.

##### `cleanup_all`

Remove all checkpoints from disk.

##### `get_best_metric`

Get the best metric value.

Returns
-------
Optional[float]
    Best metric value, or None if no checkpoints with metrics exist

##### `load`

Load a checkpoint from file.

Parameters
----------
checkpoint_path : Union[str, Path]
    Path to checkpoint file

Returns
-------
Dict[str, Any]
    Loaded checkpoint state

Examples
--------
>>> state = manager.load('checkpoint_epoch_0100.pt')
>>> model.load_state_dict(state['model'])

##### `load_best`

Load the best checkpoint according to the metric.

Returns
-------
Optional[Dict[str, Any]]
    Best checkpoint state, or None if no best checkpoints exist

##### `load_latest`

Load the most recent checkpoint.

Returns
-------
Optional[Dict[str, Any]]
    Latest checkpoint state, or None if no checkpoints exist

##### `save`

Save a checkpoint and manage old checkpoints.

Parameters
----------
state : Dict[str, Any]
    State dictionary to save (model, optimizer, etc.)
epoch : int, optional
    Current epoch number
metric : float, optional
    Metric value for this checkpoint (e.g., loss, accuracy)
is_best : bool, optional
    Whether to mark this as a best checkpoint
**metadata : Any
    Additional metadata to store

Returns
-------
Path
    Path to saved checkpoint

Examples
--------
>>> manager.save(
...     {'model': model.state_dict(), 'epoch': epoch},
...     epoch=epoch,
...     metric=val_loss,
...     learning_rate=lr
... )

##### `save_best_if_improved`

Save checkpoint only if it's better than existing best checkpoints.

Parameters
----------
state : Dict[str, Any]
    State dictionary to save
metric : float
    Metric value to compare
epoch : int, optional
    Current epoch number
**metadata : Any
    Additional metadata

Returns
-------
Optional[Path]
    Path to saved checkpoint if it was saved, None otherwise

## Functions

### save_checkpoint_with_cleanup

```python
save_checkpoint_with_cleanup(state: Dict[str, Any], checkpoint_dir: Union[str, pathlib.Path], filename: str = 'checkpoint.pt', max_to_keep: int = 3) -> pathlib.Path
```

Simple function to save checkpoint with automatic cleanup.

This is a simplified interface for checkpoint management without
the full CheckpointManager class.

Parameters
----------
state : Dict[str, Any]
    State dictionary to save
checkpoint_dir : Union[str, Path]
    Directory for checkpoints
filename : str, optional
    Checkpoint filename (default: 'checkpoint.pt')
max_to_keep : int, optional
    Maximum checkpoints to keep (default: 3)

Returns
-------
Path
    Path to saved checkpoint

Examples
--------
>>> save_checkpoint_with_cleanup(
...     {'model': model.state_dict(), 'epoch': epoch},
...     'runs/experiment/checkpoints',
...     filename=f'checkpoint_epoch_{epoch:04d}.pt'
... )

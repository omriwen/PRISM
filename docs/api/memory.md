# prism.utils.memory

Memory management utilities for PyTorch tensors.

This module provides utilities for tracking and managing GPU/CPU memory
during training to prevent memory leaks and optimize memory usage.

## Classes

### MemoryTracker

```python
MemoryTracker(device: Optional[torch.device] = None, track_cpu: bool = False) -> None
```

Track memory usage during training.

This class monitors GPU memory allocation and reserved memory
over time, useful for identifying memory leaks and optimizing
memory usage.

Parameters
----------
device : torch.device, optional
    Device to track (default: current CUDA device)
track_cpu : bool, optional
    Whether to track CPU memory as well (default: False)

Attributes
----------
history : List[Dict[str, Any]]
    History of memory measurements
device : torch.device
    Device being tracked
track_cpu : bool
    Whether CPU memory is being tracked

Examples
--------
>>> tracker = MemoryTracker()
>>> for epoch in range(num_epochs):
...     # Training code
...     tracker.log(epoch=epoch, phase='train')
>>> tracker.print_summary()
>>> tracker.save_history('memory_log.pt')

#### Methods

##### `__init__`

Initialize memory tracker.

##### `get_current_usage`

Get current memory usage without logging.

Returns
-------
Dict[str, float]
    Current memory statistics in MB

##### `load_history`

Load memory history from file.

Parameters
----------
path : str
    Path to load history from

##### `log`

Log current memory usage.

Parameters
----------
**metadata : Any
    Additional metadata to store with this measurement
    (e.g., epoch, batch, phase)

Returns
-------
Dict[str, Any]
    Current memory statistics

##### `print_summary`

Print summary of memory usage over time.

##### `reset_peak_stats`

Reset peak memory statistics.

##### `save_history`

Save memory history to file.

Parameters
----------
path : str
    Path to save history (as PyTorch file)

## Functions

### cleanup_tensors

```python
cleanup_tensors(*tensors: torch.Tensor) -> None
```

Explicitly cleanup tensors and free GPU memory.

This function deletes tensor references and triggers garbage collection
and CUDA cache clearing to ensure memory is released.

Parameters
----------
*tensors : torch.Tensor
    Tensors to cleanup

Examples
--------
>>> large_tensor = torch.randn(10000, 10000).cuda()
>>> # Do some computation
>>> cleanup_tensors(large_tensor)

Notes
-----
This is particularly important for large intermediate tensors
that are no longer needed. PyTorch's autograd can keep references
to tensors longer than necessary.

### get_memory_summary

```python
get_memory_summary() -> str
```

Get formatted string with current memory usage.

Returns
-------
str
    Formatted memory usage summary

Examples
--------
>>> print(get_memory_summary())
GPU: 1234.5 MB allocated, 2000.0 MB reserved

### print_memory_usage

```python
print_memory_usage(prefix: str = '') -> None
```

Print current memory usage with optional prefix.

Parameters
----------
prefix : str, optional
    Prefix for the output message

Examples
--------
>>> print_memory_usage("After training epoch 1")

### temporary_tensor

```python
temporary_tensor(*tensors: torch.Tensor)
```

Context manager for temporary tensors with automatic cleanup.

Parameters
----------
*tensors : torch.Tensor
    Tensors to manage

Yields
------
tuple or torch.Tensor
    The input tensors

Examples
--------
>>> with temporary_tensor(torch.randn(1000, 1000)) as temp:
...     result = temp.sum()
# temp is automatically cleaned up here

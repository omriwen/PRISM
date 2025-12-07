# prism.training.parallel

Parallel training utilities for PRISM.

This module provides utilities for data parallelism across multiple GPUs
and distributed training support. It includes functions to automatically
detect available hardware and configure models for optimal parallel execution.

Main Functions:
    - setup_parallel_model: Configure model for multi-GPU training
    - get_device_count: Get number of available CUDA devices
    - is_parallel_available: Check if parallel training is available
    - parallelize_if_available: Automatically parallelize model if multiple GPUs exist
    - get_recommended_batch_size: Suggest batch size based on available GPUs

Example:
    >>> import torch.nn as nn
    >>> from prism.training.parallel import setup_parallel_model
    >>>
    >>> model = MyModel()
    >>> model = setup_parallel_model(model)  # Automatically uses all GPUs
    >>> # Or specify specific devices:
    >>> model = setup_parallel_model(model, device_ids=[0, 1])

## Classes

## Functions

### get_device_count

```python
get_device_count() -> int
```

Get the number of available CUDA devices.

Returns:
    int: Number of CUDA devices available, or 0 if CUDA is not available

Example:
    >>> from prism.training.parallel import get_device_count
    >>> n_gpus = get_device_count()
    >>> print(f"Found {n_gpus} GPUs")

### get_gpu_memory_info

```python
get_gpu_memory_info() -> List[dict]
```

Get memory information for all available GPUs.

Returns:
    List[dict]: List of dictionaries with memory info for each GPU
               Keys: 'device_id', 'name', 'total_memory', 'allocated', 'cached'

Example:
    >>> from prism.training.parallel import get_gpu_memory_info
    >>> for gpu_info in get_gpu_memory_info():
    ...     print(f"GPU {gpu_info['device_id']}: {gpu_info['allocated']/1e9:.2f}GB used")

### get_recommended_batch_size

```python
get_recommended_batch_size(base_batch_size: int = 1, scale_with_gpus: bool = True) -> int
```

Get recommended batch size based on available GPUs.

When using DataParallel, you can often increase batch size proportionally
to the number of GPUs for better GPU utilization.

Parameters:
    base_batch_size: The batch size for a single GPU
    scale_with_gpus: If True, scale batch size by number of GPUs

Returns:
    int: Recommended batch size

Example:
    >>> from prism.training.parallel import get_recommended_batch_size
    >>> # With 4 GPUs and base_batch_size=8
    >>> batch_size = get_recommended_batch_size(8)  # Returns 32

### is_parallel_available

```python
is_parallel_available() -> bool
```

Check if parallel training is available (i.e., multiple GPUs exist).

Returns:
    bool: True if multiple CUDA devices are available, False otherwise

Example:
    >>> from prism.training.parallel import is_parallel_available
    >>> if is_parallel_available():
    ...     print("Multi-GPU training is available!")

### parallelize_if_available

```python
parallelize_if_available(model: torch.nn.modules.module.Module, min_gpus: int = 2, **kwargs) -> torch.nn.modules.module.Module
```

Automatically parallelize model if multiple GPUs are available.

This is a convenience function that only parallelizes if the number
of available GPUs meets or exceeds min_gpus.

Parameters:
    model: The PyTorch model to potentially parallelize
    min_gpus: Minimum number of GPUs required for parallelization
    **kwargs: Additional arguments passed to setup_parallel_model

Returns:
    nn.Module: The potentially parallelized model

Example:
    >>> model = ProgressiveDecoder()
    >>> # Only parallelize if 2+ GPUs available
    >>> model = parallelize_if_available(model, min_gpus=2)

### print_gpu_memory_info

```python
print_gpu_memory_info()
```

Print formatted GPU memory information for all available GPUs.

Example:
    >>> from prism.training.parallel import print_gpu_memory_info
    >>> print_gpu_memory_info()
    GPU 0 (NVIDIA A100): 12.5 GB / 40.0 GB (allocated), 15.2 GB (cached)
    GPU 1 (NVIDIA A100): 8.3 GB / 40.0 GB (allocated), 10.1 GB (cached)

### setup_parallel_model

```python
setup_parallel_model(model: torch.nn.modules.module.Module, device_ids: Optional[List[int]] = None, output_device: Optional[int] = None, distributed: bool = False, find_unused_parameters: bool = False) -> torch.nn.modules.module.Module
```

Setup model for parallel training across multiple GPUs.

This function wraps a model with DataParallel or DistributedDataParallel
if multiple GPUs are available. If only one GPU or no GPUs are available,
the model is returned unchanged.

Parameters:
    model: The PyTorch model to parallelize
    device_ids: List of GPU device IDs to use. If None, uses all available GPUs
    output_device: GPU device ID for output. If None, uses device_ids[0]
    distributed: If True, use DistributedDataParallel instead of DataParallel.
                Requires torch.distributed to be initialized.
    find_unused_parameters: Only used with DistributedDataParallel. Set to True
                           if your model has parameters that don't receive gradients.

Returns:
    nn.Module: The parallelized model (or original if no parallelization needed)

Example:
    >>> model = ProgressiveDecoder()
    >>> model = setup_parallel_model(model)
    >>> # Model is now ready for multi-GPU training

Notes:
    - DataParallel is easier to use but may be slower for some workloads
    - DistributedDataParallel is faster but requires more setup
    - If model is already wrapped in DataParallel/DistributedDataParallel,
      it will be returned unchanged

### unwrap_parallel_model

```python
unwrap_parallel_model(model: torch.nn.modules.module.Module) -> torch.nn.modules.module.Module
```

Unwrap a model from DataParallel or DistributedDataParallel wrapper.

Useful when saving checkpoints or accessing the underlying model.

Parameters:
    model: The potentially wrapped model

Returns:
    nn.Module: The unwrapped model

Example:
    >>> parallel_model = DataParallel(model)
    >>> original_model = unwrap_parallel_model(parallel_model)
    >>> # original_model is now the unwrapped version

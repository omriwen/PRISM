# prism.utils.caching

Memory-efficient caching utilities for tensor operations.

This module provides decorators and utilities for caching expensive tensor
computations, particularly useful for repeated operations with the same parameters.

## Classes

### CacheManager

```python
CacheManager() -> None
```

Global cache manager for controlling multiple cached functions.

Useful for clearing caches when memory is limited or when operations
change state (e.g., moving tensors between devices).

Example:
    >>> manager = CacheManager()
    >>>
    >>> @manager.register
    ... @tensor_cache(maxsize=128)
    ... def func1(x):
    ...     return x * 2
    ...
    >>> @manager.register
    ... @tensor_cache(maxsize=128)
    ... def func2(x):
    ...     return x ** 2
    ...
    >>> # Clear all registered caches
    >>> manager.clear_all()
    >>>
    >>> # Get statistics for all caches
    >>> manager.get_stats()

#### Methods

##### `__init__`

Initialize cache manager.

##### `clear_all`

Clear all registered caches.

##### `get_stats`

Get cache statistics for all registered functions.

Returns:
    Dict mapping function names to cache info

##### `register`

Register a cached function with the manager.

Args:
    func: Function decorated with @tensor_cache or @simple_cache

Returns:
    The same function (for use as decorator)

##### `total_cached_items`

Get total number of cached items across all functions.

## Functions

### create_tensor_hash

```python
create_tensor_hash(*args: Any, **kwargs: Any) -> str
```

Create a hash key from function arguments including tensors.

Handles various input types:
- Tensors: Hash shape, dtype, device, and small sample of values
- Lists/tuples: Recursively hash contents
- Primitive types: Direct hashing

Args:
    *args: Positional arguments to hash
    **kwargs: Keyword arguments to hash

Returns:
    str: MD5 hash of the inputs

Example:
    >>> tensor = torch.randn(10, 10)
    >>> key = create_tensor_hash(tensor, r=5.0)

### simple_cache

```python
simple_cache(maxsize: int = 64) -> Callable
```

Simplified cache for primitive arguments (no tensor hashing).

Much faster than tensor_cache for functions with only primitive arguments
(int, float, str, bool, None).

Args:
    maxsize (int): Maximum number of cached results. Default: 64

Returns:
    Callable: Decorated function with caching

Example:
    >>> @simple_cache(maxsize=100)
    ... def compute_coefficient(n, wavelength):
    ...     return some_expensive_calculation(n, wavelength)

Notes:
    - Only use for functions without tensor/array arguments
    - ~10x faster than tensor_cache for primitive types
    - Uses tuple hashing (native Python, very fast)

### tensor_cache

```python
tensor_cache(maxsize: int = 128) -> Callable
```

Decorator to cache tensor operations with LRU eviction.

Caches function results based on input tensor properties (shape, dtype, values).
Automatically evicts least recently used items when cache is full.

Args:
    maxsize (int): Maximum number of cached results. Default: 128

Returns:
    Callable: Decorated function with caching

Example:
    >>> @tensor_cache(maxsize=256)
    ... def expensive_operation(tensor, param):
    ...     return tensor.sum() * param

Notes:
    - Cache is stored in function closure, persists across calls
    - Thread-safe for CPU operations, but not GPU (PyTorch limitation)
    - Cache invalidation: automatic LRU, or manual via clear_cache()
    - Memory: Be cautious with large tensors and large maxsize

Performance:
    - Cache hit: ~O(1) with small overhead for hashing
    - Cache miss: Original function cost + hash computation
    - Hash computation: ~O(1) for tensors (samples only)

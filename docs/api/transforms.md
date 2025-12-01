# prism.utils.transforms

Module: prism.utils.transforms
Purpose: Fourier transforms and related operations for optical simulations
Dependencies: torch
Main Functions:
    - fft(im, norm): Fourier transform with proper centering
    - ifft(im, norm): Inverse Fourier transform with proper centering
    - create_mask(image, mask_type, mask_size, center): Generate circular or square masks

Main Classes:
    - FFTCache: Caches FFT computations for repeated operations on same-sized tensors

Description:
    This module provides Fourier transform utilities for PRISM optical simulations.
    All Fourier transforms use fftshift/ifftshift to ensure the DC component is centered,
    which is essential for telescope aperture simulations in k-space.

    FFTCache can provide 10-30% speedup when repeatedly transforming tensors of the
    same shape, which is common in iterative reconstruction algorithms.

## Classes

### FFTCache

```python
FFTCache(max_cache_size: int = 128, use_fast_shifts: bool = True) -> None
```

Cache for FFT operations to improve performance on repeated tensor sizes.

When processing many tensors of the same size (common in iterative algorithms),
this cache can provide 10-30% speedup by avoiding redundant setup costs and
potentially reusing optimized FFT plans.

The cache stores a limited number of recent computations based on tensor shape,
dtype, device, and normalization mode.

Attributes:
    max_cache_size: Maximum number of cached entries (default: 128)
    cache_hits: Number of cache hits (for profiling)
    cache_misses: Number of cache misses (for profiling)
    use_fast_shifts: Whether to use optimized roll-based shifts (default: True)

Example:
    >>> fft_cache = FFTCache(max_cache_size=64)
    >>> tensor = torch.randn(1024, 1024, dtype=torch.cfloat)
    >>>
    >>> # First call - cache miss
    >>> result1 = fft_cache.fft2(tensor)
    >>>
    >>> # Subsequent calls with same shape - cache hit
    >>> result2 = fft_cache.fft2(tensor)
    >>>
    >>> # Check cache statistics
    >>> print(f"Hits: {fft_cache.cache_hits}, Misses: {fft_cache.cache_misses}")
    >>> print(f"Hit rate: {fft_cache.hit_rate():.1%}")

Performance Notes:
    - Most beneficial when repeatedly processing tensors of the same size
    - Memory overhead is minimal (only shape/dtype info cached, not actual tensors)
    - Use clear_cache() if processing very different tensor sizes over time
    - Thread-safe for read operations, but not for concurrent writes
    - With use_fast_shifts=True, provides 20-50% speedup on GPU via torch.roll

#### Methods

##### `__init__`

Initialize FFT cache.

Args:
    max_cache_size: Maximum number of cached tensor shapes (default: 128)
    use_fast_shifts: Use optimized roll-based shifts for 20-50% speedup on GPU
                    (default: True)

##### `cache_info`

Get cache statistics.

Returns:
    Dictionary with cache statistics

##### `clear_cache`

Clear the cache and reset statistics.

##### `fft2`

Cached 2D FFT with proper centering.

Equivalent to fft() function but with internal caching for performance.
When use_fast_shifts=True, uses torch.roll for 20-50% speedup on GPU.

Args:
    im: Input image tensor
    norm: Normalization mode ('ortho' for orthonormal)

Returns:
    Fourier-transformed tensor with DC component centered

##### `hit_rate`

Calculate cache hit rate.

Returns:
    Hit rate as a fraction (0.0 to 1.0)

##### `ifft2`

Cached 2D inverse FFT with proper centering.

Equivalent to ifft() function but with internal caching for performance.
When use_fast_shifts=True, uses torch.roll for 20-50% speedup on GPU.

Args:
    im: Input Fourier-space tensor
    norm: Normalization mode ('ortho' for orthonormal)

Returns:
    Inverse Fourier-transformed tensor

## Functions

### create_mask

```python
create_mask(image: torch.Tensor, mask_type: Literal['circular', 'square'] = 'circular', mask_size: float = 0.5, center: Optional[List[float]] = None) -> torch.Tensor
```

Generate circular or square mask for image.

Uses cached coordinate grid generation for improved performance when
repeatedly creating masks of the same size.

Args:
    image: Reference image for determining mask dimensions
    mask_type: Type of mask ('circular' or 'square')
    mask_size: Relative size of mask (0-1)
    center: Center position [y, x] (default: [0, 0])

Returns:
    Binary mask tensor

Performance:
    - First call: Normal speed (creates and caches coordinate grids)
    - Subsequent calls with same image size: ~30-50% faster (uses cached grids)

### fft

```python
fft(im: torch.Tensor, norm: str = 'ortho') -> torch.Tensor
```

Perform 2D Fourier transform with proper centering.

Args:
    im: Input image tensor
    norm: Normalization mode ('ortho' for orthonormal)

Returns:
    Fourier-transformed tensor with DC component centered

### fft_fast

```python
fft_fast(im: torch.Tensor, norm: str = 'ortho') -> torch.Tensor
```

Optimized 2D Fourier transform with proper centering.

Uses torch.roll for shifts instead of fftshift/ifftshift,
which is 20-50% faster on GPU.

Args:
    im: Input image tensor
    norm: Normalization mode ('ortho' for orthonormal)

Returns:
    Fourier-transformed tensor with DC component centered

### ifft

```python
ifft(im: torch.Tensor, norm: str = 'ortho') -> torch.Tensor
```

Perform 2D inverse Fourier transform with proper centering.

Args:
    im: Input Fourier-space tensor
    norm: Normalization mode ('ortho' for orthonormal)

Returns:
    Inverse Fourier-transformed tensor

### ifft_fast

```python
ifft_fast(im: torch.Tensor, norm: str = 'ortho') -> torch.Tensor
```

Optimized 2D inverse Fourier transform with proper centering.

Uses torch.roll for shifts instead of fftshift/ifftshift,
which is 20-50% faster on GPU.

Args:
    im: Input Fourier-space tensor
    norm: Normalization mode ('ortho' for orthonormal)

Returns:
    Inverse Fourier-transformed tensor

# prism.utils.sampling

Parallel sampling utilities for PRISM.

This module provides parallel execution utilities for sampling operations,
enabling faster generation of multiple sampling patterns or energy calculations
across multiple telescope positions.

Main Functions:
    - generate_samples_parallel: Generate multiple sampling patterns in parallel
    - compute_energies_parallel: Compute measurement energies in parallel
    - sort_samples_by_energy_parallel: Sort samples using parallel energy computation
    - parallel_telescope_measurements: Generate multiple telescope measurements in parallel

Example:
    >>> from prism.utils.sampling import generate_samples_parallel
    >>> # Generate 100 patterns in parallel using 4 workers
    >>> patterns = generate_samples_parallel(
    ...     n_patterns=100,
    ...     pattern_generator=lambda: generate_fermat_spiral(50, 100),
    ...     n_workers=4
    ... )

## Classes

## Functions

### compute_energies_parallel

```python
compute_energies_parallel(object_tensor: torch.Tensor, points: torch.Tensor, telescope: Any, n_workers: int = 4, batch_size: Optional[int] = None) -> torch.Tensor
```

Compute measurement energies for multiple points in parallel.

This function parallelizes the computation of energies across different
sampling points, which can be beneficial when dealing with a large number
of points or computationally expensive telescope simulations.

Parameters:
    object_tensor: The object to be measured [B, C, H, W]
    points: Sampling points to evaluate [N, 2]
    telescope: Telescope instance for measurements
    n_workers: Number of parallel workers
    batch_size: If specified, process points in batches of this size.
               If None, automatically determine based on n_workers.

Returns:
    torch.Tensor: Energy values for each point [N]

Example:
    >>> from optics import Telescope
    >>> telescope = Telescope(n=256, r=10)
    >>> object_img = torch.randn(1, 1, 256, 256)
    >>> points = torch.randn(100, 2)  # 100 sample points
    >>>
    >>> # Compute energies in parallel
    >>> energies = compute_energies_parallel(
    ...     object_img, points, telescope, n_workers=4
    ... )

Notes:
    - For small numbers of points (< 50), parallel overhead may not be worth it
    - The telescope's forward pass should be thread-safe
    - Results are returned in the same order as input points

### generate_samples_parallel

```python
generate_samples_parallel(n_patterns: int, pattern_generator: Callable[[], torch.Tensor], n_workers: int = 4, use_processes: bool = False, show_progress: bool = False) -> List[torch.Tensor]
```

Generate multiple sampling patterns in parallel.

This function is useful when you need to generate many independent
sampling patterns (e.g., for Monte Carlo simulations or ensemble methods).

Parameters:
    n_patterns: Number of patterns to generate
    pattern_generator: Function that generates a single pattern (takes no args)
    n_workers: Number of parallel workers
    use_processes: If True, use ProcessPoolExecutor (better for CPU-bound tasks).
                  If False, use ThreadPoolExecutor (better for I/O-bound tasks).
    show_progress: If True, print progress updates

Returns:
    List[torch.Tensor]: List of generated patterns

Example:
    >>> from sampling import generate_fermat_spiral
    >>> from functools import partial
    >>>
    >>> # Create pattern generator with fixed parameters
    >>> generator = partial(generate_fermat_spiral, n_points=50, r_max=100)
    >>>
    >>> # Generate 10 patterns in parallel
    >>> patterns = generate_samples_parallel(10, generator, n_workers=4)

Notes:
    - Use ProcessPoolExecutor for CPU-intensive pattern generation
    - Use ThreadPoolExecutor for I/O-bound operations or when sharing memory
    - Ensure pattern_generator is picklable when using processes

### get_optimal_worker_count

```python
get_optimal_worker_count(task_type: str = 'cpu_bound', max_workers: Optional[int] = None) -> int
```

Get recommended number of workers based on available hardware.

Parameters:
    task_type: Type of task - "cpu_bound" or "io_bound"
    max_workers: Maximum number of workers to use (None for automatic)

Returns:
    int: Recommended number of workers

Example:
    >>> n_workers = get_optimal_worker_count("cpu_bound")
    >>> # Use n_workers for parallel operations

### parallel_telescope_measurements

```python
parallel_telescope_measurements(object_tensor: torch.Tensor, centers_list: List[List[float]], telescope: Any, n_workers: int = 4, batch_size: Optional[int] = None) -> torch.Tensor
```

Generate telescope measurements for multiple centers in parallel.

This can be faster than sequential measurement generation when dealing
with many measurement positions or complex telescope simulations.

Parameters:
    object_tensor: The object to be measured [B, C, H, W]
    centers_list: List of measurement centers [[y1, x1], [y2, x2], ...]
    telescope: Telescope instance for measurements
    n_workers: Number of parallel workers
    batch_size: Number of centers to process per batch

Returns:
    torch.Tensor: Stacked measurements [N, C, H, W]

Example:
    >>> from optics import Telescope
    >>> telescope = Telescope(n=256, r=10)
    >>> object_img = torch.randn(1, 1, 256, 256)
    >>> centers = [[0, 0], [10, 10], [20, 20]]
    >>>
    >>> # Generate measurements in parallel
    >>> measurements = parallel_telescope_measurements(
    ...     object_img, centers, telescope, n_workers=2
    ... )

Notes:
    - For small numbers of centers (< 10), sequential may be faster
    - Measurements are returned in the same order as centers_list

### sort_samples_by_energy_parallel

```python
sort_samples_by_energy_parallel(object_tensor: torch.Tensor, points: torch.Tensor, telescope: Any, descending: bool = True, n_workers: int = 4) -> torch.Tensor
```

Sort sampling points by their measurement energy using parallel computation.

Parameters:
    object_tensor: The object to be measured [B, C, H, W]
    points: Sampling points to sort [N, 2] or [N, 1, 2]
    telescope: Telescope instance for measurements
    descending: If True, sort from highest to lowest energy
    n_workers: Number of parallel workers

Returns:
    torch.Tensor: Points sorted by energy (same shape as input)

Example:
    >>> from optics import Telescope
    >>> telescope = Telescope(n=256, r=10)
    >>> object_img = torch.randn(1, 1, 256, 256)
    >>> points = torch.randn(100, 1, 2)
    >>>
    >>> # Sort by energy in parallel
    >>> sorted_points = sort_samples_by_energy_parallel(
    ...     object_img, points, telescope, n_workers=4
    ... )

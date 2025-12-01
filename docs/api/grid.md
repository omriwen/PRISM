# prism.core.grid

Grid management for optical simulations.

This module provides coordinate systems for diffraction simulations using
the discrete Fourier transform (FFT). Coordinates follow the fftshift convention
with zero-frequency (DC component) at the center.

Coordinate Systems
------------------
Spatial (x, y):
    Physical coordinates in meters, centered at origin.
    - Range: [-FOV/2, FOV/2] where FOV = nx * dx
    - Zero at: (nx//2, ny//2) after fftshift
    - Units: meters

Frequency (kx, ky):
    Fourier conjugate variables in inverse meters.
    - Relation: kx = x / (nx * dx²)
    - Range: [-1/(2*dx), 1/(2*dx)] (Nyquist limit)
    - Units: 1/meters (spatial frequency)

FFT Convention:
    Uses fftshift before and after FFT to center DC component.
    Coordinates match numpy.fft.fftshift convention.

Physical Parameters
-------------------
wavelength (λ):
    Optical wavelength in meters.
    - Visible light: 400-700 nm (4e-7 to 7e-7 m)
    - Common: 520 nm (green) = 520e-9 m
    - Infrared: 1-10 µm

dx, dy:
    Pixel pitch (sampling interval) in meters.
    - Typical: 1-100 µm (1e-6 to 1e-4 m)
    - Must satisfy Nyquist criterion (see below)

nx, ny:
    Number of pixels in each dimension.
    - Power of 2 recommended for FFT efficiency: 64, 128, 256, 512, 1024, 2048
    - Minimum: ~32 for meaningful diffraction
    - Typical: 256-1024

Sampling Requirements
---------------------
Nyquist Sampling:
    To avoid aliasing in propagation:

        dx < λ * z / (2 * FOV)

    where z is propagation distance, FOV = nx * dx.

    Example:
        λ = 520nm, z = 1m, FOV = 2.56mm
        → dx_max = 520e-9 * 1 / (2 * 2.56e-3) = 101 µm
        → dx = 10 µm satisfies criterion ✓

Field of View:
    FOV = nx * dx

    Trade-offs:
    - Large FOV: Capture wide scene, but dx increases (lower resolution)
    - Small FOV: High resolution, but limited scene coverage

Resolution Limit:
    Diffraction-limited resolution:

        Δx_min ≈ λ * f / D

    where f is focal length, D is aperture diameter.

Usage Examples
--------------
Basic grid creation:
    >>> from prism.core.grid import Grid
    >>> grid = Grid(nx=256, dx=10e-6, wavelength=520e-9, device='cuda')
    >>> print(f"FOV: {grid.fov[0]*1e3:.2f} mm")
    FOV: 2.56 mm

Access coordinates:
    >>> x, y = grid.grid  # Spatial coordinates
    >>> kx, ky = grid.kx, grid.ky  # Frequency coordinates
    >>> print(f"x range: [{x.min():.2e}, {x.max():.2e}] m")
    x range: [-1.28e-03, 1.28e-03] m

Grid operations:
    >>> padded = grid.pad(padding_scale=2)  # Zero-padding
    >>> print(f"Padded size: {padded.nx}")
    Padded size: 512

    >>> upsampled = grid.upsample(scale=2)  # Finer sampling
    >>> print(f"Upsampled dx: {upsampled.dx:.2e}")
    Upsampled dx: 5.00e-06

Lens Fourier transform:
    >>> focal_length = 0.1  # 10 cm
    >>> focal_grid = grid.lens_ft_grid(f=focal_length)
    >>> print(f"Focal plane FOV: {focal_grid.fov[0]*1e3:.2f} mm")

From custom coordinates:
    >>> import torch
    >>> x = torch.linspace(-1e-3, 1e-3, 256).unsqueeze(0)
    >>> y = torch.linspace(-1e-3, 1e-3, 256).unsqueeze(1)
    >>> grid = Grid.from_coordinates(x, y, wavelength=520e-9)

Physics Notes
-------------
Fraunhofer (Far-Field) Regime:
    Valid when Fresnel number F << 1, where

        F = a² / (λ * z)

    a = aperture size, z = distance.

    For astronomical imaging: F ~ 10⁻¹² (always far-field)

Fresnel (Near-Field) Regime:
    Valid when 0.1 < F < 10.
    Requires more careful grid design.

Evanescent Waves:
    Frequencies where kx² + ky² > (1/λ)² represent evanescent waves.
    These decay exponentially and are filtered in propagation.

## Classes

### Grid

```python
Grid(nx: int = 256, dx: float = 1e-05, ny: Optional[int] = None, dy: Optional[float] = None, wavelength: float = 5.2e-07, device: Union[str, torch.device] = 'cpu') -> None
```

Manages spatial and frequency grids for optical simulations.

The Grid class handles coordinate transformations between spatial (x, y) and
frequency (kx, ky) domains, and provides utilities for padding, upsampling,
and lens Fourier transform grids.

Coordinates are cached for performance - repeated access returns the same
tensor object without regeneration.

Attributes:
    nx (int): Number of pixels in x direction
    ny (int): Number of pixels in y direction
    dx (float): Pixel spacing in x direction (meters)
    dy (float): Pixel spacing in y direction (meters)
    wl (float): Wavelength (meters)
    device (str): Device for tensor operations ('cpu' or 'cuda')

Properties:
    fov: Field of view (nx*dx, ny*dy)
    x: Spatial x coordinates (cached)
    y: Spatial y coordinates (cached)
    grid: Tuple of (x, y) coordinates
    kx: Frequency x coordinates (cached)
    ky: Frequency y coordinates (cached)
    kmax: Maximum frequency

Methods:
    lens_ft(f): Compute lens Fourier transform coordinates
    pad(padding_scale): Create padded grid
    upsample(scale): Create upsampled grid
    clone(): Create a copy of the grid
    lens_ft_grid(f): Create grid in lens Fourier transform plane
    to(device): Move grid to specified device
    from_coordinates(x, y, wavelength, device): Create grid from coordinate tensors

Example:
    >>> grid = Grid(nx=256, dx=1e-5, wavelength=520e-9)
    >>> x, y = grid.grid
    >>> kx, ky = grid.kx, grid.ky

#### Methods

##### `__init__`

Initialize Grid with validation.

Args:
    nx (int): Number of pixels in x direction. Must be positive.
    dx (float): Pixel spacing in x direction (meters). Must be positive.
    ny (int, optional): Number of pixels in y direction. Defaults to nx.
    dy (float, optional): Pixel spacing in y direction. Defaults to dx.
    wavelength (float): Wavelength in meters. Must be positive. Defaults to 520nm.
    device (str): Device for tensor operations. Defaults to 'cpu'.

Raises:
    ValueError: If any parameter is invalid (non-positive values).

Warns:
    If nx or ny is not a power of 2 (FFT will be slower).

##### `clone`

Create a deep copy of the grid.

Returns:
    Grid: Cloned grid object

##### `lens_ft`

Compute lens Fourier transform coordinates.

Args:
    f (float): Focal length (meters)

Returns:
    tuple: (x_f, y_f) coordinates in focal plane

##### `lens_ft_grid`

Create a grid in the lens Fourier transform plane.

Args:
    f (float): Focal length (meters)

Returns:
    Grid: New grid in focal plane coordinates

##### `pad`

Create a padded version of the grid.

Args:
    padding_scale (int): Padding scale factor

Returns:
    Grid: New grid with increased size

##### `to`

Move grid to specified device.

Cache is invalidated on device change since cached tensors are device-specific.

Args:
    device (str or torch.device): Target device

Returns:
    Grid: Self for chaining

##### `upsample`

Create an upsampled version of the grid.

Args:
    scale (int): Upsampling scale factor

Returns:
    Grid: New grid with finer spacing

# prism.core.propagators

Optical Propagation Methods for PRISM.

This module provides implementations of different optical propagation methods
for simulating light propagation through free space. The choice of method
depends on the propagation distance and desired accuracy.

Propagation Methods
-------------------

Fraunhofer Propagator:
    Far-field diffraction (z >> a²/λ).
    - Method: Simple FFT/IFFT
    - Speed: Fastest
    - Accuracy: Excellent for far field
    - PRISM default: Used for astronomical imaging

Fresnel Propagator (1-Step Impulse Response):
    Near-field diffraction using impulse response method.
    - Method: Single FFT with pre/post quadratic phase chirps
    - Speed: Fast - O(N² log N), ~2x faster than 2-step method
    - Accuracy: Good for intermediate distances (0.1 < F < 10)
    - Output grid scaling: dx_out = λ·z / (N·dx_in)
    - Use when: 0.1 < F < 10, z > z_crit = N·dx²/λ

Angular Spectrum Propagator:
    Exact propagation (within paraxial approximation).
    - Method: Transfer function with exact phase
    - Speed: Fast (same as Fresnel)
    - Accuracy: Excellent for all distances
    - Use when: High accuracy needed, near field (F > 1)

Propagator Selection Guide
---------------------------

┌─────────────────────┬──────────────────┬─────────┬──────────────────────┐
│ Method              │ Valid Range      │ Speed   │ Use Case             │
├─────────────────────┼──────────────────┼─────────┼──────────────────────┤
│ Fraunhofer          │ F << 1           │ Fastest │ Astronomy (PRISM)    │
│                     │ z >> a²/λ        │         │ F ~ 10⁻¹²            │
├─────────────────────┼──────────────────┼─────────┼──────────────────────┤
│ Fresnel             │ 0.1 < F < 10     │ Fast    │ Intermediate         │
│ (Quadratic Phase)   │                  │         │ distances            │
├─────────────────────┼──────────────────┼─────────┼──────────────────────┤
│ Angular Spectrum    │ All F            │ Fast    │ High accuracy        │
│                     │ (exact)          │         │ Near field (F > 1)   │
└─────────────────────┴──────────────────┴─────────┴──────────────────────┘

Fresnel Number: F = a²/(λz)
    a = characteristic aperture size
    λ = wavelength
    z = propagation distance

Usage Examples
--------------

Fraunhofer (PRISM Default):
    >>> from prism.core.propagators import FraunhoferPropagator
    >>> prop = FraunhoferPropagator(normalize=True)
    >>>
    >>> # Forward: spatial → k-space
    >>> k_field = prop(spatial_field, direction='forward')
    >>>
    >>> # Backward: k-space → spatial
    >>> spatial_field = prop(k_field, direction='backward')

Fresnel (1-Step Method - Grid-based API):
    >>> from prism.core.grid import Grid
    >>> grid = Grid(nx=256, dx=10e-6, wavelength=520e-9)
    >>> prop = FresnelPropagator(grid=grid, distance=0.1)  # 10 cm
    >>>
    >>> output_field = prop(input_field)
    >>> # Output grid scales with distance:
    >>> print(f"Output pixel size: {prop.output_grid.dx:.2e} m")

Angular Spectrum (High Accuracy):
    >>> from prism.core.grid import Grid
    >>> grid = Grid(nx=256, dx=10e-6, wavelength=520e-9)
    >>> prop = AngularSpectrumPropagator(grid, distance=0.05)
    >>> output_field = prop(input_field)

Factory Function:
    >>> from prism.core.propagators import create_propagator
    >>>
    >>> # Auto-select based on method string
    >>> prop = create_propagator('fraunhofer', normalize=True)
    >>> prop = create_propagator('fresnel', dx=10e-6, ...)
    >>> prop = create_propagator('angular_spectrum', grid=grid, ...)

References
----------
- Goodman, J. W. "Introduction to Fourier Optics" (2005), Chapter 3-4
- Born & Wolf, "Principles of Optics" (1999), Chapter 8

## Classes

## Functions

### create_propagator

```python
create_propagator(method: Literal['auto', 'fraunhofer', 'fresnel', 'angular_spectrum', 'otf', 'extended_source', 'incoherent_auto', 'partially_coherent_auto'], *, normalize: bool = True, fft_cache: Optional[prism.utils.transforms.FFTCache] = None, grid: Optional[prism.core.grid.Grid] = None, aperture: Optional[torch.Tensor] = None, **kwargs: Any) -> prism.core.propagators.base.Propagator
```

Factory function to create propagator instances.

Parameters
----------
method : PropagationMethod
    Propagation method to use:
        - 'fraunhofer': Far-field (FFT-based), fastest
        - 'fresnel': 1-step Impulse Response, intermediate distances (0.1 < F < 10)
        - 'angular_spectrum': Exact method, all distances
        - 'otf': OTF-based incoherent propagation
        - 'extended_source': Extended source propagation (not yet implemented)
normalize : bool, optional
    Whether to use normalized FFTs. Default: True
fft_cache : FFTCache, optional
    Shared FFT cache for performance optimization
grid : Grid, optional
    Coordinate system (required for angular_spectrum and otf methods)
aperture : Tensor, optional
    Aperture/pupil function (required for otf method)
**kwargs : Any
    Additional method-specific parameters

Returns
-------
Propagator
    Configured propagator instance

Raises
------
ValueError
    If method is unknown or required parameters are missing

Examples
--------
>>> # Fraunhofer (PRISM default for astronomy)
>>> prop = create_propagator('fraunhofer', normalize=True)

>>> # Angular spectrum (high accuracy)
>>> from prism.core.grid import Grid
>>> grid = Grid(nx=256, dx=10e-6, wavelength=520e-9)
>>> prop = create_propagator('angular_spectrum', grid=grid, distance=0.05)

>>> # OTF propagator for incoherent illumination
>>> aperture = torch.ones(256, 256, dtype=torch.cfloat)
>>> prop = create_propagator('otf', aperture=aperture, grid=grid)

Selection Guide
---------------
For coherent illumination (astronomical imaging, PRISM default):
    - Use 'fraunhofer' (F ~ 10⁻¹², far field)
    - Use 'angular_spectrum' for high accuracy at any distance

For incoherent illumination (extended sources):
    - Use 'otf' with aperture function

Where F = a²/(λz), a = aperture size, z = distance

### select_propagator

```python
select_propagator(wavelength: float, obj_distance: float, fov: float, *, method: Literal['auto', 'fraunhofer', 'fresnel', 'angular_spectrum', 'otf', 'extended_source', 'incoherent_auto', 'partially_coherent_auto'] = 'auto', illumination: Literal['coherent', 'incoherent', 'partially_coherent'] = 'coherent', aperture: Optional[torch.Tensor] = None, grid: Optional[prism.core.grid.Grid] = None, image_size: Optional[int] = None, dx: Optional[float] = None, dxf: Optional[float] = None, fft_cache: Optional[prism.utils.transforms.FFTCache] = None, **kwargs: Any) -> prism.core.propagators.base.Propagator
```

Select and create appropriate propagator based on physical parameters.

This function intelligently chooses the best propagation method based on
the Fresnel number F = fov²/(wavelength × obj_distance) and the illumination
mode, eliminating the need for users to understand optical physics regimes.

Parameters
----------
wavelength : float
    Optical wavelength in meters
obj_distance : float
    Propagation distance in meters
fov : float
    Field of view (aperture size) in meters
method : PropagationMethod, optional
    Propagation method selection:
        - 'auto': Automatically select based on F (default)
        - 'fraunhofer': Force Fraunhofer (FFT-based, fast)
        - 'angular_spectrum': Force Angular Spectrum (exact, recommended)
        - 'fresnel': 1-step Impulse Response (valid for 0.1 < F < 10)
        - 'otf': OTF-based incoherent propagation
        - 'incoherent_auto': Auto-select incoherent propagator
illumination : IlluminationMode, optional
    Illumination type: "coherent" or "incoherent". Default: "coherent"
aperture : Tensor, optional
    Aperture/pupil function (required for incoherent illumination)
grid : Grid, optional
    Coordinate system (optional, will be created if needed)
image_size : int, optional
    Image size in pixels (for creating grid if not provided)
dx : float, optional
    Spatial sampling interval in meters (for creating grid if not provided)
dxf : float, optional
    Frequency sampling interval in 1/meters (required for Fresnel only)
fft_cache : FFTCache, optional
    Shared FFT cache for performance optimization
**kwargs : Any
    Additional parameters passed to propagator constructor

Returns
-------
Propagator
    Configured propagator instance

Raises
------
ValueError
    If required parameters are missing for selected method

Examples
--------
>>> # Auto-select for coherent Europa observation
>>> prop = select_propagator(
...     wavelength=520e-9,
...     obj_distance=628e9,
...     fov=1024 * 10e-6,
...     method='auto'
... )
>>> # Logs: "Auto-selected Fraunhofer propagator (F=2.5e-13 << 0.1)"

>>> # Incoherent illumination for extended source
>>> aperture = torch.ones(256, 256, dtype=torch.cfloat)
>>> prop = select_propagator(
...     wavelength=550e-9,
...     obj_distance=1e6,
...     fov=256 * 1e-5,
...     illumination="incoherent",
...     aperture=aperture,
... )
>>> # Returns OTFPropagator

>>> # Manual override (with warning if inappropriate)
>>> prop = select_propagator(
...     wavelength=520e-9,
...     obj_distance=0.01,  # 1 cm
...     fov=0.001,  # 1 mm
...     method='fraunhofer',  # Force far-field
...     fft_cache=shared_cache
... )
>>> # Logs: "Using Fraunhofer with F=0.19 >= 0.1. Accuracy may be reduced."

Notes
-----
Selection Logic for Coherent Illumination:
    - F < 0.1 → FraunhoferPropagator (far field, fast)
    - F ≥ 0.1 → AngularSpectrumPropagator (exact, all distances)

For Incoherent Illumination:
    - OTFPropagator is used (requires aperture)

Physics Notes:
    - Fresnel number F determines diffraction regime
    - F < 0.1: Far field → Fraunhofer (fast, FFT-based)
    - 0.1 < F < 10: Fresnel regime → FresnelPropagator (balanced)
    - F ≥ 10 or high accuracy needed: Angular Spectrum (exact for all distances)
    - **Note**: FresnelPropagator now uses corrected 1-step method

See Also
--------
create_propagator : Lower-level factory (no auto-selection)
OTFPropagator : Incoherent propagation using optical transfer function

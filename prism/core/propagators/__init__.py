"""
Optical Propagation Methods for SPIDS.

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
    - SPIDS default: Used for astronomical imaging

Fresnel Propagator (1-Step Impulse Response):
    Near-field diffraction using 1-step method with grid scaling.
    - Method: Single FFT with pre/post chirps
    - Speed: Fast (~2x faster than old 2-step)
    - Accuracy: Good for intermediate distances (< 5% error vs ASM)
    - Use when: 0.1 < F < 10
    - Note: Output grid scales with distance (dx_out = λz/(N·dx_in))

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
│ Fraunhofer          │ F << 1           │ Fastest │ Astronomy (SPIDS)    │
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

Fraunhofer (SPIDS Default):
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
    >>> output_field = prop(input_field)
    >>> # Note: Output grid scales with distance
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
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger
from torch import Tensor

# Import dependencies for factory functions
from prism.core.grid import Grid

# Import propagator implementations
from prism.core.propagators.angular_spectrum import AngularSpectrumPropagator

# Import base classes and types
from prism.core.propagators.base import (
    CoherenceMode,
    IlluminationMode,
    PropagationMethod,
    Propagator,
    SamplingMethod,
)
from prism.core.propagators.fraunhofer import FraunhoferPropagator
from prism.core.propagators.fresnel import FreeSpacePropagator, FresnelPropagator

# Import incoherent propagators and helpers
from prism.core.propagators.incoherent import (
    ExtendedSourcePropagator,
    OTFPropagator,
    create_binary_source,
    create_gaussian_source,
    create_ring_source,
    create_stellar_disk,
    estimate_required_samples,
)

# Import validation utilities
from prism.core.propagators.utils import validate_coherent_input, validate_intensity_input
from prism.utils.transforms import FFTCache


__all__ = [
    # Base classes and types
    "Propagator",
    "PropagationMethod",
    "IlluminationMode",
    "SamplingMethod",
    "CoherenceMode",
    # Coherent propagators
    "FraunhoferPropagator",
    "FresnelPropagator",
    "FreeSpacePropagator",  # Backward compatibility alias
    "AngularSpectrumPropagator",
    # Incoherent propagators
    "OTFPropagator",
    "ExtendedSourcePropagator",
    # Source geometry helpers
    "create_stellar_disk",
    "create_gaussian_source",
    "create_binary_source",
    "create_ring_source",
    "estimate_required_samples",
    # Validation utilities
    "validate_intensity_input",
    "validate_coherent_input",
    # Factory functions
    "create_propagator",
    "select_propagator",
]


def create_propagator(
    method: PropagationMethod,
    *,
    normalize: bool = True,
    fft_cache: Optional[FFTCache] = None,
    grid: Optional[Grid] = None,
    aperture: Optional[Tensor] = None,
    **kwargs: Any,
) -> Propagator:
    """
    Factory function to create propagator instances.

    Parameters
    ----------
    method : PropagationMethod
        Propagation method to use:
            - 'fraunhofer': Far-field (FFT-based), fastest
            - 'fresnel': 1-step Impulse Response, intermediate distances (0.1 < F < 10)
            - 'angular_spectrum': Exact method, all distances
            - 'otf': OTF-based incoherent propagation
            - 'extended_source': Extended source propagation
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
    >>> # Fraunhofer (SPIDS default for astronomy)
    >>> prop = create_propagator('fraunhofer', normalize=True)

    >>> # Fresnel (NEW Grid-based API)
    >>> from prism.core.grid import Grid
    >>> grid = Grid(nx=256, dx=10e-6, wavelength=520e-9)
    >>> prop = create_propagator('fresnel', grid=grid, distance=0.1)

    >>> # Angular spectrum (high accuracy)
    >>> grid = Grid(nx=256, dx=10e-6, wavelength=520e-9)
    >>> prop = create_propagator('angular_spectrum', grid=grid, distance=0.05)

    >>> # OTF propagator for incoherent illumination
    >>> aperture = torch.ones(256, 256, dtype=torch.cfloat)
    >>> prop = create_propagator('otf', aperture=aperture, grid=grid)

    Selection Guide
    ---------------
    For coherent illumination (astronomical imaging, SPIDS default):
        - Use 'fraunhofer' (F ~ 10⁻¹², far field)
        - Use 'angular_spectrum' for high accuracy at any distance

    For incoherent illumination (extended sources):
        - Use 'otf' with aperture function

    Where F = a²/(λz), a = aperture size, z = distance
    """
    if method == "fraunhofer":
        return FraunhoferPropagator(normalize=normalize, fft_cache=fft_cache)

    elif method == "fresnel":
        # Grid-based signature (breaking change from old dx/dxf API)
        if grid is None:
            # Try to extract distance from kwargs for better error message
            distance = kwargs.get("distance", None)
            raise ValueError(
                "FresnelPropagator requires 'grid' parameter. "
                "Old API (dx, dxf, image_size) is no longer supported. "
                "Use: create_propagator('fresnel', grid=grid, distance=0.1)"
            )

        distance = kwargs.pop("distance", None)
        if distance is None:
            raise ValueError(
                "FresnelPropagator requires 'distance' parameter. "
                "Specify propagation distance in meters."
            )

        return FresnelPropagator(grid=grid, distance=distance, fft_cache=fft_cache, **kwargs)

    elif method == "angular_spectrum":
        if grid is None:
            raise ValueError(
                "Grid required for angular_spectrum propagator. "
                "Provide grid=Grid(nx=..., dx=..., wavelength=...)"
            )
        return AngularSpectrumPropagator(grid=grid, fft_cache=fft_cache, **kwargs)

    elif method == "otf":
        if aperture is None:
            raise ValueError(
                "Aperture required for OTF propagator. "
                "Provide aperture tensor representing the pupil function."
            )
        return OTFPropagator(
            aperture=aperture,
            grid=grid,
            normalize=normalize,
            fft_cache=fft_cache,
        )

    elif method == "extended_source":
        if grid is None:
            raise ValueError(
                "Grid required for extended_source propagator. "
                "Provide grid=Grid(nx=..., dx=..., wavelength=...)"
            )

        # Use provided coherent propagator or default to Fraunhofer
        coherent_prop = kwargs.pop("coherent_propagator", None)
        if coherent_prop is None:
            coherent_prop = FraunhoferPropagator(fft_cache=fft_cache)

        return ExtendedSourcePropagator(
            coherent_propagator=coherent_prop,
            grid=grid,
            n_source_points=kwargs.pop("n_source_points", 1000),
            sampling_method=kwargs.pop("sampling_method", "adaptive"),
            batch_size=kwargs.pop("batch_size", 32),
            fft_cache=fft_cache,
            **kwargs,
        )

    elif method == "incoherent_auto":
        # Auto-select incoherent propagator (currently only OTF available)
        if aperture is None:
            raise ValueError(
                "Aperture required for incoherent_auto propagator. "
                "Provide aperture tensor representing the pupil function."
            )
        return OTFPropagator(
            aperture=aperture,
            grid=grid,
            normalize=normalize,
            fft_cache=fft_cache,
        )

    else:
        raise ValueError(
            f"Unknown propagation method: '{method}'. "
            f"Valid options: 'fraunhofer', 'fresnel', 'angular_spectrum', 'otf', "
            f"'extended_source', 'incoherent_auto'"
        )


def select_propagator(
    wavelength: float,
    obj_distance: float,
    fov: float,
    *,
    method: PropagationMethod = "auto",
    illumination: IlluminationMode = "coherent",
    aperture: Optional[Tensor] = None,
    grid: Optional[Grid] = None,
    image_size: Optional[int] = None,
    dx: Optional[float] = None,
    dxf: Optional[float] = None,
    fft_cache: Optional[FFTCache] = None,
    **kwargs: Any,
) -> Propagator:
    """
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
            - 'fresnel': 1-step method for Fresnel regime (0.1 < F < 10)
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
        (Deprecated - no longer used) Frequency sampling interval in 1/meters
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
        - 0.1 ≤ F < 10: Fresnel regime → FresnelPropagator (1-step, efficient)
        - F ≥ 0.1: Angular Spectrum (exact for all distances)
        - Auto-selection uses F < 0.1 threshold (conservative)

    See Also
    --------
    create_propagator : Lower-level factory (no auto-selection)
    OTFPropagator : Incoherent propagation using optical transfer function
    """
    from prism.config.constants import fresnel_number as calculate_fresnel_number

    # Handle incoherent illumination
    if illumination == "incoherent" or method in ("otf", "incoherent_auto"):
        if aperture is None:
            raise ValueError(
                "Aperture required for incoherent illumination. "
                "Provide aperture tensor representing the pupil function."
            )

        # Create grid if not provided
        if grid is None and (image_size is not None or dx is not None):
            _image_size = image_size if image_size is not None else 256
            _dx = dx if dx is not None else 10e-6
            grid = Grid(nx=_image_size, dx=_dx, wavelength=wavelength)

        logger.info("Using OTF propagator for incoherent illumination")
        return OTFPropagator(
            aperture=aperture,
            grid=grid,
            fft_cache=fft_cache,
            **kwargs,
        )

    # Handle partially coherent illumination
    if illumination == "partially_coherent" or method in (
        "extended_source",
        "partially_coherent_auto",
    ):
        # Create grid if not provided
        if grid is None:
            _image_size = image_size if image_size is not None else 256
            _dx = dx if dx is not None else 10e-6
            grid = Grid(nx=_image_size, dx=_dx, wavelength=wavelength)

        coherent_prop = FraunhoferPropagator(fft_cache=fft_cache)

        logger.info("Using ExtendedSourcePropagator for partially coherent illumination")
        return ExtendedSourcePropagator(
            coherent_propagator=coherent_prop,
            grid=grid,
            fft_cache=fft_cache,
            **kwargs,
        )

    # Calculate Fresnel number for coherent propagation
    fresnel_number = calculate_fresnel_number(fov, obj_distance, wavelength)

    # Coherent auto-selection
    selected_method = method
    if method == "auto":
        # Automatic selection based on Fresnel number
        if fresnel_number < 0.1:
            selected_method = "fraunhofer"
            logger.info(f"Auto-selected Fraunhofer propagator (F={fresnel_number:.2e} < 0.1)")
        else:
            selected_method = "angular_spectrum"
            logger.info(f"Auto-selected Angular Spectrum propagator (F={fresnel_number:.2e} ≥ 0.1)")
    else:
        # Manual selection - warn if inappropriate
        if method == "fraunhofer" and fresnel_number >= 0.1:
            logger.warning(
                f"Using Fraunhofer with F={fresnel_number:.2f} ≥ 0.1. "
                f"Accuracy may be reduced. Consider 'angular_spectrum' or 'auto'."
            )
        elif method == "fresnel":
            # Fresnel is now valid with 1-step method (accuracy fixed)
            if 0.1 <= fresnel_number < 10:
                logger.info(
                    f"Using Fresnel propagator (1-step method) with F={fresnel_number:.2e}. "
                    f"Good choice for Fresnel regime (0.1 < F < 10)."
                )
            elif fresnel_number < 0.1:
                logger.warning(
                    f"Fresnel number F={fresnel_number:.2e} < 0.1 (far field). "
                    f"Consider using FraunhoferPropagator for better performance."
                )
            else:  # fresnel_number >= 10
                logger.warning(
                    f"Fresnel number F={fresnel_number:.2e} > 10 (near field). "
                    f"Consider using AngularSpectrumPropagator for better accuracy."
                )
        elif method == "angular_spectrum" and fresnel_number < 0.1:
            logger.info(
                f"Using Angular Spectrum with F={fresnel_number:.2e} < 0.1. "
                f"Fraunhofer would be faster with similar accuracy."
            )

    # Create propagator based on selected method
    if selected_method == "fraunhofer":
        return FraunhoferPropagator(normalize=True, fft_cache=fft_cache)

    elif selected_method == "fresnel":
        # Fresnel uses Grid-based API (breaking change from old dx/dxf API)
        # Create Grid if not provided
        if grid is None:
            if image_size is None or dx is None:
                raise ValueError(
                    "FresnelPropagator requires either 'grid' or 'image_size' + 'dx'. "
                    "Provide grid=Grid(...) or image_size and dx parameters."
                )

            grid = Grid(nx=image_size, ny=image_size, dx=dx, dy=dx, wavelength=wavelength)

        # Pass to factory with new signature
        return create_propagator(
            "fresnel", grid=grid, distance=obj_distance, fft_cache=fft_cache, **kwargs
        )

    elif selected_method == "angular_spectrum":
        # Angular spectrum requires Grid
        _image_size = image_size if image_size is not None else 256
        _dx = dx if dx is not None else 10e-6

        if grid is None:
            grid = Grid(nx=_image_size, dx=_dx, wavelength=wavelength)
        return AngularSpectrumPropagator(grid=grid, distance=obj_distance, fft_cache=fft_cache)

    else:
        raise ValueError(
            f"Unknown method: '{selected_method}'. "
            f"Choose from: 'auto', 'fraunhofer', 'fresnel', 'angular_spectrum', "
            f"'otf', 'incoherent_auto'"
        )

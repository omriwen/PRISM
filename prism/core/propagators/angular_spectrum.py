"""
Angular Spectrum propagator for exact optical propagation.

This module provides the AngularSpectrumPropagator class, which implements
exact propagation valid for all distances (within paraxial approximation).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
from torch import Tensor

from prism.core.grid import Grid
from prism.core.propagators.base import Propagator
from prism.utils.transforms import FFTCache


class AngularSpectrumPropagator(Propagator):
    """
    Angular spectrum method (exact propagation).

    This is the most accurate propagation method, valid for ALL distances.
    Uses the exact transfer function in k-space.

    Method:
        Transfer function approach:

        U(z) = IFFT{ FFT(U(0)) · H(kx, ky, z) }

        where H(kx, ky, z) = exp(i·k·z·sqrt(1 - (λkx)² - (λky)²))
        and k = 2π/λ

    Evanescent Wave Handling:
        Frequencies where kx² + ky² > (1/λ)² represent evanescent waves.
        These are properly filtered (set to zero) as they decay exponentially.

    Attributes:
        grid: Grid object defining coordinate system
        distance: Propagation distance (optional, can specify in forward())
        diff_limit: Boolean mask for propagating vs. evanescent waves
        k_sqrt: sqrt(1/λ² - kx² - ky²) for transfer function
        fft_cache: FFT cache for performance optimization

    Example:
        >>> from prism.core.grid import Grid
        >>> grid = Grid(nx=256, dx=10e-6, wavelength=520e-9)
        >>> prop = AngularSpectrumPropagator(grid, distance=0.05)
        >>>
        >>> output_field = prop(input_field)
        >>> # Or with variable distance:
        >>> output_field = prop(input_field, distance=0.1)

    Physics Notes:
        - Valid for ALL distances (near field to far field)
        - Exact within paraxial approximation
        - Properly handles evanescent waves
        - More computationally expensive than Fraunhofer/Fresnel

    When to Use:
        - High accuracy required
        - Near-field propagation (F > 1)
        - When Fresnel approximation breaks down
        - Scientific applications where exactness matters

    Based on:
        deepTIMP implementation (/home/omri/deepTIMP/optics.py)
        Goodman, "Introduction to Fourier Optics", Chapter 3
    """

    def __init__(
        self,
        grid: Grid,
        distance: Optional[float] = None,
        fft_cache: Optional[FFTCache] = None,
    ):
        """
        Initialize angular spectrum propagator.

        Args:
            grid: Grid object defining spatial/frequency coordinates
            distance: Fixed propagation distance in meters (optional)
                     If None, distance must be specified in forward()
            fft_cache: Shared FFT cache for performance (optional)

        Raises:
            ValueError: If wavelength <= 0 or grid is invalid
        """
        super().__init__(fft_cache=fft_cache)

        # Validate
        if grid.wl <= 0:
            raise ValueError(f"Wavelength must be positive, got {grid.wl}")

        # Calculate angular spectrum transfer function components
        # Phase: sqrt(1/λ² - kx² - ky²)
        quad_phase = 1 / grid.wl**2 - grid.kx**2 - grid.ky**2

        # Diffraction limit: propagating waves have kx² + ky² < 1/λ²
        # Evanescent waves (kx² + ky² > 1/λ²) are filtered out
        self.register_buffer("diff_limit", quad_phase > 0)

        # Compute sqrt safely (zero out evanescent components)
        quad_phase_safe = torch.where(quad_phase > 0, quad_phase, torch.zeros_like(quad_phase))
        self.register_buffer("k_sqrt", quad_phase_safe.sqrt())

        # Store distance (if fixed)
        self.const_distance = distance is not None
        if distance is not None:
            self.register_buffer("distance", torch.tensor(distance))
        else:
            self.distance = None

    @property
    def diff_limit_tensor(self) -> Tensor:
        """Diffraction limit mask as Tensor (type-safe accessor for buffer)."""
        assert isinstance(self.diff_limit, Tensor)
        return self.diff_limit

    @property
    def k_sqrt_tensor(self) -> Tensor:
        """k_sqrt as Tensor (type-safe accessor for buffer)."""
        assert isinstance(self.k_sqrt, Tensor)
        return self.k_sqrt

    def forward(self, field: Tensor, **kwargs: Any) -> Tensor:
        """
        Propagate field using angular spectrum method.

        Args:
            field: Complex field tensor [H, W] or [B, C, H, W]
            **kwargs: Keyword arguments:
                distance: Propagation distance in meters
                    Required if not specified in __init__ (unless using direction)
                direction: Propagation direction for Telescope compatibility
                    - 'forward': spatial domain → k-space (FFT with transfer function)
                    - 'backward': k-space → spatial domain (IFFT with inverse transfer)
                    When specified, uses self.distance for propagation distance.

        Returns:
            Propagated complex field (same shape as input)

        Raises:
            ValueError: If distance not specified and not set in __init__

        Notes:
            - Transfer function: H = exp(i·2π·z·sqrt(1/λ² - kx² - ky²))
            - Evanescent waves (where kx²+ky² > 1/λ²) are filtered to zero
            - Exact within paraxial approximation
            - When using direction='backward', inverse transfer function is applied
        """
        # Extract parameters from kwargs
        distance: Optional[float] = kwargs.get("distance")
        direction: Optional[Literal["forward", "backward"]] = kwargs.get("direction")

        # Handle direction-based interface (for Telescope compatibility)
        if direction is not None:
            # Get propagation distance
            if distance is not None:
                z = distance
            elif self.distance is not None:
                z = self.distance.item()
            else:
                # Default to zero distance (pure FFT behavior) for Telescope compatibility
                z = 0.0

            if direction == "forward":
                # Spatial → k-space: FFT with transfer function
                field_k = self.fft_cache.fft2(field, norm="ortho")
                if z != 0.0:
                    prop_phase = 2 * torch.pi * z * self.k_sqrt_tensor
                    h_transfer = torch.exp(1j * prop_phase)
                    h_transfer = torch.where(
                        self.diff_limit_tensor, h_transfer, torch.zeros_like(h_transfer)
                    )
                    field_k = field_k * h_transfer
                return field_k
            elif direction == "backward":
                # k-space → spatial: IFFT with inverse transfer function
                if z != 0.0:
                    # Inverse transfer function: conjugate of H (equivalent to -z)
                    prop_phase = 2 * torch.pi * z * self.k_sqrt_tensor
                    h_inverse = torch.exp(-1j * prop_phase)
                    h_inverse = torch.where(
                        self.diff_limit_tensor, h_inverse, torch.zeros_like(h_inverse)
                    )
                    field = field * h_inverse
                return self.fft_cache.ifft2(field, norm="ortho")
            else:
                raise ValueError(f"Unknown direction: {direction}. Use 'forward' or 'backward'.")

        # Original distance-based interface (full ASP propagation)
        if distance is not None:
            z = distance
        elif self.distance is not None:
            z = self.distance.item()
        else:
            raise ValueError(
                "Propagation distance must be specified either in __init__ or forward() call"
            )

        # FFT to k-space (with cache)
        field_k = self.fft_cache.fft2(field, norm="ortho")

        # Angular spectrum transfer function
        # H(kx, ky, z) = exp(i · 2π · z · sqrt(1/λ² - kx² - ky²))
        prop_phase = 2 * torch.pi * z * self.k_sqrt_tensor
        h_transfer = torch.exp(1j * prop_phase)

        # Zero out evanescent waves
        h_transfer = torch.where(self.diff_limit_tensor, h_transfer, torch.zeros_like(h_transfer))

        # Apply transfer function and inverse FFT (with cache)
        propagated_k = field_k * h_transfer
        return self.fft_cache.ifft2(propagated_k, norm="ortho")

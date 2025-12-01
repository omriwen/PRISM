"""
Fresnel (near-field) diffraction using 1-step Impulse Response method.

This propagator implements the single-FFT Fresnel propagation algorithm,
which is efficient for medium-to-long distance propagation in the Fresnel
regime (0.1 < F < 10, where F = a²/(λz) is the Fresnel number).

Algorithm:
    U_out = (e^ikz / iλz) · e^(ik/(2z)r₂²) · FFT{U_in · e^(ik/(2z)r₁²)}

Output Grid Scaling:
    The output pixel size differs from input:
        dx_out = λ·z / (N·dx_in)

    This is a fundamental property of Fresnel diffraction - the observation
    grid "zooms out" as distance increases.

Valid Range:
    - Distance: z > z_crit = N·dx²/λ (critical distance for Nyquist sampling)
    - Fresnel number: 0.1 < F < 10 (intermediate field)
    - Use AngularSpectrumPropagator for z < z_crit
    - Use FraunhoferPropagator for F << 0.1 (far field)

Performance:
    - Computational cost: O(N² log N) - single FFT
    - Memory: O(N²) - same as input
    - ~2x faster than 2-step Transfer Function method
    - Comparable speed to AngularSpectrum but less accurate

Attributes:
    grid: Input spatial grid
    distance: Fixed propagation distance
    output_grid: Computed output grid (property)

Example:
    >>> from prism.core.grid import Grid
    >>> grid = Grid(nx=256, dx=10e-6, wavelength=520e-9)
    >>> prop = FresnelPropagator(grid, distance=0.1)
    >>>
    >>> output = prop(input_field)
    >>> print(f"Input px: {grid.dx:.2e}, Output px: {prop.output_grid.dx:.2e}")

References:
    - Goodman, J. W. "Introduction to Fourier Optics" (2005), Chapter 4
    - Schmidt, J. D. "Numerical Simulation of Optical Wave Propagation" (2010)

See Also:
    AngularSpectrumPropagator: Exact propagation for all distances
    FraunhoferPropagator: Fast far-field propagation (F << 1)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from loguru import logger
from torch import Tensor

from prism.core.grid import Grid
from prism.core.propagators.base import Propagator
from prism.utils.transforms import FFTCache


class FresnelPropagator(Propagator):
    """
    Fresnel (near-field) diffraction using 1-step Impulse Response method.

    This propagator implements the single-FFT Fresnel propagation algorithm,
    which is efficient for medium-to-long distance propagation in the Fresnel
    regime (0.1 < F < 10, where F = a²/(λz) is the Fresnel number).

    Algorithm:
        U_out = (e^ikz / iλz) · e^(ik/(2z)r₂²) · FFT{U_in · e^(ik/(2z)r₁²)}

    Output Grid Scaling:
        The output pixel size differs from input:
            dx_out = λ·z / (N·dx_in)

        This is a fundamental property of Fresnel diffraction - the observation
        grid "zooms out" as distance increases.

    Valid Range:
        - Distance: z > z_crit = N·dx²/λ (critical distance for Nyquist sampling)
        - Fresnel number: 0.1 < F < 10 (intermediate field)
        - Use AngularSpectrumPropagator for z < z_crit
        - Use FraunhoferPropagator for F << 0.1 (far field)

    Performance:
        - Computational cost: O(N² log N) - single FFT
        - Memory: O(N²) - same as input
        - ~2x faster than 2-step Transfer Function method
        - Comparable speed to AngularSpectrum but less accurate

    Attributes:
        grid: Input spatial grid
        distance: Fixed propagation distance
        output_grid: Computed output grid (property)

    Example:
        >>> from prism.core.grid import Grid
        >>> grid = Grid(nx=256, dx=10e-6, wavelength=520e-9)
        >>> prop = FresnelPropagator(grid, distance=0.1)
        >>>
        >>> output = prop(input_field)
        >>> print(f"Input px: {grid.dx:.2e}, Output px: {prop.output_grid.dx:.2e}")

    References:
        - Goodman, J. W. "Introduction to Fourier Optics" (2005), Chapter 4
        - Schmidt, J. D. "Numerical Simulation of Optical Wave Propagation" (2010)

    See Also:
        AngularSpectrumPropagator: Exact propagation for all distances
        FraunhoferPropagator: Fast far-field propagation (F << 1)
    """

    def __init__(
        self,
        grid: Grid,
        distance: float,
        fft_cache: Optional[FFTCache] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize FresnelPropagator with Grid-based API.

        Args:
            grid: Input spatial grid (contains wavelength, pixel size, and grid size)
            distance: Fixed propagation distance (meters). Can be negative for backward propagation.
            fft_cache: Shared FFT cache for performance (optional)
            **kwargs: Additional arguments (for future compatibility)

        Raises:
            ValueError: If grid or distance parameters are invalid

        Example:
            >>> grid = Grid(nx=256, dx=10e-6, wavelength=520e-9)
            >>> prop = FresnelPropagator(grid, distance=0.1)
        """
        super().__init__(fft_cache=fft_cache)

        # Store grid as attribute
        self.grid = grid

        # Validate distance
        if abs(distance) < grid.wl:
            raise ValueError(
                f"Distance ({distance:.2e} m) too small. Must be >> wavelength ({grid.wl:.2e} m)"
            )

        # Register distance as buffer (allows device movement)
        self.register_buffer("distance", torch.tensor(distance, dtype=torch.get_default_dtype()))

        # Critical distance check (Nyquist sampling requirement)
        z_crit = (grid.nx * grid.dx**2) / grid.wl
        if abs(distance) < z_crit:
            logger.warning(
                f"Distance z={distance:.3e}m is below critical distance "
                f"z_crit={z_crit:.3e}m. Nyquist sampling may be violated. "
                f"Consider using AngularSpectrumPropagator for z < z_crit."
            )

        # Fresnel number check (informational)
        fov = grid.nx * grid.dx
        fresnel_number = (fov / 2) ** 2 / (grid.wl * abs(distance))
        logger.info(
            f"Fresnel propagator initialized: z={distance:.3e}m, "
            f"F={fresnel_number:.2e} (valid range: 0.1 < F < 10)"
        )

        if fresnel_number < 0.1:
            logger.warning(
                f"Fresnel number F={fresnel_number:.2e} < 0.1. "
                f"Far-field regime - consider using FraunhoferPropagator (faster)."
            )
        elif fresnel_number > 10:
            logger.warning(
                f"Fresnel number F={fresnel_number:.2e} > 10. "
                f"Near-field regime - consider using AngularSpectrumPropagator (more accurate)."
            )

    def _compute_pre_chirp(self, z: float) -> Tensor:
        """
        Compute spatial domain quadratic phase (pre-chirp).

        This is applied to the input field before FFT.

        Args:
            z: Propagation distance (can be negative for backward)

        Returns:
            Pre-chirp phase factor: exp(i*k/(2z) * (x² + y²))

        Reference:
            Goodman (2005), Eq. 4-13 for Fresnel diffraction
        """
        k = 2 * torch.pi / self.grid.wl

        # Get input grid coordinates (already in broadcasting-compatible shapes)
        x, y = self.grid.x, self.grid.y  # Shape: (1, nx) and (ny, 1)

        # Quadratic phase (broadcasting handles meshgrid automatically)
        phase = (k / (2 * z)) * (x**2 + y**2)
        return torch.exp(1j * phase)

    def _compute_post_chirp(self, z: float) -> Tensor:
        """
        Compute output domain quadratic phase (post-chirp).

        This is applied after FFT, using OUTPUT grid coordinates.

        Args:
            z: Propagation distance

        Returns:
            Post-chirp phase factor: exp(i*k/(2z) * (x₂² + y₂²))

        Note:
            Output grid has different pixel size than input!
            dx_out = λz / (N·dx_in)
        """
        k = 2 * torch.pi / self.grid.wl
        n_pixels = self.grid.nx

        # Compute OUTPUT grid coordinates (scaled!)
        dx_out = (self.grid.wl * abs(z)) / (n_pixels * self.grid.dx)

        # Create output coordinate arrays (broadcasting-compatible shapes)
        x2 = (
            torch.arange(
                -n_pixels // 2,
                n_pixels // 2,
                dtype=torch.get_default_dtype(),
                device=self.grid.x.device,
            )
            * dx_out
        ).unsqueeze(0)  # Shape: (1, n_pixels)
        y2 = (
            torch.arange(
                -n_pixels // 2,
                n_pixels // 2,
                dtype=torch.get_default_dtype(),
                device=self.grid.y.device,
            )
            * dx_out
        ).unsqueeze(1)  # Shape: (n_pixels, 1)

        # Quadratic phase (use signed z for direction, broadcasting handles 2D)
        phase = (k / (2 * z)) * (x2**2 + y2**2)
        return torch.exp(1j * phase)

    @property
    def output_grid(self) -> Grid:
        """
        Compute output grid after propagation.

        The 1-step Fresnel method changes pixel size:
            dx_out = λ·z / (N·dx_in)

        Returns:
            Grid object with output coordinates

        Example:
            >>> prop = FresnelPropagator(grid, distance=0.1)
            >>> output_field = prop(input_field)
            >>> print(f"Output pixel size: {prop.output_grid.dx:.2e} m")
        """
        dx_out = (self.grid.wl * abs(self.distance.item())) / (self.grid.nx * self.grid.dx)

        return Grid(
            nx=self.grid.nx,
            ny=self.grid.ny,
            dx=dx_out,
            dy=dx_out,
            wavelength=self.grid.wl,
        )

    def forward(self, field: Tensor, **kwargs: Any) -> Tensor:  # type: ignore[override]
        """
        Propagate field using 1-step Fresnel (Impulse Response) method.

        Algorithm:
            U_out = (e^ikz / iλz) · e^(ik/(2z)r₂²) · FFT{U_in · e^(ik/(2z)r₁²)}

        where:
            - r₁ = input coordinates (x₁, y₁)
            - r₂ = output coordinates (x₂, y₂) with dx₂ = λz/(N·dx₁)
            - k = 2π/λ

        Args:
            field: Complex input field [N, N]
            **kwargs: Optional parameters
                - direction: "forward" (default) or "backward"

        Returns:
            Complex output field [N, N] on scaled output grid

        Note:
            Output has same pixel count but different pixel size!
            Access via self.output_grid property.

        Reference:
            Goodman (2005), Eq. 4-13, 4-14, 4-15
        """
        # Get direction
        direction = kwargs.get("direction", "forward")
        z = self.distance.item() if direction == "forward" else -self.distance.item()

        k = 2 * torch.pi / self.grid.wl

        # Step 1: Apply pre-chirp (spatial domain)
        pre_chirp = self._compute_pre_chirp(z)
        field_chirped = field * pre_chirp

        # Step 2: FFT with orthonormal normalization
        field_fft = self.fft_cache.fft2(field_chirped, norm="ortho")

        # Step 3: Apply post-chirp (output domain, scaled coordinates)
        post_chirp = self._compute_post_chirp(z)
        field_propagated = field_fft * post_chirp

        # Step 4: Apply normalization factor
        # Factor: (e^ikz / iλz) * dx²
        propagation_phase = torch.exp(torch.tensor(1j * k * z, dtype=torch.complex64))
        # Convert 1j to torch complex tensor
        normalization = propagation_phase / torch.tensor(
            1j * self.grid.wl * z, dtype=torch.complex64
        )
        discrete_scaling = self.grid.dx**2

        factor = normalization * discrete_scaling

        # Step 5: Final result
        output = factor * field_propagated

        return output


# Backward compatibility: FreeSpacePropagator is now FresnelPropagator
FreeSpacePropagator = FresnelPropagator

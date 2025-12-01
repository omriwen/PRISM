"""Thin lens optical element implementation.

This module provides a ``ThinLens`` class that implements the thin lens
approximation for modeling converging and diverging lenses in optical systems.

The thin lens introduces a quadratic phase shift that converts plane waves
to spherical waves (converging) or vice versa (diverging), which is the
fundamental operation for imaging and Fourier transformation.

Notes
-----
The thin lens approximation assumes:

1. The lens thickness is negligible compared to the focal length
2. Paraxial (small angle) approximation is valid
3. No aberrations (ideal lens)

For microscope objectives, these assumptions are typically valid for
the mathematical model, though real objectives have aberration corrections.

See Also
--------
MicroscopeForwardModel : Forward model using thin lenses
spids.core.propagators : Wave propagation methods

References
----------
.. [1] Goodman, J. W. "Introduction to Fourier Optics", Chapter 5.
.. [2] Born, M. and Wolf, E. "Principles of Optics", Chapter 4.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from prism.core.grid import Grid


class ThinLens(nn.Module):
    """Thin lens element with quadratic phase transformation.

    A thin lens introduces a phase shift proportional to the square
    of the radial distance from the optical axis:

        φ(x, y) = -k/(2f) × (x² + y²)

    where k = 2π/λ is the wavenumber and f is the focal length.

    Parameters
    ----------
    focal_length : float
        Focal length in meters. Positive for converging lens.
    grid : Grid
        Spatial grid defining coordinates.
    aperture_diameter : float, optional
        Lens aperture diameter in meters. If None, no aperture clipping.

    Attributes
    ----------
    focal_length : float
        Focal length of the lens.
    grid : Grid
        Associated spatial grid.
    lens_phase : Tensor
        Precomputed complex lens phase factor (registered buffer).
    pupil : Tensor
        Binary aperture mask (registered buffer).

    Notes
    -----
    The lens phase is precomputed during initialization and stored as a
    registered buffer, making repeated applications efficient.

    For a converging lens (f > 0), the phase is negative at the edges,
    causing wavefront delay that focuses parallel rays. For a diverging
    lens (f < 0), the phase is positive at the edges.

    See Also
    --------
    MicroscopeForwardModel : Uses ThinLens for objective and tube lenses
    spids.core.grid.Grid.lens_ft_grid : Compute focal plane grid scaling

    Examples
    --------
    >>> from prism.core.grid import Grid
    >>> grid = Grid(nx=128, dx=1e-6, wavelength=532e-9)
    >>> lens = ThinLens(focal_length=0.01, grid=grid)
    >>> field = torch.ones(128, 128, dtype=torch.complex64)
    >>> output = lens(field)  # Apply lens transformation

    Creating a lens with aperture clipping:

    >>> lens_with_aperture = ThinLens(
    ...     focal_length=0.01,
    ...     grid=grid,
    ...     aperture_diameter=50e-6,  # 50 µm aperture
    ... )
    """

    def __init__(
        self,
        focal_length: float,
        grid: Grid,
        aperture_diameter: Optional[float] = None,
    ) -> None:
        super().__init__()

        if focal_length == 0:
            raise ValueError("Focal length cannot be zero")

        self.focal_length = focal_length
        self.grid = grid
        self.aperture_diameter = aperture_diameter

        # Precompute lens phase and pupil
        self._precompute_lens_function()

    def _precompute_lens_function(self) -> None:
        """Precompute lens phase factor and aperture pupil."""
        x, y = self.grid.x, self.grid.y
        k = 2 * torch.pi / self.grid.wl

        # Quadratic phase: φ = -k·r²/(2f)
        # x is shape (1, nx), y is shape (ny, 1), broadcasting gives (ny, nx)
        r_squared = x**2 + y**2
        phase = -k / (2 * self.focal_length) * r_squared
        lens_phase = torch.exp(1j * phase)
        self.register_buffer("lens_phase", lens_phase)

        # Aperture pupil (if specified)
        if self.aperture_diameter is not None:
            r = torch.sqrt(r_squared)
            pupil = (r <= self.aperture_diameter / 2).to(lens_phase.dtype)
        else:
            pupil = torch.ones_like(lens_phase)
        self.register_buffer("pupil", pupil)

    def forward(self, field: Tensor) -> Tensor:
        """Apply thin lens transformation to input field.

        Parameters
        ----------
        field : Tensor
            Complex input field.

        Returns
        -------
        Tensor
            Field after lens transformation.
        """
        return field * self.pupil * self.lens_phase  # type: ignore[operator]  # lens_phase is Tensor at runtime

    @property
    def output_grid(self) -> Grid:
        """Get grid in the lens focal plane (Fourier transform plane).

        Returns
        -------
        Grid
            Grid with scaled pixel size for the focal plane.
        """
        return self.grid.lens_ft_grid(self.focal_length)

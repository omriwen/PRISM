"""Unified aperture mask generator for all instruments.

This module provides a unified interface for generating aperture and pupil masks
across different optical instruments (microscopes, telescopes, cameras). It supports
various mask geometries and can work with both numerical aperture (NA) specifications
and physical/pixel radius specifications.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from ..grid import Grid


class ApertureMaskGenerator:
    """Unified aperture mask generator for all instruments.

    This class provides methods to generate various types of aperture and pupil
    masks for optical systems. It handles coordinate transformations and supports
    both physical (NA-based) and geometric (pixel/radius-based) specifications.

    Supported mask types:
        - Circular pupils (NA-limited for microscopes, physical for telescopes)
        - Annular pupils (darkfield microscopy)
        - Phase rings (phase contrast microscopy)
        - Sub-apertures (SPIDS synthetic aperture)
        - Hexagonal apertures (segmented telescopes)
        - Obscured apertures (central obstruction)

    Parameters
    ----------
    grid : Grid
        Spatial/frequency grid defining the coordinate system.
    cutoff_type : str, default='na'
        How to interpret radius specifications:
        - 'na': Use numerical aperture (NA) and wavelength to compute frequency cutoff
        - 'physical': Use physical radius in meters
        - 'pixels': Use radius directly in pixel units
    wavelength : float, optional
        Wavelength in meters. Required when cutoff_type='na'.
    medium_index : float, default=1.0
        Refractive index of the medium (1.0 for air, 1.33 for water, 1.515 for oil).

    Examples
    --------
    Create mask generator for microscope (NA-based):
    >>> grid = Grid(nx=512, dx=10e-6, wavelength=550e-9)
    >>> generator = ApertureMaskGenerator(grid, cutoff_type='na', wavelength=550e-9, medium_index=1.0)
    >>> mask = generator.circular(na=1.4)

    Create mask generator for telescope (pixel-based):
    >>> grid = Grid(nx=512, dx=1e-6, wavelength=550e-9)
    >>> generator = ApertureMaskGenerator(grid, cutoff_type='pixels')
    >>> mask = generator.circular(radius=50)

    Create sub-aperture for SPIDS:
    >>> mask = generator.sub_aperture(center=[10, 5], radius=15)
    """

    def __init__(
        self,
        grid: Grid,
        cutoff_type: str = "na",
        wavelength: Optional[float] = None,
        medium_index: float = 1.0,
    ) -> None:
        """Initialize aperture mask generator.

        Parameters
        ----------
        grid : Grid
            Spatial/frequency grid defining the coordinate system.
        cutoff_type : str, default='na'
            How to interpret radius specifications ('na', 'physical', or 'pixels').
        wavelength : float, optional
            Wavelength in meters. Required when cutoff_type='na'.
        medium_index : float, default=1.0
            Refractive index of the medium.

        Raises
        ------
        ValueError
            If cutoff_type='na' but wavelength is not provided.
            If cutoff_type is not one of 'na', 'physical', 'pixels'.
        """
        self.grid = grid
        self.cutoff_type = cutoff_type
        self.wavelength = wavelength if wavelength is not None else grid.wl
        self.medium_index = medium_index

        # Validate cutoff_type
        valid_types = {"na", "physical", "pixels"}
        if cutoff_type not in valid_types:
            raise ValueError(f"cutoff_type must be one of {valid_types}, got '{cutoff_type}'")

        # Validate wavelength for NA-based cutoffs
        if cutoff_type == "na" and self.wavelength is None:
            raise ValueError("wavelength must be provided when cutoff_type='na'")

        # Device tracking
        self._device = torch.device("cpu")

    def _radius_to_frequency_cutoff(
        self,
        radius: Optional[float] = None,
        na: Optional[float] = None,
    ) -> float:
        """Convert radius specification to frequency cutoff.

        Parameters
        ----------
        radius : float, optional
            Radius in units determined by cutoff_type.
        na : float, optional
            Numerical aperture (only used when cutoff_type='na').

        Returns
        -------
        float
            Frequency cutoff in 1/meters.

        Raises
        ------
        ValueError
            If neither radius nor na is provided, or if na is provided but cutoff_type != 'na'.
        """
        if self.cutoff_type == "na":
            if na is None:
                raise ValueError("NA must be provided when cutoff_type='na'")
            # Frequency cutoff: f_cutoff = NA / (n * λ)
            return na / (self.medium_index * self.wavelength)

        elif self.cutoff_type == "physical":
            if radius is None:
                raise ValueError("radius must be provided when cutoff_type='physical'")
            # Convert physical radius to frequency
            # For physical aperture: f_cutoff ≈ radius / (grid.fov)
            return radius / max(self.grid.fov)

        elif self.cutoff_type == "pixels":
            if radius is None:
                raise ValueError("radius must be provided when cutoff_type='pixels'")
            # Convert pixel radius to frequency
            # Frequency resolution: df = 1 / FOV
            df = 1.0 / max(self.grid.fov)
            return radius * df

        else:
            raise ValueError(f"Unknown cutoff_type: {self.cutoff_type}")

    def circular(
        self,
        radius: Optional[float] = None,
        na: Optional[float] = None,
        center: Optional[List[float]] = None,
    ) -> Tensor:
        """Generate circular aperture/pupil mask.

        Creates a binary circular mask in the frequency domain (k-space).
        For microscopes, this represents the objective NA. For telescopes,
        this represents the aperture diameter.

        Parameters
        ----------
        radius : float, optional
            Radius in units specified by cutoff_type.
        na : float, optional
            Numerical aperture (only for cutoff_type='na').
        center : List[float], optional
            Center position [y, x] in pixels from DC. Defaults to [0, 0].

        Returns
        -------
        Tensor
            Binary mask of shape (n_pixels, n_pixels), dtype float32.

        Examples
        --------
        >>> # Microscope: NA-based circular pupil
        >>> mask = generator.circular(na=1.4)
        >>>
        >>> # Telescope: pixel-based circular aperture
        >>> mask = generator.circular(radius=50)
        >>>
        >>> # Off-center sub-aperture
        >>> mask = generator.circular(radius=25, center=[10, 5])
        """
        if center is None:
            center = [0.0, 0.0]

        # Get frequency cutoff
        cutoff_freq = self._radius_to_frequency_cutoff(radius=radius, na=na)

        # Get frequency coordinates
        fx = self.grid.kx.to(self._device)
        fy = self.grid.ky.to(self._device)

        # Center offset in frequency space
        cy, cx = center
        # Convert pixel offset to frequency offset (if needed)
        if self.cutoff_type == "pixels":
            df = 1.0 / max(self.grid.fov)
            cy_freq = cy * df
            cx_freq = cx * df
        else:
            cy_freq = cy
            cx_freq = cx

        # Compute radial distance from center
        r_freq = torch.sqrt((fx - cx_freq) ** 2 + (fy - cy_freq) ** 2)

        # Create circular mask
        mask = (r_freq <= cutoff_freq).float()

        return mask

    def annular(
        self,
        inner_radius: Optional[float] = None,
        outer_radius: Optional[float] = None,
        inner_na: Optional[float] = None,
        outer_na: Optional[float] = None,
        center: Optional[List[float]] = None,
    ) -> Tensor:
        """Generate annular aperture/pupil mask.

        Creates a ring-shaped mask (outer circle minus inner circle).
        Used for darkfield microscopy where the direct beam is blocked.

        Parameters
        ----------
        inner_radius : float, optional
            Inner radius in units specified by cutoff_type.
        outer_radius : float, optional
            Outer radius in units specified by cutoff_type.
        inner_na : float, optional
            Inner numerical aperture (only for cutoff_type='na').
        outer_na : float, optional
            Outer numerical aperture (only for cutoff_type='na').
        center : List[float], optional
            Center position [y, x] in pixels from DC. Defaults to [0, 0].

        Returns
        -------
        Tensor
            Binary mask of shape (n_pixels, n_pixels), dtype float32.

        Examples
        --------
        >>> # Darkfield microscopy: annular illumination
        >>> mask = generator.annular(inner_na=0.8, outer_na=1.3)
        >>>
        >>> # Pixel-based annular mask
        >>> mask = generator.annular(inner_radius=20, outer_radius=50)
        """
        if center is None:
            center = [0.0, 0.0]

        # Get frequency cutoffs
        if self.cutoff_type == "na":
            if inner_na is None or outer_na is None:
                raise ValueError("inner_na and outer_na must be provided when cutoff_type='na'")
            inner_cutoff = self._radius_to_frequency_cutoff(na=inner_na)
            outer_cutoff = self._radius_to_frequency_cutoff(na=outer_na)
        else:
            if inner_radius is None or outer_radius is None:
                raise ValueError("inner_radius and outer_radius must be provided")
            inner_cutoff = self._radius_to_frequency_cutoff(radius=inner_radius)
            outer_cutoff = self._radius_to_frequency_cutoff(radius=outer_radius)

        # Get frequency coordinates
        fx = self.grid.kx.to(self._device)
        fy = self.grid.ky.to(self._device)

        # Center offset
        cy, cx = center
        if self.cutoff_type == "pixels":
            df = 1.0 / max(self.grid.fov)
            cy_freq = cy * df
            cx_freq = cx * df
        else:
            cy_freq = cy
            cx_freq = cx

        # Compute radial distance
        r_freq = torch.sqrt((fx - cx_freq) ** 2 + (fy - cy_freq) ** 2)

        # Create annular mask
        mask = ((r_freq > inner_cutoff) & (r_freq <= outer_cutoff)).float()

        return mask

    def phase_ring(
        self,
        radius: Optional[float] = None,
        ring_inner: float = 0.6,
        ring_outer: float = 0.8,
        phase_shift: float = np.pi / 2,
        na: Optional[float] = None,
    ) -> Tensor:
        """Generate phase contrast ring mask.

        Creates a mask with a phase-shifted ring region for phase contrast microscopy.
        The ring typically corresponds to the direct (unscattered) light, while the
        rest of the pupil transmits scattered light without phase shift.

        Parameters
        ----------
        radius : float
            Overall pupil radius in units specified by cutoff_type.
        ring_inner : float
            Inner ring radius (normalized to radius, 0-1 range).
        ring_outer : float
            Outer ring radius (normalized to radius, 0-1 range).
        phase_shift : float, default=π/2
            Phase shift applied to the ring region (radians).
        na : float, optional
            Numerical aperture for overall pupil (only for cutoff_type='na').

        Returns
        -------
        Tensor
            Complex-valued mask of shape (n_pixels, n_pixels), dtype complex64.

        Examples
        --------
        >>> # Phase contrast microscopy with π/2 phase ring
        >>> mask = generator.phase_ring(na=1.4, ring_inner=0.6, ring_outer=0.8)
        >>>
        >>> # Custom phase shift
        >>> mask = generator.phase_ring(radius=50, ring_inner=0.5, ring_outer=0.7, phase_shift=np.pi/4)
        """
        # Get frequency cutoff for overall pupil
        cutoff_freq = self._radius_to_frequency_cutoff(radius=radius, na=na)

        # Get frequency coordinates
        fx = self.grid.kx.to(self._device)
        fy = self.grid.ky.to(self._device)

        # Compute normalized radial distance
        r_freq = torch.sqrt(fx**2 + fy**2)
        r_norm = r_freq / cutoff_freq

        # Initialize complex mask (unity transmission)
        mask = torch.ones_like(r_norm, dtype=torch.complex64)

        # Apply phase shift to ring region
        ring_mask = (r_norm > ring_inner) & (r_norm < ring_outer)
        phase_shift_tensor = torch.tensor(phase_shift, dtype=r_norm.dtype, device=r_norm.device)
        mask[ring_mask] *= torch.exp(1j * phase_shift_tensor)

        # Zero out regions beyond pupil
        mask[r_norm > 1.0] = 0

        return mask

    def sub_aperture(
        self,
        center: List[float],
        radius: Optional[float] = None,
        na: Optional[float] = None,
    ) -> Tensor:
        """Generate sub-aperture mask for SPIDS synthetic aperture.

        Creates a circular aperture at a specified k-space position.
        This is used for SPIDS progressive synthetic aperture reconstruction,
        where different regions of k-space are sampled sequentially.

        Parameters
        ----------
        center : List[float]
            Center position [y, x] in pixels from DC.
        radius : float
            Sub-aperture radius in units specified by cutoff_type.
        na : float, optional
            Numerical aperture for sub-aperture (only for cutoff_type='na').

        Returns
        -------
        Tensor
            Binary mask of shape (n_pixels, n_pixels), dtype float32.

        Examples
        --------
        >>> # SPIDS sub-aperture at offset position
        >>> mask = generator.sub_aperture(center=[10, 5], radius=15)
        >>>
        >>> # NA-based sub-aperture
        >>> mask = generator.sub_aperture(center=[0, 20], na=0.3)
        """
        return self.circular(radius=radius, na=na, center=center)

    def hexagonal(
        self,
        radius: Optional[float] = None,
        na: Optional[float] = None,
        center: Optional[List[float]] = None,
    ) -> Tensor:
        """Generate hexagonal aperture mask.

        Creates a regular hexagon aperture, useful for segmented telescope mirrors
        (e.g., James Webb Space Telescope) or hexagonal pixel arrays.

        Parameters
        ----------
        radius : float
            Circumradius (center to vertex) in units specified by cutoff_type.
        na : float, optional
            Numerical aperture for circumscribed circle (only for cutoff_type='na').
        center : List[float], optional
            Center position [y, x] in pixels from DC. Defaults to [0, 0].

        Returns
        -------
        Tensor
            Binary mask of shape (n_pixels, n_pixels), dtype float32.

        Examples
        --------
        >>> # Hexagonal telescope aperture
        >>> mask = generator.hexagonal(radius=50)
        >>>
        >>> # Hexagonal microscope pupil
        >>> mask = generator.hexagonal(na=1.2)

        Notes
        -----
        The hexagon is oriented with a flat edge at the top (vertex pointing up).
        """
        if center is None:
            center = [0.0, 0.0]

        # Get frequency cutoff (for circumradius)
        cutoff_freq = self._radius_to_frequency_cutoff(radius=radius, na=na)

        # Get frequency coordinates
        fx = self.grid.kx.to(self._device)
        fy = self.grid.ky.to(self._device)

        # Center offset
        cy, cx = center
        if self.cutoff_type == "pixels":
            df = 1.0 / max(self.grid.fov)
            cy_freq = cy * df
            cx_freq = cx * df
        else:
            cy_freq = cy
            cx_freq = cx

        # Shift coordinates to center
        fx_shifted = fx - cx_freq
        fy_shifted = fy - cy_freq

        # Hexagon is defined by 6 half-planes
        # Vertices at angles: 0°, 60°, 120°, 180°, 240°, 300°
        # Half-plane normals point inward at angles: 30°, 90°, 150°, 210°, 270°, 330°

        # For a regular hexagon with circumradius R:
        # - The inradius (center to edge center) is R * sqrt(3)/2
        # - Distance from center to edge along normal is R * sqrt(3)/2

        inradius = cutoff_freq * np.sqrt(3) / 2

        # Six half-plane constraints (all must be satisfied)
        # Normal vectors at 30° intervals
        angles = torch.tensor(
            [30, 90, 150, 210, 270, 330], dtype=torch.float32, device=self._device
        )
        angles_rad = angles * np.pi / 180

        # Initialize mask (all True)
        mask = torch.ones_like(fx_shifted, dtype=torch.bool)

        # Apply each half-plane constraint
        for angle in angles_rad:
            # Normal vector components
            nx = torch.cos(angle)
            ny = torch.sin(angle)

            # Distance from point to edge along normal
            # Constraint: nx * x + ny * y <= inradius
            distance = nx * fx_shifted + ny * fy_shifted
            mask = mask & (distance <= inradius)

        return mask.float()

    def obscured(
        self,
        outer_radius: Optional[float] = None,
        inner_radius: Optional[float] = None,
        outer_na: Optional[float] = None,
        inner_na: Optional[float] = None,
        center: Optional[List[float]] = None,
    ) -> Tensor:
        """Generate obscured aperture mask (central obstruction).

        Creates a circular aperture with a central circular obstruction.
        Common in reflecting telescopes where the secondary mirror blocks
        the center of the primary mirror.

        Parameters
        ----------
        outer_radius : float, optional
            Outer aperture radius in units specified by cutoff_type.
        inner_radius : float, optional
            Inner obstruction radius in units specified by cutoff_type.
        outer_na : float, optional
            Outer numerical aperture (only for cutoff_type='na').
        inner_na : float, optional
            Inner numerical aperture (only for cutoff_type='na').
        center : List[float], optional
            Center position [y, x] in pixels from DC. Defaults to [0, 0].

        Returns
        -------
        Tensor
            Binary mask of shape (n_pixels, n_pixels), dtype float32.

        Examples
        --------
        >>> # Cassegrain telescope with 30% central obstruction
        >>> mask = generator.obscured(outer_radius=50, inner_radius=15)
        >>>
        >>> # NA-based obscured pupil
        >>> mask = generator.obscured(outer_na=1.4, inner_na=0.4)

        Notes
        -----
        This is equivalent to annular() but with clearer naming for telescope applications.
        """
        return self.annular(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            inner_na=inner_na,
            outer_na=outer_na,
            center=center,
        )

    def to(self, device: torch.device) -> "ApertureMaskGenerator":
        """Move generator to specified device.

        Parameters
        ----------
        device : torch.device
            Target device (e.g., torch.device("cuda")).

        Returns
        -------
        ApertureMaskGenerator
            Self for method chaining.
        """
        self._device = device
        return self

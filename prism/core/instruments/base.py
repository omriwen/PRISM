"""Base classes for optical instruments in SPIDS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch

from ..grid import Grid
from ..propagators.base import Propagator


@dataclass
class InstrumentConfig:
    """Base configuration for all optical instruments.

    Attributes:
        wavelength: Operating wavelength in meters
        n_pixels: Number of pixels in computational grid
        pixel_size: Physical pixel size of detector in meters
        grid_size: Optional total grid size in meters
    """

    wavelength: float = 550e-9  # Default green light
    n_pixels: int = 1024
    pixel_size: float = 6.5e-6  # Typical camera sensor
    grid_size: Optional[float] = None

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.wavelength <= 0:
            raise ValueError(f"Wavelength must be positive, got {self.wavelength}")
        if self.n_pixels <= 0:
            raise ValueError(f"Number of pixels must be positive, got {self.n_pixels}")
        if self.pixel_size <= 0:
            raise ValueError(f"Pixel size must be positive, got {self.pixel_size}")
        if self.grid_size is not None and self.grid_size <= 0:
            raise ValueError(f"Grid size must be positive, got {self.grid_size}")


class Instrument(ABC):
    """Abstract base class for optical instruments.

    This class defines the common interface for all optical instruments
    (telescopes, microscopes, cameras) in SPIDS.
    """

    def __init__(self, config: InstrumentConfig):
        """Initialize instrument with configuration.

        Args:
            config: Instrument configuration parameters
        """
        config.validate()
        self.config = config
        self._grid: Optional[Grid] = None
        self._propagator: Optional[Propagator] = None

    @property
    def grid(self) -> Grid:
        """Get computational grid (lazy initialization)."""
        if self._grid is None:
            self._grid = self._create_grid()
        return self._grid

    @property
    def propagator(self) -> Propagator:
        """Get propagator (lazy initialization)."""
        if self._propagator is None:
            self._propagator = self._select_propagator()
        return self._propagator

    @abstractmethod
    def _create_grid(self) -> Grid:
        """Create computational grid for instrument.

        Returns:
            Grid instance configured for this instrument
        """
        pass

    @abstractmethod
    def _select_propagator(self) -> Propagator:
        """Select appropriate propagator for instrument.

        Returns:
            Propagator instance suitable for this instrument
        """
        pass

    @abstractmethod
    def compute_psf(self, **kwargs: Any) -> torch.Tensor:
        """Compute point spread function.

        Args:
            **kwargs: Instrument-specific PSF parameters

        Returns:
            PSF tensor (2D or 3D depending on instrument)
        """
        pass

    @abstractmethod
    def forward(
        self,
        field: torch.Tensor,
        illumination_mode: Optional[str] = None,
        illumination_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward propagation through instrument.

        Args:
            field: Input field or intensity
            illumination_mode: Type of illumination ('brightfield', 'darkfield',
                             'phase', 'dic', 'custom', etc.). None uses default.
            illumination_params: Parameters for illumination (e.g., NA_illumination,
                               annular_ratio, phase_ring_params, custom_pupil, etc.)
            **kwargs: Additional instrument-specific parameters

        Returns:
            Output field or intensity at detector

        Notes:
            - Brightfield: Direct illumination collected
            - Darkfield: Only scattered light collected (direct blocked)
            - Mixed modes: Partial overlap of illumination/detection pupils
            - Custom: User-defined illumination/detection pupils
        """
        pass

    @property
    @abstractmethod
    def resolution_limit(self) -> float:
        """Theoretical resolution limit.

        Returns:
            Resolution limit in appropriate units (radians for telescopes,
            meters for microscopes/cameras)
        """
        pass

    def get_instrument_type(self) -> str:
        """Return instrument type identifier.

        Returns:
            Lowercase instrument class name
        """
        return self.__class__.__name__.lower()

    def get_info(self) -> dict:
        """Get instrument information summary.

        Returns:
            Dictionary with instrument parameters and characteristics
        """
        return {
            "type": self.get_instrument_type(),
            "wavelength": self.config.wavelength,
            "n_pixels": self.config.n_pixels,
            "pixel_size": self.config.pixel_size,
            "resolution_limit": self.resolution_limit,
            "grid_fov": self.grid.fov if hasattr(self, "_grid") and self._grid else None,
        }

    def validate_field(
        self,
        field: torch.Tensor,
        input_mode: str = "auto",
        input_pixel_size: Optional[float] = None,
    ) -> torch.Tensor:
        """Validate and prepare input field for wave propagation.

        Parameters
        ----------
        field : torch.Tensor
            Input field tensor with shape (..., H, W).
        input_mode : str
            How to interpret input values:
            - 'intensity': Field is I = |E|^2, converted via sqrt(I)
            - 'amplitude': Field is |E|, values >= 0
            - 'complex': Field is already complex E = |E|*exp(i*phi)
            - 'auto' (default): Auto-detect from dtype and values
        input_pixel_size : float, optional
            Physical size of input pixels (meters) for FOV validation.

        Returns
        -------
        torch.Tensor
            Validated and converted complex field.

        Raises
        ------
        ValueError
            If field shape doesn't match expected grid size.
        """
        from prism.core.optics.input_handling import (
            InputMode,
            prepare_field,
            validate_fov_consistency,
        )

        expected_shape = (self.config.n_pixels, self.config.n_pixels)

        mode_map = {
            "amplitude": InputMode.AMPLITUDE,
            "intensity": InputMode.INTENSITY,
            "complex": InputMode.COMPLEX,
            "auto": InputMode.AUTO,
        }

        if input_mode not in mode_map:
            raise ValueError(
                f"Invalid input_mode '{input_mode}'. Valid options: {list(mode_map.keys())}"
            )

        # FOV consistency check
        if hasattr(self, "_grid") and self._grid is not None:
            validate_fov_consistency(input_pixel_size, self._grid.dx)

        return prepare_field(
            field=field,
            expected_shape=expected_shape,
            input_mode=mode_map[input_mode],
        )

    def __repr__(self) -> str:
        """String representation of instrument."""
        return (
            f"{self.__class__.__name__}("
            f"wavelength={self.config.wavelength:.2e}, "
            f"n_pixels={self.config.n_pixels}, "
            f"resolution={self.resolution_limit:.2e})"
        )

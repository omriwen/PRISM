"""Microscope scenario configuration for SPIDS.

This module provides user-friendly configuration for microscope systems with
automatic parameter calculation and validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import cast

from prism.core.instruments import MicroscopeConfig

from .base import ScenarioConfig


@dataclass
class ObjectiveSpec:
    """Microscope objective specification.

    Attributes:
        magnification: Total magnification (e.g., 40, 100)
        numerical_aperture: NA value (0 < NA <= medium_index)
        immersion_medium: 'air', 'water', or 'oil'
        medium_index: Refractive index of immersion medium
    """

    magnification: float
    numerical_aperture: float
    immersion_medium: str = "air"
    medium_index: float = 1.0

    def __post_init__(self) -> None:
        """Set medium index based on immersion medium."""
        if self.immersion_medium == "air":
            self.medium_index = 1.0
        elif self.immersion_medium == "water":
            self.medium_index = 1.33
        elif self.immersion_medium == "oil":
            self.medium_index = 1.515
        else:
            # Custom medium, keep the provided index
            pass

    @classmethod
    def from_string(cls, spec: str) -> ObjectiveSpec:
        """Parse objective specification string.

        Args:
            spec: Specification string like "100x_1.4NA_oil" or "40x_0.9NA_air"

        Returns:
            ObjectiveSpec instance

        Raises:
            ValueError: If specification format is invalid

        Examples:
            >>> ObjectiveSpec.from_string("100x_1.4NA_oil")
            >>> ObjectiveSpec.from_string("40x_0.9NA_air")
            >>> ObjectiveSpec.from_string("60x_1.2NA_water")
        """
        # Pattern: <mag>x_<na>NA_<medium>
        pattern = r"(\d+(?:\.\d+)?)x[_\s]+([\d.]+)NA[_\s]+(\w+)"
        match = re.match(pattern, spec, re.IGNORECASE)

        if not match:
            raise ValueError(
                f"Invalid objective spec '{spec}'. "
                f"Expected format: '100x_1.4NA_oil' or '40x_0.9NA_air'"
            )

        mag = float(match.group(1))
        na = float(match.group(2))
        medium = match.group(3).lower()

        return cls(magnification=mag, numerical_aperture=na, immersion_medium=medium)

    def __str__(self) -> str:
        """String representation."""
        return f"{self.magnification:.0f}x_{self.numerical_aperture:.1f}NA_{self.immersion_medium}"


@dataclass
class MicroscopeScenarioConfig(ScenarioConfig):
    """Microscope scenario configuration with automatic parameter calculation.

    This class provides user-friendly configuration for microscope systems,
    automatically calculating resolution limits, FOV, and sampling requirements.

    Attributes:
        objective_spec: Objective specification (string or ObjectiveSpec)
        illumination_mode: 'brightfield', 'darkfield', 'phase', 'dic'
        wavelength: Operating wavelength in meters
        sensor_pixel_size: Physical pixel size at sensor in meters
        n_pixels: Number of pixels in computational grid
        tube_lens_focal: Tube lens focal length in meters

    Auto-computed attributes:
        lateral_resolution_nm: Abbe lateral resolution limit in nm
        axial_resolution_um: Axial resolution limit in um
        field_of_view_um: Field of view in um
    """

    objective_spec: str | ObjectiveSpec = "40x_0.9NA_air"
    illumination_mode: str = "brightfield"
    wavelength: float = 550e-9  # Green light
    sensor_pixel_size: float = 3.45e-6  # Scientific camera (e.g., Andor Zyla)
    n_pixels: int = 1024
    tube_lens_focal: float = 0.2  # 200mm standard

    # Auto-computed attributes
    lateral_resolution_nm: float = field(init=False)
    axial_resolution_um: float = field(init=False)
    field_of_view_um: float = field(init=False)

    @property
    def _obj(self) -> ObjectiveSpec:
        """Get objective spec as ObjectiveSpec (for type checking)."""
        return cast(ObjectiveSpec, self.objective_spec)

    def __post_init__(self) -> None:
        """Initialize scenario type and compute derived parameters."""
        # Set base class attributes
        object.__setattr__(self, "scenario_type", "microscope")

        # Parse objective spec if string
        if isinstance(self.objective_spec, str):
            object.__setattr__(
                self, "objective_spec", ObjectiveSpec.from_string(self.objective_spec)
            )

        # After parsing, objective_spec is always ObjectiveSpec
        assert isinstance(self.objective_spec, ObjectiveSpec)

        # Generate name if not set
        if not hasattr(self, "name") or self.name == "":
            obj_str = str(self._obj)
            object.__setattr__(self, "name", f"Microscope {obj_str} ({self.illumination_mode})")

        # Compute derived parameters
        self._compute_resolution()
        self._compute_field_of_view()

        # Validate configuration
        self.validate()

    def _compute_resolution(self) -> None:
        """Compute lateral and axial resolution limits using Abbe formulas."""
        na = self._obj.numerical_aperture
        wavelength_nm = self.wavelength * 1e9

        # Abbe lateral resolution: Δx = 0.61λ / NA
        lateral_res_nm = 0.61 * wavelength_nm / na
        object.__setattr__(self, "lateral_resolution_nm", lateral_res_nm)

        # Store as resolution_limit in base class (in meters)
        object.__setattr__(self, "resolution_limit", lateral_res_nm * 1e-9)

        # Axial resolution: Δz = 2λn / NA² (for widefield)
        n = self._obj.medium_index
        axial_res_um = (2 * wavelength_nm * n / (na**2)) / 1000
        object.__setattr__(self, "axial_resolution_um", axial_res_um)

    def _compute_field_of_view(self) -> None:
        """Compute field of view and sampling parameters."""
        # Object space pixel size (before magnification)
        mag = self._obj.magnification
        object_pixel_size = self.sensor_pixel_size / mag

        # Total FOV in object space
        fov_m = object_pixel_size * self.n_pixels
        fov_um = fov_m * 1e6
        object.__setattr__(self, "field_of_view_um", fov_um)

        # Store object distance (working distance)
        # Default: object at focal plane to use SIMPLIFIED model (no lens phase aliasing)
        # f_objective = f_tube / magnification
        f_objective = self.tube_lens_focal / mag
        working_distance_m = f_objective  # At focal plane by default
        object.__setattr__(self, "object_distance", working_distance_m)

    def validate(self) -> None:
        """Validate microscope scenario configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        obj = self._obj

        # Check NA vs medium index
        if obj.numerical_aperture > obj.medium_index:
            raise ValueError(
                f"NA ({obj.numerical_aperture:.2f}) cannot exceed "
                f"medium index ({obj.medium_index:.2f}) for {obj.immersion_medium}"
            )

        # Check Nyquist sampling
        # Nyquist limit: pixel size < λ/(4*NA) in object space
        min_sampling = self.wavelength / (4 * obj.numerical_aperture)
        object_pixel_size = self.sensor_pixel_size / obj.magnification

        if object_pixel_size > min_sampling * 1.5:  # Allow 1.5x margin
            raise ValueError(
                f"Severe undersampling: object pixel size {object_pixel_size * 1e6:.3f} µm "
                f"exceeds 1.5x Nyquist limit {min_sampling * 1e6:.3f} µm. "
                f"Increase magnification or use smaller sensor pixels."
            )

        # Check illumination mode
        valid_modes = ["brightfield", "darkfield", "phase", "dic"]
        if self.illumination_mode not in valid_modes:
            raise ValueError(
                f"Illumination mode must be one of {valid_modes}, got '{self.illumination_mode}'"
            )

    def to_instrument_config(self) -> MicroscopeConfig:
        """Convert to SPIDS MicroscopeConfig.

        Returns:
            MicroscopeConfig instance ready for use with SPIDS
        """
        obj = self._obj

        return MicroscopeConfig(
            wavelength=self.wavelength,
            n_pixels=self.n_pixels,
            pixel_size=self.sensor_pixel_size,
            numerical_aperture=obj.numerical_aperture,
            magnification=obj.magnification,
            medium_index=obj.medium_index,
            tube_lens_focal=self.tube_lens_focal,
            working_distance=self.object_distance,
            default_illumination_na=0.8 * obj.numerical_aperture,  # Standard Köhler
        )

    def get_info(self) -> dict:
        """Get detailed scenario information.

        Returns:
            Dictionary with all key parameters
        """
        base_info = super().get_info()
        base_info.update(
            {
                "objective": str(self.objective_spec),
                "magnification": self._obj.magnification,
                "numerical_aperture": self._obj.numerical_aperture,
                "immersion_medium": self._obj.immersion_medium,
                "illumination_mode": self.illumination_mode,
                "lateral_resolution_nm": self.lateral_resolution_nm,
                "axial_resolution_um": self.axial_resolution_um,
                "field_of_view_um": self.field_of_view_um,
                "object_pixel_size_nm": (self.sensor_pixel_size / self._obj.magnification) * 1e9,
            }
        )
        return base_info


class MicroscopeBuilder:
    """Fluent builder for microscope scenarios.

    Example:
        >>> scenario = (MicroscopeBuilder()
        ...     .objective("100x_1.4NA_oil")
        ...     .illumination("phase")
        ...     .wavelength_nm(488)
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize builder with default values."""
        self._objective_spec: str | ObjectiveSpec = "40x_0.9NA_air"
        self._illumination_mode: str = "brightfield"
        self._wavelength: float = 550e-9
        self._sensor_pixel_size: float = 3.45e-6
        self._n_pixels: int = 1024
        self._tube_lens_focal: float = 0.2
        self._name: str = ""
        self._description: str = ""

    def objective(self, spec: str | ObjectiveSpec) -> MicroscopeBuilder:
        """Set objective specification."""
        self._objective_spec = spec
        return self

    def illumination(self, mode: str) -> MicroscopeBuilder:
        """Set illumination mode."""
        self._illumination_mode = mode
        return self

    def wavelength_nm(self, wavelength_nm: float) -> MicroscopeBuilder:
        """Set wavelength in nanometers."""
        self._wavelength = wavelength_nm * 1e-9
        return self

    def sensor_pixels(self, n_pixels: int, pixel_size_um: float) -> MicroscopeBuilder:
        """Set sensor parameters."""
        self._n_pixels = n_pixels
        self._sensor_pixel_size = pixel_size_um * 1e-6
        return self

    def name(self, name: str) -> MicroscopeBuilder:
        """Set scenario name."""
        self._name = name
        return self

    def description(self, description: str) -> MicroscopeBuilder:
        """Set scenario description."""
        self._description = description
        return self

    def build(self) -> MicroscopeScenarioConfig:
        """Build the microscope scenario configuration."""
        config = MicroscopeScenarioConfig(
            objective_spec=self._objective_spec,
            illumination_mode=self._illumination_mode,
            wavelength=self._wavelength,
            sensor_pixel_size=self._sensor_pixel_size,
            n_pixels=self._n_pixels,
            tube_lens_focal=self._tube_lens_focal,
        )

        if self._name:
            object.__setattr__(config, "name", self._name)
        if self._description:
            object.__setattr__(config, "description", self._description)

        return config

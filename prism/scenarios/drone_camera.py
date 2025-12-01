"""Drone camera scenario configuration for SPIDS.

This module provides user-friendly configuration for drone-mounted camera systems
with automatic GSD, swath width, and motion blur calculations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import cast

from prism.core.instruments import CameraConfig

from .base import ScenarioConfig


@dataclass
class LensSpec:
    """Camera lens specification.

    Attributes:
        focal_length_mm: Focal length in millimeters
        f_number: f-number (focal ratio) like 2.8, 4.0, etc.
        aperture_diameter_mm: Physical aperture diameter (computed)
    """

    focal_length_mm: float
    f_number: float
    aperture_diameter_mm: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute aperture diameter from focal length and f-number."""
        object.__setattr__(self, "aperture_diameter_mm", self.focal_length_mm / self.f_number)

    @classmethod
    def from_string(cls, spec: str) -> LensSpec:
        """Parse lens specification string.

        Args:
            spec: Specification string like "50mm_f2.8" or "35mm_f4.0"

        Returns:
            LensSpec instance

        Raises:
            ValueError: If specification format is invalid

        Examples:
            >>> LensSpec.from_string("50mm_f2.8")
            >>> LensSpec.from_string("35mm_f4.0")
        """
        # Pattern: <focal>mm_f<fnumber>
        pattern = r"(\d+(?:\.\d+)?)mm[_\s]+f([\d.]+)"
        match = re.match(pattern, spec, re.IGNORECASE)

        if not match:
            raise ValueError(
                f"Invalid lens spec '{spec}'. Expected format: '50mm_f2.8' or '35mm_f4.0'"
            )

        focal_mm = float(match.group(1))
        f_num = float(match.group(2))

        return cls(focal_length_mm=focal_mm, f_number=f_num)

    def __str__(self) -> str:
        """String representation."""
        return f"{self.focal_length_mm:.0f}mm_f{self.f_number:.1f}"


@dataclass
class SensorSpec:
    """Camera sensor specification.

    Attributes:
        name: Sensor name (e.g., "full_frame", "aps_c")
        width_mm: Sensor width in millimeters
        height_mm: Sensor height in millimeters
        pixel_pitch_um: Pixel pitch in micrometers
        megapixels: Total megapixels (computed)
    """

    name: str
    width_mm: float
    height_mm: float
    pixel_pitch_um: float
    megapixels: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute megapixels."""
        n_pixels_x = int(self.width_mm * 1000 / self.pixel_pitch_um)
        n_pixels_y = int(self.height_mm * 1000 / self.pixel_pitch_um)
        object.__setattr__(self, "megapixels", (n_pixels_x * n_pixels_y) / 1e6)

    @classmethod
    def from_name(cls, name: str) -> SensorSpec:
        """Get sensor specification by name.

        Args:
            name: Sensor type name

        Returns:
            SensorSpec instance

        Raises:
            ValueError: If sensor name not recognized

        Supported sensors:
            - full_frame: 36x24mm, 6.5µm pixels (Sony A7R series)
            - aps_c: 23.5x15.6mm, 3.9µm pixels (Canon 80D)
            - micro_four_thirds: 17.3x13mm, 3.3µm pixels (Olympus)
            - 1_inch: 13.2x8.8mm, 2.4µm pixels (Sony RX100)
            - 1_2.3_inch: 6.17x4.55mm, 1.6µm pixels (DJI Phantom)
        """
        sensors = {
            "full_frame": cls("full_frame", 36.0, 24.0, 6.5),
            "aps_c": cls("aps_c", 23.5, 15.6, 3.9),
            "micro_four_thirds": cls("micro_four_thirds", 17.3, 13.0, 3.3),
            "1_inch": cls("1_inch", 13.2, 8.8, 2.4),
            "1_2.3_inch": cls("1_2.3_inch", 6.17, 4.55, 1.6),
        }

        if name not in sensors:
            available = list(sensors.keys())
            raise ValueError(f"Unknown sensor type '{name}'. Available: {available}")

        return sensors[name]


@dataclass
class DroneScenarioConfig(ScenarioConfig):
    """Drone camera scenario configuration with automatic GSD calculation.

    This class provides user-friendly configuration for drone-mounted cameras,
    automatically calculating ground sampling distance (GSD), swath width,
    and selecting appropriate propagation methods.

    Attributes:
        lens_spec: Lens specification (string or LensSpec)
        sensor_spec: Sensor specification (string or SensorSpec)
        altitude_m: Flight altitude in meters
        ground_speed_mps: Ground speed in meters per second (for motion blur)
        n_pixels: Number of pixels in computational grid

    Auto-computed attributes:
        actual_gsd_cm: Ground sampling distance in centimeters
        swath_width_m: Swath width on ground in meters
        fresnel_number: Fresnel number for propagation regime
        motion_blur_pixels: Motion blur in pixels
    """

    lens_spec: str | LensSpec = "50mm_f4.0"
    sensor_spec: str | SensorSpec = "full_frame"
    altitude_m: float = 50.0
    ground_speed_mps: float = 0.0  # Hover by default
    n_pixels: int = 1024

    # Auto-computed attributes
    actual_gsd_cm: float = field(init=False)
    swath_width_m: float = field(init=False)
    fresnel_number: float = field(init=False)
    motion_blur_pixels: float = field(init=False)

    @property
    def _lens(self) -> LensSpec:
        """Get lens spec as LensSpec (for type checking)."""
        return cast(LensSpec, self.lens_spec)

    @property
    def _sensor(self) -> SensorSpec:
        """Get sensor spec as SensorSpec (for type checking)."""
        return cast(SensorSpec, self.sensor_spec)

    def __post_init__(self) -> None:
        """Initialize scenario type and compute derived parameters."""
        # Set base class attributes
        object.__setattr__(self, "scenario_type", "drone")

        # Parse specs if strings
        if isinstance(self.lens_spec, str):
            object.__setattr__(self, "lens_spec", LensSpec.from_string(self.lens_spec))

        if isinstance(self.sensor_spec, str):
            object.__setattr__(self, "sensor_spec", SensorSpec.from_name(self.sensor_spec))

        # After parsing, specs are always the proper types
        assert isinstance(self.lens_spec, LensSpec)
        assert isinstance(self.sensor_spec, SensorSpec)

        # Generate name if not set
        if not hasattr(self, "name") or self.name == "":
            lens_str = str(self.lens_spec)
            object.__setattr__(self, "name", f"Drone {lens_str} @ {self.altitude_m:.0f}m")

        # Compute derived parameters
        self._compute_gsd()
        self._compute_swath_width()
        self._compute_fresnel_number()
        self._compute_motion_blur()

        # Validate configuration
        self.validate()

    def _compute_gsd(self) -> None:
        """Compute ground sampling distance (GSD).

        GSD = (altitude * pixel_pitch) / focal_length
        """
        h = self.altitude_m  # Altitude in meters
        p = self._sensor.pixel_pitch_um * 1e-6  # Pixel pitch in meters
        f = self._lens.focal_length_mm * 1e-3  # Focal length in meters

        gsd_m = (h * p) / f
        gsd_cm = gsd_m * 100

        object.__setattr__(self, "actual_gsd_cm", gsd_cm)
        object.__setattr__(self, "resolution_limit", gsd_m)  # Base class attribute

    def _compute_swath_width(self) -> None:
        """Compute swath width on ground."""
        # Use sensor width for swath width calculation
        sensor_width_m = self._sensor.width_mm * 1e-3
        focal_length_m = self._lens.focal_length_mm * 1e-3

        # Similar triangles: swath_width / altitude = sensor_width / focal_length
        swath_width_m = (self.altitude_m * sensor_width_m) / focal_length_m
        object.__setattr__(self, "swath_width_m", swath_width_m)

    def _compute_fresnel_number(self) -> None:
        """Compute Fresnel number to determine propagation regime.

        F = a² / (λ * z)
        where a = aperture radius, λ = wavelength, z = distance

        F >> 1: Geometric optics (can use ray tracing)
        F ~ 1: Fresnel diffraction (use angular spectrum)
        F << 1: Fraunhofer diffraction (far field)
        """
        a = (self._lens.aperture_diameter_mm * 1e-3) / 2  # Aperture radius in meters
        wavelength = self.wavelength
        z = self.altitude_m

        fresnel_num = (a**2) / (wavelength * z)
        object.__setattr__(self, "fresnel_number", fresnel_num)

        # Select propagator based on Fresnel number
        if fresnel_num > 10:
            object.__setattr__(self, "propagator_method", "angular_spectrum")
        else:
            object.__setattr__(self, "propagator_method", "fraunhofer")

    def _compute_motion_blur(self) -> None:
        """Compute motion blur in pixels.

        Motion blur occurs when drone moves during exposure.
        blur_distance = ground_speed * exposure_time
        blur_pixels = blur_distance / GSD
        """
        if self.ground_speed_mps == 0:
            object.__setattr__(self, "motion_blur_pixels", 0.0)
            return

        # Assume typical exposure time of 1/1000s for sunny conditions
        exposure_time_s = 1e-3
        blur_distance_m = self.ground_speed_mps * exposure_time_s
        gsd_m = self.actual_gsd_cm * 0.01

        blur_pixels = blur_distance_m / gsd_m
        object.__setattr__(self, "motion_blur_pixels", blur_pixels)

    def validate(self) -> None:
        """Validate drone scenario configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Check altitude is reasonable
        if self.altitude_m <= 0:
            raise ValueError(f"Altitude must be positive, got {self.altitude_m} m")

        if self.altitude_m < 5:
            raise ValueError(f"Altitude {self.altitude_m} m is too low (minimum 5m)")

        if self.altitude_m > 500:
            raise ValueError(f"Altitude {self.altitude_m} m exceeds typical drone limit (500m)")

        # Check ground speed
        if self.ground_speed_mps < 0:
            raise ValueError(f"Ground speed must be non-negative, got {self.ground_speed_mps} m/s")

        if self.ground_speed_mps > 30:
            raise ValueError(
                f"Ground speed {self.ground_speed_mps} m/s exceeds typical drone limit (30 m/s)"
            )

        # Warn about severe motion blur
        if self.motion_blur_pixels > 2.0:
            import warnings

            warnings.warn(
                f"Severe motion blur: {self.motion_blur_pixels:.1f} pixels. "
                f"Reduce ground speed or increase exposure time."
            )

    def to_instrument_config(self) -> CameraConfig:
        """Convert to SPIDS CameraConfig.

        Returns:
            CameraConfig instance ready for use with SPIDS
        """
        # Convert specs to SI units
        focal_length_m = self._lens.focal_length_mm * 1e-3
        pixel_size_m = self._sensor.pixel_pitch_um * 1e-6
        sensor_size_m = (
            self._sensor.width_mm * 1e-3,
            self._sensor.height_mm * 1e-3,
        )

        return CameraConfig(
            wavelength=self.wavelength,
            n_pixels=self.n_pixels,
            pixel_size=pixel_size_m,
            focal_length=focal_length_m,
            f_number=self._lens.f_number,
            sensor_size=sensor_size_m,
            object_distance=self.altitude_m,
            focus_distance=self.altitude_m,
            lens_type="thin",
        )

    def get_info(self) -> dict:
        """Get detailed scenario information.

        Returns:
            Dictionary with all key parameters
        """
        base_info = super().get_info()
        base_info.update(
            {
                "lens": str(self.lens_spec),
                "sensor": self._sensor.name,
                "altitude_m": self.altitude_m,
                "gsd_cm": self.actual_gsd_cm,
                "swath_width_m": self.swath_width_m,
                "fresnel_number": self.fresnel_number,
                "focal_length_mm": self._lens.focal_length_mm,
                "f_number": self._lens.f_number,
                "aperture_diameter_mm": self._lens.aperture_diameter_mm,
                "sensor_megapixels": self._sensor.megapixels,
                "ground_speed_mps": self.ground_speed_mps,
                "motion_blur_pixels": self.motion_blur_pixels,
            }
        )
        return base_info


class DroneBuilder:
    """Fluent builder for drone camera scenarios.

    Example:
        >>> scenario = (DroneBuilder()
        ...     .lens("50mm_f2.8")
        ...     .sensor("aps_c")
        ...     .altitude(100)
        ...     .ground_speed(10)
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize builder with default values."""
        self._lens_spec: str | LensSpec = "50mm_f4.0"
        self._sensor_spec: str | SensorSpec = "full_frame"
        self._altitude_m: float = 50.0
        self._ground_speed_mps: float = 0.0
        self._wavelength: float = 550e-9
        self._n_pixels: int = 1024
        self._name: str = ""
        self._description: str = ""

    def lens(self, spec: str | LensSpec) -> DroneBuilder:
        """Set lens specification."""
        self._lens_spec = spec
        return self

    def sensor(self, spec: str | SensorSpec) -> DroneBuilder:
        """Set sensor specification."""
        self._sensor_spec = spec
        return self

    def altitude(self, altitude_m: float) -> DroneBuilder:
        """Set flight altitude in meters."""
        self._altitude_m = altitude_m
        return self

    def ground_speed(self, speed_mps: float) -> DroneBuilder:
        """Set ground speed in meters per second."""
        self._ground_speed_mps = speed_mps
        return self

    def wavelength_nm(self, wavelength_nm: float) -> DroneBuilder:
        """Set wavelength in nanometers."""
        self._wavelength = wavelength_nm * 1e-9
        return self

    def n_pixels(self, n_pixels: int) -> DroneBuilder:
        """Set computational grid size."""
        self._n_pixels = n_pixels
        return self

    def name(self, name: str) -> DroneBuilder:
        """Set scenario name."""
        self._name = name
        return self

    def description(self, description: str) -> DroneBuilder:
        """Set scenario description."""
        self._description = description
        return self

    def build(self) -> DroneScenarioConfig:
        """Build the drone scenario configuration."""
        config = DroneScenarioConfig(
            lens_spec=self._lens_spec,
            sensor_spec=self._sensor_spec,
            altitude_m=self._altitude_m,
            ground_speed_mps=self._ground_speed_mps,
            n_pixels=self._n_pixels,
        )

        object.__setattr__(config, "wavelength", self._wavelength)

        if self._name:
            object.__setattr__(config, "name", self._name)
        if self._description:
            object.__setattr__(config, "description", self._description)

        return config

"""Base classes for optical scenario configurations.

This module provides abstract base classes for defining optical imaging scenarios
in a user-friendly way that automatically converts to SPIDS instrument configurations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from prism.core.instruments import InstrumentConfig


@dataclass
class ScenarioConfig(ABC):
    """Base configuration for optical imaging scenarios.

    All scenario configs provide:
    1. User-friendly parameters (objective spec, altitude, lens spec)
    2. Auto-computed physics parameters (wavelength, resolution, SNR)
    3. Conversion to SPIDS instrument configs via to_instrument_config()
    4. Validation with helpful error messages

    Attributes:
        scenario_type: Type identifier ("microscope", "drone", "satellite")
        name: Human-readable scenario name
        description: Detailed description of the scenario
        wavelength: Operating wavelength in meters
        object_distance: Distance to object in meters
        resolution_limit: Theoretical resolution in meters
        snr: Expected signal-to-noise ratio in dB
        propagator_method: Propagation method ('auto', 'fraunhofer', 'angular_spectrum')
    """

    scenario_type: str = ""
    name: str = ""
    description: str = ""

    # Common derived parameters (computed from scenario specifics)
    wavelength: float = 550e-9
    object_distance: float = 1.0
    resolution_limit: float = 1e-6
    snr: float = 40.0
    propagator_method: str = "auto"

    @abstractmethod
    def to_instrument_config(self) -> InstrumentConfig:
        """Convert scenario config to SPIDS instrument config.

        Returns:
            InstrumentConfig appropriate for this scenario (MicroscopeConfig,
            CameraConfig, etc.)
        """
        pass

    @abstractmethod
    def validate(self) -> None:
        """Validate scenario-specific parameters.

        Raises:
            ValueError: If configuration is invalid with helpful message
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get scenario information summary.

        Returns:
            Dictionary with key scenario parameters
        """
        return {
            "scenario_type": self.scenario_type,
            "name": self.name,
            "wavelength_nm": self.wavelength * 1e9,
            "object_distance_m": self.object_distance,
            "resolution_limit_um": self.resolution_limit * 1e6,
            "snr_db": self.snr,
            "propagator": self.propagator_method,
        }

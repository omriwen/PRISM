"""
Base classes and type definitions for optical propagators.

This module provides the abstract base class for all propagators and common
type definitions used throughout the propagation system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, Optional

from torch import Tensor, nn

from prism.utils.transforms import FFTCache


PropagationMethod = Literal[
    "auto",
    "fraunhofer",
    "fresnel",
    "angular_spectrum",
    "otf",
    "extended_source",
    "incoherent_auto",
    "partially_coherent_auto",
]

IlluminationMode = Literal["coherent", "incoherent", "partially_coherent"]

SamplingMethod = Literal["grid", "monte_carlo", "adaptive"]


class CoherenceMode(str, Enum):
    """Illumination coherence mode."""

    COHERENT = "coherent"
    INCOHERENT = "incoherent"
    PARTIALLY_COHERENT = "partially_coherent"


class Propagator(nn.Module, ABC):
    """
    Abstract base class for optical propagators.

    All propagators implement the forward() method to propagate
    complex optical fields through free space.
    """

    def __init__(self, fft_cache: Optional[FFTCache] = None):
        """
        Initialize propagator with optional FFT cache.

        Args:
            fft_cache: Shared FFT cache for performance optimization.
                      If None, a new cache is created.
        """
        super().__init__()
        # Share cache across propagators or create new one
        self.fft_cache = fft_cache if fft_cache is not None else FFTCache()

    @abstractmethod
    def forward(self, field: Tensor, **kwargs: Any) -> Tensor:
        """
        Propagate complex field through free space.

        Args:
            field: Complex-valued field tensor
            **kwargs: Propagator-specific parameters

        Returns:
            Propagated complex field
        """
        pass

    @property
    def illumination_mode(self) -> str:
        """
        Return the illumination mode for this propagator.

        Returns
        -------
        str
            Either "coherent" or "incoherent"
        """
        return "coherent"  # Default for existing propagators

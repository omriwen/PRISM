"""
Fraunhofer (far-field) diffraction propagator.

This module provides the FraunhoferPropagator class for far-field diffraction,
which is the default propagation method used in SPIDS for astronomical imaging.
"""

from __future__ import annotations

from typing import Literal, Optional

from torch import Tensor

from prism.core.propagators.base import Propagator
from prism.utils.transforms import FFTCache


class FraunhoferPropagator(Propagator):
    """
    Far-field (Fraunhofer) diffraction propagator.

    This is the actual propagation method used in SPIDS via Telescope.propagate_to_kspace/propagate_to_spatial.
    Valid when Fresnel number F = a²/(λz) << 1, i.e., z >> a²/λ.

    For astronomical imaging: F ~ (1m)²/(520nm × 1000km) ~ 10⁻¹² ✓ Excellent!

    Method:
        Forward (spatial → k-space): U_k = FFT(U_spatial)
        Backward (k-space → spatial): U_spatial = IFFT(U_k)

    This is the simplest and fastest propagation method - just an FFT or IFFT.

    Attributes:
        normalize: Whether to use orthonormal FFT (default: True)
        fft_cache: FFT cache for performance optimization

    Example:
        >>> from prism.core.propagators import FraunhoferPropagator
        >>> prop = FraunhoferPropagator(normalize=True)
        >>>
        >>> # Forward: spatial → k-space
        >>> spatial_field = torch.randn(256, 256, dtype=torch.cfloat)
        >>> k_field = prop(spatial_field, direction='forward')
        >>>
        >>> # Backward: k-space → spatial
        >>> reconstructed = prop(k_field, direction='backward')

    Physics Notes:
        - Fraunhofer regime: observation distance z >> a²/λ
        - Also called "far-field diffraction"
        - Fourier transform relationship between object and diffraction pattern
        - No phase curvature (plane wave approximation)

    SPIDS Usage:
        This is extracted from Telescope.propagate_to_kspace (forward) and Telescope.propagate_to_spatial (backward).
        Previously these were inline FFT calls, now properly encapsulated.
    """

    def __init__(self, normalize: bool = True, fft_cache: Optional[FFTCache] = None):
        """
        Initialize Fraunhofer propagator.

        Args:
            normalize: Use orthonormal FFT normalization (default: True)
                      - True: FFT has 1/√N scaling (preserves energy)
                      - False: FFT has 1/N scaling on inverse only
            fft_cache: Shared FFT cache for performance (optional)
        """
        super().__init__(fft_cache=fft_cache)
        self.normalize = normalize

    def forward(  # type: ignore[override]
        self, field: Tensor, direction: Literal["forward", "backward"] = "forward"
    ) -> Tensor:
        """
        Propagate field using Fraunhofer approximation.

        Args:
            field: Complex field tensor [H, W] or [B, C, H, W]
            direction: Propagation direction
                - 'forward': spatial domain → k-space (FFT)
                - 'backward': k-space → spatial domain (IFFT)

        Returns:
            Propagated complex field (same shape as input)

        Notes:
            - Forward: Equivalent to Fourier transform (object → diffraction)
            - Backward: Equivalent to inverse Fourier transform
            - With normalize=True, forward and backward are inverses
        """
        norm = "ortho" if self.normalize else "backward"

        if direction == "forward":
            # Spatial → k-space: FFT with cache
            return self.fft_cache.fft2(field, norm=norm)
        elif direction == "backward":
            # k-space → spatial: IFFT with cache
            return self.fft_cache.ifft2(field, norm=norm)
        else:
            raise ValueError(f"Unknown direction: {direction}. Use 'forward' or 'backward'.")

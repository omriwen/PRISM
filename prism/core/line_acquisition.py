"""GPU-optimized incoherent line acquisition for motion blur simulation.

This module implements the physically correct formula:
    I = (1/N) × Σᵢ |IFFT(F_kspace × Aperture_i)|²

This is the incoherent sum: intensity (squared magnitude) is summed,
NOT the complex field amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import torch
from torch import Tensor


if TYPE_CHECKING:
    from prism.core.instruments.base import Instrument


@dataclass
class LineAcquisitionConfig:
    """Configuration for GPU-optimized line acquisition.

    Attributes
    ----------
    samples_per_pixel : float
        Sampling density for accurate mode (default: 1.0)
    min_samples : int
        Minimum samples even for short lines (default: 2)
    batch_size : int
        GPU batch size, prefer power of 2 (default: 64)
    memory_limit_gb : float
        Max GPU memory for batched operations (default: 4.0)
    mode : Literal["accurate", "fast"]
        "accurate" = 1 sample/pixel, "fast" = half-diameter spacing
    """

    samples_per_pixel: float = 1.0
    min_samples: int = 2
    batch_size: int = 64
    memory_limit_gb: float = 4.0
    mode: Literal["accurate", "fast"] = "accurate"

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.samples_per_pixel <= 0:
            raise ValueError(f"samples_per_pixel must be > 0, got {self.samples_per_pixel}")
        if self.min_samples < 2:
            raise ValueError(f"min_samples must be >= 2, got {self.min_samples}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.mode not in ("accurate", "fast"):
            raise ValueError(f"mode must be 'accurate' or 'fast', got {self.mode}")


class IncoherentLineAcquisition:
    """GPU-optimized incoherent line measurement.

    Implements: I = (1/N) × Σᵢ |IFFT(F_kspace × Aperture_i)|²

    Modes
    -----
    - "accurate": 1 sample per pixel along line (default)
    - "fast": ~diameter/2 spacing (legacy behavior, but batched)

    Both modes use batched FFT operations for GPU efficiency.

    Parameters
    ----------
    config : LineAcquisitionConfig
        Configuration parameters
    instrument : Instrument
        Optical instrument for mask generation and propagation

    Example
    -------
    >>> config = LineAcquisitionConfig(mode="accurate")
    >>> telescope = Telescope(TelescopeConfig(n_pixels=512))
    >>> line_acq = IncoherentLineAcquisition(config, telescope)
    >>> measurement = line_acq.forward(field_kspace, start, end)
    """

    def __init__(
        self,
        config: LineAcquisitionConfig,
        instrument: "Instrument",
    ) -> None:
        config.validate()
        self.config = config
        self.instrument = instrument
        self._effective_batch_size = self._compute_batch_size()

    def _compute_batch_size(self) -> int:
        """Determine batch size based on GPU memory."""
        n = self.instrument.config.n_pixels
        bytes_per_complex = 16  # complex64 = 8, complex128 = 16
        bytes_per_batch = n * n * bytes_per_complex
        max_batches = int(self.config.memory_limit_gb * 1e9 / bytes_per_batch)
        return min(self.config.batch_size, max(max_batches, 1))

    def compute_n_samples(self, line_length: float) -> int:
        """Calculate number of samples based on line length and mode.

        Parameters
        ----------
        line_length : float
            Length of line in pixels

        Returns
        -------
        int
            Number of aperture samples along the line
        """
        if self.config.mode == "fast":
            # Legacy: half-diameter spacing
            # Note: aperture_radius_pixels is Telescope-specific
            diameter = float(self.instrument.config.aperture_radius_pixels) * 2  # type: ignore[attr-defined]
            n_samples = int(line_length / (diameter / 2)) + 1
        else:
            # Accurate: samples_per_pixel samples per pixel
            n_samples = int(line_length * self.config.samples_per_pixel)

        return max(self.config.min_samples, n_samples)

    def compute_line_positions(
        self,
        start: Tensor,
        end: Tensor,
        n_samples: Optional[int] = None,
    ) -> Tensor:
        """Generate evenly-spaced aperture positions along line.

        Parameters
        ----------
        start : Tensor
            [2] tensor with (y, x) start coordinates
        end : Tensor
            [2] tensor with (y, x) end coordinates
        n_samples : int, optional
            Number of samples (computed from line length if None)

        Returns
        -------
        Tensor
            [N, 2] tensor of (y, x) positions along line
        """
        if n_samples is None:
            line_length = torch.norm(end - start).item()
            n_samples = self.compute_n_samples(line_length)

        t = torch.linspace(0, 1, n_samples, device=start.device, dtype=start.dtype)
        positions = start.unsqueeze(0) + t.unsqueeze(1) * (end - start).unsqueeze(0)
        return positions

    def forward(
        self,
        field_kspace: Tensor,
        start: Tensor,
        end: Tensor,
        add_noise: bool = False,
    ) -> Tensor:
        """Compute incoherent line measurement with GPU batching.

        Parameters
        ----------
        field_kspace : Tensor
            [H, W] complex k-space field
        start : Tensor
            [2] tensor (y, x) start position
        end : Tensor
            [2] tensor (y, x) end position
        add_noise : bool
            Whether to add noise to final measurement

        Returns
        -------
        Tensor
            [H, W] intensity measurement (incoherent sum along line)
        """
        from prism.utils.transforms import batched_ifft2

        positions = self.compute_line_positions(start, end)
        n_positions = positions.shape[0]
        batch_size = self._effective_batch_size
        device = field_kspace.device

        # Accumulator for incoherent intensity sum
        intensity_sum = torch.zeros(
            field_kspace.shape[-2:],
            device=device,
            dtype=torch.float32,
        )

        # Process in batches for memory efficiency
        for batch_start in range(0, n_positions, batch_size):
            batch_end = min(batch_start + batch_size, n_positions)
            batch_positions = positions[batch_start:batch_end]

            # Generate masks for this batch [B, H, W]
            # Note: generate_aperture_masks is Telescope-specific
            masks = self.instrument.generate_aperture_masks(batch_positions)  # type: ignore[attr-defined]

            # Apply masks to k-space field [B, H, W]
            masked_fields = field_kspace.unsqueeze(0) * masks.to(
                device=device, dtype=field_kspace.dtype
            )

            # BATCHED IFFT to spatial domain [B, H, W]
            spatial_fields = batched_ifft2(masked_fields)

            # Compute intensities and accumulate
            # Key: Sum of |F|², NOT |sum(F)|² (incoherent!)
            intensities = spatial_fields.abs() ** 2
            intensity_sum = intensity_sum + intensities.sum(dim=0)

        # Normalize: incoherent MEAN intensity
        measurement = intensity_sum / n_positions

        # Add noise if requested
        if add_noise and hasattr(self.instrument, "noise_model"):
            if self.instrument.noise_model is not None:
                measurement = self.instrument.noise_model(measurement.sqrt(), add_noise=True) ** 2

        return measurement

    def generate_line_mask(
        self,
        start: Tensor,
        end: Tensor,
    ) -> Tensor:
        """Generate cumulative mask showing full line k-space coverage.

        Parameters
        ----------
        start : Tensor
            [2] tensor (y, x) start position
        end : Tensor
            [2] tensor (y, x) end position

        Returns
        -------
        Tensor
            [H, W] boolean mask covering entire line
        """
        positions = self.compute_line_positions(start, end)

        n_pixels = self.instrument.config.n_pixels
        combined_mask = torch.zeros((n_pixels, n_pixels), dtype=torch.bool, device=start.device)

        for pos in positions:
            # Note: generate_aperture_mask is Telescope-specific
            mask = self.instrument.generate_aperture_mask(pos.tolist())  # type: ignore[attr-defined]
            combined_mask = torch.logical_or(combined_mask, mask.to(combined_mask.device))

        return combined_mask

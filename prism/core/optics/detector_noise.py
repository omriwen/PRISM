"""Detector noise model for realistic imaging simulations.

This module provides a PyTorch nn.Module for simulating realistic detector noise
including shot noise (Poisson statistics), read noise (Gaussian), dark current,
and optional quantization.

Classes
-------
DetectorNoiseModel
    Realistic detector noise model with SNR-based or component-based configuration.

Examples
--------
SNR-based noise model:

>>> from prism.core.optics import DetectorNoiseModel
>>> import torch
>>> noise_model = DetectorNoiseModel(snr_db=40.0)
>>> clean_image = torch.rand(1, 1, 256, 256)
>>> noisy_image = noise_model(clean_image, add_noise=True)

Component-based noise model:

>>> noise_model = DetectorNoiseModel(
...     photon_scale=1000.0,
...     read_noise_fraction=0.01,
...     dark_current_fraction=0.002,
... )
>>> noisy_image = noise_model(clean_image, add_noise=True)

Disable noise:

>>> noise_model.disable()
>>> clean_output = noise_model(clean_image, add_noise=True)  # No noise added
>>> noise_model.enable()

See Also
--------
spids.models.noise : Alternative noise models (ShotNoise, PoissonNoise)
spids.core.instruments.microscope : Microscope with detector noise
"""

from __future__ import annotations

from typing import Optional

import torch
from loguru import logger
from torch import Tensor, nn


class DetectorNoiseModel(nn.Module):
    """Realistic detector noise model for imaging systems.

    This class implements realistic detector noise including shot noise (Poisson
    statistics), read noise (Gaussian), dark current, and optional quantization.
    It can operate in two modes:

    1. **SNR-based mode**: Specify target SNR in dB, noise is automatically scaled
    2. **Component-based mode**: Specify individual noise components explicitly

    Parameters
    ----------
    snr_db : float, optional
        Target signal-to-noise ratio in dB. If provided, uses SNR-based noise model.
        The noise standard deviation is computed as: σ = signal_max / (10^(snr_db/20))
    photon_scale : float, default=1000.0
        Photon count scaling factor for shot noise. Higher values = higher photon counts
        = less relative noise. Only used when snr_db is None.
    read_noise_fraction : float, default=0.01
        Read noise as fraction of maximum signal (Gaussian, additive).
        Only used when snr_db is None.
    dark_current_fraction : float, default=0.002
        Dark current noise as fraction of maximum signal (Gaussian, additive).
        Only used when snr_db is None.
    enabled : bool, default=True
        Whether noise is enabled. Can be toggled with enable()/disable().

    Attributes
    ----------
    snr_db : float or None
        Target SNR in dB (None if using component-based mode)
    photon_scale : float
        Photon count scaling factor
    read_noise_fraction : float
        Read noise fraction
    dark_current_fraction : float
        Dark current fraction
    enabled : bool
        Whether noise is currently enabled

    Methods
    -------
    forward(intensity, add_noise=True)
        Add detector noise to intensity image
    set_snr(snr_db)
        Update SNR level (switches to SNR-based mode)
    enable()
        Enable noise addition
    disable()
        Disable noise addition

    Notes
    -----
    - Shot noise follows Poisson statistics but is approximated with Gaussian
      when photon counts are high (mean >> 1), which is valid for most imaging scenarios
    - The existing implementation in Microscope._add_detector_noise() scales to photon
      counts, adds Poisson-like noise, then scales back
    - Read noise is additive Gaussian noise representing detector electronics noise
    - Dark current represents thermal electrons accumulated during exposure
    - Output is always clamped to non-negative values (physical constraint)
    - During eval() mode, noise is still added if add_noise=True (controlled by flag,
      not by module training state)

    Examples
    --------
    Create SNR-based noise model:

    >>> noise_model = DetectorNoiseModel(snr_db=40.0)
    >>> clean = torch.rand(256, 256)
    >>> noisy = noise_model(clean, add_noise=True)

    Create component-based noise model:

    >>> noise_model = DetectorNoiseModel(
    ...     photon_scale=2000.0,
    ...     read_noise_fraction=0.005,
    ...     dark_current_fraction=0.001,
    ... )

    Dynamically adjust SNR:

    >>> noise_model.set_snr(50.0)  # Increase SNR (less noise)
    >>> less_noisy = noise_model(clean, add_noise=True)

    Temporarily disable noise:

    >>> noise_model.disable()
    >>> clean_output = noise_model(clean, add_noise=True)  # No noise
    >>> noise_model.enable()

    References
    ----------
    .. [1] Janesick, J. R. (2001). "Scientific Charge-Coupled Devices".
           SPIE Press. Chapter 4: Noise Sources and Signal-to-Noise Ratio.
    """

    def __init__(
        self,
        snr_db: Optional[float] = None,
        photon_scale: float = 1000.0,
        read_noise_fraction: float = 0.01,
        dark_current_fraction: float = 0.002,
        enabled: bool = True,
    ) -> None:
        """Initialize DetectorNoiseModel.

        Parameters
        ----------
        snr_db : float, optional
            Target SNR in dB. If provided, uses SNR-based noise model.
        photon_scale : float, default=1000.0
            Photon count scaling factor (used when snr_db is None)
        read_noise_fraction : float, default=0.01
            Read noise as fraction of max signal (used when snr_db is None)
        dark_current_fraction : float, default=0.002
            Dark current as fraction of max signal (used when snr_db is None)
        enabled : bool, default=True
            Whether noise is enabled
        """
        super().__init__()

        # Validate inputs
        if snr_db is not None and snr_db <= 0:
            raise ValueError(f"snr_db must be positive, got {snr_db}")
        if photon_scale <= 0:
            raise ValueError(f"photon_scale must be positive, got {photon_scale}")
        if read_noise_fraction < 0:
            raise ValueError(f"read_noise_fraction must be non-negative, got {read_noise_fraction}")
        if dark_current_fraction < 0:
            raise ValueError(
                f"dark_current_fraction must be non-negative, got {dark_current_fraction}"
            )

        self.snr_db = snr_db
        self.photon_scale = photon_scale
        self.read_noise_fraction = read_noise_fraction
        self.dark_current_fraction = dark_current_fraction
        self.enabled = enabled

        # Log initialization
        if snr_db is not None:
            logger.debug(f"DetectorNoiseModel initialized with SNR={snr_db} dB")
        else:
            logger.debug(
                f"DetectorNoiseModel initialized with component-based noise: "
                f"photon_scale={photon_scale}, read_noise={read_noise_fraction}, "
                f"dark_current={dark_current_fraction}"
            )

    def forward(self, intensity: Tensor, add_noise: bool = True) -> Tensor:
        """Add detector noise to intensity image.

        Parameters
        ----------
        intensity : Tensor
            Input intensity image (clean, non-negative). Shape: (H, W) or (B, C, H, W)
        add_noise : bool, default=True
            Whether to add noise. If False or if enabled=False, returns input unchanged.

        Returns
        -------
        Tensor
            Noisy intensity image with same shape as input, clamped to non-negative values

        Notes
        -----
        - If enabled=False or add_noise=False, returns input unchanged
        - Uses SNR-based noise if snr_db is set, otherwise uses component-based noise
        - Output is always non-negative (clamped to >= 0)
        - Shot noise variance scales with local signal intensity
        - Read noise and dark current are additive with constant variance
        """
        # Early exit if noise is disabled or not requested
        if not self.enabled or not add_noise:
            return intensity

        # Use SNR-based noise if snr_db is specified
        if self.snr_db is not None:
            return self._add_snr_based_noise(intensity)
        else:
            return self._add_component_based_noise(intensity)

    def _add_snr_based_noise(self, intensity: Tensor) -> Tensor:
        """Add noise based on target SNR.

        Parameters
        ----------
        intensity : Tensor
            Clean intensity image

        Returns
        -------
        Tensor
            Noisy intensity image

        Notes
        -----
        Noise standard deviation is computed as σ = signal_max / (10^(snr_db/20))
        where signal_max is the maximum value in the intensity image.
        """
        if intensity.max() == 0:
            # No signal, return zeros
            return intensity

        # Compute noise standard deviation from SNR
        # SNR(dB) = 20 * log10(signal / noise_std)
        # => noise_std = signal / 10^(SNR_dB / 20)
        assert self.snr_db is not None, "snr_db must be set for SNR-based noise"
        signal_max = intensity.max()
        noise_std = signal_max / (10 ** (self.snr_db / 20))

        # Add Gaussian noise
        noise = torch.randn_like(intensity) * noise_std
        noisy_intensity = intensity + noise

        # Ensure non-negative (physical constraint)
        return torch.clamp(noisy_intensity, min=0)

    def _add_component_based_noise(self, intensity: Tensor) -> Tensor:
        """Add component-based noise (shot + read + dark current).

        This follows the implementation pattern from Microscope._add_detector_noise().

        Parameters
        ----------
        intensity : Tensor
            Clean intensity image

        Returns
        -------
        Tensor
            Noisy intensity image

        Notes
        -----
        - Shot noise: Poisson approximated as Gaussian with variance = mean
        - Read noise: Gaussian with constant variance
        - Dark current: Gaussian with constant variance
        """
        # Shot noise (Poisson approximation using Gaussian)
        if intensity.max() > 0:
            # Scale to photon counts (arbitrary scaling)
            intensity_photons = intensity * self.photon_scale

            # Poisson noise approximated by Gaussian with variance = mean
            # Add small constant to avoid sqrt(0)
            shot_noise = (
                torch.randn_like(intensity) * torch.sqrt(intensity_photons + 1) / self.photon_scale
            )
        else:
            shot_noise = torch.zeros_like(intensity)

        # Read noise (Gaussian, signal-independent)
        read_noise = torch.randn_like(intensity) * self.read_noise_fraction * intensity.max()

        # Dark current (small constant + Gaussian)
        dark_current = torch.randn_like(intensity) * self.dark_current_fraction * intensity.max()

        # Combine all noise sources
        noisy_intensity = intensity + shot_noise + read_noise + dark_current

        # Ensure non-negative
        return torch.clamp(noisy_intensity, min=0)

    def set_snr(self, snr_db: float) -> None:
        """Update SNR level.

        This switches the model to SNR-based noise mode.

        Parameters
        ----------
        snr_db : float
            New target SNR in dB (must be positive)

        Raises
        ------
        ValueError
            If snr_db is not positive

        Examples
        --------
        >>> noise_model = DetectorNoiseModel(photon_scale=1000)  # Component-based
        >>> noise_model.set_snr(40.0)  # Switch to SNR-based
        >>> noise_model.set_snr(50.0)  # Update SNR
        """
        if snr_db <= 0:
            raise ValueError(f"snr_db must be positive, got {snr_db}")

        self.snr_db = snr_db
        logger.debug(f"DetectorNoiseModel SNR updated to {snr_db} dB")

    def enable(self) -> None:
        """Enable noise addition.

        Examples
        --------
        >>> noise_model.disable()
        >>> noise_model.enable()
        """
        self.enabled = True
        logger.debug("DetectorNoiseModel noise enabled")

    def disable(self) -> None:
        """Disable noise addition.

        When disabled, forward() returns input unchanged regardless of add_noise flag.

        Examples
        --------
        >>> noise_model = DetectorNoiseModel(snr_db=40.0)
        >>> noise_model.disable()
        >>> clean_output = noise_model(image, add_noise=True)  # No noise added
        """
        self.enabled = False
        logger.debug("DetectorNoiseModel noise disabled")

    def __repr__(self) -> str:
        """Return string representation of the noise model."""
        if self.snr_db is not None:
            mode = f"SNR={self.snr_db} dB"
        else:
            mode = (
                f"photon_scale={self.photon_scale}, "
                f"read_noise={self.read_noise_fraction}, "
                f"dark_current={self.dark_current_fraction}"
            )

        enabled_str = "enabled" if self.enabled else "disabled"
        return f"DetectorNoiseModel({mode}, {enabled_str})"

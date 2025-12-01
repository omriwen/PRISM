"""
Noise models for realistic simulations.

This module provides noise models for simulating realistic measurement
conditions in astronomical imaging, including shot noise, readout noise,
and composite noise models.

Classes:
    - NoiseModel (ABC): Abstract base class for all noise models
    - ShotNoise: Photon shot noise (Poisson)
    - PoissonNoise: Alias for ShotNoise
    - ReadoutNoise: Gaussian readout noise
    - CompositeNoise: Combination of multiple noise sources
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


class NoiseModel(nn.Module, ABC):
    """
    Abstract base class for noise models.

    All noise models should inherit from this class and implement the
    add_noise() and get_stats() methods. This enables extensible noise
    modeling with a consistent interface.

    Methods:
        add_noise(tensor): Add noise to input tensor
        get_stats(): Get noise statistics (mean, variance, etc.)

    Example:
        >>> class CustomNoise(NoiseModel):
        ...     def add_noise(self, tensor):
        ...         return tensor + torch.randn_like(tensor) * 0.1
        ...     def get_stats(self):
        ...         return {"noise_type": "custom", "sigma": 0.1}
    """

    @abstractmethod
    def add_noise(self, tensor: Tensor) -> Tensor:
        """
        Add noise to input tensor.

        Args:
            tensor (Tensor): Input tensor to add noise to

        Returns:
            Tensor: Noisy tensor

        Notes:
            - Implementations should preserve input tensor shape
            - Noise should be appropriate for the tensor value range
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get noise statistics.

        Returns:
            Dict[str, Any]: Dictionary of noise statistics including
                           noise type, parameters, and derived quantities

        Example:
            >>> noise = PoissonNoise(snr=40.0)
            >>> stats = noise.get_stats()
            >>> print(stats)
            {'noise_type': 'poisson', 'snr_db': 40.0, 'snr_linear': 10000.0}
        """
        pass


class ShotNoise(NoiseModel):
    """
    Photon shot noise simulation with controllable SNR.

    Shot noise arises from the discrete nature of photon detection and follows
    Poisson statistics. This implementation approximates Poisson noise as
    Gaussian (valid for high photon counts) with variance proportional to signal.

    The noise is scaled to achieve a specified signal-to-noise ratio (SNR) in dB:
    SNR(dB) = 10 * log10(signal_power / noise_power)

    Attributes:
        desired_snr_db (float): Target SNR in decibels
        desired_snr_linear (float): Target SNR in linear scale

    Methods:
        forward(x, add_noise=False): Add shot noise to intensity measurement

    Example:
        >>> noise = ShotNoise(desired_snr_db=40)  # 40 dB SNR
        >>> noisy_measurement = noise(clean_measurement, add_noise=True)

    Notes:
        - Noise is only added when add_noise=True (measurement mode)
        - Noise power is signal-dependent (per-pixel variance)
        - Output is clamped to non-negative values (ReLU)
        - Gaussian approximation valid for mean photon count >> 1
    """

    def __init__(self, desired_snr_db: float) -> None:
        """
        Initialize ShotNoise model.

        Args:
            desired_snr_db (float): Desired signal-to-noise ratio in decibels
        """
        super().__init__()
        self.desired_snr_db: float = desired_snr_db
        # Convert SNR from dB to linear scale
        self.desired_snr_linear: float = 10 ** (desired_snr_db / 10)

    def forward(self, x: Tensor, add_noise: bool = False) -> Tensor:
        """
        Add shot noise to measurement.

        Args:
            x (Tensor): Input intensity measurement (field amplitude)
            add_noise (bool): Whether to add noise (False for reconstruction pass)

        Returns:
            Tensor: Noisy intensity measurement if add_noise=True, else squared input

        Notes:
            - Input is squared to convert amplitude to intensity
            - Noise variance scales with local signal intensity
            - Output is guaranteed non-negative via ReLU
        """
        if add_noise:  # Only add noise when required (measurement)
            x2 = x**2
            signal_power = x2

            # Calculate desired noise power based on desired SNR
            noise_power = signal_power / self.desired_snr_linear

            # Generate Gaussian noise with mean=0 and std=1
            gaussian_noise = torch.randn_like(x2)

            # Scale the noise to achieve the desired noise power
            # First, normalize noise to have unit power
            gaussian_noise = gaussian_noise / torch.std(gaussian_noise)
            # Then, scale to desired noise power
            noise_scaling_factor = torch.sqrt(noise_power)
            scaled_noise = gaussian_noise * noise_scaling_factor
            x_noisy = F.relu(x2 + scaled_noise)
            return x_noisy
        else:
            return x

    def add_noise(self, tensor: Tensor) -> Tensor:
        """
        Add shot noise to tensor (NoiseModel interface).

        Args:
            tensor (Tensor): Input tensor

        Returns:
            Tensor: Noisy tensor
        """
        return self.forward(tensor, add_noise=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get shot noise statistics.

        Returns:
            Dict[str, Any]: Noise statistics
        """
        return {
            "noise_type": "shot",
            "snr_db": self.desired_snr_db,
            "snr_linear": self.desired_snr_linear,
        }


class PoissonNoise(NoiseModel):
    """
    Poisson (shot) noise model.

    Implements photon shot noise following Poisson statistics. Approximates
    Poisson noise as Gaussian for high photon counts.

    Attributes:
        snr (float): Target signal-to-noise ratio in dB

    Methods:
        add_noise(tensor): Add Poisson noise to tensor
        get_stats(): Get noise statistics

    Example:
        >>> noise = PoissonNoise(snr=40.0)
        >>> clean_image = torch.randn(1, 1, 256, 256)
        >>> noisy_image = noise.add_noise(clean_image)
        >>> print(noise.get_stats())
        {'noise_type': 'poisson', 'snr_db': 40.0, ...}
    """

    def __init__(self, snr: float = 40.0):
        """
        Initialize PoissonNoise model.

        Args:
            snr (float): Target SNR in decibels (default: 40.0)
        """
        super().__init__()
        self.snr = snr
        self.snr_linear = 10 ** (snr / 10)

    def add_noise(self, tensor: Tensor) -> Tensor:
        """
        Add Poisson noise to tensor.

        Args:
            tensor (Tensor): Input tensor (field amplitude)

        Returns:
            Tensor: Noisy intensity measurement

        Notes:
            - Converts amplitude to intensity (squares input)
            - Adds Gaussian approximation of Poisson noise
            - Clamps output to non-negative values
        """
        x2 = tensor**2
        signal_power = x2

        # Calculate noise power based on SNR
        noise_power = signal_power / self.snr_linear

        # Generate Gaussian noise
        gaussian_noise = torch.randn_like(x2)
        gaussian_noise = gaussian_noise / torch.std(gaussian_noise)
        scaled_noise = gaussian_noise * torch.sqrt(noise_power)

        return F.relu(x2 + scaled_noise)

    def get_stats(self) -> Dict[str, Any]:
        """Get Poisson noise statistics."""
        return {
            "noise_type": "poisson",
            "snr_db": self.snr,
            "snr_linear": self.snr_linear,
        }


class ReadoutNoise(NoiseModel):
    """
    Readout noise (Gaussian) model.

    Simulates detector readout noise as additive Gaussian noise with
    constant variance. Unlike shot noise, readout noise is signal-independent.

    Attributes:
        sigma (float): Standard deviation of readout noise

    Methods:
        add_noise(tensor): Add readout noise to tensor
        get_stats(): Get noise statistics

    Example:
        >>> noise = ReadoutNoise(sigma=0.01)
        >>> image = torch.randn(1, 1, 256, 256)
        >>> noisy_image = noise.add_noise(image)
    """

    def __init__(self, sigma: float = 0.01):
        """
        Initialize ReadoutNoise model.

        Args:
            sigma (float): Standard deviation of Gaussian noise (default: 0.01)
        """
        super().__init__()
        self.sigma = sigma

    def add_noise(self, tensor: Tensor) -> Tensor:
        """
        Add Gaussian readout noise to tensor.

        Args:
            tensor (Tensor): Input tensor

        Returns:
            Tensor: Noisy tensor with additive Gaussian noise

        Notes:
            - Noise is signal-independent (constant variance)
            - No clamping applied (can produce negative values)
        """
        return tensor + torch.randn_like(tensor) * self.sigma

    def get_stats(self) -> Dict[str, Any]:
        """Get readout noise statistics."""
        return {
            "noise_type": "readout",
            "sigma": self.sigma,
            "variance": self.sigma**2,
        }


class CompositeNoise(NoiseModel):
    """
    Composite noise model combining multiple noise sources.

    Applies multiple noise models sequentially to simulate realistic
    measurement conditions with multiple noise sources (e.g., shot noise
    + readout noise).

    Attributes:
        noise_models (List[NoiseModel]): List of noise models to apply

    Methods:
        add_noise(tensor): Apply all noise models sequentially
        get_stats(): Get combined statistics from all noise sources

    Example:
        >>> # Realistic measurement: shot noise + readout noise
        >>> shot = PoissonNoise(snr=40.0)
        >>> readout = ReadoutNoise(sigma=0.01)
        >>> composite = CompositeNoise([shot, readout])
        >>>
        >>> clean_image = torch.randn(1, 1, 256, 256)
        >>> noisy_image = composite.add_noise(clean_image)
        >>> print(composite.get_stats())
    """

    def __init__(self, noise_models: List[NoiseModel]):
        """
        Initialize CompositeNoise model.

        Args:
            noise_models (List[NoiseModel]): List of noise models to apply
                                             in sequence

        Raises:
            ValueError: If noise_models is empty
        """
        super().__init__()
        if not noise_models:
            raise ValueError("CompositeNoise requires at least one noise model")

        self.noise_models = nn.ModuleList(noise_models)

    def add_noise(self, tensor: Tensor) -> Tensor:
        """
        Apply all noise models sequentially.

        Args:
            tensor (Tensor): Input tensor

        Returns:
            Tensor: Tensor with all noise sources applied

        Notes:
            - Noise models are applied in the order specified
            - Each model receives the output of the previous model
        """
        result = tensor
        for noise_model in self.noise_models:
            assert isinstance(noise_model, NoiseModel)
            result = noise_model.add_noise(result)
        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all noise sources.

        Returns:
            Dict[str, Any]: Combined statistics including individual noise stats

        Example:
            >>> composite = CompositeNoise([PoissonNoise(40), ReadoutNoise(0.01)])
            >>> stats = composite.get_stats()
            >>> stats['noise_type']
            'composite'
            >>> len(stats['components'])
            2
        """
        return {
            "noise_type": "composite",
            "n_components": len(self.noise_models),
            "components": [
                noise_model.get_stats()
                for noise_model in self.noise_models
                if isinstance(noise_model, NoiseModel)
            ],
        }

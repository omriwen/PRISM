"""
Validation utilities for optical propagators.

This module provides functions for validating tensor inputs
for coherent and incoherent propagation.
"""

from __future__ import annotations

import warnings

from torch import Tensor


def validate_intensity_input(tensor: Tensor, name: str = "input") -> None:
    """
    Validate that a tensor is suitable for incoherent propagation.

    Incoherent propagation operates on intensity (real, non-negative) tensors.
    This function validates the input and warns if negative values are present.

    Parameters
    ----------
    tensor : Tensor
        Input tensor to validate
    name : str, optional
        Name for error messages. Default: "input"

    Raises
    ------
    ValueError
        If tensor is complex

    Warns
    -----
    UserWarning
        If tensor contains negative values (will be clamped to zero)

    Examples
    --------
    >>> intensity = torch.rand(64, 64)
    >>> validate_intensity_input(intensity, "source_intensity")  # OK

    >>> field = torch.randn(64, 64, dtype=torch.cfloat)
    >>> validate_intensity_input(field)  # Raises ValueError
    """
    if tensor.is_complex():
        raise ValueError(
            f"{name} must be real for incoherent propagation, got complex tensor. "
            "For coherent propagation, use FraunhoferPropagator or "
            "AngularSpectrumPropagator instead."
        )
    if (tensor < 0).any():
        warnings.warn(
            f"{name} contains negative values which will be clamped to zero. "
            "Intensity values should be non-negative.",
            UserWarning,
            stacklevel=2,
        )


def validate_coherent_input(tensor: Tensor, name: str = "input") -> None:
    """
    Validate that a tensor is suitable for coherent propagation.

    Coherent propagation operates on complex field tensors.
    This function validates that the input is complex-valued.

    Parameters
    ----------
    tensor : Tensor
        Input tensor to validate
    name : str, optional
        Name for error messages. Default: "input"

    Raises
    ------
    ValueError
        If tensor is not complex

    Examples
    --------
    >>> field = torch.randn(64, 64, dtype=torch.cfloat)
    >>> validate_coherent_input(field, "aperture_field")  # OK

    >>> intensity = torch.rand(64, 64)
    >>> validate_coherent_input(intensity)  # Raises ValueError

    Notes
    -----
    To convert a real tensor to complex for coherent propagation:
        >>> complex_tensor = real_tensor.to(torch.cfloat)
    """
    if not tensor.is_complex():
        raise ValueError(
            f"{name} must be complex for coherent propagation, got real tensor. "
            f"For incoherent propagation, use OTFPropagator instead. "
            f"To convert to complex: tensor = tensor.to(torch.cfloat)"
        )

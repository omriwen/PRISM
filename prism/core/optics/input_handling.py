"""Input handling utilities for wave optics simulation.

This module provides utilities for preparing input fields for microscopy
forward models, including automatic detection and conversion between
intensity, amplitude, and complex field representations.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import Tensor


if TYPE_CHECKING:
    from prism.core.grid import Grid


class InputMode(Enum):
    """Input field representation mode.

    Attributes
    ----------
    AMPLITUDE : str
        Input is field amplitude |E|. Used directly as complex magnitude.
        Values should be >= 0.

    INTENSITY : str
        Input is intensity I = |E|^2. Converted to amplitude via sqrt(I).
        Values should be >= 0. This is the most common case for targets.

    COMPLEX : str
        Input is already complex field E = |E| * exp(i*phi).
        No conversion needed.

    AUTO : str
        Auto-detect based on tensor dtype and values:
        - complex64/128 -> COMPLEX
        - real + all >= 0 -> INTENSITY (with warning)
        - real + has negative -> raises ValueError
    """

    AMPLITUDE = "amplitude"
    INTENSITY = "intensity"
    COMPLEX = "complex"
    AUTO = "auto"


def detect_input_mode(field: Tensor) -> InputMode:
    """Auto-detect input mode from tensor properties.

    Parameters
    ----------
    field : Tensor
        Input field tensor.

    Returns
    -------
    InputMode
        Detected input mode.

    Raises
    ------
    ValueError
        If field contains negative values (invalid for optical simulation).
    """
    if field.is_complex():
        return InputMode.COMPLEX

    if (field < 0).any():
        raise ValueError(
            "Input field contains negative values, which is invalid for "
            "optical simulation. Intensity and amplitude must be >= 0. "
            "If this is a phase pattern, use input_mode='complex' with "
            "torch.exp(1j * phase)."
        )

    # Non-negative real - assume intensity (most common user case)
    warnings.warn(
        "Auto-detected input as INTENSITY (non-negative real values). "
        "Converting to amplitude via sqrt(). If your input is already "
        "amplitude, use input_mode='amplitude' to suppress this warning.",
        UserWarning,
        stacklevel=3,
    )
    return InputMode.INTENSITY


def convert_to_complex_field(field: Tensor, input_mode: InputMode) -> Tensor:
    """Convert input to complex field for wave propagation.

    Parameters
    ----------
    field : Tensor
        Input field tensor.
    input_mode : InputMode
        How to interpret the input values.

    Returns
    -------
    Tensor
        Complex field tensor (complex64).
    """
    if input_mode == InputMode.COMPLEX:
        if field.is_complex():
            return field
        return field.to(torch.complex64)

    if input_mode == InputMode.AMPLITUDE:
        # Amplitude -> complex with zero phase
        return field.float().to(torch.complex64)

    if input_mode == InputMode.INTENSITY:
        # Intensity -> amplitude via sqrt, then complex
        intensity = torch.clamp(field.float(), min=0.0)
        amplitude = torch.sqrt(intensity)
        return amplitude.to(torch.complex64)

    raise ValueError(f"Unknown input mode: {input_mode}")


def prepare_field(
    field: Tensor,
    expected_shape: Tuple[int, int],
    input_mode: InputMode = InputMode.AUTO,
) -> Tensor:
    """Complete input preparation pipeline.

    Parameters
    ----------
    field : Tensor
        Input field tensor with shape (..., H, W).
    expected_shape : tuple of int
        Expected (H, W) spatial dimensions.
    input_mode : InputMode
        How to interpret input: INTENSITY, AMPLITUDE, COMPLEX, or AUTO.

    Returns
    -------
    Tensor
        Prepared complex field ready for forward model.

    Raises
    ------
    ValueError
        If field shape doesn't match expected shape.
    """
    # Step 1: Validate shape
    if field.shape[-2:] != expected_shape:
        raise ValueError(
            f"Field spatial dimensions {tuple(field.shape[-2:])} don't match "
            f"expected grid size {expected_shape}. Ensure your target uses "
            f"the same resolution as the microscope scenario."
        )

    # Step 2: Auto-detect mode if needed
    if input_mode == InputMode.AUTO:
        input_mode = detect_input_mode(field)

    # Step 3: Convert to complex field
    return convert_to_complex_field(field, input_mode)


def validate_fov_consistency(
    input_pixel_size: Optional[float],
    grid_pixel_size: float,
    tolerance: float = 0.1,
) -> None:
    """Warn if input field's pixel size doesn't match grid.

    Parameters
    ----------
    input_pixel_size : float or None
        Physical size of input pixels (meters). If None, skip validation.
    grid_pixel_size : float
        Instrument grid pixel size (meters).
    tolerance : float
        Maximum allowed relative difference (default 10%).
    """
    if input_pixel_size is None:
        return

    relative_diff = abs(input_pixel_size - grid_pixel_size) / grid_pixel_size

    if relative_diff > tolerance:
        warnings.warn(
            f"Input field pixel size ({input_pixel_size:.2e} m) differs from "
            f"instrument grid ({grid_pixel_size:.2e} m) by {relative_diff * 100:.1f}%. "
            f"This may indicate a FOV mismatch that will cause artifacts. "
            f"Ensure target field_size matches scenario.field_of_view_um.",
            UserWarning,
            stacklevel=3,
        )


class InputFieldHandler:
    """Unified input field validation and conversion for 4f optical systems.

    This class provides a complete input handling pipeline for optical forward models,
    supporting multiple input representations (intensity, amplitude, complex), automatic
    dimension standardization, and optional pixel size resampling.

    Supports input modes:
    - 'intensity': Real-valued intensity I = |E|^2 (converted via sqrt to amplitude)
    - 'amplitude': Real-valued amplitude |E| (complexified with zero phase)
    - 'complex': Complex-valued field E = |E| * exp(i*phi) (used directly)
    - 'auto': Automatically detect based on dtype and values

    The handler manages:
    - Input mode detection and conversion to complex fields
    - Dimension standardization to [B, C, H, W] format
    - Shape validation against expected grid dimensions
    - Pixel size validation and optional resampling
    - Dimension restoration to match input format

    Parameters
    ----------
    grid : Grid
        Computational grid defining expected spatial dimensions and pixel size.
    expected_shape : tuple of int, optional
        Expected (H, W) spatial shape. If None, uses grid.nx for both dimensions.
    allow_resampling : bool, default=True
        Whether to allow resampling when input_pixel_size differs from grid.
        If False, only validation warnings are issued.

    Examples
    --------
    Basic usage with automatic mode detection:

    >>> from prism.core.grid import Grid
    >>> grid = Grid(nx=256, dx=10e-6, wavelength=520e-9)
    >>> handler = InputFieldHandler(grid)
    >>> intensity = torch.rand(256, 256)  # Auto-detected as intensity
    >>> field = handler.validate_and_convert(intensity)
    >>> field.shape
    torch.Size([256, 256])

    Explicit mode specification with batched input:

    >>> amplitude = torch.rand(4, 1, 256, 256)  # [B, C, H, W]
    >>> field = handler.validate_and_convert(amplitude, input_mode='amplitude')
    >>> field.shape
    torch.Size([4, 1, 256, 256])

    With pixel size validation:

    >>> target_pixel_size = 10e-6  # Matches grid
    >>> field = handler.validate_and_convert(
    ...     intensity,
    ...     input_mode='intensity',
    ...     input_pixel_size=target_pixel_size
    ... )

    Notes
    -----
    This class is designed for use by FourFSystem base class and instrument
    forward models. It consolidates input handling logic that was previously
    scattered across multiple instrument implementations.

    The dimension standardization ensures internal operations work with consistent
    [B, C, H, W] tensors, while dimension restoration maintains API compatibility
    with code expecting various input shapes.
    """

    def __init__(
        self,
        grid: "Grid",
        expected_shape: Optional[Tuple[int, int]] = None,
        allow_resampling: bool = True,
    ) -> None:
        """Initialize input field handler.

        Parameters
        ----------
        grid : Grid
            Computational grid defining expected spatial dimensions.
        expected_shape : tuple of int, optional
            Expected (H, W) shape. Defaults to (grid.nx, grid.nx).
        allow_resampling : bool, default=True
            Whether to allow pixel size resampling.
        """
        self.grid = grid
        self.expected_shape = expected_shape or (grid.nx, grid.nx)
        self.allow_resampling = allow_resampling

    def validate_and_convert(
        self,
        field: Tensor,
        input_mode: str = "auto",
        input_pixel_size: Optional[float] = None,
    ) -> Tensor:
        """Validate input field and convert to standardized complex tensor.

        This is the main entry point for input handling. It performs the complete
        pipeline: dimension standardization, mode detection/conversion, shape
        validation, pixel size checking, and dimension restoration.

        Parameters
        ----------
        field : Tensor
            Input field of any supported shape/dtype:
            - [H, W]: Single 2D field
            - [C, H, W]: Multi-channel field (interpreted as batch if C != 1)
            - [B, C, H, W]: Batched multi-channel field
        input_mode : str, default='auto'
            One of 'intensity', 'amplitude', 'complex', 'auto'.
            In 'auto' mode, complex dtype → 'complex', non-negative real → 'intensity'.
        input_pixel_size : float, optional
            Physical pixel size of input field (meters).
            If provided and differs from grid.dx, issues warning (or resamples if enabled).

        Returns
        -------
        Tensor
            Complex field with same dimensionality as input:
            - Input [H, W] → Output [H, W]
            - Input [C, H, W] → Output [C, H, W]
            - Input [B, C, H, W] → Output [B, C, H, W]

        Raises
        ------
        ValueError
            If field shape doesn't match expected_shape, if input_mode is invalid,
            or if field contains negative values with non-complex dtype.

        Examples
        --------
        >>> handler = InputFieldHandler(grid)
        >>> # 2D intensity input
        >>> intensity = torch.rand(256, 256)
        >>> field = handler.validate_and_convert(intensity, input_mode='intensity')
        >>> field.dtype
        torch.complex64

        >>> # Batched amplitude input
        >>> amplitude = torch.rand(8, 3, 256, 256)
        >>> field = handler.validate_and_convert(amplitude, input_mode='amplitude')
        >>> field.shape
        torch.Size([8, 3, 256, 256])
        """
        # Step 1: Standardize dimensions to [B, C, H, W]
        field_standardized, squeeze_info = self._standardize_dimensions(field)

        # Step 2: Validate spatial shape
        actual_shape = field_standardized.shape[-2:]
        if actual_shape != self.expected_shape:
            raise ValueError(
                f"Field spatial dimensions {actual_shape} don't match "
                f"expected grid size {self.expected_shape}. Ensure your input uses "
                f"the same resolution as the instrument grid."
            )

        # Step 3: Validate pixel size consistency
        if input_pixel_size is not None:
            validate_fov_consistency(input_pixel_size, self.grid.dx)
            # For now, we only validate and warn

        # Step 4: Detect input mode if auto
        mode_enum = self._parse_input_mode(input_mode)
        if mode_enum == InputMode.AUTO:
            mode_enum = detect_input_mode(field_standardized)

        # Step 5: Convert to complex field
        field_complex = convert_to_complex_field(field_standardized, mode_enum)

        # Step 6: Restore original dimensions
        field_output = self._restore_dimensions(field_complex, squeeze_info)

        return field_output

    def detect_input_mode(self, field: Tensor) -> str:
        """Auto-detect the input mode based on dtype and values.

        Parameters
        ----------
        field : Tensor
            Input field tensor.

        Returns
        -------
        str
            Detected mode as string: 'complex', 'intensity', or 'amplitude'.

        Examples
        --------
        >>> handler = InputFieldHandler(grid)
        >>> complex_field = torch.randn(256, 256, dtype=torch.complex64)
        >>> handler.detect_input_mode(complex_field)
        'complex'

        >>> intensity = torch.rand(256, 256)
        >>> handler.detect_input_mode(intensity)  # Warns and returns 'intensity'
        'intensity'
        """
        mode_enum = detect_input_mode(field)
        return mode_enum.value

    def _parse_input_mode(self, input_mode: str) -> InputMode:
        """Parse string input mode to InputMode enum.

        Parameters
        ----------
        input_mode : str
            Input mode string.

        Returns
        -------
        InputMode
            Corresponding enum value.

        Raises
        ------
        ValueError
            If input_mode is not recognized.
        """
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

        return mode_map[input_mode]

    def _standardize_dimensions(self, field: Tensor) -> Tuple[Tensor, dict]:
        """Convert field to [B, C, H, W] format and save squeeze information.

        Parameters
        ----------
        field : Tensor
            Input field with shape [H, W], [C, H, W], or [B, C, H, W].

        Returns
        -------
        field_standardized : Tensor
            Field with shape [B, C, H, W].
        squeeze_info : dict
            Information needed to restore original dimensions:
            - 'ndim': Original number of dimensions
            - 'batch_squeezed': Whether batch dimension was added
            - 'channel_squeezed': Whether channel dimension was added

        Examples
        --------
        >>> field_2d = torch.rand(256, 256)
        >>> field_std, info = handler._standardize_dimensions(field_2d)
        >>> field_std.shape
        torch.Size([1, 1, 256, 256])
        >>> info
        {'ndim': 2, 'batch_squeezed': True, 'channel_squeezed': True}
        """
        original_ndim = field.ndim
        batch_squeezed = False
        channel_squeezed = False

        if field.ndim == 2:
            # [H, W] → [1, 1, H, W]
            field = field.unsqueeze(0).unsqueeze(0)
            batch_squeezed = True
            channel_squeezed = True
        elif field.ndim == 3:
            # [C, H, W] → [1, C, H, W]
            # Note: We assume [C, H, W] not [B, H, W]
            field = field.unsqueeze(0)
            batch_squeezed = True
        elif field.ndim == 4:
            # Already [B, C, H, W]
            pass
        else:
            raise ValueError(
                f"Input field must have 2, 3, or 4 dimensions, got {field.ndim}. "
                f"Expected shapes: [H, W], [C, H, W], or [B, C, H, W]."
            )

        squeeze_info = {
            "ndim": original_ndim,
            "batch_squeezed": batch_squeezed,
            "channel_squeezed": channel_squeezed,
        }

        return field, squeeze_info

    def _restore_dimensions(self, field: Tensor, squeeze_info: dict) -> Tensor:
        """Restore original dimensions from [B, C, H, W] format.

        Parameters
        ----------
        field : Tensor
            Standardized field with shape [B, C, H, W].
        squeeze_info : dict
            Information from _standardize_dimensions about original shape.

        Returns
        -------
        Tensor
            Field with original dimensionality restored.

        Examples
        --------
        >>> field_std = torch.rand(1, 1, 256, 256, dtype=torch.complex64)
        >>> squeeze_info = {'ndim': 2, 'batch_squeezed': True, 'channel_squeezed': True}
        >>> field_restored = handler._restore_dimensions(field_std, squeeze_info)
        >>> field_restored.shape
        torch.Size([256, 256])
        """
        if squeeze_info["ndim"] == 2:
            # [1, 1, H, W] → [H, W]
            field = field.squeeze(0).squeeze(0)
        elif squeeze_info["ndim"] == 3:
            # [1, C, H, W] → [C, H, W]
            field = field.squeeze(0)
        # else: ndim == 4, keep as is

        return field

"""Four-F optical system forward model.

This module implements a unified 4f optical system forward model that propagates
a complex field through the classical four-focal-length (4f) imaging system.

The 4f system is the canonical configuration for Fourier optics, consisting of
two lenses separated by the sum of their focal lengths, with pupils placed at
the common back focal plane.

Physical Model
--------------
The forward model implements:

    I(x,y) = |IFFT{ P_det · P_illum · FFT{E_object} }|²

where:
    - E_object: Complex field at object plane
    - FFT: Fourier transform (object to pupil plane)
    - P_illum: Illumination pupil function
    - P_det: Detection pupil function
    - IFFT: Inverse Fourier transform (pupil to image plane)
    - I: Detected intensity

This simplified model assumes:
    - Object at the front focal plane of the first lens
    - Pupils at the common back focal plane (Fourier plane)
    - Image at the back focal plane of the second lens
    - Thin lens approximation

For defocused or more complex systems, use MicroscopeForwardModel with
full propagation mode.

Key Features
------------
- Batch dimension support: handles [B, C, H, W] or [H, W] inputs
- FFT padding: prevents wraparound artifacts with configurable padding factor
- Power-of-2 sizing: automatic rounding for FFT efficiency
- Normalization: optional output normalization to [0, 1]
- Complex field support: return complex field or intensity

See Also
--------
MicroscopeForwardModel : Full microscope model with lens phases and defocus
Grid : Spatial grid management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from loguru import logger
from torch import Tensor, nn


if TYPE_CHECKING:
    from prism.core.grid import Grid


class FourFForwardModel(nn.Module):
    """Unified 4f optical system forward model.

    Implements the classical four-focal-length (4f) imaging system with
    automatic FFT padding to prevent wraparound artifacts. The model supports
    both intensity and complex field outputs.

    The 4f system model:
        1. Pad input field to prevent FFT artifacts
        2. FFT to Fourier plane (pupil plane)
        3. Apply illumination pupil
        4. Apply detection pupil
        5. IFFT to image plane
        6. Crop to original size
        7. Convert to intensity (or return complex field)

    Parameters
    ----------
    grid : Grid
        Spatial grid for the optical system. Defines the sampling and
        wavelength for the simulation.
    padding_factor : float, optional
        Factor by which to pad the grid for FFT operations. Must be >= 1.0.
        Default 2.0. Padding helps reduce FFT wraparound artifacts.
        The padded size is rounded to the next power of 2 for efficiency.
    normalize_output : bool, optional
        If True, normalize output intensity to [0, 1] range. Default True.
        Only applies when return_complex=False.

    Attributes
    ----------
    grid : Grid
        The spatial grid for the system.
    padding_factor : float
        The padding factor used for FFT operations.
    normalize_output : bool
        Whether to normalize output intensity.
    original_size : tuple[int, int]
        Original grid size (nx, ny) before padding.
    padded_size : tuple[int, int]
        Padded grid size for FFT operations, rounded to power of 2.

    Examples
    --------
    >>> from prism.core.grid import Grid
    >>> from prism.core.optics import FourFForwardModel
    >>> import torch
    >>>
    >>> # Create grid
    >>> grid = Grid(nx=128, dx=1e-6, wavelength=532e-9)
    >>>
    >>> # Create 4f model
    >>> model = FourFForwardModel(grid, padding_factor=2.0)
    >>>
    >>> # Create input field
    >>> field = torch.randn(128, 128, dtype=torch.complex64)
    >>>
    >>> # Create pupil functions
    >>> pupil_illum = torch.ones(128, 128, dtype=torch.complex64)
    >>> pupil_det = torch.ones(128, 128, dtype=torch.complex64)
    >>>
    >>> # Forward pass (returns intensity)
    >>> intensity = model(field, pupil_illum, pupil_det)
    >>> print(intensity.shape)  # (128, 128)
    >>>
    >>> # Get complex field
    >>> field_out = model(field, pupil_illum, pupil_det, return_complex=True)
    >>> print(field_out.dtype)  # torch.complex64

    Notes
    -----
    This is a simplified 4f model that assumes the object is at the front
    focal plane. For more complex scenarios with defocus or z-stacks, use
    MicroscopeForwardModel with the FULL regime.

    The padding and cropping operations maintain the field centering
    convention, ensuring that DC remains at the center of the grid
    throughout the propagation.
    """

    def __init__(
        self,
        grid: Grid,
        padding_factor: float = 2.0,
        normalize_output: bool = True,
    ) -> None:
        """Initialize the 4f forward model.

        Parameters
        ----------
        grid : Grid
            Spatial grid for the optical system.
        padding_factor : float, optional
            Padding factor for FFT operations (>= 1.0). Default 2.0.
        normalize_output : bool, optional
            Whether to normalize output to [0, 1]. Default True.

        Raises
        ------
        ValueError
            If padding_factor < 1.0.
        """
        super().__init__()

        self.grid = grid
        self.normalize_output = normalize_output

        # Validate and store padding factor
        if padding_factor < 1.0:
            raise ValueError(f"padding_factor must be >= 1.0, got {padding_factor}")
        self.padding_factor = padding_factor
        self.original_size = (grid.nx, grid.ny)

        # Compute padded size (round to power of 2 for FFT efficiency)
        if padding_factor > 1.0:
            padded_n = int(grid.nx * padding_factor)
            # Round up to next power of 2
            padded_n = 2 ** int(torch.tensor(padded_n).float().log2().ceil())
            self.padded_size = (padded_n, padded_n)
            logger.debug(
                f"4f model: padding from {self.original_size} to {self.padded_size} "
                f"(factor={padding_factor:.1f})"
            )
        else:
            self.padded_size = self.original_size
            logger.debug(f"4f model: no padding (factor={padding_factor})")

    def _pad(self, field: Tensor) -> Tensor:
        """Zero-pad field for FFT anti-aliasing.

        Pads the field by centering the original data in a larger array.
        This prevents wraparound artifacts during FFT operations.

        Parameters
        ----------
        field : Tensor
            Input field of shape (..., H, W).

        Returns
        -------
        Tensor
            Padded field with size based on padding_factor.

        Notes
        -----
        If padding_factor <= 1.0, returns the input unchanged.
        """
        if self.padding_factor <= 1.0:
            return field

        h, w = field.shape[-2:]
        target_h, target_w = self.padded_size

        # Calculate padding for each side (center the original)
        pad_h = target_h - h
        pad_w = target_w - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # F.pad uses (left, right, top, bottom) order
        return torch.nn.functional.pad(
            field,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

    def _crop(self, field: Tensor) -> Tensor:
        """Crop padded field back to original size.

        Extracts the center region of the padded field, restoring the
        original dimensions.

        Parameters
        ----------
        field : Tensor
            Padded field of shape (..., H_pad, W_pad).

        Returns
        -------
        Tensor
            Cropped field with original shape (..., H, W).

        Notes
        -----
        If padding_factor <= 1.0, returns the input unchanged.
        """
        if self.padding_factor <= 1.0:
            return field

        h, w = field.shape[-2:]
        target_h, target_w = self.original_size

        # Calculate crop indices (center crop)
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2

        return field[..., start_h : start_h + target_h, start_w : start_w + target_w]

    def _handle_input_dimensions(self, field: Tensor) -> tuple[Tensor, dict]:
        """Standardize input to [B, C, H, W] format.

        Converts various input shapes to a standard batch format, recording
        the transformations needed to restore the original shape.

        Parameters
        ----------
        field : Tensor
            Input field. Can be:
            - [H, W]: Single field
            - [C, H, W]: Multi-channel field
            - [B, C, H, W]: Batched multi-channel field

        Returns
        -------
        Tensor
            Field in [B, C, H, W] format.
        dict
            Information needed to restore original dimensions:
            - 'ndim': Original number of dimensions
            - 'shape': Original shape

        Raises
        ------
        ValueError
            If input has fewer than 2 or more than 4 dimensions.
        """
        squeeze_info = {
            "ndim": field.ndim,
            "shape": field.shape,
        }

        if field.ndim == 2:
            # [H, W] -> [1, 1, H, W]
            field = field.unsqueeze(0).unsqueeze(0)
        elif field.ndim == 3:
            # [C, H, W] -> [1, C, H, W]
            field = field.unsqueeze(0)
        elif field.ndim == 4:
            # [B, C, H, W] - already in correct format
            pass
        else:
            raise ValueError(f"Input field must be 2D, 3D, or 4D, got shape {field.shape}")

        return field, squeeze_info

    def _restore_dimensions(self, output: Tensor, squeeze_info: dict) -> Tensor:
        """Restore original dimensions after processing.

        Removes the batch and channel dimensions that were added during
        input standardization.

        Parameters
        ----------
        output : Tensor
            Field in [B, C, H, W] format.
        squeeze_info : dict
            Information from _handle_input_dimensions about original shape.

        Returns
        -------
        Tensor
            Field with original dimensions restored.
        """
        ndim = squeeze_info["ndim"]

        if ndim == 2:
            # [1, 1, H, W] -> [H, W]
            output = output.squeeze(0).squeeze(0)
        elif ndim == 3:
            # [1, C, H, W] -> [C, H, W]
            output = output.squeeze(0)
        # ndim == 4: no change needed

        return output

    def forward(
        self,
        field: Tensor,
        illumination_pupil: Optional[Tensor] = None,
        detection_pupil: Optional[Tensor] = None,
        return_complex: bool = False,
    ) -> Tensor:
        """Forward propagation through 4f system.

        Propagates a complex field through the 4f optical system, applying
        illumination and detection pupil functions at the Fourier plane.

        Parameters
        ----------
        field : Tensor
            Complex field at object plane. Shape can be:
            - [H, W]: Single field
            - [C, H, W]: Multi-channel field
            - [B, C, H, W]: Batched field
        illumination_pupil : Tensor, optional
            Illumination pupil function in Fourier domain. If None, uses
            all-pass (ones). Shape should match grid size or be broadcastable.
        detection_pupil : Tensor, optional
            Detection pupil function in Fourier domain. If None, uses
            all-pass (ones). Shape should match grid size or be broadcastable.
        return_complex : bool, optional
            If True, return complex field at image plane. If False, return
            intensity. Default False.

        Returns
        -------
        Tensor
            If return_complex=True: Complex field at image plane
            If return_complex=False: Intensity at image plane
            Output shape matches input shape.

        Notes
        -----
        The pupils are applied at the Fourier plane (back focal plane of
        the first lens). Both pupils should be provided as complex-valued
        tensors that include both amplitude and phase modulation.

        If normalize_output=True and return_complex=False, the output
        intensity is normalized to [0, 1].
        """
        # Handle input dimensions
        field_batched, squeeze_info = self._handle_input_dimensions(field)

        # Create default pupils if not provided
        device = field_batched.device
        dtype = field_batched.dtype

        if illumination_pupil is None:
            # All-pass pupil
            illum_pupil = torch.ones(self.original_size, dtype=dtype, device=device)
        else:
            illum_pupil = illumination_pupil.to(device)

        if detection_pupil is None:
            # All-pass pupil
            detect_pupil = torch.ones(self.original_size, dtype=dtype, device=device)
        else:
            detect_pupil = detection_pupil.to(device)

        # Step 1: Pad input field (operates on last two dimensions)
        field_padded = self._pad(field_batched)

        # Step 2: Pad pupils to match padded size (only once)
        illum_padded = self._pad(illum_pupil)
        detect_padded = self._pad(detect_pupil)

        # Step 3: Object to back focal plane (Fourier transform)
        # Process all batch and channel dimensions at once
        field_bfp = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.fftshift(field_padded, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )

        # Step 4: Apply illumination pupil (broadcast over batch and channel)
        field_illum = field_bfp * illum_padded

        # Step 5: Apply detection pupil (broadcast over batch and channel)
        field_filtered = field_illum * detect_padded

        # Step 6: Back focal plane to image plane (inverse Fourier transform)
        field_image = torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.ifftshift(field_filtered, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )

        # Step 7: Crop back to original size
        output = self._crop(field_image)

        # Convert to intensity if requested
        if not return_complex:
            output = torch.abs(output) ** 2

            # Normalize if requested
            if self.normalize_output:
                max_val = output.max()
                if max_val > 0:
                    output = output / max_val

        # Restore original dimensions
        output = self._restore_dimensions(output, squeeze_info)

        return output

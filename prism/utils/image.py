"""
Module: spids.utils.image
Purpose: Image loading and basic operations for astronomical imaging
Dependencies: torch, torchvision, PIL, numpy
Main Functions:
    - load_image(path, size, padded_size, invert): Load and preprocess images from disk
    - generate_point_sources(image_size, number_of_sources, sample_diameter, spacing):
      Create synthetic point source images
    - pad_image(tensor, target_size, mode, value): Pad tensor to target size
    - crop_image(tensor, target_size): Crop tensor to target size
    - crop_pad(tensor, target_size, mode, value): Crop or pad tensors to target size
    - get_image_size(path): Get maximum dimension of image

Description:
    This module provides image loading and basic manipulation functionality for SPIDS.
    It handles image I/O with automatic preprocessing (resizing, padding, normalization),
    and provides utilities for cropping, padding, and generating synthetic test images.
"""

from __future__ import annotations

from typing import Literal, Optional, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor

from prism.types import PathLike


if int(torch.__version__[0]) >= 2:
    from torchvision import disable_beta_transforms_warning

    disable_beta_transforms_warning()
    from torchvision.transforms import v2 as T  # noqa: N812
else:
    from torchvision import transforms as T  # noqa: N812


def load_image(
    path: PathLike,
    size: Optional[int] = None,
    padded_size: Optional[int] = None,
    invert: bool = False,
) -> Tensor:
    """
    Load image from path and apply preprocessing.

    Args:
        path: Path to image file
        size: Target size for the image (default: max dimension)
        padded_size: Size after padding (default: same as size)
        invert: Whether to invert image colors

    Returns:
        Preprocessed image tensor, normalized and square-rooted
    """
    with Image.open(path) as image:
        if size is None:
            size = max(image.size)
        if padded_size is None:
            padded_size = size
        padding_pre = (padded_size - size) // 2
        padding_post = padded_size - size - padding_pre
        # Create a transform to load and preprocess the image
        if int(torch.__version__[0]) >= 2:
            transform = T.Compose(
                [
                    T.PILToTensor(),
                    T.Grayscale(),
                    T.Resize(size),
                    T.RandomInvert(1 if invert else 0),
                    T.Pad((padding_pre, padding_pre, padding_post, padding_post), fill=0),
                    T.ToDtype(torch.float32),
                ]
            )
            image_tensor = transform(image)
        else:
            transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Resize(size),
                    T.RandomInvert(1 if invert else 0),
                    T.Pad((padding_pre, padding_pre, padding_post, padding_post), fill=0),
                ]
            )
            image_tensor = transform(image).to(torch.float32)

    image_tensor -= image_tensor.min()
    max_val = image_tensor.max()
    if max_val > 0:
        image_tensor /= max_val
    result = image_tensor.sqrt()
    return cast(Tensor, result)


def generate_point_sources(
    image_size: int, number_of_sources: Literal[1, 2, 3, 4], sample_diameter: float, spacing: float
) -> Tensor:
    """
    Generate synthetic point source image for testing.

    Args:
        image_size: Size of output image
        number_of_sources: Number of point sources (1-4)
        sample_diameter: Diameter of each point source
        spacing: Spacing between point sources

    Returns:
        Binary tensor with point sources
    """
    x = torch.arange(0, image_size).unsqueeze(0) - image_size // 2
    y = torch.arange(0, image_size).unsqueeze(1) - image_size // 2

    if number_of_sources == 1:
        centers = torch.tensor([[0, 0]])
    elif number_of_sources == 2:
        centers = torch.tensor([[0, -spacing / 2], [0, spacing / 2]])
    elif number_of_sources == 3:
        centers = torch.tensor(
            [
                [spacing / np.sqrt(3), 0],
                [-spacing / np.sqrt(3) / 2, spacing / 2],
                [-spacing / np.sqrt(3) / 2, -spacing / 2],
            ]
        )
    elif number_of_sources == 4:
        centers = torch.tensor(
            [
                [spacing / 2, spacing / 2],
                [spacing / 2, -spacing / 2],
                [-spacing / 2, spacing / 2],
                [-spacing / 2, -spacing / 2],
            ]
        )
    else:
        raise NotImplementedError
    return torch.stack(
        [
            ((x - center[1]) ** 2 + (y - center[0]) ** 2) <= (sample_diameter / 2) ** 2
            for center in centers
        ],
        dim=0,
    )


def get_image_size(path: PathLike) -> int:
    """
    Get the maximum dimension of an image without loading it fully.

    Args:
        path: Path to image file

    Returns:
        Maximum dimension (width or height) as integer
    """
    with Image.open(path) as image:
        width, height = image.size
        return int(max(width, height))


def pad_image(tensor: Tensor, target_size: int, mode: str = "constant", value: float = 0) -> Tensor:
    """
    Pad tensor to target size.

    Args:
        tensor: Input tensor to pad
        target_size: Target size after padding
        mode: Padding mode (constant, reflect, etc.)
        value: Fill value for constant padding

    Returns:
        Padded tensor
    """
    # Make sure the tensor is a PyTorch tensor
    assert isinstance(tensor, torch.Tensor)

    # Get the current shape of the tensor
    shape = tensor.size()

    # Determine the padding needed for each dimension
    pad_dim1 = max(0, target_size - shape[-2]) // 2
    pad_dim2 = max(0, target_size - shape[-1]) // 2

    # Account for odd padding
    pad_dim1_post = max(0, target_size - shape[-2] - pad_dim1)
    pad_dim2_post = max(0, target_size - shape[-1] - pad_dim2)

    # Create padding configuration
    padding = [pad_dim2, pad_dim2_post, pad_dim1, pad_dim1_post]

    # Return the padded tensor
    return F.pad(tensor, padding, mode, value)


def crop_image(tensor: Tensor, target_size: int) -> Tensor:
    """
    Crop tensor to target size from center.

    Args:
        tensor: Input tensor to crop
        target_size: Target size after cropping

    Returns:
        Cropped tensor
    """
    ny, nx = tensor.size()[-2], tensor.size()[-1]
    dy = int((ny - target_size) / 2) + 1
    dx = int((nx - target_size) / 2) + 1
    return tensor[..., dy : dy + target_size, dx : dx + target_size]


def crop_pad(
    tensor: Tensor, target_size: Optional[int], mode: str = "constant", value: float = 0
) -> Tensor:
    """
    Crop or pad tensor to target size as needed.

    Args:
        tensor: Input tensor
        target_size: Target size (if None, returns tensor unchanged)
        mode: Padding mode if padding is needed
        value: Fill value for constant padding

    Returns:
        Tensor resized to target_size, or unchanged if target_size is None
    """
    if target_size is None:
        return tensor

    return (
        pad_image(tensor, target_size, mode, value)
        if target_size >= max(tensor.size())
        else crop_image(tensor, target_size)
    )

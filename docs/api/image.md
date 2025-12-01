# prism.utils.image

Module: prism.utils.image
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
    This module provides image loading and basic manipulation functionality for PRISM.
    It handles image I/O with automatic preprocessing (resizing, padding, normalization),
    and provides utilities for cropping, padding, and generating synthetic test images.

## Classes

## Functions

### crop_image

```python
crop_image(tensor: torch.Tensor, target_size: int) -> torch.Tensor
```

Crop tensor to target size from center.

Args:
    tensor: Input tensor to crop
    target_size: Target size after cropping

Returns:
    Cropped tensor

### crop_pad

```python
crop_pad(tensor: torch.Tensor, target_size: Optional[int], mode: str = 'constant', value: float = 0) -> torch.Tensor
```

Crop or pad tensor to target size as needed.

Args:
    tensor: Input tensor
    target_size: Target size (if None, returns tensor unchanged)
    mode: Padding mode if padding is needed
    value: Fill value for constant padding

Returns:
    Tensor resized to target_size, or unchanged if target_size is None

### generate_point_sources

```python
generate_point_sources(image_size: int, number_of_sources: Literal[1, 2, 3, 4], sample_diameter: float, spacing: float) -> torch.Tensor
```

Generate synthetic point source image for testing.

Args:
    image_size: Size of output image
    number_of_sources: Number of point sources (1-4)
    sample_diameter: Diameter of each point source
    spacing: Spacing between point sources

Returns:
    Binary tensor with point sources

### get_image_size

```python
get_image_size(path: Union[str, ForwardRef('Path')]) -> int
```

Get the maximum dimension of an image without loading it fully.

Args:
    path: Path to image file

Returns:
    Maximum dimension (width or height) as integer

### load_image

```python
load_image(path: Union[str, ForwardRef('Path')], size: Optional[int] = None, padded_size: Optional[int] = None, invert: bool = False) -> torch.Tensor
```

Load image from path and apply preprocessing.

Args:
    path: Path to image file
    size: Target size for the image (default: max dimension)
    padded_size: Size after padding (default: same as size)
    invert: Whether to invert image colors

Returns:
    Preprocessed image tensor, normalized and square-rooted

### pad_image

```python
pad_image(tensor: torch.Tensor, target_size: int, mode: str = 'constant', value: float = 0) -> torch.Tensor
```

Pad tensor to target size.

Args:
    tensor: Input tensor to pad
    target_size: Target size after padding
    mode: Padding mode (constant, reflect, etc.)
    value: Fill value for constant padding

Returns:
    Padded tensor

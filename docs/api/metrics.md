# prism.utils.metrics

Module: prism.utils.metrics
Purpose: Image quality metrics for reconstruction evaluation
Dependencies: torch, skimage, numpy
Main Functions:
    - compute_ssim(img1, img2, size): Structural Similarity Index metric
    - compute_rmse(img1, img2, size): Normalized Root Mean Square Error
    - psnr(img1, img2, size, data_range): Peak Signal-to-Noise Ratio in dB

Description:
    This module provides quality metrics (SSIM, RMSE, PSNR) for evaluating
    reconstruction quality in PRISM experiments.

## Classes

## Functions

### compute_rmse

```python
compute_rmse(img1: torch.Tensor, img2: torch.Tensor, size: Optional[int] = None) -> float
```

Calculate normalized Root Mean Square Error (RMSE) between two images.

Args:
    img1: First image tensor
    img2: Second image tensor (reference)
    size: Optional size to crop both images before comparison

Returns:
    Normalized RMSE as percentage

### compute_ssim

```python
compute_ssim(img1: torch.Tensor, img2: torch.Tensor, size: Optional[int] = None) -> float
```

Calculate Structural Similarity Index (SSIM) between two images.

Args:
    img1: First image tensor
    img2: Second image tensor
    size: Optional size to crop both images before comparison

Returns:
    SSIM value (higher is better, max 1.0)

### psnr

```python
psnr(img1: torch.Tensor, img2: torch.Tensor, size: Optional[int] = None, data_range: float = 1.0) -> float
```

Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

Args:
    img1: First image tensor
    img2: Second image tensor (reference)
    size: Optional size to crop both images
    data_range: Data range of the images (default: 1.0)

Returns:
    PSNR value in dB

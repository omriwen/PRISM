# prism.core.optics.four_f_forward

Four-F optical system forward model.

This module implements a unified 4f optical system forward model that propagates a complex field through the classical four-focal-length (4f) imaging system.

## Overview

The 4f system is the canonical configuration for Fourier optics, consisting of two lenses separated by the sum of their focal lengths, with pupils placed at the common back focal plane.

**Physical Model**:

The forward model implements:
```
I(x,y) = |IFFT{ P_det · P_illum · FFT{E_object} }|²
```

where:
- `E_object`: Complex field at object plane
- `FFT`: Fourier transform (object to pupil plane)
- `P_illum`: Illumination pupil function
- `P_det`: Detection pupil function
- `IFFT`: Inverse Fourier transform (pupil to image plane)
- `I`: Detected intensity

**Assumptions**:
- Object at the front focal plane of the first lens
- Pupils at the common back focal plane (Fourier plane)
- Image at the back focal plane of the second lens
- Thin lens approximation

For defocused or more complex systems, use `MicroscopeForwardModel` with full propagation mode.

**Key Features**:
- Batch dimension support: handles [B, C, H, W] or [H, W] inputs
- FFT padding: prevents wraparound artifacts with configurable padding factor
- Power-of-2 sizing: automatic rounding for FFT efficiency
- Normalization: optional output normalization to [0, 1]
- Complex field support: return complex field or intensity

## Classes

### FourFForwardModel

```python
FourFForwardModel(
    grid: Grid,
    padding_factor: float = 2.0,
    normalize_output: bool = True
)
```

Unified 4f optical system forward model.

Implements the classical four-focal-length (4f) imaging system with automatic FFT padding to prevent wraparound artifacts. The model supports both intensity and complex field outputs.

**The 4f system model**:
1. Pad input field to prevent FFT artifacts
2. FFT to Fourier plane (pupil plane)
3. Apply illumination pupil
4. Apply detection pupil
5. IFFT to image plane
6. Crop to original size
7. Convert to intensity (or return complex field)

#### Parameters

- **grid** : `Grid`

  Spatial grid for the optical system. Defines the sampling and wavelength for the simulation.

- **padding_factor** : `float`, default=2.0

  Factor by which to pad the grid for FFT operations. Must be >= 1.0. Padding helps reduce FFT wraparound artifacts. The padded size is rounded to the next power of 2 for efficiency.

- **normalize_output** : `bool`, default=True

  If True, normalize output intensity to [0, 1] range. Only applies when return_complex=False.

#### Attributes

- **grid** : `Grid`

  The spatial grid for the system.

- **padding_factor** : `float`

  The padding factor used for FFT operations.

- **normalize_output** : `bool`

  Whether to normalize output intensity.

- **original_size** : `tuple[int, int]`

  Original grid size (nx, ny) before padding.

- **padded_size** : `tuple[int, int]`

  Padded grid size for FFT operations, rounded to power of 2.

#### Methods

##### `__init__`

Initialize the 4f forward model.

```python
__init__(
    grid: Grid,
    padding_factor: float = 2.0,
    normalize_output: bool = True
) -> None
```

**Parameters**:
- **grid** : Spatial grid for the optical system.
- **padding_factor** : Padding factor for FFT operations (>= 1.0). Default 2.0.
- **normalize_output** : Whether to normalize output to [0, 1]. Default True.

**Raises**:
- **ValueError** : If padding_factor < 1.0.

##### `forward`

Forward propagation through 4f system.

```python
forward(
    field: Tensor,
    illumination_pupil: Optional[Tensor] = None,
    detection_pupil: Optional[Tensor] = None,
    return_complex: bool = False
) -> Tensor
```

Propagates a complex field through the 4f optical system, applying illumination and detection pupil functions at the Fourier plane.

**Parameters**:
- **field** : Complex field at object plane. Shape can be:
  - [H, W]: Single field
  - [C, H, W]: Multi-channel field
  - [B, C, H, W]: Batched field
- **illumination_pupil** : Illumination pupil function in Fourier domain. If None, uses all-pass (ones). Shape should match grid size or be broadcastable.
- **detection_pupil** : Detection pupil function in Fourier domain. If None, uses all-pass (ones). Shape should match grid size or be broadcastable.
- **return_complex** : If True, return complex field at image plane. If False, return intensity. Default False.

**Returns**:
- **Tensor** : If return_complex=True: Complex field at image plane. If return_complex=False: Intensity at image plane. Output shape matches input shape.

**Notes**:

The pupils are applied at the Fourier plane (back focal plane of the first lens). Both pupils should be provided as complex-valued tensors that include both amplitude and phase modulation.

If normalize_output=True and return_complex=False, the output intensity is normalized to [0, 1].

## Usage Examples

### Basic Usage

```python
from prism.core.grid import Grid
from prism.core.optics.four_f_forward import FourFForwardModel
import torch

# Create grid
grid = Grid(nx=128, dx=1e-6, wavelength=532e-9)

# Create 4f model
model = FourFForwardModel(grid, padding_factor=2.0)

# Create input field
field = torch.randn(128, 128, dtype=torch.complex64)

# Create pupil functions
pupil_illum = torch.ones(128, 128, dtype=torch.complex64)
pupil_det = torch.ones(128, 128, dtype=torch.complex64)

# Forward pass (returns intensity)
intensity = model(field, pupil_illum, pupil_det)
print(intensity.shape)  # (128, 128)

# Get complex field
field_out = model(field, pupil_illum, pupil_det, return_complex=True)
print(field_out.dtype)  # torch.complex64
```

### Batch Processing

```python
# Batch of fields
batch_fields = torch.randn(8, 1, 128, 128, dtype=torch.complex64)

# Forward pass with batching
batch_intensity = model(batch_fields, pupil_illum, pupil_det)
print(batch_intensity.shape)  # (8, 1, 128, 128)
```

### Custom Pupil Functions

```python
import numpy as np

# Create NA-limited circular pupil
def create_circular_pupil(grid, na, wavelength):
    kx = grid.kx
    ky = grid.ky
    k_cutoff = na / wavelength
    r_k = torch.sqrt(kx**2 + ky**2)
    pupil = (r_k <= k_cutoff).float()
    return pupil.to(torch.complex64)

# Use custom pupil
detect_pupil = create_circular_pupil(grid, na=1.4, wavelength=532e-9)
intensity = model(field, illumination_pupil=None, detection_pupil=detect_pupil)
```

### Padding for Anti-Aliasing

```python
# No padding (risk of wraparound artifacts)
model_no_pad = FourFForwardModel(grid, padding_factor=1.0)

# 2x padding (recommended)
model_2x = FourFForwardModel(grid, padding_factor=2.0)

# 4x padding (extra margin for high-frequency content)
model_4x = FourFForwardModel(grid, padding_factor=4.0)

# Compare outputs
intensity_no_pad = model_no_pad(field, pupil_illum, pupil_det)
intensity_2x = model_2x(field, pupil_illum, pupil_det)
intensity_4x = model_4x(field, pupil_illum, pupil_det)
```

### Different Input Shapes

```python
# 2D input (single field)
field_2d = torch.randn(128, 128, dtype=torch.complex64)
output_2d = model(field_2d)  # Output: (128, 128)

# 3D input (multi-channel)
field_3d = torch.randn(3, 128, 128, dtype=torch.complex64)
output_3d = model(field_3d)  # Output: (3, 128, 128)

# 4D input (batched multi-channel)
field_4d = torch.randn(8, 3, 128, 128, dtype=torch.complex64)
output_4d = model(field_4d)  # Output: (8, 3, 128, 128)
```

### Complex Field Output

```python
# Get complex field instead of intensity
complex_output = model(field, pupil_illum, pupil_det, return_complex=True)

# Extract amplitude and phase
amplitude = torch.abs(complex_output)
phase = torch.angle(complex_output)

# Convert to intensity manually
intensity_manual = amplitude ** 2
```

## Implementation Details

### Padding and Cropping

The model uses symmetric padding to prevent FFT wraparound artifacts:

```python
# Input: [H, W]
# After padding: [H_pad, W_pad] where H_pad = 2^ceil(log2(H * padding_factor))
# After cropping: [H, W] (original size restored)
```

Padding is centered, so DC component remains at the center of the grid.

### Dimension Handling

The model automatically handles different input dimensions:

1. **[H, W]** → Add batch and channel dimensions → **[1, 1, H, W]**
2. **[C, H, W]** → Add batch dimension → **[1, C, H, W]**
3. **[B, C, H, W]** → Already in correct format

After processing, the original dimensions are restored.

### FFT Convention

The model uses the following FFT convention:

```python
# Forward FFT (object → k-space)
field_k = fftshift(fft2(fftshift(field)))

# Inverse FFT (k-space → image)
field_img = ifftshift(ifft2(ifftshift(field_k)))
```

This ensures DC is at the center of the k-space grid, consistent with pupil mask conventions.

### Normalization

When `normalize_output=True` (default):

```python
intensity = torch.abs(field_out) ** 2
if intensity.max() > 0:
    intensity = intensity / intensity.max()
```

This ensures output is in [0, 1] range, making it easier to visualize and compare.

## Performance Considerations

### Power-of-2 Sizing

The padded size is automatically rounded up to the next power of 2 for FFT efficiency:

```python
# Example: grid size 128 with padding_factor=2.0
# Target: 128 * 2.0 = 256
# Actual: 256 (already power of 2)

# Example: grid size 100 with padding_factor=2.0
# Target: 100 * 2.0 = 200
# Actual: 256 (next power of 2 above 200)
```

This provides optimal FFT performance while ensuring sufficient padding.

### Memory Usage

Memory scales with the padded size:

```
Memory ≈ B × C × H_pad × W_pad × dtype_size
```

For large grids with high padding factors, consider:
- Reducing padding_factor (minimum 1.0, no padding)
- Processing in smaller batches
- Using GPU acceleration for large FFTs

### GPU Acceleration

The model works seamlessly with CUDA:

```python
# Move model and data to GPU
device = torch.device('cuda')
grid_gpu = grid.to(device)
model_gpu = FourFForwardModel(grid_gpu)
field_gpu = field.to(device)
pupil_gpu = pupil_det.to(device)

# Forward pass on GPU
intensity_gpu = model_gpu(field_gpu, detection_pupil=pupil_gpu)
```

## Notes

This is a simplified 4f model that assumes the object is at the front focal plane. For more complex scenarios with defocus or z-stacks, use `MicroscopeForwardModel` with the FULL regime.

The padding and cropping operations maintain the field centering convention, ensuring that DC remains at the center of the grid throughout the propagation.

## See Also

- `prism.core.instruments.four_f_base.FourFSystem` : Instrument base class using this forward model
- `prism.core.grid.Grid` : Spatial grid management
- `prism.core.optics.aperture_masks.ApertureMaskGenerator` : Pupil mask generation
- `prism.core.propagators.microscope.MicroscopeForwardModel` : Full microscope model with defocus

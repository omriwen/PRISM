# prism.core.optics.detector_noise

Detector noise model for realistic imaging simulations.

This module provides a PyTorch nn.Module for simulating realistic detector noise including shot noise (Poisson statistics), read noise (Gaussian), dark current, and optional quantization.

## Overview

The `DetectorNoiseModel` class provides a unified interface for adding realistic detector noise to simulated images. It can operate in two modes:

1. **SNR-based mode**: Specify target SNR in dB, noise is automatically scaled
2. **Component-based mode**: Specify individual noise components explicitly

**Noise Components**:
- **Shot noise**: Poisson statistics from photon counting (signal-dependent)
- **Read noise**: Gaussian noise from detector electronics (signal-independent)
- **Dark current**: Thermal electrons accumulated during exposure
- **Quantization** (future): Analog-to-digital conversion effects

This model consolidates noise generation that was previously duplicated in Microscope (`_add_detector_noise()`) and Camera (`_add_sensor_noise()`).

## Classes

### DetectorNoiseModel

```python
DetectorNoiseModel(
    snr_db: Optional[float] = None,
    photon_scale: float = 1000.0,
    read_noise_fraction: float = 0.01,
    dark_current_fraction: float = 0.002,
    enabled: bool = True
)
```

Realistic detector noise model for imaging systems.

This class implements realistic detector noise including shot noise (Poisson statistics), read noise (Gaussian), dark current, and optional quantization.

#### Parameters

- **snr_db** : `float`, optional

  Target signal-to-noise ratio in dB. If provided, uses SNR-based noise model. The noise standard deviation is computed as: σ = signal_max / (10^(snr_db/20))

- **photon_scale** : `float`, default=1000.0

  Photon count scaling factor for shot noise. Higher values = higher photon counts = less relative noise. Only used when snr_db is None.

- **read_noise_fraction** : `float`, default=0.01

  Read noise as fraction of maximum signal (Gaussian, additive). Only used when snr_db is None.

- **dark_current_fraction** : `float`, default=0.002

  Dark current noise as fraction of maximum signal (Gaussian, additive). Only used when snr_db is None.

- **enabled** : `bool`, default=True

  Whether noise is enabled. Can be toggled with enable()/disable().

#### Attributes

- **snr_db** : `float` or `None`

  Target SNR in dB (None if using component-based mode)

- **photon_scale** : `float`

  Photon count scaling factor

- **read_noise_fraction** : `float`

  Read noise fraction

- **dark_current_fraction** : `float`

  Dark current fraction

- **enabled** : `bool`

  Whether noise is currently enabled

#### Methods

##### `__init__`

Initialize DetectorNoiseModel.

```python
__init__(
    snr_db: Optional[float] = None,
    photon_scale: float = 1000.0,
    read_noise_fraction: float = 0.01,
    dark_current_fraction: float = 0.002,
    enabled: bool = True
) -> None
```

**Parameters**:
- **snr_db** : Target SNR in dB. If provided, uses SNR-based noise model.
- **photon_scale** : Photon count scaling factor (used when snr_db is None)
- **read_noise_fraction** : Read noise as fraction of max signal (used when snr_db is None)
- **dark_current_fraction** : Dark current as fraction of max signal (used when snr_db is None)
- **enabled** : Whether noise is enabled

**Raises**:
- **ValueError** : If snr_db <= 0, photon_scale <= 0, or fractions < 0

##### `forward`

Add detector noise to intensity image.

```python
forward(intensity: Tensor, add_noise: bool = True) -> Tensor
```

**Parameters**:
- **intensity** : Input intensity image (clean, non-negative). Shape: (H, W) or (B, C, H, W)
- **add_noise** : Whether to add noise. If False or if enabled=False, returns input unchanged.

**Returns**:
- **Tensor** : Noisy intensity image with same shape as input, clamped to non-negative values

**Notes**:
- If enabled=False or add_noise=False, returns input unchanged
- Uses SNR-based noise if snr_db is set, otherwise uses component-based noise
- Output is always non-negative (clamped to >= 0)
- Shot noise variance scales with local signal intensity
- Read noise and dark current are additive with constant variance

##### `set_snr`

Update SNR level.

```python
set_snr(snr_db: float) -> None
```

This switches the model to SNR-based noise mode.

**Parameters**:
- **snr_db** : New target SNR in dB (must be positive)

**Raises**:
- **ValueError** : If snr_db is not positive

**Examples**:
```python
>>> noise_model = DetectorNoiseModel(photon_scale=1000)  # Component-based
>>> noise_model.set_snr(40.0)  # Switch to SNR-based
>>> noise_model.set_snr(50.0)  # Update SNR
```

##### `enable`

Enable noise addition.

```python
enable() -> None
```

**Examples**:
```python
>>> noise_model.disable()
>>> noise_model.enable()
```

##### `disable`

Disable noise addition.

```python
disable() -> None
```

When disabled, forward() returns input unchanged regardless of add_noise flag.

**Examples**:
```python
>>> noise_model = DetectorNoiseModel(snr_db=40.0)
>>> noise_model.disable()
>>> clean_output = noise_model(image, add_noise=True)  # No noise added
```

## Usage Examples

### SNR-Based Noise Model

```python
from prism.core.optics.detector_noise import DetectorNoiseModel
import torch

# Create SNR-based noise model
noise_model = DetectorNoiseModel(snr_db=40.0)

# Clean image
clean_image = torch.rand(256, 256)

# Add noise
noisy_image = noise_model(clean_image, add_noise=True)

# Adjust SNR dynamically
noise_model.set_snr(50.0)  # Less noise
less_noisy = noise_model(clean_image, add_noise=True)

noise_model.set_snr(30.0)  # More noise
more_noisy = noise_model(clean_image, add_noise=True)
```

### Component-Based Noise Model

```python
# Create component-based noise model
noise_model = DetectorNoiseModel(
    photon_scale=2000.0,      # High photon counts (low shot noise)
    read_noise_fraction=0.005, # 0.5% read noise
    dark_current_fraction=0.001, # 0.1% dark current
)

# Add noise
noisy_image = noise_model(clean_image, add_noise=True)
```

### Batch Processing

```python
# Batch of images
batch_images = torch.rand(8, 1, 256, 256)

# Add noise to entire batch
noisy_batch = noise_model(batch_images, add_noise=True)
print(noisy_batch.shape)  # (8, 1, 256, 256)
```

### Enable/Disable Noise

```python
# Temporarily disable noise
noise_model.disable()
clean_output = noise_model(image, add_noise=True)  # No noise added

# Re-enable noise
noise_model.enable()
noisy_output = noise_model(image, add_noise=True)  # Noise added
```

### Integration with Instruments

```python
from prism.core.instruments import MicroscopeConfig, create_instrument

# Create microscope with noise model
config = MicroscopeConfig(wavelength=532e-9, numerical_aperture=1.4)
microscope = create_instrument(config)

# Attach noise model to instrument
microscope._noise_model = DetectorNoiseModel(snr_db=45.0)

# Forward pass with noise
field = torch.randn(512, 512, dtype=torch.complex64)
noisy_image = microscope.forward(field, add_noise=True)

# Forward pass without noise
clean_image = microscope.forward(field, add_noise=False)
```

### Comparing Noise Levels

```python
import matplotlib.pyplot as plt

# Create different noise models
noise_models = {
    'SNR 60dB': DetectorNoiseModel(snr_db=60.0),
    'SNR 40dB': DetectorNoiseModel(snr_db=40.0),
    'SNR 20dB': DetectorNoiseModel(snr_db=20.0),
}

# Apply to same clean image
clean = torch.rand(256, 256)
results = {}

for name, model in noise_models.items():
    results[name] = model(clean, add_noise=True)

# Plot comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(clean, cmap='gray')
axes[0].set_title('Clean')
for idx, (name, noisy) in enumerate(results.items(), 1):
    axes[idx].imshow(noisy, cmap='gray')
    axes[idx].set_title(name)
plt.tight_layout()
plt.show()
```

### Custom Noise Configuration

```python
# Simulate high-quality scientific camera
scientific_camera = DetectorNoiseModel(
    photon_scale=5000.0,        # High quantum efficiency
    read_noise_fraction=0.002,  # Low read noise (< 3 electrons)
    dark_current_fraction=0.0005, # Very low dark current
)

# Simulate smartphone camera
smartphone_camera = DetectorNoiseModel(
    photon_scale=500.0,         # Lower quantum efficiency
    read_noise_fraction=0.02,   # Higher read noise
    dark_current_fraction=0.01,  # Higher dark current
)

# Apply to same scene
scene = torch.rand(512, 512)
scientific_image = scientific_camera(scene, add_noise=True)
smartphone_image = smartphone_camera(scene, add_noise=True)
```

## Noise Model Details

### SNR-Based Noise

When `snr_db` is specified, the noise is computed as:

```
σ_noise = signal_max / (10^(SNR_dB / 20))
noise = N(0, σ_noise)
I_noisy = I_clean + noise
```

where N(0, σ) is Gaussian noise with mean 0 and standard deviation σ.

**SNR Definition**:
```
SNR(dB) = 20 * log10(signal / noise_std)
```

### Component-Based Noise

When `snr_db` is None, noise components are added individually:

#### 1. Shot Noise (Signal-Dependent)

Shot noise arises from the discrete nature of photon detection (Poisson statistics). For high photon counts, Poisson noise can be approximated as Gaussian:

```
I_photons = I_clean * photon_scale
σ_shot = sqrt(I_photons)
shot_noise = N(0, σ_shot) / photon_scale
```

#### 2. Read Noise (Signal-Independent)

Read noise from detector electronics:

```
σ_read = read_noise_fraction * max(I_clean)
read_noise = N(0, σ_read)
```

#### 3. Dark Current (Signal-Independent)

Thermal electrons accumulated during exposure:

```
σ_dark = dark_current_fraction * max(I_clean)
dark_current = N(0, σ_dark)
```

#### Combined Noise

```
I_noisy = I_clean + shot_noise + read_noise + dark_current
I_noisy = clamp(I_noisy, min=0)  # Enforce physical constraint
```

## Physical Interpretation

### Typical Parameter Values

| Detector Type | photon_scale | read_noise_fraction | dark_current_fraction |
|---------------|--------------|---------------------|----------------------|
| Scientific CCD/CMOS | 5000-10000 | 0.001-0.005 | 0.0001-0.001 |
| Consumer Camera | 500-2000 | 0.01-0.05 | 0.002-0.01 |
| Smartphone | 100-500 | 0.02-0.1 | 0.01-0.05 |
| Low-Light Camera | 10000-50000 | 0.0005-0.002 | 0.00001-0.0001 |

### SNR to Component Mapping

Approximate relationship between SNR and photon counts:

```
SNR(dB) ≈ 20 * log10(sqrt(photon_count))
```

| SNR (dB) | Approximate Photon Count | Quality |
|----------|-------------------------|---------|
| 20 | 100 | Low |
| 30 | 1,000 | Moderate |
| 40 | 10,000 | Good |
| 50 | 100,000 | Excellent |
| 60 | 1,000,000 | Near-perfect |

## Notes

- Shot noise follows Poisson statistics but is approximated with Gaussian when photon counts are high (mean >> 1), which is valid for most imaging scenarios
- The existing implementation in Microscope._add_detector_noise() scales to photon counts, adds Poisson-like noise, then scales back
- Read noise is additive Gaussian noise representing detector electronics noise
- Dark current represents thermal electrons accumulated during exposure
- Output is always clamped to non-negative values (physical constraint)
- During eval() mode, noise is still added if add_noise=True (controlled by flag, not by module training state)

## References

[1] Janesick, J. R. (2001). "Scientific Charge-Coupled Devices". SPIE Press. Chapter 4: Noise Sources and Signal-to-Noise Ratio.

[2] Holst, G. C., & Lomheim, T. S. (2007). "CMOS/CCD sensors and camera systems". JCD publishing.

## See Also

- `prism.core.instruments.four_f_base.FourFSystem` : Uses DetectorNoiseModel for noise simulation
- `prism.core.instruments.microscope.Microscope` : Microscope with detector noise
- `prism.core.instruments.camera.Camera` : Camera with sensor noise
- `prism.models.noise` : Alternative noise models (ShotNoise, PoissonNoise)

# prism.core.instruments.four_f_base

Abstract base class for 4f optical system instruments.

This module provides a unified implementation for all instruments based on the four-focal-length (4f) optical system architecture. It consolidates common functionality including forward modeling, k-space propagation, aperture mask generation, and noise modeling.

## Overview

The 4f system is the canonical configuration for Fourier optics, consisting of two lenses separated by the sum of their focal lengths, with pupils placed at the common back focal plane. This architecture is shared by microscopes, telescopes, and cameras when operating in their simplified (paraxial, in-focus) imaging modes.

**Key Benefits**:
- Single source of truth for 4f forward model implementation
- Reduces code duplication by ~500 lines across instruments
- Unified interface for PRISM synthetic aperture reconstruction
- Easier to maintain and extend

**Physical Model**:
```
I(x,y) = |IFFT{ P_det(u,v) · P_illum(u,v) · FFT{E_object(x,y)} }|²
```

where:
- `E_object`: Complex field at object plane
- `FFT/IFFT`: Fourier transforms
- `P_illum`: Illumination pupil function
- `P_det`: Detection pupil function
- `I`: Detected intensity

## Classes

### FourFSystem

```python
FourFSystem(
    config: InstrumentConfig,
    padding_factor: float = 2.0,
    aperture_cutoff_type: str = 'na',
    medium_index: float = 1.0,
    noise_model: Optional[DetectorNoiseModel] = None
)
```

Abstract base class for 4f optical system instruments.

This class provides unified implementation for all instruments based on the four-focal-length (4f) optical system. It consolidates common functionality that was previously duplicated across Microscope, Telescope, and PRISM classes.

**Provided Functionality**:
- Forward model: Unified 4f propagation with padding to prevent aliasing
- K-space propagation: FFT-based object ↔ pupil plane transformations
- Aperture mask generation: Unified interface for all mask types
- Input validation: Automatic detection of intensity/amplitude/complex inputs
- Noise modeling: Optional detector noise (shot, read, dark current)

**Subclass Requirements**:

Subclasses must implement:
- `_create_pupils()`: Return illumination and detection pupil functions
- `resolution_limit`: Property returning theoretical resolution limit

Subclasses may override:
- `compute_psf()`: Custom PSF computation
- `get_info()`: Additional instrument-specific info
- `_create_grid()`: Custom grid configuration
- `_select_propagator()`: Custom propagator selection

#### Parameters

- **config** : `InstrumentConfig`

  Instrument configuration (wavelength, n_pixels, pixel_size, etc.)

- **padding_factor** : `float`, default=2.0

  FFT padding factor to prevent wraparound artifacts. Must be >= 1.0.
  Default 2.0 provides good anti-aliasing. Set to 1.0 for no padding.

- **aperture_cutoff_type** : `str`, default='na'

  How to interpret aperture radius specifications:
  - `'na'`: Use numerical aperture (microscopes)
  - `'physical'`: Use physical radius in meters (telescopes)
  - `'pixels'`: Use radius in pixel units (generic)

- **medium_index** : `float`, default=1.0

  Refractive index of medium (1.0=air, 1.33=water, 1.515=oil)

- **noise_model** : `DetectorNoiseModel`, optional

  Optional detector noise model. If None, no noise is added.

#### Attributes

- **padding_factor** : `float`

  FFT padding factor used by forward model

- **medium_index** : `float`

  Refractive index of the medium

- **_forward_model** : `FourFForwardModel`

  Core 4f forward model (pad → FFT → pupils → IFFT → crop)

- **_aperture_generator** : `ApertureMaskGenerator`

  Unified aperture mask generator

- **_noise_model** : `DetectorNoiseModel` or `None`

  Optional detector noise model

- **_default_aperture_radius** : `float`

  Default sub-aperture radius for generate_aperture_mask

#### Methods

##### `__init__`

Initialize FourFSystem with configuration and components.

```python
__init__(
    config: InstrumentConfig,
    padding_factor: float = 2.0,
    aperture_cutoff_type: str = 'na',
    medium_index: float = 1.0,
    noise_model: Optional[DetectorNoiseModel] = None
) -> None
```

**Parameters**:
- **config** : Instrument configuration parameters
- **padding_factor** : FFT padding factor (>= 1.0)
- **aperture_cutoff_type** : Aperture specification type ('na', 'physical', 'pixels')
- **medium_index** : Refractive index of medium
- **noise_model** : Optional detector noise model

**Raises**:
- **ValueError** : If padding_factor < 1.0

##### `forward`

Forward propagation through 4f system.

```python
forward(
    field: Tensor,
    illumination_mode: Optional[str] = None,
    illumination_params: Optional[dict] = None,
    add_noise: bool = False,
    input_mode: str = 'auto',
    input_pixel_size: Optional[float] = None,
    coherence_mode: CoherenceMode = CoherenceMode.COHERENT,
    source_intensity: Optional[Tensor] = None,
    n_source_points: int = 100,
    **kwargs: Any
) -> Tensor
```

This is the unified forward method for all 4f-based instruments. Instrument-specific behavior (pupil shapes, illumination modes) is controlled by the `_create_pupils()` abstract method.

**Parameters**:
- **field** : Input field at object plane. Shape: (H, W), (C, H, W), or (B, C, H, W)
- **illumination_mode** : Type of illumination. Interpretation is instrument-specific. Common modes: 'brightfield', 'darkfield', 'phase', 'dic', 'custom'
- **illumination_params** : Parameters for illumination mode (NA values, phase shifts, etc.)
- **add_noise** : Whether to add detector noise (requires noise_model to be set)
- **input_mode** : How to interpret input values:
  - `'intensity'`: Field is I = |E|², converted via sqrt(I)
  - `'amplitude'`: Field is |E|, values >= 0
  - `'complex'`: Field is already complex E = |E|*exp(i*phi)
  - `'auto'`: Auto-detect from dtype and values
- **input_pixel_size** : Physical size of input pixels (meters) for FOV validation
- **coherence_mode** : `CoherenceMode`, default=`CoherenceMode.COHERENT`
  Illumination coherence mode:
  - `COHERENT`: Standard coherent amplitude transfer (laser illumination)
  - `INCOHERENT`: OTF-based propagation for self-luminous objects (fluorescence). Uses only detection pupil; illumination pupil is ignored.
  - `PARTIALLY_COHERENT`: Extended source integration for LED/extended illumination. Requires `source_intensity` parameter.
- **source_intensity** : `Tensor`, optional
  Source intensity distribution for `PARTIALLY_COHERENT` mode. Required when `coherence_mode=CoherenceMode.PARTIALLY_COHERENT`. Shape should match field spatial dimensions.
- **n_source_points** : `int`, default=100
  Number of source points for `PARTIALLY_COHERENT` sampling. Higher values give more accurate results but slower computation.
- **kwargs** : Additional instrument-specific parameters (passed to _create_pupils)

**Returns**:
- **Tensor** : Output intensity at detector. Shape matches input.

**Notes**:

The forward model depends on `coherence_mode`:

**COHERENT** (default):
```
I(x,y) = |IFFT{ P_det · P_illum · FFT{E_object} }|²
```
Standard amplitude transfer through 4f system. Used for laser illumination.

**INCOHERENT** (fluorescence):
```
I_out = IFFT{ OTF · FFT{I_in} }
where OTF = Autocorr(P_det) and I_in = |field|²
```
Note: `P_illum` is IGNORED because incoherent imaging models self-luminous objects (fluorescence emission) where only the detection optics matter.

**PARTIALLY_COHERENT** (extended source):
```
I = Σ_i w_i · |coherent_propagate(E, phase_i)|²
```
Integrates over source intensity distribution. Each source point contributes a coherent image with a different illumination tilt; the total is their incoherent (intensity) sum.

**Illumination Mode Compatibility**:

| illumination_mode | INCOHERENT | Note |
|-------------------|------------|------|
| brightfield | ✅ Allowed | P_det is circular, works for emission |
| darkfield | ⚠️ Warning | Darkfield requires coherent scattering |
| phase | ⚠️ Warning | Phase info lost in incoherent |
| DIC | ⚠️ Warning | Interference requires coherence |

**Examples**:

Basic forward pass (coherent, default):
```python
>>> field = torch.randn(512, 512)
>>> intensity = instrument.forward(field)
```

With illumination mode:
```python
>>> intensity = instrument.forward(
...     field,
...     illumination_mode='darkfield',
...     illumination_params={'annular_ratio': 0.8}
... )
```

Fluorescence imaging (incoherent):
```python
>>> from prism.core.propagators import CoherenceMode
>>> intensity = instrument.forward(
...     field,
...     coherence_mode=CoherenceMode.INCOHERENT,
... )
```

LED brightfield (partially coherent):
```python
>>> # Create Gaussian extended source
>>> source = torch.exp(-((x**2 + y**2) / (2 * sigma**2)))
>>> intensity = instrument.forward(
...     field,
...     coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
...     source_intensity=source,
...     n_source_points=200,
... )
```

With noise:
```python
>>> instrument._noise_model = DetectorNoiseModel(snr_db=40.0)
>>> noisy_intensity = instrument.forward(field, add_noise=True)
```

##### `propagate_to_kspace`

Propagate field to k-space (Fourier domain).

```python
propagate_to_kspace(field: Tensor) -> Tensor
```

Performs FFT from object plane to pupil plane (back focal plane of first lens). The output is fftshift'd so DC is at the center.

**Parameters**:
- **field** : Complex field at object plane. Shape: (H, W) or (B, C, H, W)

**Returns**:
- **Tensor** : Complex k-space field (centered, DC at N//2)

**Notes**:

This is equivalent to the first two steps of the 4f model:
1. FFT to Fourier plane
2. fftshift to center DC

The coordinate system has DC at [N//2, N//2], matching the convention used by `generate_aperture_mask()`.

**Examples**:
```python
>>> field_obj = torch.randn(512, 512, dtype=torch.complex64)
>>> field_kspace = instrument.propagate_to_kspace(field_obj)
>>> print(field_kspace.shape)  # (512, 512)
```

##### `propagate_to_spatial`

Propagate k-space field back to spatial domain and compute intensity.

```python
propagate_to_spatial(field_kspace: Tensor) -> Tensor
```

Performs inverse FFT from pupil plane back to image plane, then computes intensity. This is the final step of the 4f model.

**Parameters**:
- **field_kspace** : Complex k-space field (centered, DC at N//2)

**Returns**:
- **Tensor** : Real-valued intensity at detector plane

**Notes**:

This implements the final steps of the 4f model:
1. ifftshift to prepare for IFFT
2. IFFT to image plane
3. Compute intensity |E|²

The input is expected to be fftshift'd (DC at center), matching the output of `propagate_to_kspace()`.

**Examples**:
```python
>>> field_kspace = instrument.propagate_to_kspace(field_obj)
>>> # Apply pupil mask
>>> field_kspace_masked = field_kspace * pupil_mask
>>> intensity = instrument.propagate_to_spatial(field_kspace_masked)
```

##### `generate_aperture_mask`

Generate circular aperture mask at specified position.

```python
generate_aperture_mask(
    center: Optional[List[float]] = None,
    radius: Optional[float] = None
) -> Tensor
```

Creates a circular aperture in the pupil plane (k-space) at the specified center position. This enables PRISM synthetic aperture construction by sampling different regions of k-space.

**Parameters**:
- **center** : Center position [y, x] in pixels from DC (0,0). Defaults to [0.0, 0.0] (centered on DC).
- **radius** : Aperture radius in units specified by aperture_cutoff_type. Defaults to _default_aperture_radius if not specified.

**Returns**:
- **Tensor** : Binary mask of shape (n_pixels, n_pixels), dtype float32

**Notes**:

The aperture specification depends on aperture_cutoff_type:
- `'na'`: radius interpreted as numerical aperture
- `'physical'`: radius in meters
- `'pixels'`: radius in pixel units

The coordinate system matches `propagate_to_kspace()` output (DC at center).

**Examples**:
```python
>>> # Centered aperture (conventional imaging)
>>> mask = instrument.generate_aperture_mask([0, 0])
>>>
>>> # Off-center sub-aperture for PRISM
>>> mask = instrument.generate_aperture_mask([10, 5], radius=15)
>>>
>>> # Use with k-space propagation
>>> field_kspace = instrument.propagate_to_kspace(field)
>>> masked_kspace = field_kspace * mask
>>> intensity = instrument.propagate_to_spatial(masked_kspace)
```

##### `compute_psf`

Compute point spread function.

```python
compute_psf(**kwargs: Any) -> Tensor
```

Default implementation creates a delta function input and propagates through the system. Subclasses may override for specialized PSF computation (e.g., 3D PSF stacks for microscopes).

**Parameters**:
- **kwargs** : Additional parameters passed to forward() (e.g., illumination_mode)

**Returns**:
- **Tensor** : PSF tensor (2D or 3D depending on instrument), normalized to max=1

**Examples**:
```python
>>> psf = instrument.compute_psf()
>>> print(psf.shape)  # (512, 512)
>>>
>>> # With illumination mode
>>> psf_df = instrument.compute_psf(illumination_mode='darkfield')
```

##### `get_info`

Get instrument information summary.

```python
get_info() -> dict
```

**Returns**:
- **dict** : Dictionary with instrument parameters and characteristics

**Notes**:

Subclasses should override to add instrument-specific information, calling `super().get_info()` first to get base information.

**Examples**:
```python
>>> info = instrument.get_info()
>>> print(info['resolution_limit'])
>>> print(info['wavelength'])
```

##### `_create_pupils` (Abstract)

Create illumination and detection pupil functions.

```python
@abstractmethod
_create_pupils(
    illumination_mode: Optional[str] = None,
    illumination_params: Optional[dict] = None
) -> tuple[Optional[Tensor], Optional[Tensor]]
```

This abstract method must be implemented by subclasses to define the instrument-specific pupil functions. Pupils can vary based on illumination mode (brightfield, darkfield, phase, DIC, etc.).

**Parameters**:
- **illumination_mode** : Type of illumination ('brightfield', 'darkfield', 'phase', 'dic', 'custom', etc.). If None, use instrument default.
- **illumination_params** : Additional parameters for illumination mode (e.g., NA values, annular ratios, phase shifts, custom pupils).

**Returns**:
- **tuple[Tensor or None, Tensor or None]** : (illumination_pupil, detection_pupil) in k-space. Either can be None for identity (all-pass, no filtering). Pupils should be complex-valued tensors of shape (n_pixels, n_pixels).

**Notes**:

Pupils are applied in the Fourier plane (back focal plane of first lens). They should be provided as complex-valued tensors that include both amplitude and phase modulation.

**Examples**:

Microscope implementation:
```python
>>> def _create_pupils(self, illumination_mode=None,
...                   illumination_params=None):
...     params = illumination_params or {}
...     if illumination_mode == 'brightfield':
...         illum_na = params.get('illumination_na', 0.8 * self.na)
...         illum = self._aperture_generator_lazy.circular(na=illum_na)
...         detect = self._aperture_generator_lazy.circular(na=self.na)
...     elif illumination_mode == 'darkfield':
...         annular_ratio = params.get('annular_ratio', 0.8)
...         illum = self._aperture_generator_lazy.annular(
...             inner_na=annular_ratio*self.na, outer_na=self.na)
...         detect = self._aperture_generator_lazy.circular(na=self.na)
...     else:
...         illum = self._aperture_generator_lazy.circular(na=0.8*self.na)
...         detect = self._aperture_generator_lazy.circular(na=self.na)
...     return illum, detect
```

##### `resolution_limit` (Abstract Property)

Theoretical resolution limit.

```python
@property
@abstractmethod
resolution_limit() -> float
```

**Returns**:
- **float** : Resolution limit in meters (for microscopes/cameras) or radians (for telescopes). The units depend on the instrument type.

**Notes**:
- For microscopes: Abbe limit = 0.61 * λ / NA
- For telescopes: Rayleigh criterion = 1.22 * λ / D

## Usage Examples

### Subclassing FourFSystem

Here's a simplified example of creating a custom instrument based on FourFSystem:

```python
from prism.core.instruments.four_f_base import FourFSystem
from prism.core.instruments.base import InstrumentConfig

class SimpleMicroscope(FourFSystem):
    def __init__(self, config, na=1.4):
        self.na = na
        super().__init__(
            config,
            padding_factor=2.0,
            aperture_cutoff_type='na',
            medium_index=1.0
        )

    def _create_pupils(self, illumination_mode=None,
                      illumination_params=None):
        # Create NA-limited circular pupils
        illum_pupil = self._aperture_generator_lazy.circular(na=self.na*0.8)
        detect_pupil = self._aperture_generator_lazy.circular(na=self.na)
        return illum_pupil, detect_pupil

    @property
    def resolution_limit(self):
        return 0.61 * self.config.wavelength / self.na
```

### Using a FourFSystem Instrument

```python
import torch
from prism.core.instruments import MicroscopeConfig, create_instrument

# Create instrument
config = MicroscopeConfig(
    wavelength=532e-9,
    numerical_aperture=1.4,
    n_pixels=512
)
microscope = create_instrument(config)

# Forward imaging
field = torch.randn(512, 512, dtype=torch.complex64)
intensity = microscope.forward(field)

# Compute PSF
psf = microscope.compute_psf()

# PRISM synthetic aperture
field_kspace = microscope.propagate_to_kspace(field)
sub_aperture = microscope.generate_aperture_mask(center=[10, 5], radius=15)
masked_kspace = field_kspace * sub_aperture
sub_aperture_image = microscope.propagate_to_spatial(masked_kspace)
```

## Coherence Modes

FourFSystem supports three illumination coherence modes via the `coherence_mode` parameter:

### CoherenceMode Enum

```python
from prism.core.propagators import CoherenceMode

CoherenceMode.COHERENT           # Default: laser illumination
CoherenceMode.INCOHERENT         # Fluorescence, self-luminous objects
CoherenceMode.PARTIALLY_COHERENT # LED, extended source illumination
```

### Mode Comparison

| Mode | Model | Input | Physical Scenario |
|------|-------|-------|-------------------|
| `COHERENT` | Amplitude transfer | Complex E | Laser illumination |
| `INCOHERENT` | OTF convolution | Intensity I | Fluorescence (self-luminous) |
| `PARTIALLY_COHERENT` | Source integration | Complex E | LED brightfield, extended sources |

### When to Use Each Mode

| Use Case | Recommended Mode |
|----------|-----------------|
| Laser microscopy | `COHERENT` |
| Holographic imaging | `COHERENT` |
| Fluorescence microscopy | `INCOHERENT` |
| Stellar/astronomical imaging | `INCOHERENT` |
| LED brightfield microscopy | `PARTIALLY_COHERENT` |
| Köhler illumination | `PARTIALLY_COHERENT` |

### Example Usage

```python
from prism.core.propagators import CoherenceMode

# Coherent (default)
intensity = microscope.forward(field)

# Incoherent (fluorescence)
intensity = microscope.forward(
    emission_pattern,
    coherence_mode=CoherenceMode.INCOHERENT,
)

# Partially coherent (LED)
source = create_gaussian_source(n_pixels=512, sigma=0.1)
intensity = microscope.forward(
    field,
    coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
    source_intensity=source,
    n_source_points=200,
)
```

For detailed examples and physics background, see the [Coherence Modes Guide](../examples/coherence_modes.md).

## See Also

- `prism.core.instruments.base.Instrument` : Base class for all optical instruments
- `prism.core.optics.four_f_forward.FourFForwardModel` : Core 4f forward model implementation
- `prism.core.optics.aperture_masks.ApertureMaskGenerator` : Unified aperture mask generation
- `prism.core.optics.detector_noise.DetectorNoiseModel` : Realistic detector noise simulation
- `prism.core.propagators.CoherenceMode` : Coherence mode enum
- `prism.core.propagators.OTFPropagator` : OTF-based incoherent propagation
- `prism.core.instruments.microscope.Microscope` : Microscope implementation using FourFSystem
- `prism.core.instruments.telescope.Telescope` : Telescope implementation using FourFSystem
- `prism.core.instruments.camera.Camera` : Camera implementation using FourFSystem
- [Coherence Modes Guide](../examples/coherence_modes.md) : Detailed examples and physics

## Migration Notes

### From Pre-FourFSystem Instruments

The FourFSystem consolidation is **backward compatible**. Existing code using Microscope, Telescope, or Camera will continue to work without changes. However, the following methods are now inherited from FourFSystem:

- `propagate_to_kspace()`
- `propagate_to_spatial()`
- `generate_aperture_mask()`
- `forward()` (core implementation)

**No action required** unless you were overriding these methods in custom instrument subclasses.

### Creating New Instruments

When creating new 4f-based instruments:

**Before** (pre-consolidation):
```python
class MyInstrument(Instrument):
    def forward(self, field):
        # Implement full FFT → pupil → IFFT chain
        field_k = torch.fft.fftshift(torch.fft.fft2(field))
        field_k = field_k * self.pupil
        field_out = torch.fft.ifft2(torch.fft.ifftshift(field_k))
        return torch.abs(field_out) ** 2

    def propagate_to_kspace(self, field):
        # Implement FFT
        ...

    # ... more boilerplate
```

**After** (with FourFSystem):
```python
class MyInstrument(FourFSystem):
    def _create_pupils(self, illumination_mode=None,
                      illumination_params=None):
        # Just define pupils - FourFSystem handles the rest
        pupil = self._aperture_generator_lazy.circular(na=self.na)
        return None, pupil

    @property
    def resolution_limit(self):
        return 0.61 * self.config.wavelength / self.na
```

The new approach reduces code by ~500 lines and provides all standard functionality automatically.

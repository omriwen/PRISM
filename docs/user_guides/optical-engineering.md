# Optical Engineering Guide for PRISM

## Table of Contents
1. [Overview](#overview)
2. [Fundamental Optical Physics](#fundamental-optical-physics)
3. [Telescope Systems](#telescope-systems)
4. [Microscope Systems](#microscope-systems)
5. [Camera Systems](#camera-systems)
6. [Propagation Models](#propagation-models)
7. [Implementation in PRISM](#implementation-in-prism)
8. [Engineering Considerations](#engineering-considerations)
9. [Mathematical Relations](#mathematical-relations)
10. [Parameter Tables](#parameter-tables)

## Overview

This document provides comprehensive engineering guidance for implementing multi-scale optical systems in PRISM. The framework supports three primary instrument types:
- **Telescopes**: Far-field imaging of astronomical objects
- **Microscopes**: Near-field imaging of microscopic samples
- **Cameras**: General-purpose imaging systems

All systems are unified under a wave optics (Fourier optics) framework that accounts for diffraction, interference, and coherence effects.

## Fundamental Optical Physics

### Wave Optics Framework

All optical propagation in PRISM is based on the scalar wave equation:
$$\nabla^2 U - \frac{1}{c^2}\frac{\partial^2 U}{\partial t^2} = 0$$

For monochromatic light with wavelength $\lambda$, the field becomes:
$$U(x,y,z,t) = u(x,y,z)e^{-i\omega t}$$

where the spatial part satisfies the Helmholtz equation:
$$\nabla^2 u + k^2 u = 0$$
with $k = 2\pi/\lambda$ being the wave number.

### Key Physical Parameters

| Parameter | Symbol | Description | Units |
|-----------|--------|-------------|-------|
| Wavelength | $\lambda$ | Light wavelength | nm, μm |
| Wave number | $k$ | $2\pi/\lambda$ | rad/m |
| Frequency | $\nu$ | $c/\lambda$ | Hz |
| Refractive index | $n$ | Medium property | dimensionless |
| Numerical aperture | NA | $n\sin\theta$ | dimensionless |
| Focal length | $f$ | Lens focal length | mm, m |
| Aperture diameter | $D$ | Physical aperture size | mm, m |
| F-number | $f/\#$ | $f/D$ | dimensionless |

### Resolution Limits

The fundamental resolution limit is determined by diffraction:

**Rayleigh Criterion:**
- Telescopes: $\theta = 1.22\lambda/D$ (angular resolution in radians)
- Microscopes: $\delta = 0.61\lambda/\text{NA}$ (spatial resolution)
- Cameras: Depends on pixel size and optics

**Abbe Limit (Microscopy):**
$$\delta = \frac{\lambda}{2\text{NA}}$$

## Telescope Systems

### Configuration Parameters

```python
telescope_params = {
    'aperture_diameter': 8.2,  # meters (e.g., VLT)
    'focal_length': 120.0,      # meters
    'wavelength': 550e-9,       # meters (green)
    'pixel_size': 13e-6,        # meters (detector)
    'object_distance': float('inf'),  # infinity
    'propagation_mode': 'fraunhofer'
}
```

### Aperture Functions

Telescopes use various aperture configurations:

1. **Circular Aperture:**
   $$P(x,y) = \begin{cases} 1 & \text{if } \sqrt{x^2+y^2} \leq D/2 \\ 0 & \text{otherwise} \end{cases}$$

2. **Hexagonal Aperture (Segmented mirrors):**
   $$P(x,y) = \text{HexMask}(x,y,D)$$

3. **Obscured Aperture (Secondary mirror):**
   $$P(x,y) = \text{Circ}(r/R_{\text{outer}}) - \text{Circ}(r/R_{\text{inner}})$$

### Point Spread Function (PSF)

For telescopes in the far-field (Fraunhofer regime):
$$\text{PSF} = |\mathcal{F}\{P(x,y)\}|^2$$

where $\mathcal{F}$ is the Fourier transform.

**Airy Pattern (Circular aperture):**
$$I(r) = I_0 \left[\frac{2J_1(\pi D r/\lambda f)}{\pi D r/\lambda f}\right]^2$$

### Field of View

$$\text{FOV} = 2\arctan\left(\frac{d_{\text{sensor}}}{2f}\right)$$

where $d_{\text{sensor}}$ is the sensor size.

## Microscope Systems

### Configuration Parameters

```python
microscope_params = {
    'numerical_aperture': 1.4,  # Oil immersion
    'magnification': 100,        # 100x objective
    'tube_lens_focal': 200e-3,  # meters
    'wavelength': 532e-9,        # meters (green laser)
    'medium_index': 1.515,       # Oil
    'pixel_size': 6.5e-6,        # Camera sensor
    'working_distance': 0.13e-3, # meters
    'propagation_mode': 'angular_spectrum'
}
```

### Numerical Aperture and Resolution

The numerical aperture defines the light-gathering ability:
$$\text{NA} = n \sin\theta_{\max}$$

Resolution limits:
- **Lateral resolution:** $r_{xy} = 0.61\lambda/\text{NA}$
- **Axial resolution:** $r_z = 2\lambda n/\text{NA}^2$

### Microscope Pupil Function

$$P(f_x, f_y) = \begin{cases}
1 & \text{if } \lambda\sqrt{f_x^2 + f_y^2} \leq \text{NA} \\
0 & \text{otherwise}
\end{cases}$$

### 4f System (Infinity-Corrected Microscope)

Modern microscopes use a 4f configuration:
1. **Object → Objective:** Creates Fourier transform at back focal plane
2. **Back focal plane:** Spatial frequency filtering
3. **Tube lens → Image:** Inverse Fourier transform

$$U_{\text{image}} = \mathcal{F}^{-1}\{\mathcal{F}\{U_{\text{object}}\} \cdot P(f_x, f_y)\}$$

### Magnification

Total magnification:
$$M = \frac{f_{\text{tube}}}{f_{\text{obj}}} \times M_{\text{eyepiece}}$$

Effective pixel size at object:
$$\Delta x_{\text{obj}} = \frac{\Delta x_{\text{camera}}}{M}$$

### Köhler Illumination

Provides uniform illumination:
- Source imaged at back focal plane of condenser
- Field diaphragm imaged at sample plane
- Condenser NA should match objective NA

### Illumination & Contrast Modes (Brightfield vs Darkfield)

The difference between Brightfield and Darkfield imaging is determined by the **source geometry** relative to the **objective pupil**.

**Brightfield (BF):**
- The background is bright; unscattered light (zero-order) **passes** through the objective
- **Condition:** Illumination angles are **inside** the objective NA
- $\text{NA}_{source} \le \text{NA}_{obj}$

**Darkfield (DF):**
- The background is dark; unscattered light is **blocked**
- Only light scattered by the sample enters the objective
- **Condition:** Illumination angles are **outside** the objective NA
- $\text{NA}_{source} > \text{NA}_{obj}$

### Illumination Source Library

#### A. LED / Standard Illumination (Partial Coherence)

**Model:** Extended source integration (grid of points).

| Configuration | Source Geometry | Constraint |
|---------------|-----------------|------------|
| **Brightfield** | Circular disk grid | $\text{NA}_{cond} \le \text{NA}_{obj}$ |
| **Darkfield** | Annular (ring) grid | $\text{NA}_{obj} < \text{NA}_{inner} < \text{NA}_{cond}$ |

- **BF:** Generate circular grid of points with max angle $\theta_{max} = \sin^{-1}(\text{NA}_{cond})$
- **DF:** Generate annular grid with inner radius just outside $\text{NA}_{obj} + \epsilon$

#### B. Laser Illumination (Coherent)

**Model:** Single point source.

| Configuration | Source Position | Result |
|---------------|-----------------|--------|
| **Brightfield** | Point at $(0, 0)$ (on-axis) | Holographic-like fringes, high contrast |
| **Darkfield** | Point at $(k_x, 0)$ tilted beyond objective | "Oblique illumination darkfield" |

#### C. Solar / Space-Originated Illumination

**Direct Solar (Collimated):**
- **BF:** Single point source or tight $3 \times 3$ grid; acts like coherent brightfield
- **DF:** Not possible unless sun is blocked or sample illuminated from side

**Diffused Solar (Ambient/Cloud):**
- Acts as standard LED source (partial coherence)
- **BF:** Standard simulation with disk grid
- **DF:** Requires physical "patch stop" mask to create ring illumination

### Source Integration Algorithm

The master algorithm for physically accurate microscope simulation:

```
Inputs: Object, NA_obj, NA_cond, Mode ('BF' or 'DF'), Light_Type

1. GENERATE SOURCE GRID:
   - Define frequency coordinates (f_x, f_y)
   - IF Mode == 'BF': Select points where ρ < NA_obj/λ
   - IF Mode == 'DF': Select points where NA_obj/λ < ρ < NA_cond/λ
   - Result: List of angle vectors [(θ_x, θ_y), ...]

2. CALCULATE OBJECTIVE PUPIL:
   - R_pupil = NA_obj / λ
   - P_mask = 1 inside R_pupil, 0 outside

3. INITIALIZE: Accumulated_Intensity = 0

4. SOURCE POINT LOOP: for (θ_x, θ_y) in Source_Grid:

   a. TILT INPUT WAVE:
      U_in = exp(i · 2π · (θ_x·x + θ_y·y))

   b. OBJECT INTERACTION:
      U_sample = U_in · T_object(x, y)

   c. FOURIER TRANSFORM:
      A_spectrum = FFT2(U_sample)

   d. APPLY OBJECTIVE PUPIL:
      A_filtered = A_spectrum · P_mask

   e. IMAGE FORMATION:
      U_image = IFFT2(A_filtered)

   f. INCOHERENT SUM:
      Accumulated_Intensity += |U_image|²

5. RETURN: Accumulated_Intensity
```

### Simulation State Summary

| Mode | Source Geometry | Visual Characteristic | Simulation Check |
|------|-----------------|----------------------|------------------|
| **Brightfield (Coherent)** | Point $(0,0)$ | High contrast, ringing, speckle | `len(Source_Grid) == 1` |
| **Brightfield (Partial)** | Disk radius $< \text{NA}_{obj}$ | Standard bio-image, clear edges | `NA_cond <= NA_obj` |
| **Darkfield (Standard)** | Ring $\text{NA}_{obj} < r < \text{NA}_{cond}$ | Bright edges on black background | `NA_cond > NA_obj` |
| **Solar (Direct)** | Point $(0,0)$ | **[BF]** Extremely coherent, harsh shadows | Single point source |
| **Solar (Diffused)** | Large disk | **[BF]** Soft, low contrast | Large `N` in grid loop |

## Camera Systems

### Configuration Parameters

```python
camera_params = {
    'focal_length': 50e-3,       # meters (50mm lens)
    'f_number': 1.4,              # f/1.4
    'sensor_size': [36e-3, 24e-3],  # Full frame
    'pixel_size': 4.3e-6,         # meters
    'object_distance': 2.0,       # meters
    'wavelength': 550e-9,         # meters
    'propagation_mode': 'auto'    # Automatic selection
}
```

### Lens Equation

Gaussian thin lens equation:
$$\frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}$$

where:
- $f$ = focal length
- $d_o$ = object distance
- $d_i$ = image distance

### Depth of Field

$$\text{DOF} = \frac{2Nc(d_o^2 - f^2)}{f^2}$$

where:
- $N$ = f-number
- $c$ = circle of confusion diameter

### Modulation Transfer Function (MTF)

The MTF describes frequency response:
$$\text{MTF}(f) = |\mathcal{F}\{\text{PSF}\}|$$

Diffraction-limited MTF for circular aperture:
$$\text{MTF}(f) = \frac{2}{\pi}\left[\cos^{-1}\left(\frac{f}{f_c}\right) - \frac{f}{f_c}\sqrt{1-\left(\frac{f}{f_c}\right)^2}\right]$$

where $f_c = 1/(\lambda \cdot f/\#)$ is the cutoff frequency.

## Propagation Models

### Fresnel Number

The Fresnel number determines propagation regime:
$$F = \frac{a^2}{z\lambda}$$

where:
- $a$ = aperture radius
- $z$ = propagation distance

**Regime Selection:**
- $F \gg 1$: Near-field (use Angular Spectrum)
- $F \approx 1$: Fresnel regime
- $F \ll 1$: Fraunhofer (far-field)

### Angular Spectrum Method (ASM)

**Exact propagation for all distances:**

Forward propagation:
$$U(x,y,z) = \mathcal{F}^{-1}\left\{\mathcal{F}\{U(x,y,0)\} \cdot H(f_x,f_y,z)\right\}$$

Transfer function:
$$H(f_x,f_y,z) = \exp\left(ikz\sqrt{1-\lambda^2(f_x^2+f_y^2)}\right)$$

**Implementation considerations:**
- Evanescent wave cutoff: $f_x^2 + f_y^2 > 1/\lambda^2$
- Sampling: $\Delta x < \lambda/(4\text{NA})$ for microscopy

### Fresnel Propagation

**Valid for intermediate distances:**

Fresnel integral:
$$U(x,y,z) = \frac{e^{ikz}}{i\lambda z}\iint U(\xi,\eta,0)\exp\left[\frac{ik}{2z}((x-\xi)^2+(y-\eta)^2)\right]d\xi d\eta$$

**Convolution form:**
$$U(x,y,z) = U(x,y,0) \ast h(x,y,z)$$

where the impulse response is:
$$h(x,y,z) = \frac{e^{ikz}}{i\lambda z}\exp\left[\frac{ik}{2z}(x^2+y^2)\right]$$

### Fraunhofer Propagation

**Valid for far-field:**

$$U(x,y,z) = \frac{e^{ikz}e^{ik(x^2+y^2)/(2z)}}{i\lambda z} \mathcal{F}\{U(\xi,\eta,0)\}\bigg|_{f_x=x/(\lambda z), f_y=y/(\lambda z)}$$

Simplified: The field is proportional to the Fourier transform of the aperture.

## Implementation in PRISM

### Automatic Instrument Detection

```python
def detect_instrument_type(config):
    """Automatically detect instrument type from physical parameters"""

    # Check object distance (astronomy vs terrestrial)
    if config.object_distance > 1e6:  # > 1000 km
        return 'telescope'

    # Check scale (microscopy vs macroscopy)
    elif config.pixel_pitch < 10e-6:  # < 10 μm
        if hasattr(config, 'numerical_aperture'):
            return 'microscope'

    # Default to camera for general imaging
    return 'camera'
```

### Propagator Selection

```python
def select_propagator(instrument_type, fresnel_number, distance):
    """Select optimal propagator based on physical parameters"""

    if instrument_type == 'telescope':
        # Always far-field for astronomy
        return FraunhoferPropagator()

    elif instrument_type == 'microscope':
        # Always use exact method for microscopy
        return AngularSpectrumPropagator()

    else:  # camera
        if fresnel_number < 0.1:
            return FraunhoferPropagator()
        elif fresnel_number > 10:
            return AngularSpectrumPropagator()
        else:
            return FresnelPropagator()
```

### Grid Configuration

```python
def configure_grid(instrument_type, params):
    """Configure computational grid for instrument"""

    if instrument_type == 'telescope':
        # Angular coordinates for astronomy
        grid = Grid(
            n_pixels=params.n_pixels,
            pixel_pitch=params.pixel_size,
            wavelength=params.wavelength,
            coordinate_system='angular'
        )

    elif instrument_type == 'microscope':
        # High resolution spatial grid
        grid = Grid(
            n_pixels=params.n_pixels,
            pixel_pitch=params.pixel_size / params.magnification,
            wavelength=params.wavelength,
            coordinate_system='spatial'
        )
        # Ensure Nyquist sampling
        assert grid.dx < params.wavelength / (4 * params.numerical_aperture)

    else:  # camera
        # Standard spatial grid
        grid = Grid(
            n_pixels=params.n_pixels,
            pixel_pitch=params.pixel_size,
            wavelength=params.wavelength,
            coordinate_system='spatial'
        )

    return grid
```

## Engineering Considerations

### Sampling Requirements

**Nyquist-Shannon Theorem:**
To properly sample the field, we need at least 2 samples per period of the highest spatial frequency.

**Telescope:**
- Angular sampling: $\Delta\theta < \lambda/(2D)$
- Detector sampling: Match to PSF size

**Microscope:**
- Spatial sampling: $\Delta x < \lambda/(4\text{NA})$
- Oversampling factor: typically 2-4×

**Camera:**
- Pixel size vs diffraction spot: $d_{\text{pixel}} < d_{\text{Airy}}/2$
- Where $d_{\text{Airy}} = 2.44\lambda(f/\#)$

### Coherence Considerations

**Spatial Coherence:**
- Coherence length: $l_c = \lambda^2/(2\Delta\lambda)$
- Van Cittert-Zernike theorem for partially coherent sources

**Temporal Coherence:**
- Coherence time: $\tau_c = 1/\Delta\nu$
- Important for interferometric systems

### Aberrations

Primary aberrations (Seidel):
1. **Spherical aberration:** $W = W_{040}r^4$
2. **Coma:** $W = W_{131}r^3\cos\theta$
3. **Astigmatism:** $W = W_{222}r^2\cos^2\theta$
4. **Field curvature:** $W = W_{220}r^2$
5. **Distortion:** $W = W_{311}r^3\cos\theta$

**Zernike Polynomials:**
Used for wavefront description:
$$W(r,\theta) = \sum_{n,m} a_{nm}Z_n^m(r,\theta)$$

### Noise Sources

1. **Shot Noise (Photon noise):**
   - Poisson statistics: $\text{SNR} = \sqrt{N_{\text{photons}}}$

2. **Read Noise:**
   - Electronic noise: typically 1-10 e⁻ RMS

3. **Dark Current:**
   - Thermal electrons: $\sim 0.01$ e⁻/pixel/s at -20°C

4. **Quantization Noise:**
   - ADC discretization: $\sigma = \Delta/\sqrt{12}$

## Mathematical Relations

### Fourier Transform Pairs

| Spatial Domain | Frequency Domain |
|----------------|------------------|
| $\text{rect}(x/a)$ | $a\cdot\text{sinc}(af_x)$ |
| $\text{circ}(r/a)$ | $2\pi a^2\cdot\text{jinc}(2\pi af_r)/f_r$ |
| $\exp(-\pi x^2/\sigma^2)$ | $\sigma\exp(-\pi\sigma^2f_x^2)$ |
| $\delta(x-x_0)$ | $\exp(-2\pi if_xx_0)$ |

### Convolution Theorem

$$\mathcal{F}\{f \ast g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}$$

### Parseval's Theorem (Energy conservation)

$$\iint |U(x,y)|^2 dxdy = \iint |\tilde{U}(f_x,f_y)|^2 df_xdf_y$$

## Parameter Tables

### Typical Wavelengths

| Source | Wavelength (nm) | Color | Application |
|--------|-----------------|-------|-------------|
| HeNe Laser | 632.8 | Red | Alignment, interferometry |
| Nd:YAG (2ω) | 532 | Green | Microscopy, fluorescence |
| LED (Blue) | 450-480 | Blue | Fluorescence excitation |
| LED (White) | 400-700 | White | Brightfield microscopy |
| Na D-line | 589 | Yellow | Spectroscopy |

### Common Objectives (Microscopy)

| Magnification | NA (Air) | NA (Oil) | WD (mm) | Resolution (nm @ 550nm) |
|---------------|----------|----------|---------|-------------------------|
| 10× | 0.25-0.30 | - | 10-15 | 1120-1340 |
| 20× | 0.40-0.50 | - | 2-4 | 670-840 |
| 40× | 0.65-0.75 | 1.30 | 0.5-0.7 | 450-520 |
| 60× | 0.85-0.95 | 1.40 | 0.15-0.3 | 350-390 |
| 100× | 0.90-0.95 | 1.40-1.45 | 0.13-0.15 | 230-370 |

### Telescope Parameters

| Telescope | Diameter (m) | f/# | Plate Scale (″/mm) | Diffraction Limit (″ @ 550nm) |
|-----------|-------------|-----|-------------------|--------------------------------|
| HST | 2.4 | 24 | 3.58 | 0.047 |
| VLT | 8.2 | 13.5 | 1.87 | 0.014 |
| Keck | 10 | 15 | 1.38 | 0.011 |
| JWST | 6.5 | 20.2 | - | 0.017 |
| E-ELT | 39 | 17.7 | 0.32 | 0.003 |

### Camera Sensor Specifications

| Sensor Type | Size (mm) | Pixels | Pixel Size (μm) | Full Well (e⁻) |
|-------------|-----------|---------|-----------------|-----------------|
| 1/2.3" | 6.2×4.6 | 12MP | 1.55 | 4,000 |
| APS-C | 23.6×15.6 | 24MP | 3.9 | 25,000 |
| Full Frame | 36×24 | 45MP | 4.0 | 40,000 |
| Medium Format | 44×33 | 100MP | 4.3 | 50,000 |
| Scientific CCD | 27.6×27.6 | 4k×4k | 6.5 | 100,000 |

## Code Examples

### Telescope Simulation

```python
from prism.core.instruments import Telescope
from prism.config import TelescopeConfig

# Configure VLT-like telescope
config = TelescopeConfig(
    aperture_diameter=8.2,  # meters
    focal_length=120.0,     # meters
    wavelength=550e-9,      # green light
    n_pixels=2048,
    pixel_size=13e-6        # detector pixel
)

telescope = Telescope(config)
psf = telescope.compute_psf()
```

### Microscope Simulation

```python
from prism.core.instruments import Microscope
from prism.config import MicroscopeConfig

# Configure high-NA oil immersion microscope
config = MicroscopeConfig(
    numerical_aperture=1.4,
    magnification=100,
    wavelength=532e-9,
    medium_index=1.515,
    n_pixels=1024,
    camera_pixel_size=6.5e-6
)

microscope = Microscope(config)
psf = microscope.compute_psf_3d()  # 3D PSF
```

### Camera Simulation

```python
from prism.core.instruments import Camera
from prism.config import CameraConfig

# Configure DSLR camera
config = CameraConfig(
    focal_length=50e-3,     # 50mm lens
    f_number=1.4,
    sensor_size=(36e-3, 24e-3),  # Full frame
    pixel_size=4.0e-6,
    object_distance=2.0     # 2 meters
)

camera = Camera(config)
image = camera.capture(scene)
```

## Validation Tests

### Resolution Validation

```python
def test_microscope_resolution():
    """Verify microscope resolution matches theory"""
    config = MicroscopeConfig(
        numerical_aperture=1.4,
        wavelength=550e-9
    )

    theoretical_resolution = 0.61 * config.wavelength / config.numerical_aperture
    measured_resolution = measure_resolution(microscope.psf)

    assert abs(measured_resolution - theoretical_resolution) < 0.01e-9
```

### Energy Conservation

```python
def test_energy_conservation():
    """Verify Parseval's theorem holds"""
    field = create_test_field()
    propagated = propagator.propagate(field, distance=1e-3)

    input_energy = np.sum(np.abs(field)**2)
    output_energy = np.sum(np.abs(propagated)**2)

    assert abs(input_energy - output_energy) / input_energy < 1e-6
```

## References

1. Goodman, J.W. (2005). *Introduction to Fourier Optics* (3rd ed.)
2. Born, M. & Wolf, E. (1999). *Principles of Optics* (7th ed.)
3. Mertz, J. (2019). *Introduction to Optical Microscopy* (2nd ed.)
4. Schroeder, D.J. (2000). *Astronomical Optics* (2nd ed.)
5. Saleh, B.E.A. & Teich, M.C. (2019). *Fundamentals of Photonics* (3rd ed.)

## Appendix: Unit Conversions

| Quantity | SI Unit | Common Units | Conversion |
|----------|---------|--------------|------------|
| Length | m | mm, μm, nm | 1m = 10³mm = 10⁶μm = 10⁹nm |
| Angle | rad | degrees, arcmin, arcsec | 1° = π/180 rad = 60' = 3600" |
| Frequency | Hz | THz | 1THz = 10¹²Hz |
| Energy | J | eV | 1eV = 1.602×10⁻¹⁹J |
| Power | W | mW, μW | 1W = 10³mW = 10⁶μW |

---

*This document serves as the authoritative reference for optical engineering in PRISM. All implementations should adhere to these physical principles and mathematical relations.*

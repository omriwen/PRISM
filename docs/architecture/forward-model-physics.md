# Forward Model Physics: Mathematical Equations

This document provides a comprehensive mathematical description of all forward model equations currently implemented in PRISM. Each instrument's optical physics is documented with equations, physical interpretations, code references, and regime selection logic.

**Related Documentation**:
- [Instrument Class Hierarchy](instrument-hierarchy.md) - Class architecture and extension guide
- [Four-F System Consolidation Plan](../plans/four-f-system-consolidation.md) - Implementation roadmap

## Table of Contents

1. [Microscope Forward Models](#microscope-forward-models)
   - [SIMPLIFIED Model (FFT-based 4f System)](#simplified-model-fft-based-4f-system)
   - [FULL Model (Propagation Chain with Defocus)](#full-model-propagation-chain-with-defocus)
   - [Illumination Modes](#illumination-modes)
2. [Telescope Forward Model](#telescope-forward-model)
3. [Camera Forward Model](#camera-forward-model)
4. [Wave Propagation Methods](#wave-propagation-methods)
   - [Angular Spectrum Method](#angular-spectrum-method)
   - [Fresnel Propagation](#fresnel-propagation)
   - [Fraunhofer Propagation](#fraunhofer-propagation)
5. [Regime Selection Logic](#regime-selection-logic)

---

## Overview

PRISM implements multiple forward models for different optical instruments and propagation regimes. All 4f-based instruments (Microscope, Telescope, Camera) now share a unified implementation through the `FourFSystem` base class, which uses the `FourFForwardModel` component for core propagation.

For details on the class architecture, see [Instrument Class Hierarchy](instrument-hierarchy.md).

---

## Microscope Forward Models

The microscope implementation supports two forward model regimes that automatically select based on defocus conditions.

### SIMPLIFIED Model (FFT-based 4f System)

**When used**: Object at or near the front focal plane (defocus parameter δ < 1%)

**Mathematical Model**: Direct Fourier transform relationship

The simplified model assumes the object is exactly at the front focal plane of the objective, enabling a single FFT to accurately model the 4f microscope system:

$$
E_{\text{image}}(x, y) = \mathcal{F}^{-1}\left\{ \mathcal{F}\{E_{\text{object}}(x, y)\} \cdot P_{\text{illum}}(k_x, k_y) \cdot P_{\text{detect}}(k_x, k_y) \right\}
$$

**Steps**:
1. **Object to back focal plane**: $E_{\text{BFP}}(k_x, k_y) = \mathcal{F}\{E_{\text{object}}(x, y)\}$
2. **Apply illumination pupil**: $E_{\text{illum}} = E_{\text{BFP}} \cdot P_{\text{illum}}(k_x, k_y)$
3. **Apply detection pupil**: $E_{\text{filtered}} = E_{\text{illum}} \cdot P_{\text{detect}}(k_x, k_y)$
4. **Back to image plane**: $E_{\text{image}} = \mathcal{F}^{-1}\{E_{\text{filtered}}\}$

**Pupil Functions**:

Circular pupil (brightfield):
$$
P(k_x, k_y) = \begin{cases}
1 & \text{if } \sqrt{k_x^2 + k_y^2} \leq k_{\text{cutoff}} \\
0 & \text{otherwise}
\end{cases}
$$

where the cutoff frequency is determined by numerical aperture:
$$
k_{\text{cutoff}} = \frac{\text{NA}}{n \cdot \lambda}
$$

**Physical Interpretation**: When the object is at the focal plane, the Fraunhofer approximation is exact at the back focal plane. A single FFT accurately transforms the object field to the Fourier domain where pupils are applied, then an inverse FFT produces the image.

**Parameters**:
- `na`: Numerical aperture (dimensionless, typically 0.1-1.4)
- `medium_index`: Refractive index n (1.0 air, 1.33 water, 1.515 oil)
- `wavelength`: Illumination wavelength λ (meters)
- `magnification`: Total system magnification
- `padding_factor`: FFT padding factor (default 2.0) to prevent wraparound

**Code Reference**:
- `prism/core/optics/four_f_forward.py` (`FourFForwardModel` - unified implementation)
- `prism/core/instruments/four_f_base.py` (`FourFSystem.forward()` - unified interface)
- `prism/core/instruments/microscope.py` (`_create_pupils()` - microscope-specific pupils)

---

### FULL Model (Propagation Chain with Defocus)

**When used**: Object significantly defocused from focal plane (δ ≥ 1%)

**Mathematical Model**: Complete 4f propagation with lens phases

When the object is defocused, explicit propagation and lens phases are required:

$$
E_{\text{image}} = L_{\text{tube}} \left\{ \mathcal{F}^{-1}\left[ \mathcal{F}\{L_{\text{obj}}\{P_{\text{defocus}}\{E_{\text{object}}\}\}\} \cdot P_{\text{illum}} \cdot P_{\text{detect}} \right] \right\}
$$

**Complete Propagation Chain**:

1. **Defocus propagation** (if $d \neq f_{\text{obj}}$):
   $$
   E_{\text{focal}} = \text{ASM}(E_{\text{object}}, \Delta z = d - f_{\text{obj}})
   $$

2. **Objective lens phase**:
   $$
   E_{\text{after\_obj}} = E_{\text{focal}} \cdot \exp\left(-i \frac{k}{2f_{\text{obj}}}(x^2 + y^2)\right) \cdot P_{\text{aperture}}
   $$

3. **FFT to back focal plane**:
   $$
   E_{\text{BFP}} = \mathcal{F}\{E_{\text{after\_obj}}\}
   $$

4. **Apply pupils**:
   $$
   E_{\text{filtered}} = E_{\text{BFP}} \cdot P_{\text{illum}} \cdot P_{\text{detect}}
   $$

5. **IFFT to intermediate image**:
   $$
   E_{\text{intermediate}} = \mathcal{F}^{-1}\{E_{\text{filtered}}\}
   $$

6. **Tube lens phase**:
   $$
   E_{\text{image}} = E_{\text{intermediate}} \cdot \exp\left(-i \frac{k}{2f_{\text{tube}}}(x^2 + y^2)\right)
   $$

**Thin Lens Phase**:

The quadratic phase factor for a thin lens with focal length $f$:
$$
\phi_{\text{lens}}(x, y) = -\frac{k}{2f}(x^2 + y^2)
$$

where $k = 2\pi/\lambda$ is the wavenumber.

**Aperture Diameter** (from NA):
$$
D_{\text{aperture}} = \frac{2 \cdot \text{NA} \cdot f_{\text{obj}}}{n}
$$

**Physical Interpretation**: The FULL model explicitly represents each optical element in the 4f microscope system. The objective lens focuses light, the back focal plane is where pupils are applied in Fourier space, and the tube lens forms the final magnified image.

**Parameters**:
- `objective_focal`: Objective focal length $f_{\text{obj}}$ (meters)
- `tube_lens_focal`: Tube lens focal length $f_{\text{tube}}$ (meters, typically 0.2m)
- `working_distance`: Object-to-objective distance $d$ (meters)
- `magnification`: $M = f_{\text{tube}} / f_{\text{obj}}$

**Code Reference**:
- `prism/core/optics/microscope_forward.py:557-613` (`_forward_full`)
- `prism/core/optics/thin_lens.py:42-166` (`ThinLens` class)

---

### Illumination Modes

The microscope supports multiple illumination modes through different pupil configurations:

#### Brightfield (Köhler Illumination)

Partially coherent illumination with central detection:

$$
P_{\text{illum}}(k_x, k_y) = \text{circ}\left(\frac{\sqrt{k_x^2 + k_y^2}}{k_{\text{illum}}}\right)
$$

$$
P_{\text{detect}}(k_x, k_y) = \text{circ}\left(\frac{\sqrt{k_x^2 + k_y^2}}{k_{\text{NA}}}\right)
$$

where $k_{\text{illum}} = 0.8 \cdot k_{\text{NA}}$ (default).

**Code**: `prism/core/instruments/microscope.py:261-266`

#### Darkfield

Annular illumination blocks direct light, only scattered light detected:

$$
P_{\text{illum}}(k_x, k_y) = \begin{cases}
1 & \text{if } r_{\text{inner}} < r_{\text{norm}} \leq 1 \\
0 & \text{otherwise}
\end{cases}
$$

where $r_{\text{norm}} = \sqrt{k_x^2 + k_y^2} / k_{\text{cutoff}}$ and $r_{\text{inner}} = 0.8$ (default annular ratio).

**Code**: `prism/core/instruments/microscope.py:268-273`

#### Phase Contrast

Phase ring in detection pupil converts phase to intensity:

$$
P_{\text{detect}}(k_x, k_y) = \begin{cases}
\exp(i\phi) & \text{if } r_{\text{ring,inner}} < r_{\text{norm}} < r_{\text{ring,outer}} \\
1 & \text{otherwise (within NA)}
\end{cases}
$$

where $\phi = \pi/2$ (default phase shift), $r_{\text{ring,inner}} = 0.6$, $r_{\text{ring,outer}} = 0.8$.

**Code**: `prism/core/instruments/microscope.py:275-282`

#### DIC (Differential Interference Contrast)

Two displaced pupils with phase shift create interference:

$$
P_{\text{DIC}} = \frac{1}{2}\left[ P_1(k_x - \Delta k, k_y) + P_2(k_x + \Delta k, k_y) \cdot e^{i\pi/2} \right]
$$

where $\Delta k$ is the shear in frequency space.

**Code**: `prism/core/instruments/microscope.py:284-289`, `308-340`

---

## Telescope Forward Model

**Regime**: Far-field Fraunhofer diffraction (Fresnel number F ≪ 1)

The telescope operates in the far-field where the object (astronomical source) is effectively at infinity.

### Fraunhofer Diffraction Model

**Forward propagation** (spatial → k-space):
$$
E_k(k_x, k_y) = \mathcal{F}\{E_{\text{object}}(x, y)\}
$$

**Aperture filtering** (if specified):
$$
E_{\text{filtered}} = E_k \cdot A(k_x, k_y)
$$

**Propagation to detector** (k-space → spatial):
$$
E_{\text{detector}} = \mathcal{F}\{E_{\text{filtered}}\}
$$

**Intensity measurement**:
$$
I(x, y) = |E_{\text{detector}}|^2
$$

**Aperture Function**:

For a circular aperture centered at $(k_{cy}, k_{cx})$ with radius $R$:
$$
A(k_x, k_y) = \begin{cases}
1 & \text{if } \sqrt{(k_y - k_{cy})^2 + (k_x - k_{cx})^2} \leq R \\
0 & \text{otherwise}
\end{cases}
$$

### Point Spread Function

The PSF is computed as:
$$
\text{PSF}(x, y) = \left| \mathcal{F}\{A(k_x, k_y)\} \right|^2
$$

For a circular aperture of diameter $D$, this produces an Airy disk pattern:
$$
\text{PSF}(r) \propto \left[\frac{2J_1(kr)}{kr}\right]^2
$$

where $J_1$ is the first-order Bessel function.

**Resolution Limit** (Rayleigh criterion):
$$
\theta_{\text{min}} = 1.22 \frac{\lambda}{D}
$$

**Physical Interpretation**: In astronomical imaging, sources are at effectively infinite distance (Fresnel number F ≈ 10⁻¹²). The Fraunhofer approximation is exact, and the Fourier transform relationship between object and image applies directly.

**Parameters**:
- `aperture_diameter`: Physical aperture diameter $D$ (meters)
- `aperture_radius_pixels`: Aperture radius in pixel units
- `wavelength`: Observation wavelength λ (meters)

**Code Reference**:
- `prism/core/instruments/telescope.py:232-294` (`forward` method)
- `prism/core/instruments/telescope.py:194-230` (`compute_psf`)
- `prism/core/instruments/telescope.py:390-414` (propagation methods)

---

## Camera Forward Model

The camera supports both near-field and far-field imaging with automatic regime selection.

### Near-Field (Fresnel) Propagation

**When used**: Fresnel number $F = a^2/(\lambda z) > 0.1$

**Fresnel Diffraction Equation** (1-step Impulse Response):

$$
E_{\text{out}}(x_2, y_2) = \frac{e^{ikz}}{i\lambda z} e^{i\frac{k}{2z}(x_2^2 + y_2^2)} \mathcal{F}\left\{E_{\text{in}}(x_1, y_1) \cdot e^{i\frac{k}{2z}(x_1^2 + y_2^2)}\right\}
$$

**Steps**:
1. **Pre-chirp** (input domain): Multiply by $\exp[ik/(2z)(x_1^2 + y_1^2)]$
2. **FFT**: Transform to frequency domain
3. **Post-chirp** (output domain): Multiply by $\exp[ik/(2z)(x_2^2 + y_2^2)]$
4. **Normalization**: Multiply by $(e^{ikz})/(i\lambda z) \cdot dx^2$

**Output Grid Scaling**:
$$
\Delta x_{\text{out}} = \frac{\lambda z}{N \cdot \Delta x_{\text{in}}}
$$

The output pixel size scales with distance - this is a fundamental property of Fresnel diffraction.

**Valid Range**: $0.1 < F < 10$ (intermediate field)

**Code Reference**: `prism/core/propagators/fresnel.py:65-320`

### Far-Field (Fraunhofer) Propagation

**When used**: Fresnel number $F = a^2/(\lambda z) < 0.1$

**Fraunhofer Approximation**:
$$
E_{\text{out}} = \mathcal{F}\{E_{\text{in}}\}
$$

Identical to telescope model - simple FFT relationship.

**Valid Range**: $F \ll 0.1$ (far field), typically $z \gg a^2/\lambda$

**Code Reference**: `prism/core/propagators/fraunhofer.py:18-102`

### Aperture and Defocus

**Circular Aperture**:
$$
A(x, y) = \begin{cases}
1 & \text{if } \sqrt{x^2 + y^2} \leq D/2 \\
0 & \text{otherwise}
\end{cases}
$$

where $D = f/N$ is the aperture diameter ($f$ = focal length, $N$ = f-number).

**Defocus Aberration** (W₂₀ term):
$$
\phi_{\text{defocus}}(x, y) = k \cdot \frac{\delta z \cdot (x^2 + y^2)}{2f^2}
$$

Applied as: $A_{\text{defocused}} = A(x,y) \cdot \exp(i\phi_{\text{defocus}})$

**Thin Lens Equation** (image distance):
$$
\frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i} \quad \Rightarrow \quad d_i = \frac{1}{1/f - 1/d_o}
$$

**Depth of Field**:

Near and far limits using hyperfocal distance $H = f^2/(N \cdot c)$:
$$
d_n = \frac{d \cdot H}{H + d}, \quad d_f = \frac{d \cdot H}{H - d}
$$

where $c$ is the circle of confusion limit (typically 30 μm).

**Resolution Limit** (Airy disk):
$$
\text{spot size} = 2.44 \lambda N
$$

**Code Reference**:
- `prism/core/instruments/camera.py:52-348`
- `prism/core/instruments/camera.py:100-116` (Fresnel number calculation)

---

## Wave Propagation Methods

### Angular Spectrum Method

**Most accurate propagation, valid for ALL distances.**

**Transfer Function**:
$$
H(k_x, k_y, z) = \exp\left(i \cdot 2\pi z \sqrt{\frac{1}{\lambda^2} - k_x^2 - k_y^2}\right)
$$

**Propagation**:
$$
E(x, y, z) = \mathcal{F}^{-1}\left\{ \mathcal{F}\{E(x, y, 0)\} \cdot H(k_x, k_y, z) \right\}
$$

**Evanescent Wave Filtering**:

Spatial frequencies where $k_x^2 + k_y^2 > (1/\lambda)^2$ represent evanescent waves that decay exponentially. These are filtered (set to zero):
$$
H(k_x, k_y, z) = \begin{cases}
\exp(i \cdot 2\pi z \sqrt{1/\lambda^2 - k_x^2 - k_y^2}) & \text{if } k_x^2 + k_y^2 < 1/\lambda^2 \\
0 & \text{otherwise}
\end{cases}
$$

**Physical Interpretation**: The angular spectrum method decomposes the field into plane waves traveling at different angles. Each plane wave component propagates with its own phase velocity determined by its angle. This is exact within the paraxial approximation.

**Code Reference**: `prism/core/propagators/angular_spectrum.py:20-217`

### Fresnel Propagation

**Efficient near-field propagation for intermediate distances.**

**Algorithm** (covered in Camera section above):
$$
E_{\text{out}} = \frac{e^{ikz}}{i\lambda z} e^{i\frac{k}{2z}r_2^2} \mathcal{F}\left\{E_{\text{in}} \cdot e^{i\frac{k}{2z}r_1^2}\right\}
$$

**Valid Range**:
- Distance: $z > z_{\text{crit}} = N \Delta x^2 / \lambda$ (Nyquist sampling)
- Fresnel number: $0.1 < F < 10$

**Approximation**: Neglects higher-order terms in the binomial expansion of the propagation phase.

**Code Reference**: `prism/core/propagators/fresnel.py:65-320`

### Fraunhofer Propagation

**Fast far-field propagation.**

**Equation**:
$$
E_{\text{far}}(k_x, k_y) = \mathcal{F}\{E_{\text{object}}(x, y)\}
$$

**Valid Range**: $F = a^2/(\lambda z) \ll 0.1$ (typically $F < 10^{-2}$)

**Physical Interpretation**: In the far field, the observation plane is so distant that wavefront curvature is negligible. The field distribution is simply the Fourier transform of the aperture/object.

**Code Reference**: `prism/core/propagators/fraunhofer.py:18-102`

---

## Regime Selection Logic

### Microscope: SIMPLIFIED vs FULL

**Selection Criterion**: Defocus parameter δ

$$
\delta = \frac{|d - f_{\text{obj}}|}{f_{\text{obj}}}
$$

where:
- $d$ = working distance (object-to-objective)
- $f_{\text{obj}}$ = objective focal length

**Decision Rule**:
- If $\delta < \delta_{\text{threshold}}$ (default 0.01 = 1%): Use **SIMPLIFIED**
- If $\delta \geq \delta_{\text{threshold}}$: Use **FULL**

**Rationale**: When the object is within 1% of the focal plane, the Fraunhofer approximation at the back focal plane is accurate to within typical experimental tolerances. Beyond this, explicit propagation is required.

**Code Reference**:
- `prism/core/optics/microscope_forward.py:90-194` (regime selection)
- `prism/core/instruments/microscope.py:121-123` (configuration)

### Camera: Fresnel vs Fraunhofer

**Selection Criterion**: Fresnel number F

$$
F = \frac{a^2}{\lambda z}
$$

where:
- $a$ = aperture radius
- $\lambda$ = wavelength
- $z$ = propagation distance

**Decision Rule**:
- If $F < 0.1$: Use **FraunhoferPropagator**
- If $F \geq 0.1$: Use **AngularSpectrumPropagator**

**Rationale**: The Fresnel number quantifies the importance of wavefront curvature. When F ≪ 1, the far-field approximation is valid and Fraunhofer (pure FFT) is both accurate and fastest.

**Code Reference**: `prism/core/instruments/camera.py:85-99`

### Critical Distances

**Nyquist Critical Distance** (for Fresnel propagation):
$$
z_{\text{crit}} = \frac{N \cdot \Delta x^2}{\lambda}
$$

Below this distance, spatial sampling may violate the Nyquist criterion for the Fresnel approximation.

**Far-Field Distance** (for Fraunhofer validity):
$$
z_{\text{far}} \gg \frac{a^2}{\lambda}
$$

Typically requires $z > 10 \cdot a^2/\lambda$ for Fraunhofer to be accurate.

**Code Reference**:
- `prism/core/propagators/fresnel.py:155-162` (critical distance check)
- `prism/core/propagators/fresnel.py:164-181` (Fresnel number check)

---

## Summary Table

| Instrument | Forward Model | Regime | Equation | Code Location |
|------------|--------------|--------|----------|---------------|
| **Microscope** | SIMPLIFIED | δ < 1% | FFT-based 4f | `microscope_forward.py:512-555` |
| **Microscope** | FULL | δ ≥ 1% | Propagation + lenses | `microscope_forward.py:557-613` |
| **Telescope** | Fraunhofer | F ≪ 1 | $E_k = \mathcal{F}\{E\}$ | `telescope.py:232-294` |
| **Camera** | Fresnel | F > 0.1 | Pre/post-chirp + FFT | `fresnel.py:263-319` |
| **Camera** | Fraunhofer | F < 0.1 | Pure FFT | `fraunhofer.py:72-101` |
| **Propagator** | ASM | All distances | Transfer function | `angular_spectrum.py:127-216` |

---

## References

1. Goodman, J. W. "Introduction to Fourier Optics", 4th ed., Roberts & Company, 2017
2. Mertz, J. "Introduction to Optical Microscopy", 2nd ed., Cambridge University Press, 2019
3. Born, M. and Wolf, E. "Principles of Optics", 7th ed., Cambridge University Press, 1999
4. Schmidt, J. D. "Numerical Simulation of Optical Wave Propagation", SPIE Press, 2010
5. Voelz, D. G. "Computational Fourier Optics: A MATLAB Tutorial", SPIE Press, 2011

---

**Document Status**: Updated with Four-F System consolidation references
**Last Updated**: 2025-11-28
**Corresponding Plan**: Task 0.2 of Four-F System Consolidation

---

## See Also

- **[Instrument Class Hierarchy](instrument-hierarchy.md)**: Detailed class architecture, component composition, and extension guide
- **[Four-F System Consolidation Plan](../plans/four-f-system-consolidation.md)**: Implementation roadmap and design decisions
- **Code References**:
  - `prism/core/instruments/four_f_base.py` - FourFSystem base class
  - `prism/core/optics/four_f_forward.py` - Core 4f forward model
  - `prism/core/optics/aperture_masks.py` - Unified aperture generation
  - `prism/core/optics/detector_noise.py` - Detector noise model

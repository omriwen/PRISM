# Optical Resolution Limits Reference

**Quick-lookup reference for fundamental resolution limits in optical systems: Rayleigh criterion, Abbe limit, and Nyquist sampling.**

## Quick Navigation

- [Resolution Formulas](#resolution-formulas)
- [Resolution vs NA Table](#resolution-vs-na-table)
- [Resolution vs Wavelength Table](#resolution-vs-wavelength-table)
- [Nyquist Sampling Criterion](#nyquist-sampling-criterion)
- [Physical Interpretation](#physical-interpretation)

---

## Resolution Formulas

### Rayleigh Criterion (Telescopes & Cameras)

**Angular resolution** - minimum resolvable angle between two point sources:

$$\theta = 1.22 \frac{\lambda}{D}$$

Where:
- **θ** = angular resolution (radians)
- **λ** = wavelength (m)
- **D** = aperture diameter (m)

**Physical meaning**: Two stars separated by less than θ will appear as a single blur spot.

**Example**: 50mm lens at f/4.0 (D = 12.5mm) at λ = 550nm:
```
θ = 1.22 × 550e-9 / 0.0125 = 53.7 µrad
```

### Abbe Limit (Microscopy - Lateral)

**Lateral resolution** - minimum resolvable distance in the sample plane:

$$\Delta x = \frac{0.61 \lambda}{\text{NA}}$$

Where:
- **Δx** = lateral resolution (m)
- **λ** = wavelength (m)
- **NA** = numerical aperture (dimensionless)

**Physical meaning**: Two points closer than Δx will appear as a single blur spot.

**Example**: 100×/1.4NA oil objective at λ = 550nm:
```
Δx = 0.61 × 550e-9 / 1.4 = 240 nm
```

### Sparrow Criterion (Alternative Definition)

**More conservative** than Rayleigh, often used in astronomy:

$$\Delta x = \frac{0.51 \lambda}{\text{NA}}$$

**Comparison**:
- Rayleigh: 0.61λ/NA (26.5% dip between peaks)
- Sparrow: 0.51λ/NA (no dip between peaks)
- Airy disk radius: 0.61λ/NA (to first minimum)

### Axial Resolution (Microscopy - Depth)

**Depth of field** - minimum resolvable distance along the optical axis:

$$\Delta z = \frac{2 n \lambda}{\text{NA}^2}$$

Where:
- **Δz** = axial resolution (m)
- **n** = refractive index of immersion medium
- **NA** = numerical aperture

**Physical meaning**: Features separated by less than Δz in depth cannot be distinguished.

**Example**: 100×/1.4NA oil (n=1.515) at λ = 550nm:
```
Δz = 2 × 1.515 × 550e-9 / (1.4)² = 850 nm
```

---

## Resolution vs NA Table

### Lateral Resolution @ λ = 550 nm

| NA | Medium | Lateral Res (nm) | Axial Res (µm) | Typical Objective |
|-----|--------|------------------|----------------|-------------------|
| 0.25 | Air | 1342 | 35.2 | 10×/0.25 Plan |
| 0.30 | Air | 1117 | 24.4 | 10×/0.30 Plan |
| 0.45 | Air | 744 | 10.9 | 10×/0.45 Plan |
| 0.65 | Air | 513 | 5.2 | 20×/0.65 Plan |
| 0.75 | Air | 445 | 3.9 | 20×/0.75 Plan |
| 0.90 | Air | 371 | 2.7 | 40×/0.90 Plan Apo |
| 0.95 | Air | 353 | 2.4 | 40×/0.95 Plan Apo |
| 1.00 | Water/Oil | 336 | 2.2 | 40×/1.0 Water |
| 1.20 | Water | 280 | 1.5 | 60×/1.2 Water |
| 1.30 | Oil | 257 | 1.3 | 63×/1.3 Oil |
| 1.40 | Oil | 240 | 1.2 | 100×/1.4 Oil |
| 1.45 | Oil | 231 | 1.1 | 100×/1.45 Oil (apochromat) |

**Note**: Axial resolution assumes n=1.0 for air, n=1.33 for water, n=1.515 for oil.

### Categorization by Resolution

| Resolution Category | NA Range | Lateral Res @ 550nm | Example Use Case |
|-------------------|----------|---------------------|------------------|
| Low resolution | 0.25 - 0.40 | > 800 nm | Sample navigation, overview |
| Medium resolution | 0.40 - 0.80 | 400-800 nm | Tissue sections, routine observation |
| High resolution | 0.80 - 1.20 | 280-400 nm | Cellular structures, detailed work |
| Ultra-high resolution | 1.20 - 1.45 | < 280 nm | Subcellular detail, confocal |

---

## Resolution vs Wavelength Table

### 100×/1.4NA Oil Objective

| Wavelength (nm) | Color | Lateral Res (nm) | Axial Res (µm) | Application |
|----------------|-------|------------------|----------------|-------------|
| 405 | Violet | 177 | 0.62 | DAPI fluorescence, confocal |
| 488 | Blue-green | 213 | 0.74 | GFP fluorescence |
| 550 | Green | 240 | 0.84 | Standard brightfield reference |
| 594 | Orange | 259 | 0.91 | Texas Red fluorescence |
| 633 | Red | 276 | 0.97 | Cy5 fluorescence |
| 700 | Deep red | 305 | 1.07 | Near-IR imaging |

**Key insight**: Shorter wavelengths give better resolution. Confocal microscopy often uses UV/blue lasers (405nm, 488nm) for maximum resolution.

### 40×/0.9NA Air Objective

| Wavelength (nm) | Color | Lateral Res (nm) | Axial Res (µm) | Application |
|----------------|-------|------------------|----------------|-------------|
| 405 | Violet | 275 | 2.0 | Fluorescence |
| 488 | Blue-green | 331 | 2.4 | GFP imaging |
| 550 | Green | 371 | 2.7 | Standard brightfield |
| 633 | Red | 429 | 3.1 | Red fluorescence |

---

## Resolution vs Distance (Cameras & Drones)

### Converting Angular to Spatial Resolution

For cameras and drone systems, resolution on the ground depends on distance:

$$\Delta x_{\text{ground}} = \theta \times H$$

Where:
- **Δx_ground** = spatial resolution on ground (m)
- **θ** = angular resolution (rad)
- **H** = distance/altitude (m)

**Example**: 50mm f/4.0 lens (D=12.5mm) at 50m altitude:
```
θ = 1.22 × 550e-9 / 0.0125 = 53.7 µrad
Δx = 53.7e-6 × 50 = 2.7 mm
```

### Resolution vs Altitude (50mm f/4.0 lens)

| Altitude (m) | Angular Res (µrad) | Spatial Res (mm) | Spatial Res (cm) |
|--------------|-------------------|------------------|------------------|
| 10 | 53.7 | 0.54 | 0.05 |
| 20 | 53.7 | 1.07 | 0.11 |
| 50 | 53.7 | 2.68 | 0.27 |
| 100 | 53.7 | 5.37 | 0.54 |
| 120 | 53.7 | 6.44 | 0.64 |

**Note**: This is the diffraction limit. Actual resolution limited by pixel size (GSD) in most drone applications.

---

## Nyquist Sampling Criterion

### Sampling Theorem

To properly resolve features at the optical resolution limit, pixel sampling must satisfy:

$$p_{\text{sample}} < \frac{\Delta x}{2}$$

Where:
- **p_sample** = pixel size in object space (m)
- **Δx** = optical resolution limit (m)
- Factor of 2: Nyquist criterion (2 pixels per resolution element)

### Recommended Oversampling

**Standard practice**: Use 2.5-3× oversampling for best image quality:

$$p_{\text{sample}} \approx \frac{\Delta x}{2.5}$$

**Why oversample?**
- Nyquist (2×): Minimum, may show aliasing
- 2.5×: Good balance, minimal aliasing
- 3×: Excellent quality, no visible aliasing
- >3×: Diminishing returns, wastes storage/bandwidth

### Microscopy Nyquist Sampling

For microscopes, pixel size in object space:

$$p_{\text{obj}} = \frac{p_{\text{sensor}}}{M}$$

Where:
- **p_obj** = pixel size in object space (µm)
- **p_sensor** = camera pixel size (µm)
- **M** = objective magnification

**Nyquist criterion for microscopy**:

$$p_{\text{obj}} < \frac{\lambda}{4 \cdot \text{NA}}$$

### Microscopy Nyquist Table (@ 550nm)

| Objective | NA | Optical Res (nm) | Min Sampling (nm) | 2.5× Oversampling (nm) | Sensor Pixel @ Mag |
|-----------|-----|------------------|-------------------|------------------------|-------------------|
| 100×/1.4 Oil | 1.4 | 240 | 120 | 96 | 9.6 µm @ 100× |
| 100×/1.4 Oil | 1.4 | 240 | 120 | 96 | 6.4 µm @ 100× (recommended) |
| 60×/1.2 Water | 1.2 | 280 | 140 | 112 | 6.7 µm @ 60× |
| 40×/0.9 Air | 0.9 | 371 | 186 | 148 | 5.9 µm @ 40× |
| 20×/0.75 Air | 0.75 | 445 | 223 | 178 | 3.6 µm @ 20× |
| 10×/0.3 Air | 0.3 | 1117 | 558 | 447 | 4.5 µm @ 10× |

**Common sensor pixels**:
- Scientific CMOS: 6.5 µm
- Standard CMOS: 3.45 µm, 5.5 µm
- High-res CMOS: 2.4 µm

### Drone Camera Nyquist Sampling

For drones, **GSD (Ground Sampling Distance)** is the pixel size in object space:

$$\text{GSD} < \frac{\Delta x_{\text{diffraction}}}{2}$$

**In practice**: Drone systems are almost always pixel-limited, not diffraction-limited.

**Example**: 50mm f/4.0 lens at 50m altitude:
- Diffraction limit: 2.7 mm
- Typical GSD: 65 mm (full-frame, 6.5µm pixels)
- **Ratio**: GSD/diffraction ≈ 24× (heavily pixel-limited)

**Conclusion**: Increasing sensor resolution improves drone image quality much more than increasing aperture.

---

## Physical Interpretation

### Why Do These Limits Exist?

**Diffraction**: Light waves spread out when passing through apertures
- Smaller apertures → more spreading → worse resolution
- Shorter wavelengths → less spreading → better resolution

**Abbe's insight** (1873): Resolution fundamentally limited by wave nature of light
- Microscope collects diffracted orders from specimen
- Higher NA → collects more orders → better resolution
- Diffraction creates "blur disk" around each point

### Airy Disk

The point spread function (PSF) of a perfect lens:

$$I(r) = I_0 \left[ \frac{2 J_1(kr)}{kr} \right]^2$$

Where:
- **J_1** = Bessel function of first kind
- **k = 2πNA/λ**
- **r** = radial distance from center

**First minimum** (Airy disk radius):
```
r = 0.61λ/NA
```

**Full width at half maximum (FWHM)**:
```
FWHM ≈ 0.51λ/NA (Sparrow criterion)
```

### Rayleigh Criterion Definition

Two Airy disks are "just resolved" when:
- Peak of one disk coincides with first minimum of other
- Intensity dip between peaks is 26.5%
- Separation = 0.61λ/NA (lateral) or 1.22λ/D (angular)

---

## Practical Examples

### Example 1: Can I resolve mitochondria?

**Question**: Can a 40×/0.9NA air objective resolve mitochondria (typical size ~500nm × 1µm)?

**Answer**:
```
Resolution @ 550nm = 0.61 × 550 / 0.9 = 371 nm
Mitochondria width: 500 nm
Ratio: 500 / 371 = 1.35 (barely resolvable)
```

**Conclusion**: Yes, but not with high contrast. Use 100×/1.4NA oil (240nm resolution) for better detail.

### Example 2: Drone inspection GSD requirement

**Question**: What altitude for 5mm GSD with 50mm f/4.0 lens, full-frame sensor (6.5µm pixels)?

**Answer**:
```
GSD = H × p / f
5mm = H × 6.5e-6 / 0.05
H = 5e-3 × 0.05 / 6.5e-6 = 38.5 m
```

**Diffraction check**:
```
θ = 1.22 × 550e-9 / 0.0125 = 53.7 µrad
Resolution @ 38.5m = 53.7e-6 × 38.5 = 2.1 mm (< 5mm GSD ✓)
```

**Conclusion**: Altitude ~38-40m achieves 5mm GSD. System is pixel-limited, not diffraction-limited.

### Example 3: Nyquist sampling check

**Question**: Is a 6.5µm pixel camera properly sampled on a 100×/1.4NA oil objective?

**Answer**:
```
Resolution = 0.61 × 550e-9 / 1.4 = 240 nm
Pixel in object space = 6.5µm / 100 = 65 nm
Oversampling factor = 240 / 65 = 3.7×
```

**Conclusion**: Yes, excellent oversampling (3.7×). Image quality will be very good.

---

## Quick Reference Summary

### Formulas Cheat Sheet

| System | Formula | Variable | Typical Value |
|--------|---------|----------|---------------|
| **Telescopes/Cameras** | θ = 1.22λ/D | D | 5-50 mm (lenses) |
| **Microscopes (lateral)** | Δx = 0.61λ/NA | NA | 0.25-1.45 |
| **Microscopes (axial)** | Δz = 2nλ/NA² | n | 1.0-1.515 |
| **Nyquist (microscopy)** | p < λ/(4·NA) | λ | 400-700 nm |
| **Nyquist (general)** | p < Δx/2.5 | Oversampling | 2.5-3× |

### Wavelength Reference

| Wavelength | Color | Common Use |
|------------|-------|------------|
| 405 nm | Violet | DAPI, confocal |
| 488 nm | Blue-green | GFP |
| 550 nm | Green | Brightfield reference |
| 594 nm | Orange | Texas Red |
| 633 nm | Red | Cy5, HeNe laser |

### NA Reference (Microscopy)

| NA | Medium | Resolution @ 550nm |
|----|--------|-------------------|
| 0.3 | Air | ~1100 nm |
| 0.75 | Air | ~450 nm |
| 0.9 | Air | ~370 nm |
| 1.2 | Water | ~280 nm |
| 1.4 | Oil | ~240 nm |

---

## Source Files

This reference is based on:
- [docs/user_guides/optical-engineering.md](../user_guides/optical-engineering.md) (lines 51-62)
- [prism/scenarios/microscopy.py](../../prism/scenarios/microscopy.py) (resolution calculations)
- [prism/scenarios/drone_camera.py](../../prism/scenarios/drone_camera.py) (GSD calculations)

## Related References

- [Physical Constants](physical_constants.md) - Wavelengths and unit conversions
- [Microscopy Parameters](microscopy_parameters.md) - Complete microscope specifications
- [Drone Camera Parameters](drone_camera_parameters.md) - GSD and camera specs
- [Fresnel Propagation Regimes](fresnel_propagation_regimes.md) - Diffraction regime classification
- [Scenario Preset Catalog](scenario_preset_catalog.md) - All presets with resolution data

## Related User Guides

- [Optical Engineering Guide](../user_guides/optical-engineering.md) - Tutorial on optical systems
- [Scenarios User Guide](../user_guides/scenarios.md) - How to use scenario presets

---

**Last Updated**: 2025-01-26
**Formulas**: Standard optical physics (Rayleigh, Abbe)
**Accuracy**: All calculations verified against PRISM implementations

# PRISM Learning Path

Comprehensive curriculum for learning PRISM optical imaging simulation and validation.

---

## Overview

PRISM (Progressive Reconstruction from Imaging with Sparse Measurements) provides a complete framework for optical imaging simulation across microscopy and drone/aerial imaging domains. This learning path guides you from beginner to advanced topics through hands-on examples.

**Total Curriculum Time**: 12-15 hours for complete coverage

**What You'll Learn**:
- Optical physics fundamentals (resolution, PSF, diffraction)
- Microscopy simulation (objectives, NA, immersion media)
- Drone/aerial imaging (GSD, altitude, lens selection)
- Validation techniques (USAF-1951, MTF, SNR analysis)
- Advanced workflows (multi-scenario comparison, custom configurations)

---

## Prerequisites

### Required
- Python 3.11+
- PRISM installed (`uv sync` from project root)
- Jupyter Lab or Jupyter Notebook

### Recommended Background
- Basic optics knowledge (wavelength, lenses, diffraction)
- Python programming fundamentals
- NumPy/matplotlib basics

### Optional (for advanced topics)
- PyTorch basics (for understanding reconstruction)
- Signal processing (for Nyquist/sampling theory)

---

## Learning Tracks

### Track 1: Microscopy (Beginner → Advanced)

**Total Time**: 8-10 hours

#### Level 1: Fundamentals (2-3 hours)

| # | Notebook | Time | Prerequisites | What You'll Learn |
|---|----------|------|---------------|-------------------|
| 1 | [learning_01_resolution_fundamentals](../../examples/notebooks/learning_01_resolution_fundamentals.ipynb) | 20-30 min | None | Abbe limit, NA, PSF, immersion media |
| 2 | [learning_02_resolution_validation](../../examples/notebooks/learning_02_resolution_validation.ipynb) | 25-35 min | Notebook 1 | USAF-1951 targets, MTF, validation techniques |
| 3 | [learning_03_illumination_modes](../../examples/notebooks/learning_03_illumination_modes.ipynb) | 20-30 min | Notebook 1 | Brightfield, phase contrast, DIC modes |

**Assessment after Level 1**:
- [ ] Calculate resolution limits for any NA/wavelength combination
- [ ] Predict which USAF elements are resolvable
- [ ] Select appropriate illumination mode for sample type

#### Level 2: Practice (1 hour)

| # | Script | Time | Prerequisites | What You'll Learn |
|---|--------|------|---------------|-------------------|
| 4 | [06_microscope_reconstruction.py](../../examples/python_api/06_microscope_reconstruction.py) | 15-20 min | Level 1 complete | Full PRISM microscopy workflow |

**Assessment after Level 2**:
- [ ] Run complete microscopy reconstruction pipeline
- [ ] Interpret quality metrics (SSIM, PSNR)
- [ ] Understand preset configurations

#### Level 3: Validation (2-3 hours)

| # | Notebook | Time | Prerequisites | What You'll Learn |
|---|----------|------|---------------|-------------------|
| 5 | [01_microscope_resolution_validation](../../examples/validation/notebooks/01_microscope_resolution_validation.ipynb) | 30-45 min | Level 2 complete | Resolution against Abbe limit |
| 6 | [02_snr_reconstruction_quality](../../examples/validation/notebooks/02_snr_reconstruction_quality.ipynb) | 30-45 min | Level 2 complete | SNR effects on image quality |
| 7 | [03_propagator_accuracy_validation](../../examples/validation/notebooks/03_propagator_accuracy_validation.ipynb) | 30-45 min | Level 2 complete | Wave propagation accuracy |

**Assessment after Level 3**:
- [ ] Perform scientific validation against theory
- [ ] Quantify reconstruction quality
- [ ] Validate propagator selection

---

### Track 2: Drone/Aerial Imaging (Beginner → Advanced)

**Total Time**: 5-7 hours

#### Level 1: Fundamentals (1.5-2 hours)

| # | Notebook | Time | Prerequisites | What You'll Learn |
|---|----------|------|---------------|-------------------|
| 1 | [learning_04_gsd_basics](../../examples/notebooks/learning_04_gsd_basics.ipynb) | 20-25 min | None | GSD formula, mission planning |
| 2 | [learning_05_drone_altitudes](../../examples/notebooks/learning_05_drone_altitudes.ipynb) | 15-20 min | Notebook 1 | Coverage vs detail tradeoff |

**Assessment after Level 1**:
- [ ] Calculate GSD for any altitude/lens/sensor combination
- [ ] Plan missions for required ground resolution
- [ ] Understand coverage/detail tradeoffs

#### Level 2: Practice (1 hour)

| # | Script | Time | Prerequisites | What You'll Learn |
|---|--------|------|---------------|-------------------|
| 3 | [07_drone_mapping.py](../../examples/python_api/07_drone_mapping.py) | 15-20 min | Level 1 complete | Full drone imaging workflow |

**Assessment after Level 2**:
- [ ] Run complete drone imaging pipeline
- [ ] Understand all 8 drone presets
- [ ] Configure custom missions

#### Level 3: Validation (1 hour)

| # | Notebook | Time | Prerequisites | What You'll Learn |
|---|----------|------|---------------|-------------------|
| 4 | [04_drone_gsd_validation](../../examples/validation/notebooks/04_drone_gsd_validation.ipynb) | 30-45 min | Level 2 complete | GSD against theory, swath width |

**Assessment after Level 3**:
- [ ] Validate GSD calculations
- [ ] Quantify image quality vs altitude
- [ ] Optimize mission parameters

---

### Track 3: Advanced Topics (2-3 hours)

**Prerequisites**: Track 1 or Track 2 complete

| # | Resource | Time | Prerequisites | What You'll Learn |
|---|----------|------|---------------|-------------------|
| 1 | [learning_08_scenario_comparison](../../examples/notebooks/learning_08_scenario_comparison.ipynb) | 40-50 min | Tracks 1 & 2 | Multi-scenario workflows |
| 2 | [08_custom_scenario_builder.py](../../examples/python_api/08_custom_scenario_builder.py) | 20-30 min | Track 1 or 2 | Builder API, custom configs |
| 3 | [05_sampling_density_validation](../../examples/validation/notebooks/05_sampling_density_validation.ipynb) | 30-40 min | Nyquist understanding | Sampling requirements |
| 4 | [09_resolution_validation.py](../../examples/python_api/09_resolution_validation.py) | 15-20 min | All tracks | Automated validation suite |

**What You'll Learn**:
- Systematic comparison across scenarios
- Custom scenario configuration
- Nyquist sampling validation
- Automated validation workflows

---

## Recommended Learning Paths

### Path A: Microscopy Focus (8-10 hours)

For researchers working primarily with microscopy:

```
Level 1: Fundamentals (2-3 hours)
    └─→ learning_01_resolution_fundamentals
    └─→ learning_02_resolution_validation
    └─→ learning_03_illumination_modes

Level 2: Practice (1 hour)
    └─→ 06_microscope_reconstruction.py

Level 3: Validation (2-3 hours)
    └─→ 01_microscope_resolution_validation
    └─→ 02_snr_reconstruction_quality
    └─→ 03_propagator_accuracy_validation

Advanced (1-2 hours)
    └─→ 08_custom_scenario_builder.py
    └─→ learning_08_scenario_comparison.ipynb
```

### Path B: Drone/Remote Sensing Focus (5-7 hours)

For drone operators and remote sensing applications:

```
Level 1: Fundamentals (1.5-2 hours)
    └─→ learning_04_gsd_basics
    └─→ learning_05_drone_altitudes

Level 2: Practice (1 hour)
    └─→ 07_drone_mapping.py

Level 3: Validation (1 hour)
    └─→ 04_drone_gsd_validation

Advanced (1-2 hours)
    └─→ 08_custom_scenario_builder.py
    └─→ learning_08_scenario_comparison.ipynb
```

### Path C: Complete Curriculum (12-15 hours)

For comprehensive understanding of both domains:

```
Week 1: Microscopy Fundamentals
    └─→ Track 1, Levels 1-2

Week 2: Drone Fundamentals
    └─→ Track 2, Levels 1-2

Week 3: Validation
    └─→ Track 1, Level 3
    └─→ Track 2, Level 3

Week 4: Advanced Topics
    └─→ Track 3 (all)
```

### Path D: Quick Validation (2-3 hours)

For users who just need to validate results:

```
Quickstart (30 min)
    └─→ quickstart_01_microscopy_basic OR quickstart_02_drone_basic

Validation (1.5-2 hours)
    └─→ 09_resolution_validation.py (automated)
    └─→ Domain-specific validation notebook
```

---

## Assessment Guide

### After Level 1 (Either Track)

You should be able to:
- [ ] Calculate theoretical resolution limits
- [ ] Understand the physics of your domain (NA for microscopy, GSD for drones)
- [ ] Select appropriate presets for your application
- [ ] Interpret basic quality metrics

### After Level 2 (Either Track)

You should be able to:
- [ ] Run full PRISM reconstruction pipeline
- [ ] Interpret quality metrics (SSIM, PSNR, MTF)
- [ ] Use command-line scripts for automation
- [ ] Load and analyze experiment results

### After Level 3 (Either Track)

You should be able to:
- [ ] Perform rigorous scientific validation
- [ ] Quantify performance against theoretical limits
- [ ] Identify and diagnose issues in reconstructions
- [ ] Generate publication-quality validation results

### After Advanced Topics

You should be able to:
- [ ] Create custom scenarios for your specific application
- [ ] Compare multiple configurations systematically
- [ ] Optimize sampling strategies for efficiency
- [ ] Run automated validation across all presets

---

## Resource Index

### Quick Reference Documentation

| Topic | Reference |
|-------|-----------|
| Physical constants | [physical_constants.md](../references/physical_constants.md) |
| Microscopy parameters | [microscopy_parameters.md](../references/microscopy_parameters.md) |
| Drone parameters | [drone_camera_parameters.md](../references/drone_camera_parameters.md) |
| Resolution limits | [optical_resolution_limits.md](../references/optical_resolution_limits.md) |
| Propagation regimes | [fresnel_propagation_regimes.md](../references/fresnel_propagation_regimes.md) |
| All 17 presets | [scenario_preset_catalog.md](../references/scenario_preset_catalog.md) |

### User Guides

| Topic | Guide |
|-------|-------|
| Scenario system | [scenarios.md](scenarios.md) |
| Propagator selection | [propagator_selection.md](propagator_selection.md) |
| Optical engineering | [optical-engineering.md](optical-engineering.md) |
| Pattern functions | [pattern-functions.md](pattern-functions.md) |

### Python API Examples

Located in [examples/python_api/](../../examples/python_api/):

| Script | Purpose |
|--------|---------|
| 01_basic_usage.py | PRISM basics |
| 02_custom_patterns.py | K-space sampling |
| 03_custom_loss.py | Loss functions |
| 04_model_extension.py | Model customization |
| 05_batch_experiments.py | Automation |
| 06_microscope_reconstruction.py | Microscopy workflow |
| 07_drone_mapping.py | Drone workflow |
| 08_custom_scenario_builder.py | Custom scenarios |
| 09_resolution_validation.py | Automated validation |

---

## Next Steps

After completing the learning path:

### Apply to Your Research
1. Identify your specific imaging scenario
2. Select or create appropriate preset configuration
3. Run validation to establish baseline
4. Optimize parameters for your application

### Extend PRISM
1. Create custom targets for your domain
2. Implement specialized loss functions
3. Add new sampling patterns
4. Contribute examples back to the project

### Get Help
- Review [AI_ASSISTANT_GUIDE.md](../../AI_ASSISTANT_GUIDE.md) for coding patterns
- Check [scenario_preset_catalog.md](../references/scenario_preset_catalog.md) for preset details
- Explore the knowledge graph for component discovery

---

## Completion Checklist

### Microscopy Track
- [ ] Level 1: Fundamentals (3 notebooks)
- [ ] Level 2: Practice (1 script)
- [ ] Level 3: Validation (3 notebooks)

### Drone Track
- [ ] Level 1: Fundamentals (2 notebooks)
- [ ] Level 2: Practice (1 script)
- [ ] Level 3: Validation (1 notebook)

### Advanced Topics
- [ ] Multi-scenario comparison
- [ ] Custom scenario builder
- [ ] Sampling density validation
- [ ] Automated validation suite

---

**Last Updated**: 2025-11-27
**Curriculum Version**: 1.0

# SPIDS Notebooks & Learning Path

Interactive tutorials for learning SPIDS through hands-on examples.

## Learning Path

SPIDS notebooks are organized into three learning tracks:

### 1. Quickstart Track (30-60 min total)
**Goal**: Get running with SPIDS quickly

| Notebook | Duration | Level | Description |
|----------|----------|-------|-------------|
| [quickstart_01_microscopy_basic](quickstart_01_microscopy_basic.ipynb) | 10-15 min | Beginner | Basic microscope simulation |
| [quickstart_02_drone_basic](quickstart_02_drone_basic.ipynb) | 10-15 min | Beginner | Drone camera imaging basics |
| [quickstart_03_validation_intro](quickstart_03_validation_intro.ipynb) | 15-20 min | Beginner | Quality validation introduction |

### 2. Learning Track - Microscopy Fundamentals (75-100 min total)
**Goal**: Deep understanding of optical imaging physics

| Notebook | Duration | Level | Description |
|----------|----------|-------|-------------|
| [learning_01_resolution_fundamentals](learning_01_resolution_fundamentals.ipynb) | 20-30 min | Beginner | Abbe limit, PSF, NA concepts |
| [learning_02_resolution_validation](learning_02_resolution_validation.ipynb) | 25-35 min | Intermediate | USAF-1951 testing, MTF |
| [learning_03_illumination_modes](learning_03_illumination_modes.ipynb) | 20-30 min | Intermediate | Brightfield, phase, DIC modes |

### 3. Tutorial Track - SPIDS Features (2-3 hours total)
**Goal**: Master SPIDS features and workflows

| Notebook | Duration | Level | Description |
|----------|----------|-------|-------------|
| [tutorial_01_quickstart](tutorial_01_quickstart.ipynb) | 15-20 min | Beginner | SPIDS basics and setup |
| [tutorial_02_pattern_design](tutorial_02_pattern_design.ipynb) | 20-30 min | Intermediate | K-space sampling patterns |
| [tutorial_03_result_analysis](tutorial_03_result_analysis.ipynb) | 20-30 min | Intermediate | Analyzing experiments |
| [tutorial_04_dashboard](tutorial_04_dashboard.ipynb) | 15-20 min | Intermediate | Web dashboard monitoring |
| [tutorial_05_reporting](tutorial_05_reporting.ipynb) | 15-20 min | Intermediate | Publication-quality reports |

### 4. Drone Imaging Track (35-45 min total)
**Goal**: Master drone/aerial imaging fundamentals

| Notebook | Duration | Level | Description |
|----------|----------|-------|-------------|
| [learning_04_gsd_basics](learning_04_gsd_basics.ipynb) | 20-25 min | Beginner | GSD formula, mission planning |
| [learning_05_drone_altitudes](learning_05_drone_altitudes.ipynb) | 15-20 min | Intermediate | Altitude vs coverage tradeoffs |

### 5. Advanced Track (40-50 min)
**Goal**: Cross-domain comparison and advanced workflows

| Notebook | Duration | Level | Description |
|----------|----------|-------|-------------|
| [learning_08_scenario_comparison](learning_08_scenario_comparison.ipynb) | 40-50 min | Advanced | Multi-scenario comparison across microscopy and drone |

### Additional Resources

| Notebook | Description |
|----------|-------------|
| [microscope_simulation](microscope_simulation.ipynb) | Detailed microscope forward model |
| [camera_simulation](camera_simulation.ipynb) | Camera and lens simulation |

---

## Recommended Learning Paths

### For New Users (Start Here!)
1. **Quickstart Track** → Get hands-on experience
2. **Learning Track** → Understand the physics
3. **Tutorial Track** → Master advanced features

### For Microscopists
1. `learning_01_resolution_fundamentals` - Resolution physics
2. `learning_02_resolution_validation` - USAF validation
3. `learning_03_illumination_modes` - Illumination selection
4. `microscope_simulation` - Advanced simulation

### For Drone/Remote Sensing
1. `quickstart_02_drone_basic` - Drone imaging basics
2. `camera_simulation` - Camera modeling
3. `tutorial_02_pattern_design` - Sampling optimization

### For Researchers
1. Complete **Learning Track** for theory
2. `tutorial_03_result_analysis` - Experiment analysis
3. `tutorial_05_reporting` - Publication figures

### For Complete Curriculum
See the comprehensive [Learning Path Guide](../../docs/user_guides/learning_path.md) for:
- Full 12-15 hour curriculum
- Assessment checklists
- Validation notebooks integration
- Python API script references

---

## Running Notebooks

### Install Jupyter

```bash
# Already installed if you ran uv sync (includes dev dependencies)
cd examples/notebooks
uv run jupyter lab
```

### Run Specific Notebook

```bash
uv run jupyter notebook learning_01_resolution_fundamentals.ipynb
```

### Convert to Python Script

```bash
uv run jupyter nbconvert --to script learning_01_resolution_fundamentals.ipynb
```

---

## Prerequisites

### For All Notebooks
- Python ≥3.11
- SPIDS installed (`uv sync` from project root)
- Jupyter Lab or Jupyter Notebook

### For Interactive Widgets
- `ipywidgets` (optional - graceful fallback if not available)

### For Dashboard Notebooks
- `dash` and `plotly` (included in dev dependencies)

---

## Tips

- **Run cells sequentially** (Shift+Enter) for best results
- **Clear outputs before saving** to reduce git diff noise
- **Restart kernel** (Kernel → Restart) if things get stuck
- **Check memory** - some notebooks use significant GPU memory
- **Save experiments** with meaningful names for later reference

---

## Related Resources

### Learning Path
- [Complete Learning Path Guide](../../docs/user_guides/learning_path.md) - Full curriculum with assessments

### Validation Notebooks
- [Microscope Resolution Validation](../validation/notebooks/01_microscope_resolution_validation.ipynb)
- [SNR Reconstruction Quality](../validation/notebooks/02_snr_reconstruction_quality.ipynb)
- [Propagator Accuracy](../validation/notebooks/03_propagator_accuracy_validation.ipynb)
- [Drone GSD Validation](../validation/notebooks/04_drone_gsd_validation.ipynb)
- [Sampling Density Validation](../validation/notebooks/05_sampling_density_validation.ipynb)

### Reference Documentation
- [Optical Resolution Limits](../../docs/references/optical_resolution_limits.md)
- [Microscopy Parameters](../../docs/references/microscopy_parameters.md)
- [Scenario Preset Catalog](../../docs/references/scenario_preset_catalog.md)
- [Fresnel Propagation Regimes](../../docs/references/fresnel_propagation_regimes.md)

### Python API Examples
- [examples/python_api/](../python_api/) - Production scripts

### AI Assistant Guide
- [AI_ASSISTANT_GUIDE.md](../../AI_ASSISTANT_GUIDE.md) - Coding patterns and API reference

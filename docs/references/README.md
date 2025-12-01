# PRISM Technical Reference Documentation

**Quick-lookup reference for physical parameters, optical specifications, and preset configurations.**

## Purpose

This directory provides authoritative, concise reference documentation for:
- Physical constants and unit conversions
- Optical resolution formulas and limits
- Microscopy objective specifications
- Drone camera/lens specifications
- Complete preset parameter catalogs

**References vs User Guides**:
- **References** = "What is the value?" (quick lookup, tables, formulas)
- **User Guides** = "How do I use it?" (tutorials, workflows, explanations)

## Quick Navigation

| Reference | Quick Lookup For | Lines |
|-----------|------------------|-------|
| [Physical Constants](physical_constants.md) | Speed of light, wavelengths, units, Fresnel number | ~120 |
| [Microscopy Parameters](microscopy_parameters.md) | Objective specs, NA values, resolution @ 550nm | ~300 |
| [Drone Camera Parameters](drone_camera_parameters.md) | Lens/sensor specs, GSD calculations | ~250 |
| [Optical Resolution Limits](optical_resolution_limits.md) | Abbe limit, Rayleigh criterion, Nyquist | ~150 |
| [Fresnel Propagation Regimes](fresnel_propagation_regimes.md) | Propagation regime by distance | ~150 |
| [Scenario Preset Catalog](scenario_preset_catalog.md) | All 17 presets (9 microscope + 8 drone) | ~200 |

## Usage Examples

### AI Assistant Queries

**"What is the resolution of a 1.4 NA objective?"**
→ See [Microscopy Parameters](microscopy_parameters.md)
→ Answer: ~240nm @ 550nm wavelength

**"What GSD at 100m altitude with 50mm lens?"**
→ See [Drone Camera Parameters](drone_camera_parameters.md)
→ Answer: ~13cm (full-frame sensor with 6.5µm pixels)

**"Which propagator for 50m distance?"**
→ See [Fresnel Propagation Regimes](fresnel_propagation_regimes.md)
→ Answer: Depends on aperture size (Fresnel number calculation)

**"List all microscope presets with oil immersion"**
→ See [Scenario Preset Catalog](scenario_preset_catalog.md)
→ Answer: `microscope_100x_oil`, `microscope_100x_oil_phase`, `microscope_fluorescence_100x`

### Python API

```python
from prism.scenarios import get_scenario_preset

# Load preset and check computed parameters
scenario = get_scenario_preset("microscope_100x_oil")
print(f"Resolution: {scenario.lateral_resolution_nm:.1f} nm")
print(f"NA: {scenario.objective_spec.numerical_aperture}")
```

### CLI

```bash
# List all presets
prism-scenario list

# Get preset details
prism-scenario info microscope_100x_oil
```

## File Naming Conventions

- All files use `snake_case.md`
- All formulas use inline LaTeX: `$...$` or block: `$$...$$`
- All tables are GitHub-flavored Markdown (sortable)
- All parameter values include explicit units
- All source file locations documented for maintainability

## Maintenance

### Updating References

When parameters change in source code:
1. Identify affected reference file (see "Source Files" section in each reference)
2. Update parameter values to match source code exactly
3. Verify all cross-references still work
4. Update "Last Updated" timestamp

### Source File Locations

| Reference | Source Files |
|-----------|--------------|
| Physical Constants | `prism/config/constants.py` |
| Microscopy Parameters | `prism/scenarios/presets.py`, `prism/scenarios/microscopy.py` |
| Drone Parameters | `prism/scenarios/presets.py`, `prism/scenarios/drone_camera.py` |
| Preset Catalog | `prism/scenarios/presets.py` |

## Verification Checklist

Before considering a reference complete:
- [ ] All parameter values match source code (spot-check 5+ values)
- [ ] All formulas include variable definitions
- [ ] All tables are properly formatted (GitHub Markdown)
- [ ] All units are explicit (no ambiguity)
- [ ] All cross-references tested and functional
- [ ] Source file locations documented
- [ ] "Last Updated" timestamp included

## Related Resources

### Learning Path
- **[Complete Learning Path](../user_guides/learning_path.md)**: Full 12-15 hour curriculum
- **[Validation Notebooks](../../examples/validation/notebooks/)**: Hands-on validation exercises
- **[Python API Examples](../../examples/python_api/)**: Production scripts

## Integration with AI_ASSISTANT_GUIDE.md

These references are integrated into [AI_ASSISTANT_GUIDE.md](../../AI_ASSISTANT_GUIDE.md) for discoverability. AI assistants should:
1. Check references FIRST before reading source code
2. Use references for parameter lookup (<20 seconds)
3. Use source code only when references are insufficient

---

**Last Updated**: 2025-11-27
**Status**: Complete (All 6 reference documents + learning path)

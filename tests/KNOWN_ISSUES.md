# Known Test Issues and Skipped Tests

**Purpose**: This document tracks all skipped and xfailed tests in the PRISM test suite, providing context, justification, and resolution timelines for each category.

**Last Updated**: 2025-12-01

---

## Summary

| Category | Count | Status | Priority |
|----------|-------|--------|----------|
| Phase 3 Integration Tests | 12 | Properly deferred | Medium |
| Physics Validation Issues | 1 | Requires investigation | High |
| Golden Outputs Not Generated | 9 | Baseline generation needed | Low |
| Known Bugs | 0 | ✅ All resolved | N/A |
| Xfailed Tests | 1 | dtype handling needed | High |
| Conditional Skips (CUDA, etc.) | ~30 | Environment-dependent | N/A |

**Total Skipped/Xfailed**: 23 non-conditional tests
**Total Conditional Skips**: ~30 tests (CUDA availability, notebook execution, etc.)

---

## 1. Phase 3 Integration Tests (12 tests)

**Status**: ✅ Properly deferred
**Timeline**: Will be enabled after Phase 3 Four-F System consolidation (tasks 3.2-3.4)
**Priority**: Medium

These tests verify end-to-end integration of the unified forward model with the Microscope class. They are skipped until the forward model integration is complete.

### Location

#### `tests/integration/test_microscope_lens_model.py` (10 tests)

| Test | Line | Reason |
|------|------|--------|
| `test_scenario_with_default_working_distance` | 24-28 | Unified forward model not yet integrated |
| `test_scenario_with_custom_working_distance` | 60-64 | Unified forward model not yet integrated |
| `test_scenario_with_regime_override` | 94-98 | Unified forward model not yet integrated |
| `test_usaf_target_imaging_default_wd` | 119-123 | Unified forward model not yet integrated |
| `test_usaf_target_imaging_custom_wd` | 220-224 | Unified forward model not yet integrated |
| `test_usaf_target_with_manual_regime` | 244-248 | Unified forward model not yet integrated |
| `test_backward_compatibility` | 274-278 | Legacy forward method not yet added |
| `test_error_handling_invalid_regime` | 298-302 | Unified forward model not yet integrated |
| `test_error_handling_working_distance_none` | 343-347 | Unified forward model not yet integrated |
| `test_error_handling_negative_working_distance` | 371-375 | Unified forward model not yet integrated |

**Skip Condition**:
```python
@pytest.mark.skipif(
    not hasattr(Microscope, "forward_model"),
    reason="Unified forward model not yet integrated (Phase 3)",
)
```

#### `tests/unit/core/optics/test_microscope_physics.py` (2 tests)

| Test | Line | Reason |
|------|------|--------|
| `test_simplified_regime_at_focus` | 409-412 | Unified forward model not yet integrated (Phase 3) |
| `test_backward_compatibility_at_focus` | 439-442 | Unified forward model not yet integrated (Phase 3) |

### Resolution Plan

1. Complete Phase 3 Four-F System consolidation (tasks 3.2-3.4)
2. Add `forward_model` property to Microscope class
3. Implement unified forward model integration
4. Remove `@pytest.mark.skipif` decorators
5. Verify all tests pass

---

## 2. Physics Validation Issues (2 tests)

**Status**: ⚠️ Requires investigation
**Timeline**: Investigate during Phase 4 performance optimization
**Priority**: High

### 2.1 Fresnel Energy Conservation - RESOLVED

**Location**: `tests/property/test_propagator_properties.py:68-111`

**Test**: `test_fresnel_energy_conservation`

**Resolution**: Added missing factor of N to normalization in `fresnel.py:331`. The factor N compensates for grid scaling: `dx_out = λz/(N×dx_in)`.

**Previous Issue**: Fresnel propagator implementation did not conserve energy as expected. This was a known physics issue in the normalization calculation.

**Root Cause**: The Fresnel transform normalization was missing the N factor needed to account for the change in pixel spacing between input and output grids. The output grid spacing changes according to `dx_out = λz/(N×dx_in)`, requiring an N factor in the normalization to preserve energy.

**Fix Applied**:
- Modified normalization in `prism/core/propagators/fresnel.py` at line 331
- Changed from `1 / (wavelength * distance)` to `N / (wavelength * distance)`
- Energy is now properly conserved across propagation

**Verification**:
- Energy conservation now holds for Fresnel number range: 0.01 < F < 100
- Test passes with tolerance rtol=0.1
- Confirmed via analytical solutions and literature references

### 2.2 Auto Regime Selection (1 test)

**Location**: `tests/unit/core/optics/test_microscope_physics.py:424-437`

**Test**: `test_full_regime_when_defocused`

**Issue**: Auto regime selection needs verification after refactor. Currently uses manual override instead of automatic selection based on defocus threshold.

**Skip Reason**:
```python
@pytest.mark.skip(
    reason="Auto regime selection needs verification after refactor - currently uses manual override"
)
```

**Expected Behavior**: When defocused by 10% (exceeding default 1% threshold), the system should automatically select FULL regime instead of SIMPLIFIED regime.

**Resolution Plan**:
1. Verify auto regime selection logic in forward model
2. Test threshold behavior with various defocus amounts
3. Enable automatic selection in implementation
4. Remove skip decorator once verified

---

## 3. Golden Outputs Not Generated (9 tests)

**Status**: ⏸️ Baseline generation needed
**Timeline**: Generate golden outputs when regression test infrastructure is ready
**Priority**: Low (tests are implementation-ready)

**Location**: `tests/regression/test_instrument_outputs.py`

These regression tests are fully implemented but skip execution because the golden output files don't exist yet. They need to be run once to generate baseline outputs for future comparison.

### Test List

| Test | Line | Golden File |
|------|------|-------------|
| `test_microscope_forward_brightfield` | 497-499 | `microscope_forward_brightfield.npz` |
| `test_microscope_forward_with_noise` | 511-513 | `microscope_forward_with_noise.npz` |
| `test_microscope_forward_darkfield` | 527-529 | `microscope_forward_darkfield.npz` |
| `test_microscope_forward_phase_contrast` | 543-545 | `microscope_forward_phase_contrast.npz` |
| `test_microscope_forward_with_aperture_center` | 579-581 | `microscope_forward_aperture_center.npz` |
| `test_microscope_forward_with_illumination_center` | 593-595 | `microscope_forward_illumination_center.npz` |
| `test_drone_forward_default` | 639-641 | `drone_forward_default.npz` |
| `test_drone_forward_with_noise` | 653-655 | `drone_forward_with_noise.npz` |
| `test_camera_forward_far_field_regression` | 672-674 | `camera_forward_far_field.npz` |

**Skip Pattern**:
```python
golden_path = GOLDEN_DIR / "filename.npz"
if not golden_path.exists():
    pytest.skip("Golden output not generated yet")
```

### Generation Instructions

To generate golden outputs:

1. **Ensure instruments are working correctly**:
   ```bash
   uv run pytest tests/unit/core/ -v
   ```

2. **Remove the skip conditions temporarily** (or use a flag to force generation)

3. **Run regression tests once to generate baselines**:
   ```bash
   uv run pytest tests/regression/test_instrument_outputs.py -v
   ```

4. **Verify golden outputs were created**:
   ```bash
   ls -lh tests/regression/golden_outputs/
   ```

5. **Commit golden outputs to git**:
   ```bash
   git add tests/regression/golden_outputs/*.npz
   git commit -m "test: add golden outputs for regression tests"
   ```

**Note**: Golden outputs should be generated on a stable, known-good implementation to serve as the reference baseline for future regression testing.

---

## 4. Known Bugs (0 tests remaining)

**Status**: ✅ All resolved
**Priority**: N/A

### 4.1 Camera Far-Field Mode Bug - RESOLVED

**Location**: `tests/regression/test_instrument_outputs.py`

**Affected Tests**:
- `test_camera_forward_far_field` (line 445-451)
- `test_camera_forward_far_field_regression` (line 667-674)

**Resolution**:
1. Original bug (distance parameter to FraunhoferPropagator) was fixed during FourFSystem consolidation
2. Aperture calculation now uses f/# instead of hardcoded 25%
3. Skip decorators removed, tests now pass

**Fix Details**:
- Modified `_create_pupils()` in `prism/core/instruments/camera.py` to calculate aperture from f-number
- NA_effective = 1 / (2 * f_number)
- k_cutoff = NA_effective / wavelength
- Aperture radius properly scaled to Nyquist frequency

**Related Files**:
- `prism/core/instruments/camera.py` - Camera implementation (fixed)
- `prism/core/propagators/fraunhofer.py` - Fraunhofer propagator

---

## 5. Xfailed Tests (1 test)

**Status**: ❌ Known failure - dtype handling needed
**Timeline**: Fix before v1.0 release
**Priority**: High

### 5.1 AMP + SSIM Type Mismatch (1 test)

**Location**: `tests/integration/test_trainer_integration.py:649-660`

**Test**: `TestPRISMTrainer.test_training_with_amp_on_cuda`

**Issue**: Automatic Mixed Precision (AMP) causes dtype mismatches in SSIM loss computation during the trainer's initialization phase. This is a known issue with AMP and convolution operations used in SSIM calculation.

**Xfail Marker**:
```python
@pytest.mark.xfail(
    reason="AMP with SSIM has known type mismatch issues in metrics computation",
    strict=False,
)
```

**Error Details**:
- SSIM uses convolutional operations that may not properly handle mixed precision
- Dtype mismatch occurs between float16 (AMP) and float32 (SSIM Gaussian kernel)
- Issue appears during initialization, not during training loop

**Expected Behavior**: Training with AMP should work seamlessly with SSIM loss function, automatically handling dtype conversions.

**Resolution Plan**:
1. Add explicit dtype handling in SSIM loss function (`prism/models/losses.py`)
2. Ensure Gaussian kernel is cast to input dtype before convolution
3. Add `torch.cuda.amp.autocast()` context manager guards where needed
4. Alternative: Use `torch.nn.functional.conv2d` with explicit dtype casting
5. Add comprehensive AMP + SSIM unit tests
6. Remove xfail marker once fixed

**Code Location**: `prism/models/losses.py` - SSIM loss implementation

**Related Tests**: All SSIM tests should be checked for AMP compatibility:
- `tests/unit/models/test_losses.py` - SSIM unit tests
- `tests/integration/test_trainer_integration.py` - Trainer integration tests

---

## 6. Conditional Skips (~30 tests)

**Status**: ✅ Environment-dependent (expected behavior)
**Priority**: N/A

These tests are conditionally skipped based on runtime environment, not due to implementation issues.

### 6.1 CUDA Availability (~20 tests)

Tests that require GPU/CUDA and skip when not available:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
```

**Locations**:
- `tests/unit/models/test_losses.py` - Multiple GPU-specific performance tests
- `tests/unit/core/test_grid.py` - GPU tensor operations
- `tests/integration/test_inference_optimization.py` - Speed benchmarks requiring CUDA
- `tests/unit/test_microscope_scanning_illum.py` - GPU-accelerated tests

**Expected Behavior**: These tests run on GPU CI runners and skip on CPU-only machines.

### 6.2 Notebook Execution Tests (~5 tests)

**Location**: `tests/e2e/test_notebooks.py`

Tests that skip when notebook files are not found:
```python
if not notebook_path.exists():
    pytest.skip(f"Notebook not found: {notebook_path}")
```

**Expected Behavior**: Notebooks may not be present in minimal installations or CI environments.

### 6.3 Example File Tests (~5 tests)

**Locations**:
- `tests/e2e/test_examples_python_api.py`
- `tests/e2e/test_examples_patterns.py`
- `tests/e2e/test_examples_demo.py`

Tests that skip when example scripts or modules are not found:
```python
if not example_path.exists():
    pytest.skip(f"Example not found: {example_path}")
```

**Expected Behavior**: Examples may be excluded from some distributions or test environments.

---

## 7. Warnings Configuration

**Status**: ✅ Properly configured
**Location**: `pyproject.toml:149-164`

### Suppressed Warnings

The test suite uses `filterwarnings` to manage warning levels:

#### Strict Mode (Error on Warnings)
```python
"error::DeprecationWarning:prism.*",  # Treat our deprecations as errors
"error::UserWarning:prism.*",          # Treat our warnings as errors
```

#### Ignored Third-Party Warnings
```python
"ignore::DeprecationWarning:torch.*",     # PyTorch deprecations
"ignore::UserWarning:matplotlib.*",       # Matplotlib warnings
"ignore::FutureWarning:numpy.*",          # NumPy future changes
```

#### Temporary Ignores
```python
"ignore:Parameters 'use_leaky'.*:DeprecationWarning",                    # Legacy parameter
"ignore:star_sample and fermat_sample are deprecated.*:DeprecationWarning",  # Old API
"ignore:torch.compile not available.*:UserWarning",                      # Optional feature
"ignore:Conv-BN fusion skipped.*:UserWarning",                           # Performance hint
```

### Rationale

- **Strict for PRISM code**: Ensures we fix deprecations before they become breaking changes
- **Lenient for dependencies**: Don't fail on third-party library warnings we can't control
- **Temporary ignores**: Allow time to migrate away from deprecated APIs without breaking CI

---

## 8. Resolution Timeline

### Immediate (v0.6.x)
- [x] Investigate Fresnel energy conservation issue ✅ Fixed
- [x] Document all skipped/xfailed tests (this document)

### Short-term (v0.7.0 - Phase 3)
- [ ] Complete Four-F System consolidation
- [x] Fix Camera far-field mode bug ✅ Fixed
- [ ] Enable Phase 3 integration tests
- [ ] Add auto regime selection for forward models

### Medium-term (v0.8.0)
- [ ] Fix AMP + SSIM dtype mismatch
- [ ] Generate golden outputs for regression tests
- [ ] Add comprehensive AMP compatibility testing

### Long-term (v1.0)
- [ ] All tests enabled and passing
- [ ] No xfail markers remaining
- [ ] Complete regression test coverage with golden outputs
- [ ] Resolved all physics validation issues

---

## 9. Test Suite Health Metrics

**Current Status** (as of 2025-12-01):

```
Total Tests: ~250+
Passing: ~220+
Skipped (deferred): 26
Skipped (conditional): ~30
Xfailed: 1
---
Success Rate: ~88% (excluding conditional skips)
```

**Quality Indicators**:
- ✅ All skipped tests are documented with reasons
- ✅ Clear resolution timeline for each category
- ✅ No unknown/undocumented test failures
- ⚠️ Some physics validation issues require investigation
- ⚠️ One xfailed test needs fixing before v1.0

---

## 10. How to Use This Document

### For Developers

**When adding a new skip/xfail**:
1. Update the relevant section in this document
2. Provide clear reason and expected resolution
3. Link to related issues/tasks if applicable
4. Update the summary table

**When fixing a skipped test**:
1. Remove the skip/xfail decorator
2. Verify test passes
3. Update this document to remove the entry
4. Update test suite health metrics

### For CI/CD

**Skip conditions that should NOT block merge**:
- Phase 3 integration tests (properly deferred)
- Conditional skips (CUDA, notebooks, examples)
- Golden output generation (baseline not yet created)

**Issues that SHOULD block release**:
- Xfailed tests (known failures)
- Physics validation issues (correctness concerns)
- Known bugs affecting user-facing features

### For QA

**Test priorities**:
1. **High**: Physics validation, xfailed tests (correctness issues)
2. **Medium**: Phase 3 integration, known bugs (functionality gaps)
3. **Low**: Golden outputs, conditional skips (infrastructure)

---

## References

- **Phase 3 Plan**: `docs/IMPROVEMENT_PLAN_PHASE3.md`
- **Regression Test Guide**: `tests/regression/TASK_0.1_COMPLETION_SUMMARY.md`
- **Testing Best Practices**: `AI_ASSISTANT_GUIDE.md#testing`
- **CI Configuration**: `.github/workflows/tests.yml`

---

**Document Maintenance**: This document should be updated whenever:
- New tests are skipped or marked as xfail
- Skipped tests are re-enabled
- Resolution timelines change
- New test categories emerge
- Test suite health metrics change significantly

**Owners**: Maintained by the PRISM development team. Contact via GitHub issues for questions or updates.

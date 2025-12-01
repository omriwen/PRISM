# Four-F System API Documentation Index

This index provides quick access to all documentation related to the Four-F System Consolidation.

**Last Updated**: 2025-11-28
**Status**: Complete

---

## Core API Documentation

### 1. FourFSystem Base Class
**File**: [four_f_base.md](/home/omri/PRISM/docs/api/four_f_base.md)

Abstract base class for all 4f optical system instruments. Provides unified implementation of forward modeling, k-space propagation, aperture mask generation, and noise modeling.

**Key Topics**:
- Class overview and hierarchy
- Constructor parameters
- Abstract methods (\_create_pupils, resolution_limit)
- Forward propagation method
- K-space propagation methods
- Aperture mask generation
- PSF computation
- Subclassing guide with examples

**Target Audience**: Developers creating custom instruments

---

### 2. FourFForwardModel
**File**: [four_f_forward.md](/home/omri/PRISM/docs/api/four_f_forward.md)

Core 4f optical system forward model implementing the unified propagation chain: pad → FFT → pupils → IFFT → crop.

**Key Topics**:
- Physical model and equations
- Constructor and configuration
- Forward propagation with pupils
- Batch dimension handling
- FFT padding for anti-aliasing
- Complex field vs intensity output
- Performance considerations
- GPU acceleration

**Target Audience**: Users needing direct control over forward model

---

### 3. ApertureMaskGenerator
**File**: [aperture_masks.md](/home/omri/PRISM/docs/api/aperture_masks.md)

Unified aperture and pupil mask generator supporting multiple geometries and radius specifications.

**Key Topics**:
- Supported mask types (circular, annular, hexagonal, phase rings, etc.)
- Radius specification modes (NA, physical, pixels)
- Coordinate conventions
- Frequency cutoff conversion
- Usage examples for each mask type
- Integration with instruments

**Target Audience**: Users creating custom pupils or apertures

---

### 4. DetectorNoiseModel
**File**: [detector_noise.md](/home/omri/PRISM/docs/api/detector_noise.md)

Realistic detector noise model including shot noise, read noise, and dark current.

**Key Topics**:
- SNR-based vs component-based noise modes
- Noise component details (shot, read, dark current)
- Parameter configuration for different detector types
- Enable/disable functionality
- Integration with instruments
- Physical interpretation and typical values

**Target Audience**: Users simulating realistic imaging conditions

---

## Migration and Usage Guides

### 5. Four-F Migration Guide
**File**: [four_f_migration_guide.md](/home/omri/PRISM/docs/api/four_f_migration_guide.md)

Comprehensive guide for migrating from pre-Four-F consolidation code.

**Key Topics**:
- What changed in the consolidation
- Backward compatibility (100%)
- New features and capabilities
- Breaking changes (none)
- Advanced usage patterns
- Performance impact
- Testing and validation
- FAQ

**Target Audience**: All users (existing and new)

---

## Quick Navigation

### By Use Case

| Use Case | Recommended Documentation |
|----------|---------------------------|
| **Using existing instruments** | [four_f_migration_guide.md](four_f_migration_guide.md) - "Existing Code" section |
| **Creating custom instruments** | [four_f_base.md](four_f_base.md) - "Subclassing FourFSystem" section |
| **Generating custom pupils** | [aperture_masks.md](aperture_masks.md) - "Usage Examples" section |
| **Adding realistic noise** | [detector_noise.md](detector_noise.md) - "Usage Examples" section |
| **Understanding the forward model** | [four_f_forward.md](four_f_forward.md) - "Overview" and "Implementation Details" |
| **Performance optimization** | [four_f_forward.md](four_f_forward.md) - "Performance Considerations" |

### By User Type

| User Type | Start Here |
|-----------|------------|
| **End User** (using existing instruments) | [four_f_migration_guide.md](four_f_migration_guide.md) |
| **Developer** (extending instruments) | [four_f_base.md](four_f_base.md) |
| **Researcher** (understanding physics) | [four_f_forward.md](four_f_forward.md) |
| **Tool Builder** (using components directly) | [aperture_masks.md](aperture_masks.md), [detector_noise.md](detector_noise.md) |

### By Component

| Component | Documentation | Source Code |
|-----------|---------------|-------------|
| `FourFSystem` | [four_f_base.md](four_f_base.md) | `/home/omri/PRISM/prism/core/instruments/four_f_base.py` |
| `FourFForwardModel` | [four_f_forward.md](four_f_forward.md) | `/home/omri/PRISM/prism/core/optics/four_f_forward.py` |
| `ApertureMaskGenerator` | [aperture_masks.md](aperture_masks.md) | `/home/omri/PRISM/prism/core/optics/aperture_masks.py` |
| `DetectorNoiseModel` | [detector_noise.md](detector_noise.md) | `/home/omri/PRISM/prism/core/optics/detector_noise.py` |

---

## Related Documentation

### Implementation Plan
**File**: `/home/omri/PRISM/docs/plans/four-f-system-consolidation.md`

Detailed plan for the Four-F System Consolidation including:
- Problem statement and motivation
- Architecture design
- Implementation phases
- Testing strategy
- Performance benchmarks
- Change history

### Architecture Documentation
**Files**:
- `/home/omri/PRISM/docs/architecture/forward-model-physics.md` - Forward model equations
- `/home/omri/PRISM/docs/architecture/propagator-usage.md` - Propagator usage guide

### Existing Instrument Documentation
**Files**:
- [instruments.md](instruments.md) - Instrument module overview
- [microscope.md](microscope.md) - Microscope class (now uses FourFSystem)
- [telescope.md](telescope.md) - Telescope class (now uses FourFSystem)
- [camera.md](camera.md) - Camera class (now uses FourFSystem)

---

## Documentation Statistics

| Documentation File | Lines | Size | Created |
|-------------------|-------|------|---------|
| four_f_base.md | ~700 | ~45 KB | 2025-11-28 |
| four_f_forward.md | ~550 | ~35 KB | 2025-11-28 |
| aperture_masks.md | ~600 | ~38 KB | 2025-11-28 |
| detector_noise.md | ~500 | ~32 KB | 2025-11-28 |
| four_f_migration_guide.md | ~450 | ~28 KB | 2025-11-28 |
| **Total** | **~2,800** | **~178 KB** | - |

---

## Code Examples Coverage

Each documentation file includes extensive code examples:

| Documentation | Example Categories | Total Examples |
|---------------|-------------------|----------------|
| four_f_base.md | Subclassing, Usage, Integration | 8+ |
| four_f_forward.md | Basic, Batch, Padding, Shapes | 12+ |
| aperture_masks.md | All mask types, Use cases | 15+ |
| detector_noise.md | SNR, Component, Integration | 10+ |
| four_f_migration_guide.md | Before/After, Advanced | 12+ |
| **Total** | - | **57+** |

All examples are:
- Executable (can be copy-pasted and run)
- Tested (verified against actual implementation)
- Documented (with explanations)
- Diverse (covering common and advanced scenarios)

---

## Quality Metrics

### Completeness
- [x] All new classes documented
- [x] All public methods documented
- [x] All parameters documented with types
- [x] Return values documented
- [x] Examples provided for each major feature
- [x] Cross-references included

### Accuracy
- [x] Extracted directly from source code docstrings
- [x] Verified against implementation
- [x] Tested examples
- [x] Reviewed for consistency

### Usability
- [x] Clear structure and organization
- [x] Progressive detail (overview → details → examples)
- [x] Multiple navigation paths (by use case, user type, component)
- [x] Consistent formatting
- [x] Helpful cross-references

---

## Feedback and Contributions

Found an issue or have a suggestion? Please:
1. Check if it's already documented in the FAQ
2. Review the migration guide for backward compatibility
3. File an issue with details and code examples
4. Submit a pull request with documentation improvements

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-28 | Initial documentation release for Four-F System Consolidation |

---

**Last Updated**: 2025-11-28
**Maintained By**: PRISM Development Team

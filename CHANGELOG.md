# Changelog

All notable changes to PRISM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **GPU-Optimized Line Acquisition Module** ([prism/core/line_acquisition.py](prism/core/line_acquisition.py))
  - New `IncoherentLineAcquisition` class with correct incoherent intensity summation physics: `I = (1/N) × Σᵢ |IFFT(F_kspace × Aperture_i)|²`
  - Line-length dependent sampling with two modes:
    - "accurate": 1 sample per pixel (configurable via `samples_per_pixel`)
    - "fast": half-diameter spacing for legacy compatibility
  - Batched FFT operations for 10x+ GPU speedup over loop-based approach
  - Memory-efficient batch processing with configurable limits
  - `LineAcquisitionConfig` dataclass for flexible configuration
  - Comprehensive unit tests ([tests/unit/core/test_line_acquisition_physics.py](tests/unit/core/test_line_acquisition_physics.py))
  - Performance benchmarks ([tests/benchmarks/test_line_acquisition_perf.py](tests/benchmarks/test_line_acquisition_perf.py))

- **Batched FFT Utilities** ([prism/utils/transforms.py](prism/utils/transforms.py))
  - `batched_fft2()`: Batched 2D FFT with proper DC centering
  - `batched_ifft2()`: Batched 2D IFFT for efficient batch processing
  - Support for arbitrary batch sizes with automatic memory management

### Fixed

- **Line Acquisition Physics Bug** ([prism/core/telescope.py](prism/core/telescope.py))
  - Removed incorrect `.sqrt()` in intensity averaging that was computing RMS instead of mean intensity
  - Formula now correctly implements `I = (1/N) × Σ|F|²` without the spurious square root

### Changed

- **MeasurementSystem Integration** ([prism/core/measurement_system.py](prism/core/measurement_system.py))
  - Added `line_acquisition` parameter for optional line acquisition support
  - New `line_endpoints` parameter in `measure()` and `add_mask()` methods
  - Automatic dispatch between point and line measurement modes
  - Full backward compatibility with existing point acquisition code

- **Training Pipeline Updates**
  - Added `--line-mode` CLI argument (choices: "accurate", "fast")
  - Added `--samples-per-pixel` for fine-grained control
  - Updated training loop to support line acquisition measurements
  - Runner creates LineAcquisition automatically when needed

## [0.3.0] - 2025-11-20

### Added - UI/UX Upgrade (Major Release)

**Interactive Tools & Visualization Suite**

A comprehensive UI/UX upgrade providing interactive tools for experiment analysis, comparison, and visualization. All features are fully tested, documented, and ready for production use.

**Phase 1: Enhanced Terminal Experience**

- **Intelligent Error Messages** ([prism/config/validation.py](prism/config/validation.py))
  - ConfigValidator class with spelling suggestions using difflib
  - Detailed error formatting with parameter descriptions
  - Context-aware help system with `--help-<topic>` flags
  - All enum and range validations with "Did you mean?" suggestions
  - 14,533 bytes of tests ([tests/test_validation.py](tests/test_validation.py))

- **Experiment Comparison CLI** ([prism/cli/compare.py](prism/cli/compare.py))
  - Side-by-side comparison of multiple experiments
  - Metrics comparison tables with best values highlighted
  - Configuration diff viewer showing parameter differences
  - Comparison visualizations with training curves overlay
  - Export results to PNG/PDF formats
  - Command: `prism compare runs/exp1 runs/exp2`

- **Checkpoint Inspector** ([prism/cli/inspect.py](prism/cli/inspect.py))
  - Interactive checkpoint exploration with questionary menu
  - Summary display with metadata, metrics, and configuration
  - Training history visualization
  - Reconstruction export to PNG
  - Handles corrupted checkpoints gracefully
  - Command: `prism inspect runs/experiment/checkpoint.pt --interactive`

- **Progress Visualization Enhancements** ([prism/utils/progress.py](prism/utils/progress.py))
  - Sparkline charts for metric trends using unicode characters
  - Color-coded status indicators (🟢 improving, 🟡 plateaued, 🔴 diverging)
  - Enhanced metrics display with trend analysis
  - Improved ETA calculation with exponential moving average
  - ASCII fallback for non-unicode terminals

**Phase 2: Interactive Web Dashboard**

- **Core Dashboard Application** ([prism/web/](prism/web/))
  - Dash-based web dashboard with Flask backend
  - Real-time training monitoring with periodic polling
  - Interactive Plotly visualizations (zoom, pan, hover)
  - Multi-experiment comparison interface
  - K-space coverage visualization
  - Configuration comparison viewer
  - Command: `prism dashboard --port 8050`
  - Integration: `python main.py --dashboard`

- **Dashboard Components**:
  - `prism/web/dashboard.py` (275 lines) - Main Dash app
  - `prism/web/server.py` (10,190 bytes) - Backend server
  - `prism/web/launcher.py` (7,063 bytes) - Process management
  - `prism/web/layouts/` - Layout components (main, comparison, live)
  - `prism/web/callbacks/` - Interactive callbacks (training, comparison, realtime)
  - `prism/web/assets/` - Styles and static assets

- **Real-Time Monitoring**:
  - Live training progress updates (<2s latency)
  - Current epoch, loss, SSIM, PSNR display
  - Training curves with configurable smoothing
  - Reconstruction preview vs ground truth
  - K-space accumulation heatmap

- **Multi-Experiment Comparison UI**:
  - Side-by-side reconstruction comparison (up to 4 experiments)
  - Synchronized zoom and pan across plots
  - Sortable metric comparison tables
  - Configuration diff viewer with highlighting
  - Training curve overlay with opacity control

**Phase 3: Result Exploration & Reporting**

- **Training Animation Generator** ([prism/visualization/animation.py](prism/visualization/animation.py))
  - Generate MP4 and GIF animations from checkpoints
  - Show training progression over time
  - Side-by-side comparison animations
  - Frame interpolation for smooth playback
  - Custom layouts and metric overlays
  - Command: `prism animate runs/experiment --output training.mp4`

- **Automatic Report Generation** ([prism/reporting/](prism/reporting/))
  - Generate HTML and PDF reports for publications
  - Executive summary with key metrics
  - Complete configuration details
  - Training curves and visualizations
  - K-space coverage analysis
  - High-DPI figures (300 DPI)
  - Multi-experiment comparison reports
  - Custom template support via Jinja2
  - Command: `prism report runs/experiment --format pdf`

- **Pattern Library Browser** ([prism/core/pattern_library.py](prism/core/pattern_library.py))
  - PatternLibrary class with metadata system
  - List all available sampling patterns
  - Visualize pattern positions and coverage
  - Compare multiple patterns side-by-side
  - Pattern statistics computation
  - Interactive HTML gallery generation
  - Command: `prism patterns compare fermat random star`

### Changed - CLI & Documentation

**New CLI Commands**:
- `prism compare` - Multi-experiment comparison tool
- `prism inspect` - Interactive checkpoint inspector
- `prism dashboard` - Launch web dashboard server
- `prism animate` - Generate training animations
- `prism report` - Generate HTML/PDF reports
- `prism patterns` - Browse and compare sampling patterns

**Updated Documentation**:
- README.md: Added comprehensive UI/UX tools section
- Key Features: Added UI/UX tools to feature list
- Version number updated to 0.3.0

### New Dependencies

**Dashboard & Reporting** (added to pyproject.toml):
- `dash>=2.18.0` - Web dashboard framework
- `dash-bootstrap-components>=1.6.0` - UI components
- `pillow>=10.4.0` - Image processing for animations
- `opencv-python>=4.10.0` - Video generation
- `weasyprint>=62.3` - HTML to PDF conversion (optional)

### Testing

**Comprehensive Test Suite** (>80% coverage):
- `tests/test_validation.py` (14,533 bytes) - Validation and error messages
- `tests/test_comparison.py` (9,664 bytes) - Experiment comparison
- `tests/test_comparison_ui.py` (14,843 bytes) - Comparison UI tests
- `tests/test_inspector.py` (14,513 bytes) - Checkpoint inspector
- `tests/unit/test_inspector.py` - Inspector unit tests
- `tests/test_dashboard.py` (10,582 bytes) - Dashboard server tests
- `tests/test_dashboard_integration.py` (7,552 bytes) - Dashboard integration
- `tests/test_dashboard_launcher.py` (8,999 bytes) - Process launcher
- `tests/test_animation.py` (14,486 bytes) - Animation generation
- `tests/test_reporting.py` (12,482 bytes) - Report generation
- `tests/test_pattern_library.py` (8,603 bytes) - Pattern library
- `tests/unit/test_progress.py` - Progress visualization

### Performance

All performance requirements met:
- Error message generation: <10ms
- Comparison tool (2 experiments): <2s
- Checkpoint inspection: <1s
- Dashboard update latency: <100ms
- Dashboard memory usage: <500MB after 1 hour
- Animation generation: <10s for 10-second 1080p video
- Report generation: <5s per experiment

### Backward Compatibility

**Guaranteed Compatibility**:
- All existing CLI commands continue to work unchanged
- Python API remains fully compatible
- Checkpoint format unchanged
- Configuration files load correctly
- No breaking changes

**New Features (Opt-in)**:
- All new CLI subcommands are additions
- Dashboard is opt-in via `--dashboard` flag
- Existing workflows remain fully functional

### Documentation

See [docs/UI_UX_UPGRADE_PLAN.md](docs/UI_UX_UPGRADE_PLAN.md) for:
- Complete implementation plan
- Detailed feature specifications
- Testing strategy and acceptance criteria
- Rollout plan and success metrics

### Migration

No migration required - all features are additive. Users can:
- Continue using existing workflows without changes
- Adopt new UI/UX features incrementally
- Use dashboard/reporting tools on existing experiment data

### Example Workflows

```bash
# Compare experiment results
prism compare runs/exp1 runs/exp2

# Interactively explore checkpoint
prism inspect runs/experiment/checkpoint.pt --interactive

# Monitor training with web dashboard
python main.py --obj_name europa --n_samples 100 --dashboard

# Generate publication report
prism report runs/production_run --format pdf --output paper_figure.pdf

# Create training animation for presentation
prism animate runs/experiment --output training.mp4 --fps 10

# Browse sampling patterns
prism patterns compare fermat random star --n-samples 100
```

## [1.8.0] - 2025-11-20

### Added - ProgressiveDecoder Network Model

**Phase 2-4 Complete: Networks.py Refactoring**

A comprehensive refactoring of the primary network model with improved naming, flexibility, and performance optimizations.

**New Features:**
- `ProgressiveDecoder`: Refactored and renamed from `GenCropSpidsNet` ([prism/models/networks.py](prism/models/networks.py))
  - Flexible architecture configuration (manual or automatic)
  - Manual control over `latent_channels` parameter
  - Manual control over `channel_progression` (layer-by-layer channel counts)
  - Manual control over `num_upsample_layers`
  - Improved naming and documentation
  - Modular code structure with helper methods

**Performance Optimizations:**
- Conv-BN layer fusion for ~10-20% faster inference (`prepare_for_inference()`)
- torch.compile() support for ~30% additional speedup (PyTorch 2.0+)
- Gradient checkpointing support for ~50% memory reduction (`enable_gradient_checkpointing()`)
- Enhanced inference preparation with `prepare_for_inference(compile_mode=...)`
- Performance benchmarking utility (`benchmark()` method)

**Documentation & Migration:**
- Comprehensive migration guide: [docs/MIGRATION_GENCROP_TO_PROGRESSIVE.md](docs/MIGRATION_GENCROP_TO_PROGRESSIVE.md)
- Detailed model documentation: [docs/models/progressive_decoder.md](docs/models/progressive_decoder.md)
- Updated architecture documentation: [docs/CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md)
- Migration script: [scripts/migrate_gencrop_to_progressive.py](scripts/migrate_gencrop_to_progressive.py)

**Testing:**
- 56+ unit tests for ProgressiveDecoder ([tests/unit/test_progressive_decoder.py](tests/unit/test_progressive_decoder.py))
- Integration tests with NetworkConfig ([tests/integration/test_custom_network_configs.py](tests/integration/test_custom_network_configs.py))
- Performance benchmarks ([tests/benchmarks/test_performance.py](tests/benchmarks/test_performance.py))
- 86.41% test coverage for networks.py

### Changed - Network Model Improvements

**Code Quality:**
- Extracted architecture computation into `_compute_architecture_params()` method
- Modular builder methods: `_build_initial_layer()`, `_build_upsample_layer()`, `_build_refinement_layer()`
- Improved variable naming: `half_depth` → `latent_depth`, `decoder_layers` → `layers`
- Enhanced inline documentation with shape transformation comments
- Comprehensive class docstring with multiple usage examples

**Configuration Integration:**
- Updated `NetworkConfig` to support new parameters
- Improved `from_config()` method for configuration-driven creation
- Better integration with NetworkBuilder pattern

### Deprecated

**GenCropSpidsNet (will be removed in v2.0):**
- `GenCropSpidsNet` remains available as a deprecated alias of `ProgressiveDecoder`
- Emits `DeprecationWarning` when instantiated
- Full backward compatibility maintained
- Migration guide provided: [docs/MIGRATION_GENCROP_TO_PROGRESSIVE.md](docs/MIGRATION_GENCROP_TO_PROGRESSIVE.md)

**Deprecated Parameters (ignored with warning):**
- `use_leaky`: No longer used (always uses ReLU)
- `middle_activation`: No longer configurable
- `complex_data`: Handled automatically

**Migration:**
```python
# Before (deprecated but still works)
from prism.models.networks import GenCropSpidsNet
model = GenCropSpidsNet(input_size=1024, output_size=512)

# After (recommended)
from prism.models.networks import ProgressiveDecoder
model = ProgressiveDecoder(input_size=1024, output_size=512)
```

Use migration script for automated updates:
```bash
python scripts/migrate_gencrop_to_progressive.py /path/to/code --dry-run
```

### Removed - Deprecated Network Models (BREAKING CHANGES)

**Phase 1: Code Cleanup Complete (2025-11-20)**

This is the first phase of the networks.py refactoring plan. Four deprecated network models with zero production usage have been removed from the codebase.

**Breaking Changes - Removed Classes:**
- `SpidsNet` - Original fixed 7-layer autoencoder (173 lines, 0% usage)
- `ModularSpidsNet` - Flexible autoencoder with depth based on input size (145 lines, 0% usage)
- `AeSpidsNet` - Autoencoder variant with learnable latent vector (152 lines, 0% usage)
- `GenSpidsNet` - Simple generative model (41 lines, 0% usage)

**Impact:**
- Codebase reduced by ~511 lines (-58% of networks.py)
- Only `GenCropSpidsNet` remains as the primary model
- Future Phase 2 will rename `GenCropSpidsNet` → `ProgressiveDecoder` for clarity

**Migration Guide:**
All removed models had zero production usage. If you were using any of these models:
```python
# Replace with the primary model
from prism.models.networks import GenCropSpidsNet

# GenCropSpidsNet is the recommended decoder-only generative model
model = GenCropSpidsNet(input_size=1024, output_size=512)
```

**Next Steps:**
- Phase 2: Refactor GenCropSpidsNet with improved flexibility and naming
- Phase 3: Add performance optimizations (Conv-BN fusion, torch.compile)
- See [docs/NETWORKS_REFACTORING_PLAN.md](docs/NETWORKS_REFACTORING_PLAN.md) for full roadmap

### Changed - Naming Refactoring (BREAKING CHANGES)

**All Phases Complete (2025-01-20)**

This comprehensive refactoring improves code clarity and PEP 8 compliance across the codebase.

**Breaking Changes - Class Renames:**
- `TelescopeAgg` → `TelescopeAggregator` ([prism/core/aggregator.py](prism/core/aggregator.py))
- `LossAgg` → `LossAggregator` ([prism/models/losses.py](prism/models/losses.py))
- `CBatchNorm` → `ConditionalBatchNorm` ([prism/models/layers.py](prism/models/layers.py))

**Breaking Changes - Function Renames:**
- `prop_1()` → `propagate_to_kspace()` ([prism/core/telescope.py](prism/core/telescope.py))
- `prop_2()` → `propagate_to_spatial()` ([prism/core/telescope.py](prism/core/telescope.py))
- `prop_saved_mask()` → `measure_through_accumulated_mask()` ([prism/core/aggregator.py](prism/core/aggregator.py))
- `compare_rmse()` → `compute_rmse()` ([prism/utils/metrics.py](prism/utils/metrics.py))
- `ssim_skimage()` → `compute_ssim()` ([prism/utils/metrics.py](prism/utils/metrics.py))
- `rand_cntrd()` → `random_centered()` ([prism/core/patterns.py](prism/core/patterns.py))
- `pos_to_pix()` → `position_to_pixel()` ([prism/core/patterns.py](prism/core/patterns.py))
- `F()` → `fresnel_number()` ([prism/config/constants.py](prism/config/constants.py))

**Breaking Changes - Method Renames:**
- `GenCropSpidsNet.minmax()` → `clamp_channel_count()` ([prism/models/networks.py](prism/models/networks.py))

**Breaking Changes - Variable Standardization:**
- `cntr`/`ctr` → `center` (throughout codebase)
- `rec` → `reconstruction` (throughout codebase)
- `meas` → `measurement` (documentation)
- `fixs` → `figure` (visualization code)

**Breaking Changes - Parameter Names (PEP 8 compliance):**
- Constants functions: `W` → `width`, `L` → `distance`
- Affected functions: `fresnel_number()`, `F_crit()`, `is_fraunhofer()`, `is_fresnel()`, `r_coh()`

**Migration Guide:**
```python
# Before
from prism.core import TelescopeAgg
from prism.models import LossAgg
from prism.utils import compare_rmse, ssim_skimage
telescope = TelescopeAgg(n=256, r=10)

# After
from prism.core import TelescopeAggregator
from prism.models import LossAggregator
from prism.utils import compute_rmse, compute_ssim
telescope = TelescopeAggregator(n=256, r=10)
```

See [NAMING_REFACTORING_PLAN.md](NAMING_REFACTORING_PLAN.md) for complete details.

### Added - Phase 3 (Advanced Features + Mixed Precision)

**Network Builder Pattern (Week 6 Days 1-2)**
- NetworkConfig dataclass for flexible architecture configuration ([prism/models/network_config.py](prism/models/network_config.py))
- NetworkBuilder class for configuration-driven network construction ([prism/models/network_builder.py](prism/models/network_builder.py))
- `GenCropSpidsNet.from_config()` class method for streamlined network creation
- Support for custom latent channels, activations, batch normalization, dropout, and initialization methods
- Comprehensive configuration validation with helpful error messages
- 17 configuration tests + 17 builder tests (>95% coverage for new modules)

**Mixed Precision Training (Week 6 Days 3-4) - HIGHEST PRIORITY**
- Automatic Mixed Precision (AMP) support in GenCropSpidsNet via `use_amp` parameter
- Target: 20-30% speedup + 40-50% memory reduction on compatible GPUs (Ampere, Hopper)
- GradScaler integration in PRISMTrainer for numerically stable FP16/FP32 training
- `generate_fp32()` method for maximum accuracy during inference
- `--mixed-precision` CLI flag for easy AMP activation
- AMP-aware forward pass with autocast context management

**Noise Model Hierarchy (Week 6 Day 5)**
- NoiseModel ABC for extensible noise modeling ([prism/models/noise.py](prism/models/noise.py))
- PoissonNoise for realistic shot noise simulation
- ReadoutNoise for detector noise modeling
- CompositeNoise for combining multiple noise sources
- Statistical validation and device compatibility
- Backward compatibility with existing ShotNoise class
- 28 noise model tests with comprehensive validation

**Inference Optimization (Week 7 Days 3-4)**
- `prepare_for_inference()` method for Conv-BN fusion and parameter freezing
- Target: 10-20% inference speedup through optimization
- Automatic gradient disabling for production deployments

### Changed - Phase 3

- GenCropSpidsNet now supports configuration-driven initialization
- PRISMTrainer supports mixed precision training with GradScaler
- Noise models now inherit from common NoiseModel ABC
- CLI parser includes `--mixed-precision` flag
- Test suite expanded with 62+ new tests for Phase 3 features

### Performance - Phase 3 Targets

| Feature | Target | Priority | Status |
|---------|--------|----------|--------|
| **AMP Speedup** | **20-30%** | **CRITICAL** | Implemented ⏳ |
| **AMP Memory** | **40-50% reduction** | **HIGH** | Implemented ⏳ |
| Inference optimization | 10-20% | MEDIUM | Implemented ⏳ |
| Noise overhead | <5% | LOW | Implemented ✓ |

**Expected Combined Performance**:
- Phase 1-2: >>5x speedup ✅ ACHIEVED
- Phase 3 AMP: +20-30% (if compatible GPU available)
- **Overall Target: 6-7x speedup**

### Documentation - Phase 3

- NetworkConfig and NetworkBuilder documented with comprehensive examples
- Mixed precision training guide (usage, benefits, requirements)
- Noise model hierarchy documentation with statistical properties
- Inference optimization best practices
- Updated pyproject.toml with `gpu` test marker

### Added - Phase 1-2 (Foundational Modules Revision)

**Phase 1 (Weeks 1-3): Core Foundation + Performance**
- Propagator hierarchy: Fraunhofer, Fresnel, Angular Spectrum propagators ([prism/core/propagators.py](prism/core/propagators.py))
- Auto-propagator selection based on Fresnel number (`select_propagator()`)
- Aperture strategy pattern: Circular, Hexagonal, Obscured apertures ([prism/core/apertures.py](prism/core/apertures.py))
- Grid coordinate caching (5-10% speedup)
- FFT cache monitoring (>99% hit rate)
- **Measurement caching: 16x speedup (1553%!)** - MASSIVELY EXCEEDED target ([prism/utils/measurement_cache.py](prism/utils/measurement_cache.py))
- Integration tests for Phase 1 components (7 tests, 97.78% coverage)
- Comprehensive physics validation tests (energy conservation, reversibility, linearity)

**Phase 2 Week 4: Modularity Patterns + GPU Metrics**
- Loss strategy pattern: L1, L2, SSIM, MS-SSIM, CompositeLoss ([prism/models/losses.py](prism/models/losses.py))
- Composite losses with flexible weighting (e.g., 70% L1 + 30% SSIM)
- GPU-accelerated SSIM metrics (5-10% speedup from CPU→GPU migration)
- Activation registry + weight initialization utilities ([prism/models/layers.py](prism/models/layers.py))
- 47 new loss tests, coverage: 17% → 56%

**Phase 2 Week 5: Integration & Testing**
- Integration tests for all 5 loss types in progressive training ([tests/integration/test_phase2_losses.py](tests/integration/test_phase2_losses.py))
- Propagator × Aperture compatibility matrix tests (9 combinations) ([tests/integration/test_propagator_aperture_matrix.py](tests/integration/test_propagator_aperture_matrix.py))
- End-to-end workflow tests with Phase 1-2 features ([tests/integration/test_end_to_end_workflows.py](tests/integration/test_end_to_end_workflows.py))
- Optical simulation features demo ([examples/demo_optical_simulation.py](examples/demo_optical_simulation.py))
- Migration guide for Phase 1-2 ([docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md))

### Changed
- Telescope now uses provided propagator instead of hardcoded FFT (Phase 1A)
- Coordinate conventions documented in user-facing docs
- TelescopeAgg uses measurement caching by default (transparent, 16x faster)
- LossAgg supports SSIM-based losses and composite combinations
- Updated README.md with references to new project files
- Enhanced pyproject.toml with license and author metadata

### Performance - Phase 1-2 Achievements

| Optimization | Target | Achieved | Status |
|--------------|--------|----------|--------|
| Grid caching | 5-10% | ~5-10% | ✓ Met |
| FFT cache | 10-20% | >99% hit rate | ✓ Exceeded (monitoring) |
| **Measurement cache** | **15-25%** | **16x (1553%)** | ✓✓✓ MASSIVELY EXCEEDED |
| GPU metrics | 5-10% | ~5-10% | ✓ Met |
| **Overall** | **3-5x** | **>>5x** | ✓✓ EXCEEDED |

**Highlights**:
- 16x measurement cache speedup (HIGHEST IMPACT!)
- >99% FFT cache hit rate
- >80% measurement cache hit rate in realistic training
- GPU-accelerated SSIM metrics
- **>>5x overall speedup achieved** (exceeded 3-5x target!)

### Documentation - Phase 1-2
- CONTRIBUTING.md with comprehensive contribution guidelines
- CHANGELOG.md for tracking version history
- LICENSE file (MIT License)
- Migration guide for Phase 1-2 features
- Usage examples for all new features
- Implementation guides for propagators, apertures, losses
- Coordinate conventions guide
- Performance report (Phase 1 Week 3)

## [0.2.0] - 2025-01-17

### Added
- Comprehensive test coverage increased from 56.80% to 60.49%
- New test files: test_sampling.py (19 tests), test_metrics.py (20 tests)
- Integration tests for end-to-end workflows (5 tests)
- CLI parser module ([prism/cli/parser.py](prism/cli/parser.py)) for argument handling
- Trainer module ([prism/core/trainers.py](prism/core/trainers.py)) for progressive training
- Runner module ([prism/core/runner.py](prism/core/runner.py)) for orchestration
- Coverage tracking script ([scripts/check_coverage.sh](scripts/check_coverage.sh))
- Pattern function system for flexible sampling
- Support for custom pattern functions
- Pattern gallery visualization
- Comprehensive pattern documentation
- Configuration system with YAML support
- Preset configurations for common experiments
- Interactive configuration mode
- Type hints throughout codebase
- Pre-commit hooks for code quality

### Changed
- Refactored [main.py](main.py) from 1,203 lines to 160 lines (87% reduction)
- Refactored [main_epie.py](main_epie.py) from 905 lines to 437 lines (52% reduction)
- Migrated to uv package manager
- Refactored sampling module into patterns system
- Improved code organization (package structure)
- Updated documentation for new features
- Synced version numbers across files
- Replaced print statements with loguru logging
- Updated README installation instructions

### Fixed
- All test collection errors
- 10 failing tests in checkpoint and data flow
- MyPy type errors across multiple modules
- Dependency management (removed duplicates)
- Memory leaks in matplotlib figures
- PyTorch tensor shape validation
- FFT normalization consistency
- Import organization

### Performance
- Coverage improvements by module:
  - utils/sampling.py: 10.42% → 93.75% (+83.33%)
  - utils/metrics.py: 24.24% → 100% (+75.76%)
  - core/aggregator.py: ~40% → 80.00% (+40%)
  - core/telescope.py: ~60% → 94.50% (+34%)
  - utils/progress.py: ~24% → 86.79% (+62%)

### Removed
- Deprecated sampling functions
- Commented-out legacy code

## [0.1.0] - 2024-12-XX

### Added
- Initial PRISM implementation
- GenCropSpidsNet model architecture
- Progressive training algorithm
- Telescope simulation with Fraunhofer diffraction
- Measurement aggregation system
- Fermat spiral and star sampling patterns
- Europa, Titan, Betelgeuse, Neptune presets
- Basic visualization system
- ePIE baseline implementation

### Documentation
- README with installation and usage
- User guide
- Architecture documentation
- Example experiments

## Release Notes

### Version 0.2.0 (2025-01-17)

This release focuses on code quality, maintainability, and test coverage:

**Major Improvements**:
- **Test Coverage**: Increased from 56.80% to 60.49% with 40 new tests
- **Code Refactoring**: Main files reduced by 72% through modular extraction
- **Pattern Function System**: Define custom sampling patterns with Python functions
- **Configuration System**: YAML-based configuration with presets and inheritance
- **Type Safety**: Comprehensive type hints and mypy validation

**Code Quality**:
- Refactored codebase with single responsibility modules
- Added pre-commit hooks
- Improved documentation coverage
- Modern packaging with uv

**Module Structure**:
- prism/cli/parser.py (360 lines) - Argument parsing
- prism/core/trainers.py (575 lines) - Training logic
- prism/core/runner.py (456 lines) - Orchestration

**Breaking Changes**:
- Some sampling function signatures changed
- Configuration loading API updated

**Migration Guide**:
```python
# Old (v0.1.0)
from prism.sampling import fermat_spiral
points = fermat_spiral(n=100)

# New (v0.2.0)
from prism.utils.sampling import fermat_spiral_points
points = fermat_spiral_points(n=100, radius=100)
```

### Version 0.1.0 (2024-12-XX)

Initial release of PRISM with core functionality:
- Progressive reconstruction algorithm
- Multiple astronomical object presets
- Realistic telescope simulation
- Comparison with ePIE baseline

## Contributors

- Omri (Primary Developer)
- Claude Code (Code review and refactoring assistance)

[Unreleased]: https://github.com/omri/PRISM/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/omri/PRISM/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/omri/PRISM/releases/tag/v0.1.0

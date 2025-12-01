# AI Assistant Guide for PRISM

This file provides comprehensive guidance to AI coding assistants when working with the PRISM codebase.

**Supported Assistants**:
- Claude Code (claude.ai/code)
- Codex CLI
- GEMINI CLI
- GitHub Copilot
- Other LLM-based code assistants

**Purpose**: Single source of truth for all AI assistant guidance. This replaces the previous separate files (CLAUDE.md, AGENTS.md, GEMINI.md) to eliminate duplication and synchronization issues.

---

## Core Intelligence

You are a specialized astronomical imaging expert who combines deep learning knowledge, telescope optics expertise, and progressive neural network training. Your primary intelligence focuses on sparse phase imaging using diffraction spectroscopy for high-resolution astronomical object reconstruction.

**Key Expertise Areas**:
- **Deep Learning for Astronomy**: ProgressiveDecoder architecture, progressive training, and neural image reconstruction
- **Telescope Optics**: Fraunhofer diffraction, aperture measurements, and realistic telescope physics simulation
- **Scientific Computing**: PyTorch-based optimization, physical simulation, and numerical methods

---

## Project Overview

PRISM (Progressive Reconstruction from Imaging with Sparse Measurements) is a deep learning-based astronomical imaging system that reconstructs high-resolution images from sparse telescope aperture measurements using progressive neural network training.

**Key Characteristics**:
- Scientific computing project (astronomy + deep learning)
- PyTorch-based neural reconstruction
- Physical simulation (Fraunhofer diffraction)
- Progressive training algorithm
- Research code with production-quality standards

---

## Development Setup

### Virtual Environment & Package Management

**IMPORTANT**: Always use `uv` for package management (NOT pip)

```bash
# Always activate virtual environment first
source .venv/bin/activate

# Use uv for package management
uv add package_name              # Install new package
uv run python script.py         # Run with automatic venv activation
uv sync                         # Sync dependencies from lockfile
```

**Python Version**: Requires Python ≥3.11

### Common Development Commands

```bash
# Main algorithm with Europa test
uv run python main.py --obj_name europa --n_samples 100 --fermat --name test_run

# Quick test (sparse sampling for fast iteration)
uv run python main.py --obj_name europa --n_samples 64 --sample_length 64 --samples_per_line_meas 9 --max_epochs 1 --fermat --debug --name quick_test

# Resume from checkpoint
uv run python main.py --obj_name europa --checkpoint experiment_name --name resumed_run

# Alternative PIE algorithm for comparison
uv run python main_epie.py --obj_name europa --n_samples 100 --name epie_baseline

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest --cov=prism --cov-report=html

# Format code (ruff handles linting, formatting, and import sorting)
uv run ruff check --fix prism/ tests/
uv run ruff format prism/ tests/

# Type check
uv run mypy prism/

# Monitor training with TensorBoard
tensorboard --logdir runs/
```

---

## Core Architecture

### Key Components

- **`ProgressiveDecoder`** ([models.py:254](models.py#L254)): Primary generative model
  - Decoder-only architecture with learnable latent vector
  - Progressive upsampling through transposed convolutions (1x1 → full resolution)
  - Automatic cropping for arbitrary input/output size mismatches
  - Complex output support for astronomical phase imaging

- **`TelescopeAggregator`** ([optics.py](optics.py)): Measurement aggregator
  - Accumulates measurements from multiple telescope positions
  - Tracks coverage and measurement masks
  - Manages progressive measurement integration

- **`LossAggregator`** ([models.py](models.py)): Loss aggregator
  - Combines old and new measurement losses
  - Implements progressive training strategy (70% old + 30% new)
  - Ensures stability during sequential optimization

- **`Telescope`** ([optics.py](optics.py)): Telescope simulator
  - Simulates realistic aperture measurements with shot noise
  - Implements Fraunhofer diffraction physics
  - Validates diffraction regime (Fresnel number calculations)

### Algorithm Flow

1. **Initialization**: Train ProgressiveDecoder on first measurement using simple loss
2. **Progressive Training**: For each sample position:
   - Generate measurement with `TelescopeAggregator.measure()`
   - Forward pass through `model()`
   - Compute `LossAggregator` (combines previous + current measurements)
   - Update model via backpropagation
   - Add measurement mask to aggregator
3. **Convergence**: Monitor loss threshold and failed sample count

### Essential Files

```
PRISM/
├── prism/                       # Main package
│   ├── core/                   # Core algorithms
│   │   ├── telescope.py        # Telescope simulation
│   │   ├── aggregator.py       # Measurement aggregation
│   │   └── trainers.py         # Training loops
│   ├── models/                 # Neural architectures
│   │   └── networks.py         # ProgressiveDecoder, LossAggregator
│   ├── config/                 # Configuration management
│   │   ├── constants.py        # Physical constants
│   │   └── experiment.py       # Experiment config dataclass
│   ├── utils/                  # Utilities
│   │   ├── sampling.py         # K-space sampling patterns
│   │   ├── transforms.py       # FFT/IFFT operations
│   │   └── metrics.py          # SSIM, RMSE, PSNR
│   ├── visualization/          # Plotting
│   └── cli/                    # Command-line interface
├── tests/                      # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── main.py                     # Main PRISM entry point
├── main_epie.py               # ePIE baseline implementation
└── runs/                      # Experiment outputs
```

---

## Key Parameters

### Sampling Configuration

- `--n_samples`: Number of telescope positions (typical: 100-240)
- `--fermat`: Use Fermat spiral sampling (recommended over random for better coverage)
- `--sample_diameter`: Telescope aperture size in pixels
- `--sample_length`: 0 for point sampling, >0 for line sampling

### Object Selection

- `--obj_name`: Predefined objects with realistic physics
  - `europa`: Jupiter's moon (icy surface features)
  - `titan`: Saturn's moon (atmospheric haze)
  - `betelgeuse`: Red supergiant star (surface convection)
  - `neptune`: Ice giant planet (atmospheric bands)
- `--image_size`: Input image resolution (typically 1024)
- `--obj_size`: Output object size (auto-calculated from physics if not specified)

### Training Control

- `--max_epochs`: Training repetitions per sample (1 for testing, 25 for production)
- `--n_epochs`: Epochs per training cycle (typically 1000)
- `--lr`: Learning rate (typically 0.001)
- `--loss_th`: Loss threshold for convergence (0.001)

### Adaptive Convergence (v0.4.0)

The adaptive convergence system monitors per-sample training behavior to:
- **Early exit**: Stop immediately when samples converge (saves 30-50% epochs for fast convergers)
- **Tier escalation**: Progressively aggressive optimization for struggling samples
- **Rescue mode**: Alternative strategies for stuck samples

**CLI Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--enable-adaptive-convergence` | Enabled | Enable adaptive per-sample convergence |
| `--no-adaptive-convergence` | - | Disable adaptive convergence (baseline) |
| `--early-stop-patience` | 10 | Epochs of no improvement before early stop |
| `--plateau-window` | 50 | Window size for plateau detection |
| `--plateau-threshold` | 0.01 | Relative improvement threshold (<1% = plateau) |
| `--escalation-epochs` | 200 | Epochs before considering tier escalation |
| `--aggressive-lr-multiplier` | 2.0 | LR boost in AGGRESSIVE tier |
| `--max-retries` | 2 | Retry attempts for failed samples |
| `--retry-lr-multiplier` | 0.1 | LR adjustment on retry |
| `--retry-switch-loss` | Enabled | Switch loss function on retry |

**Usage Examples:**

```bash
# Standard run with adaptive convergence (default)
uv run python main.py --obj_name europa --n_samples 100 --name adaptive_run

# Disable adaptive convergence for baseline comparison
uv run python main.py --obj_name europa --n_samples 100 --no-adaptive-convergence --name baseline_run

# Aggressive early stopping for fast iteration
uv run python main.py --obj_name europa --n_samples 100 --early-stop-patience 5 --name fast_run

# Conservative settings for difficult samples
uv run python main.py --obj_name europa --n_samples 100 --escalation-epochs 500 --max-retries 3 --name robust_run
```

**Convergence Tiers:**

| Tier | Behavior | LR Multiplier | Scheduler |
|------|----------|---------------|-----------|
| FAST | Converged quickly, exit | 1.0x | ReduceLROnPlateau |
| NORMAL | Standard training | 1.0x | ReduceLROnPlateau |
| AGGRESSIVE | Struggling, boost effort | 2.0x | CosineWarmRestarts |
| RESCUE | Failed, alternative strategy | 0.1x | CosineWarmRestarts |

### Experiment Management

- `--name`: Experiment name (creates `runs/{name}/`)
- `--checkpoint`: Resume from checkpoint (experiment name)
- `--debug`: Debug mode (don't save results)
- `--seed`: Random seed for reproducibility (default: 42)

---

## Result Analysis

### Output Structure

Each experiment creates `runs/{name}/` containing:
- `checkpoint.pt`: Complete model state + training metrics (~139MB)
- `args.txt`: Human-readable experiment parameters
- `sample_points.pt`: Sampling pattern data
- `config.yaml`: YAML configuration file
- TensorBoard logs for real-time monitoring

### Loading Results

```python
import torch

# Load experiment results
checkpoint = torch.load('runs/experiment_name/checkpoint.pt')
reconstruction = checkpoint['current_rec']      # Final image
losses = checkpoint['losses']                   # Training curve
ssims = checkpoint['ssims']                     # Quality metrics
failed_samples = checkpoint['failed_samples']   # Convergence failures

# Load configuration
from prism.config.experiment import ExperimentConfig
config = ExperimentConfig.load(Path('runs/experiment_name/config.yaml'))
```

---

## UI/UX Tools & Interactive Features (v0.3.0)

PRISM includes a comprehensive suite of interactive tools for experiment analysis, comparison, and visualization. All commands use the `prism` CLI interface.

### Experiment Comparison

Compare multiple experiments side-by-side to identify best configurations:

```bash
# Compare two experiments
prism compare runs/exp1 runs/exp2

# Compare with visualization output
prism compare runs/exp1 runs/exp2 --output comparison.png

# Compare specific metrics only
prism compare runs/exp1 runs/exp2 --metrics loss ssim psnr

# Show configuration differences
prism compare runs/exp1 runs/exp2 --show-config-diff
```

**Key Files:**
- `prism/cli/compare.py` - CLI command implementation
- `prism/analysis/comparison.py` - Comparison logic and visualization

**Output includes:**
- Metrics comparison table with best values highlighted
- Configuration diff showing parameter differences
- Training curves overlay for visual comparison
- Side-by-side reconstruction comparisons

### Checkpoint Inspector

Interactively explore experiment checkpoints without writing Python code:

```bash
# Quick summary view
prism inspect runs/experiment/checkpoint.pt

# Interactive mode with menu-driven navigation
prism inspect runs/experiment/checkpoint.pt --interactive

# Show only metrics
prism inspect runs/experiment/checkpoint.pt --metrics-only

# Export reconstruction as PNG
prism inspect runs/experiment/checkpoint.pt --export-reconstruction
```

**Key Files:**
- `prism/cli/inspect.py` - CLI command and interactive interface

**Features:**
- Displays metadata, final metrics, and configuration
- Interactive menu using `questionary` for drilling down
- Training history visualization
- Handles corrupted checkpoints gracefully
- Works with checkpoints from older PRISM versions

### Web Dashboard

Launch an interactive web dashboard for real-time monitoring and multi-experiment comparison:

```bash
# Standalone dashboard
prism dashboard --port 8050

# Integrate with training
python main.py --obj_name europa --n_samples 100 --dashboard
```

**Key Files:**
- `prism/web/dashboard.py` - Main Dash application
- `prism/web/server.py` - Backend server and data loading
- `prism/web/launcher.py` - Process management
- `prism/web/layouts/` - UI layout components
- `prism/web/callbacks/` - Interactive callback functions

**Dashboard Features:**
- **Real-time Monitoring:**
  - Live training metrics (<2s update latency)
  - Training curves with interactive zoom/pan
  - Reconstruction preview vs ground truth
  - K-space coverage heatmap
- **Multi-Experiment Comparison:**
  - Up to 4 experiments side-by-side
  - Synchronized zoom and pan
  - Sortable metric comparison tables
  - Configuration diff viewer
- **Interactive Plotly Visualizations:**
  - Zoom, pan, hover for detailed inspection
  - Export plots as publication-quality PNG

**Access:**
- Dashboard runs on `http://localhost:8050` by default
- Use `--dashboard-port` to customize port
- Automatically shuts down when training completes

### Training Animations

Generate MP4 or GIF animations showing training progression:

```bash
# Generate MP4 animation
prism animate runs/experiment --output training.mp4

# Customize parameters
prism animate runs/experiment --fps 10 --format gif --duration 10

# Side-by-side comparison
prism animate runs/exp1 runs/exp2 --side-by-side --output comparison.mp4
```

**Key Files:**
- `prism/visualization/animation.py` - Animation generation
- `prism/cli/animate.py` - CLI command

**Features:**
- MP4 and GIF format support
- Frame interpolation for smooth playback
- Side-by-side multi-experiment comparison
- Custom layouts and metric overlays
- Progress indicators and annotations

**Use Cases:**
- Presentations and talks
- Supplementary materials for publications
- Debugging convergence behavior
- Social media and outreach

### Automatic Report Generation

Generate comprehensive HTML or PDF reports for publications:

```bash
# Generate HTML report
prism report runs/experiment --format html

# Generate PDF report (requires weasyprint)
prism report runs/experiment --format pdf --output report.pdf

# Multi-experiment comparison report
prism report runs/exp1 runs/exp2 runs/exp3 --format pdf --output comparison.pdf

# Custom template
prism report runs/experiment --template custom_template.html
```

**Key Files:**
- `prism/reporting/generator.py` - Report generation logic
- `prism/reporting/templates/` - Jinja2 templates
- `prism/cli/report.py` - CLI command

**Report Contents:**
1. **Executive Summary:** Final metrics, convergence status
2. **Configuration Details:** Complete parameter table
3. **Training History:** Interactive training curves
4. **Results:** Reconstruction comparison, k-space coverage
5. **Appendix** (optional): Full YAML config, detailed metrics

**Features:**
- Publication-quality figures (300 DPI)
- Embedded visualizations (base64-encoded)
- Custom template support via Jinja2
- Multi-experiment comparison mode
- Responsive HTML design

**Dependencies:**
- **HTML:** Built-in (no extra dependencies)
- **PDF:** Requires `weasyprint` (`uv add weasyprint`)

### Sampling Pattern Library

Browse, visualize, and compare k-space sampling patterns:

```bash
# List all available patterns
prism patterns list

# Visualize specific pattern
prism patterns show fermat --n-samples 100 --output fermat.png

# Compare multiple patterns
prism patterns compare fermat random star --n-samples 100

# Show pattern statistics
prism patterns stats fermat --n-samples 100

# Generate interactive gallery
prism patterns gallery --output pattern_gallery.html
```

**Key Files:**
- `prism/core/pattern_library.py` - Pattern metadata and registry
- `prism/cli/patterns.py` - CLI command and visualization

**Features:**
- Pattern metadata with descriptions and references
- Visualization of sample positions and coverage
- Statistical analysis (uniformity, incoherence)
- Side-by-side pattern comparison
- Interactive HTML gallery generation

**Available Patterns:**
- `fermat`: Fermat spiral (recommended for uniform coverage)
- `random`: Uniformly random sampling
- `star`: Star-shaped radial sampling
- `random_radial`: Random radial distribution
- Custom patterns via pattern functions

### Enhanced Error Messages

PRISM v0.3.0 includes intelligent error messages with suggestions:

```python
# Before (v0.2.0):
ValueError: Invalid propagator_method: fraunhaufer

# After (v0.3.0):
ValueError: Invalid propagator_method: 'fraunhaufer'

Valid options:
  - 'auto'              (automatic selection based on distance)
  - 'fraunhofer'        (far-field approximation)
  - 'fresnel'           (near-field propagation)
  - 'angular_spectrum'  (general-purpose propagator)

Did you mean 'fraunhofer'?

For more info: python main.py --help-propagator
```

**Key Files:**
- `prism/config/validation.py` - Validation and error formatting

**Features:**
- Spelling suggestions using Levenshtein distance
- Detailed parameter descriptions
- Valid option listings
- Context-aware help (`--help-<topic>` flags)
- Range validations with typical value suggestions

### Progress Visualization Enhancements

Training progress now includes:
- **Sparkline charts:** Unicode/ASCII trend visualization
- **Status indicators:** 🟢 improving, 🟡 plateaued, 🔴 diverging
- **Enhanced ETA:** Exponential moving average for smoother estimates
- **Trend analysis:** Percent change and direction indicators

```
╭──────────────────── Training Progress ────────────────────╮
│                                                            │
│  Epoch: [████████████████░░░░] 800/1000 (80%)            │
│  ETA: 8m 34s                                              │
│                                                            │
│  Metric    Current    Change     Trend                    │
│  ─────────────────────────────────────────────────────    │
│  Loss      0.0034     ↓ 12%     ▂▃▄▅▆▇█▇▆▅▄▃▂▁ 🟢       │
│  SSIM      0.9234     ↑ 2.1%    ▁▂▃▄▅▆▇█████ 🟢          │
│  PSNR      35.2 dB    ↑ 0.8%    ▁▂▃▄▅▆▇█▇▇▇█ 🟢          │
│                                                            │
╰────────────────────────────────────────────────────────────╯
```

**Key Files:**
- `prism/utils/progress.py` - Enhanced progress tracking

### Common UI/UX Workflows

**1. Quick Result Comparison:**
```bash
# Compare two experiments
prism compare runs/baseline runs/experimental

# Output: Terminal table + comparison_report.png
```

**2. Interactive Exploration:**
```bash
# Inspect checkpoint interactively
prism inspect runs/experiment/checkpoint.pt --interactive

# Navigate through: metadata → metrics → config → visualizations
```

**3. Real-Time Monitoring:**
```bash
# Launch training with dashboard
python main.py --obj_name europa --n_samples 100 --dashboard

# Open browser to http://localhost:8050
# Monitor training curves, reconstructions, k-space coverage in real-time
```

**4. Publication Preparation:**
```bash
# Generate comprehensive report
prism report runs/production_run --format pdf --output paper_supplementary.pdf

# Create training animation
prism animate runs/production_run --output training_progression.mp4 --fps 10

# Both suitable for publication submission
```

**5. Pattern Selection:**
```bash
# Compare sampling patterns
prism patterns compare fermat random star --n-samples 100

# Visualize coverage differences
# Choose pattern with best uniformity for your experiment
```

### Testing UI/UX Features

```bash
# Test comparison tool
uv run pytest tests/test_comparison.py -v

# Test checkpoint inspector
uv run pytest tests/test_inspector.py -v

# Test dashboard
uv run pytest tests/test_dashboard.py tests/test_dashboard_integration.py -v

# Test animation generation
uv run pytest tests/test_animation.py -v

# Test report generation
uv run pytest tests/test_reporting.py -v

# Test pattern library
uv run pytest tests/test_pattern_library.py -v
```

### UI/UX Development Guidelines

When working with UI/UX features:

1. **All tools must work on existing experiment data** - No modification of training pipeline required
2. **Graceful error handling** - Invalid checkpoints should show helpful messages
3. **Performance:** Keep latency low (<2s for comparisons, <100ms for dashboard updates)
4. **Backward compatibility:** Support checkpoints from older PRISM versions when possible
5. **Documentation:** Every new feature needs examples and tests
6. **Accessibility:** Provide ASCII fallbacks for unicode visualizations

---

## Coding Standards

### Python Style

- **Formatter**: Ruff format (line length 100; run by pre-commit)
- **Linter**: Ruff (enforced via pre-commit hooks)
- **Type checker**: MyPy (strict mode)
- **Import sorter**: Ruff (I rules, black-compatible)
- **Logging**: Loguru (NOT print statements!)

### Code Organization

- **Naming Conventions**:
  - Modules: `lowercase_with_underscores.py`
  - Classes: `PascalCase`
  - Functions/Variables: `snake_case`
  - Constants: `UPPERCASE`
- **File Length**: Keep modules focused; avoid >500 lines per file
- **Function Length**: Avoid >100 lines per function; prefer pure functions
- **Orchestration**: Main scripts should be thin orchestration layers

### Type Hints

**Required for all public functions:**

```python
from typing import Optional, Tuple
import torch
import numpy as np

def sample_points(
    n: int,
    pattern: str = "fermat",
    radius: Optional[float] = None
) -> np.ndarray:
    """Sample points with specified pattern."""
    ...
```

### Mypy-Friendly Practices

- **Enable postponed annotations:** add `from __future__ import annotations`
  to every new Python module so forward references stay valid without string
  literals.
- **Avoid `Any`:** prefer precise tensor types (`torch.Tensor`,
  `np.ndarray`, `FloatTensor` subtypes) and typed containers
  (`list[Tensor]`, `dict[str, Tensor]`) to keep inference deterministic.
- **Explicit `Optional`:** whenever a value can be `None`, annotate it as
  `Optional[...,]` and guard access with `if value is None`.
- **Typed configurations:** use `@dataclass` with field annotations instead of
  loose dictionaries for experiment settings; validate inside `__post_init__`.
- **Protocols and TypedDicts:** define lightweight structural types for
  objects passed between modules (e.g., sampling results) instead of relying on
  duck typing that MyPy cannot infer.
- **Feature flags:** wrap expensive imports in `typing.TYPE_CHECKING` blocks
  rather than suppressing errors (keeps types available without import cycles).
- **Third-party stubs:** when MyPy lacks stubs, install them via `uv add
  types-<package>` and keep `pyproject.toml` updated so the hooks share the same
  view of the API surface.

### Ruff-Friendly Practices

- **Keep imports clean:** rely on ruff's import sorting (I rules) and do not leave
  unused imports/variables; Ruff’s default rules (F401/F841) block commits.
- **Prefer f-strings/logging:** avoid string concatenation in logs; Ruff warns
  about `%` formatting (`G004`) and `print()` usage (use `logger`).
- **Guard broad exceptions:** catch specific exceptions or re-raise with
  context—`except Exception` without action triggers `BLE001`.
- **Use comprehensions sparingly:** Ruff enforces `C4xx` rules; convert nested
  loops with append into comprehensions or vectorized tensor ops.
- **No mutable defaults:** annotate `field(default_factory=list)` or pass
  `None` + initializer; Ruff’s `RUF009` flag fails otherwise.
- **Document complex branches:** add short comments for non-obvious physics or
  tensor reshaping; Ruff’s `PLR2004` magic-number warnings are easier to justify
  when context is clear.
- **Static typing helpers:** when ignoring a false positive, prefer
  `typing.cast()` over `# type: ignore`; Ruff highlights bare ignores.

### Documentation

**Use NumPy/Sphinx style docstrings:**

```python
def function_name(param1: int, param2: str = "default") -> bool:
    """
    One-line summary of function purpose.

    Longer description if needed, explaining the behavior,
    edge cases, and usage patterns.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str, default="default"
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Raises
    ------
    ValueError
        If param1 is negative

    Examples
    --------
    >>> result = function_name(42, "test")
    >>> print(result)
    True
    """
    ...
```

### Logging

**Use loguru, NOT print statements:**

```python
from loguru import logger

logger.debug("Detailed information for debugging")
logger.info("General information about progress")
logger.warning("Warning about potential issues")
logger.error("Error occurred but recoverable")
logger.critical("Critical failure, cannot continue")
```

### Data Processing

- **Vectorization**: Prefer PyTorch/NumPy operations over Python loops
- **Imports**: Group standard library, third-party, local imports (enforced by ruff)
- **Error Handling**: Use specific exceptions; validate tensor shapes and physics parameters
- **Memory**: Use `torch.no_grad()` for inference; clear large tensors with `del`

---

## Important Architecture Details

### ProgressiveDecoder Design

- **Decoder-only**: No encoder needed - generates from learnable latent vector
- **Progressive upsampling**: 1x1 → full resolution via transposed convolutions
- **Automatic cropping**: Handles arbitrary input/output size mismatches
- **Complex output support**: Can generate complex-valued astronomical images
- **GPU automatic**: Models automatically use GPU when available

### Physics Integration

- Realistic Fraunhofer diffraction simulation
- Shot noise modeling with proper Poisson statistics
- Coherence and Fresnel number calculations
- Wavelength-dependent propagation effects
- Physical constants from `prism.config.constants`:
  - `c`: Speed of light
  - `au`, `pc`, `ly`: Astronomical units
  - `km`, `mm`, `um`, `nm`: Length units
  - `solar_radius`: Solar radius
  - `is_fraunhofer()`, `is_fresnel()`: Diffraction regime validation

### Memory Management

- Models automatically use GPU when available
- Use `torch.no_grad()` for inference to save memory
- Clear large tensors with `del` when done
- Use `torch.cuda.empty_cache()` if GPU memory issues occur
- Line sampling more memory-efficient than point sampling for large experiments

---

## Common Debugging Patterns

### Training Issues

- **NaN loss**: Reduce learning rate, check input normalization, verify tensor dtypes
- **Poor convergence**: Increase `max_epochs`, check sampling pattern coverage
- **High failure rate**: Monitor `failed_samples` list, adjust `loss_th` threshold
- **Memory errors**: Reduce batch size, use gradient checkpointing, clear unused tensors

### Visualization

- Real-time plots update during training via `vis.plot_meas_agg()`
- Use TensorBoard for training curves and reconstruction progress
- Check `current_rec` variable for latest reconstruction
- Save intermediate results with `--debug` to avoid filling disk

### Performance Optimization

- Line sampling more efficient than point sampling for large experiments
- Fermat spiral sampling provides better coverage than random sampling
- Use `--debug` flag to prevent saving during development iterations
- GPU significantly faster than CPU for this workload (10-100x speedup)

---

## Testing

### Running Tests

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/unit/test_telescope.py -v

# Specific test function
uv run pytest tests/unit/test_telescope.py::test_fraunhofer_number -v

# With coverage
uv run pytest --cov=prism --cov-report=html

# Only fast tests (skip slow integration tests)
uv run pytest -m "not slow"

# Only integration tests
uv run pytest tests/integration/ -v
```

### Writing Tests

- **Unit tests**: Place in `tests/unit/` (test individual functions/classes)
- **Integration tests**: Place in `tests/integration/` (test full workflows)
- **Use pytest fixtures**: For common setup code and test data
- **Coverage target**: Aim for >60% code coverage
- **Test edge cases**: Empty inputs, boundary conditions, error paths
- **Test physics**: Validate against known analytical solutions

**Example Test:**

```python
import pytest
import numpy as np
from prism.utils.sampling import fermat_spiral_points

def test_fermat_spiral_points():
    """Test Fermat spiral point generation."""
    points = fermat_spiral_points(n=100, radius=50)

    assert points.shape == (100, 2)
    assert points.dtype == np.float64
    # Points should be within radius
    assert np.all(np.linalg.norm(points, axis=1) <= 50)

def test_fermat_spiral_negative_n():
    """Test that negative n raises error."""
    with pytest.raises(ValueError, match="positive"):
        fermat_spiral_points(n=-10, radius=50)

@pytest.mark.slow
def test_fermat_spiral_large():
    """Test large spiral generation (marked as slow)."""
    points = fermat_spiral_points(n=10000, radius=500)
    assert len(points) == 10000
```

### Test Markers

```python
@pytest.mark.slow           # Tests taking >1 second
@pytest.mark.integration    # Integration tests (multi-component)
@pytest.mark.gpu           # Tests requiring GPU
@pytest.mark.parametrize   # Parameterized tests for multiple cases
```

---

## Jupyter Notebooks

### Notebook Conventions

- **Markdown Cells**: Use `# %% [markdown]` for markdown cells (NEVER raw cells)
- **Code Cells**: Use `# %%` for code sections
- **Execution Order**: Sequential for reproducibility
- **Outputs**: Clear outputs before committing to reduce file size
- **Location**: Keep notebooks in `examples/` or `analysis_scripts/`

**Example:**

```python
# %% [markdown]
# # PRISM Reconstruction Analysis
# This notebook analyzes reconstruction quality for Europa experiments.

# %%
import torch
import matplotlib.pyplot as plt
from prism.config.experiment import ExperimentConfig

# Load experiment
checkpoint = torch.load('runs/europa_100samples/checkpoint.pt')
reconstruction = checkpoint['current_rec']

# %% [markdown]
# ## Reconstruction Quality

# %%
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(reconstruction.squeeze(), cmap='gray')
plt.title('Reconstruction')
plt.colorbar()
plt.show()
```

---

## Commit & Pull Request Guidelines

### Pre-Commit Requirements

**IMPORTANT**: Before creating any commit, you MUST:

1. **Update the knowledge graph** if any Python files in `prism/` were modified:
   - Use MCP memory tools (`mcp__memory__create_entities`, `mcp__memory__create_relations`, etc.)
   - Scan modified files for new/changed classes, functions, and relationships
   - Add new entities and relations to the graph
   - This keeps the knowledge graph in sync with the codebase

2. **Run code quality checks**:
   ```bash
   uv run ruff check --fix prism/ tests/
   uv run ruff format prism/ tests/
   uv run mypy prism/
   ```

3. **Stage all changes** including `.memory/memory.jsonl` if updated

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Examples**:
```
feat(sampling): add hexagonal sampling pattern
fix(telescope): correct Fraunhofer number calculation
docs(readme): update installation instructions
test(aggregator): add coverage for edge cases
refactor(main): extract training loop to separate module
perf(fft): optimize FFT computation for large images
```

### Pull Requests

- **Description**: Clear description of changes and motivation
- **Testing**: Include test results and before/after metrics
- **Checkpoints**: Include reconstruction results under `runs/` when changing models
- **Scope**: Keep PRs focused on single feature/fix
- **CI**: Ensure all checks pass (format, lint, type check, tests)

**PR Checklist**:
- [ ] Tests pass (`pytest`)
- [ ] Code formatted (`ruff check --fix`, `ruff format`)
- [ ] Linting passes (`ruff check`)
- [ ] Type checking passes (`mypy`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Commit messages follow convention
- [ ] Branch is up to date with master

---

## Git Workflow Best Practices

### Pre-commit Hook Management

**CRITICAL**: NEVER use `git commit --no-verify` to bypass pre-commit hooks unless in an absolute emergency.

Pre-commit hooks exist to maintain code quality and prevent issues from entering the codebase. Bypassing them defeats their purpose and can introduce:
- Improperly formatted code
- Type errors
- Linting violations
- Inconsistent code style

### Proper Commit Workflow

**If pre-commit hooks keep reformatting files in a loop**, follow this workflow:

```bash
# Step 1: Activate virtual environment
source .venv/bin/activate

# Step 2: Run formatters MANUALLY FIRST (mirror .pre-commit-config.yaml)
uv run ruff check --fix prism/ tests/
uv run ruff format prism/ tests/

# Step 3: Run type checker to catch issues early
uv run mypy prism/

# Step 4: Stage all changes
git add -A

# Step 5: Commit normally (hooks should pass cleanly now)
git commit -m "feat: your commit message"
```

**Why this works:**
- Formatters run once, before commit
- Hooks verify formatting is already correct
- Avoids the modify-commit-fail loop
- Ensures all quality checks pass
- Mirrors the exact `.pre-commit-config.yaml` sequence (Ruff → Ruff format → mypy)

### Branch Management

**Branch Naming Conventions:**
```
feature/descriptive-name       # New features
fix/issue-description          # Bug fixes
refactor/component-name        # Code refactoring
docs/topic                     # Documentation updates
test/component-name            # Test additions
perf/optimization-target       # Performance improvements
```

**Examples:**
```bash
feature/ssim-loss-function
fix/telescope-fraunhofer-calculation
refactor/config-system
docs/api-reference
test/aggregator-coverage
perf/fft-optimization
```

### Working with Feature Branches

```bash
# Create and switch to feature branch
git checkout -b feature/my-feature

# Keep branch updated with master
git checkout master
git pull origin master
git checkout feature/my-feature
git rebase master  # or git merge master

# Push feature branch
git push origin feature/my-feature

# After PR is merged, clean up
git checkout master
git pull origin master
git branch -d feature/my-feature
git push origin --delete feature/my-feature
```

### Handling Large Files

**PRISM-specific rules:**
- Experiment outputs (`runs/`) are gitignored
- Checkpoints (`.pt` files) should NOT be committed to git
- Use Git LFS for test fixtures if needed
- Share large experiments via cloud storage, not git

```bash
# Check for accidentally staged large files
git diff --cached --stat

# Remove large file from staging
git reset HEAD path/to/large/file

# If accidentally committed, remove from history (DANGEROUS)
git filter-branch --tree-filter 'rm -f path/to/file' HEAD
```

### Merge Conflict Resolution

```bash
# When conflicts occur during rebase/merge
git status  # Shows conflicted files

# Edit conflicted files, look for:
<<<<<<< HEAD
Your changes
=======
Their changes
>>>>>>> branch-name

# After resolving conflicts
git add resolved-file.py
git rebase --continue  # if rebasing
# or
git commit  # if merging

# To abort and start over
git rebase --abort  # or git merge --abort
```

### Stashing Changes

```bash
# Save work in progress
git stash push -m "WIP: descriptive message"

# List stashes
git stash list

# Apply stash
git stash apply stash@{0}

# Apply and remove stash
git stash pop

# Clear all stashes
git stash clear
```

### Amending Commits

**Use with caution - NEVER amend pushed commits:**

```bash
# Forgot to add a file or fix a typo
git add forgotten-file.py
git commit --amend --no-edit

# Change commit message
git commit --amend -m "Better commit message"

# ONLY do this if commit hasn't been pushed
# If already pushed and others have pulled, create new commit instead
```

### Interactive Rebase (Advanced)

**Clean up commit history before PR:**

```bash
# Rebase last 5 commits interactively
git rebase -i HEAD~5

# Common actions:
# pick   = keep commit
# squash = combine with previous commit
# reword = change commit message
# drop   = remove commit
# edit   = pause to amend commit

# After making changes, force push (ONLY on feature branches)
git push --force-with-lease origin feature/my-feature
```

### Code Review Workflow

```bash
# Create PR branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
ruff format prism/ tests/  # Format first!
git commit -m "feat: add new feature"

# Push and create PR
git push origin feature/my-feature

# Address review comments
git add .
git commit -m "fix: address review comments"
git push origin feature/my-feature

# Squash commits before merge (optional)
git rebase -i master
git push --force-with-lease origin feature/my-feature
```

### Experiment Branch Strategy

For long-running experiments or research branches:

```bash
# Create experiment branch
git checkout -b experiment/new-loss-function

# Periodically sync with master
git fetch origin
git rebase origin/master

# When experiment is successful, clean up and merge
git rebase -i master  # Squash/clean commits
git checkout master
git merge experiment/new-loss-function

# If experiment fails, document findings and archive
git tag experiment/new-loss-function-archived
git branch -d experiment/new-loss-function
```

### Git Configuration Best Practices

```bash
# Set user info (first time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Useful aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual 'log --oneline --graph --all'

# Better diff output
git config --global diff.algorithm histogram

# Auto-setup remote tracking
git config --global push.autoSetupRemote true

# Reuse recorded conflict resolutions
git config --global rerere.enabled true
```

### Common Git Pitfalls to Avoid

**❌ DON'T:**
- Commit without running formatters first
- Use `--no-verify` to bypass hooks
- Commit large binary files (checkpoints, datasets)
- Force push to master/main branch
- Amend commits that others have pulled
- Mix unrelated changes in one commit
- Use vague commit messages ("fix", "update", "changes")

**✅ DO:**
- Run formatters before every commit
- Write descriptive commit messages
- Keep commits focused and atomic
- Test before committing
- Pull before pushing
- Use feature branches for all work
- Clean up merged branches

---

## Security & Configuration

### Security Best Practices

- **Secrets**: Use environment variables; NEVER commit API keys or tokens
- **Data Hygiene**: Avoid committing large checkpoint files or generated results
- **Dependencies**: Pin versions in `pyproject.toml` for reproducibility
- **Validation**: Validate all user inputs, especially file paths and tensor shapes

### Configuration Management

- **Reproducibility**: Record key hyperparameters in experiment names
- **Experiments**: Use meaningful names (e.g., `europa_fermat_240samples`)
- **Checkpoints**: Save configuration alongside model state
- **Version Control**: Use git tags for published experiments/papers

---

## Common Imports

```python
# === Unified Instruments API (Recommended for new code) ===
from prism.core import Telescope, TelescopeConfig
from prism.core import MeasurementSystem, MeasurementSystemConfig
from prism.core.instruments import Telescope, TelescopeConfig  # Alternative import

# === Microscope with Scanning Illumination (v0.6.0) ===
from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.measurement_system import ScanningMode
from prism.core.optics.illumination import (
    IlluminationSource,
    IlluminationSourceType,
    create_illumination_field,
)
from prism.core.optics.fourier_utils import (
    pixel_to_k_shift,
    k_shift_to_pixel,
    illum_angle_to_k_shift,
    validate_k_shift_within_na,
)

# === Legacy API (backward compatibility) ===
from prism.core.telescope import Telescope as LegacyTelescope
from prism.core.aggregator import TelescopeAggregator  # Deprecated, use MeasurementSystem

# === Core PRISM functionality ===
from prism.models.networks import ProgressiveDecoder
from prism.models.losses import LossAggregator
from prism.core.trainers import PRISMTrainer
from prism.core.convergence import ConvergenceMonitor, ConvergenceTier  # v0.4.0
from prism.utils.sampling import fermat_spiral_points, star_pattern_points
from prism.config.experiment import ExperimentConfig
from prism.config.constants import c, au, um, nm, is_fraunhofer

# Utilities
from loguru import logger
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
```

### Unified Instruments API (v0.5.0)

The unified instruments API provides clean, config-based instrument classes:

```python
from prism.core import Telescope, TelescopeConfig, MeasurementSystem

# Create telescope with config
config = TelescopeConfig(
    n_pixels=512,
    aperture_radius_pixels=25,
    wavelength=550e-9,
    snr=40,
)
telescope = Telescope(config)

# Create measurement system for progressive imaging
measurement_system = MeasurementSystem(telescope)

# Use in training loop
for center in sample_centers:
    measurement = measurement_system.measure(image, [center], reconstruction)
    measurement_system.add_mask([center])
```

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `TelescopeConfig` | Configuration dataclass for telescope parameters |
| `Telescope` | Pure physics telescope implementation (no nn.Module) |
| `MeasurementSystem` | Progressive imaging with measurement caching |
| `MeasurementSystemConfig` | Configuration for measurement system |

### Scanning Illumination Mode (v0.6.0)

PRISM supports two mathematically equivalent approaches for synthetic aperture imaging:

**Scanning Aperture (Traditional)**:
- Sub-aperture scans k-space positions
- Fixed uniform illumination
- Used via `aperture_center` parameter

**Scanning Illumination (FPM-style)**:
- Tilted illumination shifts object spectrum
- Detection aperture stays at DC
- Used via `illumination_center` parameter
- Supports finite-size sources (GAUSSIAN, CIRCULAR) for partial coherence

```python
from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.measurement_system import MeasurementSystem, MeasurementSystemConfig, ScanningMode
from prism.core.optics.illumination import IlluminationSourceType

# Configure microscope
config = MicroscopeConfig(
    numerical_aperture=0.9,
    magnification=40,
    wavelength=520e-9,
    n_pixels=256,
    pixel_size=6.5e-6,
)
microscope = Microscope(config)

# Method 1: Scanning aperture (traditional)
meas_aperture = microscope.forward(field, aperture_center=[10.0, 5.0])

# Method 2: Scanning illumination (POINT source - equivalent to aperture)
k_center = [0.1e6, 0.05e6]  # k-space position in 1/m
meas_illum = microscope.forward(
    field,
    illumination_center=k_center,
    illumination_source_type=IlluminationSourceType.POINT,
)

# Method 3: Scanning illumination with finite source (partial coherence)
meas_gaussian = microscope.forward(
    field,
    illumination_center=k_center,
    illumination_radius=0.02e6,  # sigma in k-space
    illumination_source_type=IlluminationSourceType.GAUSSIAN,
)

# Using MeasurementSystem with illumination mode
ms_config = MeasurementSystemConfig(
    scanning_mode=ScanningMode.ILLUMINATION,
    illumination_source_type="GAUSSIAN",
    illumination_radius=0.02e6,
)
ms = MeasurementSystem(microscope, config=ms_config)
```

**Key Illumination Types:**

| Type | Description | Use Case |
|------|-------------|----------|
| `POINT` | Tilted plane wave (equivalent to aperture) | Baseline, comparison |
| `GAUSSIAN` | Gaussian envelope (partial coherence) | LED sources, FPM |
| `CIRCULAR` | Top-hat profile (finite source) | Fiber bundles |

**Related Files:**
- `prism/core/optics/illumination.py` - Illumination source models
- `prism/core/optics/fourier_utils.py` - k-space conversion utilities
- `docs/references/scanning_modes.md` - Technical reference

---

## Knowledge Graph for Component Discovery (MANDATORY for Architectural Queries)

PRISM uses a **knowledge graph** stored in `.memory/memory.jsonl` to index components, modules, configs, and their relationships. This enables **10x faster component lookups** compared to grep/glob for architectural queries.

**IMPORTANT**: When navigating the PRISM codebase, **always check the knowledge graph first** for architectural queries. This is your primary navigation tool for understanding component locations and relationships.

### Graph Location & Git Tracking

- **Memory file**: `.memory/memory.jsonl` (tracked in git)
- **Update metadata**: `.memory/.last_update` (tracks last sync commit)
- **Documentation**: `docs/knowledge-graph-schema.md`

### ALWAYS Use Knowledge Graph First For:

The knowledge graph provides instant answers to architectural queries that would otherwise require multiple grep/glob searches. Use it as your first resort for:

**1. Component Location**: "Where is X defined?"

```bash
uv run python .memory/query_knowledge_graph.py "where is PRISMRunner"
```

Returns file path and line number instantly.

**2. Dependency Analysis**: "What uses X?" or "What does X depend on?"

```bash
uv run python .memory/query_knowledge_graph.py "what uses MeasurementSystem"
```

Shows all components that depend on or use the target component.

**3. Type-based Discovery**: "List all configs/pipelines/propagators"

```bash
uv run python .memory/query_knowledge_graph.py "list all Pipeline"
```

Enumerates all components of a specific type.

**4. Configuration Mapping**: "What configures X?"

```bash
uv run python .memory/query_knowledge_graph.py "what configures Telescope"
```

Shows configuration classes and their relationships.

### Use Codebase-Retrieval For:

Switch to grep/glob/read when you need:

- **Implementation details** and code examples
- **Free-text queries** about algorithms or methods
- **Current code state** (if graph might be stale)
- **Queries not covered** by graph relations (e.g., "find all instances of deprecated pattern")

### Hybrid Approach (Recommended):

Combine both tools for maximum efficiency:

1. **Query knowledge graph** to identify relevant components
2. **Use codebase-retrieval** for implementation details
3. **Use Read tool** to examine specific files

**Example Workflow:**

```
User: "How does the training loop work?"

Step 1: Knowledge Graph Query
  uv run python .memory/query_knowledge_graph.py "list all Pipeline"
  → Result: PRISMRunner, PRISMTrainer

Step 2: Knowledge Graph Query
  uv run python .memory/query_knowledge_graph.py "what uses PRISMTrainer"
  → Result: PRISMRunner (orchestrates training)

Step 3: Codebase-Retrieval
  Search for "PRISMTrainer training loop implementation"
  → Identify key methods and logic

Step 4: Read Specific File
  Read prism/core/trainers.py
  → Examine detailed implementation
```

This approach typically reduces discovery time from 2-3 minutes to under 30 seconds.

### When to Use Knowledge Graph vs Grep/Glob

**Use Knowledge Graph for:**
- ✅ Component lookups: "Where is Telescope class?"
- ✅ Dependency queries: "What uses MeasurementSystem?"
- ✅ Architecture questions: "Show all propagators"
- ✅ Relationship queries: "What configures Telescope?"
- ✅ Type-based searches: "List all pipelines"

**Use Grep for:**
- ✅ String searches: "Find all TODO comments"
- ✅ Code patterns: "Search for deprecated functions"
- ✅ Usage examples: "Find examples of create_pattern"

**Use Glob for:**
- ✅ File discovery: "Find all test files"
- ✅ File patterns: "List notebooks"

### MCP Memory Tools

Query the knowledge graph using MCP memory tools:

```python
# Search for components
mcp__memory__search_nodes("Telescope")

# Get detailed information on specific nodes
mcp__memory__open_nodes(["Telescope", "TelescopeConfig", "MeasurementSystem"])

# Read entire graph
mcp__memory__read_graph()
```

### Command-Line Query Tool

Use the query tool for quick lookups:

```bash
# Find component location
uv run python tools/query_knowledge_graph.py "where is Telescope?"
# 📍 Telescope
#    Type: Class
#    Location: prism/core/instruments/telescope.py:45

# Find dependencies
uv run python tools/query_knowledge_graph.py "what uses Telescope?"
# 🔗 Telescope is used by:
#    - MeasurementSystem
#    - PRISMRunner

# List by type
uv run python tools/query_knowledge_graph.py "show all pipelines"
# 📋 Found 2 pipelines:
#    - PRISMRunner
#    - PRISMTrainer
```

### Graph Contents

The knowledge graph includes:

**Core Components:**
- Telescope, TelescopeConfig, MeasurementSystem, MeasurementSystemConfig
- Grid, Instrument (protocol)

**Propagators:**
- FreeSpacePropagator (auto-selecting)
- FresnelPropagator (near-field), FraunhoferPropagator (far-field)
- AngularSpectrumPropagator (general)

**Pipelines:**
- PRISMRunner (experiment orchestrator)
- PRISMTrainer (training loop)

**Scenarios:**
- MicroscopeScenarioConfig, DroneScenarioConfig
- MicroscopeBuilder, DroneBuilder
- get_scenario_preset (17 presets)

**Models:**
- ProgressiveDecoder (decoder architecture)
- LossAggregator (progressive loss)

**Visualization:**
- ReconstructionComparisonPlotter, LearningCurvesPlotter
- SyntheticAperturePlotter, VisualizationConfig

**Full Documentation:**
- Schema: [docs/knowledge-graph-schema.md](docs/knowledge-graph-schema.md)
- Summary: [docs/knowledge-graph-summary.md](docs/knowledge-graph-summary.md)

### Updating the Knowledge Graph

When the codebase changes, update the knowledge graph:

```bash
# Check current status
uv run python tools/update_knowledge_graph.py --status

# Incremental update (only changed files)
uv run python tools/update_knowledge_graph.py --incremental

# Full rescan
uv run python tools/update_knowledge_graph.py --full
```

---

## Quick Reference Documentation

For authoritative physical parameter lookup, see the **Technical References**:

| Reference | Quick Lookup For |
|-----------|------------------|
| [Physical Constants](docs/references/physical_constants.md) | Speed of light, wavelengths, units, conversions |
| [Microscopy Parameters](docs/references/microscopy_parameters.md) | Objective specs, NA values, resolution limits |
| [Drone Parameters](docs/references/drone_camera_parameters.md) | Lens/sensor specs, GSD calculations |
| [Resolution Limits](docs/references/optical_resolution_limits.md) | Abbe limit, Rayleigh criterion, Nyquist sampling |
| [Fresnel Regimes](docs/references/fresnel_propagation_regimes.md) | Propagation regime selection by distance |
| [Preset Catalog](docs/references/scenario_preset_catalog.md) | All 17 presets in sortable tables |

**When to Use References**:
- "What is the resolution of a 1.4 NA objective?" → [Microscopy Parameters](docs/references/microscopy_parameters.md)
- "What GSD at 100m with 50mm lens?" → [Drone Parameters](docs/references/drone_camera_parameters.md)
- "Which propagator for 50m distance?" → [Fresnel Regimes](docs/references/fresnel_propagation_regimes.md)
- "List all microscope presets" → [Preset Catalog](docs/references/scenario_preset_catalog.md)
- "What wavelength for GFP?" → [Physical Constants](docs/references/physical_constants.md)
- "Nyquist sampling for microscopy?" → [Resolution Limits](docs/references/optical_resolution_limits.md)

**References vs User Guides**:
- **References** = "What is the value?" (quick lookup, tables, formulas)
- **User Guides** = "How do I use it?" (tutorials, explanations, workflows)

**Example Use Cases**:
```python
# AI: "What is the resolution of microscope_100x_oil?"
# → Check docs/references/scenario_preset_catalog.md
# → Answer: 240 nm lateral, 1.2 µm axial @ 550nm

# AI: "Calculate GSD for 75m altitude with drone_agriculture_50m setup"
# → Check docs/references/drone_camera_parameters.md
# → Formula: GSD = H × p / f = 75 × 3.9e-6 / 0.035 = 8.4 cm

# AI: "Which propagator should I use for microscopy?"
# → Check docs/references/fresnel_propagation_regimes.md
# → Answer: Angular Spectrum (F >> 10 for all microscopy)
```

---

### ConvergenceMonitor API (v0.4.0)

The `ConvergenceMonitor` class tracks per-sample convergence behavior during training:

```python
from prism.core.convergence import ConvergenceMonitor, ConvergenceTier

# Create monitor with custom settings
monitor = ConvergenceMonitor(
    loss_threshold=1e-3,     # Target loss for convergence
    patience=10,             # Epochs without improvement before stop
    plateau_window=50,       # Window for plateau detection
    plateau_threshold=0.01,  # <1% improvement = plateau
    escalation_epochs=200,   # Epochs before tier escalation
)

# Usage in training loop
for epoch in range(max_epochs):
    loss = train_step()
    monitor.update(loss)

    if monitor.should_stop():
        if monitor.is_converged():
            print("Converged!")
        else:
            print("Plateau detected, stopping")
        break

    if monitor.should_escalate():
        # Increase learning rate, switch scheduler
        print(f"Escalating to {monitor.get_current_tier()}")

# Get statistics
stats = monitor.get_statistics()
print(f"Epochs: {stats['epochs']}, Tier: {stats['tier']}")
```

**Key Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `update(loss)` | None | Record new loss value |
| `is_converged()` | bool | Check if below threshold with stability |
| `is_plateau()` | bool | Check if loss stopped improving |
| `should_stop()` | bool | Check if training should stop |
| `should_escalate()` | bool | Check if tier should increase |
| `get_current_tier()` | ConvergenceTier | Get current optimization tier |
| `get_statistics()` | dict | Get all convergence metrics |
| `reset()` | None | Reset for new sample |

---

## GLOBAL RULES

### 🚫 FORBIDDEN

- **NEVER add shebang lines** (`#!/usr/bin/env python3`) to any files
- **NEVER use pip** - always use `uv` for package management
- **NEVER use print statements** - use `loguru` logger instead
- **NEVER commit without formatting** - run `ruff check --fix` and `ruff format` first
- **NEVER skip type hints** - all public functions must have type annotations

### ✅ REQUIRED

- **ALWAYS activate venv first** or use `uv run`
- **ALWAYS validate tensor shapes** in functions accepting tensors
- **ALWAYS use NumPy/Sphinx docstrings** for public functions
- **ALWAYS add tests** for new functionality
- **ALWAYS check diffraction regime** (Fraunhofer vs Fresnel) for optical code
- **ALWAYS format code BEFORE committing** (`ruff check --fix`, `ruff format`) - never bypass pre-commit hooks

---

## PRISM-Specific Development Best Practices

### Working with Loss Functions

When implementing new loss functions:

```python
# ✅ GOOD: Match existing patterns
class LossAggregator(nn.Module):
    """Aggregated loss with dual outputs for progressive training."""

    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type
        # Initialize loss function

    def forward(self, inputs, target, telescope=None, cntr=None):
        # Return dual losses: (loss_old, loss_new)
        return loss_old, loss_new
```

**Key requirements:**
- Must return dual losses `(loss_old, loss_new)` for progressive training
- Support both measurement-space and image-space operations
- Maintain compatibility with existing `TelescopeAggregator` interface
- Add comprehensive tests comparing to reference implementations

### Adding New Sampling Patterns

```python
# ✅ GOOD: Follow existing pattern structure
def my_sampling_pattern(
    n: int,
    radius: float,
    *,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate sampling pattern with my algorithm.

    Parameters
    ----------
    n : int
        Number of sample points
    radius : float
        Maximum radius for samples
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) with [x, y] coordinates
    """
    # Implementation
    return points
```

**Requirements:**
- Return `np.ndarray` of shape `(n, 2)`
- Support reproducible seeding
- Validate coverage efficiency
- Add visualization in `analysis_scripts/`

### Implementing New Network Architectures

When adding neural network components:

```python
# ✅ GOOD: Extend existing base classes
class MyNetwork(nn.Module):
    """
    Custom network for astronomical imaging.

    Follows PRISM conventions for complex-valued outputs.
    """

    def __init__(self, latent_dim: int = 256, output_size: int = 1024):
        super().__init__()
        # Architecture definition

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Optional input (decoder-only supports None)

        Returns:
            Reconstructed image [B, C, H, W]
        """
        # Implementation
        return output
```

**Key patterns:**
- Support decoder-only mode (x=None) like ProgressiveDecoder
- Handle complex outputs if needed
- Auto-detect and use GPU
- Validate output shapes match expected dimensions

### Configuration Management

```python
# ✅ GOOD: Use dataclass-based configs
from dataclasses import dataclass
from prism.config.base import PRISMConfig

@dataclass
class MyExperimentConfig:
    """Configuration for my experiment."""

    # Required fields with defaults
    loss_type: str = "ssim"
    window_size: int = 11
    sigma: float = 1.5

    def validate(self):
        """Validate configuration parameters."""
        if self.window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

# Usage
config = MyExperimentConfig(loss_type="ms-ssim")
config.validate()
```

**Best practices:**
- Use dataclasses for all configurations
- Provide sensible defaults
- Add validation methods
- Support YAML serialization/deserialization

### Testing Physics Simulations

```python
# ✅ GOOD: Test against analytical solutions
def test_fraunhofer_diffraction():
    """Test Fraunhofer diffraction matches analytical solution."""
    # Setup: circular aperture
    aperture = create_circular_aperture(radius=50)
    wavelength = 500e-9  # 500 nm
    distance = 10  # meters

    # Compute diffraction pattern
    pattern = telescope.diffract(aperture, wavelength, distance)

    # Compare to analytical Airy pattern
    analytical = airy_pattern(aperture.radius, wavelength, distance)

    # Should match within numerical precision
    assert np.allclose(pattern, analytical, rtol=1e-3)
```

**Testing requirements:**
- Validate against known analytical solutions
- Check dimensional analysis (units match)
- Verify Fresnel/Fraunhofer regime calculations
- Test edge cases (zero aperture, infinite distance, etc.)

### Memory Management for Large Experiments

```python
# ✅ GOOD: Manage GPU memory explicitly
import torch
from contextlib import contextmanager

@contextmanager
def managed_inference(model):
    """Context manager for memory-efficient inference."""
    model.eval()
    with torch.no_grad():
        yield model
    torch.cuda.empty_cache()

# Usage
with managed_inference(model) as m:
    reconstruction = m()
    # Use reconstruction
# Memory automatically cleared after block
```

**Best practices:**
- Use `torch.no_grad()` for inference
- Clear cache with `torch.cuda.empty_cache()` after large operations
- Delete large tensors explicitly: `del large_tensor`
- Monitor GPU memory: `torch.cuda.memory_allocated()`

### Experiment Reproducibility

```python
# ✅ GOOD: Set all random seeds
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Always call at experiment start
set_seed(args.seed)
```

**Reproducibility checklist:**
- Set random seeds at start
- Save full configuration with results
- Record package versions (uv.lock)
- Document hardware used (GPU model, CUDA version)
- Save intermediate checkpoints for long runs

### Performance Profiling

```python
# ✅ GOOD: Profile critical sections
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Critical code section
    for _ in range(10):
        output = model()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Save for visualization
prof.export_chrome_trace("trace.json")
```

**Profiling best practices:**
- Profile on representative workloads
- Focus on bottlenecks (>10% of total time)
- Compare before/after optimization
- Use PyTorch profiler for GPU-heavy code
- Use cProfile for CPU-bound code

### Documentation Standards for New Features

When adding new features, always update:

1. **Docstrings** - NumPy/Sphinx format with examples
2. **Type hints** - Full type annotations
3. **Tests** - Unit + integration tests
4. **README** - If user-facing feature
5. **Implementation guide** - For complex features (like SSIM loss)
6. **CHANGELOG.md** - Document the change
7. **Example scripts** - Show usage in `examples/`

**Example implementation guide structure:**
```markdown
# Feature Name Implementation Guide

## Overview
Brief description and motivation

## Implementation Details
Technical specifications and algorithms

## Usage Examples
Code examples showing how to use

## Testing Requirements
What needs to be tested

## References
Papers, documentation, related work
```

---

## Additional Resources

- **[README.md](README.md)**: Project overview and quick start
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Detailed contribution guidelines
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and changes
- **[docs/IMPROVEMENT_PLAN_PHASE2.md](docs/IMPROVEMENT_PLAN_PHASE2.md)**: Current development roadmap
- **[User Guide](docs/user_guide.md)**: Detailed usage instructions (if exists)

---

## Development Notes

- Always use `ProgressiveDecoder` as primary model (deprecated models have been removed in Phase 1 cleanup)
- `TelescopeAggregator` accumulates measurements - crucial for progressive learning
- System designed specifically for astronomical imaging with realistic physics
- Checkpoint resumption common for long experiments (hours to days)
- Line sampling reduces effective sample count due to spatial constraints
- Failed samples (<1-2%) are normal; high failure rate (>10%) indicates issues

---

**Last Updated**: 2025-11-30
**Version**: 0.6.0
**Replaces**: CLAUDE.md, AGENTS.md, GEMINI.md (now archived)

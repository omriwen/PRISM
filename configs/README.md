# PRISM Configuration Files

This directory contains pre-configured YAML files for common PRISM experiment scenarios.

## Quick Start with Presets

**NEW**: Use built-in presets for common workflows:

```bash
# Quick test (fast iteration, no saving)
uv run python main.py --preset quick_test --obj europa --name test

# Production run (high quality, full checkpointing)
uv run python main.py --preset production --obj titan --name titan_prod

# High quality (maximum fidelity)
uv run python main.py --preset high_quality --obj betelgeuse --name hq

# Mo-PIE baseline (traditional phase retrieval)
uv run python main_mopie.py --preset mopie_baseline --obj europa --name mopie_test

# List all available presets
python main.py --list-presets

# Show what a preset contains
python main.py --show-preset production
```

## Interactive Mode (NEW!)

For an even easier experience, use the interactive configuration wizard:

```bash
# Launch interactive mode
python main.py --interactive

# Or for Mo-PIE
python main_mopie.py --interactive
```

The wizard guides you through:
1. **Preset selection** - Choose from common experiment patterns
2. **Object selection** - Select astronomical target (Europa, Titan, etc.)
3. **Telescope parameters** - Configure samples, SNR, sampling pattern
4. **Training parameters** - Set epochs, learning rate, loss threshold
5. **Review & save** - See summary, optionally save config, and run

**Benefits**:
- No need to memorize CLI flags
- No need to write YAML files
- Guided parameter selection with helpful descriptions
- Configuration validation before running
- Optional config file saving for reproducibility

**Perfect for**:
- New users trying PRISM for the first time
- Occasional users who don't remember all parameters
- Quick experiments without diving into documentation

## Configuration Methods

PRISM supports four methods for configuration (in priority order):

### 0. Interactive Mode (Easiest for beginners)
Use `--interactive` for guided setup with no prior knowledge needed.

```bash
python main.py --interactive
```

### 1. Built-in Presets (Easiest)
Use `--preset` for common experiment patterns:
- `quick_test` - Fast iteration, no saving, 64 samples
- `production` - High quality, 200 samples, SNR=40dB
- `high_quality` - Maximum fidelity, 240 samples, SNR=50dB
- `debug` - Minimal testing, 16 samples
- `line_sampling` - Efficient line sampling mode
- `mopie_baseline` - Traditional Mo-PIE algorithm (for main_mopie.py)

Presets can be overridden:
```bash
# Use production preset but with more samples
uv run python main.py --preset production --n_samples 300 --obj europa --name custom
```

### 2. YAML Config Files (Flexible)
Use `--config` to load configurations from YAML files:

```bash
# Load from file
uv run python main.py --config configs/production_europa.yaml --name my_run

# Override specific parameters
uv run python main.py --config configs/quick_test.yaml --n_samples 128 --name custom
```

### 3. Pure Command-Line (Advanced)
Specify all parameters via CLI arguments:

```bash
uv run python main.py --obj europa --n_samples 100 --fermat --max_epochs 25 --name manual
```

**Priority order**:
1. Explicit CLI arguments (highest priority)
2. Config file or preset values
3. Argparse defaults (lowest priority)

## Preset Templates Directory

The `presets/` subdirectory contains template configuration files:

### `presets/quick_test.yaml`
Fast iteration preset for debugging:
- 64 samples with line sampling
- max_epochs = 1
- save_data = false

### `presets/production.yaml`
High-quality production settings:
- 200 point samples with Fermat spiral
- SNR = 40 dB
- max_epochs = 25
- Full checkpointing enabled

### `presets/high_quality.yaml`
Maximum fidelity preset:
- 240 samples
- SNR = 50 dB
- max_epochs = 50, n_epochs = 2000
- Tighter loss threshold (0.0005)

### `presets/debug.yaml`
Minimal testing without saving:
- 16 samples
- max_epochs = 1
- save_data = false

### `presets/mopie_baseline.yaml`
Traditional Mo-PIE algorithm (for main_mopie.py):
- 200 samples
- 500 epochs
- Fixed probe mode

### `presets/europa_custom.yaml`
Example showing config inheritance (see below)

## Config Inheritance with `extends`

**NEW**: Config files can inherit from other configs using the `extends` keyword:

```yaml
# configs/my_custom.yaml
extends: presets/production  # Inherit all production settings

comment: "Custom experiment based on production"

# Override only what you need to change
telescope:
  n_samples: 150
  snr: 45

physics:
  obj_name: europa
```

Supports:
- Preset names: `extends: quick_test`
- Relative paths: `extends: presets/production.yaml`
- Absolute paths: `extends: /full/path/to/config.yaml`

Child configs override parent values (deep merge for nested dicts).

## Legacy Config Files

These configs existed before the preset system and continue to work:

### `default.yaml`
Sensible defaults for general PRISM experiments.

### `quick_test.yaml`
Minimal configuration for fast iteration (now available as `--preset quick_test`)

### `production_europa.yaml`
High-quality Europa settings (now use `--preset production --obj europa`)

### `point_source_test.yaml`
Configuration for point source simulations

### `mopie_example.yaml`
Mo-PIE algorithm configuration (now use `--preset mopie_baseline`)

## Configuration Inspection

Explore and validate configurations before running:

```bash
# List all available presets
python main.py --list-presets

# Show details of a specific preset
python main.py --show-preset production

# Show parameters for a predefined object
python main.py --show-object europa

# Show effective config after all merging
python main.py --preset production --obj europa --show-config

# Validate config without running
python main.py --config my_config.yaml --validate-only
```

## Using Configuration Files

### Load entire configuration
```bash
uv run python main.py --config configs/default.yaml --name my_experiment
```

### Override specific parameters
CLI arguments override config file settings:
```bash
uv run python main.py --config configs/default.yaml --n_samples 150 --lr 0.0005
```

### Resume from checkpoint
Automatically loads config from checkpoint directory:
```bash
# Config is auto-loaded from runs/my_experiment/config.yaml
uv run python main.py --checkpoint my_experiment --name my_experiment_resumed

# Override specific parameters
uv run python main.py --checkpoint my_experiment --lr 0.0001 --name fine_tuned
```

### Programmatic usage
```python
from prism.config.loader import load_config

config = load_config("configs/default.yaml")
config.telescope.n_samples = 150
config.training.lr = 0.0005
```

## Creating Custom Configurations

### Method 1: Extend a preset (Recommended)
```yaml
# configs/my_experiment.yaml
extends: production

comment: "My custom experiment"

telescope:
  n_samples: 200
  snr: 35

physics:
  obj_name: titan
```

### Method 2: Start from scratch
1. Copy an existing config as a template
2. Modify parameters as needed
3. Save with a descriptive name
4. Document your changes in the `comment` field

Example custom config:
```yaml
# Custom configuration for Titan observations
comment: "High-res Titan with 300 Fermat samples"

image:
  image_size: 2048  # Higher resolution

telescope:
  n_samples: 300
  fermat_sample: true
  snr: 35

physics:
  obj_name: titan
```

## Configuration Structure

Configurations are organized into logical sections:

- **image**: Image loading and preprocessing
- **telescope**: Aperture sampling parameters
- **model**: Neural network architecture
- **training**: Optimization parameters
- **physics**: Physical constants for astronomy
- **point_source**: Point source simulation
- **mopie**: Mo-PIE-specific parameters (for main_mopie.py only)

See `prism/config/base.py` for complete parameter documentation.

## Predefined Astronomical Objects

Use `--obj` or `--obj_name` to select a predefined object:
- `europa` (default) - Jupiter's moon
- `titan` - Saturn's moon
- `betelgeuse` - Red supergiant star
- `neptune` - Ice giant planet

Each object has pre-configured physics parameters (wavelength, distance, diameter, etc.).

```bash
# Show what parameters an object sets
python main.py --show-object europa
```

## Output Structure

When you run an experiment with a config file, the following files are saved to `runs/{name}/`:

```
runs/my_experiment/
├── config.yaml              # Complete configuration used (for reproducibility)
├── args.txt                 # Human-readable parameter list
├── args.pt                  # PyTorch serialized parameters
├── checkpoint.pt            # Model state, metrics, optimizer state
├── sample_points.pt         # Sampling pattern data
├── final_reconstruction.png # Ground truth vs reconstruction comparison
├── synthetic_aperture.png   # K-space coverage visualization
├── learning_curves.png      # Loss/SSIM/PSNR progression
└── events.out.tfevents.*   # TensorBoard logs
```

**Key benefit**: The saved `config.yaml` allows exact experiment reproduction:
```bash
# Reproduce exact experiment
uv run python main.py --config runs/my_experiment/config.yaml --name reproduced
```

## Examples

### Quick iteration during development
```bash
python main.py --preset quick_test --obj europa --name dev_test
```

### Production run for publication
```bash
python main.py --preset production --obj europa --name europa_final
```

### Custom experiment with inheritance
```yaml
# configs/my_snr_study.yaml
extends: production
comment: "SNR sensitivity study"
telescope:
  snr: 30  # Lower SNR than production's 40dB
```

```bash
python main.py --config configs/my_snr_study.yaml --name snr_study_30db
```

### Compare deep learning vs traditional Mo-PIE
```bash
# Deep learning approach
python main.py --preset production --obj europa --name dl_approach

# Traditional Mo-PIE
python main_mopie.py --preset mopie_baseline --obj europa --name mopie_approach
```

## Troubleshooting

### Validate configuration before running
```bash
python main.py --config my_config.yaml --validate-only
```

### Check effective configuration
```bash
python main.py --preset production --obj europa --n_samples 150 --show-config
```

### Configuration error messages
The system now provides helpful error messages:

```
Configuration Error: Missing required physics parameters.
  → Either set 'obj_name' OR manually specify: wavelength, obj_diameter, obj_distance
  → Available obj_name values: europa, titan, betelgeuse, neptune
  → Suggestion: Try --obj europa or --preset quick_test --obj europa
```

## Migration from Legacy Configs

If you have old config files, they continue to work! But consider:

**Old way:**
```bash
python main.py --config configs/quick_test.yaml --name test
```

**New way (simpler):**
```bash
python main.py --preset quick_test --obj europa --name test
```

**New way (with customization):**
```yaml
# configs/my_test.yaml
extends: quick_test
telescope:
  n_samples: 100  # Override preset's 64
```

## Additional Resources

- **Full parameter reference**: See `prism/config/base.py`
- **Preset definitions**: See `prism/config/presets.py`
- **Example configs**: Browse `configs/` and `configs/presets/`
- **Comprehensive guide**: See `docs/CONFIGURATION_GUIDE.md`

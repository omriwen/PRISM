# Baseline Algorithms for Comparison

This directory contains reference implementations of alternative algorithms for comparison with SPIDS.

## Mo-PIE Baseline

### Active Implementation

**Use [main_mopie.py](../../main_mopie.py) at the project root** - This is the current, working Mo-PIE implementation integrated with the modern SPIDS codebase.

### Historical Implementation

[mopie_baseline.py](mopie_baseline.py) - Historical Mo-PIE implementation from legacy codebase. **Note**: This file uses old import paths and requires updates to work with the refactored spids package structure. Use `main_mopie.py` instead for a working Mo-PIE baseline.

### Algorithm Overview

Traditional phase retrieval algorithm using iterative projections. This is a physics-based optimization approach that alternately updates the object and probe estimates, unlike SPIDS which uses a generative neural network.

**Mo-PIE (Motion-aware Ptychographic Iterative Engine)**:
- Iterative phase retrieval method
- Updates complex-valued object and probe
- Physics-based constraints (Fourier ptychography)
- No neural network - pure optimization

**Key Differences from SPIDS**:
| Aspect | SPIDS | Mo-PIE |
|--------|-------|------|
| Approach | Deep learning (generative model) | Iterative optimization |
| Model | Neural network decoder | Direct object/probe update |
| Training | Backpropagation | Physics-based updates |
| Prior | Learned from data structure | Physics constraints only |
| Speed | Fast after training | Iterative convergence |

### Usage

```bash
# Run Mo-PIE reconstruction (from project root)
uv run python main_mopie.py \
    --obj_name europa \
    --n_samples 100 \
    --fermat \
    --name mopie_europa_test

# With custom parameters
uv run python main_mopie.py \
    --obj_name titan \
    --n_samples 200 \
    --fermat \
    --lr_obj 1.0 \
    --lr_probe 1.0 \
    --fix_probe \
    --name mopie_titan_custom
```

### Command-line Arguments

- `--obj_name`: Target object (europa, titan, betelgeuse, neptune)
- `--n_samples`: Number of aperture measurements
- `--fermat`: Use Fermat spiral sampling (recommended)
- `--lr_obj`: Object update learning rate (default: 0.5)
- `--lr_probe`: Probe update learning rate (default: 0.1)
- `--fix_probe`: Fix probe (don't update), only estimate object
- `--name`: Experiment name for saving results

### Comparing SPIDS vs Mo-PIE

Run both algorithms on the same configuration:

```bash
# 1. Run SPIDS
uv run python main.py \
    --obj_name europa \
    --n_samples 100 \
    --fermat \
    --name spids_europa_comparison

# 2. Run Mo-PIE
uv run python main_mopie.py \
    --obj_name europa \
    --n_samples 100 \
    --fermat \
    --name mopie_europa_comparison

# 3. Compare results
# Both save to runs/ directory
# - runs/spids_europa_comparison/
# - runs/mopie_europa_comparison/

# Load and compare metrics:
# - SSIM (structural similarity)
# - RMSE (root mean squared error)
# - Convergence curves
# - Computational time
```

### Expected Performance

**Reconstruction Quality**:
- Mo-PIE typically achieves good quality with sufficient iterations
- SPIDS may be faster but requires more samples initially
- Quality depends on sampling pattern and number of measurements

**Computational Cost**:
- Mo-PIE: ~0.1-1 second per iteration per sample
- SPIDS: Slower training but faster inference
- Mo-PIE better for small experiments, SPIDS better at scale

### Output Structure

Results saved to `runs/{experiment_name}/`:
- `checkpoint.pt`: Final object estimate and probe
- `args.txt`: Experiment parameters
- `losses.png`: Convergence curves
- `reconstruction.png`: Final reconstructed object

### Scientific References

**Mo-PIE Algorithm**:
- Maiden, A. M., & Rodenburg, J. M. (2009). "An improved ptychographical phase retrieval algorithm for diffractive imaging." *Ultramicroscopy*, 109(10), 1256-1262.

**Fourier Ptychography**:
- Zheng, G., Horstmeyer, R., & Yang, C. (2013). "Wide-field, high-resolution Fourier ptychographic microscopy." *Nature Photonics*, 7(9), 739-745.

**Phase Retrieval**:
- Fienup, J. R. (1982). "Phase retrieval algorithms: a comparison." *Applied Optics*, 21(15), 2758-2769.

## Adding New Baselines

To add another comparison algorithm:

1. Create new file: `examples/baselines/{algorithm}_baseline.py`
2. Implement with same interface as Mo-PIE baseline
3. Use same command-line arguments where possible
4. Save results to `runs/` directory
5. Document in this README

Example algorithms to consider:
- Gerchberg-Saxton (basic phase retrieval)
- Hybrid Input-Output (HIO)
- Difference Map (DM)
- RAAR (Relaxed Averaged Alternating Reflections)

## Notes

- Baselines maintained for scientific validation
- Code quality may vary (historical implementations)
- Use same test objects as SPIDS for fair comparison
- Consider sampling pattern effects (Fermat vs random)
- Report multiple metrics (SSIM, RMSE, convergence time)

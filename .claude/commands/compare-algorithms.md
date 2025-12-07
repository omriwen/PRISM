# Compare Algorithms

Run PRISM and Mo-PIE algorithms on the same configuration and compare reconstruction quality metrics.

## Arguments

`$ARGUMENTS` should contain:
- `object_name`: Name of the test object (e.g., `europa`, `moon`, `astronaut`)
- `n_samples`: Number of samples to use (e.g., `100`, `500`)
- `--fermat` (optional): Use Fermat spiral sampling pattern
- `--skip-run` (optional): Skip running, just compare existing checkpoints

Examples:
```bash
/compare-algorithms europa 100 --fermat
/compare-algorithms moon 500
/compare-algorithms europa 100 --skip-run
```

## Instructions

### Step 1: Parse Arguments

Extract from `$ARGUMENTS`:
- `object_name`: Required, first positional argument
- `n_samples`: Required, second positional argument
- `use_fermat`: True if `--fermat` flag present
- `skip_run`: True if `--skip-run` flag present

### Step 2: Set Up Experiment Names

Generate standardized experiment names:
```python
prism_name = f"compare_prism_{object_name}_{n_samples}"
mopie_name = f"compare_mopie_{object_name}_{n_samples}"
```

### Step 3: Run Both Algorithms (unless --skip-run)

#### Run PRISM
```bash
uv run python main.py \
    --obj_name {object_name} \
    --n_samples {n_samples} \
    --name {prism_name} \
    --fermat  # if --fermat flag
```

#### Run Mo-PIE
```bash
uv run python main_mopie.py \
    --obj_name {object_name} \
    --n_samples {n_samples} \
    --name {mopie_name} \
    --fermat  # if --fermat flag
```

Wait for both to complete before proceeding.

### Step 4: Load Checkpoints

Load checkpoints from both experiments:

```python
import torch
from pathlib import Path

# Find checkpoint paths
runs_dir = Path("runs")
prism_ckpt = runs_dir / prism_name / "checkpoint.pt"
mopie_ckpt = runs_dir / mopie_name / "checkpoint.pt"

# Load checkpoints
prism_data = torch.load(prism_ckpt, map_location='cpu')
mopie_data = torch.load(mopie_ckpt, map_location='cpu')
```

### Step 5: Extract Metrics

Extract metrics from both checkpoints:

```python
def extract_metrics(checkpoint_data):
    """Extract metrics from checkpoint."""
    metrics = {}

    # Final metrics (may be in different locations)
    if 'ssim' in checkpoint_data:
        metrics['ssim'] = checkpoint_data['ssim']
    elif 'metrics' in checkpoint_data:
        metrics['ssim'] = checkpoint_data['metrics'].get('ssim')

    if 'psnr' in checkpoint_data:
        metrics['psnr'] = checkpoint_data['psnr']
    elif 'metrics' in checkpoint_data:
        metrics['psnr'] = checkpoint_data['metrics'].get('psnr')

    if 'rmse' in checkpoint_data:
        metrics['rmse'] = checkpoint_data['rmse']
    elif 'metrics' in checkpoint_data:
        metrics['rmse'] = checkpoint_data['metrics'].get('rmse')

    # Training info
    metrics['final_loss'] = checkpoint_data.get('loss', checkpoint_data.get('final_loss'))
    metrics['failed_samples'] = len(checkpoint_data.get('failed_samples', []))

    return metrics

prism_metrics = extract_metrics(prism_data)
mopie_metrics = extract_metrics(mopie_data)
```

### Step 6: Generate Comparison Report

Create a formatted comparison table:

```markdown
## PRISM vs Mo-PIE Algorithm Comparison

### Configuration
- **Object**: {object_name}
- **Samples**: {n_samples}
- **Sampling**: {Fermat spiral | Random}

### Results

| Metric | PRISM | Mo-PIE | Difference | Winner |
|--------|-------|--------|------------|--------|
| SSIM | {prism_ssim:.4f} | {mopie_ssim:.4f} | {diff:+.4f} | {winner} |
| PSNR (dB) | {prism_psnr:.2f} | {mopie_psnr:.2f} | {diff:+.2f} | {winner} |
| RMSE | {prism_rmse:.6f} | {mopie_rmse:.6f} | {diff:+.6f} | {winner} |
| Final Loss | {prism_loss:.6f} | {mopie_loss:.6f} | {diff:+.6f} | {winner} |
| Failed Samples | {prism_failed} | {mopie_failed} | {diff:+d} | {winner} |

### Winner Determination
- **SSIM**: Higher is better
- **PSNR**: Higher is better
- **RMSE**: Lower is better
- **Final Loss**: Lower is better
- **Failed Samples**: Lower is better

### Overall Assessment
[Summarize which algorithm performed better and by how much]

### Checkpoint Locations
- PRISM: `runs/{prism_name}/checkpoint.pt`
- Mo-PIE: `runs/{mopie_name}/checkpoint.pt`

### Next Steps
- Use `/analyze-checkpoint` for detailed analysis of either checkpoint
- Run `prism compare {prism_name} {mopie_name}` for visual comparison
```

## Error Handling

### Missing Checkpoints
If a checkpoint doesn't exist:
```markdown
**Error**: Checkpoint not found at `{path}`

Suggestions:
1. Run the experiment first (remove --skip-run flag)
2. Check if the experiment name is correct
3. Verify the runs/ directory exists
```

### Invalid Arguments
If arguments are missing or invalid:
```markdown
**Error**: Invalid arguments

Usage: /compare-algorithms <object_name> <n_samples> [--fermat] [--skip-run]

Examples:
  /compare-algorithms europa 100 --fermat
  /compare-algorithms moon 500
```

## Performance Notes

- **100 samples**: ~2-5 minutes per algorithm
- **500 samples**: ~10-20 minutes per algorithm
- **1000 samples**: ~30-60 minutes per algorithm

Consider using `--skip-run` to re-analyze existing experiments without re-running.

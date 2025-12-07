# Analyze Checkpoint

Analyze a PRISM or Mo-PIE checkpoint file and display key metrics, configuration, and quality assessment.

## Arguments

`$ARGUMENTS` should contain:
- Path to checkpoint file (required)
- `--show-image` (optional): Extract and display reconstruction image

Examples:
```bash
/analyze-checkpoint runs/experiment/checkpoint.pt
/analyze-checkpoint runs/europa_100samples
/analyze-checkpoint runs/my_experiment/checkpoint.pt --show-image
```

## Instructions

### Step 1: Parse Arguments

Extract from `$ARGUMENTS`:
- `checkpoint_path`: Path to checkpoint file or experiment directory
- `show_image`: True if `--show-image` flag present

If path is a directory, append `checkpoint.pt`:
```python
from pathlib import Path

path = Path(checkpoint_path)
if path.is_dir():
    path = path / "checkpoint.pt"
```

### Step 2: Load Checkpoint

Load the checkpoint file:

```python
import torch

# Load checkpoint
checkpoint = torch.load(path, map_location='cpu')

# Also try to load args.pt if it exists
args_path = path.parent / "args.pt"
if args_path.exists():
    args = torch.load(args_path, map_location='cpu')
else:
    # Try to get args from checkpoint
    args = checkpoint.get('freq_pattern_args', checkpoint.get('args', {}))
```

### Step 3: Extract Metadata

```python
def extract_metadata(checkpoint, path):
    """Extract experiment metadata."""
    metadata = {
        'experiment_name': path.parent.name,
        'checkpoint_path': str(path),
    }

    # Timestamp (from file or checkpoint)
    if 'timestamp' in checkpoint:
        metadata['timestamp'] = checkpoint['timestamp']
    else:
        import os
        mtime = os.path.getmtime(path)
        from datetime import datetime
        metadata['timestamp'] = datetime.fromtimestamp(mtime).isoformat()

    # Version info
    metadata['prism_version'] = checkpoint.get('version', 'Unknown')

    return metadata
```

### Step 4: Extract Metrics

```python
def extract_metrics(checkpoint):
    """Extract final metrics from checkpoint."""
    metrics = {}

    # Try multiple locations for metrics
    if 'metrics' in checkpoint:
        m = checkpoint['metrics']
        metrics['ssim'] = m.get('ssim', m.get('final_ssim'))
        metrics['psnr'] = m.get('psnr', m.get('final_psnr'))
        metrics['rmse'] = m.get('rmse', m.get('final_rmse'))
    else:
        metrics['ssim'] = checkpoint.get('ssim', checkpoint.get('final_ssim'))
        metrics['psnr'] = checkpoint.get('psnr', checkpoint.get('final_psnr'))
        metrics['rmse'] = checkpoint.get('rmse', checkpoint.get('final_rmse'))

    metrics['loss'] = checkpoint.get('loss', checkpoint.get('final_loss'))

    return metrics
```

### Step 5: Extract Training Info

```python
def extract_training_info(checkpoint):
    """Extract training information."""
    info = {}

    # Sample counts
    info['total_samples'] = checkpoint.get('n_samples', checkpoint.get('total_samples'))
    info['last_sample_idx'] = checkpoint.get('last_sample_idx', checkpoint.get('sample_idx'))

    # Failed samples
    failed = checkpoint.get('failed_samples', [])
    info['failed_samples'] = failed
    info['failed_count'] = len(failed)
    info['failure_rate'] = len(failed) / info['total_samples'] * 100 if info['total_samples'] else 0

    # Epochs (for Mo-PIE style training)
    info['epochs'] = checkpoint.get('epochs', checkpoint.get('n_epochs'))

    return info
```

### Step 6: Extract Configuration

```python
def extract_config(checkpoint, args):
    """Extract experiment configuration."""
    config = {}

    # From args
    if isinstance(args, dict):
        config['object_name'] = args.get('obj_name', args.get('object_name'))
        config['n_samples'] = args.get('n_samples')
        config['pattern'] = args.get('pattern_preset', args.get('pattern'))
        config['learning_rate'] = args.get('lr', args.get('learning_rate'))
        config['loss_type'] = args.get('loss_type', args.get('loss'))
    elif hasattr(args, '__dict__'):
        # Namespace object
        config['object_name'] = getattr(args, 'obj_name', None)
        config['n_samples'] = getattr(args, 'n_samples', None)
        config['pattern'] = getattr(args, 'pattern_preset', None)
        config['learning_rate'] = getattr(args, 'lr', None)
        config['loss_type'] = getattr(args, 'loss_type', None)

    return config
```

### Step 7: Quality Assessment

```python
def assess_quality(metrics):
    """Provide quality assessment based on metrics."""
    assessment = []

    # SSIM assessment
    ssim = metrics.get('ssim')
    if ssim is not None:
        if ssim >= 0.95:
            assessment.append(("SSIM", "Excellent", f"{ssim:.4f} (>=0.95)"))
        elif ssim >= 0.90:
            assessment.append(("SSIM", "Good", f"{ssim:.4f} (0.90-0.95)"))
        elif ssim >= 0.80:
            assessment.append(("SSIM", "Fair", f"{ssim:.4f} (0.80-0.90)"))
        else:
            assessment.append(("SSIM", "Poor", f"{ssim:.4f} (<0.80)"))

    # PSNR assessment
    psnr = metrics.get('psnr')
    if psnr is not None:
        if psnr >= 40:
            assessment.append(("PSNR", "Excellent", f"{psnr:.2f} dB (>=40)"))
        elif psnr >= 30:
            assessment.append(("PSNR", "Good", f"{psnr:.2f} dB (30-40)"))
        elif psnr >= 20:
            assessment.append(("PSNR", "Fair", f"{psnr:.2f} dB (20-30)"))
        else:
            assessment.append(("PSNR", "Poor", f"{psnr:.2f} dB (<20)"))

    return assessment
```

### Step 8: Generate Report

Output a comprehensive analysis:

```markdown
## Checkpoint Analysis: {experiment_name}

### Metadata
- **Experiment**: {experiment_name}
- **Checkpoint**: {checkpoint_path}
- **Timestamp**: {timestamp}
- **PRISM Version**: {version}

### Final Metrics
| Metric | Value |
|--------|-------|
| SSIM | {ssim:.4f} ({ssim_percent:.1f}% similarity) |
| PSNR | {psnr:.2f} dB |
| RMSE | {rmse:.6f} |
| Final Loss | {loss:.6f} |

### Training Information
- **Total Samples**: {total_samples}
- **Last Sample Index**: {last_idx}
- **Failed Samples**: {failed_count} / {total_samples} ({failure_rate:.1f}%)
- **Epochs**: {epochs} (if applicable)

### Failed Samples
{list of failed sample indices, or "None" if empty}

### Configuration Summary
| Parameter | Value |
|-----------|-------|
| Object | {object_name} |
| Samples | {n_samples} |
| Pattern | {pattern} |
| Learning Rate | {lr} |
| Loss Type | {loss_type} |

### Quality Assessment
| Metric | Rating | Details |
|--------|--------|---------|
| SSIM | {rating} | {details} |
| PSNR | {rating} | {details} |
| Failure Rate | {rating} | {details} |

### Recommendations
{Based on quality assessment, provide recommendations}
- If SSIM < 0.90: "Consider more training iterations or adjusting learning rate"
- If failure rate > 5%: "High failure rate indicates potential convergence issues"
- If PSNR < 30: "Low signal quality, check input data or model parameters"
```

### Step 9: Show Image (if requested)

If `--show-image` flag is present:

```python
def extract_reconstruction(checkpoint, save_path):
    """Extract and save reconstruction image."""
    import matplotlib.pyplot as plt

    # Try to find reconstruction tensor
    recon = None
    for key in ['reconstruction', 'final_reconstruction', 'image', 'output']:
        if key in checkpoint:
            recon = checkpoint[key]
            break

    if recon is None:
        return None

    # Convert to numpy
    if hasattr(recon, 'cpu'):
        recon = recon.cpu().numpy()

    # Handle different shapes
    if recon.ndim == 4:  # (B, C, H, W)
        recon = recon[0]
    if recon.ndim == 3 and recon.shape[0] in [1, 3]:  # (C, H, W)
        recon = recon.transpose(1, 2, 0)
    if recon.shape[-1] == 1:  # Single channel
        recon = recon.squeeze(-1)

    # Save image
    plt.figure(figsize=(8, 8))
    plt.imshow(recon, cmap='gray' if recon.ndim == 2 else None)
    plt.title(f"Reconstruction: {save_path.stem}")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

    return save_path
```

## Error Handling

### File Not Found
```markdown
**Error**: Checkpoint not found at `{path}`

Suggestions:
1. Check the path is correct
2. List available experiments: `ls runs/`
3. Provide full path including `checkpoint.pt`
```

### Corrupted Checkpoint
```markdown
**Error**: Failed to load checkpoint: {error}

The checkpoint may be corrupted or incomplete.

Suggestions:
1. Check if training completed successfully
2. Look for backup checkpoints: `ls {dir}/*.pt`
3. Re-run the experiment
```

### Missing Metrics
```markdown
**Warning**: Some metrics not found in checkpoint

This may be an older checkpoint format. Available keys:
{list of available keys}
```

## Example Output

```markdown
## Checkpoint Analysis: europa_100samples

### Metadata
- **Experiment**: europa_100samples
- **Checkpoint**: runs/europa_100samples/checkpoint.pt
- **Timestamp**: 2025-12-07T14:32:15
- **PRISM Version**: 0.6.0

### Final Metrics
| Metric | Value |
|--------|-------|
| SSIM | 0.9823 (98.2% similarity) |
| PSNR | 42.15 dB |
| RMSE | 0.001234 |
| Final Loss | 0.000234 |

### Training Information
- **Total Samples**: 100
- **Last Sample Index**: 99
- **Failed Samples**: 2 / 100 (2.0%)

### Quality Assessment
| Metric | Rating | Details |
|--------|--------|---------|
| SSIM | Excellent | 0.9823 (>=0.95) |
| PSNR | Excellent | 42.15 dB (>=40) |
| Failure Rate | Good | 2.0% (<5%) |

### Recommendations
- Reconstruction quality is excellent
- Consider this as a baseline for comparison
```

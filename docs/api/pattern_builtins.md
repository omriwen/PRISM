# prism.core.pattern_builtins

Builtin pattern functions.

Wraps existing pattern generators in the pattern function interface.

## Classes

## Functions

### fermat_builtin

```python
fermat_builtin(config: Any) -> torch.Tensor
```

Fermat spiral pattern (golden angle spiral).

Provides optimal k-space coverage with logarithmic spiral sampling.
This is the recommended pattern for most PRISM applications.

Uses parameters from config:
- n_samples: Number of sampling positions
- roi_diameter: K-space region diameter
- sample_length: Line length (0 for point sampling)
- samples_r_cutoff: Maximum radius for sample centers
- line_angle: Fixed line angle (None for random)

### random_builtin

```python
random_builtin(config: Any) -> torch.Tensor
```

Random uniform sampling pattern.

Samples positions uniformly at random within k-space region.
Generally less optimal than Fermat spiral but useful for comparison.

Uses parameters from config:
- n_samples: Number of sampling positions
- sample_length: Line length (0 for point sampling)
- roi_diameter: K-space region diameter

### star_builtin

```python
star_builtin(config: Any) -> torch.Tensor
```

Star pattern (radial lines from center).

Generates evenly-spaced radial lines, useful for testing
rotational symmetry and basic reconstruction capabilities.

Uses parameters from config:
- n_angs: Number of radial angles
- sample_length: Line length
- roi_diameter: K-space region diameter

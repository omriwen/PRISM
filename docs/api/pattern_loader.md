# prism.core.pattern_loader

Pattern function loader and executor.

Provides infrastructure for loading pattern generation functions from Python files
and executing them to create sampling patterns.

## Classes

### PatternLoader

```python
PatternLoader() -> None
```

Loads and executes pattern generation functions.

Pattern functions must have signature:
    def generate_pattern(config) -> torch.Tensor

And return tensor of shape (n_samples, 1, 2) or (n_samples, 2, 2).

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

##### `execute_pattern_function`

Execute pattern function and validate output.

Args:
    func: Pattern generation function
    config: Configuration object

Returns:
    Generated sample positions tensor

Raises:
    ValueError: If output shape is invalid
    TypeError: If output is not a tensor

##### `load_pattern_function`

Load pattern function from specification.

Args:
    pattern_spec: Either:
        - "builtin:name" for builtin patterns
        - "/path/to/pattern.py" for custom patterns

Returns:
    (function, metadata) tuple where metadata contains:
        - 'source': Source code or path
        - 'hash': Hash of source for verification
        - 'docstring': Function docstring
        - 'is_builtin': Boolean flag

## Functions

### load_and_generate_pattern

```python
load_and_generate_pattern(pattern_spec: str, config: Any) -> tuple[torch.Tensor, dict[str, typing.Any]]
```

Convenience function to load and execute pattern in one call.

Args:
    pattern_spec: Pattern specification (builtin:name or file path)
    config: Configuration object

Returns:
    (sample_centers, metadata) tuple

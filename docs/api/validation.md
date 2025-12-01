# prism.config.validation

Enhanced configuration validation with intelligent error messages and suggestions.

This module provides comprehensive validation for PRISM configuration parameters
with helpful error messages, spelling suggestions, and detailed guidance.

## Classes

### ConfigValidator

```python
ConfigValidator(/, *args, **kwargs)
```

Validates configuration with helpful error messages and suggestions.

#### Methods

##### `suggest_correction`

Suggest closest match using Levenshtein distance.

Parameters
----------
invalid : str
    Invalid value provided by user
valid_options : List[str]
    List of valid options
n : int, optional
    Number of suggestions to return, by default 1
cutoff : float, optional
    Similarity threshold (0-1), by default 0.6

Returns
-------
Optional[str]
    Closest match if found, None otherwise

### ValidationError

```python
ValidationError(/, *args, **kwargs)
```

Enhanced validation error with suggestions and detailed guidance.

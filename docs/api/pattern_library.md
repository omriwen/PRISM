# prism.core.pattern_library

Pattern library and metadata registry.

Provides centralized metadata and information about available sampling patterns.

## Classes

### PatternInfo

```python
PatternInfo(name: str, spec: str, description: str, properties: List[str], parameters: List[str], reference: str = '', recommended: bool = False) -> None
```

Metadata and information about a sampling pattern.

Attributes
----------
name : str
    Display name of the pattern
spec : str
    Pattern specification string (e.g., "builtin:fermat")
description : str
    Brief description of the pattern
properties : List[str]
    Pattern properties (e.g., 'uniform', 'incoherent', 'radial')
parameters : List[str]
    Required/optional configuration parameters
reference : str
    Academic reference or citation (if applicable)
recommended : bool
    Whether this is a recommended pattern for general use

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

### PatternLibrary

```python
PatternLibrary(/, *args, **kwargs)
```

Central registry of sampling patterns with metadata.

Provides information about built-in patterns and utilities for
listing, searching, and comparing patterns.

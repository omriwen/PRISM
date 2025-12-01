# prism.types

Type definitions for PRISM.

This module provides type aliases and protocols used throughout the PRISM codebase
for better type safety and documentation.

## Classes

### HasForward

```python
HasForward(*args, **kwargs)
```

Protocol for objects with a forward method (like nn.Module).

#### Methods

##### `__init__`

No documentation available.

##### `forward`

Forward pass.

### LRScheduler

```python
LRScheduler(*args, **kwargs)
```

Protocol for learning rate schedulers.

#### Methods

##### `__init__`

No documentation available.

##### `get_last_lr`

Get current learning rate.

##### `step`

Update learning rate.

### Optimizer

```python
Optimizer(*args, **kwargs)
```

Protocol for PyTorch optimizers.

#### Methods

##### `__init__`

No documentation available.

##### `step`

Update parameters.

##### `zero_grad`

Clear gradients.

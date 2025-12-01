# prism.config.loader

Configuration loading and saving utilities.

## Classes

## Functions

### args_to_config

```python
args_to_config(args: argparse.Namespace) -> prism.config.base.PRISMConfig
```

Convert argparse.Namespace to PRISMConfig.

Args:
    args: argparse.Namespace from argument parser

Returns:
    PRISMConfig object

### config_to_args

```python
config_to_args(config: prism.config.base.PRISMConfig) -> argparse.Namespace
```

Convert PRISMConfig to argparse.Namespace for backward compatibility.

Args:
    config: PRISMConfig object

Returns:
    argparse.Namespace with all parameters

### load_config

```python
load_config(config_path: str, _visited: Optional[set] = None) -> prism.config.base.PRISMConfig
```

Load PRISM configuration from YAML file with support for inheritance.

Supports the 'extends' keyword for config inheritance:
- extends: path/to/parent.yaml (relative path)
- extends: presets/quick_test.yaml (preset template)
- extends: quick_test (preset name - will be resolved)

Child configs override parent values (deep merge for nested dicts).

Args:
    config_path: Path to YAML configuration file
    _visited: Internal parameter to track visited configs (prevents circular refs)

Returns:
    PRISMConfig object

Raises:
    FileNotFoundError: If config file doesn't exist
    ValueError: If config is invalid or has circular inheritance

### merge_config_with_args

```python
merge_config_with_args(config: prism.config.base.PRISMConfig, cli_args: argparse.Namespace, cli_provided_args: Optional[set] = None) -> argparse.Namespace
```

Merge config with CLI arguments (CLI args take precedence).

Args:
    config: Base configuration from YAML
    cli_args: Command-line arguments from argparse
    cli_provided_args: Set of argument names that were explicitly provided on CLI.
                      If None, will attempt to detect from sys.argv

Returns:
    Merged argparse.Namespace

### save_config

```python
save_config(config: prism.config.base.PRISMConfig, output_path: str, minimal: bool = False) -> None
```

Save PRISM configuration to YAML file.

Args:
    config: PRISMConfig object to save
    output_path: Path to output YAML file
    minimal: If True, save only non-default values (cleaner, more readable)
            If False, save all values (complete, verbose)

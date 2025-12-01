# prism.cli.animate

CLI command for generating training animations from PRISM experiments.

Usage:
    python -m prism.cli.animate runs/experiment --output training.mp4 [options]

## Classes

## Functions

### animate_command

```python
animate_command(args: argparse.Namespace) -> int
```

Execute animate command.

Parameters
----------
args : argparse.Namespace
    Parsed command-line arguments

Returns
-------
int
    Exit code (0 for success, 1 for error)

### create_animate_parser

```python
create_animate_parser() -> argparse.ArgumentParser
```

Create argument parser for animate command.

Returns
-------
argparse.ArgumentParser
    Configured argument parser for animation generation

### main

```python
main(argv: Optional[List[str]] = None) -> int
```

Main entry point for animate command.

Parameters
----------
argv : List[str], optional
    Command-line arguments. If None, uses sys.argv.

Returns
-------
int
    Exit code

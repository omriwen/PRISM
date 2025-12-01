# prism.cli.compare

CLI command for comparing PRISM experiments.

Usage:
    python -m prism.cli.compare runs/exp1 runs/exp2 [options]

## Classes

## Functions

### compare_command

```python
compare_command(args: argparse.Namespace) -> int
```

Execute comparison command.

Parameters
----------
args : argparse.Namespace
    Parsed command-line arguments

Returns
-------
int
    Exit code (0 for success, 1 for error)

### create_compare_parser

```python
create_compare_parser() -> argparse.ArgumentParser
```

Create argument parser for compare command.

Returns
-------
argparse.ArgumentParser
    Configured argument parser for experiment comparison

### main

```python
main(argv: Optional[List[str]] = None) -> int
```

Main entry point for compare command.

Parameters
----------
argv : List[str], optional
    Command-line arguments. If None, uses sys.argv.

Returns
-------
int
    Exit code

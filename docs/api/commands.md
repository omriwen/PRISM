# prism.cli.patterns.commands

Command handlers for patterns CLI.

## Classes

## Functions

### compare_command

```python
compare_command(args: argparse.Namespace, console: rich.console.Console) -> int
```

Compare multiple patterns side-by-side.

Parameters
----------
args : argparse.Namespace
    Parsed command-line arguments
console : Console
    Rich console for formatted output

Returns
-------
int
    Exit code (0 for success, 1 for error)

### gallery_command

```python
gallery_command(args: argparse.Namespace, console: rich.console.Console) -> int
```

Generate HTML gallery of all patterns.

Parameters
----------
args : argparse.Namespace
    Parsed command-line arguments
console : Console
    Rich console for formatted output

Returns
-------
int
    Exit code (0 for success, 1 for error)

### list_command

```python
list_command(args: argparse.Namespace, console: rich.console.Console) -> int
```

List all available patterns.

Parameters
----------
args : argparse.Namespace
    Parsed command-line arguments
console : Console
    Rich console for formatted output

Returns
-------
int
    Exit code (0 for success)

### patterns_command

```python
patterns_command(args: argparse.Namespace) -> int
```

Execute patterns command based on subcommand.

Parameters
----------
args : argparse.Namespace
    Parsed command-line arguments

Returns
-------
int
    Exit code (0 for success, 1 for error)

### show_command

```python
show_command(args: argparse.Namespace, console: rich.console.Console) -> int
```

Visualize a specific pattern.

Parameters
----------
args : argparse.Namespace
    Parsed command-line arguments
console : Console
    Rich console for formatted output

Returns
-------
int
    Exit code (0 for success, 1 for error)

### stats_command

```python
stats_command(args: argparse.Namespace, console: rich.console.Console) -> int
```

Show statistics for a pattern.

Parameters
----------
args : argparse.Namespace
    Parsed command-line arguments
console : Console
    Rich console for formatted output

Returns
-------
int
    Exit code (0 for success, 1 for error)

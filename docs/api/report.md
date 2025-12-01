# prism.cli.report

Report Generation CLI
=====================

Command-line interface for generating experiment reports.

## Classes

## Functions

### add_report_parser

```python
add_report_parser(subparsers: Any) -> None
```

Add report command to CLI parser.

Parameters
----------
subparsers : argparse._SubParsersAction
    Subparser action to add report command to

### main

```python
main() -> None
```

Main entry point for standalone execution.

### report_command

```python
report_command(args: argparse.Namespace) -> None
```

Execute report generation command.

Parameters
----------
args : argparse.Namespace
    Parsed command-line arguments

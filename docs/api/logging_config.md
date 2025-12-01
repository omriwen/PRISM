# prism.utils.logging_config

Centralized logging configuration for PRISM.

This module configures loguru for consistent logging across the package.

## Classes

## Functions

### setup_logging

```python
setup_logging(level: str = 'INFO', log_file: Optional[pathlib.Path] = None, show_time: bool = True, show_level: bool = True) -> Any
```

Configure logging for PRISM package.

Parameters
----------
level : str, default="INFO"
    Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
log_file : Path | None, default=None
    If provided, also log to this file
show_time : bool, default=True
    Whether to show timestamp in logs
show_level : bool, default=True
    Whether to show log level in logs

Returns
-------
logger
    Configured loguru logger instance

Examples
--------
>>> from prism.utils.logging_config import setup_logging
>>> setup_logging(level="DEBUG")
>>> from loguru import logger
>>> logger.info("This will be logged")

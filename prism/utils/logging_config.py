"""
Centralized logging configuration for SPIDS.

This module configures loguru for consistent logging across the package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    show_time: bool = True,
    show_level: bool = True,
) -> Any:
    """
    Configure logging for SPIDS package.

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
    """
    # Remove default handler
    logger.remove()

    # Build format string
    format_parts = []
    if show_time:
        format_parts.append("<green>{time:YYYY-MM-DD HH:mm:ss}</green>")
    if show_level:
        format_parts.append("<level>{level: <8}</level>")
    format_parts.append("<level>{message}</level>")

    format_str = " | ".join(format_parts)

    # Add console handler
    logger.add(
        sys.stderr,
        format=format_str,
        level=level,
        colorize=True,
    )

    # Add file handler if requested
    if log_file:
        logger.add(
            log_file,
            format=format_str,
            level=level,
            rotation="10 MB",
            retention="7 days",
        )

    return logger


# Default logger for package
_default_logger = setup_logging()

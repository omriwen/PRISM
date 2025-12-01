"""
Pattern function loader and executor.

Provides infrastructure for loading pattern generation functions from Python files
and executing them to create sampling patterns.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable

import torch


# Type alias for pattern functions
PatternFunction = Callable[[Any], torch.Tensor]


class PatternLoader:
    """
    Loads and executes pattern generation functions.

    Pattern functions must have signature:
        def generate_pattern(config) -> torch.Tensor

    And return tensor of shape (n_samples, 1, 2) or (n_samples, 2, 2).
    """

    def __init__(self) -> None:
        self._builtin_patterns = self._register_builtins()

    def _register_builtins(self) -> dict[str, PatternFunction]:
        """Register builtin patterns (fermat, star, random)."""
        from prism.core.pattern_builtins import (
            fermat_builtin,
            random_builtin,
            star_builtin,
        )

        return {
            "fermat": fermat_builtin,
            "star": star_builtin,
            "random": random_builtin,
        }

    def load_pattern_function(self, pattern_spec: str) -> tuple[PatternFunction, dict[str, Any]]:
        """
        Load pattern function from specification.

        Args:
            pattern_spec: Either:
                - "builtin:name" for builtin patterns
                - "/path/to/pattern.py" for custom patterns

        Returns:
            (function, metadata) tuple where metadata contains:
                - 'source': Source code or path
                - 'hash': Hash of source for verification
                - 'docstring': Function docstring
                - 'is_builtin': Boolean flag
        """
        if pattern_spec.startswith("builtin:"):
            name = pattern_spec.split(":", 1)[1]
            return self._load_builtin(name)
        else:
            return self._load_from_file(pattern_spec)

    def _load_builtin(self, name: str) -> tuple[PatternFunction, dict[str, Any]]:
        """Load builtin pattern."""
        if name not in self._builtin_patterns:
            available = ", ".join(self._builtin_patterns.keys())
            raise ValueError(f"Unknown builtin pattern '{name}'. Available: {available}")

        func = self._builtin_patterns[name]
        metadata = {
            "source": f"builtin:{name}",
            "hash": None,
            "docstring": inspect.getdoc(func),
            "is_builtin": True,
        }
        return func, metadata

    def _load_from_file(self, file_path: str) -> tuple[PatternFunction, dict[str, Any]]:
        """Load pattern function from Python file."""
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Pattern file not found: {path}")

        if path.suffix != ".py":
            raise ValueError(f"Pattern file must be .py, got: {path.suffix}")

        # Read source for hashing and storage
        source_code = path.read_text()
        source_hash = hashlib.sha256(source_code.encode()).hexdigest()[:16]

        # Load module
        spec = importlib.util.spec_from_file_location("pattern_module", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load pattern from {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get generate_pattern function
        if not hasattr(module, "generate_pattern"):
            raise AttributeError(
                f"Pattern file must define 'generate_pattern' function. Found: {dir(module)}"
            )

        func = module.generate_pattern

        # Validate signature
        sig = inspect.signature(func)
        if len(sig.parameters) != 1:
            raise TypeError(
                f"generate_pattern must take exactly 1 argument (config), got {len(sig.parameters)}"
            )

        metadata = {
            "source": source_code,
            "source_path": str(path),
            "hash": source_hash,
            "docstring": inspect.getdoc(func),
            "is_builtin": False,
        }

        return func, metadata

    def execute_pattern_function(
        self,
        func: PatternFunction,
        config: Any,
    ) -> torch.Tensor:
        """
        Execute pattern function and validate output.

        Args:
            func: Pattern generation function
            config: Configuration object

        Returns:
            Generated sample positions tensor

        Raises:
            ValueError: If output shape is invalid
            TypeError: If output is not a tensor
        """
        # Execute function
        result = func(config)

        # Validate output type
        if not isinstance(result, torch.Tensor):
            raise TypeError(f"Pattern function must return torch.Tensor, got {type(result)}")

        # Validate output shape
        if result.ndim != 3:
            raise ValueError(
                f"Pattern output must be 3D tensor (n_samples, n_points, 2), "
                f"got shape {result.shape}"
            )

        if result.shape[1] not in [1, 2]:
            raise ValueError(
                f"Pattern output shape[1] must be 1 (points) or 2 (lines), got {result.shape[1]}"
            )

        if result.shape[2] != 2:
            raise ValueError(
                f"Pattern output shape[2] must be 2 (x, y coordinates), got {result.shape[2]}"
            )

        return result


def load_and_generate_pattern(
    pattern_spec: str,
    config: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Convenience function to load and execute pattern in one call.

    Args:
        pattern_spec: Pattern specification (builtin:name or file path)
        config: Configuration object

    Returns:
        (sample_centers, metadata) tuple
    """
    loader = PatternLoader()
    func, metadata = loader.load_pattern_function(pattern_spec)
    sample_centers = loader.execute_pattern_function(func, config)
    return sample_centers, metadata

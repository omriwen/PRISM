"""
Runner factory for creating appropriate runners based on algorithm type.

This module provides the RunnerFactory class that implements the Factory pattern
for creating algorithm-specific runners from a unified interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from prism.core.runner.base import AbstractRunner


class RunnerFactory:
    """Factory for creating appropriate runner based on algorithm type.

    This class implements the Factory pattern to create algorithm-specific
    runners (PRISMRunner, MoPIERunner) from a unified interface.

    The factory supports both built-in algorithms and custom runner registration.

    Examples
    --------
    >>> # Create a PRISM runner
    >>> runner = RunnerFactory.create("prism", args)
    >>> result = runner.run()

    >>> # Create a MoPIE runner
    >>> runner = RunnerFactory.create("mopie", args)
    >>> result = runner.run()

    >>> # Register a custom runner
    >>> RunnerFactory.register("custom", MyCustomRunner)
    >>> runner = RunnerFactory.create("custom", args)
    """

    # Registry of algorithm names to runner classes
    _runners: dict[str, type[AbstractRunner]] = {}

    @classmethod
    def _ensure_registered(cls) -> None:
        """Ensure built-in runners are registered.

        This method lazily registers the built-in runners to avoid
        circular imports at module load time.
        """
        if not cls._runners:
            from prism.core.runner.mopie_runner import MoPIERunner
            from prism.core.runner.prism_runner import PRISMRunner

            cls._runners = {
                "prism": PRISMRunner,
                "spids": PRISMRunner,  # Alias for backward compatibility
                "mopie": MoPIERunner,
                "mo-pie": MoPIERunner,  # Alias with hyphen
            }

    @classmethod
    def create(cls, algorithm: str, args: Any) -> AbstractRunner:
        """Create a runner for the specified algorithm.

        Parameters
        ----------
        algorithm : str
            Algorithm name: "prism", "spids", "mopie", or "mo-pie"
        args : Any
            Parsed command-line arguments (argparse.Namespace)

        Returns
        -------
        AbstractRunner
            Configured runner instance for the specified algorithm

        Raises
        ------
        ValueError
            If the algorithm name is not recognized

        Examples
        --------
        >>> runner = RunnerFactory.create("prism", args)
        >>> result = runner.run()
        """
        cls._ensure_registered()

        algorithm_lower = algorithm.lower()
        runner_class = cls._runners.get(algorithm_lower)

        if runner_class is None:
            available = ", ".join(sorted(set(cls._runners.keys())))
            raise ValueError(f"Unknown algorithm: '{algorithm}'. Available algorithms: {available}")

        return runner_class(args)

    @classmethod
    def register(cls, name: str, runner_class: type[AbstractRunner]) -> None:
        """Register a custom runner class.

        Parameters
        ----------
        name : str
            Algorithm name to register
        runner_class : type[AbstractRunner]
            Runner class to register

        Examples
        --------
        >>> from prism.core.runner.base import AbstractRunner
        >>> class MyRunner(AbstractRunner):
        ...     # Implementation
        ...     pass
        >>> RunnerFactory.register("my_algorithm", MyRunner)
        """
        cls._ensure_registered()
        cls._runners[name.lower()] = runner_class

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """List all available algorithm names.

        Returns
        -------
        list[str]
            Sorted list of available algorithm names (without aliases)

        Examples
        --------
        >>> algorithms = RunnerFactory.list_algorithms()
        >>> print(algorithms)
        ['mopie', 'prism']
        """
        cls._ensure_registered()
        # Return unique algorithms (exclude aliases)
        primary_names = {"prism", "mopie"}
        return sorted(primary_names & set(cls._runners.keys()))

    @classmethod
    def is_valid_algorithm(cls, algorithm: str) -> bool:
        """Check if an algorithm name is valid.

        Parameters
        ----------
        algorithm : str
            Algorithm name to check

        Returns
        -------
        bool
            True if the algorithm is registered, False otherwise

        Examples
        --------
        >>> RunnerFactory.is_valid_algorithm("prism")
        True
        >>> RunnerFactory.is_valid_algorithm("unknown")
        False
        """
        cls._ensure_registered()
        return algorithm.lower() in cls._runners

"""Enhanced configuration validation with intelligent error messages and suggestions.

This module provides comprehensive validation for SPIDS configuration parameters
with helpful error messages, spelling suggestions, and detailed guidance.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table


class ValidationError(Exception):
    """Enhanced validation error with suggestions and detailed guidance."""

    pass


class ConfigValidator:
    """Validates configuration with helpful error messages and suggestions."""

    # Valid options for enum-like parameters
    VALID_PROPAGATORS = ["auto", "fraunhofer", "fresnel", "angular_spectrum"]
    VALID_PATTERNS = ["fermat", "star", "random"]
    VALID_SAMPLE_SHAPES = ["circle", "line"]
    VALID_ROI_SHAPES = ["circle", "square"]
    VALID_SAMPLE_SORTS = ["center", "rand", "energy"]
    VALID_LOSS_TYPES = ["l1", "l2", "ssim", "ms-ssim"]
    VALID_INIT_TARGETS = ["measurement", "circle", "synthetic_aperture"]
    VALID_ACTIVATIONS = ["none", "sigmoid", "hardsigmoid", "scalesigmoid", "relu", "tanh"]
    VALID_OBJECTS = ["europa", "titan", "betelgeuse", "neptune"]

    # Descriptions for enum values
    PROPAGATOR_DESCRIPTIONS = {
        "auto": "Automatic selection based on distance (Fresnel number)",
        "fraunhofer": "Far-field approximation (Fresnel << 1)",
        "fresnel": "Near-field propagation (Fresnel ~ 1)",
        "angular_spectrum": "General-purpose propagator (all cases)",
    }

    PATTERN_DESCRIPTIONS = {
        "fermat": "Fermat spiral - uniform, incoherent sampling",
        "star": "Star pattern - multiple radial lines",
        "random": "Random sampling - uniformly distributed points",
    }

    LOSS_TYPE_DESCRIPTIONS = {
        "l1": "L1 loss (Mean Absolute Error) - robust to outliers",
        "l2": "L2 loss (Mean Squared Error) - penalizes large errors",
        "ssim": "Structural Similarity Index - perceptual quality",
        "ms-ssim": "Multi-Scale SSIM - best for image quality (5 scales)",
    }

    ACTIVATION_DESCRIPTIONS = {
        "none": "No activation (linear)",
        "sigmoid": "Sigmoid - outputs in [0, 1]",
        "hardsigmoid": "Hard sigmoid - faster approximation",
        "scalesigmoid": "Scaled sigmoid - custom range",
        "relu": "ReLU - outputs in [0, ∞)",
        "tanh": "Tanh - outputs in [-1, 1]",
    }

    @staticmethod
    def suggest_correction(
        invalid: str, valid_options: List[str], n: int = 1, cutoff: float = 0.6
    ) -> Optional[str]:
        """Suggest closest match using Levenshtein distance.

        Parameters
        ----------
        invalid : str
            Invalid value provided by user
        valid_options : List[str]
            List of valid options
        n : int, optional
            Number of suggestions to return, by default 1
        cutoff : float, optional
            Similarity threshold (0-1), by default 0.6

        Returns
        -------
        Optional[str]
            Closest match if found, None otherwise
        """
        matches = get_close_matches(invalid, valid_options, n=n, cutoff=cutoff)
        return matches[0] if matches else None

    @classmethod
    def format_enum_error(
        cls,
        param_name: str,
        invalid_value: str,
        valid_options: List[str],
        descriptions: Optional[Dict[str, str]] = None,
        help_flag: Optional[str] = None,
    ) -> str:
        """Format error message for invalid enum values with suggestions.

        Parameters
        ----------
        param_name : str
            Name of the parameter
        invalid_value : str
            Invalid value provided
        valid_options : List[str]
            List of valid options
        descriptions : Optional[Dict[str, str]], optional
            Descriptions for each option
        help_flag : Optional[str], optional
            CLI help flag for more info

        Returns
        -------
        str
            Formatted error message
        """
        # Build error message
        lines = [f"Invalid {param_name}: '{invalid_value}'\n"]

        # Add valid options with descriptions
        lines.append("Valid options:")
        for option in valid_options:
            desc = descriptions.get(option, "") if descriptions else ""
            if desc:
                lines.append(f"  - '{option}' → {desc}")
            else:
                lines.append(f"  - '{option}'")

        # Add spelling suggestion
        suggestion = cls.suggest_correction(invalid_value, valid_options)
        if suggestion:
            lines.append(f"\nDid you mean '{suggestion}'?")

        # Add help flag
        if help_flag:
            lines.append(f"\nFor more info: python main.py {help_flag}")

        return "\n".join(lines)

    @classmethod
    def format_range_error(
        cls,
        param_name: str,
        invalid_value: Any,
        valid_range: str,
        typical_values: Optional[str] = None,
        help_flag: Optional[str] = None,
    ) -> str:
        """Format error message for out-of-range values.

        Parameters
        ----------
        param_name : str
            Name of the parameter
        invalid_value : Any
            Invalid value provided
        valid_range : str
            Description of valid range (e.g., "positive", "> 0", "[0, 1]")
        typical_values : Optional[str], optional
            Examples of typical values
        help_flag : Optional[str], optional
            CLI help flag for more info

        Returns
        -------
        str
            Formatted error message
        """
        lines = [f"Invalid {param_name}: {invalid_value}"]
        lines.append(f"  → Must be {valid_range}")

        if typical_values:
            lines.append(f"  → Typical values: {typical_values}")

        if help_flag:
            lines.append(f"\nFor more info: python main.py {help_flag}")

        return "\n".join(lines)

    @classmethod
    def validate_propagator(cls, value: Optional[str]) -> None:
        """Validate propagator method parameter.

        Parameters
        ----------
        value : Optional[str]
            Propagator method value

        Raises
        ------
        ValidationError
            If propagator method is invalid
        """
        if value is None:
            return

        if value not in cls.VALID_PROPAGATORS:
            error_msg = cls.format_enum_error(
                param_name="propagator_method",
                invalid_value=value,
                valid_options=cls.VALID_PROPAGATORS,
                descriptions=cls.PROPAGATOR_DESCRIPTIONS,
                help_flag="--help-propagator",
            )
            raise ValidationError(error_msg)

    @classmethod
    def validate_pattern(cls, value: Optional[str]) -> None:
        """Validate pattern name (for builtin patterns).

        Parameters
        ----------
        value : Optional[str]
            Pattern name value

        Raises
        ------
        ValidationError
            If pattern name is invalid
        """
        if value is None:
            return

        if value not in cls.VALID_PATTERNS:
            error_msg = cls.format_enum_error(
                param_name="pattern",
                invalid_value=value,
                valid_options=cls.VALID_PATTERNS,
                descriptions=cls.PATTERN_DESCRIPTIONS,
                help_flag="--help-patterns",
            )
            raise ValidationError(error_msg)

    @classmethod
    def validate_loss_type(cls, value: str) -> None:
        """Validate loss type parameter.

        Parameters
        ----------
        value : str
            Loss type value

        Raises
        ------
        ValidationError
            If loss type is invalid
        """
        if value not in cls.VALID_LOSS_TYPES:
            error_msg = cls.format_enum_error(
                param_name="loss_type",
                invalid_value=value,
                valid_options=cls.VALID_LOSS_TYPES,
                descriptions=cls.LOSS_TYPE_DESCRIPTIONS,
                help_flag="--help-loss",
            )
            raise ValidationError(error_msg)

    @classmethod
    def validate_activation(cls, value: str, param_name: str = "activation") -> None:
        """Validate activation function parameter.

        Parameters
        ----------
        value : str
            Activation function value
        param_name : str, optional
            Parameter name for error message, by default "activation"

        Raises
        ------
        ValidationError
            If activation function is invalid
        """
        if value not in cls.VALID_ACTIVATIONS:
            error_msg = cls.format_enum_error(
                param_name=param_name,
                invalid_value=value,
                valid_options=cls.VALID_ACTIVATIONS,
                descriptions=cls.ACTIVATION_DESCRIPTIONS,
                help_flag="--help-model",
            )
            raise ValidationError(error_msg)

    @classmethod
    def validate_positive(
        cls, value: float, param_name: str, typical_values: Optional[str] = None
    ) -> None:
        """Validate that a value is positive.

        Parameters
        ----------
        value : float
            Value to validate
        param_name : str
            Parameter name for error message
        typical_values : Optional[str], optional
            Examples of typical values

        Raises
        ------
        ValidationError
            If value is not positive
        """
        if value <= 0:
            error_msg = cls.format_range_error(
                param_name=param_name,
                invalid_value=value,
                valid_range="positive (> 0)",
                typical_values=typical_values,
            )
            raise ValidationError(error_msg)

    @classmethod
    def validate_non_negative(
        cls, value: float, param_name: str, typical_values: Optional[str] = None
    ) -> None:
        """Validate that a value is non-negative.

        Parameters
        ----------
        value : float
            Value to validate
        param_name : str
            Parameter name for error message
        typical_values : Optional[str], optional
            Examples of typical values

        Raises
        ------
        ValidationError
            If value is negative
        """
        if value < 0:
            error_msg = cls.format_range_error(
                param_name=param_name,
                invalid_value=value,
                valid_range="non-negative (≥ 0)",
                typical_values=typical_values,
            )
            raise ValidationError(error_msg)

    @classmethod
    def validate_in_range(
        cls,
        value: float,
        param_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        typical_values: Optional[str] = None,
    ) -> None:
        """Validate that a value is within a specified range.

        Parameters
        ----------
        value : float
            Value to validate
        param_name : str
            Parameter name for error message
        min_val : Optional[float], optional
            Minimum allowed value
        max_val : Optional[float], optional
            Maximum allowed value
        typical_values : Optional[str], optional
            Examples of typical values

        Raises
        ------
        ValidationError
            If value is out of range
        """
        if min_val is not None and value < min_val:
            range_desc = f">= {min_val}"
            if max_val is not None:
                range_desc = f"[{min_val}, {max_val}]"
            error_msg = cls.format_range_error(
                param_name=param_name,
                invalid_value=value,
                valid_range=range_desc,
                typical_values=typical_values,
            )
            raise ValidationError(error_msg)

        if max_val is not None and value > max_val:
            range_desc = f"<= {max_val}"
            if min_val is not None:
                range_desc = f"[{min_val}, {max_val}]"
            error_msg = cls.format_range_error(
                param_name=param_name,
                invalid_value=value,
                valid_range=range_desc,
                typical_values=typical_values,
            )
            raise ValidationError(error_msg)

    @classmethod
    def print_help_topic(cls, topic: str) -> None:
        """Print detailed help for a specific topic.

        Parameters
        ----------
        topic : str
            Help topic (propagator, patterns, loss, model, objects)
        """
        console = Console()

        if topic == "propagator":
            cls._print_propagator_help(console)
        elif topic == "patterns":
            cls._print_patterns_help(console)
        elif topic == "loss":
            cls._print_loss_help(console)
        elif topic == "model":
            cls._print_model_help(console)
        elif topic == "objects":
            cls._print_objects_help(console)
        else:
            console.print(f"[red]Unknown help topic: {topic}[/red]")
            console.print("\nAvailable topics: propagator, patterns, loss, model, objects")

    @classmethod
    def _print_propagator_help(cls, console: Console) -> None:
        """Print propagator help."""
        table = Table(title="Propagator Methods", show_header=True, header_style="bold magenta")
        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Best For", style="green")

        table.add_row(
            "auto",
            cls.PROPAGATOR_DESCRIPTIONS["auto"],
            "Most cases (default)",
        )
        table.add_row(
            "fraunhofer",
            cls.PROPAGATOR_DESCRIPTIONS["fraunhofer"],
            "Far-field imaging",
        )
        table.add_row(
            "fresnel",
            cls.PROPAGATOR_DESCRIPTIONS["fresnel"],
            "Near-field imaging",
        )
        table.add_row(
            "angular_spectrum",
            cls.PROPAGATOR_DESCRIPTIONS["angular_spectrum"],
            "Any distance",
        )

        console.print(table)
        console.print("\nUsage: --propagator-method <method>")
        console.print("Example: --propagator-method fraunhofer")

    @classmethod
    def _print_patterns_help(cls, console: Console) -> None:
        """Print patterns help."""
        table = Table(title="Sampling Patterns", show_header=True, header_style="bold magenta")
        table.add_column("Pattern", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Properties", style="green")

        table.add_row(
            "fermat",
            cls.PATTERN_DESCRIPTIONS["fermat"],
            "Uniform, incoherent",
        )
        table.add_row(
            "star",
            cls.PATTERN_DESCRIPTIONS["star"],
            "Radial coverage",
        )
        table.add_row(
            "random",
            cls.PATTERN_DESCRIPTIONS["random"],
            "Baseline",
        )

        console.print(table)
        console.print("\nUsage: --pattern-fn builtin:<name>")
        console.print("Example: --pattern-fn builtin:fermat")
        console.print("\nCustom patterns: --pattern-fn /path/to/pattern.py")

    @classmethod
    def _print_loss_help(cls, console: Console) -> None:
        """Print loss function help."""
        table = Table(title="Loss Functions", show_header=True, header_style="bold magenta")
        table.add_column("Loss Type", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Best For", style="green")

        table.add_row("l1", cls.LOSS_TYPE_DESCRIPTIONS["l1"], "General use (default)")
        table.add_row("l2", cls.LOSS_TYPE_DESCRIPTIONS["l2"], "Smooth optimization")
        table.add_row("ssim", cls.LOSS_TYPE_DESCRIPTIONS["ssim"], "Perceptual quality")
        table.add_row(
            "ms-ssim",
            cls.LOSS_TYPE_DESCRIPTIONS["ms-ssim"],
            "Best image quality",
        )

        console.print(table)
        console.print("\nUsage: --loss_type <type>")
        console.print("Example: --loss_type ms-ssim")

    @classmethod
    def _print_model_help(cls, console: Console) -> None:
        """Print model configuration help."""
        table = Table(title="Activation Functions", show_header=True, header_style="bold magenta")
        table.add_column("Activation", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Output Range", style="green")

        for act in cls.VALID_ACTIVATIONS:
            desc = cls.ACTIVATION_DESCRIPTIONS[act]
            if "outputs" in desc:
                output_range = desc.split("outputs in ")[-1]
                desc = desc.split(" - outputs")[0]
            else:
                output_range = "N/A"
            table.add_row(act, desc, output_range)

        console.print(table)
        console.print("\nUsage:")
        console.print("  --output_activation <activation>")
        console.print("  --middle_activation <activation>")
        console.print("\nExample: --output_activation sigmoid")

    @classmethod
    def _print_objects_help(cls, console: Console) -> None:
        """Print predefined objects help."""
        table = Table(title="Predefined Objects", show_header=True, header_style="bold magenta")
        table.add_column("Object", style="cyan", no_wrap=True)
        table.add_column("Type", style="white")
        table.add_column("Notes", style="green")

        table.add_row("europa", "Moon of Jupiter", "Default object")
        table.add_row("titan", "Moon of Saturn", "Large moon")
        table.add_row("betelgeuse", "Red supergiant star", "Very large")
        table.add_row("neptune", "Ice giant planet", "Blue planet")

        console.print(table)
        console.print("\nUsage: --obj <name>")
        console.print("Example: --obj europa")
        console.print("\nTo see object parameters: --show-object <name>")

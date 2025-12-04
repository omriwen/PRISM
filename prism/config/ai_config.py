"""
Module: prism.config.ai_config
Purpose: AI-powered natural language configuration using local LLM
Dependencies: ollama, argparse, rich, shlex, yaml, json
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table


@dataclass
class ArgumentSchema:
    """Schema for a single CLI argument.

    Attributes
    ----------
    name : str
        Argument name (snake_case, from action.dest)
    option_strings : List[str]
        CLI flags (e.g., ['--n-samples', '-n'])
    arg_type : Optional[str]
        Type converter name (int, float, str, etc.)
    default : Any
        Default value
    choices : Optional[List[Any]]
        Valid choices if constrained
    help : str
        Help text describing the argument
    required : bool
        Whether the argument is required
    is_flag : bool
        Whether this is a boolean flag (store_true/store_false)
    """

    name: str
    option_strings: List[str] = field(default_factory=list)
    arg_type: Optional[str] = None
    default: Any = None
    choices: Optional[List[Any]] = None
    help: str = ""
    required: bool = False
    is_flag: bool = False


@dataclass
class ConfigDelta:
    """Represents a configuration change from LLM.

    Attributes
    ----------
    changes : Dict[str, Any]
        Parameter name -> new value mappings
    explanation : str
        LLM's explanation of why these changes were made
    warnings : List[str]
        Any warnings about the changes
    """

    changes: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)


SYSTEM_PROMPT = """You are an AI assistant that configures PRISM experiments.
PRISM is a deep learning-based astronomical imaging system that reconstructs
high-resolution images from sparse telescope aperture measurements.

Your task is to interpret natural language instructions and return ONLY a JSON
object containing parameter changes. Do not include any explanation outside the JSON.

CRITICAL RULES:
1. ONLY use parameter names from the provided schema
2. ONLY use values that match the expected types
3. For parameters with choices, ONLY use values from the choices list
4. Return EMPTY JSON {{}} if the instruction doesn't require changes
5. Return ONLY the JSON object, no markdown, no explanation outside JSON

{schema}

## Current Configuration
{current_config}

## Response Format
Return a JSON object with this structure:
{{
    "changes": {{
        "parameter_name": new_value,
        ...
    }},
    "explanation": "Brief explanation of changes",
    "warnings": ["any warnings about the changes"]
}}

Examples:

Instruction: "use 200 samples with fermat spiral"
Response: {{"changes": {{"n_samples": 200, "fermat_sample": true}}, "explanation": "Set 200 samples with Fermat spiral sampling pattern", "warnings": []}}

Instruction: "increase learning rate to 0.01"
Response: {{"changes": {{"lr": 0.01}}, "explanation": "Increased learning rate from default to 0.01", "warnings": ["Higher learning rate may cause instability"]}}

Instruction: "hello"
Response: {{"changes": {{}}, "explanation": "No configuration changes requested", "warnings": []}}
"""


class AIConfigurator:
    """AI-powered configuration assistant using local LLM.

    Uses ollama with a local model (e.g., llama3.2:3b) to interpret
    natural language instructions and generate configuration changes.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to extract schema from
    model : str, default="llama3.2:3b"
        Ollama model to use for inference

    Examples
    --------
    >>> parser = create_main_parser()
    >>> configurator = AIConfigurator(parser)
    >>> base_config = configurator.load_base("configs/production.yaml")
    >>> delta = configurator.get_delta("use 200 samples", base_config)
    >>> final_config = configurator.apply_delta(base_config, delta.changes)
    """

    DEFAULT_MODEL = "llama3.2:3b"

    def __init__(
        self,
        parser: argparse.ArgumentParser,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.parser = parser
        self.model = model
        self.console = Console()
        self.schema = self._extract_schema()

    def _extract_schema(self) -> List[ArgumentSchema]:
        """Extract argument schema from parser._actions.

        Returns
        -------
        List[ArgumentSchema]
            List of schema objects for all parser arguments
        """
        schemas: List[ArgumentSchema] = []

        for action in self.parser._actions:
            # Skip help action
            if isinstance(action, argparse._HelpAction):
                continue

            # Determine if this is a boolean flag
            is_flag = isinstance(
                action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
            )

            # Get type name
            type_name = None
            if action.type is not None:
                if hasattr(action.type, "__name__"):
                    type_name = action.type.__name__
                else:
                    type_name = str(action.type)
            elif is_flag:
                type_name = "bool"

            # Handle choices
            choices = None
            if action.choices is not None:
                choices = list(action.choices)

            schema = ArgumentSchema(
                name=action.dest,
                option_strings=list(action.option_strings) if action.option_strings else [],
                arg_type=type_name,
                default=action.default,
                choices=choices,
                help=action.help or "",
                required=getattr(action, "required", False),
                is_flag=is_flag,
            )
            schemas.append(schema)

        return schemas

    def _schema_to_prompt_text(self) -> str:
        """Convert schema to text for LLM prompt.

        Returns
        -------
        str
            Formatted schema text for inclusion in prompt
        """
        lines = ["## Available Parameters\n"]

        # Group by category based on help text patterns
        categories: Dict[str, List[ArgumentSchema]] = {
            "sampling": [],
            "training": [],
            "physics": [],
            "model": [],
            "other": [],
        }

        for schema in self.schema:
            # Skip internal/hidden args
            if schema.name.startswith("_") or not schema.option_strings:
                continue

            # Categorize based on help text keywords
            help_lower = schema.help.lower()
            if any(kw in help_lower for kw in ["sample", "telescope", "aperture", "pattern"]):
                categories["sampling"].append(schema)
            elif any(kw in help_lower for kw in ["epoch", "lr", "learning", "loss", "train"]):
                categories["training"].append(schema)
            elif any(kw in help_lower for kw in ["wavelength", "distance", "diameter", "object"]):
                categories["physics"].append(schema)
            elif any(kw in help_lower for kw in ["model", "network", "activation", "bn"]):
                categories["model"].append(schema)
            else:
                categories["other"].append(schema)

        for category, cat_schemas in categories.items():
            if not cat_schemas:
                continue

            lines.append(f"\n### {category.title()} Parameters\n")

            for schema in cat_schemas:
                line = f"- `{schema.name}`"
                if schema.arg_type:
                    line += f" ({schema.arg_type})"
                if schema.choices:
                    line += f" - choices: {schema.choices}"
                if schema.default is not None and schema.default != argparse.SUPPRESS:
                    line += f" - default: {schema.default}"
                if schema.help:
                    help_text = schema.help[:80] + "..." if len(schema.help) > 80 else schema.help
                    line += f" - {help_text}"
                lines.append(line)

        return "\n".join(lines)

    def load_base(self, path: str) -> argparse.Namespace:
        """Load base configuration from file.

        Supports three formats:
        - YAML (.yaml, .yml): Load as dict, convert to args
        - JSON (.json): Same approach as YAML
        - Shell (.sh): shlex.split() the file, parse_known_args()

        Parameters
        ----------
        path : str
            Path to configuration file

        Returns
        -------
        argparse.Namespace
            Parsed configuration

        Raises
        ------
        FileNotFoundError
            If configuration file doesn't exist
        ValueError
            If file format is not supported
        """
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        suffix = config_path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            return self._load_yaml(config_path)
        elif suffix == ".json":
            return self._load_json(config_path)
        elif suffix == ".sh":
            return self._load_shell(config_path)
        else:
            raise ValueError(
                f"Unsupported config format: {suffix}\n"
                f"Supported formats: .yaml, .yml, .json, .sh"
            )

    def _load_yaml(self, path: Path) -> argparse.Namespace:
        """Load YAML configuration."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        flat_dict = self._flatten_config_dict(data)
        args_list = self._dict_to_args_list(flat_dict)
        namespace, _ = self.parser.parse_known_args(args_list)
        return namespace

    def _load_json(self, path: Path) -> argparse.Namespace:
        """Load JSON configuration."""
        with open(path, "r") as f:
            data = json.load(f)

        flat_dict = self._flatten_config_dict(data)
        args_list = self._dict_to_args_list(flat_dict)
        namespace, _ = self.parser.parse_known_args(args_list)
        return namespace

    def _load_shell(self, path: Path) -> argparse.Namespace:
        """Load shell script configuration (the 'Self-Parsing Cheat Code').

        Parse a shell script that contains CLI invocations, extracting
        the arguments using shlex.split().

        Parameters
        ----------
        path : Path
            Path to shell script

        Returns
        -------
        argparse.Namespace
            Parsed configuration from shell script
        """
        with open(path, "r") as f:
            content = f.read()

        args_list: List[str] = []
        for line in content.split("\n"):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Look for lines that invoke main.py
            if "main.py" in line or "python" in line:
                try:
                    tokens = shlex.split(line)
                    in_args = False
                    for token in tokens:
                        if "main.py" in token:
                            in_args = True
                            continue
                        if in_args:
                            args_list.append(token)
                except ValueError:
                    # shlex parse error, skip this line
                    continue

        namespace, _ = self.parser.parse_known_args(args_list)
        return namespace

    def _flatten_config_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested config dict to argument names.

        Handles the PRISM config structure:
        {"telescope": {"n_samples": 100}, "training": {"lr": 0.001}}

        Parameters
        ----------
        data : Dict[str, Any]
            Potentially nested configuration dictionary

        Returns
        -------
        Dict[str, Any]
            Flattened dictionary with argument names as keys
        """
        flat: Dict[str, Any] = {}

        # Direct top-level args
        for key, value in data.items():
            if not isinstance(value, dict):
                flat[key] = value

        # Nested sections (matching PRISM config structure)
        nested_sections = [
            "telescope",
            "training",
            "physics",
            "image",
            "model",
            "point_source",
            "mopie",
        ]

        for section in nested_sections:
            if section in data and isinstance(data[section], dict):
                for key, value in data[section].items():
                    flat[key] = value

        return flat

    def _dict_to_args_list(self, flat_dict: Dict[str, Any]) -> List[str]:
        """Convert flat dict to CLI args list.

        Parameters
        ----------
        flat_dict : Dict[str, Any]
            Flat dictionary of parameter name -> value

        Returns
        -------
        List[str]
            CLI arguments list suitable for parse_known_args()
        """
        args_list: List[str] = []

        for name, value in flat_dict.items():
            if value is None:
                continue

            # Find schema for this argument
            schema = next((s for s in self.schema if s.name == name), None)
            if schema is None or not schema.option_strings:
                continue

            opt = schema.option_strings[0]

            if schema.is_flag:
                if value:
                    args_list.append(opt)
            else:
                args_list.append(opt)
                args_list.append(str(value))

        return args_list

    def get_delta(
        self,
        instruction: str,
        current_config: argparse.Namespace,
    ) -> ConfigDelta:
        """Get configuration delta from natural language instruction.

        Parameters
        ----------
        instruction : str
            Natural language instruction from user
        current_config : argparse.Namespace
            Current configuration values

        Returns
        -------
        ConfigDelta
            Validated configuration changes

        Raises
        ------
        ConnectionError
            If Ollama is not running
        ValueError
            If LLM response cannot be parsed
        """
        schema_text = self._schema_to_prompt_text()
        config_text = self._config_to_text(current_config)

        system_prompt = SYSTEM_PROMPT.format(
            schema=schema_text,
            current_config=config_text,
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                ],
                options={
                    "temperature": 0.1,  # Low temperature for consistent output
                    "num_predict": 500,
                },
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to ollama: {e}\n"
                "Make sure ollama is running: `ollama serve`"
            ) from e

        content = response["message"]["content"]

        try:
            delta_dict = self._extract_json(content)
            validated_changes = self._validate_changes(delta_dict.get("changes", {}))

            return ConfigDelta(
                changes=validated_changes,
                explanation=delta_dict.get("explanation", ""),
                warnings=delta_dict.get("warnings", []),
            )
        except Exception as e:
            raise ValueError(
                f"Failed to parse LLM response: {e}\n" f"Raw response: {content}"
            ) from e

    def _config_to_text(self, config: argparse.Namespace) -> str:
        """Convert current config to text for prompt.

        Parameters
        ----------
        config : argparse.Namespace
            Current configuration

        Returns
        -------
        str
            Formatted configuration text
        """
        lines: List[str] = []
        config_dict = vars(config)

        # Include most important/commonly modified parameters
        important_keys = [
            "n_samples",
            "sample_diameter",
            "sample_length",
            "fermat_sample",
            "star_sample",
            "snr",
            "max_epochs",
            "n_epochs",
            "lr",
            "loss_type",
            "loss_th",
            "obj_name",
            "image_size",
            "wavelength",
        ]

        for key in important_keys:
            if key in config_dict:
                value = config_dict[key]
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response (handles markdown code blocks).

        Parameters
        ----------
        content : str
            Raw LLM response content

        Returns
        -------
        Dict[str, Any]
            Parsed JSON object

        Raises
        ------
        ValueError
            If JSON cannot be extracted
        """
        # Try direct JSON parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks
        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"\{.*\}",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if "```" in pattern else match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        raise ValueError("Could not extract JSON from response")

    def _validate_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and type-convert changes against schema.

        Parameters
        ----------
        changes : Dict[str, Any]
            Raw changes from LLM

        Returns
        -------
        Dict[str, Any]
            Validated and type-converted changes

        Raises
        ------
        ValueError
            If a change is invalid
        """
        validated: Dict[str, Any] = {}

        for name, value in changes.items():
            schema = next((s for s in self.schema if s.name == name), None)

            if schema is None:
                logger.warning(f"Unknown parameter '{name}' - skipping")
                continue

            # Validate choices
            if schema.choices and value not in schema.choices:
                raise ValueError(
                    f"Invalid value '{value}' for '{name}'. "
                    f"Valid choices: {schema.choices}"
                )

            # Type conversion
            if schema.arg_type == "int":
                value = int(value)
            elif schema.arg_type == "float":
                value = float(value)
            elif schema.arg_type == "bool" or schema.is_flag:
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                else:
                    value = bool(value)

            validated[name] = value

        return validated

    def apply_delta(
        self,
        config: argparse.Namespace,
        delta: Dict[str, Any],
    ) -> argparse.Namespace:
        """Apply delta to configuration namespace.

        Parameters
        ----------
        config : argparse.Namespace
            Base configuration
        delta : Dict[str, Any]
            Changes to apply

        Returns
        -------
        argparse.Namespace
            New namespace with changes applied
        """
        # Create a copy to avoid modifying the original
        new_config = argparse.Namespace(**vars(config))

        for name, value in delta.items():
            setattr(new_config, name, value)

        return new_config

    def show_delta(self, delta: ConfigDelta, current: argparse.Namespace) -> None:
        """Display delta with rich formatting.

        Parameters
        ----------
        delta : ConfigDelta
            Proposed changes
        current : argparse.Namespace
            Current configuration for comparison
        """
        if not delta.changes:
            self.console.print("[yellow]No changes requested.[/yellow]")
            return

        table = Table(title="Proposed Configuration Changes", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Current", style="dim")
        table.add_column("New", style="green")

        current_dict = vars(current)

        for name, new_value in delta.changes.items():
            current_value = current_dict.get(name, "N/A")
            table.add_row(name, str(current_value), str(new_value))

        self.console.print(table)

        if delta.explanation:
            self.console.print(f"\n[bold]Explanation:[/bold] {delta.explanation}")

        if delta.warnings:
            self.console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in delta.warnings:
                self.console.print(f"  [yellow]- {warning}[/yellow]")

    def confirm_delta(self, delta: ConfigDelta) -> bool:
        """Prompt user to confirm delta application.

        Parameters
        ----------
        delta : ConfigDelta
            Proposed changes

        Returns
        -------
        bool
            True if user confirms, False otherwise
        """
        return Confirm.ask("\n[bold]Apply these changes?[/bold]", default=True)

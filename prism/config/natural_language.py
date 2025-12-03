"""Natural language configuration parsing via Ollama."""
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import ollama
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

DEFAULT_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = '''You are a configuration parser for PRISM, a scientific imaging CLI.
Extract parameter values from the user's instruction and return ONLY valid JSON.

Available parameters:
- lr (float): Learning rate. Example: 0.001, 0.01
- n_samples (int): Number of telescope positions. Example: 100, 200
- n_epochs (int): Epochs per sample. Example: 1000, 2000
- max_epochs (int): Training repetitions. Example: 1, 25, 50
- loss_type (str): One of "l1", "l2", "ssim", "ms-ssim"
- propagator_method (str): One of "auto", "fraunhofer", "fresnel", "angular_spectrum"
- obj_name (str): One of "europa", "titan", "neptune", "betelgeuse"
- log_dir (str): Output directory path
- name (str): Experiment name
- preset (str): One of "quick_test", "production", "high_quality", "debug"
- use_cuda (bool): Use GPU (true/false)
- fermat_sample (bool): Use Fermat spiral sampling
- save_data (bool): Save experiment results
- sample_diameter (float): Telescope aperture diameter in pixels
- image_size (int): Image resolution in pixels

Rules:
1. Only extract parameters explicitly mentioned
2. Use exact parameter names and valid values from lists above
3. Return empty {} if nothing can be extracted
4. For paths, use the exact path mentioned

Example: "train europa with lr 0.01 using fresnel"
Output: {"obj_name": "europa", "lr": 0.01, "propagator_method": "fresnel"}'''


@dataclass
class ParsedConfig:
    """Result of parsing a natural language instruction."""
    parameters: Dict[str, Any]
    raw_response: str
    model_used: str


def ensure_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """Ensure Ollama is installed and model is available."""
    console = Console()

    # Check if ollama CLI exists
    if not shutil.which("ollama"):
        if sys.platform == "linux":
            console.print("[yellow]Ollama not found. Installing...[/yellow]")
            result = subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True,
                capture_output=True,
            )
            if result.returncode != 0:
                console.print("[red]Failed to install Ollama.[/red]")
                return False
        else:
            console.print(
                "[red]Ollama not found.[/red]\n"
                "Install from: https://ollama.com/download"
            )
            return False

    # Check if model is available, pull if not
    try:
        models = ollama.list()
        model_names = [m.get("name", "") for m in models.get("models", [])]
        if not any(model in name for name in model_names):
            console.print(f"[yellow]Pulling {model} model (~1.9GB)...[/yellow]")
            ollama.pull(model)
    except Exception as e:
        console.print(f"[red]Ollama error: {e}[/red]")
        console.print("Is the Ollama service running? Try: ollama serve")
        return False

    return True


def parse_instruction(
    instruction: str, model: str = DEFAULT_MODEL
) -> ParsedConfig:
    """Parse natural language instruction into config parameters."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        format="json",
    )

    raw = response["message"]["content"]
    parameters = json.loads(raw)

    return ParsedConfig(
        parameters=parameters,
        raw_response=raw,
        model_used=model,
    )


def show_confirmation(parsed: ParsedConfig, instruction: str) -> bool:
    """Display parsed config and get user confirmation."""
    console = Console()

    console.print(f"\n[dim]Instruction: {instruction}[/dim]\n")

    if not parsed.parameters:
        console.print("[yellow]No parameters could be extracted.[/yellow]")
        return False

    table = Table(title="Parsed Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")

    for name, value in parsed.parameters.items():
        table.add_row(name, str(value))

    console.print(table)
    console.print(f"\n[dim]Model: {parsed.model_used}[/dim]")

    return Confirm.ask("\nProceed with these settings?", default=True)


def apply_parsed_config(args, parsed: ParsedConfig):
    """Apply parsed parameters to argparse namespace."""
    for name, value in parsed.parameters.items():
        if hasattr(args, name):
            setattr(args, name, value)
        # Handle special mappings (negated flags)
        elif name == "use_cuda" and hasattr(args, "no_cuda"):
            args.no_cuda = not value
    return args


def process_instruction(args, interactive: bool = True) -> Optional:
    """Main entry point: parse instruction and apply to args.

    Returns None if user cancels or Ollama unavailable.
    """
    if not ensure_ollama_available():
        return None

    parsed = parse_instruction(args.instruction)

    if interactive:
        if not show_confirmation(parsed, args.instruction):
            return None

    return apply_parsed_config(args, parsed)

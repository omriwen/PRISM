"""Dashboard server for managing experiment data loading and parsing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import torch
from loguru import logger


class ExperimentData:
    """Container for experiment data."""

    def __init__(
        self,
        exp_id: str,
        path: Path,
        config: Dict[str, Any],
        metrics: Dict[str, List[float]],
        final_metrics: Dict[str, float],
        timestamp: Optional[datetime] = None,
        reconstruction: Optional[np.ndarray] = None,
    ):
        self.exp_id = exp_id
        self.path = path
        self.config = config
        self.metrics = metrics
        self.final_metrics = final_metrics
        self.timestamp = timestamp
        self.reconstruction = reconstruction

    def __repr__(self) -> str:
        return f"ExperimentData(exp_id={self.exp_id}, timestamp={self.timestamp})"


class DashboardServer:
    """Manages dashboard server and data loading."""

    def __init__(self, runs_dir: Path = Path("runs")):
        """Initialize dashboard server.

        Args:
            runs_dir: Path to directory containing experiment runs
        """
        self.runs_dir = Path(runs_dir)
        self.experiments_cache: Dict[str, ExperimentData] = {}

        if not self.runs_dir.exists():
            logger.warning(f"Runs directory not found: {self.runs_dir}")
            self.runs_dir.mkdir(parents=True, exist_ok=True)

    def scan_experiments(self) -> List[Dict[str, Any]]:
        """Scan runs directory for experiments.

        Returns:
            List of experiment metadata dictionaries
        """
        experiments = []

        for exp_dir in self.runs_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            checkpoint_path = exp_dir / "checkpoint.pt"
            if not checkpoint_path.exists():
                continue

            try:
                stat = checkpoint_path.stat()
                experiments.append(
                    {
                        "id": exp_dir.name,
                        "path": exp_dir,
                        "last_modified": stat.st_mtime,
                        "last_modified_str": datetime.fromtimestamp(stat.st_mtime).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "size_mb": stat.st_size / (1024 * 1024),
                    }
                )
            except Exception as e:  # noqa: BLE001 - Continue scanning other experiments on failure
                logger.warning(f"Error scanning experiment {exp_dir.name}: {e}")
                continue

        # Sort by most recent first
        experiments.sort(key=lambda x: float(cast(float, x["last_modified"])), reverse=True)
        return experiments

    def load_experiment_data(self, exp_id: str, use_cache: bool = True) -> Optional[ExperimentData]:
        """Load experiment metrics and checkpoints.

        Args:
            exp_id: Experiment ID (directory name)
            use_cache: Whether to use cached data if available

        Returns:
            ExperimentData object or None if loading fails
        """
        # Check cache first
        if use_cache and exp_id in self.experiments_cache:
            return self.experiments_cache[exp_id]

        exp_path = self.runs_dir / exp_id
        if not exp_path.exists():
            logger.error(f"Experiment path not found: {exp_path}")
            return None

        try:
            # Load checkpoint
            checkpoint_path = exp_path / "checkpoint.pt"
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return None

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Load configuration
            config = self._load_config(exp_path)

            # Extract metrics history
            loss_list = self._tensor_to_list(checkpoint.get("losses", []))
            ssim_list = self._tensor_to_list(checkpoint.get("ssims", []))
            psnr_list = self._tensor_to_list(checkpoint.get("psnrs", []))
            rmse_list = self._tensor_to_list(checkpoint.get("rmses", []))
            epoch_list = [float(i) for i in range(len(loss_list))]

            metrics: Dict[str, List[float]] = {
                "epoch": epoch_list,
                "loss": loss_list,
                "ssim": ssim_list,
                "psnr": psnr_list,
                "rmse": rmse_list,
            }

            # Extract final metrics
            final_metrics: Dict[str, float] = {
                "loss": float(loss_list[-1]) if loss_list else 0.0,
                "ssim": float(ssim_list[-1]) if ssim_list else 0.0,
                "psnr": float(psnr_list[-1]) if psnr_list else 0.0,
                "rmse": float(rmse_list[-1]) if rmse_list else 0.0,
                "epochs": float(len(epoch_list)),
            }

            # Extract reconstruction if available
            reconstruction = None
            if "current_rec" in checkpoint:
                rec = checkpoint["current_rec"]
                if isinstance(rec, torch.Tensor):
                    reconstruction = rec.cpu().numpy()
                    # Squeeze batch and channel dims: [1, 1, H, W] -> [H, W]
                    while reconstruction.ndim > 2:
                        reconstruction = reconstruction[0]

            # Get timestamp
            timestamp = None
            stat = checkpoint_path.stat()
            timestamp = datetime.fromtimestamp(stat.st_mtime)

            # Create experiment data object
            exp_data = ExperimentData(
                exp_id=exp_id,
                path=exp_path,
                config=config,
                metrics=metrics,
                final_metrics=final_metrics,
                timestamp=timestamp,
                reconstruction=reconstruction,
            )

            # Cache the data
            self.experiments_cache[exp_id] = exp_data

            return exp_data

        except Exception as e:  # noqa: BLE001 - Return None on any loading failure
            logger.error(f"Error loading experiment {exp_id}: {e}")
            return None

    def _load_config(self, exp_path: Path) -> Dict[str, Any]:
        """Load configuration from experiment directory.

        Args:
            exp_path: Path to experiment directory

        Returns:
            Configuration dictionary
        """
        config = {}

        # Try to load args.pt first
        args_pt_path = exp_path / "args.pt"
        if args_pt_path.exists():
            try:
                args = torch.load(args_pt_path, map_location="cpu")
                # Convert Namespace or dict to dict
                if hasattr(args, "__dict__"):
                    config = vars(args)
                else:
                    config = dict(args)
            except Exception as e:  # noqa: BLE001 - Fallback to other config sources
                logger.warning(f"Error loading args.pt: {e}")

        # Fallback to args.txt
        if not config:
            args_txt_path = exp_path / "args.txt"
            if args_txt_path.exists():
                try:
                    config = self._parse_args_txt(args_txt_path)
                except Exception as e:  # noqa: BLE001 - Config loading failure is non-fatal
                    logger.warning(f"Error loading args.txt: {e}")

        return config

    def _parse_args_txt(self, args_txt_path: Path) -> Dict[str, Any]:
        """Parse args.txt file into dictionary.

        Args:
            args_txt_path: Path to args.txt file

        Returns:
            Configuration dictionary
        """
        config = {}

        with open(args_txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue

                key, value_str = line.split(":", 1)
                key = key.strip()
                value_str = value_str.strip()

                # Try to convert to appropriate type
                parsed_value: Union[bool, None, int, float, str]
                if value_str.lower() == "true":
                    parsed_value = True
                elif value_str.lower() == "false":
                    parsed_value = False
                elif value_str.lower() == "none":
                    parsed_value = None
                else:
                    try:
                        # Try int first
                        parsed_value = int(value_str)
                    except ValueError:
                        try:
                            # Try float
                            parsed_value = float(value_str)
                        except ValueError:
                            # Keep as string
                            parsed_value = value_str

                config[key] = parsed_value

        return config

    def _tensor_to_list(self, data: Any) -> List[float]:
        """Convert tensor or array to list of floats.

        Args:
            data: Tensor, array, or list

        Returns:
            List of floats
        """
        if isinstance(data, torch.Tensor):
            result: List[float] = data.cpu().numpy().tolist()
            return result
        elif isinstance(data, np.ndarray):
            result = data.tolist()
            return result
        elif isinstance(data, list):
            return list(data)
        else:
            return []

    def parse_tensorboard_logs(self, exp_path: Path) -> Dict[str, List[float]]:
        """Parse TensorBoard event files (if available).

        Args:
            exp_path: Path to experiment directory

        Returns:
            Dictionary of metric histories from TensorBoard
        """
        # This is a placeholder for TensorBoard parsing
        # Implementation would use tensorboard.backend.event_processing
        # For now, we rely on checkpoint metrics
        return {}

    def clear_cache(self) -> None:
        """Clear the experiment data cache."""
        self.experiments_cache.clear()

    def refresh_experiment(self, exp_id: str) -> Optional[ExperimentData]:
        """Force refresh experiment data from disk.

        Args:
            exp_id: Experiment ID to refresh

        Returns:
            Updated ExperimentData or None if loading fails
        """
        if exp_id in self.experiments_cache:
            del self.experiments_cache[exp_id]
        return self.load_experiment_data(exp_id, use_cache=False)

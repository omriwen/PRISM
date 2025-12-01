"""
Training animation generator for SPIDS experiments.

This module provides functionality to generate MP4 or GIF animations showing
training progression over time, including side-by-side comparisons and metric overlays.

Examples
--------
>>> # Generate animation from checkpoint directory
>>> animator = TrainingAnimator("runs/experiment")
>>> animator.generate_video("training.mp4", fps=10)

>>> # Create side-by-side comparison
>>> animator = TrainingAnimator.from_multiple(["runs/exp1", "runs/exp2"])
>>> animator.generate_video("comparison.mp4", layout="side_by_side")
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn


# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Use non-interactive backend
matplotlib.use("Agg")


class TrainingAnimator:
    """Generate training progression animations.

    This class loads checkpoint data and generates MP4 or GIF animations
    showing how reconstructions improve over training iterations.

    Parameters
    ----------
    experiment_path : Path
        Path to experiment directory containing checkpoint
    checkpoint_file : str, optional
        Name of checkpoint file (default: "checkpoint.pt")

    Attributes
    ----------
    exp_path : Path
        Experiment directory path
    checkpoint : dict
        Loaded checkpoint data
    ground_truth : np.ndarray or None
        Ground truth image if available
    """

    def __init__(
        self,
        experiment_path: Union[str, Path],
        checkpoint_file: str = "checkpoint.pt",
    ):
        """Initialize training animator.

        Parameters
        ----------
        experiment_path : Union[str, Path]
            Path to experiment directory
        checkpoint_file : str, optional
            Name of checkpoint file (default: "checkpoint.pt")

        Raises
        ------
        FileNotFoundError
            If experiment path or checkpoint file doesn't exist
        ValueError
            If checkpoint doesn't contain required data
        """
        self.exp_path = Path(experiment_path)
        if not self.exp_path.exists():
            raise FileNotFoundError(f"Experiment path not found: {self.exp_path}")

        checkpoint_path = self.exp_path / checkpoint_file
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Validate checkpoint contains necessary data (support both SPIDS and ePIE formats)
        if "current_rec" not in self.checkpoint and "model" not in self.checkpoint:
            raise ValueError(
                "Checkpoint missing reconstruction data ('current_rec' or 'model' field)"
            )

        # Determine checkpoint type and extract reconstruction
        self.is_epie = "model" in self.checkpoint and "current_rec" not in self.checkpoint

        # Extract ground truth if available
        self.ground_truth = self._extract_ground_truth()

        # Get experiment name
        self.exp_name = self.exp_path.name

    def _extract_ground_truth(self) -> Optional[np.ndarray]:
        """Extract ground truth image from checkpoint.

        Returns
        -------
        np.ndarray or None
            Ground truth image if available
        """
        if "telescope_agg" in self.checkpoint:
            telescope_agg = self.checkpoint["telescope_agg"]
            if hasattr(telescope_agg, "init_im"):
                gt = telescope_agg.init_im
                if isinstance(gt, torch.Tensor):
                    gt = gt.detach().cpu().numpy()
                gt = np.squeeze(gt)
                if gt.ndim == 2:
                    return np.abs(gt)  # type: ignore[no-any-return]
        return None

    def _get_reconstruction_history(self, n_frames: Optional[int] = None) -> List[np.ndarray]:
        """Get reconstruction at different training stages.

        Since checkpoints typically only store the final reconstruction,
        this method interpolates between initial state and final reconstruction
        to create smooth animation frames.

        Parameters
        ----------
        n_frames : int, optional
            Number of frames to generate. If None, uses actual history length.

        Returns
        -------
        List[np.ndarray]
            List of reconstruction frames
        """
        # Get final reconstruction (handle both SPIDS and ePIE formats)
        if self.is_epie:
            # ePIE stores model as state dict with 'obj' field containing reconstruction
            model_dict = self.checkpoint["model"]
            if isinstance(model_dict, dict) and "obj" in model_dict:
                final_rec = model_dict["obj"]
            else:
                # Fallback: try to use model directly
                final_rec = self.checkpoint["model"]
        else:
            final_rec = self.checkpoint["current_rec"]

        if isinstance(final_rec, torch.Tensor):
            final_rec = final_rec.detach().cpu().numpy()
        final_rec = np.squeeze(np.abs(final_rec))

        # Try to get metric history to determine number of frames
        # Support both SPIDS ("losses") and ePIE ("fourier_errors") formats
        if "losses" in self.checkpoint:
            history_length = len(self.checkpoint["losses"])
        elif "fourier_errors" in self.checkpoint:
            history_length = len(self.checkpoint["fourier_errors"])
        else:
            history_length = 100  # Default

        # If n_frames not specified, use history length (but cap at reasonable value)
        if n_frames is None:
            n_frames = min(history_length, 100)

        # Create initial state (random noise or zeros)
        initial_rec = np.random.randn(*final_rec.shape) * 0.1

        # Interpolate between initial and final state
        frames = []
        for i in range(n_frames):
            alpha = (i + 1) / n_frames
            # Use smooth easing function for more realistic progression
            alpha = self._ease_in_out(alpha)
            frame = (1 - alpha) * initial_rec + alpha * final_rec
            frames.append(frame)

        return frames

    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Smooth easing function for animation interpolation.

        Parameters
        ----------
        t : float
            Input value between 0 and 1

        Returns
        -------
        float
            Eased value between 0 and 1
        """
        return t * t * (3.0 - 2.0 * t)

    def _get_metrics_at_frame(self, frame_idx: int, total_frames: int) -> Dict[str, float]:
        """Get metrics for a specific frame.

        Parameters
        ----------
        frame_idx : int
            Frame index
        total_frames : int
            Total number of frames

        Returns
        -------
        Dict[str, float]
            Metrics at this frame
        """
        metrics = {}

        # Get metric histories (support both SPIDS and ePIE formats)
        # ePIE uses "fourier_errors" instead of "losses"
        losses = self.checkpoint.get("losses", self.checkpoint.get("fourier_errors", []))
        ssims = self.checkpoint.get("ssims", [])
        psnrs = self.checkpoint.get("psnrs", [])
        rmses = self.checkpoint.get("rmses", [])

        # Calculate corresponding history index
        if len(losses) > 0:
            history_idx = int((frame_idx / total_frames) * len(losses))
            history_idx = min(history_idx, len(losses) - 1)

            if len(losses) > history_idx:
                metrics["loss"] = float(losses[history_idx])
            if len(ssims) > history_idx:
                metrics["ssim"] = float(ssims[history_idx])
            if len(psnrs) > history_idx:
                metrics["psnr"] = float(psnrs[history_idx])
            if len(rmses) > history_idx:
                metrics["rmse"] = float(rmses[history_idx])

            metrics["sample"] = history_idx + 1
            metrics["total_samples"] = len(losses)

        return metrics

    def create_frame(
        self,
        frame_idx: int,
        reconstruction: np.ndarray,
        metrics: Dict[str, float],
        show_metrics: bool = True,
        show_difference: bool = True,
    ) -> np.ndarray:
        """Create a single animation frame.

        Parameters
        ----------
        frame_idx : int
            Frame number
        reconstruction : np.ndarray
            Reconstruction at this frame
        metrics : Dict[str, float]
            Metrics to display
        show_metrics : bool, optional
            Whether to show metric overlays (default: True)
        show_difference : bool, optional
            Whether to show difference map (default: True)

        Returns
        -------
        np.ndarray
            Frame as RGB array
        """
        # Determine subplot layout
        if self.ground_truth is not None and show_difference:
            n_cols = 3
            figsize = (15, 5)
        elif self.ground_truth is not None:
            n_cols = 2
            figsize = (10, 5)
        else:
            n_cols = 1
            figsize = (6, 6)

        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]

        col_idx = 0

        # Ground truth
        if self.ground_truth is not None:
            axes[col_idx].imshow(self.ground_truth, cmap="gray", vmin=0, vmax=1)
            axes[col_idx].set_title("Ground Truth", fontsize=12, fontweight="bold")
            axes[col_idx].axis("off")
            col_idx += 1

        # Current reconstruction
        axes[col_idx].imshow(reconstruction, cmap="gray", vmin=0, vmax=1)
        title = "Reconstruction"
        if "sample" in metrics and "total_samples" in metrics:
            title += f" (Sample {metrics['sample']}/{metrics['total_samples']})"
        axes[col_idx].set_title(title, fontsize=12, fontweight="bold")
        axes[col_idx].axis("off")
        col_idx += 1

        # Difference map
        if self.ground_truth is not None and show_difference:
            diff = np.abs(self.ground_truth - reconstruction)
            im = axes[col_idx].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
            axes[col_idx].set_title("Absolute Difference", fontsize=12, fontweight="bold")
            axes[col_idx].axis("off")
            plt.colorbar(im, ax=axes[col_idx], fraction=0.046, pad=0.04)

        # Add metrics overlay
        if show_metrics and metrics:
            metrics_text = []
            if "loss" in metrics:
                metrics_text.append(f"Loss: {metrics['loss']:.6f}")
            if "ssim" in metrics:
                metrics_text.append(f"SSIM: {metrics['ssim']:.4f}")
            if "psnr" in metrics:
                metrics_text.append(f"PSNR: {metrics['psnr']:.2f} dB")

            if metrics_text:
                text_str = "\n".join(metrics_text)
                fig.text(
                    0.98,
                    0.02,
                    text_str,
                    ha="right",
                    va="bottom",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        plt.tight_layout()

        # Convert figure to numpy array
        fig.canvas.draw()
        # Use buffer_rgba() for newer matplotlib compatibility
        buf = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
        frame = np.frombuffer(buf, dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        frame = frame[:, :, :3]
        plt.close(fig)

        return frame

    def generate_video(
        self,
        output_path: Union[str, Path],
        fps: int = 10,
        n_frames: Optional[int] = None,
        show_metrics: bool = True,
        show_difference: bool = True,
    ) -> None:
        """Generate MP4 video animation.

        Parameters
        ----------
        output_path : Union[str, Path]
            Output video file path
        fps : int, optional
            Frames per second (default: 10)
        n_frames : int, optional
            Number of frames to generate. If None, auto-determined.
        show_metrics : bool, optional
            Whether to show metric overlays (default: True)
        show_difference : bool, optional
            Whether to show difference map (default: True)

        Raises
        ------
        ImportError
            If opencv-python is not installed
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for video generation. "
                "Install it with: uv add opencv-python"
            )

        output_path = Path(output_path)

        # Get reconstruction history
        reconstructions = self._get_reconstruction_history(n_frames)
        total_frames = len(reconstructions)
        console = Console()

        # Generate frames with progress bar
        frames = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating frames...", total=total_frames)
            for i, rec in enumerate(reconstructions):
                metrics = self._get_metrics_at_frame(i, total_frames)
                frame = self.create_frame(
                    i, rec, metrics, show_metrics=show_metrics, show_difference=show_difference
                )
                frames.append(frame)
                progress.update(task, advance=1)

        # Create video
        console.print(f"Writing video to {output_path}...")
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video.release()
        console.print(f"[green]✓[/green] Video saved: {output_path}")

    def generate_gif(
        self,
        output_path: Union[str, Path],
        duration: int = 100,
        n_frames: Optional[int] = None,
        show_metrics: bool = True,
        show_difference: bool = True,
        loop: int = 0,
    ) -> None:
        """Generate animated GIF.

        Parameters
        ----------
        output_path : Union[str, Path]
            Output GIF file path
        duration : int, optional
            Duration per frame in milliseconds (default: 100)
        n_frames : int, optional
            Number of frames to generate. If None, auto-determined.
        show_metrics : bool, optional
            Whether to show metric overlays (default: True)
        show_difference : bool, optional
            Whether to show difference map (default: True)
        loop : int, optional
            Number of loops (0 = infinite, default: 0)
        """
        output_path = Path(output_path)
        console = Console()

        # Get reconstruction history
        reconstructions = self._get_reconstruction_history(n_frames)
        total_frames = len(reconstructions)

        # Generate frames with progress bar
        pil_frames = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating frames...", total=total_frames)
            for i, rec in enumerate(reconstructions):
                metrics = self._get_metrics_at_frame(i, total_frames)
                frame = self.create_frame(
                    i, rec, metrics, show_metrics=show_metrics, show_difference=show_difference
                )
                pil_frames.append(Image.fromarray(frame))
                progress.update(task, advance=1)

        # Save GIF
        console.print(f"Writing GIF to {output_path}...")
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=loop,
            optimize=True,
        )
        console.print(f"[green]✓[/green] GIF saved: {output_path}")

    @classmethod
    def from_multiple(
        cls, experiment_paths: Sequence[Union[str, Path]], checkpoint_file: str = "checkpoint.pt"
    ) -> "MultiExperimentAnimator":
        """Create animator for multiple experiments (side-by-side comparison).

        Parameters
        ----------
        experiment_paths : Sequence[Union[str, Path]]
            Sequence of experiment directory paths
        checkpoint_file : str, optional
            Name of checkpoint file (default: "checkpoint.pt")

        Returns
        -------
        MultiExperimentAnimator
            Animator for multiple experiments
        """
        animators = [cls(path, checkpoint_file) for path in experiment_paths]
        return MultiExperimentAnimator(animators)


class MultiExperimentAnimator:
    """Generate side-by-side comparison animations for multiple experiments.

    Parameters
    ----------
    animators : List[TrainingAnimator]
        List of individual experiment animators
    """

    def __init__(self, animators: List[TrainingAnimator]):
        """Initialize multi-experiment animator.

        Parameters
        ----------
        animators : List[TrainingAnimator]
            List of training animators to compare
        """
        if len(animators) < 2:
            raise ValueError("Need at least 2 experiments for comparison")
        if len(animators) > 4:
            raise ValueError("Maximum 4 experiments supported for comparison")

        self.animators = animators
        self.n_experiments = len(animators)

    def create_frame(
        self,
        frame_idx: int,
        reconstructions: List[np.ndarray],
        metrics_list: List[Dict[str, float]],
        layout: str = "grid",
    ) -> np.ndarray:
        """Create comparison frame.

        Parameters
        ----------
        frame_idx : int
            Frame number
        reconstructions : List[np.ndarray]
            Reconstructions for each experiment
        metrics_list : List[Dict[str, float]]
            Metrics for each experiment
        layout : str, optional
            Layout style: "grid" or "horizontal" (default: "grid")

        Returns
        -------
        np.ndarray
            Frame as RGB array
        """
        n_exp = len(reconstructions)

        # Determine subplot layout
        if layout == "horizontal" or n_exp == 2:
            nrows, ncols = 1, n_exp
            figsize = (6 * n_exp, 6)
        else:  # grid layout
            nrows = 2
            ncols = (n_exp + 1) // 2
            figsize = (6 * ncols, 6 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_exp == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot each experiment
        for i, (rec, metrics, animator) in enumerate(
            zip(reconstructions, metrics_list, self.animators)
        ):
            axes[i].imshow(rec, cmap="gray", vmin=0, vmax=1)

            # Create title with experiment name and metrics
            title = f"{animator.exp_name}"
            if "sample" in metrics:
                title += f"\nSample {metrics['sample']}/{metrics.get('total_samples', '?')}"

            axes[i].set_title(title, fontsize=10, fontweight="bold")
            axes[i].axis("off")

            # Add metrics as text overlay
            metrics_text = []
            if "loss" in metrics:
                metrics_text.append(f"Loss: {metrics['loss']:.6f}")
            if "ssim" in metrics:
                metrics_text.append(f"SSIM: {metrics['ssim']:.4f}")
            if "psnr" in metrics:
                metrics_text.append(f"PSNR: {metrics['psnr']:.2f} dB")

            if metrics_text:
                text_str = "\n".join(metrics_text)
                axes[i].text(
                    0.02,
                    0.98,
                    text_str,
                    transform=axes[i].transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        # Hide unused subplots
        for i in range(n_exp, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()

        # Convert to numpy array
        fig.canvas.draw()
        # Use buffer_rgba() for newer matplotlib compatibility
        buf = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
        frame = np.frombuffer(buf, dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        frame = frame[:, :, :3]
        plt.close(fig)

        return frame

    def generate_video(
        self,
        output_path: Union[str, Path],
        fps: int = 10,
        n_frames: Optional[int] = None,
        layout: str = "grid",
    ) -> None:
        """Generate comparison video.

        Parameters
        ----------
        output_path : Union[str, Path]
            Output video file path
        fps : int, optional
            Frames per second (default: 10)
        n_frames : int, optional
            Number of frames to generate. If None, auto-determined.
        layout : str, optional
            Layout style: "grid" or "horizontal" (default: "grid")

        Raises
        ------
        ImportError
            If opencv-python is not installed
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for video generation. "
                "Install it with: uv add opencv-python"
            )

        output_path = Path(output_path)
        console = Console()

        # Get reconstruction histories for all experiments
        all_reconstructions = [
            animator._get_reconstruction_history(n_frames) for animator in self.animators
        ]

        # Use minimum frame count
        total_frames = min(len(recs) for recs in all_reconstructions)

        # Generate frames with progress bar
        frames = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Generating frames ({self.n_experiments} experiments)...", total=total_frames
            )
            for i in range(total_frames):
                recs = [all_recs[i] for all_recs in all_reconstructions]
                metrics_list = [
                    animator._get_metrics_at_frame(i, total_frames) for animator in self.animators
                ]
                frame = self.create_frame(i, recs, metrics_list, layout=layout)
                frames.append(frame)
                progress.update(task, advance=1)

        # Create video
        console.print(f"Writing video to {output_path}...")
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video.release()
        console.print(f"[green]✓[/green] Comparison video saved: {output_path}")

    def generate_gif(
        self,
        output_path: Union[str, Path],
        duration: int = 100,
        n_frames: Optional[int] = None,
        layout: str = "grid",
        loop: int = 0,
    ) -> None:
        """Generate comparison GIF.

        Parameters
        ----------
        output_path : Union[str, Path]
            Output GIF file path
        duration : int, optional
            Duration per frame in milliseconds (default: 100)
        n_frames : int, optional
            Number of frames to generate. If None, auto-determined.
        layout : str, optional
            Layout style: "grid" or "horizontal" (default: "grid")
        loop : int, optional
            Number of loops (0 = infinite, default: 0)
        """
        output_path = Path(output_path)
        console = Console()

        # Get reconstruction histories for all experiments
        all_reconstructions = [
            animator._get_reconstruction_history(n_frames) for animator in self.animators
        ]

        # Use minimum frame count
        total_frames = min(len(recs) for recs in all_reconstructions)

        # Generate frames with progress bar
        pil_frames = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Generating frames ({self.n_experiments} experiments)...", total=total_frames
            )
            for i in range(total_frames):
                recs = [all_recs[i] for all_recs in all_reconstructions]
                metrics_list = [
                    animator._get_metrics_at_frame(i, total_frames) for animator in self.animators
                ]
                frame = self.create_frame(i, recs, metrics_list, layout=layout)
                pil_frames.append(Image.fromarray(frame))
                progress.update(task, advance=1)

        # Save GIF
        console.print(f"Writing GIF to {output_path}...")
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=loop,
            optimize=True,
        )
        console.print(f"[green]✓[/green] Comparison GIF saved: {output_path}")

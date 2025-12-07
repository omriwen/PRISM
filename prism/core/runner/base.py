"""
Abstract base class for experiment runners.

This module provides the AbstractRunner base class that defines the template
method pattern for running experiments in PRISM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter


@dataclass
class ExperimentResult:
    """Result of an experiment run.

    Parameters
    ----------
    ssims : list[float]
        SSIM values for each sample
    psnrs : list[float]
        PSNR values for each sample (dB)
    rmses : list[float]
        RMSE values for each sample
    final_reconstruction : torch.Tensor
        Final reconstructed image
    log_dir : Path
        Directory where logs were saved
    elapsed_time : float
        Total training time in seconds
    failed_samples : list[int]
        Indices of samples that failed to converge
    """
    ssims: list[float] = field(default_factory=list)
    psnrs: list[float] = field(default_factory=list)
    rmses: list[float] = field(default_factory=list)
    final_reconstruction: Optional[torch.Tensor] = None
    log_dir: Optional[Path] = None
    elapsed_time: float = 0.0
    failed_samples: list[int] = field(default_factory=list)


class AbstractRunner(ABC):
    """Abstract base class for experiment runners.

    This class implements the Template Method pattern for experiment execution.
    Subclasses must implement the abstract methods to provide algorithm-specific
    behavior.

    The `run()` method defines the workflow:
    1. setup() - Environment and configuration setup
    2. load_data() - Load input data (images, patterns)
    3. create_components() - Create model, telescope, trainer
    4. run_experiment() - Execute the training loop
    5. save_results() - Save checkpoints and figures
    6. cleanup() - Release resources

    Parameters
    ----------
    args : Any
        Parsed command-line arguments (argparse.Namespace)

    Examples
    --------
    >>> class MyRunner(AbstractRunner):
    ...     def setup(self): ...
    ...     def load_data(self): ...
    ...     def create_components(self): ...
    ...     def run_experiment(self): ...
    ...     def save_results(self, result): ...
    >>> runner = MyRunner(args)
    >>> result = runner.run()
    """

    def __init__(self, args: Any) -> None:
        """Initialize the runner with command-line arguments.

        Parameters
        ----------
        args : Any
            Parsed command-line arguments
        """
        self.args = args
        self.config: Any = None
        self.device: Optional[torch.device] = None
        self.log_dir: Optional[Path] = None
        self.writer: Optional[SummaryWriter] = None

    def run(self) -> ExperimentResult:
        """Execute the complete experiment workflow.

        This is the template method that defines the experiment execution order.
        Subclasses should not override this method; instead, override the
        abstract hook methods.

        Returns
        -------
        ExperimentResult
            Results of the experiment including metrics and reconstruction

        Raises
        ------
        Exception
            Any exception from the workflow is re-raised after cleanup
        """
        try:
            self.setup()
            self.load_data()
            self.create_components()
            result = self.run_experiment()
            self.save_results(result)
            return result
        finally:
            self.cleanup()

    @abstractmethod
    def setup(self) -> None:
        """Set up the experiment environment.

        This method should:
        - Configure the device (CPU/GPU)
        - Set up logging directories
        - Initialize TensorBoard writer
        - Load and validate configuration

        Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def load_data(self) -> None:
        """Load input data for the experiment.

        This method should:
        - Load or generate input images
        - Generate or load sampling patterns
        - Prepare data tensors on the appropriate device

        Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def create_components(self) -> None:
        """Create experiment components.

        This method should:
        - Create the model (decoder/algorithm)
        - Create the optical system (telescope/microscope)
        - Create the trainer
        - Set up optimizers and schedulers

        Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def run_experiment(self) -> ExperimentResult:
        """Execute the main experiment logic.

        This method should:
        - Run initialization training (if applicable)
        - Run the main training loop
        - Collect and return results

        Must be implemented by subclasses.

        Returns
        -------
        ExperimentResult
            Results of the experiment
        """
        ...

    @abstractmethod
    def save_results(self, result: ExperimentResult) -> None:
        """Save experiment results.

        This method should:
        - Save final checkpoint
        - Generate and save visualization figures
        - Log final metrics

        Must be implemented by subclasses.

        Parameters
        ----------
        result : ExperimentResult
            The experiment results to save
        """
        ...

    def cleanup(self) -> None:
        """Clean up resources after experiment completion.

        Default implementation closes TensorBoard writer.
        Subclasses can override to add additional cleanup.
        """
        if self.writer is not None:
            self.writer.close()

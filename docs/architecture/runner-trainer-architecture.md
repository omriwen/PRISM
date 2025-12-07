# Runner and Trainer Architecture

This document describes the unified runner and trainer architecture in PRISM,
which provides a common framework for running both PRISM (deep learning) and
MoPIE (iterative phase retrieval) experiments.

## Overview

The architecture implements the Template Method design pattern for experiment
execution and provides a clean separation between:

- **Runners**: Orchestrate the complete experiment workflow
- **Trainers**: Handle the training loop and metric collection
- **Entry Points**: Provide unified CLI access to all algorithms

## Class Hierarchy

### Runner Hierarchy

```
AbstractRunner (base.py)
├── PRISMRunner (prism_runner.py)
│   └── Uses ProgressiveTrainer for deep learning reconstruction
└── MoPIERunner (mopie_runner.py)
    └── Uses EpochalTrainer for iterative phase retrieval
```

### Trainer Hierarchy

```
AbstractTrainer (base.py)
├── PRISMTrainer (progressive.py)
│   └── Progressive sample-by-sample training with convergence checking
└── EpochalTrainer (epochal.py)
    └── Epochal training over all samples (MoPIE algorithm)
```

## AbstractRunner

The `AbstractRunner` class defines the template for experiment execution:

```python
class AbstractRunner(ABC):
    def run(self) -> ExperimentResult:
        """Template method defining experiment workflow."""
        try:
            self.setup()           # 1. Environment setup
            self.load_data()       # 2. Load images and patterns
            self.create_components()  # 3. Create model, telescope, trainer
            result = self.run_experiment()  # 4. Run training
            self.save_results(result)  # 5. Save outputs
            return result
        finally:
            self.cleanup()         # 6. Cleanup resources
```

### Abstract Methods

Subclasses must implement these methods:

| Method | Purpose |
|--------|---------|
| `setup()` | Configure device, logging, directories |
| `load_data()` | Load images and generate sampling patterns |
| `create_components()` | Create model, telescope, trainer |
| `run_experiment()` | Execute training and return results |
| `save_results(result)` | Save checkpoints and visualizations |

### Concrete Methods

These methods have default implementations:

| Method | Default Behavior |
|--------|-----------------|
| `cleanup()` | Close TensorBoard writer |

## AbstractTrainer

The `AbstractTrainer` class defines the interface for training algorithms:

```python
class AbstractTrainer(ABC):
    def __init__(self, model, device, config=None):
        self.model = model
        self.device = device
        self.config = config
        self.metrics = MetricsCollector()
```

### Abstract Methods

| Method | Purpose |
|--------|---------|
| `train(**kwargs)` | Execute complete training |
| `train_epoch(epoch, **kwargs)` | Execute single epoch |
| `compute_metrics(output, target)` | Compute SSIM, PSNR, RMSE |

### Concrete Methods

| Method | Default Behavior |
|--------|-----------------|
| `should_stop(loss)` | Check loss threshold |
| `save_checkpoint(path, **kwargs)` | Save model state |

## Data Classes

### ExperimentResult

Returned by `AbstractRunner.run()`:

```python
@dataclass
class ExperimentResult:
    ssims: list[float]          # SSIM values per sample
    psnrs: list[float]          # PSNR values in dB
    rmses: list[float]          # RMSE values
    final_reconstruction: Tensor  # Final image
    log_dir: Path               # Output directory
    elapsed_time: float         # Total time in seconds
    failed_samples: list[int]   # Failed sample indices
```

### TrainingResult

Returned by `AbstractTrainer.train()`:

```python
@dataclass
class TrainingResult:
    losses: list[float]         # Loss per sample
    ssims: list[float]          # SSIM per sample
    psnrs: list[float]          # PSNR per sample
    rmses: list[float]          # RMSE per sample
    final_reconstruction: Tensor
    failed_samples: list[int]
    wall_time_seconds: float
    epochs_per_sample: list[int]
```

### TrainingConfig

Configuration for trainers:

```python
@dataclass
class TrainingConfig:
    n_epochs: int = 100         # Epochs per iteration
    max_epochs: int = 100       # Max iterations
    loss_threshold: float = 1e-4
    learning_rate: float = 1e-4
```

## Unified Entry Point

The unified CLI entry point supports both algorithms:

```bash
# Run PRISM (default)
python main.py --algorithm prism --obj_name europa

# Run MoPIE
python main.py --algorithm mopie --obj_name europa
```

### RunnerFactory

Creates the appropriate runner based on algorithm:

```python
class RunnerFactory:
    @classmethod
    def create(cls, algorithm: str, args) -> AbstractRunner:
        runners = {
            "prism": PRISMRunner,
            "mopie": MoPIERunner,
        }
        return runners[algorithm.lower()](args)
```

## Usage Examples

### Running an Experiment

```python
from prism.core.runner.factory import RunnerFactory

# Create runner for chosen algorithm
runner = RunnerFactory.create("prism", args)

# Run experiment
result = runner.run()

# Access results
print(f"Final SSIM: {result.ssims[-1]:.4f}")
print(f"Final PSNR: {result.psnrs[-1]:.2f} dB")
```

### Creating a Custom Runner

```python
from prism.core.runner.base import AbstractRunner, ExperimentResult

class CustomRunner(AbstractRunner):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = Path(self.args.log_dir) / self.args.name

    def load_data(self):
        self.image = load_image(self.args.input)
        self.sample_centers = generate_pattern(self.args)

    def create_components(self):
        self.model = create_model(self.args)
        self.trainer = create_trainer(self.model, self.device)

    def run_experiment(self) -> ExperimentResult:
        result = self.trainer.train()
        return ExperimentResult(
            ssims=result.ssims,
            psnrs=result.psnrs,
            rmses=result.rmses,
            final_reconstruction=result.final_reconstruction,
            log_dir=self.log_dir,
            elapsed_time=result.wall_time_seconds,
        )

    def save_results(self, result):
        torch.save(result.final_reconstruction, self.log_dir / "reconstruction.pt")
```

### Creating a Custom Trainer

```python
from prism.core.trainers.base import AbstractTrainer, TrainingResult

class CustomTrainer(AbstractTrainer):
    def train(self, **kwargs) -> TrainingResult:
        self.metrics.start()

        for sample in samples:
            loss = self._train_sample(sample)
            metrics = self.compute_metrics(output, target)
            self.metrics.record_sample(
                loss=loss,
                ssim=metrics["ssim"],
                psnr=metrics["psnr"],
                rmse=metrics["rmse"],
                epochs=epochs_used,
            )

            if self.should_stop(loss):
                break

        return self.metrics.finalize(self.current_reconstruction)

    def train_epoch(self, epoch, **kwargs):
        # Implement epoch training
        return EpochResult(loss=total_loss)

    def compute_metrics(self, output, target):
        return {
            "ssim": compute_ssim(output, target),
            "psnr": compute_psnr(output, target),
            "rmse": compute_rmse(output, target),
        }
```

## Mixins for Shared Functionality

Common setup logic is shared via mixins in `mixins.py`:

```python
class SetupMixin:
    def _setup_device(self) -> torch.device
    def _setup_logging(self) -> Path

class DataLoadingMixin:
    def _load_image(self) -> torch.Tensor
    def _generate_pattern(self) -> tuple

class LineSamplingMixin:
    def _configure_line_sampling(self) -> dict
```

## Metrics Collection

The `MetricsCollector` class tracks metrics during training:

```python
collector = MetricsCollector()
collector.start()

for sample in samples:
    # Train sample...
    collector.record_sample(loss, ssim, psnr, rmse, epochs)

if failed:
    collector.record_failure(sample_idx)

result = collector.finalize(final_reconstruction)
```

## File Locations

| File | Purpose |
|------|---------|
| `prism/core/runner/base.py` | AbstractRunner, ExperimentResult |
| `prism/core/runner/prism_runner.py` | PRISMRunner implementation |
| `prism/core/runner/mopie_runner.py` | MoPIERunner implementation |
| `prism/core/runner/factory.py` | RunnerFactory |
| `prism/core/runner/mixins.py` | Shared setup mixins |
| `prism/core/trainers/base.py` | AbstractTrainer, TrainingConfig, etc. |
| `prism/core/trainers/progressive.py` | PRISMTrainer |
| `prism/core/trainers/epochal.py` | EpochalTrainer |
| `prism/cli/entry_points.py` | Unified CLI entry point |

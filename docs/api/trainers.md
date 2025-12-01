# prism.core.trainers

Training loops for PRISM algorithms.

This module provides trainers for progressive reconstruction using
PRISM and ePIE algorithms.

## Classes

### PRISMTrainer

```python
PRISMTrainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, telescope_agg: prism.core.aggregator.TelescopeAggregator, args: Any, device: torch.device, log_dir: str | None = None, writer: torch.utils.tensorboard.writer.SummaryWriter | None = None, use_amp: bool = False)
```

Progressive trainer for PRISM algorithm.

Handles both initialization and progressive training phases with
convergence checking, checkpoint saving, and TensorBoard logging.

Parameters
----------
model : nn.Module
    Generative model to train (ProgressiveDecoder)
optimizer : torch.optim.Optimizer
    PyTorch optimizer
scheduler : torch.optim.lr_scheduler._LRScheduler
    Learning rate scheduler
telescope_agg : TelescopeAggregator
    Telescope measurement aggregator
args : argparse.Namespace
    Training arguments and configuration
device : torch.device
    Device to run training on
log_dir : str | None, optional
    Directory for saving checkpoints and logs
writer : SummaryWriter | None, optional
    TensorBoard writer for logging

Examples
--------
>>> model = ProgressiveDecoder(...)
>>> optimizer = torch.optim.Adam(model.parameters())
>>> trainer = PRISMTrainer(model, optimizer, ...)
>>> trainer.run_initialization(measurement)
>>> results = trainer.run_progressive_training(sample_centers, image_gt)

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

##### `retry_failed_samples`

Retry failed samples with alternative strategies.

Parameters
----------
sample_centers : torch.Tensor
    All sample center positions
image : torch.Tensor
    Input image
image_gt : torch.Tensor
    Ground truth image
samples_per_line_meas : int
    Samples per line measurement
criterion : LossAggregator
    Loss criterion

Returns
-------
int
    Number of samples recovered during retry

##### `run_initialization`

Run initialization phase: train model on initial target.

Parameters
----------
measurement : torch.Tensor
    Initial measurement or circle mask to train on
center : torch.Tensor
    Sample center for visualization
image_gt : torch.Tensor
    Ground truth image for SSIM computation
figure : Any, optional
    Matplotlib figure handle for visualization

Returns
-------
tuple[torch.Tensor, Any]
    Current reconstruction and updated figure handle

##### `run_progressive_training`

Run progressive training over multiple sample positions.

Parameters
----------
sample_centers : torch.Tensor
    Sample center positions [N, 2]
image : torch.Tensor
    Input image (padded)
image_gt : torch.Tensor
    Ground truth image for metrics
samples_per_line_meas : int
    Number of samples per line measurement
figure : Any, optional
    Matplotlib figure handle for visualization
pattern_metadata : dict, optional
    Pattern generation metadata
pattern_spec : str, optional
    Pattern specification string

Returns
-------
dict[str, Any]
    Training results including losses, metrics, and reconstruction

## Functions

### create_scheduler

```python
create_scheduler(optimizer: torch.optim.optimizer.Optimizer, scheduler_type: str, **kwargs: Any) -> torch.optim.lr_scheduler.LRScheduler
```

Create scheduler based on type.

Parameters
----------
optimizer : torch.optim.Optimizer
    The optimizer to schedule
scheduler_type : str
    Type of scheduler: 'plateau' or 'cosine_warm_restarts'
**kwargs : Any
    Additional arguments passed to scheduler

Returns
-------
torch.optim.lr_scheduler.LRScheduler
    Configured scheduler

Raises
------
ValueError
    If scheduler_type is unknown

# prism.core.runner

Experiment runner for PRISM algorithm.

This module orchestrates the complete experimental workflow including
setup, initialization, training, and result generation.

## Classes

### PRISMRunner

```python
PRISMRunner(args: Any)
```

Orchestrates PRISM experiment from setup to completion.

Handles all phases of a PRISM experiment:
- Argument validation and setup
- Image loading and preprocessing
- Pattern generation
- Model and telescope initialization
- Training (initialization + progressive)
- Checkpoint and figure generation

Parameters
----------
args : argparse.Namespace
    Parsed command-line arguments

Examples
--------
>>> from prism.cli.parser import create_main_parser
>>> parser = create_main_parser()
>>> args = parser.parse_args(['--obj_name', 'europa', '--name', 'test'])
>>> runner = PRISMRunner(args)
>>> runner.run()

#### Methods

##### `__init__`

Initialize self.  See help(type(self)) for accurate signature.

##### `cleanup`

Cleanup resources.

##### `create_model_and_telescope`

Initialize model and telescope components.

##### `create_trainer`

Create trainer instance.

##### `load_checkpoint_if_needed`

Load checkpoint if specified. Returns True if checkpoint loaded.

##### `load_image_and_pattern`

Load input image and generate sampling pattern.

##### `run`

Run complete PRISM experiment.

##### `run_initialization`

Run initialization phase and return figure handle.

##### `run_training`

Run progressive training and return results.

##### `save_final_checkpoint`

Save final checkpoint for single-sample runs.

##### `save_final_figures`

Generate and save final visualization figures.

##### `setup`

Setup experiment: device, logging, directories, and configuration.

##### `start_dashboard_if_requested`

Launch dashboard if --dashboard flag is set.

##### `stop_dashboard`

Stop dashboard if it was started.

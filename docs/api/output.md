# prism.output

Output management system for PRISM experiments.

This module provides structured output management including:
- Organized directory structure for experiment outputs
- Checkpoint management with versioning
- Metadata tracking with hardware information
- Configuration saving and loading
- Metrics tracking and visualization support

Examples
--------
>>> from prism.output import OutputManager, ExperimentMetadata
>>>
>>> # Create output manager
>>> manager = OutputManager('runs/', 'my_experiment')
>>>
>>> # Save configuration
>>> config = {'lr': 0.001, 'epochs': 100}
>>> manager.save_config(config)
>>>
>>> # Track metadata
>>> metadata = ExperimentMetadata(experiment_name='my_experiment')
>>> metadata.start()
>>> # ... training ...
>>> metadata.record_best_metrics({'ssim': 0.95})
>>> metadata.end()
>>> metadata.save(manager.base_dir / 'metadata.json')

## Classes

# PRISM API Reference

Complete API documentation for the PRISM package.

## Package Organization

### Core

- [prism.core](core.md)
- [prism.core.aggregator](aggregator.md)
- [prism.core.algorithms](algorithms.md)
- [prism.core.algorithms.epie](epie.md)
- [prism.core.apertures](apertures.md)
- [prism.core.convergence](convergence.md)
- [prism.core.grid](grid.md)
- [prism.core.instruments](instruments.md)
- [prism.core.instruments.base](base.md)
- [prism.core.instruments.camera](camera.md)
- [prism.core.instruments.microscope](microscope.md)
- [prism.core.instruments.utils](utils.md)
- [prism.core.pattern_builtins](pattern_builtins.md)
- [prism.core.pattern_library](pattern_library.md)
- [prism.core.pattern_loader](pattern_loader.md)
- [prism.core.pattern_preview](pattern_preview.md)
- [prism.core.patterns](patterns.md)
- [prism.core.propagators](propagators.md)
- [prism.core.propagators.angular_spectrum](angular_spectrum.md)
- [prism.core.propagators.base](base.md)
- [prism.core.propagators.fraunhofer](fraunhofer.md)
- [prism.core.propagators.fresnel](fresnel.md)
- [prism.core.propagators.incoherent](incoherent.md)
- [prism.core.propagators.utils](utils.md)
- [prism.core.runner](runner.md)
- [prism.core.telescope](telescope.md)
- [prism.core.trainers](trainers.md)

### Models

- [prism.models](models.md)
- [prism.models.layers](layers.md)
- [prism.models.losses](losses.md)
- [prism.models.network_builder](network_builder.md)
- [prism.models.network_config](network_config.md)
- [prism.models.networks](networks.md)
- [prism.models.noise](noise.md)

### Utils

- [prism.utils](utils.md)
- [prism.utils.image](image.md)
- [prism.utils.io](io.md)
- [prism.utils.logging_config](logging_config.md)
- [prism.utils.measurement_cache](measurement_cache.md)
- [prism.utils.metrics](metrics.md)
- [prism.utils.progress](progress.md)
- [prism.utils.sampling](sampling.md)
- [prism.utils.training_helpers](training_helpers.md)
- [prism.utils.transforms](transforms.md)
- [prism.utils.visualization](visualization.md)

### Config

- [prism.cli.patterns.config](config.md)
- [prism.config](config.md)
- [prism.config.base](base.md)
- [prism.config.constants](constants.md)
- [prism.config.inspector](inspector.md)
- [prism.config.interactive](interactive.md)
- [prism.config.loader](loader.md)
- [prism.config.objects](objects.md)
- [prism.config.presets](presets.md)
- [prism.config.validation](validation.md)

### Visualization

- [prism.visualization](visualization.md)
- [prism.visualization.animation](animation.md)

### Reporting

- [prism.reporting](reporting.md)
- [prism.reporting.generator](generator.md)

### Training

- [prism.web.callbacks.training](training.md)

## Quick Links

### Essential Classes

- [Telescope](telescope.md) - Telescope aperture simulation
- [Microscope](microscope.md) - High-NA microscope simulation
- [Camera](camera.md) - General camera simulation
- [Instruments](instruments.md) - Multi-instrument factory and base classes
- [TelescopeAgg](aggregator.md) - Progressive measurement aggregation
- [ProgressiveDecoder](networks.md) - Primary generative model
- [LossAgg](losses.md) - Progressive loss function

### Key Utilities

- [Image Operations](image.md) - Loading and preprocessing
- [Sampling](sampling.md) - k-space sampling patterns
- [Metrics](metrics.md) - SSIM, PSNR, RMSE
- [Visualization](publication.md) - Publication-quality plots

### Configuration

- [Base Config](base.md) - Configuration dataclasses
- [Config Loader](loader.md) - YAML/JSON loading
- [Constants](constants.md) - Physical constants

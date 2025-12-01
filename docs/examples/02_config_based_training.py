"""Example 2: Configuration-Based Training

This example demonstrates using YAML configuration files for
reproducible experiments.
"""

from pathlib import Path

from prism.config.base import ExperimentConfig, ImageConfig, TelescopeConfig, TrainingConfig
from prism.config.loader import load_config, save_config


# Create configuration programmatically
config = ExperimentConfig(
    name="europa_reconstruction",
    image=ImageConfig(
        obj_name="europa",  # Predefined astronomical object
        image_size=1024,
        obj_size=512,
        invert=False,
        crop=True,
    ),
    telescope=TelescopeConfig(sample_diameter=100, sample_shape="circle", roi_diameter=600, snr=40),
    training=TrainingConfig(
        n_epochs=1000, max_epochs=25, lr=0.001, loss_type="l1", loss_threshold=0.001
    ),
)

# Save configuration to YAML
config_dir = Path("configs")
config_dir.mkdir(exist_ok=True)
config_path = config_dir / "europa_example.yaml"

save_config(config, config_path)
print(f"Configuration saved to {config_path}")

# Load configuration from YAML
loaded_config = load_config(config_path)
print("\nLoaded configuration:")
print(f"  Name: {loaded_config.name}")
print(f"  Object: {loaded_config.image.obj_name}")
print(f"  Image size: {loaded_config.image.image_size}")
print(f"  Object size: {loaded_config.image.obj_size}")
print(f"  Aperture diameter: {loaded_config.telescope.sample_diameter}")
print(f"  SNR: {loaded_config.telescope.snr} dB")
print(f"  Learning rate: {loaded_config.training.lr}")

# Example: Using configuration in training
print("\nConfiguration can now be used in training loop:")
print(
    f"  model = ProgressiveDecoder(input_size={loaded_config.image.image_size}, "
    f"output_size={loaded_config.image.obj_size})"
)
print(
    f"  telescope = TelescopeAgg(n={loaded_config.image.image_size}, "
    f"r={loaded_config.telescope.sample_diameter / 2})"
)
print(f"  optimizer = Adam(model.parameters(), lr={loaded_config.training.lr})")

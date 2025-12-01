"""Test script for SPIDS configuration system."""

from __future__ import annotations

import sys
from pathlib import Path


# Test imports
print("Testing configuration imports...")
try:
    from prism.config import (
        ImageConfig,
        TelescopeConfig,
        TrainingConfig,
        args_to_config,
        load_config,
        save_config,
    )
    from prism.config import PRISMConfig as ExperimentConfig

    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test creating config programmatically
print("\nTesting programmatic config creation...")
try:
    config = ExperimentConfig(
        name="test_experiment",
        image=ImageConfig(image_size=512),
        telescope=TelescopeConfig(n_samples=100, fermat_sample=True),
        training=TrainingConfig(lr=0.0005, max_epochs=10),
    )
    print("✓ Programmatic config creation successful")
    print(f"  Config name: {config.name}")
    print(f"  Image size: {config.image.image_size}")
    print(f"  Samples: {config.telescope.n_samples}")
    print(f"  Learning rate: {config.training.lr}")
except Exception as e:
    print(f"✗ Config creation failed: {e}")
    sys.exit(1)

# Test loading from YAML
print("\nTesting YAML config loading...")
yaml_configs = [
    "configs/default.yaml",
    "configs/quick_test.yaml",
    "configs/production_europa.yaml",
    "configs/point_source_test.yaml",
]

for yaml_path in yaml_configs:
    try:
        config = load_config(yaml_path)
        print(f"✓ Loaded {yaml_path}")
        print(f"  Samples: {config.telescope.n_samples}")
        print(f"  Object: {config.physics.obj_name}")
    except FileNotFoundError:
        print(f"✗ File not found: {yaml_path}")
    except Exception as e:
        print(f"✗ Failed to load {yaml_path}: {e}")
        sys.exit(1)

# Test saving config
print("\nTesting config saving...")
try:
    test_config = ExperimentConfig(name="save_test")
    save_path = Path("/tmp/test_config.yaml")
    save_config(test_config, str(save_path))
    print(f"✓ Saved config to {save_path}")

    # Load it back
    loaded = load_config(str(save_path))
    print("✓ Loaded saved config back")
    assert loaded.name == "save_test", "Config name mismatch"
    print("✓ Config round-trip successful")

    # Cleanup
    save_path.unlink()
except Exception as e:
    print(f"✗ Save/load test failed: {e}")
    sys.exit(1)

# Test argparse integration
print("\nTesting argparse integration...")
try:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--obj_name", type=str, default="europa")
    parser.add_argument("--name", type=str, default="test")

    # Simulate command line args
    args = parser.parse_args(["--n_samples", "150", "--lr", "0.0005", "--name", "argparse_test"])

    # Convert to config
    config = args_to_config(args)
    print("✓ Args to config conversion successful")
    print(f"  Samples: {config.telescope.n_samples} (should be 150)")
    print(f"  Learning rate: {config.training.lr} (should be 0.0005)")
    print(f"  Name: {config.name} (should be argparse_test)")

    assert config.telescope.n_samples == 150, "Sample count not updated"
    assert config.training.lr == 0.0005, "Learning rate not updated"
    assert config.name == "argparse_test", "Name not updated"
    print("✓ All argparse values correctly transferred")
except Exception as e:
    print(f"✗ Argparse test failed: {e}")
    sys.exit(1)

# Test string representation
print("\nTesting config string representation...")
try:
    config = ExperimentConfig(name="display_test")
    config_str = str(config)
    print("✓ Config string representation:")
    print(config_str)
except Exception as e:
    print(f"✗ String representation failed: {e}")
    sys.exit(1)

# Test to_dict conversion
print("\nTesting to_dict conversion...")
try:
    config = ExperimentConfig(name="dict_test")
    config_dict = config.to_dict()
    print("✓ Config to_dict conversion successful")
    assert "name" in config_dict, "Missing 'name' in dict"
    assert "image" in config_dict, "Missing 'image' in dict"
    assert "telescope" in config_dict, "Missing 'telescope' in dict"
    print(f"  Dict keys: {list(config_dict.keys())}")
except Exception as e:
    print(f"✗ to_dict conversion failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All configuration tests passed! ✓")
print("=" * 60)

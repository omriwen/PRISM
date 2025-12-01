"""Test script for SPIDS configuration system."""

from __future__ import annotations

import os
import tempfile

from prism.config import PRISMConfig, load_config, save_config


def test_load_default_config():
    """Test loading default.yaml"""
    print("=" * 60)
    print("Test 1: Load default.yaml")
    print("=" * 60)

    config = load_config("configs/default.yaml")
    print("✓ Loaded config successfully")
    print(f"  - obj_name: {config.physics.obj_name}")
    print(f"  - n_samples: {config.telescope.n_samples}")
    print(f"  - fermat_sample: {config.telescope.fermat_sample}")
    print(f"  - image_size: {config.image.image_size}")
    print()


def test_load_quick_test():
    """Test loading quick_test.yaml"""
    print("=" * 60)
    print("Test 2: Load quick_test.yaml")
    print("=" * 60)

    config = load_config("configs/quick_test.yaml")
    print("✓ Loaded config successfully")
    print(f"  - n_samples: {config.telescope.n_samples}")
    print(f"  - sample_length: {config.telescope.sample_length}")
    print(f"  - max_epochs: {config.training.max_epochs}")
    print(f"  - save_data: {config.save_data}")
    print()


def test_load_production_europa():
    """Test loading production_europa.yaml"""
    print("=" * 60)
    print("Test 3: Load production_europa.yaml")
    print("=" * 60)

    config = load_config("configs/production_europa.yaml")
    print("✓ Loaded config successfully")
    print(f"  - n_samples: {config.telescope.n_samples}")
    print(f"  - snr: {config.telescope.snr}")
    print(f"  - max_epochs: {config.training.max_epochs}")
    print()


def test_load_mopie_example():
    """Test loading mopie_example.yaml"""
    print("=" * 60)
    print("Test 4: Load mopie_example.yaml")
    print("=" * 60)

    config = load_config("configs/mopie_example.yaml")
    print("✓ Loaded config successfully")
    print(f"  - n_epochs: {config.training.n_epochs}")
    assert config.mopie is not None, "Mo-PIE config should not be None"
    print(f"  - mopie.lr_obj: {config.mopie.lr_obj}")
    print(f"  - mopie.fix_probe: {config.mopie.fix_probe}")
    print(f"  - mopie.plot_every: {config.mopie.plot_every}")
    print()


def test_save_and_reload():
    """Test saving and reloading a config"""
    print("=" * 60)
    print("Test 5: Save and reload config")
    print("=" * 60)

    # Load a config
    config1 = load_config("configs/quick_test.yaml")

    # Modify it
    config1.telescope.n_samples = 999
    config1.name = "test_save"

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, "test_config.yaml")
        save_config(config1, temp_path)
        print(f"✓ Saved config to {temp_path}")

        # Reload it
        config2 = load_config(temp_path)
        print("✓ Reloaded config successfully")

        # Verify
        assert config2.telescope.n_samples == 999
        assert config2.name == "test_save"
        print(f"  - n_samples: {config2.telescope.n_samples} (modified)")
        print(f"  - name: {config2.name}")
    print()


def test_validation():
    """Test config validation"""
    print("=" * 60)
    print("Test 6: Config validation")
    print("=" * 60)

    config = PRISMConfig()

    # Valid config
    try:
        config.validate()
        print("✓ Valid config passed validation")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")

    # Invalid config (negative learning rate)
    config.training.lr = -0.001
    try:
        config.validate()
        print("✗ Invalid config should have failed validation")
    except ValueError as e:
        print(f"✓ Invalid config caught: {e}")

    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("SPIDS Configuration System Tests")
    print("=" * 60)
    print()

    try:
        test_load_default_config()
        test_load_quick_test()
        test_load_production_europa()
        test_load_mopie_example()
        test_save_and_reload()
        test_validation()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()

    except Exception as e:
        print("=" * 60)
        print("✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

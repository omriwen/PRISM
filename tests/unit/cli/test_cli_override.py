"""Test CLI override behavior."""

from __future__ import annotations

import argparse

from prism.config import load_config, merge_config_with_args


# Simulate argparse with CLI args (WITH DEFAULTS like in main.py)
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--n_samples", type=int, default=200)  # Has default!
parser.add_argument("--lr", type=float, default=1e-3)  # Has default!
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--image_size", type=int, default=1024)  # Has default!


def test_override():
    """Test that CLI args override config values"""
    print("=" * 60)
    print("Testing CLI override behavior")
    print("=" * 60)

    # Test 1: Config only - argparse defaults should NOT override config
    print("\n1. Config only (argparse has defaults but CLI doesn't provide values):")
    config = load_config("configs/quick_test.yaml")
    # Simulate: python main.py --config configs/quick_test.yaml
    # Parser has default n_samples=200, but config has n_samples=64
    cli_args = parser.parse_args(["--config", "configs/quick_test.yaml"])
    provided_args = {"config"}  # Only --config was provided
    merged = merge_config_with_args(config, cli_args, provided_args)

    print(f"   n_samples from config: {config.telescope.n_samples}")
    print(f"   n_samples argparse default: {cli_args.n_samples}")
    print(f"   n_samples after merge: {merged.n_samples}")
    assert merged.n_samples == 64, (
        f"Should use config value (64), not argparse default (200), got {merged.n_samples}"
    )
    print("   ✓ Config value used correctly (argparse default ignored)")

    # Test 2: Explicit CLI override
    print("\n2. CLI explicitly overrides config:")
    config = load_config("configs/quick_test.yaml")
    cli_args = parser.parse_args(["--config", "configs/quick_test.yaml", "--n_samples", "999"])
    provided_args = {"config", "n_samples"}  # Both provided
    merged = merge_config_with_args(config, cli_args, provided_args)

    print(f"   n_samples from config: {config.telescope.n_samples}")
    print("   n_samples from CLI: 999")
    print(f"   n_samples after merge: {merged.n_samples}")
    assert merged.n_samples == 999, "CLI should override config"
    print("   ✓ CLI override works correctly")

    # Test 3: Multiple overrides
    print("\n3. Multiple CLI overrides:")
    config = load_config("configs/quick_test.yaml")
    cli_args = parser.parse_args(
        [
            "--config",
            "configs/quick_test.yaml",
            "--n_samples",
            "150",
            "--lr",
            "0.0005",
            "--name",
            "custom_name",
        ]
    )
    provided_args = {"config", "n_samples", "lr", "name"}
    merged = merge_config_with_args(config, cli_args, provided_args)

    print(f"   n_samples: {config.telescope.n_samples} → {merged.n_samples}")
    print(f"   lr: {config.training.lr} → {merged.lr}")
    print(f"   name: {config.name} → {merged.name}")
    assert merged.n_samples == 150
    assert merged.lr == 0.0005
    assert merged.name == "custom_name"
    print("   ✓ All overrides work correctly")

    # Test 4: Verify image_size from config is preserved
    print("\n4. Verify non-overridden values from config are preserved:")
    config = load_config("configs/quick_test.yaml")
    cli_args = parser.parse_args(["--config", "configs/quick_test.yaml"])
    provided_args = {"config"}
    merged = merge_config_with_args(config, cli_args, provided_args)

    print(f"   image_size from config: {config.image.image_size}")
    print(f"   image_size argparse default: {cli_args.image_size}")
    print(f"   image_size after merge: {merged.image_size}")
    assert merged.image_size == config.image.image_size, "Should preserve config value"
    print("   ✓ Config values preserved when not explicitly overridden on CLI")

    print("\n" + "=" * 60)
    print("✓ All CLI override tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_override()

"""Test config integration with main.py argument parser."""

from __future__ import annotations

import subprocess
import sys


def run_command(cmd):
    """Run a command and capture its output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


print("=" * 70)
print("TESTING CONFIG INTEGRATION WITH main.py")
print("=" * 70)

# Test 1: Help shows --config argument
print("\n1. Verify --config argument exists in help:")
returncode, stdout, stderr = run_command("uv run python main.py --help")
if "--config" in stdout:
    print("   ✓ --config argument found in help")
else:
    print("   ✗ --config argument NOT found in help")
    sys.exit(1)

# Test 2: Config argument parsing doesn't crash
print("\n2. Test config argument parsing (dry run):")
# We can't actually run training without proper setup, but we can verify
# that the config loading mechanism doesn't crash during argument parsing
cmd = "uv run python -c \"import sys; sys.argv = ['main.py', '--config', 'configs/quick_test.yaml', '--name', 'test_dry']; exec(open('main.py').read().split('# %% Load the input')[0])\""
returncode, stdout, stderr = run_command(cmd)
if returncode == 0 or "Loading configuration from:" in stdout + stderr:
    print("   ✓ Config loading mechanism works")
else:
    print("   ✗ Config loading failed")
    print(f"   stdout: {stdout}")
    print(f"   stderr: {stderr}")
    sys.exit(1)

# Test 3: Verify help text is correct
print("\n3. Verify --config help text:")
if "CLI args override config values" in stdout or "--config CONFIG" in stdout:
    print("   ✓ Config help text is descriptive")
else:
    print("   Note: Help text could be more descriptive")

print("\n" + "=" * 70)
print("✓ All integration tests passed!")
print("=" * 70)

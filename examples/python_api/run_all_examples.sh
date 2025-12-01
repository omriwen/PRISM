#!/bin/bash
# Run all Python API examples

set -e

echo "Running all Python API examples..."
echo ""

cd "$(dirname "$0")"

examples=(
    "01_basic_usage.py"
    "02_custom_patterns.py"
    "03_custom_loss.py"
    "04_model_extension.py"
    "05_batch_experiments.py"
)

for example in "${examples[@]}"; do
    echo "========================================"
    echo "Running: $example"
    echo "========================================"
    echo ""

    uv run python "$example"

    echo ""
    echo "✓ $example completed"
    echo ""
done

echo "========================================"
echo "All examples completed successfully!"
echo "========================================"

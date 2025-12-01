# Python API Examples

This directory contains examples of using SPIDS programmatically via the Python API,
rather than the command-line interface.

## Examples

1. **`01_basic_usage.py`** - Basic reconstruction workflow
2. **`02_custom_patterns.py`** - Custom sampling pattern functions
3. **`03_custom_loss.py`** - Custom loss functions
4. **`04_model_extension.py`** - Extending model architectures
5. **`05_batch_experiments.py`** - Running multiple experiments programmatically

## Running Examples

```bash
cd examples/python_api

# Run individual example
uv run python 01_basic_usage.py

# Run all examples
for f in *.py; do
    echo "Running $f..."
    uv run python "$f"
done
```

## Prerequisites

All examples assume SPIDS is installed:

```bash
cd ../..
uv sync
```

## Notes

- Examples are self-contained and can be run independently
- Each example includes comments explaining the code
- Modify parameters to experiment with different configurations

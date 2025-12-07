"""
PRISM - Progressive Reconstruction from Incomplete Sparse Measurements.

This is the unified entry point for PRISM experiments. It supports both:
- PRISM algorithm (deep learning reconstruction)
- MoPIE algorithm (iterative phase retrieval)

Usage
-----
Run with PRISM (default):
    python main.py --obj_name europa --n_samples 100

Run with MoPIE:
    python main.py --algorithm mopie --obj_name europa --n_epochs 100

For full usage options:
    python main.py --help
    python main.py --help-patterns
    python main.py --help-objects

Interactive mode:
    python main.py --interactive
"""

from prism.cli.entry_points import main


if __name__ == "__main__":
    main()

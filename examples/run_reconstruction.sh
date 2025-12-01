#!/bin/bash
# ==============================================================================
# PRISM Reconstruction Example Script
# ==============================================================================
# Based on: runs/gencrop_eur_nsamples_240_len_80
# This script runs a reconstruction from scratch (no checkpoint) using the
# most novel features available in this project.
#
# NOVEL FEATURES USED:
# --------------------
# 1. Circle Initialization (--initialization_target circle)
#    - Original/default initialization approach
#    - Uses a circular mask as the initialization target
#
# 2. Modern Pattern Function System (--pattern-fn builtin:fermat)
#    - Newer, more flexible pattern specification system
#    - Supports both builtin patterns and custom pattern files
#
# Parameters from original run (runs/gencrop_eur_nsamples_240_len_80):
#   - Object: europa
#   - n_samples: 240 (line sampling, ~204 effective)
#   - sample_length: 80 (line sampling mode)
#   - fermat spiral sampling pattern
#   - 25 max epochs per sample
#   - L1 loss with threshold 0.001
# ==============================================================================

set -e  # Exit on error

# Get the project root directory (parent of examples/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Change to project root for proper relative paths
cd "$PROJECT_ROOT"

# Run name with timestamp
RUN_NAME="novel_europa_$(date +%Y%m%d_%H%M%S)"

echo "======================================================================"
echo "PRISM Reconstruction Example"
echo "======================================================================"
echo "Run Name: $RUN_NAME"
echo ""
echo "Novel Features Enabled:"
echo "  ✓ Circle Initialization (original approach)"
echo "  ✓ Modern Pattern Function System"
echo "======================================================================"
echo ""

uv run python main.py \
    --name "$RUN_NAME" \
    --obj_name europa \
    --n_samples 240 \
    --image_size 1024 \
    --sample_diameter 17 \
    --sample_length 80 \
    --samples_per_line_meas 11 \
    --samples_per_line_rec 11 \
    --sample_sort center \
    --pattern-fn "builtin:fermat" \
    --n_epochs 1000 \
    --max_epochs 25 \
    --n_epochs_init 100 \
    --max_epochs_init 100 \
    --loss_type l1 \
    --lr 0.001 \
    --loss_th 0.001 \
    --new_weight 1 \
    --f_weight 0.0001 \
    --output_activation none \
    --middle_activation sigmoid \
    --initialization_target circle \
    --no-adaptive-convergence \
    --comment "Novel reconstruction with circle init, no adaptive convergence"

echo ""
echo "======================================================================"
echo "Run completed: $RUN_NAME"
echo "Results saved to: runs/$RUN_NAME/"
echo "======================================================================"

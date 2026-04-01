#!/bin/bash
set -e

VENV="/workspace/spar-team-recon/.venv/bin/python"
TRAIN_SCRIPT="/workspace/spar-team-recon/projects/ip/src/finetuning/open_model.py"
EVAL_SCRIPT="/workspace/spar-team-recon/projects/ip/src/evaluation/eval_pipeline.py"
CONFIG_DIR="/workspace/spar-team-recon/projects/ip/configs/exp_10"
RESULTS_DIR="/workspace/spar-team-recon/projects/ip/results/exp_10"

for arm in 7 8 9; do
    echo "=========================================="
    echo "  arm_${arm}: Starting training"
    echo "=========================================="
    mkdir -p "${RESULTS_DIR}/arm_${arm}/logs"
    $VENV $TRAIN_SCRIPT "${CONFIG_DIR}/arm_${arm}/training.json"

    echo "=========================================="
    echo "  arm_${arm}: Starting evaluation"
    echo "=========================================="
    $VENV $EVAL_SCRIPT "${CONFIG_DIR}/arm_${arm}/eval.json"

    echo "=========================================="
    echo "  arm_${arm}: Done"
    echo "=========================================="
done

echo "All arms (7, 8, 9) complete."

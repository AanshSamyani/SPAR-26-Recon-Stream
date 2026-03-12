#!/bin/bash
# Sequentially finetune all 16 arms (arm_0 to arm_15) for exp_6/XY.
# Each arm's training runs under its own nohup, with output saved to that arm's log dir.
# This entire script is intended to be run under nohup itself:
#   nohup bash run_exp6_XY_all_arms.sh > /workspace/spar-team-recon/projects/ip/results/exp_6/XY/nohup.out 2>&1 &

set -e

PYTHON="/workspace/spar-team-recon/.venv/bin/python"
TRAIN_SCRIPT="/workspace/spar-team-recon/projects/ip/src/finetuning/open_model.py"
CONFIG_BASE="/workspace/spar-team-recon/projects/ip/configs/exp_6/XY"
LOG_BASE="/workspace/spar-team-recon/projects/ip/results/exp_6/XY"

for i in $(seq 0 15); do
    ARM="arm_${i}"
    CONFIG="${CONFIG_BASE}/${ARM}/training.json"
    LOG_DIR="${LOG_BASE}/${ARM}/logs"

    mkdir -p "$LOG_DIR"

    echo "========================================"
    echo "[$(date)] Starting training for ${ARM}"
    echo "  Config: ${CONFIG}"
    echo "  Nohup log: ${LOG_DIR}/nohup.out"
    echo "========================================"

    # Run under nohup; use 'wait' to block until this arm finishes before starting the next
    nohup "$PYTHON" "$TRAIN_SCRIPT" "$CONFIG" > "${LOG_DIR}/nohup.out" 2>&1 &
    NOHUP_PID=$!
    echo "[$(date)] ${ARM} PID: ${NOHUP_PID}"
    wait $NOHUP_PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date)] ERROR: ${ARM} exited with code ${EXIT_CODE}. Continuing to next arm."
    else
        echo "[$(date)] ${ARM} completed successfully."
    fi
done

echo "========================================"
echo "[$(date)] All arms finished."
echo "========================================"

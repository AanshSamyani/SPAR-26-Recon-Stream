#!/bin/bash
# Sequentially evaluate all 16 arms (arm_0 to arm_15) for exp_6/YX.
# Each arm's eval runs under its own nohup, with output saved to that arm's log dir.
# This entire script is intended to be run under nohup itself:
#   nohup bash run_exp6_YX_all_arms_eval.sh > /workspace/spar-team-recon/projects/ip/results/exp_6/YX/eval_nohup.out 2>&1 &

set -e

PYTHON="/workspace/spar-team-recon/.venv/bin/python"
EVAL_SCRIPT="/workspace/spar-team-recon/projects/ip/src/evaluation/eval_pipeline.py"
CONFIG_BASE="/workspace/spar-team-recon/projects/ip/configs/exp_6/YX"
LOG_BASE="/workspace/spar-team-recon/projects/ip/results/exp_6/YX"

for i in $(seq 0 15); do
    ARM="arm_${i}"
    CONFIG="${CONFIG_BASE}/${ARM}/eval.json"
    LOG_DIR="${LOG_BASE}/${ARM}/logs"

    mkdir -p "$LOG_DIR"

    echo "========================================"
    echo "[$(date)] Starting eval for ${ARM}"
    echo "  Config: ${CONFIG}"
    echo "  Nohup log: ${LOG_DIR}/eval_nohup.out"
    echo "========================================"

    nohup "$PYTHON" "$EVAL_SCRIPT" "$CONFIG" > "${LOG_DIR}/eval_nohup.out" 2>&1 &
    NOHUP_PID=$!
    echo "[$(date)] ${ARM} PID: ${NOHUP_PID}"
    wait $NOHUP_PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date)] ERROR: ${ARM} eval exited with code ${EXIT_CODE}. Continuing to next arm."
    else
        echo "[$(date)] ${ARM} eval completed successfully."
    fi
done

echo "========================================"
echo "[$(date)] All arm evals finished."
echo "========================================"

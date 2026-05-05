#!/usr/bin/env bash
# Run every exp_12 arm sequentially.
#
# Order: off-policy arms first (~10-20 min each, fast feedback),
# then on-policy arms (~hours each).
#
# - Idempotent: skips any arm whose final_eval_summary.json already exists.
# - Each arm logs to projects/ip/results/exp_12/<arm>/logs/run.out.
# - Master timeline log at projects/ip/results/exp_12/master_run_all.log.
# - Continues past per-arm failures (warns in the master log).
#
# Usage on the pod:
#     cd /workspace/spar-team-recon/projects/ip/src
#     nohup bash run_exp12.sh \
#         > /workspace/spar-team-recon/projects/ip/results/exp_12/master_nohup.out 2>&1 &
#     echo $! > /workspace/spar-team-recon/projects/ip/results/exp_12/master.pid
#     tail -f /workspace/spar-team-recon/projects/ip/results/exp_12/master_run_all.log

set -uo pipefail   # NOT -e: we want to keep going past arm failures

REPO=/workspace/spar-team-recon
SRC=$REPO/projects/ip/src
CFG=$REPO/projects/ip/configs/exp_12
RES=$REPO/projects/ip/results/exp_12
DATA=$REPO/projects/ip/data/exp_12
MASTER_LOG=$RES/master_run_all.log

mkdir -p "$RES"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"
}

cd "$SRC"

# --- Pre-flight ----------------------------------------------------------
for f in "$DATA/train_prompts.jsonl" "$DATA/test_prompts.jsonl"; do
    if [ ! -s "$f" ]; then
        log "FATAL: required data file missing: $f"
        log "Run: python -m data_generation.exp_12_sycophancy"
        exit 1
    fi
done

if [ ! -s "$DATA/train_offpolicy_sft.jsonl" ]; then
    log "Generating $DATA/train_offpolicy_sft.jsonl ..."
    python -m data_generation.exp_12_offpolicy_sft 2>&1 | tee -a "$MASTER_LOG"
    if [ ! -s "$DATA/train_offpolicy_sft.jsonl" ]; then
        log "FATAL: off-policy SFT data file was not produced"
        exit 1
    fi
fi

# --- Arm runners ---------------------------------------------------------
run_on_policy() {
    local arm=$1
    local arm_dir=$CFG/$arm
    local res_dir=$RES/$arm
    local logf=$res_dir/logs/run.out
    mkdir -p "$res_dir/logs"
    log "=== START on-policy $arm  (log: $logf)"
    python expert_iteration.py "$arm_dir/ei.json" "$arm_dir/eval.json" \
        > "$logf" 2>&1
    local rc=$?
    log "=== END   on-policy $arm  (exit $rc)"
    return $rc
}

run_off_policy() {
    local arm=$1
    local arm_dir=$CFG/$arm
    local res_dir=$RES/$arm
    local logf=$res_dir/logs/run.out
    mkdir -p "$res_dir/logs"
    log "=== START off-policy $arm (log: $logf)"
    python offpolicy_sft.py "$arm_dir/sft.json" "$arm_dir/eval.json" \
        > "$logf" 2>&1
    local rc=$?
    log "=== END   off-policy $arm (exit $rc)"
    return $rc
}

# --- Run order: off-policy first (fast), then on-policy (slow) -----------
ORDER=(
    "off:arm_2"
    "off:arm_3"
    "off:arm_4"
    "off:arm_8"
    "off:arm_9"
    "on:arm_0"
    "on:arm_1"
    "on:arm_5"
    "on:arm_6"
    "on:arm_7"
)

log "Total arms queued: ${#ORDER[@]}"
log "Order: ${ORDER[*]}"

for entry in "${ORDER[@]}"; do
    kind=${entry%%:*}
    arm=${entry##*:}

    if [ -s "$RES/$arm/final_eval_summary.json" ]; then
        log "SKIP $arm (already has final_eval_summary.json)"
        continue
    fi

    case "$kind" in
        on)  run_on_policy  "$arm" || log "WARN: $arm failed; continuing" ;;
        off) run_off_policy "$arm" || log "WARN: $arm failed; continuing" ;;
        *)   log "WARN: unknown kind '$kind' for $arm; skipping" ;;
    esac
done

log "All arms processed."

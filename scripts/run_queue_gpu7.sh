#!/usr/bin/env bash
# Queue: D4 → E4 CC3M (prep + eval) → E5
# Runs on a specified GPU. Each step is run independently; a failure is
# logged but the queue continues to the next step.
#
# Usage: bash scripts/run_queue_gpu7.sh [GPU_ID] [WAIT_PID]
#   GPU_ID   defaults to 2
#   WAIT_PID if given, queue waits for this PID to finish before starting

set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-2}"
WAIT_PID="${2:-}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run_step() {
    local name="$1"; shift
    local log="$LOG_DIR/${name}_gpu${GPU}.log"
    echo "[queue] === $name === ($(date))"
    if CUDA_VISIBLE_DEVICES="$GPU" python -u "$@" 2>&1 | tee "$log"; then
        echo "[queue] $name DONE at $(date)"
    else
        echo "[queue] $name FAILED (exit $?) — see $log — continuing queue"
    fi
}

if [[ -n "$WAIT_PID" ]]; then
    echo "[queue] Waiting for PID $WAIT_PID to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    echo "[queue] PID $WAIT_PID done. Starting GPU $GPU queue at $(date)"
fi

run_step D4            experiments/run_D4.py  --config configs/D4.yaml
run_step prepare_cc3m  scripts/prepare_cc3m.py --max-images 200000 --timeout 5
run_step E4_cc3m       experiments/run_E4.py  --config configs/E4_cc3m.yaml
run_step E5            experiments/run_E5.py  --config configs/_rerun_fix_E5.yaml

echo "[queue] All steps complete at $(date)"

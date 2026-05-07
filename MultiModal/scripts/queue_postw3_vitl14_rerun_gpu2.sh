#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/next_run_suite/post_w3_sequence"
MARK_DIR="$MM_ROOT/results/next_run_suite/post_w3_sequence/markers"
mkdir -p "$LOG_DIR" "$MARK_DIR"

DONE_MARK="$MARK_DIR/gpu2.vitl14_stage44_rerun.done"
FAIL_MARK="$MARK_DIR/gpu2.vitl14_stage44_rerun.fail"
rm -f "$DONE_MARK" "$FAIL_MARK"

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

until [[ -f "$MARK_DIR/gpu0.vitl14_prebuild.done" ]]; do sleep 10; done

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

if python -m MultiModal.multimodal.experiments.run_stage44_zero_shot_baseline_control       --config MultiModal/configs/stage44_vitl14_rerun_gpu2.yaml 2>&1 | tee "$LOG_DIR/stage44_vitl14_rerun_gpu2.log"; then
  touch "$DONE_MARK"
  echo "[$(date '+%F %T')] queue_postw3_vitl14_rerun_gpu2 complete"
else
  touch "$FAIL_MARK"
  echo "[$(date '+%F %T')] queue_postw3_vitl14_rerun_gpu2 FAILED"
  exit 1
fi

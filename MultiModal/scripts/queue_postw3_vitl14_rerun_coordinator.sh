#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/next_run_suite/post_w3_sequence"
MARK_DIR="$MM_ROOT/results/next_run_suite/post_w3_sequence/markers"
mkdir -p "$LOG_DIR" "$MARK_DIR"

READY_MARK="$MARK_DIR/vitl14_stage44.ready.done"
FAIL_MARK="$MARK_DIR/vitl14_stage44.ready.fail"
rm -f "$READY_MARK" "$FAIL_MARK"

while true; do
  for g in 0 1 2 3 4 5 6 7; do
    if [[ -f "$MARK_DIR/gpu${g}.vitl14_stage44_rerun.fail" ]]; then
      touch "$FAIL_MARK"
      echo "[$(date '+%F %T')] vitl14 stage44 rerun FAILED at gpu${g}" | tee -a "$LOG_DIR/stage44_vitl14_rerun_coordinator.log"
      exit 1
    fi
  done

  ok=1
  for g in 0 1 2 3 4 5 6 7; do
    if [[ ! -f "$MARK_DIR/gpu${g}.vitl14_stage44_rerun.done" ]]; then
      ok=0
      break
    fi
  done

  if [[ "$ok" -eq 1 ]]; then
    # Legacy markers expected by old stage39 scripts.
    touch "$MARK_DIR/gpu1.vitl14_stage44.done"
    touch "$MARK_DIR/gpu2.vitl14_stage44.done"
    touch "$MARK_DIR/gpu3.vitl14_stage44.done"
    touch "$MARK_DIR/gpu4.vitl14_stage44.done"
    touch "$READY_MARK"
    echo "[$(date '+%F %T')] vitl14 stage44 rerun READY" | tee -a "$LOG_DIR/stage44_vitl14_rerun_coordinator.log"
    exit 0
  fi

  sleep 15
done

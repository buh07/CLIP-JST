#!/usr/bin/env bash
set -euo pipefail
ROOT='/jumbo/lisp/f004ndc/CLIP JST'
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/domain_gap_closure_suite"
MARK_DIR="$MM_ROOT/results/domain_gap_closure_suite/markers/shards"
mkdir -p "$LOG_DIR" "$MARK_DIR"
export PYTHONUNBUFFERED=1
cd "$ROOT"
source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
while true; do
  if [[ -f "$MARK_DIR/stage31_wavcaps_cache.failed.json" ]]; then
    echo "stage31 cache build failed; see $MARK_DIR/stage31_wavcaps_cache.failed.json"
    exit 1
  fi
  if [[ -f "$MARK_DIR/gpu4.stage31.done" && -f "$MARK_DIR/gpu5.stage31.done" && -f "$MARK_DIR/gpu6.stage31.done" && -f "$MARK_DIR/gpu7.stage31.done" ]]; then
    break
  fi
  sleep 60
done
python -m MultiModal.multimodal.experiments.run_stage33_domain_gap_aggregate --config MultiModal/configs/stage33_domain_gap_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage33_refresh_after_stage31_all4.log"

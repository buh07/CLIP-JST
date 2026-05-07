#!/usr/bin/env bash
set -euo pipefail
ROOT='/jumbo/lisp/f004ndc/CLIP JST'
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/domain_gap_closure_suite"
MARK_DIR="$MM_ROOT/results/domain_gap_closure_suite/markers/shards"
mkdir -p "$LOG_DIR" "$MARK_DIR"
export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1
cd "$ROOT"
source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
while [[ ! -f "$MARK_DIR/stage31_wavcaps_cache.done" ]]; do sleep 60; done
python -m MultiModal.multimodal.experiments.run_stage31_wavcaps_scaling --config MultiModal/configs/stage31_domain_gap_gpu6.yaml 2>&1 | tee "$LOG_DIR/stage31_rerun_gpu6.log"
touch "$MARK_DIR/gpu6.stage31.done"

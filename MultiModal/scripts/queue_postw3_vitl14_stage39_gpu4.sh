#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/next_run_suite/post_w3_sequence"
MARK_DIR="$MM_ROOT/results/next_run_suite/post_w3_sequence/markers"
mkdir -p "$LOG_DIR" "$MARK_DIR"

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

until [[ -f "$MARK_DIR/vitl14_stage44.ready.done" || -f "$MARK_DIR/vitl14_stage44.ready.fail" ]]; do
  sleep 20
done
if [[ -f "$MARK_DIR/vitl14_stage44.ready.fail" ]]; then
  echo "[$(date '+%F %T')] stage39 gpu4 aborting: vitl14 stage44 failed"
  exit 1
fi

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage39_modality_gap_linear_vs_jl \
  --config MultiModal/configs/stage39_vitl14_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage39_vitl14_gpu4.log"

touch "$MARK_DIR/gpu4.vitl14_stage39.done"
echo "[$(date '+%F %T')] queue_postw3_vitl14_stage39_gpu4 complete"

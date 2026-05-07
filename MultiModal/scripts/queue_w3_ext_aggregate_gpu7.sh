#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/next_run_suite/w3_holdout_ext"
LOG_DIR="$MM_ROOT/logs/next_run_suite/w3_holdout_ext"
MARK="$OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

while [[ ! -f "$MARK/gpu0.stage55.done" || ! -f "$MARK/gpu1.stage55.done" || ! -f "$MARK/gpu2.stage55.done" || ! -f "$MARK/gpu3.stage55.done" || ! -f "$MARK/gpu4.stage55.done" || ! -f "$MARK/gpu5.stage55.done" || ! -f "$MARK/gpu6.stage55.done" || ! -f "$MARK/gpu7.stage55.done" ]]; do
  sleep 30
done

python -m MultiModal.multimodal.experiments.run_stage56_wavcaps_holdout_aggregate \
  --config MultiModal/configs/stage56_w3_ext_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage56_w3_ext_aggregate.log"

touch "$MARK/gpu7.stage56.done"
echo "[$(date '+%F %T')] queue_w3_ext_aggregate_gpu7 complete"

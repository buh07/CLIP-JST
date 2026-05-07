#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/neurips_strengthen_suite"
LOG_DIR="$MM_ROOT/logs/neurips_cleanup"
MARK="$OUT/markers/cleanup"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m py_compile \
  MultiModal/multimodal/experiments/run_stage40_strengthen_suite_aggregate.py

until [[ -f "$MARK/gpu0.stage38.cleanup.done" && -f "$MARK/gpu1.stage37relabel.cleanup.done" ]]; do
  echo "[$(date '+%F %T')] waiting for stage38 + stage37 relabel..."
  sleep 30
done

rm -f "$OUT/markers/stage40_strengthen_suite_aggregate.done.json"
python -m MultiModal.multimodal.experiments.run_stage40_strengthen_suite_aggregate \
  --config MultiModal/configs/stage40_strengthen_suite_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage40_gpu2_cleanup.log"

touch "$MARK/gpu2.stage40.cleanup.done"
echo "[$(date '+%F %T')] queue_cleanup_gpu2 complete"

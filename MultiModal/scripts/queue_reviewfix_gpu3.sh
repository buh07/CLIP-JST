#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/reviewer_fixes_suite"
LOG_DIR="$MM_ROOT/logs/reviewer_fixes"
MARK="$OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m py_compile \
  MultiModal/multimodal/experiments/run_stage41_semantic_metadata_topk.py \
  MultiModal/multimodal/experiments/run_stage42_semantic_metadata_aggregate.py \
  MultiModal/multimodal/experiments/run_stage43_bottleneck_oos_validation.py

rm -f "$OUT/split/gpu3/markers/stage41_semantic_metadata_topk.done.json"
python -m MultiModal.multimodal.experiments.run_stage41_semantic_metadata_topk \
  --config MultiModal/configs/stage41_semantic_metadata_gpu3.yaml 2>&1 | tee "$LOG_DIR/stage41_gpu3.log"

touch "$MARK/gpu3.stage41.done"

until [[ -f "$MARK/gpu0.stage41.done" && -f "$MARK/gpu1.stage41.done" && -f "$MARK/gpu2.stage41.done" && -f "$MARK/gpu3.stage41.done" ]]; do
  echo "[$(date '+%F %T')] waiting for all stage41 shards..."
  sleep 30
done

rm -f "$OUT/aggregate/markers/stage42_semantic_metadata_aggregate.done.json"
python -m MultiModal.multimodal.experiments.run_stage42_semantic_metadata_aggregate \
  --config MultiModal/configs/stage42_semantic_metadata_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage42_aggregate_gpu3.log"

touch "$MARK/stage42.aggregate.done"

rm -f "$OUT/aggregate/markers/stage43_bottleneck_oos_validation.done.json"
python -m MultiModal.multimodal.experiments.run_stage43_bottleneck_oos_validation \
  --config MultiModal/configs/stage43_bottleneck_oos_validation.yaml 2>&1 | tee "$LOG_DIR/stage43_oos_gpu3.log"

touch "$MARK/stage43.oos.done"
echo "[$(date '+%F %T')] queue_reviewfix_gpu3 complete"

#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/semantic_followup"
LOG_DIR="$MM_ROOT/logs/semantic_followup"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python -m MultiModal.multimodal.experiments.run_stage34_semantic_topk \
  --config MultiModal/configs/stage34_semantic_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage34_gpu7.log"
touch "$MARK/gpu7.stage34.done"

for g in 0 1 2 3 4 5 6 7; do
  while [[ ! -f "$MARK/gpu${g}.stage34.done" ]]; do
    sleep 20
  done
done

python -m MultiModal.multimodal.experiments.run_stage35_semantic_topk_aggregate \
  --config MultiModal/configs/stage35_semantic_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage35_aggregate.log"
touch "$MARK/stage35.aggregate.done"

echo "[$(date '+%F %T')] queue_semantic_gpu7 complete"

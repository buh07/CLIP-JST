#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/semantic_jlablation"
LOG_DIR="$MM_ROOT/logs/semantic_jlablation"
MARK="$OUT/markers/shards"
FULL_MARK="$MM_ROOT/results/semantic_full_aggregate/markers/shards"
mkdir -p "$LOG_DIR" "$MARK" "$FULL_MARK"
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# Stage 34: semantic topk for JL ablation gpu7 shard (m=512)
python -m MultiModal.multimodal.experiments.run_stage34_semantic_topk \
  --config MultiModal/configs/stage34_jlablation_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage34_jlablation_gpu7.log"
touch "$MARK/gpu7.stage34.done"

# Wait for all 4 JL ablation shards before running aggregation
echo "[$(date '+%F %T')] Waiting for jlablation shards gpu4..7..."
for g in 4 5 6 7; do
  while [[ ! -f "$MARK/gpu${g}.stage34.done" ]]; do
    sleep 20
  done
  echo "[$(date '+%F %T')] shard gpu${g} done"
done

# Stage 35 full aggregate (combines existing semantic_followup + new jlablation shards)
python -m MultiModal.multimodal.experiments.run_stage35_semantic_topk_aggregate \
  --config MultiModal/configs/stage35_semantic_full_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage35_full_aggregate.log"
touch "$FULL_MARK/stage35_full_aggregate.done"
echo "[$(date '+%F %T')] Stage 35 full aggregate complete"

# Stage 36 bottleneck decomposition (runs on CPU, reads all stage result JSONs)
python -m MultiModal.multimodal.experiments.run_stage36_bottleneck_decomposition \
  --config MultiModal/configs/stage36_bottleneck_decomposition.yaml 2>&1 | tee "$LOG_DIR/stage36_bottleneck_decomposition.log"
touch "$FULL_MARK/stage36_bottleneck_decomposition.done"
echo "[$(date '+%F %T')] Stage 36 bottleneck decomposition complete"

echo "[$(date '+%F %T')] queue_semantic_jlablation_gpu7 ALL DONE"

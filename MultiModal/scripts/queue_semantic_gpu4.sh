#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/semantic_followup"
LOG_DIR="$MM_ROOT/logs/semantic_followup"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"
export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python -m MultiModal.multimodal.experiments.run_stage34_semantic_topk \
  --config MultiModal/configs/stage34_semantic_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage34_gpu4.log"
touch "$MARK/gpu4.stage34.done"
echo "[$(date '+%F %T')] queue_semantic_gpu4 complete"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/sdpi_aggressive_suite"
LOG_DIR="$MM_ROOT/logs/sdpi_aggressive_suite"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

while [[ ! -f "$OUT/markers/stage47_sdpi_manifest.done.json" ]]; do
  sleep 15
done

python -m MultiModal.multimodal.experiments.run_stage48_sdpi_embed_export \
  --config MultiModal/configs/stage48_sdpi_gpu6.yaml 2>&1 | tee "$LOG_DIR/stage48_gpu6.log"
touch "$MARK/gpu6.stage48.done"

while [[ ! -f "$MARK/gpu4.stage48.done" || ! -f "$MARK/gpu5.stage48.done" || ! -f "$MARK/gpu6.stage48.done" || ! -f "$MARK/gpu7.stage48.done" ]]; do
  sleep 20
done

python -m MultiModal.multimodal.experiments.run_stage49_sdpi_mi_diagnostics \
  --config MultiModal/configs/stage49_sdpi_gpu6.yaml 2>&1 | tee "$LOG_DIR/stage49_gpu6.log"
touch "$MARK/gpu6.stage49.done"

echo "[$(date '+%F %T')] queue_sdpi_gpu6 complete"

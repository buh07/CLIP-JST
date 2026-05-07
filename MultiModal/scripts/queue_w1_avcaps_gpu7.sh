#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/reinforce_suite/w1_second_triple_avcaps"
LOG_DIR="$MM_ROOT/logs/reinforce_suite/w1_second_triple_avcaps"
MARK="$OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

while [[ ! -f "$MARK/avcaps_cache_ready.done" ]]; do sleep 30; done

python -m MultiModal.multimodal.experiments.run_stage57_second_triple_avcaps \
  --config MultiModal/configs/stage57_w1_avcaps_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage57_gpu7.log"

touch "$MARK/gpu7.stage57.done"

while [[ ! -f "$MARK/gpu4.stage57.done" || ! -f "$MARK/gpu5.stage57.done" || ! -f "$MARK/gpu6.stage57.done" || ! -f "$MARK/gpu7.stage57.done" ]]; do
  sleep 30
done

python -m MultiModal.multimodal.experiments.run_stage58_second_triple_aggregate \
  --config MultiModal/configs/stage58_w1_avcaps_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage58_gpu7.log"

touch "$MARK/gpu7.stage58.done"
echo "[$(date '+%F %T')] queue_w1_avcaps_gpu7 complete"

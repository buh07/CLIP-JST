#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/next_run_suite/w1_avcaps_full"
LOG_DIR="$MM_ROOT/logs/next_run_suite/w1_avcaps_full"
MARK="$OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

while [[ ! -f "$MARK/gpu1.stage57_headline.done" || ! -f "$MARK/gpu2.stage57_headline.done" || ! -f "$MARK/gpu3.stage57_headline.done" || ! -f "$MARK/gpu4.stage57_headline.done" || ! -f "$MARK/gpu5.stage57_headline.done" || ! -f "$MARK/gpu6.stage57_headline.done" || ! -f "$MARK/gpu7.stage57_headline.done" ]]; do
  sleep 30
done

python -m MultiModal.multimodal.experiments.run_stage57_second_triple_avcaps   --config MultiModal/configs/stage57_w1_lora_gpu6.yaml 2>&1 | tee "$LOG_DIR/stage57_lora_gpu6.log"

touch "$MARK/gpu6.stage57_lora.done"
echo "[$(date '+%F %T')] queue_w1_lora_gpu6 complete"

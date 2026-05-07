#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/reinforce_suite"
MARK="$MM_ROOT/results/reinforce_suite/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_w8_1k_split_eval \
  --config MultiModal/configs/w8_reinforce_gpu5.yaml 2>&1 | tee "$LOG_DIR/w8_1k_split_eval.gpu5.log"

touch "$MARK/gpu5.w8.done"
echo "[$(date '+%F %T')] queue_reinforce_gpu5 complete"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/reinforce_suite"
MARK="$MM_ROOT/results/reinforce_suite/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# Fast validation pass for W8 schema + deterministic split behavior
python -m MultiModal.multimodal.experiments.run_w8_1k_split_eval \
  --config MultiModal/configs/w8_reinforce_gpu7_smoke.yaml 2>&1 | tee "$LOG_DIR/w8_1k_split_eval_smoke.gpu7.log"

touch "$MARK/gpu7.w8smoke.done"
echo "[$(date '+%F %T')] queue_reinforce_gpu7 complete"

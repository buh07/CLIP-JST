#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/strengthen_suite"
HELPER_DIR="$MM_ROOT/results/strengthen_suite/stage2_strength_split/helper_markers"
mkdir -p "$LOG_DIR" "$HELPER_DIR"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

S2_SELF_DONE="$MM_ROOT/results/strengthen_suite/stage2_dimtax_split/gpu7/markers/stage2_e7_karpathy.done.json"
while [[ ! -f "$S2_SELF_DONE" ]]; do
  echo "[$(date '+%F %T')] helper-gpu7 waiting for its current stage2 task to finish..."
  sleep 60
done

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage2_e7_karpathy \
  --config MultiModal/configs/stage2_e7_strength_gpu4_helper_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage2_strength_gpu4_helper_gpu7.log"

echo "[$(date '+%F %T')] helper-gpu7 complete" | tee "$HELPER_DIR/gpu7.done.txt"

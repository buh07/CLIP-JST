#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT_ROOT="$MM_ROOT/results/concern_suite"
LOG_DIR="$MM_ROOT/logs/concern_suite"
MARKERS="$OUT_ROOT/markers"
mkdir -p "$LOG_DIR" "$MARKERS"

export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

echo "[$(date '+%F %T')] WAIT stage8_e8d checkpoint run to finish"
while [[ ! -f "$MARKERS/stage8_e8d.done.json" ]]; do
  sleep 30
done
echo "[$(date '+%F %T')] FOUND stage8 marker"

log="$LOG_DIR/stage11_mia_lira_gpu6.log"
echo "[$(date '+%F %T')] START stage11_mia_lira"
python -m MultiModal.multimodal.experiments.run_stage11_mia_lira \
  --config MultiModal/configs/stage11_mia_lira.yaml 2>&1 | tee "$log"
echo "[$(date '+%F %T')] DONE stage11_mia_lira"

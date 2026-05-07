#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/modular_transitivity_suite"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

M19="$MM_ROOT/results/modular_transitivity_suite/stage19_merged/markers/stage19_merge_gate.done.json"
while [[ ! -f "$M19" ]]; do
  echo "[$(date '+%F %T')] gpu6 waiting for stage19 merged gate..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage20_modular_audio_transitivity \
  --config MultiModal/configs/stage20_modtrans_gpu6.yaml 2>&1 | tee "$LOG_DIR/stage20_gpu6.log"

echo "[$(date '+%F %T')] queue_modtrans_gpu6 complete"

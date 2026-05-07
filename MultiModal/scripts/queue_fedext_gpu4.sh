#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/federated_extension"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage13_federated_comparison \
  --config MultiModal/configs/stage13_federated_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage13_gpu4.log"

M="$MM_ROOT/results/federated_extension/markers/stage13_federated.done.json"
while [[ ! -f "$M" ]]; do
  echo "[$(date '+%F %T')] gpu4 waiting for merged stage13 marker..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage14_stronger_attacks \
  --config MultiModal/configs/stage14_stronger_attacks_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage14_gpu4.log"
python - <<'PY'
from pathlib import Path
from MultiModal.multimodal.common import mark_done
mark_done(Path('/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results/federated_extension/markers/stage14_gpu4.done.json'))
PY

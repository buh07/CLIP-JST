#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/federated_extension"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# Stage13 shard on GPU7
python -m MultiModal.multimodal.experiments.run_stage13_federated_comparison \
  --config MultiModal/configs/stage13_federated_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage13_gpu7.log"

M4="$MM_ROOT/results/federated_extension_split/gpu4/markers/stage13_federated.done.json"
M5="$MM_ROOT/results/federated_extension_split/gpu5/markers/stage13_federated.done.json"
M6="$MM_ROOT/results/federated_extension_split/gpu6/markers/stage13_federated.done.json"
M7="$MM_ROOT/results/federated_extension_split/gpu7/markers/stage13_federated.done.json"

while [[ ! -f "$M4" || ! -f "$M5" || ! -f "$M6" || ! -f "$M7" ]]; do
  echo "[$(date '+%F %T')] waiting for all stage13 shard markers..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage13_merge \
  --config MultiModal/configs/stage13_merge.yaml 2>&1 | tee "$LOG_DIR/stage13_merge.log"

# Stage14 shard on GPU7 (dir_a0p1)
python -m MultiModal.multimodal.experiments.run_stage14_stronger_attacks \
  --config MultiModal/configs/stage14_stronger_attacks_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage14_gpu7.log"
python - <<'PY'
from pathlib import Path
from MultiModal.multimodal.common import mark_done
mark_done(Path('/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results/federated_extension/markers/stage14_gpu7.done.json'))
PY

# Wait for other stage14 shards, then aggregate
S14_4="$MM_ROOT/results/federated_extension/markers/stage14_gpu4.done.json"
S14_5="$MM_ROOT/results/federated_extension/markers/stage14_gpu5.done.json"
S14_6="$MM_ROOT/results/federated_extension/markers/stage14_gpu6.done.json"
S14_7="$MM_ROOT/results/federated_extension/markers/stage14_gpu7.done.json"
while [[ ! -f "$S14_4" || ! -f "$S14_5" || ! -f "$S14_6" || ! -f "$S14_7" ]]; do
  echo "[$(date '+%F %T')] waiting for all stage14 shard markers..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage15_federated_aggregate \
  --config MultiModal/configs/stage15_federated_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage15_gpu7.log"

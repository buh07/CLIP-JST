#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/strengthen_suite"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage2_e7_karpathy \
  --config MultiModal/configs/stage2_e7_dimtax_gpu7_flickr.yaml 2>&1 | tee "$LOG_DIR/stage2_dimtax_gpu7.log"

# Wait for all stage2 shards, then merge both stage2 tracks.
S2A4="$MM_ROOT/results/strengthen_suite/stage2_strength_split/gpu4/markers/stage2_e7_karpathy.done.json"
S2A5="$MM_ROOT/results/strengthen_suite/stage2_strength_split/gpu5/markers/stage2_e7_karpathy.done.json"
S2B6="$MM_ROOT/results/strengthen_suite/stage2_dimtax_split/gpu6/markers/stage2_e7_karpathy.done.json"
S2B7="$MM_ROOT/results/strengthen_suite/stage2_dimtax_split/gpu7/markers/stage2_e7_karpathy.done.json"
while [[ ! -f "$S2A4" || ! -f "$S2A5" || ! -f "$S2B6" || ! -f "$S2B7" ]]; do
  echo "[$(date '+%F %T')] waiting for stage2 shards..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage2_merge \
  --config MultiModal/configs/stage2_e7_strength_merge.yaml 2>&1 | tee "$LOG_DIR/stage2_strength_merge.log"
python -m MultiModal.multimodal.experiments.run_stage2_merge \
  --config MultiModal/configs/stage2_e7_dimtax_merge.yaml 2>&1 | tee "$LOG_DIR/stage2_dimtax_merge.log"

python -m MultiModal.multimodal.experiments.run_stage13_federated_comparison \
  --config MultiModal/configs/stage13_fedfix_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage13_fedfix_gpu7.log"

M13_4="$MM_ROOT/results/strengthen_suite/stage13_fedfix_split/gpu4/markers/stage13_federated.done.json"
M13_5="$MM_ROOT/results/strengthen_suite/stage13_fedfix_split/gpu5/markers/stage13_federated.done.json"
M13_6="$MM_ROOT/results/strengthen_suite/stage13_fedfix_split/gpu6/markers/stage13_federated.done.json"
M13_7="$MM_ROOT/results/strengthen_suite/stage13_fedfix_split/gpu7/markers/stage13_federated.done.json"
while [[ ! -f "$M13_4" || ! -f "$M13_5" || ! -f "$M13_6" || ! -f "$M13_7" ]]; do
  echo "[$(date '+%F %T')] waiting for stage13 shards..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage13_merge \
  --config MultiModal/configs/stage13_fedfix_merge.yaml 2>&1 | tee "$LOG_DIR/stage13_fedfix_merge.log"

python -m MultiModal.multimodal.experiments.run_stage14_stronger_attacks \
  --config MultiModal/configs/stage14_fedfix_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage14_fedfix_gpu7.log"

M14_4="$MM_ROOT/results/strengthen_suite/stage14_fedfix_split/gpu4/markers/stage14_stronger_attacks.done.json"
M14_5="$MM_ROOT/results/strengthen_suite/stage14_fedfix_split/gpu5/markers/stage14_stronger_attacks.done.json"
M14_6="$MM_ROOT/results/strengthen_suite/stage14_fedfix_split/gpu6/markers/stage14_stronger_attacks.done.json"
M14_7="$MM_ROOT/results/strengthen_suite/stage14_fedfix_split/gpu7/markers/stage14_stronger_attacks.done.json"
while [[ ! -f "$M14_4" || ! -f "$M14_5" || ! -f "$M14_6" || ! -f "$M14_7" ]]; do
  echo "[$(date '+%F %T')] waiting for stage14 shards..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage14_merge \
  --config MultiModal/configs/stage14_fedfix_merge.yaml 2>&1 | tee "$LOG_DIR/stage14_fedfix_merge.log"

python -m MultiModal.multimodal.experiments.run_stage15_federated_aggregate \
  --config MultiModal/configs/stage15_fedfix_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage15_fedfix_aggregate.log"

echo "[$(date '+%F %T')] queue_strength_gpu7 complete"

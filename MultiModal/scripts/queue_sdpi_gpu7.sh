#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/sdpi_aggressive_suite"
LOG_DIR="$MM_ROOT/logs/sdpi_aggressive_suite"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m py_compile \
  MultiModal/multimodal/experiments/sdpi_common.py \
  MultiModal/multimodal/experiments/run_stage47_sdpi_manifest.py \
  MultiModal/multimodal/experiments/run_stage48_sdpi_embed_export.py \
  MultiModal/multimodal/experiments/run_stage49_sdpi_mi_diagnostics.py \
  MultiModal/multimodal/experiments/run_stage50_sdpi_inequality.py \
  MultiModal/multimodal/experiments/run_stage51_sdpi_bridge_link.py \
  MultiModal/multimodal/experiments/run_stage52_sdpi_interventions.py \
  MultiModal/multimodal/experiments/run_stage53_sdpi_constrained_channel.py \
  MultiModal/multimodal/experiments/run_stage54_sdpi_aggressive_aggregate.py

python -m MultiModal.multimodal.experiments.run_stage47_sdpi_manifest \
  --config MultiModal/configs/stage47_sdpi_manifest.yaml 2>&1 | tee "$LOG_DIR/stage47_manifest_gpu7.log"

# smoke: one condition through stage49
if [[ "${RUN_SDPI_SMOKE:-0}" == "1" ]]; then
  python -m MultiModal.multimodal.experiments.run_stage48_sdpi_embed_export \
    --config MultiModal/configs/stage48_sdpi_smoke.yaml 2>&1 | tee "$LOG_DIR/stage48_smoke_gpu7.log"
  python -m MultiModal.multimodal.experiments.run_stage49_sdpi_mi_diagnostics \
    --config MultiModal/configs/stage49_sdpi_smoke.yaml 2>&1 | tee "$LOG_DIR/stage49_smoke_gpu7.log"
fi

python -m MultiModal.multimodal.experiments.run_stage48_sdpi_embed_export \
  --config MultiModal/configs/stage48_sdpi_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage48_gpu7.log"
touch "$MARK/gpu7.stage48.done"

while [[ ! -f "$MARK/gpu4.stage48.done" || ! -f "$MARK/gpu5.stage48.done" || ! -f "$MARK/gpu6.stage48.done" || ! -f "$MARK/gpu7.stage48.done" ]]; do
  sleep 20
done

python -m MultiModal.multimodal.experiments.run_stage49_sdpi_mi_diagnostics \
  --config MultiModal/configs/stage49_sdpi_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage49_gpu7.log"
touch "$MARK/gpu7.stage49.done"

while [[ ! -f "$MARK/gpu4.stage49.done" || ! -f "$MARK/gpu5.stage49.done" || ! -f "$MARK/gpu6.stage49.done" || ! -f "$MARK/gpu7.stage49.done" ]]; do
  sleep 30
done

python -m MultiModal.multimodal.experiments.run_stage50_sdpi_inequality \
  --config MultiModal/configs/stage50_sdpi_inequality.yaml 2>&1 | tee "$LOG_DIR/stage50_gpu7.log"
python -m MultiModal.multimodal.experiments.run_stage51_sdpi_bridge_link \
  --config MultiModal/configs/stage51_sdpi_bridge_link.yaml 2>&1 | tee "$LOG_DIR/stage51_gpu7.log"
python -m MultiModal.multimodal.experiments.run_stage52_sdpi_interventions \
  --config MultiModal/configs/stage52_sdpi_interventions.yaml 2>&1 | tee "$LOG_DIR/stage52_gpu7.log"
python -m MultiModal.multimodal.experiments.run_stage53_sdpi_constrained_channel \
  --config MultiModal/configs/stage53_sdpi_constrained_channel.yaml 2>&1 | tee "$LOG_DIR/stage53_gpu7.log"
python -m MultiModal.multimodal.experiments.run_stage54_sdpi_aggressive_aggregate \
  --config MultiModal/configs/stage54_sdpi_aggressive_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage54_gpu7.log"

touch "$MARK/gpu7.stage54.done"
echo "[$(date '+%F %T')] queue_sdpi_gpu7 complete"

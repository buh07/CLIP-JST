#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/modular_transitivity_followup"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage22_modtrans_followup \
  --config MultiModal/configs/stage22_modfollow_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage22_gpu4.log"

FCACHE="$MM_ROOT/results/modular_transitivity_followup/caches/audiocaps_av_clap_htsat_fused"
while [[ ! -f "$FCACHE/image_feats_clip_raw.pt" || ! -f "$FCACHE/audio_feats_clap_raw.pt" || ! -f "$FCACHE/text_feats_clip_raw.pt" || ! -f "$FCACHE/metadata.json" ]]; do
  echo "[$(date '+%F %T')] gpu4 waiting for fused AV cache..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage23_modtrans_audio_swap \
  --config MultiModal/configs/stage23_modfollow_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage23_gpu4.log"

echo "[$(date '+%F %T')] queue_modfollow_gpu4 complete"

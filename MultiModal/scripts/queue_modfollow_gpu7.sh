#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/modular_transitivity_followup"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

FCACHE="$MM_ROOT/results/modular_transitivity_followup/caches/audiocaps_av_clap_htsat_fused"
while [[ ! -f "$FCACHE/image_feats_clip_raw.pt" || ! -f "$FCACHE/audio_feats_clap_raw.pt" || ! -f "$FCACHE/text_feats_clip_raw.pt" || ! -f "$FCACHE/metadata.json" ]]; do
  echo "[$(date '+%F %T')] gpu7 waiting for fused AV cache..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage23_modtrans_audio_swap \
  --config MultiModal/configs/stage23_modfollow_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage23_gpu7.log"

S22_4="$MM_ROOT/results/modular_transitivity_followup/stage22/split/gpu4/markers/stage22_modtrans_followup.done.json"
S22_5="$MM_ROOT/results/modular_transitivity_followup/stage22/split/gpu5/markers/stage22_modtrans_followup.done.json"
S23_4="$MM_ROOT/results/modular_transitivity_followup/stage23/split/gpu4/markers/stage23_modtrans_audio_swap.done.json"
S23_5="$MM_ROOT/results/modular_transitivity_followup/stage23/split/gpu5/markers/stage23_modtrans_audio_swap.done.json"
S23_6="$MM_ROOT/results/modular_transitivity_followup/stage23/split/gpu6/markers/stage23_modtrans_audio_swap.done.json"
S23_7="$MM_ROOT/results/modular_transitivity_followup/stage23/split/gpu7/markers/stage23_modtrans_audio_swap.done.json"

while [[ ! -f "$S22_4" || ! -f "$S22_5" || ! -f "$S23_4" || ! -f "$S23_5" || ! -f "$S23_6" || ! -f "$S23_7" ]]; do
  echo "[$(date '+%F %T')] gpu7 waiting for stage22/stage23 shard markers..."
  sleep 120
done

python -m MultiModal.multimodal.experiments.run_stage24_modtrans_followup_aggregate \
  --config MultiModal/configs/stage24_modfollow_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage24_aggregate.log"

echo "[$(date '+%F %T')] queue_modfollow_gpu7 complete"

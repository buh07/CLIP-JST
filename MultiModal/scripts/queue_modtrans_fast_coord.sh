#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/modular_transitivity_suite"
mkdir -p "$LOG_DIR"

source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

S19_6="$MM_ROOT/results/modular_transitivity_suite/split/gpu6/markers/stage19_pseudomodality_transitivity.done.json"
S19_7="$MM_ROOT/results/modular_transitivity_suite/split/gpu7/markers/stage19_pseudomodality_transitivity.done.json"
while [[ ! -f "$S19_6" || ! -f "$S19_7" ]]; do
  echo "[$(date '+%F %T')] coord waiting for stage19 gpu6+gpu7 markers..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage19_merge_gate \
  --config MultiModal/configs/stage19_modtrans_merge_fast.yaml 2>&1 | tee "$LOG_DIR/stage19_merge_fast.log"

CACHE_ROOT="$MM_ROOT/results/modular_transitivity_suite/caches/audiocaps_av"
while [[ ! -f "$CACHE_ROOT/image_feats_clip_raw.pt" || ! -f "$CACHE_ROOT/audio_feats_clap_raw.pt" || ! -f "$CACHE_ROOT/text_feats_clip_raw.pt" || ! -f "$CACHE_ROOT/metadata.json" ]]; do
  echo "[$(date '+%F %T')] coord waiting for AV cache artifacts..."
  sleep 60
done

if ! tmux has-session -t mm_modtrans_s20_g6 2>/dev/null; then
  tmux new-session -d -s mm_modtrans_s20_g6 "cd '$ROOT' && source '$MM_ROOT/.venv/bin/activate' && export PYTHONPATH='$MM_ROOT:\${PYTHONPATH:-}' && export CUDA_VISIBLE_DEVICES=6 && export PYTHONUNBUFFERED=1 && python -m MultiModal.multimodal.experiments.run_stage20_modular_audio_transitivity --config MultiModal/configs/stage20_modtrans_gpu6.yaml 2>&1 | tee '$LOG_DIR/stage20_gpu6_fast.log'"
fi
if ! tmux has-session -t mm_modtrans_s20_g7 2>/dev/null; then
  tmux new-session -d -s mm_modtrans_s20_g7 "cd '$ROOT' && source '$MM_ROOT/.venv/bin/activate' && export PYTHONPATH='$MM_ROOT:\${PYTHONPATH:-}' && export CUDA_VISIBLE_DEVICES=7 && export PYTHONUNBUFFERED=1 && python -m MultiModal.multimodal.experiments.run_stage20_modular_audio_transitivity --config MultiModal/configs/stage20_modtrans_gpu7.yaml 2>&1 | tee '$LOG_DIR/stage20_gpu7_fast.log'"
fi

S20_6="$MM_ROOT/results/modular_transitivity_suite/split/gpu6/markers/stage20_modular_audio_transitivity.done.json"
S20_7="$MM_ROOT/results/modular_transitivity_suite/split/gpu7/markers/stage20_modular_audio_transitivity.done.json"
while [[ ! -f "$S20_6" || ! -f "$S20_7" ]]; do
  echo "[$(date '+%F %T')] coord waiting for stage20 gpu6+gpu7 markers..."
  sleep 120
done

python -m MultiModal.multimodal.experiments.run_stage21_modular_transitivity_aggregate \
  --config MultiModal/configs/stage21_modtrans_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage21_aggregate_fast.log"

echo "[$(date '+%F %T')] queue_modtrans_fast_coord complete"

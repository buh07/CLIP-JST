#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/modular_transitivity_followup"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python - << 'PY' 2>&1 | tee "$LOG_DIR/cache_fused_gpu6.log"
from pathlib import Path
from MultiModal.multimodal.data import extract_audiocaps_av_cache
extract_audiocaps_av_cache(
    out_dir=Path('/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results/modular_transitivity_followup/caches/audiocaps_av_clap_htsat_fused'),
    dataset_name='JackyHoCL/AudioCaps-mp3',
    clap_model_name='laion/clap-htsat-fused',
    clip_backbone_name='openai/clip-vit-base-patch32',
    device='cuda',
    audio_batch_size=64,
    image_batch_size=128,
    text_batch_size=256,
    target_sampling_rate=48000,
    max_examples_per_split=None,
    thumbnail_timeout_sec=10.0,
    thumbnail_retries=2,
    thumbnail_backoff_sec=1.0,
    reuse_image_text_from_dir=Path('/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results/modular_transitivity_followup/caches/audiocaps_av_clap_htsat_unfused'),
)
print('fused AV cache ready')
PY

python -m MultiModal.multimodal.experiments.run_stage23_modtrans_audio_swap \
  --config MultiModal/configs/stage23_modfollow_gpu6.yaml 2>&1 | tee "$LOG_DIR/stage23_gpu6.log"

echo "[$(date '+%F %T')] queue_modfollow_gpu6 complete"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/domain_gap_closure_suite"
LOG_DIR="$MM_ROOT/logs/domain_gap_closure_suite"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# Start wavcaps cache early.
python - <<'PY' 2>&1 | tee "$LOG_DIR/wavcaps_cache_warmup_gpu7.log"
from pathlib import Path
from MultiModal.multimodal.data import extract_wavcaps_audio_text_cache
try:
    extract_wavcaps_audio_text_cache(
        out_dir=Path('/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results/domain_gap_closure_suite/caches/wavcaps_humanify'),
        dataset_name='humanify/AS-WavCaps',
        clap_model_name='laion/clap-htsat-unfused',
        clip_backbone_name='openai/clip-vit-base-patch32',
        target_sampling_rate=48000,
        max_examples=200000,
        sampling_policy='stratified',
        device='cuda',
        audio_batch_size=64,
        text_batch_size=256,
        split_name='train',
        stream=True,
    )
    print('wavcaps warmup: ok')
except Exception as e:
    print('wavcaps warmup: failed', type(e).__name__, str(e))
PY

python -m MultiModal.multimodal.experiments.run_stage29_cc3m_phaseA_modular \
  --config MultiModal/configs/stage29_domain_gap_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage29_gpu7.log"
touch "$MARK/gpu7.stage29.done"

python -m MultiModal.multimodal.experiments.run_stage30_modular_vs_nonmodular \
  --config MultiModal/configs/stage30_domain_gap_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage30_gpu7.log"
touch "$MARK/gpu7.stage30.done"

while [[ ! -f "$MARK/gpu4.stage29.done" || ! -f "$MARK/gpu5.stage29.done" || ! -f "$MARK/gpu6.stage29.done" || ! -f "$MARK/gpu7.stage29.done" ]]; do
  sleep 30
done

python -m MultiModal.multimodal.experiments.run_stage31_wavcaps_scaling \
  --config MultiModal/configs/stage31_domain_gap_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage31_gpu7.log"
touch "$MARK/gpu7.stage31.done"

# Final aggregate waits for all shard completions.
while [[ ! -f "$MARK/gpu4.stage27.done" || ! -f "$MARK/gpu5.stage28.done" || \
         ! -f "$MARK/gpu4.stage30.done" || ! -f "$MARK/gpu5.stage30.done" || ! -f "$MARK/gpu6.stage30.done" || ! -f "$MARK/gpu7.stage30.done" || \
         ! -f "$MARK/gpu6.stage31.done" || ! -f "$MARK/gpu7.stage31.done" || \
         ! -f "$MARK/gpu4.stage32.done" || ! -f "$MARK/gpu5.stage32.done" ]]; do
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage33_domain_gap_aggregate \
  --config MultiModal/configs/stage33_domain_gap_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage33_gpu7.log"
touch "$MARK/gpu7.stage33.done"

echo "[$(date '+%F %T')] queue_domain_gap_gpu7 complete"

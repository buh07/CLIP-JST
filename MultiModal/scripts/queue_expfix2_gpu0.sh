#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/experimental_fixes_suite"
LOG_DIR="$MM_ROOT/logs/experimental_fixes_suite"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# Build WavCaps/WavCaps-only 46k cache from existing shard caches.
python - <<'PY' 2>&1 | tee "$LOG_DIR/wavcaps_clean_source_46k_prepare_gpu0.log"
from pathlib import Path
from MultiModal.multimodal.data import merge_wavcaps_audio_text_cache_shards
ROOT = Path('/jumbo/lisp/f004ndc/CLIP JST')
MM = ROOT / 'MultiModal'
sh = MM / 'results' / 'domain_gap_closure_suite' / 'caches' / 'wavcaps_humanify_shards'
out = MM / 'results' / 'experimental_fixes_suite' / 'caches' / 'wavcaps_clean_source_46k'
res = merge_wavcaps_audio_text_cache_shards(
    shard_dirs=[sh / 'shard0', sh / 'shard1', sh / 'shard2', sh / 'shard3'],
    out_dir=out,
    dataset_name='humanify/AS-WavCaps',
    clap_model_name='laion/clap-htsat-unfused',
    clip_backbone_name='openai/clip-vit-base-patch32',
    target_sampling_rate=48_000,
    max_examples=46_000,
    sampling_policy='stratified',
    merge_seed=2026,
    include_sources=['WavCaps/WavCaps'],
)
print('prepared wavcaps clean_source_46k cache:', res)
PY
touch "$MARK/wavcaps_clean_source_46k.ready"

# stage45 requires m=512 phase-a checkpoints from stage44 (gpu3.stage44.done already exists).
while [[ ! -f "$MARK/gpu3.stage44.done" ]]; do
  sleep 20
done

python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_expfix_mixed46k_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage45_mixed46k_gpu0.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_expfix_mixed200k_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage45_mixed200k_gpu0.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_expfix_clean_source_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage45_clean_source_gpu0.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_expfix_clean_source_46k_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage45_clean_source_46k_gpu0.log"

touch "$MARK/gpu0.stage45.done"

# Wait for all GPUs to finish stage45, then run aggregate.
while [[ ! -f "$MARK/gpu1.stage45.done" || ! -f "$MARK/gpu2.stage45.done" || ! -f "$MARK/gpu3.stage45.done" ]]; do
  sleep 30
done

python -m MultiModal.multimodal.experiments.run_stage46_experimental_fixes_aggregate \
  --config MultiModal/configs/stage46_expfix_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage46_aggregate.log"
touch "$MARK/gpu0.stage46.done"
echo "[$(date '+%F %T')] queue_expfix2_gpu0 complete"

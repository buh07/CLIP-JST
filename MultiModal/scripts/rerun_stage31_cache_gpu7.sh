#!/usr/bin/env bash
set -euo pipefail
ROOT='/jumbo/lisp/f004ndc/CLIP JST'
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/domain_gap_closure_suite"
MARK_DIR="$MM_ROOT/results/domain_gap_closure_suite/markers/shards"
mkdir -p "$LOG_DIR" "$MARK_DIR"
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
cd "$ROOT"
source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
python - <<'PY' 2>&1 | tee "$LOG_DIR/stage31_cache_rerun_gpu7.log"
from pathlib import Path
import json
from MultiModal.multimodal.data import extract_wavcaps_audio_text_cache
out=Path('/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results/domain_gap_closure_suite/caches/wavcaps_humanify')
res=extract_wavcaps_audio_text_cache(
    out_dir=out,
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
print('CACHE_BUILD_OK',res)
meta=json.loads((out/'metadata.json').read_text())
print('CACHE_STATUS',meta.get('status'),'NUM_PAIRS',meta.get('num_pairs'))
Path('/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results/domain_gap_closure_suite/markers/shards/stage31_wavcaps_cache.done').write_text('ok\\n', encoding='utf-8')
PY

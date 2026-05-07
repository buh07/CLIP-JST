#!/usr/bin/env bash
set -euo pipefail
ROOT='/jumbo/lisp/f004ndc/CLIP JST'
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/domain_gap_closure_suite"
MARK_DIR="$MM_ROOT/results/domain_gap_closure_suite/markers/shards"
SHARD_ROOT="$MM_ROOT/results/domain_gap_closure_suite/caches/wavcaps_humanify_shards"
mkdir -p "$LOG_DIR" "$MARK_DIR" "$SHARD_ROOT"
export PYTHONUNBUFFERED=1
cd "$ROOT"
source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"

rm -f "$MARK_DIR/wavcaps_cache.shard2.done.json" "$MARK_DIR/wavcaps_cache.shard2.fail.json"

while true; do
  if [[ -f "$MARK_DIR/wavcaps_cache.shard2a.fail.json" || -f "$MARK_DIR/wavcaps_cache.shard2b.fail.json" || -f "$MARK_DIR/wavcaps_cache.shard2c.fail.json" || -f "$MARK_DIR/wavcaps_cache.shard2d.fail.json" ]]; then
    echo '{"status":"failed","reason":"subshard failure"}' > "$MARK_DIR/wavcaps_cache.shard2.fail.json"
    exit 1
  fi
  if [[ -f "$MARK_DIR/wavcaps_cache.shard2a.done.json" && -f "$MARK_DIR/wavcaps_cache.shard2b.done.json" && -f "$MARK_DIR/wavcaps_cache.shard2c.done.json" && -f "$MARK_DIR/wavcaps_cache.shard2d.done.json" ]]; then
    break
  fi
  sleep 20
done

python - << 'PY'
from pathlib import Path
from MultiModal.multimodal.data import merge_wavcaps_audio_text_cache_shards
ROOT=Path('/jumbo/lisp/f004ndc/CLIP JST')
MM=ROOT/'MultiModal'
SHARD_ROOT=MM/'results'/'domain_gap_closure_suite'/'caches'/'wavcaps_humanify_shards'
out=SHARD_ROOT/'shard2'
res=merge_wavcaps_audio_text_cache_shards(
    shard_dirs=[SHARD_ROOT/'shard2_split_a', SHARD_ROOT/'shard2_split_b', SHARD_ROOT/'shard2_split_c', SHARD_ROOT/'shard2_split_d'],
    out_dir=out,
    dataset_name='humanify/AS-WavCaps',
    clap_model_name='laion/clap-htsat-unfused',
    clip_backbone_name='openai/clip-vit-base-patch32',
    target_sampling_rate=48_000,
    max_examples=70_000,
    sampling_policy='stratified',
    merge_seed=2026,
)
print('shard2 merged:', res)
PY

python - << 'PY'
import json
from pathlib import Path
m=Path('/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results/domain_gap_closure_suite/markers/shards/wavcaps_cache.shard2.done.json')
m.write_text(json.dumps({'status':'ok','split_mode':'mod16_2_6_10_14','merged':'shard2_split_a+shard2_split_b+shard2_split_c+shard2_split_d'},indent=2),encoding='utf-8')
print('wrote',m)
PY

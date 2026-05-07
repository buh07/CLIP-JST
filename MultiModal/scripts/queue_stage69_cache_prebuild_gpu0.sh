#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/stage69_prereg_suite"
LOG_DIR="$MM_ROOT/logs/stage69_prereg_suite"
MARK="$OUT/markers"
CACHE_ROOT="$OUT/caches/speechcoco_av"
mkdir -p "$LOG_DIR" "$MARK"

TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
export TMPDIR

HF_HOME="$OUT/hf_home"
XDG_CACHE_HOME="$OUT/xdg_cache"
HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
export HF_HOME XDG_CACHE_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

while [[ ! -f "$MARK/stage69_prereg_locked.done" && ! -f "$MARK/stage69_prereg_locked.done.json" ]]; do sleep 20; done

# smoke cache build (tiny caps) + disjoint sanity
python -m MultiModal.multimodal.experiments.run_stage69_third_triple_speechcoco \
  --config MultiModal/configs/stage69_speechcoco_cache_prebuild_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage69_cache_smoke_gpu0.log"

python - <<'PY'
import json
from pathlib import Path
meta = json.loads(Path('/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results/stage69_prereg_suite/caches/speechcoco_av/metadata.json').read_text())
assert meta['overlap_train_eval_count'] == 0, meta['overlap_train_eval_count']
assert meta['overlap_val_eval_count'] == 0, meta['overlap_val_eval_count']
assert meta['split_counts']['eval_test'] >= 64
assert meta['split_counts']['phase_b_val'] >= 32
assert meta['split_counts']['phase_b_train'] >= 96
print('smoke disjoint checks passed')
PY

# full cache build
python -m MultiModal.multimodal.experiments.run_stage69_third_triple_speechcoco \
  --config MultiModal/configs/stage69_speechcoco_cache_full_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage69_cache_full_gpu0.log"

[[ -f "$CACHE_ROOT/image_feats_clip_raw.pt" ]]
[[ -f "$CACHE_ROOT/audio_feats_clap_raw.pt" ]]
[[ -f "$CACHE_ROOT/text_feats_clip_raw.pt" ]]
[[ -f "$CACHE_ROOT/metadata.json" ]]

touch "$MARK/stage69_cache_ready.done"
touch "$MARK/gpu0.stage69_cache_prebuild.done"
echo "[$(date '+%F %T')] queue_stage69_cache_prebuild_gpu0 complete"

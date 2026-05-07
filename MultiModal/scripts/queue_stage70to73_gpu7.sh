#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/stage69_prereg_suite/speechcoco_full/aggregate"
TOP_OUT="$MM_ROOT/results/stage69_prereg_suite"
LOG_DIR="$MM_ROOT/logs/stage69_prereg_suite"
MARK="$TOP_OUT/markers"
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

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

for g in 0 1 2 3 4 5 6 7; do
  while [[ ! -f "$MARK/gpu${g}.stage69.done" ]]; do
    if [[ -f "$MARK/gpu${g}.stage69.failed" ]]; then
      echo "GPU shard ${g} failed; aborting coordinator."
      exit 1
    fi
    sleep 30
  done
done

python -m MultiModal.multimodal.experiments.run_stage70_third_triple_aggregate --config MultiModal/configs/stage70_third_triple_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage70_aggregate.log"
# Stage72 precedes Stage71 so Stage71 can evaluate prereg geometric-top2 criterion.
python -m MultiModal.multimodal.experiments.run_stage72_third_triple_form_compare --config MultiModal/configs/stage72_third_triple_form_compare.yaml 2>&1 | tee "$LOG_DIR/stage72_form_compare.log"
python -m MultiModal.multimodal.experiments.run_stage71_third_triple_prospective_check --config MultiModal/configs/stage71_third_triple_prospective_check.yaml 2>&1 | tee "$LOG_DIR/stage71_prospective.log"
python -m MultiModal.multimodal.experiments.run_stage73_three_triple_meta_update --config MultiModal/configs/stage73_three_triple_meta_update.yaml 2>&1 | tee "$LOG_DIR/stage73_meta_update.log"

touch "$MARK/stage70to73.done"
echo "[$(date '+%F %T')] queue_stage70to73_gpu7 complete"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/reinforce_completion_suite"
MARK_DIR="$MM_ROOT/results/reinforce_completion_suite/markers"
mkdir -p "$LOG_DIR" "$MARK_DIR"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' >/dev/null 2>&1 || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

run_or_skip() {
  local cfg="$1"
  local mod="$2"
  local log="$3"
  if [[ -f "$cfg" ]]; then
    python -m "$mod" --config "$cfg" 2>&1 | tee "$log"
  else
    echo "[$(date '+%F %T')] missing cfg: $cfg (skip)" | tee -a "$log"
  fi
}

wait_markers() {
  local prefix="$1"
  local n="$2"
  while true; do
    local c
    c=$(ls "$MARK_DIR" 2>/dev/null | grep -c "^${prefix}_gpu" || true)
    if [[ "$c" -ge "$n" ]]; then
      break
    fi
    sleep 20
  done
}

# W2
run_or_skip   "$MM_ROOT/configs/stage20_w2_joint_clap_gpu7.yaml"   MultiModal.multimodal.experiments.run_stage20_modular_audio_transitivity   "$LOG_DIR/w2_joint_clap.gpu7.log"
touch "$MARK_DIR/w2_gpu7.done"
wait_markers w2 8
if [[ "7" == "7" ]]; then
  python -m MultiModal.multimodal.experiments.run_stage21_modular_transitivity_aggregate     --config "$MM_ROOT/configs/stage21_w2_joint_clap_aggregate.yaml"     2>&1 | tee "$LOG_DIR/w2_aggregate.gpu7.log"
  touch "$MARK_DIR/w2_aggregate.done"
fi
while [[ ! -f "$MARK_DIR/w2_aggregate.done" ]]; do sleep 20; done

# W9
run_or_skip   "$MM_ROOT/configs/stage25_w9_replication_gpu7.yaml"   MultiModal.multimodal.experiments.run_stage25_modtrans_jl_ablation   "$LOG_DIR/w9_replication.gpu7.log"
touch "$MARK_DIR/w9_gpu7.done"
wait_markers w9 8
if [[ "7" == "7" ]]; then
  python -m MultiModal.multimodal.experiments.run_stage26_modtrans_jl_aggregate     --config "$MM_ROOT/configs/stage26_w9_replication_aggregate.yaml"     2>&1 | tee "$LOG_DIR/w9_aggregate.gpu7.log"
  touch "$MARK_DIR/w9_aggregate.done"
fi
while [[ ! -f "$MARK_DIR/w9_aggregate.done" ]]; do sleep 20; done

# W11
run_or_skip   "$MM_ROOT/configs/stage61_w11_shuffled_gpu7.yaml"   MultiModal.multimodal.experiments.run_stage61_shuffled_caption_control   "$LOG_DIR/w11_shuffled.gpu7.log"
touch "$MARK_DIR/w11_gpu7.done"
wait_markers w11 8

# W3 cache prebuild gate
if [[ "7" == "0" ]]; then
  run_or_skip     "$MM_ROOT/configs/stage62_w3_clotho_prebuild_gpu0.yaml"     MultiModal.multimodal.experiments.run_stage62_clotho_intermediate_eval     "$LOG_DIR/w3_clotho_prebuild.gpu0.log"
  touch "$MARK_DIR/w3_clotho_cache_ready.done"
fi
while [[ ! -f "$MARK_DIR/w3_clotho_cache_ready.done" ]]; do sleep 20; done

# W3 Clotho evaluation shards
run_or_skip   "$MM_ROOT/configs/stage62_w3_clotho_gpu7.yaml"   MultiModal.multimodal.experiments.run_stage62_clotho_intermediate_eval   "$LOG_DIR/w3_clotho_eval.gpu7.log"
touch "$MARK_DIR/w3_gpu7.done"
wait_markers w3 8
if [[ "7" == "7" ]]; then
  python -m MultiModal.multimodal.experiments.run_stage63_clotho_intermediate_aggregate     --config "$MM_ROOT/configs/stage63_w3_clotho_aggregate.yaml"     2>&1 | tee "$LOG_DIR/w3_clotho_aggregate.gpu7.log"
  touch "$MARK_DIR/w3_aggregate.done"
fi
while [[ ! -f "$MARK_DIR/w3_aggregate.done" ]]; do sleep 20; done

# W7 AudioCLIP baseline (GPU0 only)
if [[ "7" == "0" ]]; then
  python -m pip install -q pytorch-ignite librosa soundfile ftfy >/dev/null 2>&1 || true
  python - <<'PY2' 2>&1 | tee "$LOG_DIR/w7_audioclip_importcheck.gpu0.log"
import importlib,sys
mods=['ignite','librosa','soundfile','ftfy']
for m in mods:
    try:
        importlib.import_module(m)
        print(m,'OK')
    except Exception as e:
        print(m,'MISSING',e)
PY2
  run_or_skip     "$MM_ROOT/configs/w7_audioclip_1k_split_gpu0.yaml"     MultiModal.multimodal.experiments.run_w7_audioclip_1k_split     "$LOG_DIR/w7_audioclip_1k_split.gpu0.log"
  touch "$MARK_DIR/w7_audioclip.done"
fi
while [[ ! -f "$MARK_DIR/w7_audioclip.done" ]]; do sleep 20; done

touch "$MARK_DIR/reinforce_missing_gpu7.all_done"
echo "[$(date '+%F %T')] queue_reinforce_missing_gpu7 complete"

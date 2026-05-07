#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT_ROOT="$MM_ROOT/results/e8_concat_suite"
LOG_DIR="$MM_ROOT/logs/e8_concat_suite_split4"
MARKERS="$OUT_ROOT/markers"
mkdir -p "$LOG_DIR" "$MARKERS"

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' opacus
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

run_step () {
  local name="$1"
  local cmd="$2"
  local log="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] START $name"
  echo "[$(date '+%F %T')] CMD   $cmd"
  eval "$cmd" 2>&1 | tee "$log"
  echo "[$(date '+%F %T')] DONE  $name"
}

wait_file () {
  local p="$1"
  echo "[$(date '+%F %T')] WAIT  $p"
  while [[ ! -f "$p" ]]; do
    sleep 30
  done
  echo "[$(date '+%F %T')] FOUND $p"
}

count_eval () {
  local p="$1"
  python - "$p" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
print(len(list(root.rglob("eval.json"))) if root.exists() else 0)
PY
}

wait_eval_count () {
  local p="$1"
  local expected="$2"
  local label="$3"
  while true; do
    local n
    n="$(count_eval "$p")"
    echo "[$(date '+%F %T')] $label progress: $n/$expected"
    if [[ "$n" -ge "$expected" ]]; then
      break
    fi
    sleep 60
  done
}

read_gate_field () {
  local key="$1"
  python - "$OUT_ROOT/stage5_gate_decision.json" "$key" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    obj = json.load(f)
v = obj.get(sys.argv[2])
if isinstance(v, bool):
    print("true" if v else "false")
else:
    print(v)
PY
}

run_step "stage5_e8a_gpu5_flickr_s12" "python -m MultiModal.multimodal.experiments.run_stage5_e8a_concat --config MultiModal/configs/stage5_e8a_gpu5_flickr_s12.yaml"

wait_file "$MARKERS/stage5_gate.done.json"
RUN_STAGE6="$(read_gate_field run_stage6)"
RUN_STAGE8="$(read_gate_field run_stage8)"

if [[ "$RUN_STAGE6" == "true" ]]; then
  run_step "stage6_e8b_gpu5_flickr_s12" "python -m MultiModal.multimodal.experiments.run_stage6_e8b_mask_concat --config MultiModal/configs/stage6_e8b_gpu5_flickr_s12.yaml"
fi

if [[ "$RUN_STAGE8" == "true" ]]; then
  wait_eval_count "$OUT_ROOT/stage6_e8b/coco" 90 "stage6_coco"
  wait_eval_count "$OUT_ROOT/stage6_e8b/flickr30k" 90 "stage6_flickr30k"
  run_step "stage8_e8d_gpu5_eps8" "python -m MultiModal.multimodal.experiments.run_stage8_e8d_dpsgd --config MultiModal/configs/stage8_e8d_dpsgd_gpu5_eps8.yaml"
fi

echo "[$(date '+%F %T')] GPU5 split queue complete."

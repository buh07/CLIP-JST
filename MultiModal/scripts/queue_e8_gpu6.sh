#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT_ROOT="$MM_ROOT/results/e8_concat_suite"
LOG_DIR="$MM_ROOT/logs/e8_concat_suite"
MARKERS="$OUT_ROOT/markers"
mkdir -p "$LOG_DIR" "$MARKERS"

export CUDA_VISIBLE_DEVICES=6
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

wait_marker () {
  local marker="$1"
  echo "[$(date '+%F %T')] WAIT  $marker"
  while [[ ! -f "$marker" ]]; do
    sleep 30
  done
  echo "[$(date '+%F %T')] FOUND $marker"
}

read_gate_field () {
  local key="$1"
  python - "$OUT_ROOT/stage5_gate_decision.json" "$key" <<'PY'
import json, sys
p = sys.argv[1]
k = sys.argv[2]
with open(p, encoding="utf-8") as f:
    obj = json.load(f)
v = obj.get(k)
if isinstance(v, bool):
    print("true" if v else "false")
else:
    print(v)
PY
}

run_step "e8_smoke_gpu6" "python -m MultiModal.multimodal.experiments.run_smoke_tests --config MultiModal/configs/smoke_tests_e8.yaml"
run_step "stage5_e8a_gpu6_coco" "python -m MultiModal.multimodal.experiments.run_stage5_e8a_concat --config MultiModal/configs/stage5_e8a_gpu6_coco.yaml"
run_step "stage5_gate" "python -m MultiModal.multimodal.experiments.run_stage5_gate --config MultiModal/configs/stage5_gate.yaml"

wait_marker "$MARKERS/stage5_e8a_flickr30k.done.json"

RUN_STAGE6="$(read_gate_field run_stage6)"
RUN_STAGE7="$(read_gate_field run_stage7)"
RUN_STAGE8="$(read_gate_field run_stage8)"
STOP_FLAG="$(read_gate_field stop)"

echo "[$(date '+%F %T')] Gate decision: run_stage6=$RUN_STAGE6 run_stage7=$RUN_STAGE7 run_stage8=$RUN_STAGE8 stop=$STOP_FLAG"

if [[ "$RUN_STAGE6" == "true" ]]; then
  run_step "stage6_e8b_gpu6_coco" "python -m MultiModal.multimodal.experiments.run_stage6_e8b_mask_concat --config MultiModal/configs/stage6_e8b_gpu6_coco.yaml"
  wait_marker "$MARKERS/stage6_e8b_flickr30k.done.json"
fi

if [[ "$RUN_STAGE7" == "true" ]]; then
  run_step "stage7_e8c_privacy_gpu6" "python -m MultiModal.multimodal.experiments.run_stage7_e8c_privacy_attacks --config MultiModal/configs/stage7_e8c_privacy.yaml"
fi

if [[ "$RUN_STAGE8" == "true" ]]; then
  wait_marker "$MARKERS/stage8_e8d.done.json"
fi

run_step "stage9_e8_aggregate_gpu6" "python -m MultiModal.multimodal.experiments.run_stage9_e8_aggregate --config MultiModal/configs/stage9_e8_aggregate.yaml"

echo "[$(date '+%F %T')] GPU6 E8 queue complete."

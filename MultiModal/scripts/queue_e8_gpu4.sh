#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT_ROOT="$MM_ROOT/results/e8_concat_suite"
LOG_DIR="$MM_ROOT/logs/e8_concat_suite_split4"
MARKERS="$OUT_ROOT/markers"
mkdir -p "$LOG_DIR" "$MARKERS"

export CUDA_VISIBLE_DEVICES=4
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

run_step "stage5_e8a_gpu4_coco_s12" "python -m MultiModal.multimodal.experiments.run_stage5_e8a_concat --config MultiModal/configs/stage5_e8a_gpu4_coco_s12.yaml"

wait_file "$MARKERS/stage5_gate.done.json"
RUN_STAGE6="$(read_gate_field run_stage6)"
if [[ "$RUN_STAGE6" == "true" ]]; then
  run_step "stage6_e8b_gpu4_coco_s12" "python -m MultiModal.multimodal.experiments.run_stage6_e8b_mask_concat --config MultiModal/configs/stage6_e8b_gpu4_coco_s12.yaml"
fi

echo "[$(date '+%F %T')] GPU4 split queue complete."

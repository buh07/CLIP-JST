#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/CLIP JST"
mkdir -p logs results/rerun_fix_20260429/markers

run_stage () {
  local name="$1"
  local cmd="$2"
  local log="logs/rerun_fix_${name}_gpu7.log"
  local start_ts
  start_ts=$(date -Iseconds)
  echo "[$start_ts] START ${name}" | tee "$log"
  eval "$cmd" 2>&1 | tee -a "$log"
  local end_ts
  end_ts=$(date -Iseconds)
  python - <<PY
import json, time, pathlib
name = ${name@Q}
out=pathlib.Path(f'results/rerun_fix_20260429/markers/{name}.done.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'stage':name,'status':'done','finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=2))
print('WROTE', out)
PY
  echo "[$end_ts] DONE ${name}" | tee -a "$log"
}

run_stage E3 "CUDA_VISIBLE_DEVICES=7 python -u experiments/run_E3.py --config configs/_rerun_fix_E3.yaml"
run_stage D3 "CUDA_VISIBLE_DEVICES=7 python -u experiments/run_D3.py --config configs/_rerun_fix_D3.yaml"
run_stage E5 "CUDA_VISIBLE_DEVICES=7 python -u experiments/run_E5.py --config configs/_rerun_fix_E5.yaml"

echo "ALL_DONE" | tee logs/rerun_fix_suite_gpu7.log

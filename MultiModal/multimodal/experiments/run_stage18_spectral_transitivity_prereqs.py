from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import yaml

from ..common import env_snapshot, load_json, mark_done, save_json
from .run_stage2_e7_karpathy import run as run_stage2


def run(cfg: dict) -> None:
    start = time.time()
    project_root = Path(cfg["project_root"]).resolve()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage2_cfg = copy.deepcopy(cfg["stage2_cfg"])
    stage2_cfg["project_root"] = str(project_root)
    stage2_cfg["output_root"] = str((output_root / "stage2_backend").resolve())
    stage2_cfg["spectral_reg_values"] = [0.1, 1.0, 10.0]

    base_methods = set(stage2_cfg.get("method_filter", []))
    base_methods.update(
        {
            "clip_head",
            "orth_jl_trainable",
            "spectral_align_trainable_reg0p1",
            "spectral_align_trainable_reg1p0",
            "spectral_align_trainable_reg10p0",
        }
    )
    stage2_cfg["method_filter"] = sorted(base_methods)

    print("Running spectral prerequisite sweep via stage2 backend...")
    run_stage2(stage2_cfg)

    stage2_summary = Path(stage2_cfg["output_root"]) / "E7_karpathy_full_results.json"
    if not stage2_summary.exists():
        raise FileNotFoundError(f"stage2 backend summary missing: {stage2_summary}")

    obj = load_json(stage2_summary)
    summary = {
        "stage": "stage18_spectral_transitivity_prereqs",
        "source": str(stage2_summary),
        "datasets": {},
        "spectral_reg_values": [0.1, 1.0, 10.0],
    }
    for ds, payload in obj.get("datasets", {}).items():
        summary["datasets"][ds] = payload.get("stats", {})

    save_json(summary, output_root / "stage18_spectral_results.json")

    provenance = env_snapshot(
        project_root,
        seeds=list(stage2_cfg.get("seeds", [])),
        extra={
            "stage": "stage18_spectral_transitivity_prereqs",
            "elapsed_sec": time.time() - start,
            "stage2_backend_cfg": stage2_cfg,
        },
    )
    save_json(provenance, output_root / "provenance_stage18.json")
    mark_done(markers / "stage18_spectral_transitivity_prereqs.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 18 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

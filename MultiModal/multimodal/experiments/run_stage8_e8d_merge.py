from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..common import load_json, mark_done, save_json
from ..eval.stats import build_metric_report


def run(cfg: dict) -> None:
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage8_e8d"
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    shard_files = [Path(p).resolve() for p in cfg["stage8_shard_files"]]
    shards = []
    for p in shard_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing Stage8 shard: {p}")
        shards.append(load_json(p))

    merged = {
        "stage": "stage8_e8d_dpsgd",
        "config": cfg,
        "raw": {},
        "stats": {},
        "merged_from": [str(p) for p in shard_files],
    }

    for sh in shards:
        for eps_key, rows in sh.get("raw", {}).items():
            if eps_key in merged["raw"]:
                raise ValueError(f"Duplicate epsilon key across shards: {eps_key}")
            merged["raw"][eps_key] = rows

    per_method = {}
    for eps_key, rows in merged["raw"].items():
        per_method[eps_key] = []
        for r in rows:
            per_method[eps_key].append(
                {
                    "seed": int(r["seed"]),
                    "avg_R": float(r["avg_R"]),
                    "mlp_rel_error": float(r["mlp_inverter"]["mean_relative_reconstruction_error"]),
                    "pseudo_rel_error": float(r["pseudoinverse"]["mean_relative_reconstruction_error"]),
                    "epsilon_spent": float(r["train_result"]["dp_meta"]["epsilon_spent_final"]),
                }
            )

    if not per_method:
        raise ValueError("No Stage8 rows found in shards")
    baseline = cfg.get("baseline_method")
    if not baseline or baseline not in per_method:
        baseline = sorted(per_method.keys())[-1]
    merged["stats"] = build_metric_report(
        per_method,
        metrics=["avg_R", "mlp_rel_error", "pseudo_rel_error", "epsilon_spent"],
        baseline_method=baseline,
    )

    out_name = cfg.get("stage8_merged_results_name", "E8d_dpsgd_results.json")
    save_json(merged, stage_root / out_name)
    mark_done(markers / cfg.get("stage8_merged_marker_name", "stage8_e8d.done.json"), {"results_file": out_name})
    print("Stage8 shard merge complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

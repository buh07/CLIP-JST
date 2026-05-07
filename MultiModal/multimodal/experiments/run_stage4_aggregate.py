from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..common import mark_done, save_json


def _seed_coverage_check(raw: dict, expected_seeds: list[int]) -> dict:
    missing = []
    for m_key, methods in raw.items():
        for method, rows in methods.items():
            have = sorted(int(r["seed"]) for r in rows)
            for s in expected_seeds:
                if s not in have:
                    missing.append({"m": m_key, "method": method, "seed": s})
    return {"missing": missing, "ok": len(missing) == 0}


def run(cfg: dict) -> None:
    output_root = Path(cfg["output_root"]).resolve()
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage2_path = Path(cfg["stage2_results"])
    stage3_path = Path(cfg["stage3_results"])

    stage2 = {}
    if stage2_path.exists():
        import json

        with open(stage2_path, encoding="utf-8") as f:
            stage2 = json.load(f)

    stage3 = {}
    if stage3_path.exists():
        import json

        with open(stage3_path, encoding="utf-8") as f:
            stage3 = json.load(f)

    summary = {
        "seed_coverage": {},
        "sanity": {},
        "artifacts": {
            "stage2_results": str(stage2_path),
            "stage3_results": str(stage3_path),
        },
    }

    seeds = list(cfg["seeds"])

    if stage2:
        # Coverage + chance-level sanity for random_jl_only.
        for ds, ds_obj in stage2.get("datasets", {}).items():
            cov = _seed_coverage_check(ds_obj.get("raw", {}), seeds)
            summary["seed_coverage"][f"stage2_{ds}"] = cov

            chance_flags = []
            for m_key, methods in ds_obj.get("raw", {}).items():
                rows = methods.get("random_jl_only", [])
                if not rows:
                    continue
                avg = sum(float(r["avg_R"]) for r in rows) / len(rows)
                chance_flags.append({"m": m_key, "avg_random_jl_only": avg, "near_chance": avg < cfg.get("chance_threshold", 0.02)})
            summary["sanity"][f"stage2_{ds}_random_jl_only"] = chance_flags

    if stage3:
        cov = _seed_coverage_check(stage3.get("raw", {}), seeds)
        summary["seed_coverage"]["stage3_trimodal"] = cov

    save_json(summary, output_root / "aggregate_summary.json")
    mark_done(markers / "stage4_aggregate.done.json")
    print("Stage 4 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

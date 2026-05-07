from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from ..common import load_json, mark_done, save_json
from ..eval.stats import build_metric_report
from .run_stage16_budget_matched_frontier import DEFAULT_METRICS


def _merge_partition_methods(dst: dict, src: dict) -> None:
    for part, methods in src.get("raw", {}).items():
        dst.setdefault("raw", {}).setdefault(part, {})
        for method, rows in methods.items():
            if method in dst["raw"][part]:
                by_seed = {int(r["seed"]): r for r in dst["raw"][part][method]}
                for r in rows:
                    by_seed[int(r["seed"])] = r
                dst["raw"][part][method] = [by_seed[s] for s in sorted(by_seed)]
            else:
                dst["raw"][part][method] = rows


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            shutil.copy2(p, out)


def run(cfg: dict) -> None:
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage16_budget_matched"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    shard_roots = [Path(p).resolve() for p in cfg["shard_output_roots"]]
    merged = {
        "stage": "stage16_budget_matched_frontier",
        "config": {"merged_from": [str(p) for p in shard_roots]},
        "raw": {},
        "stats": {},
    }

    for shard in shard_roots:
        s16 = shard / "stage16_budget_matched" / "E16_budget_matched_results.json"
        if not s16.exists():
            raise FileNotFoundError(f"missing shard result: {s16}")
        obj = load_json(s16)
        _merge_partition_methods(merged, obj)
        _copy_tree(shard / "stage16_budget_matched", stage_root)

    metrics = list(cfg.get("metrics_for_stats", DEFAULT_METRICS))
    baseline = cfg.get("baseline_method", "mask_concat")
    for part, methods in merged["raw"].items():
        merged["stats"][part] = build_metric_report(
            methods,
            metrics=metrics,
            baseline_method=baseline,
        )

    merged_path = stage_root / cfg.get("stage16_merged_results_name", "E16_budget_matched_results.json")
    save_json(merged, merged_path)
    mark_done(markers / cfg.get("stage16_merged_marker_name", "stage16_budget_matched.done.json"), {"results_file": str(merged_path)})
    print("Stage16 shard merge complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

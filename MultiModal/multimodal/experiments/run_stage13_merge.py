from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from ..common import load_json, mark_done, save_json


def _merge_partition_methods(dst: dict, src: dict) -> None:
    for part, methods in src.get("raw", {}).items():
        dst.setdefault("raw", {}).setdefault(part, {})
        for method, rows in methods.items():
            if method in dst["raw"][part]:
                # Merge by seed to avoid duplicates.
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
    stage_root = output_root / "stage13_federated"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    shard_roots = [Path(p).resolve() for p in cfg["shard_output_roots"]]
    merged = {
        "stage": "stage13_federated",
        "config": {"merged_from": [str(p) for p in shard_roots]},
        "raw": {},
        "stats": {},
    }

    for shard in shard_roots:
        s13 = shard / "stage13_federated" / "E13_federated_results.json"
        if not s13.exists():
            raise FileNotFoundError(f"missing shard result: {s13}")
        obj = load_json(s13)
        _merge_partition_methods(merged, obj)

        # Copy per-seed artifacts and checkpoints for downstream attacks.
        _copy_tree(shard / "stage13_federated", stage_root)

    from ..eval.stats import build_metric_report

    metrics = [
        "i2t_R@1", "i2t_R@5", "i2t_R@10",
        "t2i_R@1", "t2i_R@5", "t2i_R@10",
        "avg_R", "mlp_rel_error", "linear_rel_error", "comm_mb", "rounds",
    ]
    baseline = cfg.get("baseline_method", "mask_concat")
    for part, methods in merged["raw"].items():
        merged["stats"][part] = build_metric_report(methods, metrics=metrics, baseline_method=baseline)

    save_json(merged, stage_root / "E13_federated_results.json")
    mark_done(markers / "stage13_federated.done.json")
    print("Stage 13 merge complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

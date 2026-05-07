from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from ..common import load_json, mark_done, save_json


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


def _merge_dataset_level(dst: dict, src: dict) -> None:
    for ds, payload in src.get("datasets", {}).items():
        if ds not in dst["datasets"]:
            dst["datasets"][ds] = payload
            continue
        # Merge raw rows by m/method/seed.
        raw_dst = dst["datasets"][ds].setdefault("raw", {})
        raw_src = payload.get("raw", {})
        for m_key, methods in raw_src.items():
            raw_dst.setdefault(m_key, {})
            for method, rows in methods.items():
                if method in raw_dst[m_key]:
                    by_seed = {int(r["seed"]): r for r in raw_dst[m_key][method]}
                    for r in rows:
                        by_seed[int(r["seed"])] = r
                    raw_dst[m_key][method] = [by_seed[s] for s in sorted(by_seed)]
                else:
                    raw_dst[m_key][method] = rows
        # Stats are recomputed below.


def run(cfg: dict) -> None:
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    shard_roots = [Path(p).resolve() for p in cfg["shard_output_roots"]]
    merged = {
        "stage": "stage2_e7_karpathy",
        "datasets": {},
        "config": {"merged_from": [str(p) for p in shard_roots]},
    }

    for shard in shard_roots:
        p = shard / "E7_karpathy_full_results.json"
        if not p.exists():
            raise FileNotFoundError(f"missing stage2 shard summary: {p}")
        obj = load_json(p)
        _merge_dataset_level(merged, obj)
        _copy_tree(shard, output_root)

    # Recompute stats from merged raw using the shared helper.
    from ..eval.stats import build_metric_report

    metrics_for_stats = [
        "i2t_R@1",
        "i2t_R@5",
        "i2t_R@10",
        "t2i_R@1",
        "t2i_R@5",
        "t2i_R@10",
        "avg_R",
    ]
    baseline = cfg.get("baseline_method", "clip_head")
    for ds, payload in merged["datasets"].items():
        payload.setdefault("stats", {})
        for m_key, per_method in payload.get("raw", {}).items():
            payload["stats"][m_key] = build_metric_report(
                per_method,
                metrics=metrics_for_stats,
                baseline_method=baseline,
            )

    save_json(merged, output_root / "E7_karpathy_full_results.json")
    mark_done(markers / "stage2_e7_karpathy.done.json")
    print("Stage2 merge complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)


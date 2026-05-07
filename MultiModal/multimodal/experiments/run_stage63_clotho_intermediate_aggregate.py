from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml

from ..common import load_json, mark_done, save_json
from ..eval.stats import build_metric_report


def _merge_seed_rows(dst_rows: list[dict[str, Any]], src_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed = {int(r["seed"]): r for r in dst_rows if "seed" in r}
    for r in src_rows:
        by_seed[int(r["seed"])] = r
    return [by_seed[s] for s in sorted(by_seed)]


def run(cfg: dict) -> None:
    start = time.time()
    stage_name = "stage63_clotho_intermediate_aggregate"

    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    shard_roots = [Path(p).resolve() for p in cfg.get("stage62_shard_roots", [])]
    merged: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    cache_meta: dict[str, Any] | None = None

    for root in shard_roots:
        p = root / "stage62_clotho_intermediate_eval" / "stage62_clotho_intermediate_eval_results.json"
        if not p.exists():
            continue
        obj = load_json(p)
        if cache_meta is None:
            cache_meta = obj.get("clotho_cache", {})
        for cond, by_m in obj.get("raw", {}).items():
            merged.setdefault(cond, {})
            for m_key, methods in by_m.items():
                merged[cond].setdefault(m_key, {})
                for method, rows in methods.items():
                    cur = merged[cond][m_key].get(method, [])
                    merged[cond][m_key][method] = _merge_seed_rows(cur, rows)

    stats: dict[str, Any] = {}
    for cond, by_m in merged.items():
        stats[cond] = {}
        for m_key, methods in by_m.items():
            if not methods:
                continue
            baseline = str(cfg.get("baseline_method", next(iter(methods))))
            stats[cond][m_key] = build_metric_report(
                methods,
                metrics=["clotho_at_avg_R"],
                baseline_method=baseline if baseline in methods else next(iter(methods)),
            )

    out = {
        "stage": stage_name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage62_shard_roots": [str(p) for p in shard_roots],
        "clotho_cache": cache_meta or {},
        "raw": merged,
        "stats": stats,
        "elapsed_sec": float(time.time() - start),
    }
    save_json(out, output_root / "stage63_clotho_intermediate_aggregate.json")

    lines = [
        "# Stage63 Clotho Intermediate Aggregate",
        "",
        f"- Generated: `{out['generated_at']}`",
        f"- Shards scanned: `{len(shard_roots)}`",
        "",
    ]
    for cond in sorted(stats.keys()):
        lines.append(f"## {cond}")
        by_m = stats[cond]
        for m_key in sorted(by_m.keys(), key=lambda x: int(str(x).lstrip('m'))):
            methods = by_m[m_key].get("methods", {})
            for method, row in methods.items():
                lines.append(f"- {m_key} {method}: clotho_at={row['clotho_at_avg_R']['mean']:.4f}")
        lines.append("")

    (output_root / "stage63_clotho_intermediate_aggregate.md").write_text("\n".join(lines), encoding="utf-8")
    mark_done(markers / f"{stage_name}.done.json", {"elapsed_sec": float(time.time() - start)})
    print(f"{stage_name} complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml

from ..common import env_snapshot, load_json, mark_done, save_json
from ..eval.stats import build_metric_report


def _row_key(r: dict[str, Any]) -> tuple[str, int, str, int]:
    return (str(r["source_id"]), int(r["embed_dim"]), str(r["method"]), int(r["seed"]))


def _metric_keys(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    keys = []
    for k in rows[0].keys():
        if k.endswith("_cat_p1") or k.endswith("_cat_p5") or k.endswith("_cat_p10"):
            keys.append(k)
        elif k.endswith("_cat_hit1") or k.endswith("_cat_hit5") or k.endswith("_cat_hit10"):
            keys.append(k)
        elif k.startswith("avg_cat_p") or k.startswith("avg_cat_hit"):
            keys.append(k)
    # keep deterministic order
    return sorted(set(keys))


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage35_semantic_topk_aggregate"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    shard_roots = [Path(p).resolve() for p in cfg.get("stage34_shard_roots", [])]
    rows_by_key: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    categories = []
    topk_values = []
    for root in shard_roots:
        p = root / "stage34_semantic_topk" / "stage34_semantic_topk_results.json"
        if not p.exists():
            print(f"[stage35] skip missing shard: {p}")
            continue
        obj = load_json(p)
        categories = categories or list(obj.get("category_prompts", []))
        topk_values = topk_values or list(obj.get("topk_values", []))
        for r in obj.get("rows", []):
            rows_by_key[_row_key(r)] = r

    rows = [rows_by_key[k] for k in sorted(rows_by_key)]
    metrics = _metric_keys(rows)

    raw: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for rec in rows:
        src = rec["source_id"]
        m_key = f"m{int(rec['embed_dim'])}"
        raw.setdefault(src, {}).setdefault(m_key, {}).setdefault(rec["method"], []).append(rec)

    baseline_method = str(cfg.get("baseline_method", "modular_shared_jl"))
    stats: dict[str, Any] = {}
    for src, by_m in raw.items():
        stats[src] = {}
        for m_key, methods in by_m.items():
            b = baseline_method if baseline_method in methods else next(iter(methods))
            stats[src][m_key] = build_metric_report(methods, metrics=metrics, baseline_method=b)
            stats[src][m_key]["baseline_method"] = b

    out = {
        "stage": "stage35_semantic_topk_aggregate",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": rows,
        "raw": raw,
        "stats": stats,
        "category_prompts": categories,
        "topk_values": topk_values,
        "elapsed_sec": time.time() - start,
    }
    out_path = stage_root / "stage35_semantic_topk_aggregate.json"
    save_json(out, out_path)

    md = [
        "# Stage35 Semantic Top-k Aggregate",
        "",
        f"- Rows merged: `{len(rows)}`",
        f"- Categories: `{len(categories)}`",
        f"- Top-k values: `{topk_values}`",
        "",
    ]
    for src in sorted(stats.keys()):
        md.append(f"## {src}")
        for m_key in sorted(stats[src].keys(), key=lambda x: int(x[1:])):
            md.append(f"### {m_key}")
            for method, blk in sorted(stats[src][m_key].get("methods", {}).items()):
                p1 = blk.get("avg_cat_p1", {})
                p5 = blk.get("avg_cat_p5", {})
                h10 = blk.get("avg_cat_hit10", {})
                md.append(
                    f"- {method}: "
                    f"avg_cat_p1={p1.get('mean', 0.0):.4f}±{p1.get('std', 0.0):.4f}, "
                    f"avg_cat_p5={p5.get('mean', 0.0):.4f}±{p5.get('std', 0.0):.4f}, "
                    f"avg_cat_hit10={h10.get('mean', 0.0):.4f}±{h10.get('std', 0.0):.4f}"
                )
            md.append("")
    (stage_root / "stage35_semantic_topk_aggregate.md").write_text("\n".join(md), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={"stage": "stage35_semantic_topk_aggregate", "elapsed_sec": time.time() - start},
    )
    save_json(provenance, stage_root / "provenance_stage35.json")
    mark_done(markers / "stage35_semantic_topk_aggregate.done.json", {"elapsed_sec": time.time() - start})
    print("Stage35 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml

from ..common import load_json, mark_done, save_json
from ..eval.stats import build_metric_report, mean_std_ci, paired_ttest


METRICS = ["combined_avg_R", "coco_avg_R", "av_it_avg_R", "av_at_avg_R", "av_ia_avg_R"]


def _merge_seed_rows(dst_rows: list[dict[str, Any]], src_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed = {int(r["seed"]): r for r in dst_rows if "seed" in r}
    for r in src_rows:
        by_seed[int(r["seed"])] = r
    return [by_seed[s] for s in sorted(by_seed)]


def _merge_stage20_from_roots(roots: list[Path]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    merged: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for root in roots:
        p = root / "stage20_modular_audio_transitivity" / "stage20_results.json"
        if not p.exists():
            continue
        obj = load_json(p)
        if obj.get("skipped"):
            continue
        for m_key, methods in obj.get("raw", {}).items():
            merged.setdefault(m_key, {})
            for method, rows in methods.items():
                cur = merged[m_key].get(method, [])
                merged[m_key][method] = _merge_seed_rows(cur, rows)
    return merged


def _stats_for_raw(raw: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m_key, methods in raw.items():
        if not methods:
            continue
        baseline = "joint_shared_jl" if "joint_shared_jl" in methods else ("modular_shared_jl" if "modular_shared_jl" in methods else next(iter(methods)))
        out[m_key] = build_metric_report(methods, metrics=METRICS, baseline_method=baseline)
    return out


def _paired_delta(
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    a_by_seed = {int(r["seed"]): r for r in a_rows if metric in r and "seed" in r}
    b_by_seed = {int(r["seed"]): r for r in b_rows if metric in r and "seed" in r}
    common = sorted(set(a_by_seed) & set(b_by_seed))
    if not common:
        return {"n": 0}
    a_vals = [float(a_by_seed[s][metric]) for s in common]
    b_vals = [float(b_by_seed[s][metric]) for s in common]
    deltas = [a - b for a, b in zip(a_vals, b_vals)]
    ds = mean_std_ci(deltas)
    tt = paired_ttest(a_vals, b_vals)
    return {
        "n": len(common),
        "seeds": common,
        "delta_mean": ds["mean"],
        "delta_std": ds["std"],
        "delta_ci95_low": ds["ci95_low"],
        "delta_ci95_high": ds["ci95_high"],
        "t_stat": tt["t_stat"],
        "p_value": tt["p_value"],
    }


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage22_roots = [Path(p).resolve() for p in cfg.get("stage22_shard_roots", [])]
    stage23_roots = [Path(p).resolve() for p in cfg.get("stage23_shard_roots", [])]

    stage22_raw = _merge_stage20_from_roots(stage22_roots)
    stage23_raw = _merge_stage20_from_roots(stage23_roots)

    stage22_stats = _stats_for_raw(stage22_raw)
    stage23_stats = _stats_for_raw(stage23_raw)

    prior_obj = None
    prior_raw: dict[str, dict[str, list[dict[str, Any]]]] = {}
    prior_path = cfg.get("prior_stage21_aggregate")
    if prior_path:
        p = Path(prior_path).resolve()
        if p.exists():
            prior_obj = load_json(p)
            prior_raw = prior_obj.get("stage20", {}).get("raw", {})

    comparisons: dict[str, Any] = {
        "stage22_separate_vs_shared": {},
        "stage23_fused_vs_unfused": {},
    }

    # Mechanism check inside stage22 (same encoder): separate vs shared.
    for m_key, methods in stage22_raw.items():
        if "modular_separate_jl" not in methods or "modular_shared_jl" not in methods:
            continue
        comparisons["stage22_separate_vs_shared"][m_key] = {
            metric: _paired_delta(methods["modular_separate_jl"], methods["modular_shared_jl"], metric)
            for metric in ["av_ia_avg_R", "combined_avg_R"]
        }

    # Fused vs unfused, method-matched and seed-paired against prior Stage21 baseline suite.
    if prior_raw:
        for m_key, methods in stage23_raw.items():
            if m_key not in prior_raw:
                continue
            comparisons["stage23_fused_vs_unfused"].setdefault(m_key, {})
            for method_name, rows_fused in methods.items():
                rows_unfused = prior_raw[m_key].get(method_name)
                if not rows_unfused:
                    continue
                comparisons["stage23_fused_vs_unfused"][m_key][method_name] = {
                    metric: _paired_delta(rows_fused, rows_unfused, metric)
                    for metric in ["av_ia_avg_R", "combined_avg_R", "av_at_avg_R"]
                }

    out = {
        "stage": "stage24_modtrans_followup_aggregate",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage22": {
            "shards": [str(p) for p in stage22_roots],
            "raw": stage22_raw,
            "stats": stage22_stats,
        },
        "stage23": {
            "shards": [str(p) for p in stage23_roots],
            "raw": stage23_raw,
            "stats": stage23_stats,
        },
        "prior_stage21_aggregate": str(Path(prior_path).resolve()) if prior_path else None,
        "comparisons": comparisons,
        "elapsed_sec": time.time() - start,
    }

    save_json(out, output_root / "stage24_followup_aggregate.json")

    md = [
        "# Stage 24 Modular Transitivity Follow-up Aggregate",
        "",
        f"- Generated: `{out['generated_at']}`",
        f"- Stage22 shards: `{len(stage22_roots)}`",
        f"- Stage23 shards: `{len(stage23_roots)}`",
        "",
    ]

    def _emit_table(title: str, stats_obj: dict[str, Any]) -> None:
        md.append(f"## {title}")
        for m_key in sorted(stats_obj.keys(), key=lambda x: int(x[1:])):
            md.append(f"### {m_key}")
            methods = stats_obj[m_key].get("methods", {})
            for method_name in ["modular_shared_jl", "modular_separate_jl", "joint_shared_jl", "joint_clip_head", "raw_cosine_baseline"]:
                if method_name not in methods:
                    continue
                row = methods[method_name]
                md.append(
                    f"- {method_name}: combined={row['combined_avg_R']['mean']:.4f}, "
                    f"av_ia={row['av_ia_avg_R']['mean']:.4f}, av_at={row['av_at_avg_R']['mean']:.4f}, coco={row['coco_avg_R']['mean']:.4f}"
                )
            md.append("")

    _emit_table("Stage22 (Unfused Mechanism Lock)", stage22_stats)
    _emit_table("Stage23 (Fused Audio Swap)", stage23_stats)

    md.append("## Key Paired Deltas")
    for m_key, comp in comparisons.get("stage22_separate_vs_shared", {}).items():
        ia = comp.get("av_ia_avg_R", {"n": 0})
        if ia.get("n", 0) > 0:
            md.append(
                f"- Stage22 {m_key} separate-shared (av_ia): Δ={ia['delta_mean']:.4f} "
                f"CI[{ia['delta_ci95_low']:.4f},{ia['delta_ci95_high']:.4f}] p={ia['p_value']:.4g}"
            )
    for m_key, methods in comparisons.get("stage23_fused_vs_unfused", {}).items():
        for method_name, metrics in methods.items():
            ia = metrics.get("av_ia_avg_R", {"n": 0})
            if ia.get("n", 0) > 0:
                md.append(
                    f"- Stage23 {m_key} {method_name} fused-unfused (av_ia): Δ={ia['delta_mean']:.4f} "
                    f"CI[{ia['delta_ci95_low']:.4f},{ia['delta_ci95_high']:.4f}] p={ia['p_value']:.4g}"
                )

    (output_root / "stage24_followup_aggregate.md").write_text("\n".join(md), encoding="utf-8")
    mark_done(markers / "stage24_modtrans_followup_aggregate.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 24 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

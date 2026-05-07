from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml

from ..common import load_json, mark_done, save_json
from ..eval.stats import build_metric_report, holm_bonferroni, mean_std_ci, paired_ttest


METRICS = ["combined_avg_R", "coco_avg_R", "av_it_avg_R", "av_at_avg_R", "av_ia_avg_R"]
METHOD_SHARED = "modular_shared_jl"
METHOD_SEPARATE = "modular_separate_jl"
METHOD_HYBRID_IT = "modular_hybrid_it_jl"
METHOD_HYBRID_AT = "modular_hybrid_at_jl"


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
        baseline = METHOD_SHARED if METHOD_SHARED in methods else next(iter(methods))
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
        return {"n": 0, "metric": metric}
    a_vals = [float(a_by_seed[s][metric]) for s in common]
    b_vals = [float(b_by_seed[s][metric]) for s in common]
    deltas = [a - b for a, b in zip(a_vals, b_vals)]
    ds = mean_std_ci(deltas)
    tt = paired_ttest(a_vals, b_vals)
    return {
        "n": len(common),
        "seeds": common,
        "metric": metric,
        "delta_mean": ds["mean"],
        "delta_std": ds["std"],
        "delta_ci95_low": ds["ci95_low"],
        "delta_ci95_high": ds["ci95_high"],
        "t_stat": tt["t_stat"],
        "p_value": tt["p_value"],
    }


def _factorial_effect(
    methods: dict[str, list[dict[str, Any]]],
    metric: str,
    effect: str,
) -> dict[str, Any]:
    needed = [METHOD_SHARED, METHOD_SEPARATE, METHOD_HYBRID_IT, METHOD_HYBRID_AT]
    if not all(name in methods for name in needed):
        return {"n": 0, "metric": metric, "effect": effect}

    by_seed = {
        name: {int(r["seed"]): r for r in methods[name] if metric in r and "seed" in r}
        for name in needed
    }
    common = sorted(set.intersection(*(set(by_seed[name]) for name in needed)))
    if not common:
        return {"n": 0, "metric": metric, "effect": effect}

    vals = []
    for s in common:
        sh = float(by_seed[METHOD_SHARED][s][metric])
        sp = float(by_seed[METHOD_SEPARATE][s][metric])
        it = float(by_seed[METHOD_HYBRID_IT][s][metric])
        at = float(by_seed[METHOD_HYBRID_AT][s][metric])

        if effect == "phase_a_main":
            v = ((sh + it) / 2.0) - ((at + sp) / 2.0)
        elif effect == "phase_b_main":
            v = ((sh + at) / 2.0) - ((it + sp) / 2.0)
        elif effect == "interaction":
            v = (sh - it) - (at - sp)
        else:
            raise ValueError(f"unknown factorial effect {effect}")
        vals.append(v)

    ds = mean_std_ci(vals)
    zeros = [0.0] * len(vals)
    tt = paired_ttest(vals, zeros)
    return {
        "n": len(common),
        "seeds": common,
        "metric": metric,
        "effect": effect,
        "mean": ds["mean"],
        "std": ds["std"],
        "ci95_low": ds["ci95_low"],
        "ci95_high": ds["ci95_high"],
        "t_stat": tt["t_stat"],
        "p_value": tt["p_value"],
    }


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage25_roots = [Path(p).resolve() for p in cfg.get("stage25_shard_roots", [])]
    raw = _merge_stage20_from_roots(stage25_roots)
    stats = _stats_for_raw(raw)

    pair_defs = {
        "shared_vs_separate": (METHOD_SHARED, METHOD_SEPARATE),
        "shared_vs_hybrid_it": (METHOD_SHARED, METHOD_HYBRID_IT),
        "shared_vs_hybrid_at": (METHOD_SHARED, METHOD_HYBRID_AT),
        "hybrid_it_vs_hybrid_at": (METHOD_HYBRID_IT, METHOD_HYBRID_AT),
        "hybrid_it_vs_separate": (METHOD_HYBRID_IT, METHOD_SEPARATE),
        "hybrid_at_vs_separate": (METHOD_HYBRID_AT, METHOD_SEPARATE),
    }

    pairwise: dict[str, Any] = {}
    factorial: dict[str, Any] = {}

    for m_key, methods in raw.items():
        pairwise[m_key] = {}
        factorial[m_key] = {}
        for metric in METRICS:
            pairwise[m_key][metric] = {}
            pvals: dict[str, float] = {}
            for label, (a_name, b_name) in pair_defs.items():
                if a_name not in methods or b_name not in methods:
                    continue
                row = _paired_delta(methods[a_name], methods[b_name], metric)
                pairwise[m_key][metric][label] = row
                if row.get("n", 0) > 0:
                    pvals[label] = float(row["p_value"])
            holm = holm_bonferroni(pvals) if pvals else {}
            for label, hrow in holm.items():
                pairwise[m_key][metric][label]["p_holm"] = hrow["p_holm"]
                pairwise[m_key][metric][label]["reject_h0"] = hrow["reject_h0"]
                pairwise[m_key][metric][label]["holm_threshold"] = hrow["threshold"]

            effects = ["phase_a_main", "phase_b_main", "interaction"]
            factorial[m_key][metric] = {}
            eff_pvals: dict[str, float] = {}
            for effect in effects:
                row = _factorial_effect(methods, metric, effect)
                factorial[m_key][metric][effect] = row
                if row.get("n", 0) > 0:
                    eff_pvals[effect] = float(row["p_value"])
            eff_holm = holm_bonferroni(eff_pvals) if eff_pvals else {}
            for effect, hrow in eff_holm.items():
                factorial[m_key][metric][effect]["p_holm"] = hrow["p_holm"]
                factorial[m_key][metric][effect]["reject_h0"] = hrow["reject_h0"]
                factorial[m_key][metric][effect]["holm_threshold"] = hrow["threshold"]

    out = {
        "stage": "stage26_modtrans_jl_aggregate",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage25": {
            "shards": [str(p) for p in stage25_roots],
            "raw": raw,
            "stats": stats,
        },
        "comparisons": {
            "pairwise": pairwise,
            "factorial_effects": factorial,
        },
        "elapsed_sec": time.time() - start,
    }
    save_json(out, output_root / "stage26_jlablation_aggregate.json")

    md = [
        "# Stage 26 JL Mechanism Aggregate",
        "",
        f"- Generated: `{out['generated_at']}`",
        f"- Stage25 shards: `{len(stage25_roots)}`",
        "",
    ]

    for m_key in sorted(stats.keys(), key=lambda x: int(x[1:])):
        md.append(f"## {m_key}")
        methods = stats[m_key].get("methods", {})
        for method_name in [METHOD_SHARED, METHOD_SEPARATE, METHOD_HYBRID_IT, METHOD_HYBRID_AT]:
            if method_name not in methods:
                continue
            row = methods[method_name]
            md.append(
                f"- {method_name}: combined={row['combined_avg_R']['mean']:.4f}, "
                f"coco={row['coco_avg_R']['mean']:.4f}, av_ia={row['av_ia_avg_R']['mean']:.4f}, "
                f"av_at={row['av_at_avg_R']['mean']:.4f}"
            )
        p = pairwise.get(m_key, {}).get("av_ia_avg_R", {}).get("hybrid_it_vs_hybrid_at", {"n": 0})
        if p.get("n", 0) > 0:
            md.append(
                f"- hybrid_it_vs_hybrid_at (av_ia): Δ={p['delta_mean']:.4f} "
                f"CI[{p['delta_ci95_low']:.4f},{p['delta_ci95_high']:.4f}] "
                f"p={p['p_value']:.4g} p_holm={p.get('p_holm', 1.0):.4g}"
            )
        f = factorial.get(m_key, {}).get("av_ia_avg_R", {}).get("phase_b_main", {"n": 0})
        if f.get("n", 0) > 0:
            md.append(
                f"- phase_b_main (av_ia): mean={f['mean']:.4f} "
                f"CI[{f['ci95_low']:.4f},{f['ci95_high']:.4f}] "
                f"p={f['p_value']:.4g} p_holm={f.get('p_holm', 1.0):.4g}"
            )
        md.append("")

    (output_root / "stage26_jlablation_aggregate.md").write_text("\n".join(md), encoding="utf-8")
    mark_done(markers / "stage26_modtrans_jl_aggregate.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 26 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

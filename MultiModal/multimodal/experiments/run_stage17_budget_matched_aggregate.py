from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..common import load_json, mark_done, save_json
from ..eval.stats import build_metric_report, holm_bonferroni, mean_std_ci, paired_ttest
from .run_stage16_budget_matched_frontier import DEFAULT_METRICS


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _coverage_check(raw: dict, cfg: dict) -> dict:
    partitions = list(cfg.get("expected_partitions", []))
    methods = list(cfg.get("expected_methods", []))
    seeds = [int(s) for s in cfg.get("expected_seeds", [])]
    expected_draws = int(cfg.get("expected_draws", 5))

    missing = []
    for part in partitions:
        part_obj = raw.get(part, {})
        for method in methods:
            rows = part_obj.get(method, [])
            by_seed = {int(r["seed"]): r for r in rows if "seed" in r}
            for seed in seeds:
                if seed not in by_seed:
                    missing.append({"partition": part, "method": method, "seed": seed, "reason": "missing_seed"})
                    continue
                draw_rows = by_seed[seed].get("draw_metrics", [])
                if len(draw_rows) != expected_draws:
                    missing.append(
                        {
                            "partition": part,
                            "method": method,
                            "seed": seed,
                            "reason": "draw_count_mismatch",
                            "found_draws": len(draw_rows),
                            "expected_draws": expected_draws,
                        }
                    )
    return {
        "expected_partitions": partitions,
        "expected_methods": methods,
        "expected_seeds": seeds,
        "expected_draws": expected_draws,
        "missing_records": missing,
        "ok": len(missing) == 0,
    }


def _global_method_rows(raw: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    part_names = sorted(raw.keys())
    for p_idx, part in enumerate(part_names):
        for method, rows in raw.get(part, {}).items():
            out.setdefault(method, [])
            for r in rows:
                rr = dict(r)
                rr["seed"] = int(p_idx * 1000 + int(r["seed"]))  # unique paired key
                rr["partition"] = part
                out[method].append(rr)
    return out


def _paired_global_comparisons(per_method: dict[str, list[dict]], metrics: list[str], baseline: str) -> dict:
    base = per_method.get(baseline, [])
    base_by_key = {(r["partition"], int(r["seed"]) % 1000): r for r in base}
    out = {}
    for metric in metrics:
        pvals = {}
        comp = {}
        for method, rows in per_method.items():
            if method == baseline:
                continue
            meth_by_key = {(r["partition"], int(r["seed"]) % 1000): r for r in rows}
            common = sorted(set(base_by_key.keys()) & set(meth_by_key.keys()))
            if not common:
                continue
            b = [float(base_by_key[k][metric]) for k in common if metric in base_by_key[k]]
            m = [float(meth_by_key[k][metric]) for k in common if metric in meth_by_key[k]]
            if len(b) != len(common) or len(m) != len(common):
                continue
            deltas = [mm - bb for mm, bb in zip(m, b)]
            dsum = mean_std_ci(deltas)
            tt = paired_ttest(m, b)
            pvals[method] = float(tt["p_value"])
            comp[method] = {
                "n_paired": len(common),
                "delta_mean": dsum["mean"],
                "delta_std": dsum["std"],
                "delta_ci95_low": dsum["ci95_low"],
                "delta_ci95_high": dsum["ci95_high"],
                "t_stat": tt["t_stat"],
                "p_value": tt["p_value"],
            }
        holm = holm_bonferroni(pvals) if pvals else {}
        for method, h in holm.items():
            if method in comp:
                comp[method]["p_holm"] = h["p_holm"]
                comp[method]["reject_h0"] = h["reject_h0"]
                comp[method]["holm_threshold"] = h["threshold"]
        out[metric] = comp
    return out


def run(cfg: dict) -> None:
    output_root = Path(cfg["output_root"]).resolve()
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage16_file = output_root / "stage16_budget_matched" / cfg.get("stage16_results_name", "E16_budget_matched_results.json")
    if not stage16_file.exists():
        raise FileNotFoundError(f"missing merged stage16 file: {stage16_file}")
    s16 = load_json(stage16_file)

    raw = s16.get("raw", {})
    coverage = _coverage_check(raw, cfg)

    partition_rows = []
    for part, methods in raw.items():
        for method, rows in methods.items():
            if not rows:
                continue
            partition_rows.append(
                {
                    "partition": part,
                    "method": method,
                    "avg_R_mean": _mean(rows, "avg_R"),
                    "mlp_rel_error_mean": _mean(rows, "mlp_rel_error"),
                    "linear_rel_error_mean": _mean(rows, "linear_rel_error"),
                    "iterative_rel_error_mean": _mean(rows, "iterative_rel_error"),
                    "comm_mb_model_update_mean": _mean(rows, "comm_mb_model_update"),
                    "embedding_bytes_per_vector": int(rows[0].get("embedding_bytes_per_vector", 0)),
                    "embedding_bytes_per_pair": int(rows[0].get("embedding_bytes_per_pair", 0)),
                    "n_seeds": len(rows),
                }
            )
    partition_rows = sorted(partition_rows, key=lambda r: (r["partition"], r["avg_R_mean"]), reverse=False)

    metrics = list(cfg.get("metrics_for_stats", DEFAULT_METRICS))
    baseline = cfg.get("baseline_method", "mask_concat")
    global_rows = _global_method_rows(raw)
    global_stats = build_metric_report(global_rows, metrics=metrics, baseline_method=baseline)
    global_pairs = _paired_global_comparisons(global_rows, metrics=metrics, baseline=baseline)

    frontier_rows = []
    for method, m in global_stats.get("methods", {}).items():
        frontier_rows.append(
            {
                "method": method,
                "avg_R": m.get("avg_R", {}),
                "mlp_rel_error": m.get("mlp_rel_error", {}),
                "linear_rel_error": m.get("linear_rel_error", {}),
                "iterative_rel_error": m.get("iterative_rel_error", {}),
                "comm_mb_model_update": m.get("comm_mb_model_update", {}),
                "embedding_bytes_per_vector": m.get("embedding_bytes_per_vector", {}),
                "embedding_bytes_per_pair": m.get("embedding_bytes_per_pair", {}),
            }
        )
    frontier_rows = sorted(frontier_rows, key=lambda r: r["avg_R"].get("mean", float("-inf")), reverse=True)

    summary = {
        "stage": "stage17_budget_matched_aggregate",
        "config": cfg,
        "coverage": coverage,
        "partition_rows": partition_rows,
        "global_stats": global_stats,
        "global_paired_comparisons": global_pairs,
        "frontier_rows": frontier_rows,
        "communication_notes": [
            "comm_mb_model_update reflects FL model update traffic from Stage13.",
            "embedding_bytes_per_vector reflects fixed budget embedding channel traffic in Stage16.",
            "These channels are reported separately by design.",
        ],
    }

    stage17_json = output_root / "stage17_budget_matched_aggregate.json"
    save_json(summary, stage17_json)

    md = [
        "# Budget-Matched Privacy Frontier",
        "",
        "## Coverage",
        "",
        f"- coverage_ok: `{coverage['ok']}`",
        f"- expected_partitions: `{coverage['expected_partitions']}`",
        f"- expected_methods: `{coverage['expected_methods']}`",
        f"- expected_seeds: `{coverage['expected_seeds']}`",
        f"- expected_draws: `{coverage['expected_draws']}`",
        f"- missing_records: `{len(coverage['missing_records'])}`",
        "",
        "## Partition Summary",
        "",
        "| partition | method | avg_R | MLP err | Iter err | model-update MB | embed bytes/vector | seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in partition_rows:
        md.append(
            f"| {r['partition']} | {r['method']} | {r['avg_R_mean']:.4f} | {r['mlp_rel_error_mean']:.4f} | "
            f"{r['iterative_rel_error_mean']:.4f} | {r['comm_mb_model_update_mean']:.2f} | "
            f"{r['embedding_bytes_per_vector']} | {r['n_seeds']} |"
        )

    md.extend(
        [
            "",
            "## Global Frontier",
            "",
            "| method | avg_R mean ± std | MLP err mean ± std | Iter err mean ± std | model-update MB mean | embed bytes/vector mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in frontier_rows:
        a = r["avg_R"]
        m = r["mlp_rel_error"]
        it = r["iterative_rel_error"]
        c = r["comm_mb_model_update"]
        eb = r["embedding_bytes_per_vector"]
        md.append(
            f"| {r['method']} | {a.get('mean', float('nan')):.4f} ± {a.get('std', 0.0):.4f} | "
            f"{m.get('mean', float('nan')):.4f} ± {m.get('std', 0.0):.4f} | "
            f"{it.get('mean', float('nan')):.4f} ± {it.get('std', 0.0):.4f} | "
            f"{c.get('mean', float('nan')):.2f} | {eb.get('mean', float('nan')):.1f} |"
        )

    (output_root / "stage17_budget_matched_aggregate.md").write_text("\n".join(md), encoding="utf-8")
    mark_done(markers / "stage17_budget_matched_aggregate.done.json")
    print("Stage 17 (budget-matched aggregate) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

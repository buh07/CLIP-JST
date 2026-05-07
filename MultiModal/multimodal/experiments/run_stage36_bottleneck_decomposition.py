"""Stage 36: Bottleneck Decomposition Analysis.

Tests the hypothesis that av_ia ≈ alpha * geometric_mean(av_it_ood, av_at) across
all conditions, quantifying how much of the zero-shot image-audio gap is explained
by the product of the two independent bridge components.

Reads eval.json files from multiple stage roots (Stage25/26, Stage29, Stage30,
Stage31, Stage32) and fits a single proportionality constant alpha.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import yaml

from ..common import env_snapshot, load_json, mark_done, save_json


def _geometric_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return math.sqrt(a * b)


def _collect_records(source_specs: list[dict]) -> list[dict[str, Any]]:
    """Walk stage result files and extract per-(method,dim,seed) metric triples."""
    records: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    for spec in source_specs:
        source_id = str(spec["source_id"])
        stage_root = Path(spec["stage_root"]).resolve()
        results_file = str(spec.get("results_file", f"{stage_root.name}_results.json"))
        results_path = stage_root / results_file

        if not results_path.exists():
            print(f"[stage36] skip missing: {results_path}")
            continue

        obj = load_json(results_path)
        raw = obj.get("raw", {})

        for m_key, methods in raw.items():
            try:
                m = int(str(m_key).lstrip("m"))
            except ValueError:
                continue

            for method, recs in methods.items():
                if not isinstance(recs, list):
                    continue
                for rec in recs:
                    seed = int(rec.get("seed", -1))
                    key = (source_id, m, method, seed)
                    if key in seen:
                        continue
                    seen.add(key)

                    # av_image_text = OOD image-text (CLIP head evaluated on AudioCaps thumbnails)
                    av_it = rec.get("av_image_text", {})
                    av_at = rec.get("av_audio_text", {})
                    av_ia = rec.get("av_image_audio", {})

                    # Support both nested dicts (from Stage20) and flat floats
                    def _get_avg(d):
                        if isinstance(d, dict):
                            return float(d.get("avg_R", d.get("mean", 0.0)))
                        return float(d) if d else 0.0

                    av_it_val = _get_avg(av_it)
                    av_at_val = _get_avg(av_at)
                    av_ia_val = _get_avg(av_ia)

                    # Also support top-level float keys (Stage30/31 format)
                    if av_it_val == 0.0:
                        av_it_val = float(rec.get("av_it_avg_R", 0.0))
                    if av_at_val == 0.0:
                        av_at_val = float(rec.get("av_at_avg_R", 0.0))
                    if av_ia_val == 0.0:
                        av_ia_val = float(rec.get("av_ia_avg_R", 0.0))

                    ceiling = _geometric_mean(av_it_val, av_at_val)

                    records.append({
                        "source_id": source_id,
                        "method": method,
                        "embed_dim": m,
                        "seed": seed,
                        "av_it_ood": av_it_val,
                        "av_at": av_at_val,
                        "av_ia": av_ia_val,
                        "ceiling": ceiling,
                    })

    return records


def _fit_alpha(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit av_ia = alpha * ceiling (no intercept) by least squares."""
    xs = [r["ceiling"] for r in records if r["ceiling"] > 0]
    ys = [r["av_ia"] for r in records if r["ceiling"] > 0]

    if not xs:
        return {"alpha": None, "r2": None, "n": 0}

    n = len(xs)
    numerator = sum(x * y for x, y in zip(xs, ys))
    denominator = sum(x * x for x in xs)
    alpha = numerator / denominator if denominator > 0 else 0.0

    # R² relative to the no-intercept model (total SS from zero)
    ss_res = sum((y - alpha * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum(y * y for y in ys)
    r2_nointercept = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Pearson r between ceiling and av_ia
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    pearson_r = cov / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else 0.0

    residuals = [y - alpha * x for x, y in zip(xs, ys)]
    abs_res = [abs(r) for r in residuals]
    mae = sum(abs_res) / n
    max_abs = max(abs_res)

    return {
        "alpha": alpha,
        "r2_nointercept": r2_nointercept,
        "pearson_r": pearson_r,
        "pearson_r2": pearson_r ** 2,
        "mae": mae,
        "max_abs_residual": max_abs,
        "n": n,
    }


def _fit_alpha_by_group(records: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    """Fit alpha separately for each value of group_key."""
    from collections import defaultdict
    groups: dict[Any, list[dict]] = defaultdict(list)
    for r in records:
        groups[r[group_key]].append(r)
    return {str(k): _fit_alpha(v) for k, v in sorted(groups.items())}


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage36_bottleneck_decomposition"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    source_specs = cfg.get("source_experiments", [])
    records = _collect_records(source_specs)
    print(f"[stage36] collected {len(records)} records from {len(source_specs)} sources")

    if not records:
        raise RuntimeError("No records found. Check source_experiments in config.")

    # Global fit
    global_fit = _fit_alpha(records)
    print(
        f"[stage36] global alpha={global_fit['alpha']:.4f}  "
        f"pearson_r={global_fit['pearson_r']:.4f}  "
        f"r2_nointercept={global_fit['r2_nointercept']:.4f}  "
        f"n={global_fit['n']}"
    )

    # Per-dim fit
    by_dim = _fit_alpha_by_group(records, "embed_dim")
    print("[stage36] per-dim alpha:")
    for dim, fit in sorted(by_dim.items(), key=lambda x: int(x[0])):
        print(f"  m={dim}: alpha={fit['alpha']:.4f}  pearson_r={fit['pearson_r']:.4f}  n={fit['n']}")

    # Per-source fit
    by_source = _fit_alpha_by_group(records, "source_id")
    print("[stage36] per-source alpha:")
    for src, fit in sorted(by_source.items()):
        print(f"  {src}: alpha={fit['alpha']:.4f}  pearson_r={fit['pearson_r']:.4f}  n={fit['n']}")

    # Per-method fit
    by_method = _fit_alpha_by_group(records, "method")
    print("[stage36] per-method alpha:")
    for method, fit in sorted(by_method.items()):
        print(f"  {method}: alpha={fit['alpha']:.4f}  pearson_r={fit['pearson_r']:.4f}  n={fit['n']}")

    # Annotate records with predicted values and residuals
    alpha = global_fit["alpha"] or 0.0
    for r in records:
        r["predicted_ia"] = alpha * r["ceiling"]
        r["residual"] = r["av_ia"] - r["predicted_ia"]
        r["relative_residual"] = r["residual"] / r["av_ia"] if r["av_ia"] > 0 else 0.0

    out = {
        "stage": "stage36_bottleneck_decomposition",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hypothesis": "av_ia = alpha * geometric_mean(av_it_ood, av_at)",
        "global_fit": global_fit,
        "by_embed_dim": by_dim,
        "by_source": by_source,
        "by_method": by_method,
        "records": records,
        "elapsed_sec": time.time() - start,
    }

    out_path = stage_root / "stage36_bottleneck_decomposition.json"
    save_json(out, out_path)

    # Write markdown summary
    md = [
        "# Stage 36: Bottleneck Decomposition",
        "",
        "**Hypothesis**: `av_ia ≈ alpha × geometric_mean(av_it_ood, av_at)`",
        "",
        f"## Global Fit (n={global_fit['n']})",
        f"- alpha = {global_fit['alpha']:.4f}",
        f"- Pearson r = {global_fit['pearson_r']:.4f} (r² = {global_fit['pearson_r2']:.4f})",
        f"- R² (no-intercept) = {global_fit['r2_nointercept']:.4f}",
        f"- MAE = {global_fit['mae']:.5f}",
        f"- Max |residual| = {global_fit['max_abs_residual']:.5f}",
        "",
        "## Per Embedding Dimension",
        "| m | alpha | Pearson r | n |",
        "|---|---|---|---|",
    ]
    for dim, fit in sorted(by_dim.items(), key=lambda x: int(x[0])):
        md.append(f"| {dim} | {fit['alpha']:.4f} | {fit['pearson_r']:.4f} | {fit['n']} |")

    md += [
        "",
        "## Per Source Experiment",
        "| Source | alpha | Pearson r | n |",
        "|---|---|---|---|",
    ]
    for src, fit in sorted(by_source.items()):
        md.append(f"| {src} | {fit['alpha']:.4f} | {fit['pearson_r']:.4f} | {fit['n']} |")

    md += [
        "",
        "## Per Method",
        "| Method | alpha | Pearson r | n |",
        "|---|---|---|---|",
    ]
    for method, fit in sorted(by_method.items()):
        md.append(f"| {method} | {fit['alpha']:.4f} | {fit['pearson_r']:.4f} | {fit['n']} |")

    md += [
        "",
        "## Interpretation",
        "",
        f"A Pearson r of {global_fit['pearson_r']:.3f} (r²={global_fit['pearson_r2']:.3f}) means that "
        f"{global_fit['pearson_r2']*100:.1f}% of the variance in av_ia is explained by the geometric mean "
        "of the two independent bridge components (OOD image-text quality × audio-text quality). "
        f"The proportionality constant alpha={global_fit['alpha']:.3f} quantifies the transmission "
        "efficiency of the text bridge.",
        "",
        "If r² > 0.85, the bottleneck model is a strong explanation of the observed av_ia values "
        "across conditions, supporting the claim that text anchoring is the causal mechanism.",
    ]

    (stage_root / "stage36_bottleneck_decomposition.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[stage36] wrote results to {stage_root}")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={"stage": "stage36_bottleneck_decomposition", "n_records": len(records)},
    )
    save_json(provenance, stage_root / "provenance_stage36.json")
    mark_done(markers / "stage36_bottleneck_decomposition.done.json", {"elapsed_sec": time.time() - start})
    print("Stage36 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

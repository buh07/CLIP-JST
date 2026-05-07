from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from ..common import env_snapshot, load_json, mark_done, save_json


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage54_sdpi_aggressive_aggregate"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    paths = {
        "stage47": Path(cfg["stage47_results_path"]).resolve(),
        "stage48": Path(cfg["stage48_results_path"]).resolve(),
        "stage49": Path(cfg["stage49_results_path"]).resolve(),
        "stage50": Path(cfg["stage50_results_path"]).resolve(),
        "stage51": Path(cfg["stage51_results_path"]).resolve(),
        "stage52": Path(cfg["stage52_results_path"]).resolve(),
        "stage53": Path(cfg["stage53_results_path"]).resolve(),
    }

    objs = {k: load_json(p) for k, p in paths.items()}

    s47 = objs["stage47"]
    s48 = objs["stage48"]
    s49 = objs["stage49"]
    s50 = objs["stage50"]
    s51 = objs["stage51"]
    s52 = objs["stage52"]
    s53 = objs["stage53"]

    out = {
        "stage": "stage54_sdpi_aggressive_aggregate",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {k: str(v) for k, v in paths.items()},
        "manifest_summary": s47.get("summary", {}),
        "stage48": {
            "n_rows": len(s48.get("rows", [])),
            "n_parity_fail": sum(1 for r in s48.get("rows", []) if not bool(r.get("parity_pass", False))),
        },
        "stage49": {
            "n_rows": len(s49.get("rows", [])),
            "shard_summaries": s49.get("shard_summaries", []),
        },
        "stage50": {
            "n_rows": s50.get("n_rows", 0),
            "by_estimator": s50.get("by_estimator", {}),
        },
        "stage51": {
            "recall_tightness": s51.get("recall_tightness", {}),
            "correlations": s51.get("correlations", {}),
            "mi_tightness": s51.get("mi_tightness", {}),
        },
        "stage52": {
            "quality_intervention": s52.get("quality_intervention", {}),
            "projection_intervention": s52.get("projection_intervention", {}),
            "sharing_intervention": s52.get("sharing_intervention", {}),
        },
        "stage53": {
            "summary": s53.get("summary", {}),
            "n_rows": s53.get("n_rows", 0),
        },
        "elapsed_sec": float(time.time() - start),
    }

    save_json(out, stage_root / "stage54_sdpi_aggressive_aggregate.json")

    b50 = out["stage50"]["by_estimator"]
    c51 = out["stage51"]["correlations"]
    q52 = out["stage52"]["quality_intervention"]
    p52 = out["stage52"]["projection_intervention"]

    lines = [
        "# Stage54 SDPI Aggressive Aggregate",
        "",
        "## Coverage",
        f"- Manifest conditions: {out['manifest_summary'].get('n_conditions', 0)}",
        f"- Stage48 exported rows: {out['stage48']['n_rows']} (parity fails: {out['stage48']['n_parity_fail']})",
        f"- Stage49 MI rows: {out['stage49']['n_rows']}",
        "",
        "## E3: SDPI-shaped Inequality",
        "| estimator | upper-bound violation rate | alpha mean | alpha(unit) mean |",
        "|---|---:|---:|---:|",
    ]
    for est in ["ksg", "cancor", "gaussian"]:
        e = b50.get(est, {})
        lines.append(
            f"| {est} | {e.get('upper_bound_violation_rate', 0.0):.4f} | "
            f"{e.get('alpha_mean', 0.0):.4f} | {e.get('alpha_unit_mean', 0.0):.4f} |"
        )

    lines += [
        "",
        "## E4+E5: Link + Tightness",
        "| estimator | corr(alphaI, alphaR) | corr(alphaI_unit, alphaR) | corr(MIia, Ria) | corr(alphaI, gap) |",
        "|---|---:|---:|---:|---:|",
    ]
    for est in ["ksg", "cancor", "gaussian"]:
        b = c51.get(est, {})
        rg = b.get("alpha_i_vs_gap", {}).get("r", 0.0) if isinstance(b.get("alpha_i_vs_gap"), dict) else 0.0
        lines.append(
            f"| {est} | {b.get('alpha_i_vs_alpha_recall', {}).get('r', 0.0):.4f} | "
            f"{b.get('alpha_i_unit_vs_alpha_recall', {}).get('r', 0.0):.4f} | "
            f"{b.get('mi_ia_vs_recall_ia', {}).get('r', 0.0):.4f} | {rg:.4f} |"
        )

    lines += [
        "",
        "## E6: Intervention Directions",
        f"- WavCaps-AudioCaps delta av_at mean: {q52.get('delta_recall_av_at', {}).get('delta_mean', 0.0):.6f}",
        f"- WavCaps-AudioCaps delta av_ia mean: {q52.get('delta_recall_av_ia', {}).get('delta_mean', 0.0):.6f}",
        f"- LinearProbe-JL delta av_ia mean: {p52.get('delta_recall_ia_mean', 0.0):.6f}",
        f"- LinearProbe-JL delta alphaI_ksg mean: {p52.get('delta_alphaI_ksg_mean', 0.0):.6f}",
        f"- LinearProbe-JL delta alphaI_unit_ksg mean: {p52.get('delta_alphaI_unit_ksg_mean', 0.0):.6f}",
        "",
        "## E7: Constrained Channel Proxy",
        f"- corr(eta_geom, alphaI_ksg): {out['stage53']['summary'].get('corr_eta_vs_alphaI_ksg', 0.0):.4f}",
        f"- corr(eta_geom, alphaI_unit_ksg): {out['stage53']['summary'].get('corr_eta_vs_alphaI_unit_ksg', 0.0):.4f}",
        f"- corr(eta_geom, alpha_recall): {out['stage53']['summary'].get('corr_eta_vs_alpha_recall', 0.0):.4f}",
        "",
        "## Reviewer-Facing Caveats",
        "- SDPI interpretation is an approximation in MI-space; recall-space mapping remains empirical.",
        "- KSG and CANCOR are estimator-dependent; agreement across both is required for strong claims.",
        "- CANCOR MI is a Gaussian-copula proxy; treat it as a structured proxy, not an assumption-free estimate.",
        "- Stage25 within-shard low-variance conditions can weaken within-source correlation despite strong global fits.",
    ]

    (stage_root / "stage54_sdpi_aggressive_aggregate.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={"stage": "stage54_sdpi_aggressive_aggregate", "elapsed_sec": float(time.time() - start)},
    )
    save_json(provenance, stage_root / "provenance_stage54.json")
    mark_done(markers / "stage54_sdpi_aggressive_aggregate.done.json", {"elapsed_sec": float(time.time() - start)})
    print("[stage54] complete")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(cfg)


if __name__ == "__main__":
    main()

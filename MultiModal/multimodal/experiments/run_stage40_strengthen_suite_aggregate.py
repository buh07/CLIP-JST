from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml

from ..common import load_json, mark_done, save_json


def _safe_load(p: Path) -> dict[str, Any] | None:
    if p.exists():
        return load_json(p)
    return None


def _extract_joint_ceiling_rows(stage_paths: dict[str, Path], target_dim: int = 512) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    m_key = f"m{int(target_dim)}"
    for name, p in stage_paths.items():
        obj = _safe_load(p)
        if not obj:
            continue
        methods = obj.get("stats", {}).get(m_key, {}).get("methods", {})
        for method, blk in methods.items():
            out.append(
                {
                    "source": name,
                    "method": method,
                    "embed_dim": int(target_dim),
                    "combined_avg_R_mean": float(blk.get("combined_avg_R", {}).get("mean", 0.0)),
                    "av_ia_avg_R_mean": float(blk.get("av_ia_avg_R", {}).get("mean", 0.0)),
                    "av_at_avg_R_mean": float(blk.get("av_at_avg_R", {}).get("mean", 0.0)),
                    "av_it_avg_R_mean": float(blk.get("av_it_avg_R", {}).get("mean", 0.0)),
                }
            )
    return out


def _extract_true_joint_reference_rows(stage20_results_path: Path, target_dim: int = 512) -> list[dict[str, Any]]:
    obj = _safe_load(stage20_results_path)
    if not obj:
        return []
    m_key = f"m{int(target_dim)}"
    methods = obj.get("stats", {}).get(m_key, {}).get("methods", {})
    out: list[dict[str, Any]] = []
    for method, blk in methods.items():
        out.append(
            {
                "source": "stage20_modular_transitivity_suite",
                "method": method,
                "embed_dim": int(target_dim),
                "combined_avg_R_mean": float(blk.get("combined_avg_R", {}).get("mean", 0.0)),
                "av_ia_avg_R_mean": float(blk.get("av_ia_avg_R", {}).get("mean", 0.0)),
                "av_at_avg_R_mean": float(blk.get("av_at_avg_R", {}).get("mean", 0.0)),
                "av_it_avg_R_mean": float(blk.get("av_it_avg_R", {}).get("mean", 0.0)),
            }
        )
    return out


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage40_strengthen_suite_aggregate"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage36 = _safe_load(Path(cfg["stage36_results_path"]).resolve())
    stage37 = _safe_load(Path(cfg["stage37_results_path"]).resolve())
    stage38 = _safe_load(Path(cfg["stage38_results_path"]).resolve())
    stage39 = _safe_load(Path(cfg["stage39_results_path"]).resolve())

    stage_paths = {
        "stage29_cc3m_phaseA_modular": Path(cfg["stage29_results_path"]).resolve(),
        "stage30_modular_vs_nonmodular": Path(cfg["stage30_results_path"]).resolve(),
        "stage31_wavcaps_scaling": Path(cfg["stage31_results_path"]).resolve(),
        "stage32_modality_order_ablation": Path(cfg["stage32_results_path"]).resolve(),
    }
    ceiling_rows = _extract_joint_ceiling_rows(stage_paths, target_dim=int(cfg.get("target_dim", 512)))
    true_joint_rows = _extract_true_joint_reference_rows(
        Path(cfg.get("stage20_results_path", "")).resolve() if cfg.get("stage20_results_path") else Path(""),
        target_dim=int(cfg.get("target_dim", 512)),
    ) if cfg.get("stage20_results_path") else []

    out = {
        "stage": "stage40_strengthen_suite_aggregate",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage36_bottleneck": stage36,
        "stage37_imagebind": stage37,
        "stage38_phaseb_quality": stage38,
        "stage39_modality_gap": stage39,
        "joint_ceiling_table_rows": ceiling_rows,
        "true_joint_reference_rows": true_joint_rows,
        "elapsed_sec": float(time.time() - start),
    }
    save_json(out, stage_root / "stage40_strengthen_suite_aggregate.json")

    md = [
        "# Stage40 Strengthen-Suite Aggregate",
        "",
        f"- Generated: `{out['generated_at']}`",
        "",
        "## Bottleneck Law (Stage36)",
    ]
    if stage36:
        gf = stage36.get("global_fit", {})
        md += [
            f"- alpha={gf.get('alpha', 0.0):.4f}",
            f"- Pearson r={gf.get('pearson_r', 0.0):.4f}",
            f"- Pearson r²={gf.get('pearson_r2', 0.0):.4f}",
            f"- R² no-intercept={gf.get('r2_nointercept', 0.0):.4f}",
        ]
    else:
        md += ["- missing"]

    md += ["", "## ImageBind Baseline (Stage37)"]
    if stage37:
        md += [
            f"- n={stage37.get('n_examples', 0)}",
            f"- chance_p1={stage37.get('chance_p1', 0.0):.6f}",
            f"- image_audio avg_R={stage37.get('image_audio', {}).get('avg_R', 0.0):.4f}",
            f"- audio_text avg_R={stage37.get('audio_text', {}).get('avg_R', 0.0):.4f}",
            f"- image_text avg_R={stage37.get('image_text', {}).get('avg_R', 0.0):.4f}",
        ]
    else:
        md += ["- missing"]

    md += ["", "## Phase-B Data Quality (Stage38)"]
    if stage38:
        a = stage38.get("audiocaps", {}).get("clap_similarity", {})
        w_mixed = stage38.get("wavcaps", {}).get("clap_similarity", {})
        w_clean = stage38.get("wavcaps_clean_source", {}).get("clap_similarity", {})
        clean_name = stage38.get("wavcaps_clean_source", {}).get("source_name", "WavCaps/WavCaps")
        md += [
            f"- AudioCaps margin_mean={a.get('margin_mean', 0.0):.4f}, pos_mean={a.get('pos_mean', 0.0):.4f}",
            f"- WavCaps mixed margin_mean={w_mixed.get('margin_mean', 0.0):.4f}, pos_mean={w_mixed.get('pos_mean', 0.0):.4f}",
            f"- WavCaps clean source ({clean_name}) margin_mean={w_clean.get('margin_mean', 0.0):.4f}, pos_mean={w_clean.get('pos_mean', 0.0):.4f}",
            f"- delta(mixed-audio) margin={stage38.get('delta_wav_minus_audio', {}).get('margin_mean', 0.0):.4f}",
            f"- delta(clean-audio) margin={stage38.get('delta_wav_clean_minus_audio', {}).get('margin_mean', 0.0):.4f}",
        ]
    else:
        md += ["- missing"]

    md += ["", "## Modality Gap and Margin (Stage39)"]
    if stage39:
        stats = stage39.get("stats", {})
        for m_key in sorted(stats.keys(), key=lambda x: int(x[1:])):
            md.append(f"### {m_key}")
            for method, blk in sorted(stats[m_key].get("methods", {}).items()):
                md.append(
                    f"- {method}: av_ia={blk.get('av_ia_avg_R', {}).get('mean', 0.0):.4f}, "
                    f"gap_ia={blk.get('gap_ia_l2', {}).get('mean', 0.0):.4f}, "
                    f"i2a_margin={blk.get('i2a_margin_mean', {}).get('mean', 0.0):.4f}"
                )
    else:
        md += ["- missing"]

    md += [
        "",
        "## m=512 Method Comparison Across Experiments",
        "",
        "> Note: this table mixes methods from Stage29/30/31/32 and is not a pure joint-training-only table.",
        "| source | method | combined | av_ia | av_at | av_it |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in sorted(ceiling_rows, key=lambda x: (x["source"], x["method"])):
        md.append(
            f"| {r['source']} | {r['method']} | {r['combined_avg_R_mean']:.4f} | "
            f"{r['av_ia_avg_R_mean']:.4f} | {r['av_at_avg_R_mean']:.4f} | {r['av_it_avg_R_mean']:.4f} |"
        )

    if true_joint_rows:
        md += [
            "",
            "## True Joint-Training Reference Rows (m=512)",
            "| source | method | combined | av_ia | av_at | av_it |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in sorted(true_joint_rows, key=lambda x: (x["source"], x["method"])):
            md.append(
                f"| {r['source']} | {r['method']} | {r['combined_avg_R_mean']:.4f} | "
                f"{r['av_ia_avg_R_mean']:.4f} | {r['av_at_avg_R_mean']:.4f} | {r['av_it_avg_R_mean']:.4f} |"
            )

    (stage_root / "stage40_strengthen_suite_aggregate.md").write_text("\n".join(md), encoding="utf-8")
    mark_done(markers / "stage40_strengthen_suite_aggregate.done.json", {"elapsed_sec": float(time.time() - start)})
    print("Stage40 complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

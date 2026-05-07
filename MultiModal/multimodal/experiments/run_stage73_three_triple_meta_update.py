from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml

from ..common import env_snapshot, load_json, mark_done, save_json
from .run_stage68_law_robustness_reanalysis import (
    _cluster_and_mixed_effects,
    _collect_stage58_seed_records,
    _evaluate_forms,
    _fit_basic_law,
    _group_cell_means,
)


def _collect_stage69_seed_records(stage70_path: Path) -> list[dict[str, Any]]:
    stage70 = load_json(stage70_path)
    roots = [Path(x).resolve() for x in stage70["stage69_shard_roots"]]
    merged: dict[tuple[int, str, int], dict[str, Any]] = {}
    for root in roots:
        p = root / "stage69_third_triple_speechcoco" / "stage69_third_triple_speechcoco_results.json"
        block = load_json(p)
        for m_key, methods in block.get("raw", {}).items():
            m = int(str(m_key).lstrip("m"))
            for method, rows in methods.items():
                for r in rows:
                    key = (m, str(method), int(r["seed"]))
                    merged[key] = {
                        "source_id": "stage69_speechcoco_full_grid",
                        "method": str(method),
                        "embed_dim": int(m),
                        "seed": int(r["seed"]),
                        "av_it_ood": float(r["av_it_avg_R"]),
                        "av_at": float(r["av_at_avg_R"]),
                        "av_ia": float(r["av_ia_avg_R"]),
                    }
    return [merged[k] for k in sorted(merged.keys())]


def run(cfg: dict[str, Any]) -> None:
    start = time.time()

    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage73_three_triple_meta_update"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage36 = load_json(Path(str(cfg["stage36_results"])).resolve())
    stage58_path = Path(str(cfg["stage58_results"])).resolve()
    stage70_path = Path(str(cfg["stage70_results"])).resolve()
    stage68_path = Path(str(cfg["stage68_results"])).resolve()
    stage72_path = Path(str(cfg["stage72_results"])).resolve()

    # Seed-level records for each triple
    rec_audio = list(stage36["records"])
    rec_avcaps = _collect_stage58_seed_records(stage58_path)
    rec_speech = _collect_stage69_seed_records(stage70_path)

    # Triple-wise robustness
    robustness = {
        "audiocaps": {
            "seed_level": _fit_basic_law(rec_audio),
            "cell_mean_level": _fit_basic_law(_group_cell_means(rec_audio)),
            "cluster_and_mixed_effects": _cluster_and_mixed_effects(rec_audio),
        },
        "avcaps": {
            "seed_level": _fit_basic_law(rec_avcaps),
            "cell_mean_level": _fit_basic_law(_group_cell_means(rec_avcaps)),
            "cluster_and_mixed_effects": _cluster_and_mixed_effects(rec_avcaps),
        },
        "speechcoco": {
            "seed_level": _fit_basic_law(rec_speech),
            "cell_mean_level": _fit_basic_law(_group_cell_means(rec_speech)),
            "cluster_and_mixed_effects": _cluster_and_mixed_effects(rec_speech),
        },
    }

    # Pooled across all three triples
    rec_all = rec_audio + rec_avcaps + rec_speech
    pooled = {
        "seed_level": _fit_basic_law(rec_all),
        "cell_mean_level": _fit_basic_law(_group_cell_means(rec_all)),
        "cluster_and_mixed_effects": _cluster_and_mixed_effects(rec_all),
    }

    # Functional-form summary: carry forward stage68 + stage72, add pooled recompute
    stage68 = load_json(stage68_path)
    stage72 = load_json(stage72_path)
    forms = {
        "from_stage68": stage68.get("functional_form_cross_suite", {}),
        "speechcoco_from_stage72": stage72.get("analysis", {}),
        "pooled_three_triples": _evaluate_forms(
            rec_all,
            suite_name="pooled_three_triples",
            kfold=int(cfg.get("kfold", 5)),
            cv_seed=int(cfg.get("cv_seed", 2026)),
            boot_n=int(cfg.get("powerlaw_bootstrap_n", 1000)),
            holdout_source=None,
        ),
    }

    out = {
        "stage": "stage73_three_triple_meta_update",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "triple_robustness": robustness,
        "pooled_three_triple_robustness": pooled,
        "functional_forms": forms,
        "n_rows": {
            "audiocaps_seed_rows": int(len(rec_audio)),
            "avcaps_seed_rows": int(len(rec_avcaps)),
            "speechcoco_seed_rows": int(len(rec_speech)),
            "pooled_seed_rows": int(len(rec_all)),
        },
        "elapsed_sec": float(time.time() - start),
    }
    save_json(out, stage_root / "stage73_three_triple_meta_update.json")

    md = [
        "# Stage73 Three-Triple Meta Update",
        "",
        "## Triple-Wise Law Robustness",
        "",
        "| Triple | n(seed) | alpha(seed) | r2(seed) | alpha(cell) | r2(cell) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name in ["audiocaps", "avcaps", "speechcoco"]:
        sec = robustness[name]
        md.append(
            f"| {name} | {int(sec['seed_level']['n'])} | {sec['seed_level']['alpha']:.4f} | {sec['seed_level']['r2']:.4f} | "
            f"{sec['cell_mean_level']['alpha']:.4f} | {sec['cell_mean_level']['r2']:.4f} |"
        )

    md += [
        "",
        "## Pooled Three-Triple Robustness",
        f"- n_seed: {int(pooled['seed_level']['n'])}",
        f"- alpha_seed: {pooled['seed_level']['alpha']:.4f}, r2_seed: {pooled['seed_level']['r2']:.4f}, mae_seed: {pooled['seed_level']['mae']:.5f}",
        f"- alpha_cell: {pooled['cell_mean_level']['alpha']:.4f}, r2_cell: {pooled['cell_mean_level']['r2']:.4f}, mae_cell: {pooled['cell_mean_level']['mae']:.5f}",
        "",
        "## Pooled Form Selection (CV-R2)",
    ]

    pooled_forms = sorted(forms["pooled_three_triples"]["models"], key=lambda m: m["cv_r2_mean"], reverse=True)
    for m in pooled_forms:
        md.append(f"- {m['name']}: cv_r2={m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}")

    (stage_root / "stage73_three_triple_meta_update.md").write_text("\n".join(md), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={"stage": "stage73_three_triple_meta_update", "elapsed_sec": float(time.time() - start)},
    )
    save_json(provenance, stage_root / "provenance_stage73.json")
    mark_done(markers / "stage73_three_triple_meta_update.done.json", {"elapsed_sec": float(time.time() - start)})
    print("stage73_three_triple_meta_update complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

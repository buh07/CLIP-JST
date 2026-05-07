from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml

from ..common import load_json, mark_done, save_json
from ..eval.stats import build_metric_report


METRICS = ["combined_avg_R", "coco_avg_R", "av_it_avg_R", "av_at_avg_R", "av_ia_avg_R"]


def _load_stage_results(root: Path, stage_name: str) -> dict[str, Any] | None:
    p = root / stage_name / f"{stage_name}_results.json"
    if p.exists():
        return load_json(p)
    p_fail = root / stage_name / f"{stage_name}_failure.json"
    if p_fail.exists():
        return load_json(p_fail)
    return None


def _merge_seed_rows(dst_rows: list[dict[str, Any]], src_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed = {int(r["seed"]): r for r in dst_rows if "seed" in r}
    for r in src_rows:
        by_seed[int(r["seed"])] = r
    return [by_seed[s] for s in sorted(by_seed)]


def _merge_raw(raws: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for raw in raws:
        for m_key, methods in raw.items():
            out.setdefault(m_key, {})
            for method, rows in methods.items():
                cur = out[m_key].get(method, [])
                out[m_key][method] = _merge_seed_rows(cur, rows)
    return out


def _namespace_raw(raw: dict[str, Any], method_prefix: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for m_key, methods in raw.items():
        if not isinstance(methods, dict):
            continue
        out.setdefault(m_key, {})
        for method, rows in methods.items():
            out[m_key][f"{method_prefix}/{method}"] = rows
    return out


def _collect_raw_from_eval_tree(stage_root: Path, *, method_prefix: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    Rebuild raw rows from per-seed eval files:
      stage_root/m{dim}/{method}/seed{n}/eval.json
    This protects Stage33 from partial shard summary overwrites.
    """
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if not stage_root.exists():
        return out
    for p in stage_root.glob("m*/**/seed*/eval.json"):
        try:
            rec = load_json(p)
        except Exception:
            continue
        m_val = rec.get("embed_dim")
        method = rec.get("method")
        seed = rec.get("seed")
        if m_val is None or method is None or seed is None:
            continue
        m_key = f"m{int(m_val)}"
        out.setdefault(m_key, {}).setdefault(f"{method_prefix}/{method}", []).append(rec)
    # dedupe/sort by seed
    for m_key, methods in out.items():
        for method, rows in methods.items():
            methods[method] = _merge_seed_rows([], rows)
    return out


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    suite_root = Path(cfg["suite_root"]).resolve()

    stage27 = load_json(suite_root / "stage27_bottleneck_decomposition" / "stage27_bottleneck_decomposition_results.json")
    stage28 = load_json(suite_root / "stage28_category_retrieval" / "stage28_category_retrieval_results.json")
    s29 = _load_stage_results(suite_root, "stage29_cc3m_phaseA_modular")
    s30 = _load_stage_results(suite_root, "stage30_modular_vs_nonmodular")
    s31 = _load_stage_results(suite_root, "stage31_wavcaps_scaling")
    s32 = _load_stage_results(suite_root, "stage32_modality_order_ablation")

    raw_candidates: list[dict[str, Any]] = []
    for stage_name, s in [
        ("stage29_cc3m_phaseA_modular", s29),
        ("stage30_modular_vs_nonmodular", s30),
        ("stage31_wavcaps_scaling", s31),
        ("stage32_modality_order_ablation", s32),
    ]:
        if s is not None and isinstance(s, dict):
            raw_candidates.append(_namespace_raw(s.get("raw", {}), stage_name))
        raw_candidates.append(_collect_raw_from_eval_tree(suite_root / stage_name, method_prefix=stage_name))

    merged_training_raw = _merge_raw(raw_candidates)
    baseline_method = str(cfg.get("baseline_method", "modular_shared_jl"))

    merged_training_stats: dict[str, Any] = {}
    baseline_candidates = [
        f"stage32_modality_order_ablation/{baseline_method}",
        f"stage29_cc3m_phaseA_modular/{baseline_method}",
        f"stage31_wavcaps_scaling/{baseline_method}",
        f"stage30_modular_vs_nonmodular/{baseline_method}",
        baseline_method,
    ]
    for m_key, methods in merged_training_raw.items():
        if not methods:
            continue
        b = next((cand for cand in baseline_candidates if cand in methods), next(iter(methods)))
        rep = build_metric_report(methods, metrics=METRICS, baseline_method=b)
        rep["baseline_method"] = b
        merged_training_stats[m_key] = rep

    out = {
        "stage": "stage33_domain_gap_aggregate",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suite_root": str(suite_root),
        "stage27": stage27,
        "stage28": stage28,
        "stage29": s29,
        "stage30": s30,
        "stage31": s31,
        "stage32": s32,
        "merged_training": {
            "raw": merged_training_raw,
            "stats": merged_training_stats,
            "baseline_method": baseline_method,
        },
        "elapsed_sec": time.time() - start,
    }

    save_json(out, output_root / "stage33_domain_gap_aggregate.json")

    md = [
        "# Stage33 Domain-Gap Closure Aggregate",
        "",
        f"- Generated: `{out['generated_at']}`",
        f"- Suite root: `{suite_root}`",
        "",
        "## Stage27 (Bottleneck Decomposition)",
    ]

    eff = stage27.get("efficiency_global", {})
    for method, blk in sorted(eff.items()):
        md.append(
            f"- {method}: efficiency={blk['efficiency']['mean']:.4f} "
            f"CI[{blk['efficiency']['ci95_low']:.4f},{blk['efficiency']['ci95_high']:.4f}]"
        )

    md += ["", "## Stage28 (Category Retrieval)"]
    md.append(f"- chance_p1={stage28.get('chance_p1', 0.0):.4f}")
    for m_key, methods in sorted(stage28.get("summary", {}).items(), key=lambda kv: int(kv[0][1:])):
        md.append(f"- {m_key}:")
        for method, blk in sorted(methods.items()):
            md.append(f"  - {method}: avg_cat_p1={blk['mean']:.4f} ± {blk['std']:.4f} (n={blk['n']})")

    md += ["", "## Merged Training Stats (Stages 29–32)"]
    for m_key, rep in sorted(merged_training_stats.items(), key=lambda kv: int(kv[0][1:])):
        md.append(f"### {m_key}")
        for method, mblk in sorted(rep.get("methods", {}).items()):
            md.append(
                f"- {method}: combined={mblk['combined_avg_R']['mean']:.4f}, "
                f"coco={mblk['coco_avg_R']['mean']:.4f}, av_it={mblk['av_it_avg_R']['mean']:.4f}, "
                f"av_at={mblk['av_at_avg_R']['mean']:.4f}, av_ia={mblk['av_ia_avg_R']['mean']:.4f}"
            )

    (output_root / "stage33_domain_gap_aggregate.md").write_text("\n".join(md), encoding="utf-8")
    mark_done(markers / "stage33_domain_gap_aggregate.done.json", {"elapsed_sec": time.time() - start})
    print("Stage33 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

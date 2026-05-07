from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..common import load_json, mark_done, save_json


def _seed_coverage_stage5(ds_obj: dict, expected_seeds: list[int]) -> dict:
    missing = []
    for m_key, methods in ds_obj.get("raw", {}).items():
        for method, rows in methods.items():
            have = sorted({int(r["seed"]) for r in rows})
            for s in expected_seeds:
                if s not in have:
                    missing.append({"m": m_key, "method": method, "seed": s})
    return {"missing": missing, "ok": len(missing) == 0}


def _seed_coverage_stage6(ds_obj: dict, expected_model_seeds: list[int], expected_mask_seeds: list[int]) -> dict:
    missing = []
    for m_key, methods in ds_obj.get("raw", {}).items():
        for method, rows in methods.items():
            if method == "clip_head":
                continue
            have = {(int(r["model_seed"]), int(r["mask_seed"])) for r in rows}
            for ms in expected_model_seeds:
                for ks in expected_mask_seeds:
                    if (ms, ks) not in have:
                        missing.append({"m": m_key, "method": method, "model_seed": ms, "mask_seed": ks})
    return {"missing": missing, "ok": len(missing) == 0}


def _seed_coverage_stage8(raw: dict, expected_seeds: list[int]) -> dict:
    missing = []
    for eps_key, rows in raw.items():
        have = sorted({int(r["seed"]) for r in rows})
        for s in expected_seeds:
            if s not in have:
                missing.append({"epsilon": eps_key, "seed": s})
    return {"missing": missing, "ok": len(missing) == 0}


def _safe_get(d: dict, keys: list[str], default=None):
    cur = d
    try:
        for k in keys:
            cur = cur[k]
        return cur
    except Exception:
        return default


def _build_markdown(summary: dict) -> str:
    lines = []
    lines.append("# E8 Concatenation Suite Aggregate")
    lines.append("")
    lines.append("## Coverage")
    for name, cov in summary.get("coverage", {}).items():
        lines.append(f"- {name}: {'OK' if cov.get('ok') else 'MISSING'}")
    lines.append("")
    lines.append("## Gate Decision")
    gate = summary.get("gate_decision")
    if gate:
        lines.append(
            f"- gate={gate.get('gate')} avg_R={gate.get('avg_R'):.4f} "
            f"run_stage6={gate.get('run_stage6')} run_stage7={gate.get('run_stage7')} run_stage8={gate.get('run_stage8')}"
        )
    else:
        lines.append("- gate decision unavailable")
    lines.append("")
    lines.append("## Pareto Table (COCO m=256)")
    lines.append("| family | id | retrieval_avg_R | inversion_mlp_rel_error | notes |")
    lines.append("|---|---:|---:|---:|---|")
    for row in summary.get("pareto_rows", []):
        inv = row.get("inversion_mlp_rel_error")
        inv_str = "NA" if inv is None else f"{inv:.4f}"
        lines.append(
            f"| {row.get('family')} | {row.get('id')} | {row.get('retrieval_avg_R'):.4f} | {inv_str} | {row.get('notes','')} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(cfg: dict) -> None:
    output_root = Path(cfg["output_root"]).resolve()
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage5_root = Path(cfg["stage5_root"]).resolve()
    stage6_root = Path(cfg["stage6_root"]).resolve()
    stage7_path = Path(cfg["stage7_results"]).resolve()
    stage8_path = Path(cfg["stage8_results"]).resolve()
    gate_path = Path(cfg["gate_decision"]).resolve()

    stage5_coco = load_json(stage5_root / "E8a_concat_coco.json") if (stage5_root / "E8a_concat_coco.json").exists() else {}
    stage5_flickr = load_json(stage5_root / "E8a_concat_flickr30k.json") if (stage5_root / "E8a_concat_flickr30k.json").exists() else {}
    stage6_coco = load_json(stage6_root / "E8b_mask_concat_coco.json") if (stage6_root / "E8b_mask_concat_coco.json").exists() else {}
    stage6_flickr = load_json(stage6_root / "E8b_mask_concat_flickr30k.json") if (stage6_root / "E8b_mask_concat_flickr30k.json").exists() else {}
    stage7 = load_json(stage7_path) if stage7_path.exists() else {}
    stage8 = load_json(stage8_path) if stage8_path.exists() else {}
    gate = load_json(gate_path) if gate_path.exists() else {}

    coverage = {}
    coverage["stage5_coco"] = _seed_coverage_stage5(stage5_coco, list(cfg["seeds"])) if stage5_coco else {"ok": False, "missing": ["missing file"]}
    coverage["stage5_flickr30k"] = _seed_coverage_stage5(stage5_flickr, list(cfg["seeds"])) if stage5_flickr else {"ok": False, "missing": ["missing file"]}
    coverage["stage6_coco"] = _seed_coverage_stage6(stage6_coco, list(cfg["seeds"]), list(cfg["mask_seeds"])) if stage6_coco else {"ok": False, "missing": ["missing file"]}
    coverage["stage6_flickr30k"] = _seed_coverage_stage6(stage6_flickr, list(cfg["seeds"]), list(cfg["mask_seeds"])) if stage6_flickr else {"ok": False, "missing": ["missing file"]}
    coverage["stage8_dpsgd"] = _seed_coverage_stage8(stage8.get("raw", {}), list(cfg["seeds"])) if stage8 else {"ok": False, "missing": ["missing file"]}

    pareto_rows = []
    # Stage5 retrieval anchor points (COCO, m=256)
    m_key = "m256"
    s5_methods = _safe_get(stage5_coco, ["stats", m_key, "methods"], {})
    for method in ["clip_head", "random_jl_mahal", "mahal_only_rfull", "concat_a1_b1"]:
        avg = _safe_get(s5_methods, [method, "avg_R", "mean"])
        if avg is not None:
            pareto_rows.append(
                {
                    "family": "concat_stage5",
                    "id": method,
                    "retrieval_avg_R": float(avg),
                    "inversion_mlp_rel_error": None,
                    "notes": "COCO m=256",
                }
            )
    # Stage6 retrieval rows for p-sweep (COCO, m=256).
    s6_methods = _safe_get(stage6_coco, ["stats", m_key, "methods"], {})
    for method, obj in s6_methods.items():
        if not method.startswith("mask_concat_p"):
            continue
        avg = _safe_get(obj, ["avg_R", "mean"])
        if avg is None:
            continue
        pareto_rows.append(
            {
                "family": "mask_stage6",
                "id": method,
                "retrieval_avg_R": float(avg),
                "inversion_mlp_rel_error": None,
                "notes": "COCO m=256",
            }
        )
    # Stage7 privacy rows mapped by p for m=256.
    s7 = _safe_get(stage7, ["stats", "m256"], {})
    s7_priv = {}
    for p_key, obj in s7.items():
        val = _safe_get(obj, ["mlp_inverter", "mean_relative_reconstruction_error", "mean"])
        if val is not None:
            s7_priv[p_key] = float(val)
    # Attach known privacy values to matching mask rows.
    for row in pareto_rows:
        rid = row["id"]
        if rid == "random_jl_mahal":
            row["inversion_mlp_rel_error"] = s7_priv.get("p0")
        elif rid == "concat_a1_b1":
            row["inversion_mlp_rel_error"] = s7_priv.get("p1")
        elif rid.startswith("mask_concat_p"):
            p = rid.split("mask_concat_p", 1)[1]
            row["inversion_mlp_rel_error"] = s7_priv.get(f"p{p}")

    # Stage8 DP rows.
    for eps_key, obj in _safe_get(stage8, ["stats", "methods"], {}).items():
        avg = _safe_get(obj, ["avg_R", "mean"])
        inv = _safe_get(obj, ["mlp_rel_error", "mean"])
        if avg is None:
            continue
        pareto_rows.append(
            {
                "family": "dpsgd_stage8",
                "id": eps_key,
                "retrieval_avg_R": float(avg),
                "inversion_mlp_rel_error": float(inv) if inv is not None else None,
                "notes": "COCO",
            }
        )

    summary = {
        "coverage": coverage,
        "gate_decision": gate,
        "pareto_rows": pareto_rows,
        "artifacts": {
            "stage5_root": str(stage5_root),
            "stage6_root": str(stage6_root),
            "stage7_results": str(stage7_path),
            "stage8_results": str(stage8_path),
            "gate_decision": str(gate_path),
        },
    }
    save_json(summary, output_root / "stage9_e8_aggregate.json")
    md = _build_markdown(summary)
    (output_root / "stage9_e8_aggregate.md").write_text(md, encoding="utf-8")
    mark_done(markers / "stage9_e8_aggregate.done.json")
    print("Stage 9 (E8 aggregate) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

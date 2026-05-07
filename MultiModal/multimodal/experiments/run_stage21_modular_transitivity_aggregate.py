from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import yaml

from ..common import load_json, mark_done, save_json
from ..eval.stats import build_metric_report


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            shutil.copy2(p, out)


def _merge_seed_rows(dst_rows: list[dict], src_rows: list[dict]) -> list[dict]:
    by_seed = {int(r["seed"]): r for r in dst_rows if "seed" in r}
    for r in src_rows:
        by_seed[int(r["seed"])] = r
    return [by_seed[s] for s in sorted(by_seed)]


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    shards = [Path(p).resolve() for p in cfg["shard_output_roots"]]

    merged = {
        "stage": "stage21_modular_transitivity_aggregate",
        "stage18": {"sources": [], "datasets": {}},
        "stage19": {"raw": {}, "stats": {}, "gate": {}, "gate_decision": {}},
        "stage20": {"raw": {}, "stats": {}},
        "coverage": {"shards": [str(s) for s in shards]},
    }

    for shard in shards:
        _copy_tree(shard, output_root)

        s18 = shard / "stage18_spectral_results.json"
        if s18.exists():
            obj = load_json(s18)
            merged["stage18"]["sources"].append(str(s18))
            for ds, payload in obj.get("datasets", {}).items():
                merged["stage18"]["datasets"][ds] = payload

        s19 = shard / "stage19_pseudomodality" / "stage19_pseudomodality_results.json"
        if s19.exists():
            obj = load_json(s19)
            for m_key, methods in obj.get("raw", {}).items():
                merged["stage19"]["raw"].setdefault(m_key, {})
                for method, rows in methods.items():
                    cur = merged["stage19"]["raw"][m_key].get(method, [])
                    merged["stage19"]["raw"][m_key][method] = _merge_seed_rows(cur, rows)

        s20 = shard / "stage20_modular_audio_transitivity" / "stage20_results.json"
        if s20.exists():
            obj = load_json(s20)
            if obj.get("skipped"):
                continue
            for m_key, methods in obj.get("raw", {}).items():
                merged["stage20"]["raw"].setdefault(m_key, {})
                for method, rows in methods.items():
                    cur = merged["stage20"]["raw"][m_key].get(method, [])
                    merged["stage20"]["raw"][m_key][method] = _merge_seed_rows(cur, rows)

    # Recompute stage19 stats+gate.
    thr = float(cfg.get("gate_relative_threshold", 0.90))
    gate_embed = int(cfg.get("gate_embed_dim", 256))
    for m_key, methods in merged["stage19"]["raw"].items():
        merged["stage19"]["stats"][m_key] = build_metric_report(
            methods,
            metrics=["pair_avg_R", "pseudo_text_avg_R"],
            baseline_method="joint_shared_jl_baseline",
        )
        mod_mean = merged["stage19"]["stats"][m_key]["methods"]["modular_shared_jl"]["pair_avg_R"]["mean"]
        joint_mean = merged["stage19"]["stats"][m_key]["methods"]["joint_shared_jl_baseline"]["pair_avg_R"]["mean"]
        rel = float(mod_mean / max(1e-12, joint_mean))
        merged["stage19"]["gate"][m_key] = {
            "modular_pair_avg_R_mean": float(mod_mean),
            "joint_pair_avg_R_mean": float(joint_mean),
            "relative_ratio": rel,
            "threshold": thr,
            "pass": bool(rel >= thr),
        }

    gk = f"m{gate_embed}"
    gobj = merged["stage19"]["gate"].get(gk, {})
    merged["stage19"]["gate_decision"] = {
        "gate_embed_dim": gate_embed,
        "gate_key": gk,
        "pass": bool(gobj.get("pass", False)),
        "relative_ratio": float(gobj.get("relative_ratio", 0.0)),
        "threshold": float(gobj.get("threshold", thr)),
    }

    # Recompute stage20 stats.
    for m_key, methods in merged["stage20"]["raw"].items():
        merged["stage20"]["stats"][m_key] = build_metric_report(
            methods,
            metrics=["combined_avg_R", "coco_avg_R", "av_it_avg_R", "av_at_avg_R", "av_ia_avg_R"],
            baseline_method="joint_shared_jl",
        )

    out_json = output_root / "stage21_modular_transitivity_aggregate.json"
    save_json(merged, out_json)

    md_lines = [
        "# Stage 21 Modular Transitivity Aggregate",
        "",
        f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Shards: `{len(shards)}`",
        f"- Stage19 gate decision (m={gate_embed}): `{merged['stage19']['gate_decision'].get('pass', False)}`",
        f"- Stage19 relative ratio: `{merged['stage19']['gate_decision'].get('relative_ratio', 0.0):.4f}`",
        "",
    ]

    for m_key, st in merged["stage20"].get("stats", {}).items():
        md_lines.append(f"## {m_key}")
        methods = st.get("methods", {})
        for name in ["modular_shared_jl", "joint_shared_jl", "joint_clip_head"]:
            if name in methods:
                row = methods[name]
                md_lines.append(
                    f"- {name}: combined={row['combined_avg_R']['mean']:.4f}, "
                    f"coco={row['coco_avg_R']['mean']:.4f}, av_ia={row['av_ia_avg_R']['mean']:.4f}, av_at={row['av_at_avg_R']['mean']:.4f}"
                )
        md_lines.append("")

    (output_root / "stage21_modular_transitivity_aggregate.md").write_text("\n".join(md_lines), encoding="utf-8")
    mark_done(markers / "stage21_modular_transitivity_aggregate.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 21 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

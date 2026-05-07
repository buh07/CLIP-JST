from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from ..common import load_json, mark_done, save_json
from ..eval.stats import build_metric_report


def _merge_rows(dst: list[dict], src: list[dict]) -> list[dict]:
    by_seed = {int(r["seed"]): r for r in dst if "seed" in r}
    for r in src:
        by_seed[int(r["seed"])] = r
    return [by_seed[s] for s in sorted(by_seed)]


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    shards = [Path(p).resolve() for p in cfg["shard_output_roots"]]
    merged = {"stage": "stage19_merge_gate", "raw": {}, "stats": {}, "gate": {}, "gate_decision": {}}

    for shard in shards:
        p = shard / "stage19_pseudomodality" / "stage19_pseudomodality_results.json"
        if not p.exists():
            raise FileNotFoundError(f"missing stage19 shard summary: {p}")
        obj = load_json(p)
        for m_key, methods in obj.get("raw", {}).items():
            merged["raw"].setdefault(m_key, {})
            for method, rows in methods.items():
                cur = merged["raw"][m_key].get(method, [])
                merged["raw"][m_key][method] = _merge_rows(cur, rows)

    thr = float(cfg.get("gate_relative_threshold", 0.90))
    for m_key, methods in merged["raw"].items():
        merged["stats"][m_key] = build_metric_report(
            methods,
            metrics=["pair_avg_R", "pseudo_text_avg_R"],
            baseline_method="joint_shared_jl_baseline",
        )
        mod_mean = merged["stats"][m_key]["methods"]["modular_shared_jl"]["pair_avg_R"]["mean"]
        joint_mean = merged["stats"][m_key]["methods"]["joint_shared_jl_baseline"]["pair_avg_R"]["mean"]
        rel = float(mod_mean / max(1e-12, joint_mean))
        merged["gate"][m_key] = {
            "modular_pair_avg_R_mean": float(mod_mean),
            "joint_pair_avg_R_mean": float(joint_mean),
            "relative_ratio": rel,
            "threshold": thr,
            "pass": bool(rel >= thr),
        }

    gate_embed = int(cfg.get("gate_embed_dim", 256))
    gk = f"m{gate_embed}"
    g = merged["gate"].get(gk, {})
    merged["gate_decision"] = {
        "gate_embed_dim": gate_embed,
        "gate_key": gk,
        "pass": bool(g.get("pass", False)),
        "relative_ratio": float(g.get("relative_ratio", 0.0)),
        "threshold": float(g.get("threshold", thr)),
    }

    save_json(merged, output_root / "stage19_merged_gate.json")
    mark_done(markers / "stage19_merge_gate.done.json", {"elapsed_sec": time.time() - start})
    print(f"Stage19 merge+gate complete. gate={merged['gate_decision']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

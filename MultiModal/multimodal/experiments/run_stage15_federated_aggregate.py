from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..common import load_json, mark_done, save_json


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def run(cfg: dict) -> None:
    output_root = Path(cfg["output_root"]).resolve()
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    s13 = load_json(output_root / "stage13_federated" / "E13_federated_results.json")
    s14_path = output_root / "stage14_stronger_attacks" / "E14_stronger_attacks_results.json"
    s14 = load_json(s14_path) if s14_path.exists() else {"raw": {}}

    rows = []
    for partition, methods in s13.get("raw", {}).items():
        for method, recs in methods.items():
            if not recs:
                continue
            key = f"{partition}/{method}"
            s14_rows = s14.get("raw", {}).get(partition, {}).get(method, [])
            row = {
                "partition": partition,
                "method": method,
                "avg_R_mean": _mean(recs, "avg_R"),
                "mlp_rel_error_mean": _mean(recs, "mlp_rel_error"),
                "linear_rel_error_mean": _mean(recs, "linear_rel_error"),
                "comm_mb_mean": _mean(recs, "comm_mb"),
                "iterative_rel_error_mean": _mean(s14_rows, "iterative_rel_error") if s14_rows else None,
                "n_seeds": len(recs),
                "key": key,
            }
            rows.append(row)

    # Retrieval-descending + privacy (higher error means stronger privacy) as tie-break.
    rows = sorted(rows, key=lambda r: (r["avg_R_mean"], r["mlp_rel_error_mean"]), reverse=True)

    summary = {
        "stage": "stage15_federated_aggregate",
        "rows": rows,
        "notes": [
            "Higher retrieval avg_R is better.",
            "Higher inversion relative error implies stronger resistance to reconstruction.",
            "Communication reflects total estimated send+receive payload per run.",
        ],
    }

    save_json(summary, output_root / "stage15_federated_aggregate.json")

    md_lines = ["# Federated Extension Aggregate", "", "| partition | method | avg_R | MLP inv err | Iter inv err | Comm MB | seeds |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        it = r["iterative_rel_error_mean"]
        it_s = "NA" if it is None else f"{it:.4f}"
        md_lines.append(
            f"| {r['partition']} | {r['method']} | {r['avg_R_mean']:.4f} | {r['mlp_rel_error_mean']:.4f} | {it_s} | {r['comm_mb_mean']:.1f} | {r['n_seeds']} |"
        )
    (output_root / "stage15_federated_aggregate.md").write_text("\n".join(md_lines), encoding="utf-8")

    mark_done(markers / "stage15_federated_aggregate.done.json")
    print("Stage 15 (federated aggregate) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

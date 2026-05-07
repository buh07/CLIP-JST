from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean, pstdev

import yaml

from ..common import env_snapshot, load_json, mark_done, save_json


def _collect_stage55(root: Path):
    p = root / "stage55_wavcaps_holdout_retrain" / "stage55_wavcaps_holdout_retrain_results.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing stage55 results: {p}")
    return load_json(p)


def _baseline_map(stage44_path: Path, method: str):
    d = load_json(stage44_path)
    out = {}
    for m_key, methods in d.get("raw", {}).items():
        rows = methods.get(method, [])
        if not rows:
            continue
        out[m_key] = {
            "av_at_avg_R": mean(float(r["av_at_avg_R"]) for r in rows),
            "av_ia_avg_R": mean(float(r["av_ia_avg_R"]) for r in rows),
        }
    return out


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage55_roots = [Path(x).resolve() for x in cfg["stage55_roots"]]
    baseline_method = str(cfg.get("baseline_method", "modular_shared_jl"))
    stage44_path = Path(cfg["stage44_results_path"]).resolve()
    baseline = _baseline_map(stage44_path, baseline_method)

    blocks = []
    merged = {}

    for root in stage55_roots:
        d = _collect_stage55(root)
        cond = str(d.get("condition_name", root.name))
        merged[cond] = d
        for m_key, methods in d.get("raw", {}).items():
            rows = methods.get(baseline_method, [])
            if not rows:
                continue
            av_at = [float(r["av_at_avg_R"]) for r in rows]
            av_ia = [float(r["av_ia_avg_R"]) for r in rows]
            wh_at = [float(r["wav_holdout_at_avg_R"]) for r in rows]
            b = baseline.get(m_key, {})
            b_at = float(b.get("av_at_avg_R", 0.0))
            b_ia = float(b.get("av_ia_avg_R", 0.0))
            blocks.append({
                "condition": cond,
                "embed_dim": int(m_key[1:]),
                "n": len(rows),
                "av_at_mean": mean(av_at),
                "av_at_std": pstdev(av_at) if len(av_at) > 1 else 0.0,
                "av_ia_mean": mean(av_ia),
                "av_ia_std": pstdev(av_ia) if len(av_ia) > 1 else 0.0,
                "wav_holdout_at_mean": mean(wh_at),
                "wav_holdout_at_std": pstdev(wh_at) if len(wh_at) > 1 else 0.0,
                "delta_holdout_minus_audiocaps_at": mean(wh_at) - mean(av_at),
                "delta_audiocaps_at_vs_stage44": mean(av_at) - b_at,
                "delta_audiocaps_ia_vs_stage44": mean(av_ia) - b_ia,
            })

    blocks = sorted(blocks, key=lambda r: (r["embed_dim"], r["condition"]))

    result = {
        "stage": "stage56_wavcaps_holdout_aggregate",
        "baseline_method": baseline_method,
        "stage55_roots": [str(p) for p in stage55_roots],
        "rows": blocks,
        "elapsed_sec": time.time() - start,
    }

    save_json(result, output_root / "stage56_wavcaps_holdout_aggregate.json")

    lines = [
        "# Stage56 WavCaps Holdout Aggregate",
        "",
        f"Baseline method: `{baseline_method}`",
        "",
        "| condition | m | n | av_at(AudioCaps eval) | av_ia(AudioCaps eval) | wav_holdout_at | holdout-at delta | delta av_at vs Stage44 | delta av_ia vs Stage44 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in blocks:
        lines.append(
            f"| {r['condition']} | {r['embed_dim']} | {r['n']} | "
            f"{r['av_at_mean']:.4f}±{r['av_at_std']:.4f} | {r['av_ia_mean']:.4f}±{r['av_ia_std']:.4f} | "
            f"{r['wav_holdout_at_mean']:.4f}±{r['wav_holdout_at_std']:.4f} | {r['delta_holdout_minus_audiocaps_at']:.4f} | "
            f"{r['delta_audiocaps_at_vs_stage44']:.4f} | {r['delta_audiocaps_ia_vs_stage44']:.4f} |"
        )
    (output_root / "stage56_wavcaps_holdout_aggregate.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={
            "stage": "stage56_wavcaps_holdout_aggregate",
            "elapsed_sec": time.time() - start,
        },
    )
    save_json(provenance, output_root / "provenance_stage56.json")
    mark_done(markers / "stage56_wavcaps_holdout_aggregate.done.json", {"elapsed_sec": time.time() - start})
    print("stage56_wavcaps_holdout_aggregate complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

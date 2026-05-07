from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml

from ..common import load_json, mark_done, save_json


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def _collect_stage_metrics(obj: dict[str, Any] | None, stage_name: str) -> dict[str, Any]:
    if not obj:
        return {"stage": stage_name, "status": "missing"}
    out: dict[str, Any] = {"stage": stage_name, "status": "ok", "by_dim": {}}
    stats = obj.get("stats", {})
    for m_key, blk in sorted(stats.items(), key=lambda kv: int(str(kv[0]).lstrip("m"))):
        methods = blk.get("methods", {})
        out["by_dim"][m_key] = {}
        for method, mblk in methods.items():
            out["by_dim"][m_key][method] = {
                "combined_avg_R": float(mblk.get("combined_avg_R", {}).get("mean", 0.0)),
                "coco_avg_R": float(mblk.get("coco_avg_R", {}).get("mean", 0.0)),
                "av_it_avg_R": float(mblk.get("av_it_avg_R", {}).get("mean", 0.0)),
                "av_at_avg_R": float(mblk.get("av_at_avg_R", {}).get("mean", 0.0)),
                "av_ia_avg_R": float(mblk.get("av_ia_avg_R", {}).get("mean", 0.0)),
                "n": int(mblk.get("combined_avg_R", {}).get("n", 0)),
            }
    return out


def run(cfg: dict) -> None:
    start = time.time()
    out_root = Path(cfg["output_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stage_root = out_root / "stage46_experimental_fixes_aggregate"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = out_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage44 = _load(Path(cfg["stage44_results_path"]).resolve())

    stage45_paths = [Path(p).resolve() for p in cfg.get("stage45_results_paths", [])]
    stage45_objs = []
    for p in stage45_paths:
        o = _load(p)
        stage45_objs.append({"path": str(p), "results": o, "summary": _collect_stage_metrics(o, "stage45_quality_quantity_deconfound")})

    out = {
        "stage": "stage46_experimental_fixes_aggregate",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage44_summary": _collect_stage_metrics(stage44, "stage44_zero_shot_baseline_control"),
        "stage45_summaries": stage45_objs,
        "elapsed_sec": float(time.time() - start),
    }

    save_json(out, stage_root / "stage46_experimental_fixes_aggregate.json")

    lines = [
        "# Stage46 Experimental Fixes Aggregate",
        "",
        f"- Generated: `{out['generated_at']}`",
        "",
        "## Stage44 Zero-Shot Baseline Control",
    ]

    s44 = out["stage44_summary"]
    if s44.get("status") != "ok":
        lines.append("- missing")
    else:
        for m_key, methods in s44["by_dim"].items():
            lines.append(f"### {m_key}")
            lines.append("| method | combined | coco | av_it | av_at | av_ia | n |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for method, vals in sorted(methods.items()):
                lines.append(
                    f"| {method} | {vals['combined_avg_R']:.4f} | {vals['coco_avg_R']:.4f} | "
                    f"{vals['av_it_avg_R']:.4f} | {vals['av_at_avg_R']:.4f} | {vals['av_ia_avg_R']:.4f} | {vals['n']} |"
                )

    lines += ["", "## Stage45 Quality-vs-Quantity Deconfound"]
    for item in out["stage45_summaries"]:
        lines += ["", f"### {Path(item['path']).parent.name}", f"- source file: `{item['path']}`"]
        s = item["summary"]
        if s.get("status") != "ok":
            lines.append("- missing")
            continue
        for m_key, methods in s["by_dim"].items():
            lines.append(f"#### {m_key}")
            lines.append("| method | combined | coco | av_it | av_at | av_ia | n |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for method, vals in sorted(methods.items()):
                lines.append(
                    f"| {method} | {vals['combined_avg_R']:.4f} | {vals['coco_avg_R']:.4f} | "
                    f"{vals['av_it_avg_R']:.4f} | {vals['av_at_avg_R']:.4f} | {vals['av_ia_avg_R']:.4f} | {vals['n']} |"
                )

    (stage_root / "stage46_experimental_fixes_aggregate.md").write_text("\n".join(lines), encoding="utf-8")
    mark_done(markers / "stage46_experimental_fixes_aggregate.done.json", {"elapsed_sec": float(time.time() - start)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

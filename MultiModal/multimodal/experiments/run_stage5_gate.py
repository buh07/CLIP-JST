from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..common import mark_done, save_json


def _fmt_float(x: float) -> str:
    s = f"{x:.6g}"
    return s.replace("-", "m").replace(".", "p")


def _concat_name(alpha: float, beta: float) -> str:
    return f"concat_a{_fmt_float(alpha)}_b{_fmt_float(beta)}"


def _decision(avg_r: float, high: float, mid: float) -> dict:
    if avg_r >= high:
        return {
            "gate": "high",
            "avg_R": avg_r,
            "run_stage6": True,
            "run_stage7": True,
            "run_stage8": True,
            "defer_stage7": False,
            "stop": False,
        }
    if avg_r >= mid:
        return {
            "gate": "mid",
            "avg_R": avg_r,
            "run_stage6": True,
            "run_stage7": False,
            "run_stage8": True,
            "defer_stage7": True,
            "stop": False,
        }
    return {
        "gate": "low",
        "avg_R": avg_r,
        "run_stage6": False,
        "run_stage7": False,
        "run_stage8": False,
        "defer_stage7": False,
        "stop": True,
    }


def run(cfg: dict) -> None:
    output_root = Path(cfg["output_root"]).resolve()
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage5_coco_path = Path(cfg["stage5_coco_results"]).resolve()
    if not stage5_coco_path.exists():
        raise FileNotFoundError(f"Missing Stage 5 COCO results: {stage5_coco_path}")

    import json

    with open(stage5_coco_path, encoding="utf-8") as f:
        stage5 = json.load(f)

    m_key = f"m{int(cfg.get('gate_embed_dim', 256))}"
    alpha = float(cfg.get("gate_alpha", 1.0))
    beta = float(cfg.get("gate_beta", 1.0))
    method = _concat_name(alpha, beta)
    try:
        avg_r = float(stage5["stats"][m_key]["methods"][method]["avg_R"]["mean"])
    except KeyError as e:
        raise KeyError(f"Could not locate gate metric for {method} {m_key} in {stage5_coco_path}") from e

    decision = _decision(
        avg_r=avg_r,
        high=float(cfg.get("gate_high", 0.51)),
        mid=float(cfg.get("gate_mid", 0.30)),
    )
    decision.update(
        {
            "source_file": str(stage5_coco_path),
            "gate_method": method,
            "gate_embed_dim": int(cfg.get("gate_embed_dim", 256)),
        }
    )
    save_json(decision, output_root / "stage5_gate_decision.json")
    mark_done(markers / "stage5_gate.done.json", decision)
    print(f"Stage 5 gate complete: {decision}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

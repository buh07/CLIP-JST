from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import yaml

from ..common import env_snapshot, load_json, mark_done, save_json
from .run_stage36_bottleneck_decomposition import _fit_alpha


def _pearson(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage43_bottleneck_oos_validation"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    src = Path(cfg["stage36_results_path"]).resolve()
    obj = load_json(src)
    records = obj.get("records", [])
    holdout_source = str(cfg.get("holdout_source", "stage31_wavcaps_scaling"))

    train_recs = [r for r in records if str(r.get("source_id")) != holdout_source and float(r.get("ceiling", 0.0)) > 0]
    test_recs = [r for r in records if str(r.get("source_id")) == holdout_source and float(r.get("ceiling", 0.0)) > 0]

    fit = _fit_alpha(train_recs)
    alpha = float(fit.get("alpha", 0.0) or 0.0)

    y_true = [float(r["av_ia"]) for r in test_recs]
    y_pred = [alpha * float(r["ceiling"]) for r in test_recs]
    abs_err = [abs(a - b) for a, b in zip(y_true, y_pred)]
    mae = sum(abs_err) / len(abs_err) if abs_err else 0.0
    r = _pearson(y_true, y_pred)

    out = {
        "stage": "stage43_bottleneck_oos_validation",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "holdout_source": holdout_source,
        "n_train": len(train_recs),
        "n_test": len(test_recs),
        "fit_train": fit,
        "oos": {
            "alpha_train": alpha,
            "pearson_r": r,
            "pearson_r2": r * r,
            "mae": mae,
        },
        "elapsed_sec": time.time() - start,
    }
    save_json(out, stage_root / "stage43_bottleneck_oos_validation.json")

    md = [
        "# Stage43 Bottleneck OOS Validation",
        "",
        f"- Holdout source: `{holdout_source}`",
        f"- Train records: `{len(train_recs)}`",
        f"- Test records: `{len(test_recs)}`",
        f"- Train alpha: `{alpha:.4f}`",
        f"- OOS Pearson r: `{r:.4f}` (r²=`{(r*r):.4f}`)",
        f"- OOS MAE: `{mae:.5f}`",
        "",
        "This is an explicit out-of-sample predictive test: alpha is fit without the holdout source and evaluated on holdout only.",
    ]
    (stage_root / "stage43_bottleneck_oos_validation.md").write_text("\n".join(md), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={"stage": "stage43_bottleneck_oos_validation", "elapsed_sec": time.time() - start},
    )
    save_json(provenance, stage_root / "provenance_stage43.json")
    mark_done(markers / "stage43_bottleneck_oos_validation.done.json", {"elapsed_sec": time.time() - start})
    print("Stage43 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from ..common import mark_done, save_json


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _tag_exists(repo: Path, tag: str) -> bool:
    p = subprocess.run(["git", "-C", str(repo), "tag", "--list", tag], capture_output=True, text=True, check=True)
    return tag in [x.strip() for x in p.stdout.splitlines()]


def _create_tag(repo: Path, tag: str, message: str) -> None:
    subprocess.run(["git", "-C", str(repo), "tag", "-a", tag, "-m", message], check=True)


def run(cfg: dict[str, Any]) -> None:
    start = time.time()
    project_root = Path(cfg["project_root"]).resolve()
    prereg_dir = Path(cfg["prereg_dir"]).resolve()
    prereg_dir.mkdir(parents=True, exist_ok=True)

    stage43_path = Path(cfg["stage43_results"]).resolve()
    stage43 = json.loads(stage43_path.read_text(encoding="utf-8"))
    alpha = float(stage43["fit_train"]["alpha"])

    methods = [str(x) for x in cfg["methods"]]
    dims = [int(x) for x in cfg["embed_dims"]]
    seeds = [int(x) for x in cfg["seeds"]]

    success = {
        "cell_mean_r_min": float(cfg.get("cell_mean_r_min", 0.85)),
        "cell_mean_mae_max": float(cfg.get("cell_mean_mae_max", 0.01)),
        "geometric_mean_top2_required": True,
    }

    retry_policy = {
        "max_retries_per_failed_unit": int(cfg.get("max_retries_per_failed_unit", 2)),
        "identical_config_only": True,
        "no_silent_replacement": True,
    }

    prediction_rows = []
    for m in methods:
        for d in dims:
            prediction_rows.append(
                {
                    "method": m,
                    "embed_dim": d,
                    "prediction_rule": "R_ia_pred = alpha_locked * sqrt(R_it_obs * R_at_obs)",
                }
            )

    pred_obj = {
        "stage": "stage69_prereg_lock",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "alpha_locked": alpha,
        "alpha_source": {
            "path": str(stage43_path),
            "field": "fit_train.alpha",
        },
        "dataset": {
            "name": str(cfg["speechcoco_dataset"]),
            "train_max_examples": int(cfg["speechcoco_train_max_examples"]),
            "phase_b_val_examples": int(cfg["speechcoco_phase_b_val_examples"]),
            "eval_test_examples": int(cfg["speechcoco_eval_test_examples"]),
            "sample_seed": int(cfg["speechcoco_sample_seed"]),
            "eval_seed": int(cfg["speechcoco_eval_seed"]),
            "phase_b_split_seed": int(cfg["phase_b_split_seed"]),
            "strict_disjoint": bool(cfg.get("strict_disjoint", True)),
        },
        "grid": {
            "methods": methods,
            "embed_dims": dims,
            "seeds": seeds,
            "n_runs": int(len(methods) * len(dims) * len(seeds)),
            "conditions": prediction_rows,
        },
        "model_selection_stage72": {
            "forms": ["geometric_mean", "arithmetic_mean", "hard_min", "product", "power_law_free"],
            "selection_metric": "cv_r2_mean",
            "criterion": "geometric_mean must rank in top-2 by cv_r2_mean",
        },
        "success_criteria": success,
        "failure_retry_policy": retry_policy,
        "notes": [
            "Prospective check uses locked alpha with observed per-condition bridge metrics.",
            "No hyperparameter changes permitted after lock except identical-config reruns for failed units.",
        ],
    }

    prereg_md = Path(prereg_dir / "PREREG_STAGE69.md")
    pred_json = Path(prereg_dir / "predictions_stage69.json")
    sha_txt = Path(prereg_dir / "PREREG_STAGE69_SHA256.txt")
    commit_txt = Path(prereg_dir / "PREREG_STAGE69_COMMIT.txt")

    md_lines = [
        "# PREREG_STAGE69",
        "",
        "## Locked Design",
        f"- Triple: Image–Speech–Text via `{cfg['speechcoco_dataset']}`",
        f"- Methods: {methods}",
        f"- Dims: {dims}",
        f"- Seeds: {seeds}",
        f"- Grid size: {len(methods) * len(dims) * len(seeds)}",
        "",
        "## Locked Prediction Rule",
        f"- alpha_locked: {alpha:.12f} (from Stage43 fit_train.alpha)",
        "- Rule: `R_ia_pred = alpha_locked * sqrt(R_it_obs * R_at_obs)`",
        "",
        "## Success Criteria",
        f"- cell_mean_r >= {success['cell_mean_r_min']}",
        f"- cell_mean_MAE <= {success['cell_mean_mae_max']}",
        "- geometric mean ranks top-2 by CV-R2 in Stage72",
        "",
        "## Failure Policy",
        f"- max retries per failed unit: {retry_policy['max_retries_per_failed_unit']}",
        "- retries must use identical configs",
        "- failed units remain explicit; no silent replacement",
    ]

    prereg_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    pred_json.write_text(json.dumps(pred_obj, indent=2), encoding="utf-8")

    md_hash = _sha256(prereg_md)
    pred_hash = _sha256(pred_json)
    sha_txt.write_text(
        f"{md_hash}  {prereg_md.name}\n{pred_hash}  {pred_json.name}\n",
        encoding="utf-8",
    )

    head = _git_head(project_root)
    commit_txt.write_text(f"HEAD {head}\n", encoding="utf-8")

    tag = str(cfg.get("tag_name", "prereg-stage69-v1"))
    if bool(cfg.get("create_local_tag", True)):
        if not _tag_exists(project_root, tag):
            _create_tag(project_root, tag, "Stage69 prereg lock")

    out = {
        "stage": "stage69_prereg_lock",
        "prereg_md": str(prereg_md),
        "predictions_json": str(pred_json),
        "sha_file": str(sha_txt),
        "commit_file": str(commit_txt),
        "tag": tag,
        "head": head,
        "elapsed_sec": float(time.time() - start),
    }
    out_path = Path(cfg["output_root"]).resolve() / "stage69_prereg_lock"
    out_path.mkdir(parents=True, exist_ok=True)
    save_json(out, out_path / "stage69_prereg_lock.json")

    markers_dir = Path(cfg["output_root"]).resolve() / "markers"
    markers_dir.mkdir(parents=True, exist_ok=True)
    # Write both JSON and plain sentinel marker for compatibility with queue scripts.
    mark_done(markers_dir / "stage69_prereg_locked.done.json", out)
    (markers_dir / "stage69_prereg_locked.done").touch()
    print("stage69_prereg_lock complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json


def _mean(xs: list[float]) -> float:
    return float(np.mean(np.asarray(xs, dtype=np.float64))) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(np.std(np.asarray(xs, dtype=np.float64), ddof=1))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3:
        return 0.0
    xv = x - float(np.mean(x))
    yv = y - float(np.mean(y))
    den = float(np.linalg.norm(xv) * np.linalg.norm(yv))
    if den <= 0:
        return 0.0
    return float(np.dot(xv, yv) / den)


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.mean(np.abs(y - yhat)))


def _fit_alpha_no_intercept(x: np.ndarray, y: np.ndarray) -> float:
    den = float(np.dot(x, x))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / den)


def _linear_fit_with_intercept(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = a + b * x
    return {
        "intercept": a,
        "slope": b,
        "pearson_r": _pearson(x, y),
        "r2": _r2(y, yhat),
        "mae": _mae(y, yhat),
    }


@dataclass
class ModelResult:
    name: str
    params: dict[str, float]
    in_sample_r2: float
    in_sample_mae: float
    heldout_r2: float
    heldout_mae: float
    cv_r2_mean: float
    cv_r2_std: float
    cv_mae_mean: float
    cv_mae_std: float


def _kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return [f.astype(np.int64) for f in folds if len(f) > 0]


def _fit_fixed_form(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    name: str,
) -> tuple[dict[str, float], np.ndarray]:
    alpha = _fit_alpha_no_intercept(x_train, y_train)
    pred = alpha * x_eval
    return {"alpha": alpha, "name": name}, pred


def _fit_powerlaw(
    it_train: np.ndarray,
    at_train: np.ndarray,
    y_train: np.ndarray,
    it_eval: np.ndarray,
    at_eval: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    mask_tr = (it_train > 0) & (at_train > 0) & (y_train > 0)
    if int(mask_tr.sum()) < 10:
        return {"alpha": 0.0, "a": 0.5, "b": 0.5}, np.zeros_like(it_eval)
    Xt = np.column_stack(
        [
            np.ones(int(mask_tr.sum()), dtype=np.float64),
            np.log(it_train[mask_tr]),
            np.log(at_train[mask_tr]),
        ]
    )
    yt = np.log(y_train[mask_tr])
    coef, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
    c, a, b = [float(v) for v in coef]
    alpha = float(math.exp(c))

    mask_ev = (it_eval > 0) & (at_eval > 0)
    pred = np.zeros_like(it_eval)
    pred[mask_ev] = alpha * np.power(it_eval[mask_ev], a) * np.power(at_eval[mask_ev], b)
    return {"alpha": alpha, "a": a, "b": b}, pred


def _bootstrap_powerlaw_ci(
    it: np.ndarray,
    at: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    mask = (it > 0) & (at > 0) & (y > 0)
    itv, atv, yv = it[mask], at[mask], y[mask]
    n = len(itv)
    if n < 20:
        return {"a_lo": 0.0, "a_hi": 0.0, "b_lo": 0.0, "b_hi": 0.0}
    rng = np.random.default_rng(seed)
    as_, bs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        params, _ = _fit_powerlaw(itv[idx], atv[idx], yv[idx], itv, atv)
        as_.append(params["a"])
        bs.append(params["b"])
    a_arr = np.asarray(as_, dtype=np.float64)
    b_arr = np.asarray(bs, dtype=np.float64)
    return {
        "a_lo": float(np.quantile(a_arr, 0.025)),
        "a_hi": float(np.quantile(a_arr, 0.975)),
        "b_lo": float(np.quantile(b_arr, 0.025)),
        "b_hi": float(np.quantile(b_arr, 0.975)),
    }


def _functional_forms_analysis(
    records: list[dict[str, Any]],
    *,
    holdout_source: str,
    kfold: int,
    cv_seed: int,
    boot_n: int,
) -> dict[str, Any]:
    it = np.asarray([float(r["av_it_ood"]) for r in records], dtype=np.float64)
    at = np.asarray([float(r["av_at"]) for r in records], dtype=np.float64)
    ia = np.asarray([float(r["av_ia"]) for r in records], dtype=np.float64)
    src = np.asarray([str(r["source_id"]) for r in records], dtype=object)

    train_mask = src != holdout_source
    test_mask = src == holdout_source
    if int(train_mask.sum()) < 20 or int(test_mask.sum()) < 10:
        raise RuntimeError("Insufficient train/test split for functional-form analysis")

    def run_one(
        name: str,
        fit_fn: Callable[..., tuple[dict[str, float], np.ndarray]],
        pred_fn_all_inputs: dict[str, Any] | None = None,
    ) -> ModelResult:
        if name == "power_law_free":
            params_all, pred_all = _fit_powerlaw(it, at, ia, it, at)
            params_hold, pred_hold = _fit_powerlaw(it[train_mask], at[train_mask], ia[train_mask], it[test_mask], at[test_mask])
        else:
            x_map = {
                "geometric_mean": np.sqrt(np.clip(it * at, 0.0, None)),
                "arithmetic_mean": 0.5 * (it + at),
                "hard_min": np.minimum(it, at),
                "product": it * at,
            }
            x = x_map[name]
            p_all, pred_all = _fit_fixed_form(x, ia, x, name=name)
            params_all = p_all
            p_hold, pred_hold = _fit_fixed_form(x[train_mask], ia[train_mask], x[test_mask], name=name)
            params_hold = p_hold

        # CV metrics
        folds = _kfold_indices(len(records), kfold, cv_seed)
        cv_r2, cv_mae = [], []
        for fold in folds:
            tr_mask = np.ones(len(records), dtype=bool)
            tr_mask[fold] = False
            if name == "power_law_free":
                _, pred = _fit_powerlaw(it[tr_mask], at[tr_mask], ia[tr_mask], it[fold], at[fold])
            else:
                x_map = {
                    "geometric_mean": np.sqrt(np.clip(it * at, 0.0, None)),
                    "arithmetic_mean": 0.5 * (it + at),
                    "hard_min": np.minimum(it, at),
                    "product": it * at,
                }
                x = x_map[name]
                _, pred = _fit_fixed_form(x[tr_mask], ia[tr_mask], x[fold], name=name)
            cv_r2.append(_r2(ia[fold], pred))
            cv_mae.append(_mae(ia[fold], pred))

        # heldout metrics
        y_hold = ia[test_mask]
        heldout_r2 = _r2(y_hold, pred_hold)
        heldout_mae = _mae(y_hold, pred_hold)

        # in-sample metrics (fit on all)
        ins_r2 = _r2(ia, pred_all)
        ins_mae = _mae(ia, pred_all)
        return ModelResult(
            name=name,
            params=params_all,
            in_sample_r2=ins_r2,
            in_sample_mae=ins_mae,
            heldout_r2=heldout_r2,
            heldout_mae=heldout_mae,
            cv_r2_mean=_mean(cv_r2),
            cv_r2_std=_std(cv_r2),
            cv_mae_mean=_mean(cv_mae),
            cv_mae_std=_std(cv_mae),
        )

    models = [
        run_one("geometric_mean", _fit_fixed_form),
        run_one("arithmetic_mean", _fit_fixed_form),
        run_one("hard_min", _fit_fixed_form),
        run_one("product", _fit_fixed_form),
        run_one("power_law_free", _fit_powerlaw),
    ]

    power = next(m for m in models if m.name == "power_law_free")
    ci = _bootstrap_powerlaw_ci(
        it[train_mask],
        at[train_mask],
        ia[train_mask],
        n_boot=boot_n,
        seed=cv_seed + 99,
    )

    return {
        "holdout_source": holdout_source,
        "n_total": int(len(records)),
        "n_train": int(train_mask.sum()),
        "n_holdout": int(test_mask.sum()),
        "models": [
            {
                "name": m.name,
                "params": m.params,
                "in_sample_r2": m.in_sample_r2,
                "in_sample_mae": m.in_sample_mae,
                "heldout_r2": m.heldout_r2,
                "heldout_mae": m.heldout_mae,
                "cv_r2_mean": m.cv_r2_mean,
                "cv_r2_std": m.cv_r2_std,
                "cv_mae_mean": m.cv_mae_mean,
                "cv_mae_std": m.cv_mae_std,
            }
            for m in models
        ],
        "power_law_free_ci95": ci,
    }


def _w1_dim_conditional_alpha(stage44: dict[str, Any], stage36: dict[str, Any]) -> dict[str, Any]:
    raw = stage44["raw"]
    dims = sorted(int(str(k).lstrip("m")) for k in raw.keys())
    methods = ["audio_linear_probe", "modular_shared_jl"]
    out_rows = []
    for m in dims:
        rows_by_m = raw[f"m{m}"]
        row = {"embed_dim": m}
        for method in methods:
            recs = rows_by_m[method]
            x = np.asarray([math.sqrt(max(0.0, float(r["av_it_avg_R"]) * float(r["av_at_avg_R"]))) for r in recs], dtype=np.float64)
            y = np.asarray([float(r["av_ia_avg_R"]) for r in recs], dtype=np.float64)
            alpha = _fit_alpha_no_intercept(x, y)
            row[f"alpha_{method}"] = float(alpha)
            row[f"mean_ia_{method}"] = float(np.mean(y))
            row[f"mean_at_{method}"] = float(np.mean([float(r["av_at_avg_R"]) for r in recs]))
            row[f"mean_it_{method}"] = float(np.mean([float(r["av_it_avg_R"]) for r in recs]))
        lp = row["alpha_audio_linear_probe"]
        jl = row["alpha_modular_shared_jl"]
        row["alpha_ratio_lp_over_jl"] = float(lp / jl) if jl > 0 else 0.0
        row["observed_ia_ratio_lp_over_jl"] = float(
            row["mean_ia_audio_linear_probe"] / row["mean_ia_modular_shared_jl"]
        ) if row["mean_ia_modular_shared_jl"] > 0 else 0.0
        row["predicted_ratio_conditional"] = float(
            row["alpha_ratio_lp_over_jl"]
            * math.sqrt(
                max(
                    0.0,
                    row["mean_it_audio_linear_probe"] * row["mean_at_audio_linear_probe"]
                    / max(1e-12, row["mean_it_modular_shared_jl"] * row["mean_at_modular_shared_jl"]),
                )
            )
        )
        out_rows.append(row)

    global_lp = float(stage36["by_method"]["audio_linear_probe"]["alpha"])
    global_jl = float(stage36["by_method"]["modular_shared_jl"]["alpha"])
    row512 = next(r for r in out_rows if int(r["embed_dim"]) == 512)
    predicted_ratio_global_512 = (global_lp / global_jl) * math.sqrt(
        max(
            0.0,
            row512["mean_it_audio_linear_probe"] * row512["mean_at_audio_linear_probe"]
            / max(1e-12, row512["mean_it_modular_shared_jl"] * row512["mean_at_modular_shared_jl"]),
        )
    )
    return {
        "global_alpha_lp": global_lp,
        "global_alpha_jl": global_jl,
        "global_alpha_ratio_lp_over_jl": float(global_lp / global_jl),
        "rows": out_rows,
        "m512_predicted_ratio_global_alpha": float(predicted_ratio_global_512),
        "m512_predicted_ratio_conditional_alpha": float(row512["predicted_ratio_conditional"]),
        "m512_observed_ratio": float(row512["observed_ia_ratio_lp_over_jl"]),
    }


def _build_rk_table(stage44: dict[str, Any], stage20: dict[str, Any], stage30: dict[str, Any]) -> dict[str, Any]:
    def _mean_metric(recs: list[dict[str, Any]], metric_path: tuple[str, str]) -> float:
        a, b = metric_path
        vals = [float(r[a][b]) for r in recs]
        return float(np.mean(np.asarray(vals, dtype=np.float64)))

    out_rows = []
    # Modular rows from stage44 COCO-matched run
    for method in ["audio_linear_probe", "modular_shared_jl"]:
        recs = stage44["raw"]["m512"][method]
        out_rows.append(
            {
                "source": "stage44_coco_phaseA",
                "method": method,
                "R@1": _mean_metric(recs, ("av_image_audio", "i2t_R@1")),
                "R@5": _mean_metric(recs, ("av_image_audio", "i2t_R@5")),
                "R@10": _mean_metric(recs, ("av_image_audio", "i2t_R@10")),
                "avg_R": float(np.mean([float(r["av_ia_avg_R"]) for r in recs])),
            }
        )

    # LoRA proxy from Stage30 baseline comparison
    recs_lora = stage30["raw"]["m512"]["audio_text_lora_proxy"]
    out_rows.append(
        {
            "source": "stage30_projection_type",
            "method": "audio_text_lora_proxy",
            "R@1": _mean_metric(recs_lora, ("av_image_audio", "i2t_R@1")),
            "R@5": _mean_metric(recs_lora, ("av_image_audio", "i2t_R@5")),
            "R@10": _mean_metric(recs_lora, ("av_image_audio", "i2t_R@10")),
            "avg_R": float(np.mean([float(r["av_ia_avg_R"]) for r in recs_lora])),
        }
    )

    # Joint references from stage20 aggregate
    for method in ["joint_clip_head", "joint_shared_jl"]:
        recs = stage20["raw"]["m512"][method]
        out_rows.append(
            {
                "source": "stage20_joint_reference",
                "method": method,
                "R@1": _mean_metric(recs, ("av_image_audio", "i2t_R@1")),
                "R@5": _mean_metric(recs, ("av_image_audio", "i2t_R@5")),
                "R@10": _mean_metric(recs, ("av_image_audio", "i2t_R@10")),
                "avg_R": float(np.mean([float(r["av_ia_avg_R"]) for r in recs])),
            }
        )
    return {"rows": out_rows}


def _w9_encoder_decomposition(stage44: dict[str, Any], stage20: dict[str, Any], stage36: dict[str, Any]) -> dict[str, Any]:
    # Compare modular_shared_jl (stage44) vs joint_clip_head (stage20)
    mod = stage44["raw"]["m512"]["modular_shared_jl"]
    joint = stage20["raw"]["m512"]["joint_clip_head"]
    mod_it = float(np.mean([float(r["av_it_avg_R"]) for r in mod]))
    mod_at = float(np.mean([float(r["av_at_avg_R"]) for r in mod]))
    mod_ia = float(np.mean([float(r["av_ia_avg_R"]) for r in mod]))
    j_it = float(np.mean([float(r["av_it_avg_R"]) for r in joint]))
    j_at = float(np.mean([float(r["av_at_avg_R"]) for r in joint]))
    j_ia = float(np.mean([float(r["av_ia_avg_R"]) for r in joint]))

    observed_ratio = mod_ia / j_ia if j_ia > 0 else 0.0
    bridge_ratio = math.sqrt(max(0.0, (mod_it * mod_at) / max(1e-12, (j_it * j_at))))
    implied_alpha_ratio = observed_ratio / bridge_ratio if bridge_ratio > 0 else 0.0

    # LP vs modular shared within stage44 (matched encoder family, supervision fixed)
    lp = stage44["raw"]["m512"]["audio_linear_probe"]
    lp_it = float(np.mean([float(r["av_it_avg_R"]) for r in lp]))
    lp_at = float(np.mean([float(r["av_at_avg_R"]) for r in lp]))
    lp_ia = float(np.mean([float(r["av_ia_avg_R"]) for r in lp]))
    lp_over_mod_obs = lp_ia / mod_ia if mod_ia > 0 else 0.0
    lp_over_mod_bridge = math.sqrt(max(0.0, (lp_it * lp_at) / max(1e-12, (mod_it * mod_at))))
    lp_over_mod_alpha_imp = lp_over_mod_obs / lp_over_mod_bridge if lp_over_mod_bridge > 0 else 0.0

    return {
        "modular_shared_jl_vs_joint_clip_head": {
            "mod_it": mod_it,
            "mod_at": mod_at,
            "mod_ia": mod_ia,
            "joint_it": j_it,
            "joint_at": j_at,
            "joint_ia": j_ia,
            "observed_ratio_mod_over_joint": observed_ratio,
            "bridge_ratio_sqrt_itat": bridge_ratio,
            "implied_alpha_ratio": implied_alpha_ratio,
        },
        "audio_linear_probe_vs_modular_shared_jl_same_stage44": {
            "lp_it": lp_it,
            "lp_at": lp_at,
            "lp_ia": lp_ia,
            "mod_it": mod_it,
            "mod_at": mod_at,
            "mod_ia": mod_ia,
            "observed_ratio_lp_over_mod": lp_over_mod_obs,
            "bridge_ratio_sqrt_itat": lp_over_mod_bridge,
            "implied_alpha_ratio": lp_over_mod_alpha_imp,
            "global_alpha_lp": float(stage36["by_method"]["audio_linear_probe"]["alpha"]),
            "global_alpha_modular_shared_jl": float(stage36["by_method"]["modular_shared_jl"]["alpha"]),
        },
    }


def _w6_margin_regression(
    stage38_base: dict[str, Any],
    stage38_mixed46k: dict[str, Any],
    stage38_clean46k: dict[str, Any],
    stage44: dict[str, Any],
    stage56: dict[str, Any],
) -> dict[str, Any]:
    # margin points
    margin_points: dict[str, float] = {
        "audiocaps": float(stage38_base["audiocaps"]["clap_similarity"]["margin_mean"]),
        "mixed200k": float(stage38_base["wavcaps"]["clap_similarity"]["margin_mean"]),
        "clean_source": float(stage38_base["wavcaps_clean_source"]["clap_similarity"]["margin_mean"]),
        "mixed46k": float(stage38_mixed46k["wavcaps"]["clap_similarity"]["margin_mean"]),
        "clean_source_46k": float(stage38_clean46k["wavcaps"]["clap_similarity"]["margin_mean"]),
    }
    # retrieval points
    at_ia: dict[str, tuple[float, float]] = {}
    mod44 = stage44["raw"]["m512"]["modular_shared_jl"]
    at_ia["audiocaps"] = (
        float(np.mean([float(r["av_at_avg_R"]) for r in mod44])),
        float(np.mean([float(r["av_ia_avg_R"]) for r in mod44])),
    )
    for row in stage56["rows"]:
        at_ia[str(row["condition"])] = (float(row["av_at_mean"]), float(row["av_ia_mean"]))

    xs, ys_at, ys_ia, names = [], [], [], []
    for name in ["audiocaps", "mixed200k", "mixed46k", "clean_source", "clean_source_46k"]:
        if name not in margin_points or name not in at_ia:
            continue
        xs.append(float(margin_points[name]))
        ys_at.append(float(at_ia[name][0]))
        ys_ia.append(float(at_ia[name][1]))
        names.append(name)
    x = np.asarray(xs, dtype=np.float64)
    y_at = np.asarray(ys_at, dtype=np.float64)
    y_ia = np.asarray(ys_ia, dtype=np.float64)
    reg_at = _linear_fit_with_intercept(x, y_at)
    reg_ia = _linear_fit_with_intercept(x, y_ia)

    return {
        "points": [
            {
                "condition": n,
                "clap_margin_mean": float(mx),
                "av_at_avg_R": float(at),
                "av_ia_avg_R": float(ia),
            }
            for n, mx, at, ia in zip(names, xs, ys_at, ys_ia)
        ],
        "regression_margin_to_av_at": reg_at,
        "regression_margin_to_av_ia": reg_ia,
    }


def run(cfg: dict[str, Any]) -> None:
    start = time.time()
    out_root = Path(cfg["output_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stage_root = out_root / "stage59_paper_revision_analyses"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = out_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    # Inputs
    stage44 = load_json(Path(cfg["stage44_results"]).resolve())
    stage20 = load_json(Path(cfg["stage20_results"]).resolve())
    stage30 = load_json(Path(cfg["stage30_results"]).resolve())
    stage36 = load_json(Path(cfg["stage36_results"]).resolve())
    stage56 = load_json(Path(cfg["stage56_results"]).resolve())
    stage38_base = load_json(Path(cfg["stage38_base_results"]).resolve())
    stage38_mixed46k = load_json(Path(cfg["stage38_mixed46k_results"]).resolve())
    stage38_clean46k = load_json(Path(cfg["stage38_clean46k_results"]).resolve())

    # W1
    w1 = _w1_dim_conditional_alpha(stage44, stage36)

    # W2 + W4 from stage36 records
    records = stage36["records"]
    forms = _functional_forms_analysis(
        records,
        holdout_source=str(cfg.get("holdout_source_id", "stage31_wavcaps_scaling")),
        kfold=int(cfg.get("kfold", 5)),
        cv_seed=int(cfg.get("cv_seed", 2026)),
        boot_n=int(cfg.get("powerlaw_bootstrap_n", 1000)),
    )

    # W6
    w6 = _w6_margin_regression(stage38_base, stage38_mixed46k, stage38_clean46k, stage44, stage56)

    # W7
    w7 = _build_rk_table(stage44, stage20, stage30)

    # W9
    w9 = _w9_encoder_decomposition(stage44, stage20, stage36)

    out = {
        "stage": "stage59_paper_revision_analyses",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "w1_dim_conditional_alpha": w1,
        "w2_w4_functional_forms": forms,
        "w6_margin_regression": w6,
        "w7_r_at_k_table": w7,
        "w9_encoder_supervision_decomposition": w9,
        "elapsed_sec": float(time.time() - start),
    }
    save_json(out, stage_root / "stage59_paper_revision_analyses.json")

    # Markdown summary
    lines = [
        "# Stage59 Paper Revision Analyses",
        "",
        "## W1: Dimension-Conditional α (LP vs JL)",
        "",
        f"- Global α ratio (LP/JL): {w1['global_alpha_ratio_lp_over_jl']:.3f}",
        f"- m=512 observed LP/JL ratio: {w1['m512_observed_ratio']:.3f}",
        f"- m=512 predicted ratio (global α): {w1['m512_predicted_ratio_global_alpha']:.3f}",
        f"- m=512 predicted ratio (conditional α): {w1['m512_predicted_ratio_conditional_alpha']:.3f}",
        "",
        "| m | α(LP) | α(JL) | α ratio LP/JL | observed ia ratio LP/JL | predicted ratio (cond α) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in w1["rows"]:
        lines.append(
            f"| {r['embed_dim']} | {r['alpha_audio_linear_probe']:.4f} | {r['alpha_modular_shared_jl']:.4f} | "
            f"{r['alpha_ratio_lp_over_jl']:.3f} | {r['observed_ia_ratio_lp_over_jl']:.3f} | {r['predicted_ratio_conditional']:.3f} |"
        )

    lines += [
        "",
        "## W2/W4: Functional-Form Comparison",
        "",
        "| Form | In-sample R² | In-sample MAE | Held-out R² | Held-out MAE | CV R² (mean±std) | CV MAE (mean±std) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in forms["models"]:
        lines.append(
            f"| {m['name']} | {m['in_sample_r2']:.4f} | {m['in_sample_mae']:.5f} | "
            f"{m['heldout_r2']:.4f} | {m['heldout_mae']:.5f} | "
            f"{m['cv_r2_mean']:.4f}±{m['cv_r2_std']:.4f} | {m['cv_mae_mean']:.5f}±{m['cv_mae_std']:.5f} |"
        )
    lines += [
        "",
        f"- Free power-law CI95: a ∈ [{forms['power_law_free_ci95']['a_lo']:.3f}, {forms['power_law_free_ci95']['a_hi']:.3f}], "
        f"b ∈ [{forms['power_law_free_ci95']['b_lo']:.3f}, {forms['power_law_free_ci95']['b_hi']:.3f}]",
        "",
        "## W6: Margin → Performance",
        "",
        "| Condition | CLAP margin | av_at | av_ia |",
        "|---|---:|---:|---:|",
    ]
    for p in w6["points"]:
        lines.append(
            f"| {p['condition']} | {p['clap_margin_mean']:.4f} | {p['av_at_avg_R']:.4f} | {p['av_ia_avg_R']:.4f} |"
        )
    reg_at = w6["regression_margin_to_av_at"]
    reg_ia = w6["regression_margin_to_av_ia"]
    lines += [
        "",
        f"- margin→av_at: r={reg_at['pearson_r']:.4f}, slope={reg_at['slope']:.4f}",
        f"- margin→av_ia: r={reg_ia['pearson_r']:.4f}, slope={reg_ia['slope']:.4f}",
        "",
        "## W7: R@k comparability table (m=512)",
        "",
        "| Method | Source | R@1 | R@5 | R@10 | avg_R |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in w7["rows"]:
        lines.append(
            f"| {r['method']} | {r['source']} | {r['R@1']:.4f} | {r['R@5']:.4f} | {r['R@10']:.4f} | {r['avg_R']:.4f} |"
        )

    d = w9["modular_shared_jl_vs_joint_clip_head"]
    lines += [
        "",
        "## W9: Encoder/Supervision Decomposition",
        "",
        f"- modular_shared_jl vs joint_clip_head observed ratio: {d['observed_ratio_mod_over_joint']:.3f}",
        f"- bridge ratio sqrt((it·at)_mod/(it·at)_joint): {d['bridge_ratio_sqrt_itat']:.3f}",
        f"- implied α ratio (observed / bridge): {d['implied_alpha_ratio']:.3f}",
    ]

    (stage_root / "stage59_paper_revision_analyses.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={
            "stage": "stage59_paper_revision_analyses",
            "elapsed_sec": float(time.time() - start),
            "inputs": {
                "stage44": cfg["stage44_results"],
                "stage20": cfg["stage20_results"],
                "stage30": cfg["stage30_results"],
                "stage36": cfg["stage36_results"],
                "stage38_base": cfg["stage38_base_results"],
                "stage38_mixed46k": cfg["stage38_mixed46k_results"],
                "stage38_clean46k": cfg["stage38_clean46k_results"],
                "stage56": cfg["stage56_results"],
            },
        },
    )
    save_json(provenance, stage_root / "provenance_stage59.json")
    mark_done(markers / "stage59_paper_revision_analyses.done.json", {"elapsed_sec": float(time.time() - start)})
    print("stage59_paper_revision_analyses complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)


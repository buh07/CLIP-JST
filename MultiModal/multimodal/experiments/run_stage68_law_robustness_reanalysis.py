from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json


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


def _kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return [f.astype(np.int64) for f in folds if len(f) > 0]


def _fit_power_law(
    it_train: np.ndarray,
    at_train: np.ndarray,
    y_train: np.ndarray,
    it_eval: np.ndarray,
    at_eval: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    mask = (it_train > 0) & (at_train > 0) & (y_train > 0)
    if int(mask.sum()) < 8:
        return {"alpha": 0.0, "a": 0.5, "b": 0.5}, np.zeros_like(it_eval)
    X = np.column_stack(
        [
            np.ones(int(mask.sum()), dtype=np.float64),
            np.log(it_train[mask]),
            np.log(at_train[mask]),
        ]
    )
    yy = np.log(y_train[mask])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    c, a, b = [float(v) for v in coef]
    alpha = float(math.exp(c))
    pred = np.zeros_like(it_eval)
    mask_ev = (it_eval > 0) & (at_eval > 0)
    pred[mask_ev] = alpha * np.power(it_eval[mask_ev], a) * np.power(at_eval[mask_ev], b)
    return {"alpha": alpha, "a": a, "b": b}, pred


def _fit_fixed_form(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    alpha = _fit_alpha_no_intercept(x_train, y_train)
    return {"alpha": alpha}, alpha * x_eval


def _predict_form(
    form_name: str,
    it_train: np.ndarray,
    at_train: np.ndarray,
    y_train: np.ndarray,
    it_eval: np.ndarray,
    at_eval: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    if form_name == "power_law_free":
        return _fit_power_law(it_train, at_train, y_train, it_eval, at_eval)
    x_train_map = {
        "geometric_mean": np.sqrt(np.clip(it_train * at_train, 0.0, None)),
        "arithmetic_mean": 0.5 * (it_train + at_train),
        "hard_min": np.minimum(it_train, at_train),
        "product": it_train * at_train,
    }
    x_eval_map = {
        "geometric_mean": np.sqrt(np.clip(it_eval * at_eval, 0.0, None)),
        "arithmetic_mean": 0.5 * (it_eval + at_eval),
        "hard_min": np.minimum(it_eval, at_eval),
        "product": it_eval * at_eval,
    }
    return _fit_fixed_form(x_train_map[form_name], y_train, x_eval_map[form_name])


def _bootstrap_powerlaw_ci(records: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, float]:
    it = np.asarray([float(r["av_it_ood"]) for r in records], dtype=np.float64)
    at = np.asarray([float(r["av_at"]) for r in records], dtype=np.float64)
    ia = np.asarray([float(r["av_ia"]) for r in records], dtype=np.float64)
    mask = (it > 0) & (at > 0) & (ia > 0)
    it, at, ia = it[mask], at[mask], ia[mask]
    n = len(it)
    if n < 8:
        return {"a_lo": 0.0, "a_hi": 0.0, "b_lo": 0.0, "b_hi": 0.0}
    rng = np.random.default_rng(seed)
    as_, bs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        params, _ = _fit_power_law(it[idx], at[idx], ia[idx], it, at)
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


def _evaluate_forms(
    records: list[dict[str, Any]],
    *,
    suite_name: str,
    kfold: int,
    cv_seed: int,
    boot_n: int,
    holdout_source: str | None = None,
) -> dict[str, Any]:
    it = np.asarray([float(r["av_it_ood"]) for r in records], dtype=np.float64)
    at = np.asarray([float(r["av_at"]) for r in records], dtype=np.float64)
    ia = np.asarray([float(r["av_ia"]) for r in records], dtype=np.float64)
    methods = np.asarray([str(r.get("method", "unknown")) for r in records], dtype=object)
    dims = np.asarray([int(r.get("embed_dim", 0)) for r in records], dtype=np.int64)
    sources = np.asarray([str(r.get("source_id", "unknown")) for r in records], dtype=object)

    model_names = ["geometric_mean", "arithmetic_mean", "hard_min", "product", "power_law_free"]
    models: list[dict[str, Any]] = []

    for name in model_names:
        params_all, pred_all = _predict_form(name, it, at, ia, it, at)

        folds = _kfold_indices(len(records), max(2, min(kfold, len(records))), cv_seed)
        cv_r2, cv_mae = [], []
        for fold in folds:
            tr_mask = np.ones(len(records), dtype=bool)
            tr_mask[fold] = False
            _, pred = _predict_form(name, it[tr_mask], at[tr_mask], ia[tr_mask], it[fold], at[fold])
            cv_r2.append(_r2(ia[fold], pred))
            cv_mae.append(_mae(ia[fold], pred))

        lomo_r2 = None
        if len(set(methods.tolist())) > 1:
            pred = np.zeros_like(ia)
            for m in sorted(set(methods.tolist())):
                te = methods == m
                tr = ~te
                _, p = _predict_form(name, it[tr], at[tr], ia[tr], it[te], at[te])
                pred[te] = p
            lomo_r2 = _r2(ia, pred)

        lodo_r2 = None
        if len(set(dims.tolist())) > 1:
            pred = np.zeros_like(ia)
            for d in sorted(set(dims.tolist())):
                te = dims == d
                tr = ~te
                _, p = _predict_form(name, it[tr], at[tr], ia[tr], it[te], at[te])
                pred[te] = p
            lodo_r2 = _r2(ia, pred)

        holdout_r2 = None
        holdout_mae = None
        if holdout_source is not None and holdout_source in set(sources.tolist()):
            te = sources == holdout_source
            tr = ~te
            if int(tr.sum()) >= 8 and int(te.sum()) >= 2:
                _, p = _predict_form(name, it[tr], at[tr], ia[tr], it[te], at[te])
                holdout_r2 = _r2(ia[te], p)
                holdout_mae = _mae(ia[te], p)

        model_row: dict[str, Any] = {
            "name": name,
            "params": params_all,
            "in_sample_r2": _r2(ia, pred_all),
            "in_sample_mae": _mae(ia, pred_all),
            "cv_r2_mean": float(np.mean(cv_r2)) if cv_r2 else 0.0,
            "cv_r2_std": float(np.std(cv_r2, ddof=1)) if len(cv_r2) > 1 else 0.0,
            "cv_mae_mean": float(np.mean(cv_mae)) if cv_mae else 0.0,
            "cv_mae_std": float(np.std(cv_mae, ddof=1)) if len(cv_mae) > 1 else 0.0,
            "lomo_r2": lomo_r2,
            "lodo_r2": lodo_r2,
            "heldout_r2": holdout_r2,
            "heldout_mae": holdout_mae,
        }
        models.append(model_row)

    models_sorted = sorted(models, key=lambda m: m["cv_r2_mean"], reverse=True)
    out = {
        "suite": suite_name,
        "n": len(records),
        "n_methods": len(set(methods.tolist())),
        "n_dims": len(set(dims.tolist())),
        "n_sources": len(set(sources.tolist())),
        "models": models,
        "best_by_cv_r2": models_sorted[0]["name"] if models_sorted else None,
        "power_law_free_ci95": _bootstrap_powerlaw_ci(records, n_boot=boot_n, seed=cv_seed + 17),
    }
    return out


def _group_cell_means(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    acc: dict[tuple[str, str, int], dict[str, list[float] | str | int]] = {}
    for r in records:
        key = (str(r["source_id"]), str(r["method"]), int(r["embed_dim"]))
        slot = acc.setdefault(
            key,
            {
                "source_id": key[0],
                "method": key[1],
                "embed_dim": key[2],
                "av_it_ood": [],
                "av_at": [],
                "av_ia": [],
            },
        )
        slot["av_it_ood"].append(float(r["av_it_ood"]))  # type: ignore[index]
        slot["av_at"].append(float(r["av_at"]))  # type: ignore[index]
        slot["av_ia"].append(float(r["av_ia"]))  # type: ignore[index]

    out = []
    for slot in acc.values():
        out.append(
            {
                "source_id": str(slot["source_id"]),
                "method": str(slot["method"]),
                "embed_dim": int(slot["embed_dim"]),
                "av_it_ood": float(np.mean(np.asarray(slot["av_it_ood"], dtype=np.float64))),  # type: ignore[arg-type]
                "av_at": float(np.mean(np.asarray(slot["av_at"], dtype=np.float64))),  # type: ignore[arg-type]
                "av_ia": float(np.mean(np.asarray(slot["av_ia"], dtype=np.float64))),  # type: ignore[arg-type]
            }
        )
    return out


def _fit_basic_law(records: list[dict[str, Any]]) -> dict[str, float]:
    it = np.asarray([float(r["av_it_ood"]) for r in records], dtype=np.float64)
    at = np.asarray([float(r["av_at"]) for r in records], dtype=np.float64)
    ia = np.asarray([float(r["av_ia"]) for r in records], dtype=np.float64)
    x = np.sqrt(np.clip(it * at, 0.0, None))
    alpha = _fit_alpha_no_intercept(x, ia)
    pred = alpha * x
    return {"n": float(len(records)), "alpha": alpha, "pearson_r": _pearson(ia, pred), "r2": _r2(ia, pred), "mae": _mae(ia, pred)}


def _cluster_and_mixed_effects(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for r in records:
        it = float(r["av_it_ood"])
        at = float(r["av_at"])
        ia = float(r["av_ia"])
        rows.append(
            {
                "y": ia,
                "x": math.sqrt(max(it * at, 0.0)),
                "cell_id": f"{r['source_id']}|{r['method']}|m{int(r['embed_dim'])}",
            }
        )
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {"n_seed_rows": int(len(df)), "n_cells": int(df["cell_id"].nunique())}

    # Cluster-robust OLS (clusters by cell)
    try:
        X = df[["x"]]
        y = df["y"]
        ols = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["cell_id"]})
        ci = ols.conf_int().loc["x"].tolist()
        out["cluster_robust_ols"] = {
            "alpha": float(ols.params["x"]),
            "std_err": float(ols.bse["x"]),
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
            "r2": float(ols.rsquared),
        }
    except Exception as e:  # pragma: no cover
        out["cluster_robust_ols_error"] = str(e)

    # Mixed effects: random intercept by cell
    try:
        md = smf.mixedlm("y ~ 0 + x", df, groups=df["cell_id"])
        mdf = md.fit(reml=False, method="lbfgs", maxiter=400, disp=False)
        ci = mdf.conf_int().loc["x"].tolist()
        out["mixed_effects"] = {
            "alpha": float(mdf.params["x"]),
            "std_err": float(mdf.bse["x"]),
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
            "loglik": float(mdf.llf),
            "converged": bool(getattr(mdf, "converged", True)),
        }
    except Exception as e:  # pragma: no cover
        out["mixed_effects_error"] = str(e)

    return out


def _collect_stage58_seed_records(stage58_path: Path) -> list[dict[str, Any]]:
    stage58 = load_json(stage58_path)
    roots = [Path(x).resolve() for x in stage58["stage57_shard_roots"]]
    merged: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for root in roots:
        p = root / "stage57_second_triple_avcaps" / "stage57_second_triple_avcaps_results.json"
        block = load_json(p)
        for m_key, methods in block.get("raw", {}).items():
            dst_m = merged.setdefault(m_key, {})
            for method, rows in methods.items():
                dst_r = dst_m.setdefault(method, {})
                for r in rows:
                    dst_r[int(r["seed"])] = r
    out: list[dict[str, Any]] = []
    for m_key, methods in merged.items():
        embed_dim = int(str(m_key).lstrip("m"))
        for method, by_seed in methods.items():
            for seed in sorted(by_seed):
                r = by_seed[seed]
                out.append(
                    {
                        "source_id": "stage58_avcaps_full_grid",
                        "method": str(method),
                        "embed_dim": embed_dim,
                        "seed": int(seed),
                        "av_it_ood": float(r["av_it_avg_R"]),
                        "av_at": float(r["av_at_avg_R"]),
                        "av_ia": float(r["av_ia_avg_R"]),
                    }
                )
    return out


def _collect_stage55_rows(stage55_results_path: Path, source_id: str) -> list[dict[str, Any]]:
    obj = load_json(stage55_results_path)
    out: list[dict[str, Any]] = []
    for m_key, methods in obj.get("raw", {}).items():
        embed_dim = int(str(m_key).lstrip("m"))
        for method, rows in methods.items():
            for r in rows:
                out.append(
                    {
                        "source_id": source_id,
                        "method": str(method),
                        "embed_dim": embed_dim,
                        "seed": int(r["seed"]),
                        "av_it_ood": float(r["av_it_avg_R"]),
                        "av_at": float(r["av_at_avg_R"]),
                        "av_ia": float(r["av_ia_avg_R"]),
                    }
                )
    return out


def _collect_stage63_clotho_proxy(stage63_path: Path) -> list[dict[str, Any]]:
    obj = load_json(stage63_path)
    out: list[dict[str, Any]] = []
    for condition, dims in obj.get("raw", {}).items():
        for m_key, methods in dims.items():
            embed_dim = int(str(m_key).lstrip("m"))
            for method, rows in methods.items():
                for r in rows:
                    eval_path = Path(str(r["source_eval_path"]))
                    ev = load_json(eval_path)
                    av_it = float(ev["av_image_text"]["avg_R"])
                    av_ia = float(ev["av_image_audio"]["avg_R"])
                    out.append(
                        {
                            "source_id": f"stage63_clotho_proxy:{condition}",
                            "method": str(method),
                            "embed_dim": embed_dim,
                            "seed": int(r["seed"]),
                            "av_it_ood": av_it,
                            "av_at": float(r["clotho_at_avg_R"]),
                            "av_ia": av_ia,
                        }
                    )
    return out


def _alpha_locked_prediction(alpha: float, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0, "error": "no records"}
    y = np.asarray([float(r["av_ia"]) for r in records], dtype=np.float64)
    x = np.sqrt(
        np.clip(
            np.asarray([float(r["av_it_ood"]) for r in records], dtype=np.float64)
            * np.asarray([float(r["av_at"]) for r in records], dtype=np.float64),
            0.0,
            None,
        )
    )
    pred = alpha * x
    return {"n": int(len(records)), "alpha_fixed": float(alpha), "pearson_r": _pearson(y, pred), "r2": _r2(y, pred), "mae": _mae(y, pred)}


def _phase_a_gap_feasibility(stage47_path: Path, stage44_path: Path) -> dict[str, Any]:
    s47 = load_json(stage47_path)
    s44 = load_json(stage44_path)

    gap_by_key: dict[tuple[int, int], float] = {}
    for m_key, methods in s47.get("raw", {}).items():
        embed_dim = int(str(m_key).lstrip("m"))
        rows = methods.get("modular_shared_jl", [])
        for r in rows:
            seed = int(r["seed"])
            gap = float(r["diagnostics"]["av"]["ia_modality_gap_l2"])
            gap_by_key[(embed_dim, seed)] = gap

    alpha_rows: list[dict[str, Any]] = []
    for m_key, methods in s44.get("raw", {}).items():
        embed_dim = int(str(m_key).lstrip("m"))
        rows = methods.get("modular_shared_jl", [])
        for r in rows:
            seed = int(r["seed"])
            key = (embed_dim, seed)
            if key not in gap_by_key:
                continue
            it = float(r["av_it_avg_R"])
            at = float(r["av_at_avg_R"])
            ia = float(r["av_ia_avg_R"])
            den = math.sqrt(max(it * at, 0.0))
            alpha = float(ia / den) if den > 1e-12 else 0.0
            alpha_rows.append(
                {
                    "embed_dim": embed_dim,
                    "seed": seed,
                    "phase_a_ia_gap": gap_by_key[key],
                    "alpha_post_phase_b": alpha,
                }
            )
    if not alpha_rows:
        return {"n": 0, "error": "no aligned rows"}

    g = np.asarray([r["phase_a_ia_gap"] for r in alpha_rows], dtype=np.float64)
    a = np.asarray([r["alpha_post_phase_b"] for r in alpha_rows], dtype=np.float64)
    X = np.column_stack([np.ones_like(g), g])
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    pred = X @ coef
    return {
        "n": int(len(alpha_rows)),
        "pearson_r": _pearson(g, a),
        "linear_fit": {"intercept": float(coef[0]), "slope": float(coef[1]), "r2": _r2(a, pred), "mae": _mae(a, pred)},
        "rows": alpha_rows,
    }


def run(cfg: dict[str, Any]) -> None:
    start = time.time()

    out_root = Path(str(cfg["output_root"])).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stage_root = out_root / "stage68_law_robustness_reanalysis"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = out_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage36 = load_json(Path(str(cfg["stage36_results"])).resolve())
    stage43 = load_json(Path(str(cfg["stage43_results"])).resolve())
    stage58_path = Path(str(cfg["stage58_results"])).resolve()
    stage56 = load_json(Path(str(cfg["stage56_results"])).resolve())
    stage63_path = Path(str(cfg["stage63_results"])).resolve()
    stage47_path = Path(str(cfg["stage47_results"])).resolve()
    stage44_path = Path(str(cfg["stage44_results"])).resolve()
    w13_path = Path(str(cfg["w13_stage55_results"])).resolve()
    w14_primary_path = Path(str(cfg["w14_primary_results"])).resolve()
    w14_cc3m_path = Path(str(cfg["w14_cc3m_results"])).resolve()

    kfold = int(cfg.get("kfold", 5))
    cv_seed = int(cfg.get("cv_seed", 2026))
    boot_n = int(cfg.get("powerlaw_bootstrap_n", 1000))

    # A) Cell-mean refit and mixed-effects robustness
    s36_seed_records = stage36["records"]
    s36_cell_records = _group_cell_means(s36_seed_records)
    s58_seed_records = _collect_stage58_seed_records(stage58_path)
    s58_cell_records = _group_cell_means(s58_seed_records)

    law_robustness = {
        "audiocaps_primary": {
            "seed_level": _fit_basic_law(s36_seed_records),
            "cell_mean_level": _fit_basic_law(s36_cell_records),
            "cluster_and_mixed_effects": _cluster_and_mixed_effects(s36_seed_records),
        },
        "avcaps_full": {
            "seed_level": _fit_basic_law(s58_seed_records),
            "cell_mean_level": _fit_basic_law(s58_cell_records),
            "cluster_and_mixed_effects": _cluster_and_mixed_effects(s58_seed_records),
        },
    }

    # B) Cross-suite functional-form adjudication
    holdout_source = str(cfg.get("holdout_source_id", "stage31_wavcaps_scaling"))
    suite_forms = {
        "audiocaps_primary": _evaluate_forms(
            s36_seed_records,
            suite_name="audiocaps_primary",
            kfold=kfold,
            cv_seed=cv_seed,
            boot_n=boot_n,
            holdout_source=holdout_source,
        ),
        "avcaps_full": _evaluate_forms(
            s58_seed_records,
            suite_name="avcaps_full",
            kfold=kfold,
            cv_seed=cv_seed,
            boot_n=boot_n,
            holdout_source=None,
        ),
    }

    # Optional proxy suite: Clotho intermediate (av_at=Clotho, av_ia from paired stage55 AudioCaps eval)
    clotho_proxy_records = _collect_stage63_clotho_proxy(stage63_path)
    suite_forms["clotho_proxy_bridge"] = _evaluate_forms(
        clotho_proxy_records,
        suite_name="clotho_proxy_bridge",
        kfold=kfold,
        cv_seed=cv_seed,
        boot_n=boot_n,
        holdout_source=None,
    )

    # C) α-locked prediction checks on W13/W14-like unseen conditions
    alpha_locked = float(stage43["fit_train"]["alpha"])
    w13_records = _collect_stage55_rows(w13_path, source_id="w13_mixed100k")
    w14_primary_records = _collect_stage55_rows(w14_primary_path, source_id="w14_coco_subsampled")
    w14_cc3m_records = _collect_stage55_rows(w14_cc3m_path, source_id="w14_cc3m_domain_gap")
    prediction_checks = {
        "alpha_reference": alpha_locked,
        "w13_mixed100k": _alpha_locked_prediction(alpha_locked, w13_records),
        "w14_coco_subsampled": _alpha_locked_prediction(alpha_locked, w14_primary_records),
        "w14_cc3m_domain_gap": _alpha_locked_prediction(alpha_locked, w14_cc3m_records),
    }

    # D) Feasibility: can pre-Phase-B geometry predict post-Phase-B alpha?
    phase_a_geometry = _phase_a_gap_feasibility(stage47_path, stage44_path)

    out = {
        "stage": "stage68_law_robustness_reanalysis",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "law_robustness_seed_vs_cell": law_robustness,
        "functional_form_cross_suite": suite_forms,
        "alpha_locked_prediction_checks": prediction_checks,
        "phase_a_geometry_feasibility": phase_a_geometry,
        "caveats": [
            "clotho_proxy_bridge uses Clotho av_at with paired stage55 av_ia from AudioCaps eval; interpret as proxy-only, not pure Clotho ia retrieval.",
            "W14 primary artifact currently reflects a COCO-subsampled Phase-A condition (phase_a_source tag remains 'coco' in stage29 schema).",
        ],
        "elapsed_sec": float(time.time() - start),
    }
    save_json(out, stage_root / "stage68_law_robustness_reanalysis.json")

    lines: list[str] = [
        "# Stage68 Law Robustness Reanalysis",
        "",
        "## A) Seed-Level vs Cell-Mean Robustness",
        "",
    ]
    for suite in ["audiocaps_primary", "avcaps_full"]:
        sec = law_robustness[suite]
        lines += [
            f"### {suite}",
            f"- seed-level: n={int(sec['seed_level']['n'])}, alpha={sec['seed_level']['alpha']:.4f}, r2={sec['seed_level']['r2']:.4f}, mae={sec['seed_level']['mae']:.5f}",
            f"- cell-mean: n={int(sec['cell_mean_level']['n'])}, alpha={sec['cell_mean_level']['alpha']:.4f}, r2={sec['cell_mean_level']['r2']:.4f}, mae={sec['cell_mean_level']['mae']:.5f}",
        ]
        crm = sec["cluster_and_mixed_effects"]
        cro = crm.get("cluster_robust_ols")
        if isinstance(cro, dict):
            lines.append(
                f"- cluster-robust OLS: alpha={cro['alpha']:.4f}, ci95=[{cro['ci95_low']:.4f}, {cro['ci95_high']:.4f}], r2={cro['r2']:.4f}, cells={crm['n_cells']}"
            )
        else:
            lines.append(f"- cluster-robust OLS unavailable: {crm.get('cluster_robust_ols_error', 'unknown')}")
        me = crm.get("mixed_effects")
        if isinstance(me, dict):
            lines.append(
                f"- mixed-effects: alpha={me['alpha']:.4f}, ci95=[{me['ci95_low']:.4f}, {me['ci95_high']:.4f}], converged={me['converged']}"
            )
        else:
            lines.append(f"- mixed-effects unavailable: {crm.get('mixed_effects_error', 'unknown')}")
        lines.append("")

    lines += [
        "## B) Cross-Suite Functional-Form Adjudication",
        "",
        "| Suite | n | Best by CV-R2 | Geometric CV-R2 | Arithmetic CV-R2 | Product CV-R2 | Free-power CV-R2 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for suite_name, res in suite_forms.items():
        by_name = {m["name"]: m for m in res["models"]}
        lines.append(
            f"| {suite_name} | {res['n']} | {res['best_by_cv_r2']} | "
            f"{by_name['geometric_mean']['cv_r2_mean']:.4f} | {by_name['arithmetic_mean']['cv_r2_mean']:.4f} | "
            f"{by_name['product']['cv_r2_mean']:.4f} | {by_name['power_law_free']['cv_r2_mean']:.4f} |"
        )

    lines += [
        "",
        "## C) α-Locked Predictions on W13/W14",
        "",
        f"- α fixed from Stage43 train fit: {alpha_locked:.4f}",
        "",
        "| Condition | n | r | R2 | MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["w13_mixed100k", "w14_coco_subsampled", "w14_cc3m_domain_gap"]:
        m = prediction_checks[name]
        lines.append(f"| {name} | {m['n']} | {m['pearson_r']:.4f} | {m['r2']:.4f} | {m['mae']:.5f} |")

    lines += [
        "",
        "## D) Phase-A Geometry Feasibility",
        "",
        f"- n={phase_a_geometry.get('n', 0)}, r(phaseA_gap, postPhaseB_alpha)={phase_a_geometry.get('pearson_r', 0.0):.4f}",
    ]
    lf = phase_a_geometry.get("linear_fit")
    if isinstance(lf, dict):
        lines.append(
            f"- linear fit alpha~gap: slope={lf['slope']:.4f}, intercept={lf['intercept']:.4f}, R2={lf['r2']:.4f}, MAE={lf['mae']:.5f}"
        )
    lines += [
        "",
        "## Caveats",
        "",
    ] + [f"- {c}" for c in out["caveats"]]

    (stage_root / "stage68_law_robustness_reanalysis.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = env_snapshot(
        Path(str(cfg["project_root"])),
        seeds=[],
        extra={
            "stage": "stage68_law_robustness_reanalysis",
            "elapsed_sec": float(time.time() - start),
            "inputs": {
                "stage36_results": str(cfg["stage36_results"]),
                "stage43_results": str(cfg["stage43_results"]),
                "stage58_results": str(cfg["stage58_results"]),
                "stage56_results": str(cfg["stage56_results"]),
                "stage63_results": str(cfg["stage63_results"]),
                "stage47_results": str(cfg["stage47_results"]),
                "stage44_results": str(cfg["stage44_results"]),
                "w13_stage55_results": str(cfg["w13_stage55_results"]),
                "w14_primary_results": str(cfg["w14_primary_results"]),
                "w14_cc3m_results": str(cfg["w14_cc3m_results"]),
            },
        },
    )
    save_json(provenance, stage_root / "provenance_stage68.json")
    mark_done(markers / "stage68_law_robustness_reanalysis.done.json", {"elapsed_sec": float(time.time() - start)})
    print("stage68_law_robustness_reanalysis complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

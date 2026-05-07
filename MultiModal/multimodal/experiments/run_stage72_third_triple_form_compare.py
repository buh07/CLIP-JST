from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json


def _mean(xs: list[float]) -> float:
    return float(np.mean(np.asarray(xs, dtype=np.float64))) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(np.std(np.asarray(xs, dtype=np.float64), ddof=1))


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


@dataclass
class ModelResult:
    name: str
    params: dict[str, float]
    in_sample_r2: float
    in_sample_mae: float
    cv_r2_mean: float
    cv_r2_std: float
    cv_mae_mean: float
    cv_mae_std: float
    lomo_r2: float
    lomo_mae: float
    lodo_r2: float
    lodo_mae: float


def _fit_fixed_form(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, *, name: str) -> tuple[dict[str, float], np.ndarray]:
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

    Xt = np.column_stack([
        np.ones(int(mask_tr.sum()), dtype=np.float64),
        np.log(it_train[mask_tr]),
        np.log(at_train[mask_tr]),
    ])
    yt = np.log(y_train[mask_tr])
    coef, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
    c, a, b = [float(v) for v in coef]
    alpha = float(math.exp(c))

    mask_ev = (it_eval > 0) & (at_eval > 0)
    pred = np.zeros_like(it_eval)
    pred[mask_ev] = alpha * np.power(it_eval[mask_ev], a) * np.power(at_eval[mask_ev], b)
    return {"alpha": alpha, "a": a, "b": b}, pred


def _bootstrap_powerlaw_ci(it: np.ndarray, at: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int) -> dict[str, float]:
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


def _merge_stage69_raw(blocks: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    merged: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for b in blocks:
        for m_key, methods in b.get("raw", {}).items():
            dst_m = merged.setdefault(m_key, {})
            for method, rows in methods.items():
                dst_r = dst_m.setdefault(method, {})
                for r in rows:
                    dst_r[int(r["seed"])] = r

    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for m_key, methods in merged.items():
        out[m_key] = {}
        for method, by_seed in methods.items():
            out[m_key][method] = [by_seed[s] for s in sorted(by_seed)]
    return out


def _collect_records(stage70_aggregate_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    stage70 = load_json(stage70_aggregate_path)
    roots = [Path(x).resolve() for x in stage70["stage69_shard_roots"]]
    blocks = []
    for r in roots:
        p = r / "stage69_third_triple_speechcoco" / "stage69_third_triple_speechcoco_results.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing stage69 shard results: {p}")
        blocks.append(load_json(p))

    raw = _merge_stage69_raw(blocks)
    recs = []
    for m_key, methods in raw.items():
        embed_dim = int(str(m_key).lstrip("m"))
        for method, rows in methods.items():
            for r in rows:
                recs.append(
                    {
                        "source_id": "stage69_speechcoco_full_grid",
                        "method": str(method),
                        "embed_dim": embed_dim,
                        "seed": int(r["seed"]),
                        "av_it_ood": float(r["av_it_avg_R"]),
                        "av_at": float(r["av_at_avg_R"]),
                        "av_ia": float(r["av_ia_avg_R"]),
                    }
                )
    return recs, [str(x) for x in roots]


def _predict_form(
    name: str,
    it_train: np.ndarray,
    at_train: np.ndarray,
    y_train: np.ndarray,
    it_eval: np.ndarray,
    at_eval: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    if name == "power_law_free":
        return _fit_powerlaw(it_train, at_train, y_train, it_eval, at_eval)

    x_map_train = {
        "geometric_mean": np.sqrt(np.clip(it_train * at_train, 0.0, None)),
        "arithmetic_mean": 0.5 * (it_train + at_train),
        "hard_min": np.minimum(it_train, at_train),
        "product": it_train * at_train,
    }
    x_map_eval = {
        "geometric_mean": np.sqrt(np.clip(it_eval * at_eval, 0.0, None)),
        "arithmetic_mean": 0.5 * (it_eval + at_eval),
        "hard_min": np.minimum(it_eval, at_eval),
        "product": it_eval * at_eval,
    }
    return _fit_fixed_form(x_map_train[name], y_train, x_map_eval[name], name=name)


def _evaluate_models(records: list[dict[str, Any]], *, kfold: int, cv_seed: int, boot_n: int) -> dict[str, Any]:
    it = np.asarray([float(r["av_it_ood"]) for r in records], dtype=np.float64)
    at = np.asarray([float(r["av_at"]) for r in records], dtype=np.float64)
    ia = np.asarray([float(r["av_ia"]) for r in records], dtype=np.float64)
    methods = np.asarray([str(r["method"]) for r in records], dtype=object)
    dims = np.asarray([int(r["embed_dim"]) for r in records], dtype=np.int64)

    model_names = ["geometric_mean", "arithmetic_mean", "hard_min", "product", "power_law_free"]
    model_results: list[ModelResult] = []

    for name in model_names:
        params_all, pred_all = _predict_form(name, it, at, ia, it, at)
        in_r2 = _r2(ia, pred_all)
        in_mae = _mae(ia, pred_all)

        folds = _kfold_indices(len(records), max(2, min(kfold, len(records))), cv_seed)
        cv_r2, cv_mae = [], []
        for fold in folds:
            tr_mask = np.ones(len(records), dtype=bool)
            tr_mask[fold] = False
            _, pred = _predict_form(name, it[tr_mask], at[tr_mask], ia[tr_mask], it[fold], at[fold])
            cv_r2.append(_r2(ia[fold], pred))
            cv_mae.append(_mae(ia[fold], pred))

        pred_lomo = np.zeros_like(ia)
        for m in sorted(set(methods.tolist())):
            te = methods == m
            tr = ~te
            _, pred = _predict_form(name, it[tr], at[tr], ia[tr], it[te], at[te])
            pred_lomo[te] = pred
        lomo_r2 = _r2(ia, pred_lomo)
        lomo_mae = _mae(ia, pred_lomo)

        pred_lodo = np.zeros_like(ia)
        for d in sorted(set(dims.tolist())):
            te = dims == d
            tr = ~te
            _, pred = _predict_form(name, it[tr], at[tr], ia[tr], it[te], at[te])
            pred_lodo[te] = pred
        lodo_r2 = _r2(ia, pred_lodo)
        lodo_mae = _mae(ia, pred_lodo)

        model_results.append(
            ModelResult(
                name=name,
                params=params_all,
                in_sample_r2=in_r2,
                in_sample_mae=in_mae,
                cv_r2_mean=_mean(cv_r2),
                cv_r2_std=_std(cv_r2),
                cv_mae_mean=_mean(cv_mae),
                cv_mae_std=_std(cv_mae),
                lomo_r2=lomo_r2,
                lomo_mae=lomo_mae,
                lodo_r2=lodo_r2,
                lodo_mae=lodo_mae,
            )
        )

    ci = _bootstrap_powerlaw_ci(it, at, ia, n_boot=boot_n, seed=cv_seed + 99)

    return {
        "n_total": int(len(records)),
        "models": [
            {
                "name": m.name,
                "params": m.params,
                "in_sample_r2": m.in_sample_r2,
                "in_sample_mae": m.in_sample_mae,
                "cv_r2_mean": m.cv_r2_mean,
                "cv_r2_std": m.cv_r2_std,
                "cv_mae_mean": m.cv_mae_mean,
                "cv_mae_std": m.cv_mae_std,
                "lomo_r2": m.lomo_r2,
                "lomo_mae": m.lomo_mae,
                "lodo_r2": m.lodo_r2,
                "lodo_mae": m.lodo_mae,
            }
            for m in model_results
        ],
        "power_law_free_ci95": ci,
    }


def run(cfg: dict[str, Any]) -> None:
    start = time.time()

    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage72_third_triple_form_compare"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage70_path = Path(cfg["stage70_aggregate"]).resolve()
    records, roots = _collect_records(stage70_path)
    analysis = _evaluate_models(
        records,
        kfold=int(cfg.get("kfold", 5)),
        cv_seed=int(cfg.get("cv_seed", 2026)),
        boot_n=int(cfg.get("powerlaw_bootstrap_n", 1000)),
    )

    out = {
        "stage": "stage72_third_triple_form_compare",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage70_aggregate": str(stage70_path),
        "stage69_shard_roots": roots,
        "analysis": analysis,
        "elapsed_sec": float(time.time() - start),
    }
    save_json(out, stage_root / "stage72_third_triple_form_compare.json")

    models_sorted = sorted(analysis["models"], key=lambda m: m["cv_r2_mean"], reverse=True)

    lines = [
        "# Stage72 Third Triple Functional-Form Comparison",
        "",
        f"- n_total: {analysis['n_total']}",
        "",
        "| form | cv_r2 | cv_mae | lomo_r2 | lodo_r2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in models_sorted:
        lines.append(
            f"| {m['name']} | {m['cv_r2_mean']:.4f}±{m['cv_r2_std']:.4f} | "
            f"{m['cv_mae_mean']:.4f}±{m['cv_mae_std']:.4f} | {m['lomo_r2']:.4f} | {m['lodo_r2']:.4f} |"
        )

    ci = analysis["power_law_free_ci95"]
    lines += [
        "",
        "## Free Power-Law CI (95%)",
        f"- a: [{ci['a_lo']:.4f}, {ci['a_hi']:.4f}]",
        f"- b: [{ci['b_lo']:.4f}, {ci['b_hi']:.4f}]",
    ]

    (stage_root / "stage72_third_triple_form_compare.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={"stage": "stage72_third_triple_form_compare", "elapsed_sec": float(time.time() - start)},
    )
    save_json(provenance, stage_root / "provenance_stage72.json")
    mark_done(markers / "stage72_third_triple_form_compare.done.json", {"elapsed_sec": float(time.time() - start)})
    print("stage72_third_triple_form_compare complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import digamma
from sklearn.covariance import LedoitWolf
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import KDTree, NearestNeighbors

from ..common import load_json
from ..eval.retrieval import recall_at_k
from .run_stage29_cc3m_phaseA_modular import _build_model, _encode_batches


def canonical_json_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def sanitize_name(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def metric_avg(x: Any) -> float:
    if isinstance(x, dict):
        if "avg_R" in x:
            return float(x["avg_R"])
        if "mean" in x:
            return float(x["mean"])
    if x is None:
        return 0.0
    return float(x)


def extract_centroid_gap_ia(rec: dict[str, Any]) -> float | None:
    v = rec.get("gap_ia_l2")
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass
    d = rec.get("diagnostics", {})
    if isinstance(d, dict):
        av = d.get("av", {})
        if isinstance(av, dict):
            cdm = av.get("centroid_distance_matrix", {})
            if isinstance(cdm, dict):
                img = cdm.get("image", {})
                if isinstance(img, dict) and img.get("audio") is not None:
                    try:
                        return float(img.get("audio"))
                    except Exception:
                        pass
    return None


def _find_checkpoint(seed_dir: Path) -> Path | None:
    cands = [
        seed_dir / "phase_b" / "best.pt",
        seed_dir / "joint" / "best.pt",
        seed_dir / "phase_a" / "best.pt",
        seed_dir / "best.pt",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def collect_source_rows(
    source_specs: list[dict[str, Any]],
    *,
    embed_dims: set[int] | None = None,
    methods: set[str] | None = None,
    seeds: set[int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for src in source_specs:
        source_id = str(src["source_id"])
        source_group = str(src.get("source_group", source_id))
        stage_root = Path(src["stage_root"]).resolve()
        stage_name = str(src.get("stage_name", stage_root.name))
        results_file = str(src.get("results_file", f"{stage_name}_results.json"))
        results_path = stage_root / results_file
        if not results_path.exists():
            continue

        obj = load_json(results_path)
        raw = obj.get("raw", {})
        for m_key, method_rows in raw.items():
            try:
                m = int(str(m_key).lstrip("m"))
            except Exception:
                continue
            if embed_dims and m not in embed_dims:
                continue
            if not isinstance(method_rows, dict):
                continue
            for method, recs in method_rows.items():
                if methods and method not in methods:
                    continue
                if not isinstance(recs, list):
                    continue
                for rec in recs:
                    seed = int(rec.get("seed", -1))
                    if seeds and seed not in seeds:
                        continue
                    key = (source_id, stage_root.as_posix(), stage_name, m, method, seed)
                    if key in seen:
                        continue
                    seen.add(key)

                    seed_dir = stage_root / f"m{m}" / method / f"seed{seed}"
                    ckpt = _find_checkpoint(seed_dir)
                    eval_path = seed_dir / "eval.json"

                    av_it = metric_avg(rec.get("av_image_text", rec.get("av_it_avg_R", 0.0)))
                    av_at = metric_avg(rec.get("av_audio_text", rec.get("av_at_avg_R", 0.0)))
                    av_ia = metric_avg(rec.get("av_image_audio", rec.get("av_ia_avg_R", 0.0)))
                    coco = metric_avg(rec.get("coco_image_text", rec.get("coco_avg_R", 0.0)))
                    ceiling = math.sqrt(max(0.0, av_it) * max(0.0, av_at)) if av_it > 0 and av_at > 0 else 0.0

                    row = {
                        "source_id": source_id,
                        "source_group": source_group,
                        "stage_name": stage_name,
                        "stage_root": str(stage_root),
                        "results_path": str(results_path),
                        "embed_dim": m,
                        "method": method,
                        "seed": seed,
                        "condition_id": f"{source_id}|{stage_name}|m{m}|{method}|seed{seed}",
                        "checkpoint_path": str(ckpt) if ckpt else None,
                        "has_checkpoint": bool(ckpt),
                        "eval_path": str(eval_path),
                        "has_eval": eval_path.exists(),
                        "phase_a_source": rec.get("phase_a_source", obj.get("phase_a_source", "unknown")),
                        "phase_b_source": rec.get("phase_b_source", obj.get("phase_b_source", "unknown")),
                        "phase_order": rec.get("phase_order", obj.get("phase_order", "unknown")),
                        "metrics": {
                            "coco_avg_R": coco,
                            "av_it_avg_R": av_it,
                            "av_at_avg_R": av_at,
                            "av_ia_avg_R": av_ia,
                            "combined_avg_R": metric_avg(rec.get("combined_avg_R", 0.0)),
                            "ceiling": ceiling,
                        },
                        "centroid_gap_ia_l2": extract_centroid_gap_ia(rec),
                    }
                    rows.append(row)
    rows.sort(key=lambda r: (r["source_id"], r["embed_dim"], r["method"], r["seed"]))
    return rows


def manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, int] = {}
    by_method: dict[str, int] = {}
    by_dim: dict[str, int] = {}
    missing = 0
    for r in rows:
        by_source[r["source_id"]] = by_source.get(r["source_id"], 0) + 1
        by_method[r["method"]] = by_method.get(r["method"], 0) + 1
        d = str(r["embed_dim"])
        by_dim[d] = by_dim.get(d, 0) + 1
        if not r.get("has_checkpoint", False):
            missing += 1
    return {
        "n_conditions": len(rows),
        "missing_checkpoint_conditions": missing,
        "by_source": dict(sorted(by_source.items())),
        "by_method": dict(sorted(by_method.items())),
        "by_embed_dim": dict(sorted(by_dim.items(), key=lambda kv: int(kv[0]))),
    }


def model_load_for_condition(cond: dict[str, Any], cfg: dict[str, Any], *, device: str):
    m = int(cond["embed_dim"])
    method = str(cond["method"])
    model = _build_model(method, m, cfg)
    if model is None:
        return None
    ckpt_path = Path(str(cond["checkpoint_path"]))
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=True)
    return model.to(device).eval()


@torch.no_grad()
def export_embeddings_for_condition(
    cond: dict[str, Any],
    cfg: dict[str, Any],
    av,
    *,
    device: str,
    batch_size: int,
    store_dtype: str = "float16",
) -> dict[str, Any]:
    model = model_load_for_condition(cond, cfg, device=device)
    if model is None:
        raise RuntimeError(f"Unsupported eval-only method for stage48 export: {cond['method']}")

    img, aud, txt = av.eval_tensors(str(cfg.get("av_test_split", "test")))
    zi = _encode_batches(model.encode_image, img, device=device, batch_size=batch_size)
    za = _encode_batches(model.encode_audio, aud, device=device, batch_size=batch_size)
    zt = _encode_batches(model.encode_text, txt, device=device, batch_size=batch_size)

    met_it = recall_at_k(zi, zt)
    met_at = recall_at_k(za, zt)
    met_ia = recall_at_k(zi, za)

    src = cond.get("metrics", {})
    parity = {
        "delta_av_it": float(abs(float(met_it["avg_R"]) - float(src.get("av_it_avg_R", 0.0)))),
        "delta_av_at": float(abs(float(met_at["avg_R"]) - float(src.get("av_at_avg_R", 0.0)))),
        "delta_av_ia": float(abs(float(met_ia["avg_R"]) - float(src.get("av_ia_avg_R", 0.0)))),
    }

    out_dtype = torch.float16 if store_dtype == "float16" else torch.float32
    return {
        "zi": zi.to(out_dtype).cpu(),
        "za": za.to(out_dtype).cpu(),
        "zt": zt.to(out_dtype).cpu(),
        "recall": {
            "av_it_avg_R": float(met_it["avg_R"]),
            "av_at_avg_R": float(met_at["avg_R"]),
            "av_ia_avg_R": float(met_ia["avg_R"]),
        },
        "parity": parity,
    }


def cond_output_relpath(cond: dict[str, Any]) -> Path:
    source = sanitize_name(cond["source_id"])
    method = sanitize_name(cond["method"])
    return Path(source) / f"m{int(cond['embed_dim'])}" / method / f"seed{int(cond['seed'])}"


def deterministic_subsample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    idx.sort()
    return idx.astype(np.int64)


def pca_reduce(x: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    n, d = x.shape
    if pca_dim <= 0 or d <= pca_dim:
        return x.astype(np.float64, copy=False)
    p = PCA(n_components=min(pca_dim, d, n - 1), svd_solver="randomized", random_state=seed)
    return p.fit_transform(x).astype(np.float64, copy=False)


def pca_reduce_joint_blocks(blocks: list[np.ndarray], pca_dim: int, seed: int) -> list[np.ndarray]:
    """
    Fit a single PCA basis on the vertical concatenation of all blocks, then
    project each block into that shared basis.
    """
    if not blocks:
        return []
    xs = [np.asarray(b, dtype=np.float64) for b in blocks]
    n0 = xs[0].shape[0]
    d0 = xs[0].shape[1]
    for b in xs:
        if b.ndim != 2:
            raise ValueError("All PCA blocks must be 2D")
        if b.shape[0] != n0:
            raise ValueError("All PCA blocks must have same number of rows")
        if b.shape[1] != d0:
            raise ValueError("All PCA blocks must have same feature dimension")
    if pca_dim <= 0 or d0 <= pca_dim:
        return [b.astype(np.float64, copy=False) for b in xs]
    n_components = min(pca_dim, d0, max(1, n0 - 1))
    p = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    cat = np.concatenate(xs, axis=0)
    p.fit(cat)
    return [p.transform(b).astype(np.float64, copy=False) for b in xs]


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x[:, None]
    return x


def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    x = _ensure_2d(np.asarray(x, dtype=np.float64))
    y = _ensure_2d(np.asarray(y, dtype=np.float64))
    n = x.shape[0]
    if n <= k + 1:
        return 0.0

    z = np.concatenate([x, y], axis=1)
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=k + 1)
    nn.fit(z)
    dists, _ = nn.kneighbors(z, n_neighbors=k + 1)
    eps = np.maximum(dists[:, k] - 1e-12, 1e-15)

    tx = KDTree(x, metric="chebyshev")
    ty = KDTree(y, metric="chebyshev")
    nx = tx.query_radius(x, r=eps, count_only=True) - 1
    ny = ty.query_radius(y, r=eps, count_only=True) - 1

    nx = np.maximum(nx, 0)
    ny = np.maximum(ny, 0)
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(max(mi, 0.0))


def estimate_cov(x: np.ndarray, shrinkage: str = "ledoit_wolf", ridge_lambda: float = 1e-6) -> np.ndarray:
    x = _ensure_2d(np.asarray(x, dtype=np.float64))
    if x.shape[0] < 2:
        return np.eye(x.shape[1], dtype=np.float64)
    if shrinkage == "ledoit_wolf":
        cov = LedoitWolf().fit(x).covariance_
    else:
        cov = np.cov(x, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    cov += float(ridge_lambda) * np.eye(cov.shape[0], dtype=np.float64)
    return cov


def _logdet_pd(cov: np.ndarray) -> float:
    cov = np.asarray(cov, dtype=np.float64)
    for jitter in [0.0, 1e-10, 1e-8, 1e-6, 1e-4]:
        m = cov + jitter * np.eye(cov.shape[0], dtype=np.float64)
        sign, ld = np.linalg.slogdet(m)
        if sign > 0 and np.isfinite(ld):
            return float(ld)
    vals = np.linalg.svd(cov, compute_uv=False)
    vals = np.maximum(vals, 1e-12)
    return float(np.sum(np.log(vals)))


def gaussian_mi(x: np.ndarray, y: np.ndarray, *, shrinkage: str = "ledoit_wolf", ridge_lambda: float = 1e-6) -> float:
    x = _ensure_2d(np.asarray(x, dtype=np.float64))
    y = _ensure_2d(np.asarray(y, dtype=np.float64))
    xy = np.concatenate([x, y], axis=1)
    cx = estimate_cov(x, shrinkage=shrinkage, ridge_lambda=ridge_lambda)
    cy = estimate_cov(y, shrinkage=shrinkage, ridge_lambda=ridge_lambda)
    cxy = estimate_cov(xy, shrinkage=shrinkage, ridge_lambda=ridge_lambda)
    mi = 0.5 * (_logdet_pd(cx) + _logdet_pd(cy) - _logdet_pd(cxy))
    return float(max(mi, 0.0))


def cancor_mi_proxy(x: np.ndarray, y: np.ndarray, *, max_components: int = 16) -> float:
    x = _ensure_2d(np.asarray(x, dtype=np.float64))
    y = _ensure_2d(np.asarray(y, dtype=np.float64))
    n = x.shape[0]
    c = min(max_components, x.shape[1], y.shape[1], n - 2)
    if c <= 0:
        return 0.0
    cca = CCA(n_components=c, max_iter=1000)
    try:
        x_c, y_c = cca.fit_transform(x, y)
    except Exception:
        return 0.0
    rhos = []
    for j in range(c):
        xs = x_c[:, j]
        ys = y_c[:, j]
        sx = float(np.std(xs))
        sy = float(np.std(ys))
        if sx <= 1e-12 or sy <= 1e-12:
            continue
        rho = float(np.corrcoef(xs, ys)[0, 1])
        if np.isfinite(rho):
            rhos.append(np.clip(abs(rho), 0.0, 1.0 - 1e-8))
    if not rhos:
        return 0.0
    rhos = np.asarray(rhos, dtype=np.float64)
    mi = -0.5 * np.sum(np.log(1.0 - rhos ** 2))
    return float(max(mi, 0.0))


def residualize(x: np.ndarray, z: np.ndarray, ridge_lambda: float = 1e-3) -> np.ndarray:
    x = _ensure_2d(np.asarray(x, dtype=np.float64))
    z = _ensure_2d(np.asarray(z, dtype=np.float64))
    model = Ridge(alpha=float(ridge_lambda), fit_intercept=True)
    model.fit(z, x)
    pred = model.predict(z)
    return (x - pred).astype(np.float64, copy=False)


def gaussian_cmi_from_cov(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    shrinkage: str = "ledoit_wolf",
    ridge_lambda: float = 1e-6,
) -> float:
    x = _ensure_2d(np.asarray(x, dtype=np.float64))
    y = _ensure_2d(np.asarray(y, dtype=np.float64))
    z = _ensure_2d(np.asarray(z, dtype=np.float64))

    xz = np.concatenate([x, z], axis=1)
    yz = np.concatenate([y, z], axis=1)
    xyz = np.concatenate([x, y, z], axis=1)

    cxz = estimate_cov(xz, shrinkage=shrinkage, ridge_lambda=ridge_lambda)
    cyz = estimate_cov(yz, shrinkage=shrinkage, ridge_lambda=ridge_lambda)
    cz = estimate_cov(z, shrinkage=shrinkage, ridge_lambda=ridge_lambda)
    cxyz = estimate_cov(xyz, shrinkage=shrinkage, ridge_lambda=ridge_lambda)

    cmi = 0.5 * (_logdet_pd(cxz) + _logdet_pd(cyz) - _logdet_pd(cz) - _logdet_pd(cxyz))
    return float(max(cmi, 0.0))


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return 0.0
    sx = np.std(x)
    sy = np.std(y)
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    v = float(np.corrcoef(x, y)[0, 1])
    return v if np.isfinite(v) else 0.0


def alpha_like(num: float, a: float, b: float, eps: float = 1e-12) -> float:
    denom = math.sqrt(max(a, eps) * max(b, eps))
    if denom <= 0:
        return 0.0
    return float(num / denom)


def alpha_like_unit(num: float, a: float, b: float, eps: float = 1e-12) -> float:
    return float(np.clip(alpha_like(num, a, b, eps=eps), 0.0, 1.0))


def bootstrap_ci(
    arr: np.ndarray,
    fn,
    *,
    n_boot: int = 1000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> dict[str, float]:
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(float(fn(arr[idx])))
    vals = np.asarray(vals, dtype=np.float64)
    lo = float(np.quantile(vals, alpha / 2.0))
    hi = float(np.quantile(vals, 1.0 - alpha / 2.0))
    return {
        "mean": float(np.mean(vals)),
        "ci_low": lo,
        "ci_high": hi,
    }


def permutation_pvalue_for_corr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int = 1000,
    seed: int = 2026,
) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    obs = abs(corrcoef_safe(x, y))
    if len(x) < 3:
        return {"obs": obs, "p_value": 1.0}
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(n_perm):
        yp = y[rng.permutation(len(y))]
        v = abs(corrcoef_safe(x, yp))
        if v >= obs:
            ge += 1
    p = ge / float(n_perm + 1)
    return {"obs": obs, "p_value": float(p)}

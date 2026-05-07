from __future__ import annotations

import math

import numpy as np


def fit_gaussian(scores: np.ndarray, min_std: float = 1e-6) -> tuple[float, float]:
    x = np.asarray(scores, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("cannot fit Gaussian on empty scores")
    mu = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return mu, float(max(std, min_std))


def gaussian_logpdf(x: np.ndarray, mu: float, std: float) -> np.ndarray:
    z = (x - mu) / std
    return -0.5 * (z * z + math.log(2.0 * math.pi) + 2.0 * math.log(std))


def lira_log_likelihood_ratio(
    scores: np.ndarray,
    *,
    in_mu: float,
    in_std: float,
    out_mu: float,
    out_std: float,
) -> np.ndarray:
    s = np.asarray(scores, dtype=float).reshape(-1)
    return gaussian_logpdf(s, in_mu, in_std) - gaussian_logpdf(s, out_mu, out_std)


def roc_curve_from_scores(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if y.size != s.size:
        raise ValueError("labels and scores must have same length")
    if y.size == 0:
        raise ValueError("empty labels/scores")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary {0,1}")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("need both positive and negative labels")

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    change = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    idx = np.where(change)[0]

    tpr = tps[idx] / float(pos)
    fpr = fps[idx] / float(neg)
    thr = s_sorted[idx]

    tpr = np.r_[0.0, tpr, 1.0]
    fpr = np.r_[0.0, fpr, 1.0]
    thr = np.r_[np.inf, thr, -np.inf]
    return fpr, tpr, thr


def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    x = np.asarray(fpr, dtype=float)
    y = np.asarray(tpr, dtype=float)
    return float(np.trapz(y, x))


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    x = np.asarray(fpr, dtype=float)
    y = np.asarray(tpr, dtype=float)
    if target_fpr <= x[0]:
        return float(y[0])
    if target_fpr >= x[-1]:
        return float(y[-1])
    idx = np.searchsorted(x, target_fpr, side="right")
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    if x1 == x0:
        return float(max(y0, y1))
    w = (target_fpr - x0) / (x1 - x0)
    return float(y0 + w * (y1 - y0))

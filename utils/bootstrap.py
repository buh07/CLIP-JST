"""
Statistical testing utilities for Section 5.6.

- paired_bootstrap_ci: 95% CI for the performance difference between two models.
- permutation_test: p-value for H0: model_a and model_b have equal performance.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """
    Paired bootstrap confidence interval for the mean difference (a - b).

    scores_a, scores_b: per-query metric scores (e.g. hit@1 per query).
    Returns dict with 'mean_diff', 'ci_low', 'ci_high'.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert len(a) == len(b), "Score arrays must have equal length."

    observed = a.mean() - b.mean()
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(a), size=len(a))
        diffs.append(a[idx].mean() - b[idx].mean())

    alpha = (1.0 - ci) / 2.0
    ci_low  = float(np.percentile(diffs, 100 * alpha))
    ci_high = float(np.percentile(diffs, 100 * (1.0 - alpha)))
    return {"mean_diff": float(observed), "ci_low": ci_low, "ci_high": ci_high}


def permutation_test(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> float:
    """
    Paired permutation test for H0: mean(a) == mean(b).
    Returns two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert len(a) == len(b)

    observed = abs(a.mean() - b.mean())
    diffs = a - b
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        count += int(abs((signs * diffs).mean()) >= observed)
    return count / n_permutations

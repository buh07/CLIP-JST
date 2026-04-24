"""
Width estimation utilities supporting Claim 1.

The effective Gaussian width of a cross-modal pair set R is:
    w(R) = E_g [ sup_{(v,t) in R - R} <g, (v, -t)> ]

For real data we estimate this by:
  1. Forming the concatenated difference set {(v, -t) : (v,t) in R}.
  2. Monte Carlo approximation of E_g[max_x <g, x>] over random unit Gaussians g.

Also provides the theoretical minimum m required by Claim 1 given a width estimate.
"""

from __future__ import annotations

import math

import torch


def cross_modal_width_estimate(
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    n_samples: int = 1000,
    subsample: int | None = 5000,
) -> float:
    """
    Monte Carlo estimate of the Gaussian width of the Minkowski difference set
    { (v, -t) : (v, t) in R } ⊂ R^{dv + dt}.

    Pairs are assumed to correspond row-wise.
    Subsamples pairs when the set is large.
    """
    N = image_feats.shape[0]
    if subsample and N > subsample:
        idx = torch.randperm(N)[:subsample]
        image_feats = image_feats[idx]
        text_feats = text_feats[idx]

    # Concatenated difference vectors in R^{dv + dt}.
    diffs = torch.cat([image_feats, -text_feats], dim=1)  # (N, dv+dt)
    d = diffs.shape[1]

    g = torch.randn(n_samples, d, device=diffs.device)
    g = g / g.norm(dim=1, keepdim=True)
    projections = diffs @ g.T  # (N, n_samples)
    width = projections.max(dim=0).values.mean().item()
    return width


def required_dim(width: float, eps: float = 0.1, delta: float = 0.05, C: float = 1.0) -> int:
    """
    Theoretical minimum m from Claim 1 (Bourgain–Dirksen–Nelson width-adaptive JL):
        m >= C * eps^{-2} * (w^2 + log(1/delta))
    """
    m = C * (1.0 / eps) ** 2 * (width ** 2 + math.log(1.0 / delta))
    return int(math.ceil(m))

"""
Width estimation utilities supporting Claim 1.

The cross-modal width measures how spread apart paired (image, text) features are
on the unit sphere — i.e., how hard the alignment problem is.  We use the RMS
cross-modal distance on L2-normalized features:

    w(R) = sqrt( E_{(v,t) in R} [ ||v - t||^2 ] )
         = sqrt( 2 - 2 * E[cosine_similarity(v, t)] )

Range: 0 (perfectly aligned) to sqrt(2) ≈ 1.414 (orthogonal / random pairs).

This is the principled estimator for this setting because:
  - It is directly computable in O(N) without Monte Carlo noise.
  - It is monotone with alignment difficulty (smaller w → easier to JL-project).
  - In high dimensions (d >= 512), the standard Monte Carlo Gaussian width estimator
    collapses due to concentration of measure: all max inner-products converge to
    the same value regardless of pair quality, making it useless as a diagnostic.
  - The RMS distance is equivalent to the standard Gaussian width up to a
    dimension-dependent constant (||v - t||^2 = expected squared projection under
    random Gaussian g with unit variance per dimension).

Also provides the theoretical minimum m required by Claim 1 given a width estimate.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def cross_modal_width_estimate(
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    n_samples: int = 10_000,   # kept for API compatibility; unused
    subsample: int | None = 10_000,
    proj_seed: int = 0,
) -> float:
    """
    RMS cross-modal distance on the unit sphere:
        w = sqrt( mean_i ||v_i - t_i||^2 )
    where v_i, t_i are L2-normalized image and text features.

    Interpretation:
      w = 0.0   -- perfect alignment (v_i = t_i for all pairs)
      w ~ 0.4   -- high-agreement pairs  (cosine_sim ~ 0.9)
      w ~ 1.2   -- typical COCO pairs    (cosine_sim ~ 0.3)
      w ~ 1.41  -- random pairs          (cosine_sim ~ 0.0)

    When image and text features have different dimensions (e.g. raw ViT-B/32
    pooler outputs: d_v=768, d_t=512), the higher-dimensional modality is
    projected to the lower dimension via a fixed random Gaussian matrix before
    computing cosine similarities.  This is a JL-style reduction that
    approximately preserves pairwise distances and is consistent across calls
    with the same proj_seed.

    Subsamples pairs when N > subsample for efficiency.
    """
    image_feats = image_feats.float()
    text_feats  = text_feats.float()

    d_v, d_t = image_feats.shape[1], text_feats.shape[1]
    if d_v != d_t:
        d_common = min(d_v, d_t)
        rng = torch.Generator().manual_seed(proj_seed)
        if d_v > d_t:
            P = torch.randn(d_v, d_common, generator=rng) / math.sqrt(d_common)
            image_feats = image_feats @ P
        else:
            P = torch.randn(d_t, d_common, generator=rng) / math.sqrt(d_common)
            text_feats = text_feats @ P

    image_feats = F.normalize(image_feats, dim=1)
    text_feats  = F.normalize(text_feats, dim=1)

    N = image_feats.shape[0]
    if subsample and N > subsample:
        idx = torch.randperm(N)[:subsample]
        image_feats = image_feats[idx]
        text_feats  = text_feats[idx]

    # ||v - t||^2 = 2 - 2*<v,t> for unit vectors.
    cosine_sims = (image_feats * text_feats).sum(dim=1)   # (N,)
    rms_dist = (2.0 - 2.0 * cosine_sims).mean().clamp(min=0.0).sqrt().item()
    return rms_dist


def required_dim(width: float, eps: float = 0.1, delta: float = 0.05, C: float = 1.0) -> int:
    """
    Theoretical minimum m from Claim 1 (Bourgain-Dirksen-Nelson width-adaptive JL):
        m >= C * eps^{-2} * (w^2 + log(1/delta))
    """
    m = C * (1.0 / eps) ** 2 * (width ** 2 + math.log(1.0 / delta))
    return int(math.ceil(m))

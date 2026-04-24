"""
Diagnostic utilities (D1–D4).

D1: Expansion/shrinkage decomposition — compare singular-value spectra of
    the trained CLIP projection head vs. the JL + Mahalanobis pipeline.
D2: Class-conditional JL distortion measurement.
"""

from __future__ import annotations

import numpy as np
import torch


def singular_value_spectrum(weight: torch.Tensor) -> np.ndarray:
    """
    Returns sorted singular values (descending) of a 2D weight matrix.
    For CLIP-head: weight is (m, d). For Mahalanobis L: weight is (m, m) or (r, m).
    """
    _, s, _ = torch.linalg.svd(weight.detach().float(), full_matrices=False)
    return s.cpu().numpy()


def expansion_shrinkage_alignment(
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    k: int = 32,
) -> dict:
    """
    Measures subspace alignment between the top-k RIGHT singular vectors of
    two weight matrices weight_a and weight_b.

    Both matrices must share the same number of columns (input dimension d),
    so their right singular vectors live in the same R^d space.
    Use composed matrices (e.g. L @ Phi) when comparing across modalities.

    Returns:
        'sv_a': singular values of weight_a.
        'sv_b': singular values of weight_b.
        'subspace_overlap': cosine similarities of principal angles (top-k).
    """
    Ua, sa, Vha = torch.linalg.svd(weight_a.detach().float(), full_matrices=False)
    Ub, sb, Vhb = torch.linalg.svd(weight_b.detach().float(), full_matrices=False)

    # Right singular vectors: rows of Vh are the right sing vecs.
    # Vha: (min(m_a, d), d),  Vhb: (min(m_b, d), d)
    k_eff = min(k, Vha.shape[0], Vhb.shape[0])
    # Cosines of principal angles via SVD of cross-Gram: (k, d) @ (d, k).
    gram = Vha[:k_eff] @ Vhb[:k_eff].T   # (k_eff, k_eff)
    overlap = torch.linalg.svd(gram, full_matrices=False)[1]   # singular values = cosines

    return {
        "sv_a": sa.cpu().numpy(),
        "sv_b": sb.cpu().numpy(),
        "subspace_overlap": overlap.cpu().numpy(),
    }


# Keep backward-compatible alias used in run_D1.py
def expansion_shrinkage_alignment_named(
    clip_weight: torch.Tensor,
    mahal_weight: torch.Tensor,
    k: int = 32,
) -> dict:
    result = expansion_shrinkage_alignment(clip_weight, mahal_weight, k=k)
    return {
        "clip_sv": result["sv_a"],
        "mahal_sv": result["sv_b"],
        "subspace_overlap": result["subspace_overlap"],
    }


def jl_distortion_per_class(
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    labels: torch.Tensor,
    jl_image: torch.Tensor,
    jl_text: torch.Tensor,
) -> dict[int, float]:
    """
    Measures empirical JL distortion ||Phi v - Phi t||^2 / ||v - t||^2
    averaged over pairs within each class label (for D2 experiment).

    Args:
        image_feats: (N, dv) raw backbone features.
        text_feats:  (N, dt) raw backbone features.
        labels:      (N,) integer class labels.
        jl_image:    (m, dv) JL matrix for images.
        jl_text:     (m, dt) JL matrix for texts.
    """
    jl_image = jl_image.detach()
    jl_text  = jl_text.detach()
    proj_img = image_feats @ jl_image.T    # (N, m)
    proj_txt = text_feats  @ jl_text.T     # (N, m)

    results: dict[int, float] = {}
    for cls in labels.unique().tolist():
        mask = labels == cls
        v  = image_feats[mask]
        t  = text_feats[mask]
        pv = proj_img[mask]
        pt = proj_txt[mask]

        # Cross-modal "distance" in the block-diagonal concatenation space:
        # ||(v, -t)||^2 = ||v||^2 + ||t||^2
        orig_sq = v.pow(2).sum(dim=1) + t.pow(2).sum(dim=1)
        orig_sq = orig_sq.clamp(min=1e-8)
        proj_sq = (pv - pt).pow(2).sum(dim=1)
        results[int(cls)] = (proj_sq / orig_sq).mean().item()

    return results


def gaussian_width_upper_bound(feats: torch.Tensor, n_samples: int = 500) -> float:
    """
    Monte Carlo estimate of the Gaussian width of a feature set S:
        w(S) ≈ E_g[ max_{x in S} <g, x> - min_{x in S} <g, x> ]
    averaged over random unit Gaussian directions.
    """
    d = feats.shape[1]
    g = torch.randn(n_samples, d, device=feats.device)
    g = g / g.norm(dim=1, keepdim=True)
    projections = feats @ g.T    # (N, n_samples)
    width = (projections.max(dim=0).values - projections.min(dim=0).values).mean()
    return width.item()

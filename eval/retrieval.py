"""
Cross-modal retrieval evaluation: Recall@K and mAP.

Supports both 1:1 pairing (Flickr30K single-caption mode) and multi-GT
pairing (COCO 5-caption mode).
"""

from __future__ import annotations

import torch


def recall_at_k(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
    gt_i2t: dict[int, list[int]] | None = None,
    gt_t2i: dict[int, list[int]] | None = None,
) -> dict[str, float]:
    """
    Computes Recall@K for cross-modal retrieval.

    With gt_i2t / gt_t2i = None, assumes diagonal 1:1 ground truth
    (image i paired with text i).  Pass explicit GT dicts for multi-caption
    datasets (e.g. COCO: each image has 5 captions).

    Args:
        image_embs: (N_img, d) L2-normalized.
        text_embs:  (N_txt, d) L2-normalized.
        ks:         tuple of K values.
        gt_i2t:     {img_idx: [txt_idx, ...]}  — ground truth for i→t.
        gt_t2i:     {txt_idx: [img_idx, ...]}  — ground truth for t→i.
    Returns:
        Dict with keys 'i2t_R@K', 't2i_R@K', and 'avg_R'.
    """
    N_img = image_embs.shape[0]
    N_txt = text_embs.shape[0]
    sims = image_embs @ text_embs.T   # (N_img, N_txt)
    results: dict[str, float] = {}

    # ---- image → text ----
    topk_max = max(ks)
    i2t_top = sims.topk(min(topk_max, N_txt), dim=1).indices  # (N_img, K)
    for k in ks:
        if gt_i2t is None:
            # 1:1 diagonal GT
            gt = torch.arange(N_img).unsqueeze(1)
            hit = (i2t_top[:, :k] == gt).any(dim=1).float().mean().item()
        else:
            hits = []
            for i in range(N_img):
                gt_set = set(gt_i2t.get(i, [i]))
                retrieved = set(i2t_top[i, :k].tolist())
                hits.append(float(bool(gt_set & retrieved)))
            hit = sum(hits) / len(hits)
        results[f"i2t_R@{k}"] = hit

    # ---- text → image ----
    t2i_top = sims.T.topk(min(topk_max, N_img), dim=1).indices  # (N_txt, K)
    for k in ks:
        if gt_t2i is None:
            if N_img == N_txt:
                gt = torch.arange(N_txt).unsqueeze(1)
                hit = (t2i_top[:, :k] == gt).any(dim=1).float().mean().item()
            else:
                # N_txt > N_img: text j belongs to image j // n_captions
                n_cap = N_txt // N_img
                gt = (torch.arange(N_txt) // n_cap).unsqueeze(1)
                hit = (t2i_top[:, :k] == gt).any(dim=1).float().mean().item()
        else:
            hits = []
            for j in range(N_txt):
                gt_set = set(gt_t2i.get(j, [j]))
                retrieved = set(t2i_top[j, :k].tolist())
                hits.append(float(bool(gt_set & retrieved)))
            hit = sum(hits) / len(hits)
        results[f"t2i_R@{k}"] = hit

    results["avg_R"] = sum(results.values()) / len(results)
    return results


def mean_average_precision(
    query_embs: torch.Tensor,
    gallery_embs: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> float:
    """
    mAP for multi-label retrieval (NUS-WIDE, MIR-Flickr).

    query_labels / gallery_labels: (N, C) binary multi-hot tensors.
    Relevant = at least one shared label.
    """
    sims = query_embs @ gallery_embs.T   # (Nq, Ng)
    aps = []
    for i in range(sims.shape[0]):
        ranked = sims[i].argsort(descending=True)
        relevant = (query_labels[i] * gallery_labels[ranked]).sum(dim=1) > 0
        n_rel = relevant.sum().item()
        if n_rel == 0:
            continue
        cum = relevant.float().cumsum(0)
        denom = torch.arange(1, len(ranked) + 1, dtype=torch.float,
                             device=relevant.device)
        precision_at_k = cum / denom
        aps.append((precision_at_k * relevant.float()).sum().item() / n_rel)
    return float(sum(aps) / len(aps)) if aps else 0.0


def recall_at_k_from_sims(
    sims: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
    gt_i2t: dict[int, list[int]] | None = None,
) -> dict[str, float]:
    """Variant that accepts a pre-computed similarity matrix (N_img, N_txt)."""
    fake_img = torch.eye(sims.shape[0], sims.shape[1])
    fake_txt = torch.eye(sims.shape[1], sims.shape[0])
    # We can't reconstruct embeddings from sims; use the sims-based path directly.
    N_img, N_txt = sims.shape
    results: dict[str, float] = {}
    topk_max = max(ks)
    i2t_top = sims.topk(min(topk_max, N_txt), dim=1).indices

    for k in ks:
        if gt_i2t is None:
            gt = torch.arange(N_img).unsqueeze(1)
            hit = (i2t_top[:, :k] == gt).any(dim=1).float().mean().item()
        else:
            hits = []
            for i in range(N_img):
                gt_set = set(gt_i2t.get(i, [i]))
                hits.append(float(bool(gt_set & set(i2t_top[i, :k].tolist()))))
            hit = sum(hits) / len(hits)
        results[f"i2t_R@{k}"] = hit

    t2i_top = sims.T.topk(min(topk_max, N_img), dim=1).indices
    n_cap = N_txt // N_img if N_txt > N_img else 1
    for k in ks:
        gt = (torch.arange(N_txt) // n_cap).unsqueeze(1)
        hit = (t2i_top[:, :k] == gt).any(dim=1).float().mean().item()
        results[f"t2i_R@{k}"] = hit

    results["avg_R"] = sum(results.values()) / len(results)
    return results

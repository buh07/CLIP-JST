from __future__ import annotations

import torch


def effective_rank(x: torch.Tensor, max_rows: int = 5000) -> dict[str, float]:
    if x.shape[0] > max_rows:
        idx = torch.randperm(x.shape[0])[:max_rows]
        x = x[idx]
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    s = s[s > 1e-12]
    if len(s) == 0:
        return {"effective_rank": 0.0, "participation_ratio": 0.0}
    p = s / s.sum()
    ent = -(p * (p + 1e-12).log()).sum()
    erank = ent.exp().item()
    pr = (s.sum().pow(2) / (s.pow(2).sum() + 1e-12)).item()
    return {"effective_rank": erank, "participation_ratio": pr}


def modality_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.mean(dim=0) - b.mean(dim=0)).norm().item()


def centroid_distance_matrix(embeddings: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    centroids = {k: v.mean(dim=0) for k, v in embeddings.items()}
    out: dict[str, dict[str, float]] = {}
    for a, ca in centroids.items():
        out[a] = {}
        for b, cb in centroids.items():
            out[a][b] = float((ca - cb).norm().item())
    return out


def pair_diagnostics(a: torch.Tensor, b: torch.Tensor, prefix: str) -> dict[str, float]:
    ra = effective_rank(a)
    rb = effective_rank(b)
    return {
        f"{prefix}_modality_gap_l2": modality_gap(a, b),
        f"{prefix}_a_effective_rank": ra["effective_rank"],
        f"{prefix}_b_effective_rank": rb["effective_rank"],
        f"{prefix}_a_participation_ratio": ra["participation_ratio"],
        f"{prefix}_b_participation_ratio": rb["participation_ratio"],
    }

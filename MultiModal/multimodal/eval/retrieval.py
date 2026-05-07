from __future__ import annotations

import torch


def recall_at_k(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
    gt_i2t: dict[int, list[int]] | None = None,
    gt_t2i: dict[int, list[int]] | None = None,
) -> dict[str, float]:
    n_img = image_embs.shape[0]
    n_txt = text_embs.shape[0]
    sims = image_embs @ text_embs.T
    topk_max = max(ks)
    results: dict[str, float] = {}

    i2t = sims.topk(min(topk_max, n_txt), dim=1).indices
    for k in ks:
        if gt_i2t is None:
            gt = torch.arange(n_img).unsqueeze(1)
            hit = (i2t[:, :k] == gt).any(dim=1).float().mean().item()
        else:
            hits = []
            for i in range(n_img):
                g = set(gt_i2t.get(i, [i]))
                r = set(i2t[i, :k].tolist())
                hits.append(float(bool(g & r)))
            hit = sum(hits) / len(hits)
        results[f"i2t_R@{k}"] = hit

    t2i = sims.T.topk(min(topk_max, n_img), dim=1).indices
    for k in ks:
        if gt_t2i is None:
            if n_img == n_txt:
                gt = torch.arange(n_txt).unsqueeze(1)
                hit = (t2i[:, :k] == gt).any(dim=1).float().mean().item()
            else:
                n_cap = n_txt // n_img
                gt = (torch.arange(n_txt) // n_cap).unsqueeze(1)
                hit = (t2i[:, :k] == gt).any(dim=1).float().mean().item()
        else:
            hits = []
            for j in range(n_txt):
                g = set(gt_t2i.get(j, [j]))
                r = set(t2i[j, :k].tolist())
                hits.append(float(bool(g & r)))
            hit = sum(hits) / len(hits)
        results[f"t2i_R@{k}"] = hit

    results["avg_R"] = sum(results.values()) / len(results)
    return results

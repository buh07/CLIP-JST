from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor | float) -> torch.Tensor:
    """
    Symmetric InfoNCE with CLIP-style logit scale:
      logits = scale * cosine_similarity
    """
    if isinstance(logit_scale, torch.Tensor):
        scale = logit_scale
    else:
        scale = torch.tensor(logit_scale, device=image_emb.device, dtype=image_emb.dtype)

    logits = scale * (image_emb @ text_emb.T)
    labels = torch.arange(image_emb.shape[0], device=image_emb.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)

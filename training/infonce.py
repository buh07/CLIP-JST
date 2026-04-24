"""
InfoNCE (NT-Xent) contrastive loss for cross-modal retrieval.

For a batch of (image_embed, text_embed) pairs, computes the symmetric
InfoNCE loss: both image-to-text and text-to-image directions averaged.
"""

import torch
import torch.nn.functional as F


def infonce_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Args:
        image_emb: (B, d) L2-normalized image embeddings.
        text_emb:  (B, d) L2-normalized text embeddings.
        temperature: softmax temperature tau.
    Returns:
        Scalar loss (average of i2t and t2i directions).
    """
    B = image_emb.shape[0]
    logits = image_emb @ text_emb.T / temperature  # (B, B)
    labels = torch.arange(B, device=image_emb.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2

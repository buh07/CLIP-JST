"""
Spectral contrastive loss (HaoChen et al., NeurIPS 2021).

Used for theoretical-transparency experiments where we need a loss that
connects directly to the spectral-contrastive framework for Claim 2 proofs.
"""

import torch


def spectral_contrastive_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    temperature: float | None = None,  # accepted but unused; keeps trainer API uniform
) -> torch.Tensor:
    """
    Spectral contrastive loss from HaoChen et al. (2021), adapted for
    cross-modal pairs.

        L = -2/B * sum_{pos} <f_v, f_t>  +  1/B^2 * ||f_v f_t^T||_F^2

    This is the spectral reformulation of SimCLR without temperature.
    Not identical to InfoNCE but amenable to the augmentation-graph analysis
    used in Claim 2's generalization bound.
    """
    B = image_emb.shape[0]
    pos_term = -2.0 * (image_emb * text_emb).sum() / B
    cross = image_emb @ text_emb.T          # (B, B)
    neg_term = (cross ** 2).sum() / (B ** 2)
    return pos_term + neg_term

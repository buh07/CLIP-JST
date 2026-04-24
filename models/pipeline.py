"""
Composed CLIP-JST pipeline:
    f_v(v) = M_v^{1/2} Phi_v psi_v(v)
    f_t(t) = M_t^{1/2} Phi_t psi_t(t)

The backbone features psi_v, psi_t are assumed pre-extracted and cached;
this module only wraps the JL + Mahalanobis layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .jl import SparseJL
from .mahalanobis import FullMahalanobis, LowRankMahalanobis


class CLIPJSTPipeline(nn.Module):
    """
    Args:
        vision_dim:   backbone output dim for images (e.g. 768 for ViT-B/32).
        text_dim:     backbone output dim for text  (e.g. 512 for CLIP text enc).
        embed_dim:    shared embedding dim m (output of JL and Mahalanobis).
        mahal_rank:   if None, use full m×m Mahalanobis; else rank-r variant.
        jl_eps:       JL distortion target epsilon.
        jl_seed:      random seed for JL matrices (fixed for reproducibility).
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 512,
        embed_dim: int = 256,
        mahal_rank: int | None = None,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.jl_v = SparseJL(vision_dim, embed_dim, eps=jl_eps, seed=jl_seed)
        self.jl_t = SparseJL(text_dim,   embed_dim, eps=jl_eps, seed=jl_seed + 1)

        if mahal_rank is None:
            self.mahal_v = FullMahalanobis(embed_dim)
            self.mahal_t = FullMahalanobis(embed_dim)
        else:
            self.mahal_v = LowRankMahalanobis(embed_dim, mahal_rank)
            self.mahal_t = LowRankMahalanobis(embed_dim, mahal_rank)

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_v(self.jl_v(v)), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self.jl_t(t)), dim=-1)

    def forward(
        self, v: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

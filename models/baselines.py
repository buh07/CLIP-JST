"""
Baseline models for comparison with CLIP-JST.

1. CLIPProjectionHead   — learnable d×m linear projection (CLIP-style upper bound).
2. RandomProjectionPipeline — JL + frozen identity Mahalanobis (no learning).
3. MahalanobisOnlyPipeline  — learned d→m linear head parameterized as Mahalanobis
                               (no JL; isolates contribution of dimension reduction).
4. PCAPlusMahalanobisPipeline — data-dependent PCA projection + learned Mahalanobis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .jl import SparseJL
from .mahalanobis import FullMahalanobis, LowRankMahalanobis


class CLIPProjectionHead(nn.Module):
    """
    Learnable d_v×m and d_t×m linear projections trained with InfoNCE.
    This is the standard CLIP projection head — the upper-bound reference.
    """

    def __init__(self, vision_dim: int = 768, text_dim: int = 512, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_v = nn.Linear(vision_dim, embed_dim, bias=False)
        self.proj_t = nn.Linear(text_dim, embed_dim, bias=False)
        nn.init.normal_(self.proj_v.weight, std=vision_dim ** -0.5)
        nn.init.normal_(self.proj_t.weight, std=text_dim ** -0.5)

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_v(v), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_t(t), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RandomProjectionPipeline(nn.Module):
    """
    Frozen sparse JL with identity Mahalanobis (no training at all).
    Isolates the contribution of the learned Mahalanobis head.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 512,
        embed_dim: int = 256,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.jl_v = SparseJL(vision_dim, embed_dim, eps=jl_eps, seed=jl_seed)
        self.jl_t = SparseJL(text_dim,   embed_dim, eps=jl_eps, seed=jl_seed + 1)

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.jl_v(v), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.jl_t(t), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return 0


class MahalanobisOnlyPipeline(nn.Module):
    """
    Learned Mahalanobis head applied directly to raw backbone features (no JL).
    Parameterized as d_v→m and d_t→m rank-r linear projections so both
    modalities map to the same m-dimensional space.

    This isolates the contribution of dimension reduction (JL vs learned).
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 512,
        embed_dim: int = 256,
        mahal_rank: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        if mahal_rank is None:
            # Full m×m Mahalanobis in the projected space — first project d→m,
            # then apply full Mahalanobis.  We fuse both into a single d×m matrix.
            self.proj_v = nn.Linear(vision_dim, embed_dim, bias=False)
            self.proj_t = nn.Linear(text_dim,   embed_dim, bias=False)
            self.mahal_v = FullMahalanobis(embed_dim)
            self.mahal_t = FullMahalanobis(embed_dim)
            self._use_mahal = True
        else:
            # Low-rank: map d→m via rank-r factor (analogous to LoRA).
            self.proj_v = LowRankMahalanobis(vision_dim, mahal_rank)
            self.proj_t = LowRankMahalanobis(text_dim,   mahal_rank)
            # proj maps d→d (L^T L), so we also need a readout to m.
            self.read_v = nn.Linear(vision_dim, embed_dim, bias=False)
            self.read_t = nn.Linear(text_dim,   embed_dim, bias=False)
            self._use_mahal = False

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        if self._use_mahal:
            return F.normalize(self.mahal_v(self.proj_v(v)), dim=-1)
        return F.normalize(self.read_v(self.proj_v(v)), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        if self._use_mahal:
            return F.normalize(self.mahal_t(self.proj_t(t)), dim=-1)
        return F.normalize(self.read_t(self.proj_t(t)), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PCAPlusMahalanobisPipeline(nn.Module):
    """
    Data-dependent PCA projection to m dims, followed by learned Mahalanobis.

    PCA components are computed once from training features via fit_pca() and
    then frozen.  Only the Mahalanobis matrices are trained.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 512,
        embed_dim: int = 256,
        mahal_rank: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # Placeholder PCA buffers; overwritten by fit_pca().
        self.register_buffer("pca_v", torch.zeros(embed_dim, vision_dim))
        self.register_buffer("mean_v", torch.zeros(vision_dim))
        self.register_buffer("pca_t", torch.zeros(embed_dim, text_dim))
        self.register_buffer("mean_t", torch.zeros(text_dim))

        if mahal_rank is None:
            self.mahal_v = FullMahalanobis(embed_dim)
            self.mahal_t = FullMahalanobis(embed_dim)
        else:
            self.mahal_v = LowRankMahalanobis(embed_dim, mahal_rank)
            self.mahal_t = LowRankMahalanobis(embed_dim, mahal_rank)

    @torch.no_grad()
    def fit_pca(self, img_feats: torch.Tensor, txt_feats: torch.Tensor) -> None:
        """Compute top-m PCA directions from training features (CPU tensors)."""
        self.mean_v.copy_(img_feats.mean(0))
        self.mean_t.copy_(txt_feats.mean(0))
        _, _, Vt_v = torch.linalg.svd(img_feats - self.mean_v, full_matrices=False)
        _, _, Vt_t = torch.linalg.svd(txt_feats - self.mean_t, full_matrices=False)
        self.pca_v.copy_(Vt_v[:self.embed_dim])
        self.pca_t.copy_(Vt_t[:self.embed_dim])

    def _project_v(self, v: torch.Tensor) -> torch.Tensor:
        return (v - self.mean_v) @ self.pca_v.T

    def _project_t(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.mean_t) @ self.pca_t.T

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_v(self._project_v(v)), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self._project_t(t)), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

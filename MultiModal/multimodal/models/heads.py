from __future__ import annotations

from functools import lru_cache
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .jl_ops import DenseSparseJL, kane_nelson_jl


@lru_cache(maxsize=8)
def _load_clip_projection_weights(model_name: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Load CLIP projection matrices once and cache tensors on CPU.

    Returns:
      visual_projection weight: (512, vision_dim)
      text_projection weight:   (512, text_dim)
      logit_scale scalar (float)
    """
    try:
        from transformers import CLIPModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required for pretrained CLIP projection baselines; "
            "install it in the active environment."
        ) from exc

    m = CLIPModel.from_pretrained(model_name)
    with torch.no_grad():
        wv = m.visual_projection.weight.detach().cpu().float().clone()
        wt = m.text_projection.weight.detach().cpu().float().clone()
        ls = float(m.logit_scale.detach().cpu().item())
    del m
    return wv, wt, ls


def _project_clip_base_to_dim(
    *,
    model_name: str,
    vision_dim: int,
    text_dim: int,
    embed_dim: int,
    jl_eps: float,
    jl_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Build frozen pretrained base projections to `embed_dim`.

    CLIP base projection is 512-d. For embed_dim < 512, a fixed shared JL map
    from 512->embed_dim is applied to both modalities so cross-modal alignment is
    preserved under the same bottleneck.
    """
    wv, wt, ls = _load_clip_projection_weights(model_name)
    if int(wv.shape[1]) != int(vision_dim):
        raise ValueError(f"vision dim mismatch: cache={vision_dim}, clip_proj={wv.shape[1]}")
    if int(wt.shape[1]) != int(text_dim):
        raise ValueError(f"text dim mismatch: cache={text_dim}, clip_proj={wt.shape[1]}")

    clip_dim = int(wv.shape[0])
    if embed_dim == clip_dim:
        return wv, wt, ls
    if embed_dim > clip_dim:
        raise ValueError(f"embed_dim={embed_dim} cannot exceed CLIP projection dim={clip_dim}")

    phi = torch.tensor(
        kane_nelson_jl(clip_dim, embed_dim, eps=jl_eps, seed=jl_seed).toarray(),
        dtype=torch.float32,
    )
    return phi @ wv, phi @ wt, ls


def _topk_right_singular_basis(weight: torch.Tensor, k: int) -> torch.Tensor:
    """
    Return top-k right singular vectors (as column basis) of a projection weight.

    Args:
      weight: (out_dim, in_dim)
      k: number of singular vectors
    Returns:
      basis: (in_dim, k_eff) with orthonormal columns.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    with torch.no_grad():
        _u, _s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        k_eff = min(int(k), int(vh.shape[0]))
        return vh[:k_eff, :].T.contiguous()


class CLIPProjectionHead(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_v = nn.Linear(vision_dim, embed_dim, bias=False)
        self.proj_t = nn.Linear(text_dim, embed_dim, bias=False)
        nn.init.normal_(self.proj_v.weight, std=vision_dim ** -0.5)
        nn.init.normal_(self.proj_t.weight, std=text_dim ** -0.5)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_v(v), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_t(t), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class OrthogonalProjectionHead(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, embed_dim: int, orth_reg: float = 1e-3):
        super().__init__()
        self.embed_dim = embed_dim
        self.orth_reg = orth_reg
        self.proj_v = nn.Linear(vision_dim, embed_dim, bias=False)
        self.proj_t = nn.Linear(text_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.proj_v.weight)
        nn.init.orthogonal_(self.proj_t.weight)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_v(v), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_t(t), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def regularization_loss(self) -> torch.Tensor:
        if self.orth_reg <= 0:
            return torch.zeros([], device=self.proj_v.weight.device)
        eye = torch.eye(self.embed_dim, device=self.proj_v.weight.device, dtype=self.proj_v.weight.dtype)
        gv = self.proj_v.weight @ self.proj_v.weight.T
        gt = self.proj_t.weight @ self.proj_t.weight.T
        return self.orth_reg * ((gv - eye).pow(2).mean() + (gt - eye).pow(2).mean())

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RandomJLMahalanobisHead(nn.Module):
    """Fixed sparse JL projection + trainable full linear Mahalanobis readout."""

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.jl_v = DenseSparseJL(vision_dim, embed_dim, eps=jl_eps, seed=jl_seed)
        self.jl_t = DenseSparseJL(text_dim, embed_dim, eps=jl_eps, seed=jl_seed + 1)
        self.mahal_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mahal_t = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.mahal_v.weight)
        nn.init.eye_(self.mahal_t.weight)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_v(self.jl_v(v)), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self.jl_t(t)), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FrozenPretrainedCLIPHead(nn.Module):
    """
    Frozen pretrained CLIP projection baseline.

    Uses CLIP pretrained projection matrices (optionally projected to embed_dim
    through a shared fixed JL map when embed_dim < 512) and keeps them frozen.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        pretrained_model_name: str = "openai/clip-vit-base-patch32",
        jl_eps: float = 0.1,
        jl_seed: int = 42,
        train_logit_scale: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)

        base_v, base_t, ls = _project_clip_base_to_dim(
            model_name=pretrained_model_name,
            vision_dim=vision_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            jl_eps=jl_eps,
            jl_seed=jl_seed,
        )
        self.base_v = nn.Linear(vision_dim, embed_dim, bias=False)
        self.base_t = nn.Linear(text_dim, embed_dim, bias=False)
        with torch.no_grad():
            self.base_v.weight.copy_(base_v)
            self.base_t.weight.copy_(base_t)
        self.base_v.weight.requires_grad_(False)
        self.base_t.weight.requires_grad_(False)
        self.logit_scale = nn.Parameter(torch.tensor(ls, dtype=torch.float32), requires_grad=bool(train_logit_scale))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.base_v(v), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.base_t(t), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FrozenPretrainedCLIPMahalHead(nn.Module):
    """
    Frozen pretrained CLIP projection + trainable full Mahalanobis.

    This is the key pretrained-frozen control:
      fixed pretrained projection -> trainable linear readout in embed space.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        pretrained_model_name: str = "openai/clip-vit-base-patch32",
        jl_eps: float = 0.1,
        jl_seed: int = 42,
        train_logit_scale: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)

        base_v, base_t, ls = _project_clip_base_to_dim(
            model_name=pretrained_model_name,
            vision_dim=vision_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            jl_eps=jl_eps,
            jl_seed=jl_seed,
        )
        self.base_v = nn.Linear(vision_dim, embed_dim, bias=False)
        self.base_t = nn.Linear(text_dim, embed_dim, bias=False)
        with torch.no_grad():
            self.base_v.weight.copy_(base_v)
            self.base_t.weight.copy_(base_t)
        self.base_v.weight.requires_grad_(False)
        self.base_t.weight.requires_grad_(False)

        self.mahal_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mahal_t = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.mahal_v.weight)
        nn.init.eye_(self.mahal_t.weight)

        self.logit_scale = nn.Parameter(torch.tensor(ls, dtype=torch.float32), requires_grad=bool(train_logit_scale))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_v(self.base_v(v)), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self.base_t(t)), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpectralAlignedProjectionHead(nn.Module):
    """
    Trainable projection with explicit spectral-subspace alignment objective.

    The model optimizes InfoNCE plus a regularizer that aligns the row-space of
    trainable projections to the top right-singular subspaces of the pretrained
    CLIP projection matrices.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        pretrained_model_name: str = "openai/clip-vit-base-patch32",
        spectral_reg: float = 1e-2,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.spectral_reg = float(spectral_reg)

        self.proj_v = nn.Linear(vision_dim, embed_dim, bias=False)
        self.proj_t = nn.Linear(text_dim, embed_dim, bias=False)
        nn.init.normal_(self.proj_v.weight, std=vision_dim ** -0.5)
        nn.init.normal_(self.proj_t.weight, std=text_dim ** -0.5)

        wv, wt, ls = _load_clip_projection_weights(pretrained_model_name)
        if int(wv.shape[1]) != int(vision_dim):
            raise ValueError(f"vision dim mismatch: expected {vision_dim}, got {wv.shape[1]}")
        if int(wt.shape[1]) != int(text_dim):
            raise ValueError(f"text dim mismatch: expected {text_dim}, got {wt.shape[1]}")

        ref_v = _topk_right_singular_basis(wv, embed_dim)
        ref_t = _topk_right_singular_basis(wt, embed_dim)
        self.register_buffer("ref_basis_v", ref_v)
        self.register_buffer("ref_basis_t", ref_t)
        self.logit_scale = nn.Parameter(torch.tensor(ls, dtype=torch.float32))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_v(v), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_t(t), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    @staticmethod
    def _rowspace_basis(weight: torch.Tensor) -> torch.Tensor:
        # weight: (m, d) -> row-space basis in R^d, shape (d, k)
        q, _r = torch.linalg.qr(weight.T, mode="reduced")
        return q

    @staticmethod
    def _subspace_overlap(q: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # q: (d, kq), ref: (d, kr), both orthonormal columns.
        k = min(int(q.shape[1]), int(ref.shape[1]))
        qk = q[:, :k]
        rk = ref[:, :k]
        m = qk.T @ rk
        return (m * m).sum() / float(max(k, 1))

    def spectral_alignment_metrics(self) -> dict[str, float]:
        with torch.no_grad():
            qv = self._rowspace_basis(self.proj_v.weight)
            qt = self._rowspace_basis(self.proj_t.weight)
            ov_v = float(self._subspace_overlap(qv, self.ref_basis_v).item())
            ov_t = float(self._subspace_overlap(qt, self.ref_basis_t).item())
        return {
            "spectral_overlap_image": ov_v,
            "spectral_overlap_text": ov_t,
            "spectral_overlap_mean": 0.5 * (ov_v + ov_t),
            "spectral_reg": float(self.spectral_reg),
        }

    def regularization_loss(self) -> torch.Tensor:
        if self.spectral_reg <= 0:
            return torch.zeros([], device=self.proj_v.weight.device)
        qv = self._rowspace_basis(self.proj_v.weight)
        qt = self._rowspace_basis(self.proj_t.weight)
        ov_v = self._subspace_overlap(qv, self.ref_basis_v)
        ov_t = self._subspace_overlap(qt, self.ref_basis_t)
        # maximize overlap => minimize (1 - overlap)
        return self.spectral_reg * ((1.0 - ov_v) + (1.0 - ov_t))

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RandomJLOnlyHead(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, embed_dim: int, jl_eps: float = 0.1, jl_seed: int = 42):
        super().__init__()
        self.embed_dim = embed_dim
        self.jl_v = DenseSparseJL(vision_dim, embed_dim, eps=jl_eps, seed=jl_seed)
        self.jl_t = DenseSparseJL(text_dim, embed_dim, eps=jl_eps, seed=jl_seed + 1)

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.jl_v(v), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.jl_t(t), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return 0


class MRLProjectionHead(nn.Module):
    """
    Matryoshka-style nested representation head.

    One shared max-dimensional projection is trained with averaged InfoNCE over
    nested prefix dimensions (e.g., 64/128/256/512).
    """

    def __init__(self, vision_dim: int, text_dim: int, max_dim: int, nested_dims: list[int]):
        super().__init__()
        self.max_dim = max_dim
        self.nested_dims = sorted(nested_dims)
        self.proj_v = nn.Linear(vision_dim, max_dim, bias=False)
        self.proj_t = nn.Linear(text_dim, max_dim, bias=False)
        nn.init.normal_(self.proj_v.weight, std=vision_dim ** -0.5)
        nn.init.normal_(self.proj_t.weight, std=text_dim ** -0.5)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def _encode(self, x: torch.Tensor, proj: nn.Linear, dim: int | None = None) -> torch.Tensor:
        z = proj(x)
        if dim is not None:
            z = z[:, :dim]
        return F.normalize(z, dim=-1)

    def encode_image(self, v: torch.Tensor, dim: int | None = None) -> torch.Tensor:
        return self._encode(v, self.proj_v, dim=dim)

    def encode_text(self, t: torch.Tensor, dim: int | None = None) -> torch.Tensor:
        return self._encode(t, self.proj_t, dim=dim)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DirectCLRProxyHead(nn.Module):
    """
    Frozen-pipeline DirectCLR proxy.

    InfoNCE is applied to a fixed random subvector during training, while full
    embedding is used at evaluation.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        full_dim: int,
        train_subdim: int,
        seed: int = 42,
    ):
        super().__init__()
        if train_subdim > full_dim:
            raise ValueError("train_subdim must be <= full_dim")
        self.full_dim = full_dim
        self.train_subdim = train_subdim
        self.proj_v = nn.Linear(vision_dim, full_dim, bias=False)
        self.proj_t = nn.Linear(text_dim, full_dim, bias=False)
        nn.init.normal_(self.proj_v.weight, std=vision_dim ** -0.5)
        nn.init.normal_(self.proj_t.weight, std=text_dim ** -0.5)

        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(full_dim, generator=g)
        self.register_buffer("train_idx", perm[:train_subdim])
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_v(v), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_t(t), dim=-1)

    def train_views(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zv = self.proj_v(v)[:, self.train_idx]
        zt = self.proj_t(t)[:, self.train_idx]
        return F.normalize(zv, dim=-1), F.normalize(zt, dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LearnedJLSparseHead(nn.Module):
    """
    Learned sparse-JL head initialized from JL matrices.

    Effective projection is W \\odot sigmoid(M), where W is trainable JL-initialized
    weights and M are trainable mask logits encouraging structured sparsity.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
        sparsity_reg: float = 1e-4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sparsity_reg = sparsity_reg

        phi_v = torch.tensor(kane_nelson_jl(vision_dim, embed_dim, eps=jl_eps, seed=jl_seed).toarray(), dtype=torch.float32)
        phi_t = torch.tensor(kane_nelson_jl(text_dim, embed_dim, eps=jl_eps, seed=jl_seed + 1).toarray(), dtype=torch.float32)

        self.weight_v = nn.Parameter(phi_v.clone())
        self.weight_t = nn.Parameter(phi_t.clone())

        # High probability where JL has support, low elsewhere.
        init_mask_v = torch.where(phi_v.abs() > 0, torch.full_like(phi_v, 2.0), torch.full_like(phi_v, -2.0))
        init_mask_t = torch.where(phi_t.abs() > 0, torch.full_like(phi_t, 2.0), torch.full_like(phi_t, -2.0))
        self.mask_logits_v = nn.Parameter(init_mask_v)
        self.mask_logits_t = nn.Parameter(init_mask_t)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def _proj(self, x: torch.Tensor, weight: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(mask_logits)
        eff_w = weight * gates
        return x @ eff_w.T

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self._proj(v, self.weight_v, self.mask_logits_v), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self._proj(t, self.weight_t, self.mask_logits_t), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def regularization_loss(self) -> torch.Tensor:
        if self.sparsity_reg <= 0:
            return torch.zeros([], device=self.weight_v.device)
        gate_v = torch.sigmoid(self.mask_logits_v)
        gate_t = torch.sigmoid(self.mask_logits_t)
        return self.sparsity_reg * (gate_v.mean() + gate_t.mean())

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SparseJLProjectedHead(nn.Module):
    """
    Run 1A: train JL-initialized projections under fixed Kane-Nelson sparsity support.

    After each optimizer step, weights are projected back to the original nonzero
    support so off-support entries remain exactly zero.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        phi_v = torch.tensor(
            kane_nelson_jl(vision_dim, embed_dim, eps=jl_eps, seed=jl_seed).toarray(),
            dtype=torch.float32,
        )
        phi_t = torch.tensor(
            kane_nelson_jl(text_dim, embed_dim, eps=jl_eps, seed=jl_seed + 1).toarray(),
            dtype=torch.float32,
        )
        support_v = (phi_v != 0).float()
        support_t = (phi_t != 0).float()
        self.register_buffer("support_v", support_v)
        self.register_buffer("support_t", support_t)

        self.weight_v = nn.Parameter(phi_v.clone())
        self.weight_t = nn.Parameter(phi_t.clone())
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))
        self.project_parameters()

    def project_parameters(self) -> None:
        with torch.no_grad():
            self.weight_v.mul_(self.support_v)
            self.weight_t.mul_(self.support_t)

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(v @ self.weight_v.T, dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(t @ self.weight_t.T, dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        # Effective trainable count: values on fixed sparse support + logit scale.
        nnz = int(self.support_v.sum().item() + self.support_t.sum().item())
        return nnz + int(self.logit_scale.numel())


class SparseJLL1Head(nn.Module):
    """
    Run 1B: train dense JL-initialized projections with L1 shrinkage.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
        lambda_l1: float = 1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.lambda_l1 = float(lambda_l1)

        phi_v = torch.tensor(
            kane_nelson_jl(vision_dim, embed_dim, eps=jl_eps, seed=jl_seed).toarray(),
            dtype=torch.float32,
        )
        phi_t = torch.tensor(
            kane_nelson_jl(text_dim, embed_dim, eps=jl_eps, seed=jl_seed + 1).toarray(),
            dtype=torch.float32,
        )
        self.weight_v = nn.Parameter(phi_v.clone())
        self.weight_t = nn.Parameter(phi_t.clone())
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(v @ self.weight_v.T, dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(t @ self.weight_t.T, dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def regularization_loss(self) -> torch.Tensor:
        if self.lambda_l1 <= 0:
            return torch.zeros([], device=self.weight_v.device)
        l1 = self.weight_v.abs().mean() + self.weight_t.abs().mean()
        return self.lambda_l1 * l1

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class OrthogonalPlusMahalanobisHead(nn.Module):
    """
    Run 2: orthogonal trainable projection followed by full trainable Mahalanobis.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        orth_reg: float = 1e-3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.orth_reg = orth_reg
        self.proj_v = nn.Linear(vision_dim, embed_dim, bias=False)
        self.proj_t = nn.Linear(text_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.proj_v.weight)
        nn.init.orthogonal_(self.proj_t.weight)

        self.mahal_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mahal_t = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.mahal_v.weight)
        nn.init.eye_(self.mahal_t.weight)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_v(self.proj_v(v)), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self.proj_t(t)), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def regularization_loss(self) -> torch.Tensor:
        if self.orth_reg <= 0:
            return torch.zeros([], device=self.proj_v.weight.device)
        eye = torch.eye(self.embed_dim, device=self.proj_v.weight.device, dtype=self.proj_v.weight.dtype)
        gv = self.proj_v.weight @ self.proj_v.weight.T
        gt = self.proj_t.weight @ self.proj_t.weight.T
        return self.orth_reg * ((gv - eye).pow(2).mean() + (gt - eye).pow(2).mean())

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MahalanobisOnlyHead(nn.Module):
    """
    Trainable full linear Mahalanobis on shared raw feature width.

    For modality alignment when text_dim < shared_raw_dim, text features are
    zero-padded to shared_raw_dim before projection.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        *,
        shared_raw_dim: int = 768,
    ):
        super().__init__()
        if shared_raw_dim < max(vision_dim, text_dim):
            raise ValueError("shared_raw_dim must be >= max(vision_dim, text_dim)")
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.shared_raw_dim = shared_raw_dim

        self.mahal_v = nn.Linear(shared_raw_dim, shared_raw_dim, bias=False)
        self.mahal_t = nn.Linear(shared_raw_dim, shared_raw_dim, bias=False)
        nn.init.eye_(self.mahal_v.weight)
        nn.init.eye_(self.mahal_t.weight)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def _pad_shared(self, x: torch.Tensor, raw_dim: int) -> torch.Tensor:
        if raw_dim == self.shared_raw_dim:
            return x
        if raw_dim > self.shared_raw_dim:
            raise ValueError("raw_dim cannot exceed shared_raw_dim")
        pad = self.shared_raw_dim - raw_dim
        return F.pad(x, (0, pad), mode="constant", value=0.0)

    def _raw_image(self, v: torch.Tensor) -> torch.Tensor:
        return self._pad_shared(v, self.vision_dim)

    def _raw_text(self, t: torch.Tensor) -> torch.Tensor:
        return self._pad_shared(t, self.text_dim)

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_v(self._raw_image(v)), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self._raw_text(t)), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MahalanobisBottleneckHead(nn.Module):
    """
    Dimension-matched Mahalanobis baseline:
      raw -> trainable bottleneck (d -> m) -> trainable full Mahalanobis (m -> m)

    This is the fair counterpart to 256-d methods in Stage13/14.
    """

    def __init__(self, vision_dim: int, text_dim: int, embed_dim: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.proj_v = nn.Linear(vision_dim, embed_dim, bias=False)
        self.proj_t = nn.Linear(text_dim, embed_dim, bias=False)
        nn.init.normal_(self.proj_v.weight, std=vision_dim ** -0.5)
        nn.init.normal_(self.proj_t.weight, std=text_dim ** -0.5)
        self.mahal_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mahal_t = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.mahal_v.weight)
        nn.init.eye_(self.mahal_t.weight)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_v(self.proj_v(v)), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self.proj_t(t)), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConcatJLMahalanobisHead(nn.Module):
    """
    Proposal A: z = [alpha * R x ; beta * x_shared], then train full Mahalanobis.

    Text raw features are zero-padded to shared_raw_dim (default 768) so both
    modalities concatenate into a shared width.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        shared_raw_dim: int = 768,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
    ):
        super().__init__()
        if shared_raw_dim < max(vision_dim, text_dim):
            raise ValueError("shared_raw_dim must be >= max(vision_dim, text_dim)")
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.shared_raw_dim = shared_raw_dim
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.concat_dim = embed_dim + shared_raw_dim

        self.jl_v = DenseSparseJL(vision_dim, embed_dim, eps=jl_eps, seed=jl_seed)
        self.jl_t = DenseSparseJL(text_dim, embed_dim, eps=jl_eps, seed=jl_seed + 1)
        self.mahal_v = nn.Linear(self.concat_dim, self.concat_dim, bias=False)
        self.mahal_t = nn.Linear(self.concat_dim, self.concat_dim, bias=False)
        nn.init.eye_(self.mahal_v.weight)
        nn.init.eye_(self.mahal_t.weight)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def _pad_shared(self, x: torch.Tensor, raw_dim: int) -> torch.Tensor:
        if raw_dim == self.shared_raw_dim:
            return x
        if raw_dim > self.shared_raw_dim:
            raise ValueError("raw_dim cannot exceed shared_raw_dim")
        pad = self.shared_raw_dim - raw_dim
        return F.pad(x, (0, pad), mode="constant", value=0.0)

    def _concat_image(self, v: torch.Tensor) -> torch.Tensor:
        jl = self.jl_v(v)
        raw = self._pad_shared(v, self.vision_dim)
        return torch.cat([self.alpha * jl, self.beta * raw], dim=-1)

    def _concat_text(self, t: torch.Tensor) -> torch.Tensor:
        jl = self.jl_t(t)
        raw = self._pad_shared(t, self.text_dim)
        return torch.cat([self.alpha * jl, self.beta * raw], dim=-1)

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_v(self._concat_image(v)), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self._concat_text(t)), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MaskConcatJLMahalanobisHead(nn.Module):
    """
    Proposal B: z = [alpha * R x ; beta * M x_shared], with fixed Bernoulli mask.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        shared_raw_dim: int = 768,
        p: float = 0.5,
        mask_seed: int = 0,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
    ):
        super().__init__()
        if shared_raw_dim < max(vision_dim, text_dim):
            raise ValueError("shared_raw_dim must be >= max(vision_dim, text_dim)")
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be in [0, 1]")
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.shared_raw_dim = shared_raw_dim
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.p = float(p)
        self.mask_seed = int(mask_seed)
        self.concat_dim = embed_dim + shared_raw_dim

        self.jl_v = DenseSparseJL(vision_dim, embed_dim, eps=jl_eps, seed=jl_seed)
        self.jl_t = DenseSparseJL(text_dim, embed_dim, eps=jl_eps, seed=jl_seed + 1)
        self.mahal_v = nn.Linear(self.concat_dim, self.concat_dim, bias=False)
        self.mahal_t = nn.Linear(self.concat_dim, self.concat_dim, bias=False)
        nn.init.eye_(self.mahal_v.weight)
        nn.init.eye_(self.mahal_t.weight)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

        gv = torch.Generator().manual_seed(mask_seed)
        gt = torch.Generator().manual_seed(mask_seed + 1)
        mask_v = (torch.rand(shared_raw_dim, generator=gv) < p).float()
        mask_t = (torch.rand(shared_raw_dim, generator=gt) < p).float()
        self.register_buffer("mask_v", mask_v)
        self.register_buffer("mask_t", mask_t)

    def _pad_shared(self, x: torch.Tensor, raw_dim: int) -> torch.Tensor:
        if raw_dim == self.shared_raw_dim:
            return x
        if raw_dim > self.shared_raw_dim:
            raise ValueError("raw_dim cannot exceed shared_raw_dim")
        pad = self.shared_raw_dim - raw_dim
        return F.pad(x, (0, pad), mode="constant", value=0.0)

    def _concat_image(self, v: torch.Tensor) -> torch.Tensor:
        jl = self.jl_v(v)
        raw = self._pad_shared(v, self.vision_dim) * self.mask_v
        return torch.cat([self.alpha * jl, self.beta * raw], dim=-1)

    def _concat_text(self, t: torch.Tensor) -> torch.Tensor:
        jl = self.jl_t(t)
        raw = self._pad_shared(t, self.text_dim) * self.mask_t
        return torch.cat([self.alpha * jl, self.beta * raw], dim=-1)

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_v(self._concat_image(v)), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self._concat_text(t)), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def mask_stats(self) -> dict[str, float]:
        return {
            "mask_p_target": self.p,
            "mask_p_image": float(self.mask_v.mean().item()),
            "mask_p_text": float(self.mask_t.mean().item()),
        }

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MaskConcatBudgetMatchedHead(nn.Module):
    """
    Budget-matched mask-concat head.

    Trains with the 256-d bottleneck in-loop:
      z = [alpha * R x ; beta * M x_shared]  (concat_dim)
      z_budget = J_budget(z)                 (fixed JL concat_dim -> budget_dim)
      z_out = M_budget(z_budget)             (trainable Mahalanobis in budget space)
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        shared_raw_dim: int = 768,
        p: float = 0.5,
        mask_seed: int = 0,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
        budget_dim: int | None = None,
        budget_jl_seed: int = 31415,
    ):
        super().__init__()
        if shared_raw_dim < max(vision_dim, text_dim):
            raise ValueError("shared_raw_dim must be >= max(vision_dim, text_dim)")
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be in [0, 1]")
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.shared_raw_dim = shared_raw_dim
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.p = float(p)
        self.mask_seed = int(mask_seed)

        self.jl_dim = int(embed_dim)
        self.concat_dim = self.jl_dim + self.shared_raw_dim
        self.budget_dim = int(embed_dim if budget_dim is None else budget_dim)
        if self.budget_dim > self.concat_dim:
            raise ValueError("budget_dim cannot exceed concat_dim")

        self.jl_v = DenseSparseJL(vision_dim, self.jl_dim, eps=jl_eps, seed=jl_seed)
        self.jl_t = DenseSparseJL(text_dim, self.jl_dim, eps=jl_eps, seed=jl_seed + 1)
        self.budget_jl = DenseSparseJL(self.concat_dim, self.budget_dim, eps=jl_eps, seed=budget_jl_seed)

        self.mahal_v = nn.Linear(self.budget_dim, self.budget_dim, bias=False)
        self.mahal_t = nn.Linear(self.budget_dim, self.budget_dim, bias=False)
        nn.init.eye_(self.mahal_v.weight)
        nn.init.eye_(self.mahal_t.weight)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

        gv = torch.Generator().manual_seed(mask_seed)
        gt = torch.Generator().manual_seed(mask_seed + 1)
        mask_v = (torch.rand(shared_raw_dim, generator=gv) < p).float()
        mask_t = (torch.rand(shared_raw_dim, generator=gt) < p).float()
        self.register_buffer("mask_v", mask_v)
        self.register_buffer("mask_t", mask_t)

    def _pad_shared(self, x: torch.Tensor, raw_dim: int) -> torch.Tensor:
        if raw_dim == self.shared_raw_dim:
            return x
        if raw_dim > self.shared_raw_dim:
            raise ValueError("raw_dim cannot exceed shared_raw_dim")
        pad = self.shared_raw_dim - raw_dim
        return F.pad(x, (0, pad), mode="constant", value=0.0)

    def _concat_image(self, v: torch.Tensor) -> torch.Tensor:
        jl = self.jl_v(v)
        raw = self._pad_shared(v, self.vision_dim) * self.mask_v
        return torch.cat([self.alpha * jl, self.beta * raw], dim=-1)

    def _concat_text(self, t: torch.Tensor) -> torch.Tensor:
        jl = self.jl_t(t)
        raw = self._pad_shared(t, self.text_dim) * self.mask_t
        return torch.cat([self.alpha * jl, self.beta * raw], dim=-1)

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        z = self.budget_jl(self._concat_image(v))
        return F.normalize(self.mahal_v(z), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        z = self.budget_jl(self._concat_text(t))
        return F.normalize(self.mahal_t(z), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def mask_stats(self) -> dict[str, float]:
        return {
            "mask_p_target": self.p,
            "mask_p_image": float(self.mask_v.mean().item()),
            "mask_p_text": float(self.mask_t.mean().item()),
        }

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FedCLIPPretrainedAdapterHead(nn.Module):
    """
    Pretrained-base FedCLIP-style adapter proxy.

    Unlike the random-base proxy, this head freezes CLIP pretrained projections
    (optionally projected to embed_dim) and trains only lightweight adapters.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        adapter_rank: int = 16,
        pretrained_model_name: str = "openai/clip-vit-base-patch32",
        jl_eps: float = 0.1,
        jl_seed: int = 42,
    ):
        super().__init__()
        if adapter_rank <= 0:
            raise ValueError("adapter_rank must be > 0")
        self.embed_dim = embed_dim
        self.adapter_rank = int(adapter_rank)

        base_v, base_t, ls = _project_clip_base_to_dim(
            model_name=pretrained_model_name,
            vision_dim=vision_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            jl_eps=jl_eps,
            jl_seed=jl_seed,
        )

        self.base_v = nn.Linear(vision_dim, embed_dim, bias=False)
        self.base_t = nn.Linear(text_dim, embed_dim, bias=False)
        with torch.no_grad():
            self.base_v.weight.copy_(base_v)
            self.base_t.weight.copy_(base_t)
        self.base_v.weight.requires_grad_(False)
        self.base_t.weight.requires_grad_(False)

        self.down_v = nn.Linear(vision_dim, self.adapter_rank, bias=False)
        self.up_v = nn.Linear(self.adapter_rank, embed_dim, bias=False)
        self.down_t = nn.Linear(text_dim, self.adapter_rank, bias=False)
        self.up_t = nn.Linear(self.adapter_rank, embed_dim, bias=False)
        nn.init.normal_(self.down_v.weight, std=vision_dim ** -0.5)
        nn.init.normal_(self.down_t.weight, std=text_dim ** -0.5)
        nn.init.zeros_(self.up_v.weight)
        nn.init.zeros_(self.up_t.weight)

        self.logit_scale = nn.Parameter(torch.tensor(ls, dtype=torch.float32))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        z = self.base_v(v) + self.up_v(self.down_v(v))
        return F.normalize(z, dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        z = self.base_t(t) + self.up_t(self.down_t(t))
        return F.normalize(z, dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FedMVPPretrainedPromptHead(nn.Module):
    """
    Pretrained-base FedMVP-style prompt proxy.

    Uses frozen CLIP pretrained base projections (optionally projected to
    embed_dim) and learns additive prompt vectors in raw feature space.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        prompt_scale: float = 0.05,
        pretrained_model_name: str = "openai/clip-vit-base-patch32",
        jl_eps: float = 0.1,
        jl_seed: int = 42,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.prompt_scale = float(prompt_scale)

        base_v, base_t, ls = _project_clip_base_to_dim(
            model_name=pretrained_model_name,
            vision_dim=vision_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            jl_eps=jl_eps,
            jl_seed=jl_seed,
        )

        self.base_v = nn.Linear(vision_dim, embed_dim, bias=False)
        self.base_t = nn.Linear(text_dim, embed_dim, bias=False)
        with torch.no_grad():
            self.base_v.weight.copy_(base_v)
            self.base_t.weight.copy_(base_t)
        self.base_v.weight.requires_grad_(False)
        self.base_t.weight.requires_grad_(False)

        self.prompt_v = nn.Parameter(torch.zeros(vision_dim))
        self.prompt_t = nn.Parameter(torch.zeros(text_dim))
        self.logit_scale = nn.Parameter(torch.tensor(ls, dtype=torch.float32))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        z = self.base_v(v + self.prompt_scale * self.prompt_v)
        return F.normalize(z, dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        z = self.base_t(t + self.prompt_scale * self.prompt_t)
        return F.normalize(z, dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FedCLIPProxyHead(nn.Module):
    """
    FedCLIP-style proxy head on frozen features.

    Uses frozen random base projections with trainable low-rank adapters
    (LoRA-style) to emulate adapter-tuning in federated settings.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        adapter_rank: int = 16,
        init_seed: int = 42,
    ):
        super().__init__()
        if adapter_rank <= 0:
            raise ValueError("adapter_rank must be > 0")
        self.embed_dim = embed_dim
        self.adapter_rank = int(adapter_rank)

        self.base_v = nn.Linear(vision_dim, embed_dim, bias=False)
        self.base_t = nn.Linear(text_dim, embed_dim, bias=False)
        g = torch.Generator().manual_seed(init_seed)
        nn.init.normal_(self.base_v.weight, std=vision_dim ** -0.5, generator=g)
        nn.init.normal_(self.base_t.weight, std=text_dim ** -0.5, generator=g)
        self.base_v.weight.requires_grad_(False)
        self.base_t.weight.requires_grad_(False)

        self.down_v = nn.Linear(vision_dim, self.adapter_rank, bias=False)
        self.up_v = nn.Linear(self.adapter_rank, embed_dim, bias=False)
        self.down_t = nn.Linear(text_dim, self.adapter_rank, bias=False)
        self.up_t = nn.Linear(self.adapter_rank, embed_dim, bias=False)
        nn.init.normal_(self.down_v.weight, std=vision_dim ** -0.5)
        nn.init.normal_(self.down_t.weight, std=text_dim ** -0.5)
        nn.init.zeros_(self.up_v.weight)
        nn.init.zeros_(self.up_t.weight)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        z = self.base_v(v) + self.up_v(self.down_v(v))
        return F.normalize(z, dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        z = self.base_t(t) + self.up_t(self.down_t(t))
        return F.normalize(z, dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FedMVPProxyHead(nn.Module):
    """
    FedMVP-style prompt proxy on frozen features.

    Uses frozen random projections and trainable additive prompts in raw feature
    space to emulate prompt-tuning behavior under federated aggregation.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        init_seed: int = 42,
        prompt_scale: float = 0.05,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.prompt_scale = float(prompt_scale)

        self.base_v = nn.Linear(vision_dim, embed_dim, bias=False)
        self.base_t = nn.Linear(text_dim, embed_dim, bias=False)
        g = torch.Generator().manual_seed(init_seed)
        nn.init.normal_(self.base_v.weight, std=vision_dim ** -0.5, generator=g)
        nn.init.normal_(self.base_t.weight, std=text_dim ** -0.5, generator=g)
        self.base_v.weight.requires_grad_(False)
        self.base_t.weight.requires_grad_(False)

        self.prompt_v = nn.Parameter(torch.zeros(vision_dim))
        self.prompt_t = nn.Parameter(torch.zeros(text_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        z = self.base_v(v + self.prompt_scale * self.prompt_v)
        return F.normalize(z, dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        z = self.base_t(t + self.prompt_scale * self.prompt_t)
        return F.normalize(z, dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

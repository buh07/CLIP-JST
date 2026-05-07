from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .jl_ops import DenseSparseJL, kane_nelson_jl


class TriModalCLIPHead(nn.Module):
    def __init__(self, image_dim: int, audio_dim: int, text_dim: int, embed_dim: int):
        super().__init__()
        self.proj_i = nn.Linear(image_dim, embed_dim, bias=False)
        self.proj_a = nn.Linear(audio_dim, embed_dim, bias=False)
        self.proj_t = nn.Linear(text_dim, embed_dim, bias=False)
        nn.init.normal_(self.proj_i.weight, std=image_dim ** -0.5)
        nn.init.normal_(self.proj_a.weight, std=audio_dim ** -0.5)
        nn.init.normal_(self.proj_t.weight, std=text_dim ** -0.5)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_i(x), dim=-1)

    def encode_audio(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_a(x), dim=-1)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_t(x), dim=-1)

    def forward(self, image_x: torch.Tensor, text_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(image_x), self.encode_text(text_x)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_trainable(
        self,
        *,
        image: bool | None = None,
        audio: bool | None = None,
        text: bool | None = None,
        logit_scale: bool | None = None,
    ) -> None:
        if image is not None:
            self.proj_i.weight.requires_grad_(bool(image))
        if audio is not None:
            self.proj_a.weight.requires_grad_(bool(audio))
        if text is not None:
            self.proj_t.weight.requires_grad_(bool(text))
        if logit_scale is not None:
            self.logit_scale.requires_grad_(bool(logit_scale))

    def freeze_text_head(self) -> None:
        self.set_trainable(text=False)

    def unfreeze_text_head(self) -> None:
        self.set_trainable(text=True)

    def freeze_image_head(self) -> None:
        self.set_trainable(image=False)

    def unfreeze_image_head(self) -> None:
        self.set_trainable(image=True)

    def freeze_audio_head(self) -> None:
        self.set_trainable(audio=False)

    def unfreeze_audio_head(self) -> None:
        self.set_trainable(audio=True)


class TriModalCLIPTextLoRAHead(nn.Module):
    """
    Feature-level LoRA proxy baseline for Phase-B audio addition.

    Base image/audio/text linear projections are trained in Phase-A.
    During Phase-B, text base can be frozen while a lightweight LoRA branch
    on text remains trainable alongside the audio projection.
    """

    def __init__(
        self,
        image_dim: int,
        audio_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        lora_rank: int = 16,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        if lora_rank <= 0:
            raise ValueError("lora_rank must be > 0")
        self.proj_i = nn.Linear(image_dim, embed_dim, bias=False)
        self.proj_a = nn.Linear(audio_dim, embed_dim, bias=False)
        self.proj_t = nn.Linear(text_dim, embed_dim, bias=False)
        nn.init.normal_(self.proj_i.weight, std=image_dim ** -0.5)
        nn.init.normal_(self.proj_a.weight, std=audio_dim ** -0.5)
        nn.init.normal_(self.proj_t.weight, std=text_dim ** -0.5)

        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = self.lora_alpha / float(self.lora_rank)
        self.lora_down_t = nn.Linear(text_dim, self.lora_rank, bias=False)
        self.lora_up_t = nn.Linear(self.lora_rank, embed_dim, bias=False)
        nn.init.normal_(self.lora_down_t.weight, std=text_dim ** -0.5)
        nn.init.zeros_(self.lora_up_t.weight)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def _text_delta(self, t: torch.Tensor) -> torch.Tensor:
        return self.lora_scaling * self.lora_up_t(self.lora_down_t(t))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_i(x), dim=-1)

    def encode_audio(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_a(x), dim=-1)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_t(x) + self._text_delta(x), dim=-1)

    def forward(self, image_x: torch.Tensor, text_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(image_x), self.encode_text(text_x)

    def set_trainable(
        self,
        *,
        image: bool | None = None,
        audio: bool | None = None,
        text: bool | None = None,
        text_lora: bool | None = None,
        logit_scale: bool | None = None,
    ) -> None:
        if image is not None:
            self.proj_i.weight.requires_grad_(bool(image))
        if audio is not None:
            self.proj_a.weight.requires_grad_(bool(audio))
        if text is not None:
            self.proj_t.weight.requires_grad_(bool(text))
        if text_lora is not None:
            self.lora_down_t.weight.requires_grad_(bool(text_lora))
            self.lora_up_t.weight.requires_grad_(bool(text_lora))
        if logit_scale is not None:
            self.logit_scale.requires_grad_(bool(logit_scale))

    def freeze_text_head(self) -> None:
        self.set_trainable(text=False, text_lora=False)

    def unfreeze_text_head(self) -> None:
        self.set_trainable(text=True, text_lora=True)

    def freeze_image_head(self) -> None:
        self.set_trainable(image=False)

    def unfreeze_image_head(self) -> None:
        self.set_trainable(image=True)

    def freeze_audio_head(self) -> None:
        self.set_trainable(audio=False)

    def unfreeze_audio_head(self) -> None:
        self.set_trainable(audio=True)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TriModalRandomJLMahalanobisHead(nn.Module):
    def __init__(
        self,
        image_dim: int,
        audio_dim: int,
        text_dim: int,
        embed_dim: int,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
    ):
        super().__init__()
        self.jl_i = DenseSparseJL(image_dim, embed_dim, eps=jl_eps, seed=jl_seed)
        self.jl_a = DenseSparseJL(audio_dim, embed_dim, eps=jl_eps, seed=jl_seed + 1)
        self.jl_t = DenseSparseJL(text_dim, embed_dim, eps=jl_eps, seed=jl_seed + 2)

        self.mahal_i = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mahal_a = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mahal_t = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.mahal_i.weight)
        nn.init.eye_(self.mahal_a.weight)
        nn.init.eye_(self.mahal_t.weight)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_i(self.jl_i(x)), dim=-1)

    def encode_audio(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_a(self.jl_a(x)), dim=-1)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self.jl_t(x)), dim=-1)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TriModalOrthogonalHead(nn.Module):
    def __init__(self, image_dim: int, audio_dim: int, text_dim: int, embed_dim: int, orth_reg: float = 1e-3):
        super().__init__()
        self.embed_dim = embed_dim
        self.orth_reg = orth_reg
        self.proj_i = nn.Linear(image_dim, embed_dim, bias=False)
        self.proj_a = nn.Linear(audio_dim, embed_dim, bias=False)
        self.proj_t = nn.Linear(text_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.proj_i.weight)
        nn.init.orthogonal_(self.proj_a.weight)
        nn.init.orthogonal_(self.proj_t.weight)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_i(x), dim=-1)

    def encode_audio(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_a(x), dim=-1)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_t(x), dim=-1)

    def regularization_loss(self) -> torch.Tensor:
        if self.orth_reg <= 0:
            return torch.zeros([], device=self.proj_i.weight.device)
        eye = torch.eye(self.embed_dim, device=self.proj_i.weight.device, dtype=self.proj_i.weight.dtype)
        gi = self.proj_i.weight @ self.proj_i.weight.T
        ga = self.proj_a.weight @ self.proj_a.weight.T
        gt = self.proj_t.weight @ self.proj_t.weight.T
        return self.orth_reg * ((gi - eye).pow(2).mean() + (ga - eye).pow(2).mean() + (gt - eye).pow(2).mean())

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TriModalLearnedSparseJLHead(nn.Module):
    def __init__(
        self,
        image_dim: int,
        audio_dim: int,
        text_dim: int,
        embed_dim: int,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
        sparsity_reg: float = 1e-4,
    ):
        super().__init__()
        self.sparsity_reg = sparsity_reg

        phi_i = torch.tensor(kane_nelson_jl(image_dim, embed_dim, eps=jl_eps, seed=jl_seed).toarray(), dtype=torch.float32)
        phi_a = torch.tensor(kane_nelson_jl(audio_dim, embed_dim, eps=jl_eps, seed=jl_seed + 1).toarray(), dtype=torch.float32)
        phi_t = torch.tensor(kane_nelson_jl(text_dim, embed_dim, eps=jl_eps, seed=jl_seed + 2).toarray(), dtype=torch.float32)

        self.w_i = nn.Parameter(phi_i.clone())
        self.w_a = nn.Parameter(phi_a.clone())
        self.w_t = nn.Parameter(phi_t.clone())

        self.m_i = nn.Parameter(torch.where(phi_i.abs() > 0, torch.full_like(phi_i, 2.0), torch.full_like(phi_i, -2.0)))
        self.m_a = nn.Parameter(torch.where(phi_a.abs() > 0, torch.full_like(phi_a, 2.0), torch.full_like(phi_a, -2.0)))
        self.m_t = nn.Parameter(torch.where(phi_t.abs() > 0, torch.full_like(phi_t, 2.0), torch.full_like(phi_t, -2.0)))

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def _proj(self, x: torch.Tensor, w: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return x @ (w * torch.sigmoid(m)).T

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self._proj(x, self.w_i, self.m_i), dim=-1)

    def encode_audio(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self._proj(x, self.w_a, self.m_a), dim=-1)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self._proj(x, self.w_t, self.m_t), dim=-1)

    def regularization_loss(self) -> torch.Tensor:
        if self.sparsity_reg <= 0:
            return torch.zeros([], device=self.w_i.device)
        return self.sparsity_reg * (
            torch.sigmoid(self.m_i).mean() + torch.sigmoid(self.m_a).mean() + torch.sigmoid(self.m_t).mean()
        )

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SharedJLTriModalMahalHead(nn.Module):
    """
    Shared-JL tri-modal head for modular transitivity experiments.

    All modalities are first zero-padded into `shared_raw_dim`, then projected by
    one fixed JL map, then passed through modality-specific trainable Mahalanobis
    heads.
    """

    def __init__(
        self,
        image_dim: int,
        audio_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        shared_raw_dim: int = 768,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
        train_logit_scale: bool = True,
    ):
        super().__init__()
        if shared_raw_dim < max(image_dim, audio_dim, text_dim):
            raise ValueError("shared_raw_dim must be >= max(image_dim, audio_dim, text_dim)")

        self.image_dim = int(image_dim)
        self.audio_dim = int(audio_dim)
        self.text_dim = int(text_dim)
        self.embed_dim = int(embed_dim)
        self.shared_raw_dim = int(shared_raw_dim)
        self.shared_jl = DenseSparseJL(self.shared_raw_dim, self.embed_dim, eps=jl_eps, seed=jl_seed)

        self.mahal_i = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.mahal_a = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.mahal_t = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.eye_(self.mahal_i.weight)
        nn.init.eye_(self.mahal_a.weight)
        nn.init.eye_(self.mahal_t.weight)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / 0.07),
            requires_grad=bool(train_logit_scale),
        )

    def _pad_shared(self, x: torch.Tensor, raw_dim: int) -> torch.Tensor:
        if raw_dim == self.shared_raw_dim:
            return x
        if raw_dim > self.shared_raw_dim:
            raise ValueError("raw_dim cannot exceed shared_raw_dim")
        pad = self.shared_raw_dim - raw_dim
        return F.pad(x, (0, pad), value=0.0)

    def _shared_project(self, x: torch.Tensor, raw_dim: int) -> torch.Tensor:
        return self.shared_jl(self._pad_shared(x, raw_dim))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_i(self._shared_project(x, self.image_dim)), dim=-1)

    def encode_audio(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_a(self._shared_project(x, self.audio_dim)), dim=-1)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self._shared_project(x, self.text_dim)), dim=-1)

    def forward(self, image_x: torch.Tensor, text_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(image_x), self.encode_text(text_x)

    def set_trainable(
        self,
        *,
        image: bool | None = None,
        audio: bool | None = None,
        text: bool | None = None,
        logit_scale: bool | None = None,
    ) -> None:
        if image is not None:
            self.mahal_i.weight.requires_grad_(bool(image))
        if audio is not None:
            self.mahal_a.weight.requires_grad_(bool(audio))
        if text is not None:
            self.mahal_t.weight.requires_grad_(bool(text))
        if logit_scale is not None:
            self.logit_scale.requires_grad_(bool(logit_scale))

    def freeze_text_head(self) -> None:
        self.set_trainable(text=False)

    def unfreeze_text_head(self) -> None:
        self.set_trainable(text=True)

    def freeze_image_head(self) -> None:
        self.set_trainable(image=False)

    def unfreeze_image_head(self) -> None:
        self.set_trainable(image=True)

    def freeze_audio_head(self) -> None:
        self.set_trainable(audio=False)

    def unfreeze_audio_head(self) -> None:
        self.set_trainable(audio=True)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SeparateJLTriModalMahalHead(nn.Module):
    """
    Padded tri-modal head with modality-specific JL maps.

    This mirrors SharedJLTriModalMahalHead's padding and trainable head layout
    but uses independent fixed JL projections per modality.
    """

    def __init__(
        self,
        image_dim: int,
        audio_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        shared_raw_dim: int = 768,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
        train_logit_scale: bool = True,
    ):
        super().__init__()
        if shared_raw_dim < max(image_dim, audio_dim, text_dim):
            raise ValueError("shared_raw_dim must be >= max(image_dim, audio_dim, text_dim)")

        self.image_dim = int(image_dim)
        self.audio_dim = int(audio_dim)
        self.text_dim = int(text_dim)
        self.embed_dim = int(embed_dim)
        self.shared_raw_dim = int(shared_raw_dim)

        self.jl_i = DenseSparseJL(self.shared_raw_dim, self.embed_dim, eps=jl_eps, seed=jl_seed)
        self.jl_a = DenseSparseJL(self.shared_raw_dim, self.embed_dim, eps=jl_eps, seed=jl_seed + 1)
        self.jl_t = DenseSparseJL(self.shared_raw_dim, self.embed_dim, eps=jl_eps, seed=jl_seed + 2)

        self.mahal_i = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.mahal_a = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.mahal_t = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.eye_(self.mahal_i.weight)
        nn.init.eye_(self.mahal_a.weight)
        nn.init.eye_(self.mahal_t.weight)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / 0.07),
            requires_grad=bool(train_logit_scale),
        )

    def _pad_shared(self, x: torch.Tensor, raw_dim: int) -> torch.Tensor:
        if raw_dim == self.shared_raw_dim:
            return x
        if raw_dim > self.shared_raw_dim:
            raise ValueError("raw_dim cannot exceed shared_raw_dim")
        pad = self.shared_raw_dim - raw_dim
        return F.pad(x, (0, pad), value=0.0)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_i(self.jl_i(self._pad_shared(x, self.image_dim))), dim=-1)

    def encode_audio(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_a(self.jl_a(self._pad_shared(x, self.audio_dim))), dim=-1)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self.jl_t(self._pad_shared(x, self.text_dim))), dim=-1)

    def forward(self, image_x: torch.Tensor, text_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(image_x), self.encode_text(text_x)

    def set_trainable(
        self,
        *,
        image: bool | None = None,
        audio: bool | None = None,
        text: bool | None = None,
        logit_scale: bool | None = None,
    ) -> None:
        if image is not None:
            self.mahal_i.weight.requires_grad_(bool(image))
        if audio is not None:
            self.mahal_a.weight.requires_grad_(bool(audio))
        if text is not None:
            self.mahal_t.weight.requires_grad_(bool(text))
        if logit_scale is not None:
            self.logit_scale.requires_grad_(bool(logit_scale))

    def freeze_text_head(self) -> None:
        self.set_trainable(text=False)

    def unfreeze_text_head(self) -> None:
        self.set_trainable(text=True)

    def freeze_image_head(self) -> None:
        self.set_trainable(image=False)

    def unfreeze_image_head(self) -> None:
        self.set_trainable(image=True)

    def freeze_audio_head(self) -> None:
        self.set_trainable(audio=False)

    def unfreeze_audio_head(self) -> None:
        self.set_trainable(audio=True)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HybridITJLTriModalMahalHead(nn.Module):
    """
    Hybrid JL head: image+text share JL; audio uses independent JL.
    """

    def __init__(
        self,
        image_dim: int,
        audio_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        shared_raw_dim: int = 768,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
        train_logit_scale: bool = True,
    ):
        super().__init__()
        if shared_raw_dim < max(image_dim, audio_dim, text_dim):
            raise ValueError("shared_raw_dim must be >= max(image_dim, audio_dim, text_dim)")

        self.image_dim = int(image_dim)
        self.audio_dim = int(audio_dim)
        self.text_dim = int(text_dim)
        self.embed_dim = int(embed_dim)
        self.shared_raw_dim = int(shared_raw_dim)

        self.jl_it = DenseSparseJL(self.shared_raw_dim, self.embed_dim, eps=jl_eps, seed=jl_seed)
        self.jl_a = DenseSparseJL(self.shared_raw_dim, self.embed_dim, eps=jl_eps, seed=jl_seed + 1)

        self.mahal_i = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.mahal_a = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.mahal_t = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.eye_(self.mahal_i.weight)
        nn.init.eye_(self.mahal_a.weight)
        nn.init.eye_(self.mahal_t.weight)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / 0.07),
            requires_grad=bool(train_logit_scale),
        )

    def _pad_shared(self, x: torch.Tensor, raw_dim: int) -> torch.Tensor:
        if raw_dim == self.shared_raw_dim:
            return x
        if raw_dim > self.shared_raw_dim:
            raise ValueError("raw_dim cannot exceed shared_raw_dim")
        pad = self.shared_raw_dim - raw_dim
        return F.pad(x, (0, pad), value=0.0)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_i(self.jl_it(self._pad_shared(x, self.image_dim))), dim=-1)

    def encode_audio(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_a(self.jl_a(self._pad_shared(x, self.audio_dim))), dim=-1)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self.jl_it(self._pad_shared(x, self.text_dim))), dim=-1)

    def forward(self, image_x: torch.Tensor, text_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(image_x), self.encode_text(text_x)

    def set_trainable(
        self,
        *,
        image: bool | None = None,
        audio: bool | None = None,
        text: bool | None = None,
        logit_scale: bool | None = None,
    ) -> None:
        if image is not None:
            self.mahal_i.weight.requires_grad_(bool(image))
        if audio is not None:
            self.mahal_a.weight.requires_grad_(bool(audio))
        if text is not None:
            self.mahal_t.weight.requires_grad_(bool(text))
        if logit_scale is not None:
            self.logit_scale.requires_grad_(bool(logit_scale))

    def freeze_text_head(self) -> None:
        self.set_trainable(text=False)

    def unfreeze_text_head(self) -> None:
        self.set_trainable(text=True)

    def freeze_image_head(self) -> None:
        self.set_trainable(image=False)

    def unfreeze_image_head(self) -> None:
        self.set_trainable(image=True)

    def freeze_audio_head(self) -> None:
        self.set_trainable(audio=False)

    def unfreeze_audio_head(self) -> None:
        self.set_trainable(audio=True)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HybridATJLTriModalMahalHead(nn.Module):
    """
    Hybrid JL head: audio+text share JL; image uses independent JL.
    """

    def __init__(
        self,
        image_dim: int,
        audio_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        shared_raw_dim: int = 768,
        jl_eps: float = 0.1,
        jl_seed: int = 42,
        train_logit_scale: bool = True,
    ):
        super().__init__()
        if shared_raw_dim < max(image_dim, audio_dim, text_dim):
            raise ValueError("shared_raw_dim must be >= max(image_dim, audio_dim, text_dim)")

        self.image_dim = int(image_dim)
        self.audio_dim = int(audio_dim)
        self.text_dim = int(text_dim)
        self.embed_dim = int(embed_dim)
        self.shared_raw_dim = int(shared_raw_dim)

        self.jl_at = DenseSparseJL(self.shared_raw_dim, self.embed_dim, eps=jl_eps, seed=jl_seed)
        self.jl_i = DenseSparseJL(self.shared_raw_dim, self.embed_dim, eps=jl_eps, seed=jl_seed + 1)

        self.mahal_i = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.mahal_a = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.mahal_t = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.eye_(self.mahal_i.weight)
        nn.init.eye_(self.mahal_a.weight)
        nn.init.eye_(self.mahal_t.weight)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / 0.07),
            requires_grad=bool(train_logit_scale),
        )

    def _pad_shared(self, x: torch.Tensor, raw_dim: int) -> torch.Tensor:
        if raw_dim == self.shared_raw_dim:
            return x
        if raw_dim > self.shared_raw_dim:
            raise ValueError("raw_dim cannot exceed shared_raw_dim")
        pad = self.shared_raw_dim - raw_dim
        return F.pad(x, (0, pad), value=0.0)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_i(self.jl_i(self._pad_shared(x, self.image_dim))), dim=-1)

    def encode_audio(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_a(self.jl_at(self._pad_shared(x, self.audio_dim))), dim=-1)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mahal_t(self.jl_at(self._pad_shared(x, self.text_dim))), dim=-1)

    def forward(self, image_x: torch.Tensor, text_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(image_x), self.encode_text(text_x)

    def set_trainable(
        self,
        *,
        image: bool | None = None,
        audio: bool | None = None,
        text: bool | None = None,
        logit_scale: bool | None = None,
    ) -> None:
        if image is not None:
            self.mahal_i.weight.requires_grad_(bool(image))
        if audio is not None:
            self.mahal_a.weight.requires_grad_(bool(audio))
        if text is not None:
            self.mahal_t.weight.requires_grad_(bool(text))
        if logit_scale is not None:
            self.logit_scale.requires_grad_(bool(logit_scale))

    def freeze_text_head(self) -> None:
        self.set_trainable(text=False)

    def unfreeze_text_head(self) -> None:
        self.set_trainable(text=True)

    def freeze_image_head(self) -> None:
        self.set_trainable(image=False)

    def unfreeze_image_head(self) -> None:
        self.set_trainable(image=True)

    def freeze_audio_head(self) -> None:
        self.set_trainable(audio=False)

    def unfreeze_audio_head(self) -> None:
        self.set_trainable(audio=True)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

from __future__ import annotations

import argparse
import fcntl
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data import (
    AudioCapsAVCache,
    CC3MCache,
    KarpathyCache,
    build_cc3m_adapter,
    extract_audiocaps_av_cache,
    extract_wavcaps_audio_text_cache,
)
from ..data.datasets import PairedFeatureDataset
from ..eval.diagnostics import centroid_distance_matrix, pair_diagnostics
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report
from ..models import (
    HybridATJLTriModalMahalHead,
    HybridITJLTriModalMahalHead,
    SeparateJLTriModalMahalHead,
    SharedJLTriModalMahalHead,
    TriModalCLIPHead,
    TriModalCLIPTextLoRAHead,
)
from ..training import train_bimodal
from ..training.losses import infonce_loss


REPORT_METRICS = ["combined_avg_R", "coco_avg_R", "av_it_avg_R", "av_at_avg_R", "av_ia_avg_R"]


def _load_best(model, ckpt_dir: Path, device: str):
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))
    return model.to(device)


@torch.no_grad()
def _encode_batches(fn, x: torch.Tensor, *, device: str, batch_size: int) -> torch.Tensor:
    outs = []
    for i in range(0, len(x), batch_size):
        outs.append(fn(x[i:i + batch_size].to(device)).cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def _eval_coco_it(
    model,
    cache: KarpathyCache,
    split_name: str,
    *,
    device: str,
    batch_size: int,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    img, txt, gt_i2t, gt_t2i = cache.eval_tensors(split_name)
    zi = _encode_batches(model.encode_image, img, device=device, batch_size=batch_size)
    zt = _encode_batches(model.encode_text, txt, device=device, batch_size=batch_size)
    met = recall_at_k(zi, zt, gt_i2t=gt_i2t, gt_t2i=gt_t2i)
    return met, zi, zt


@torch.no_grad()
def _eval_av_all(model, cache: AudioCapsAVCache, split_name: str, *, device: str, batch_size: int) -> dict:
    img, aud, txt = cache.eval_tensors(split_name)
    zi = _encode_batches(model.encode_image, img, device=device, batch_size=batch_size)
    za = _encode_batches(model.encode_audio, aud, device=device, batch_size=batch_size)
    zt = _encode_batches(model.encode_text, txt, device=device, batch_size=batch_size)

    m_it = recall_at_k(zi, zt)
    m_at = recall_at_k(za, zt)
    m_ia = recall_at_k(zi, za)

    diag = {}
    diag.update(pair_diagnostics(zi, zt, prefix="it"))
    diag.update(pair_diagnostics(za, zt, prefix="at"))
    diag.update(pair_diagnostics(zi, za, prefix="ia"))
    diag["centroid_distance_matrix"] = centroid_distance_matrix({"image": zi, "audio": za, "text": zt})

    return {
        "image_text": m_it,
        "audio_text": m_at,
        "image_audio": m_ia,
        "diagnostics": diag,
    }


@torch.no_grad()
def _eval_raw_cosine_all(
    coco: KarpathyCache,
    av: AudioCapsAVCache,
    *,
    shared_raw_dim: int,
    coco_split: str,
    av_split: str,
) -> dict:
    def _pad(x: torch.Tensor, d: int) -> torch.Tensor:
        if d == shared_raw_dim:
            return x
        if d > shared_raw_dim:
            raise ValueError(f"raw dim {d} exceeds shared_raw_dim {shared_raw_dim}")
        return F.pad(x, (0, shared_raw_dim - d), value=0.0)

    img_coco, txt_coco, gt_i2t, gt_t2i = coco.eval_tensors(coco_split)
    zic = F.normalize(_pad(img_coco, int(img_coco.shape[1])), dim=-1)
    ztc = F.normalize(_pad(txt_coco, int(txt_coco.shape[1])), dim=-1)
    coco_it = recall_at_k(zic, ztc, gt_i2t=gt_i2t, gt_t2i=gt_t2i)

    img_av, aud_av, txt_av = av.eval_tensors(av_split)
    zia = F.normalize(_pad(img_av, int(img_av.shape[1])), dim=-1)
    zaa = F.normalize(_pad(aud_av, int(aud_av.shape[1])), dim=-1)
    zta = F.normalize(_pad(txt_av, int(txt_av.shape[1])), dim=-1)

    av_it = recall_at_k(zia, zta)
    av_at = recall_at_k(zaa, zta)
    av_ia = recall_at_k(zia, zaa)

    diag = {}
    diag.update(pair_diagnostics(zia, zta, prefix="it"))
    diag.update(pair_diagnostics(zaa, zta, prefix="at"))
    diag.update(pair_diagnostics(zia, zaa, prefix="ia"))
    diag["centroid_distance_matrix"] = centroid_distance_matrix({"image": zia, "audio": zaa, "text": zta})

    return {
        "coco_image_text": coco_it,
        "av_image_text": av_it,
        "av_audio_text": av_at,
        "av_image_audio": av_ia,
        "diagnostics": {"coco_it": pair_diagnostics(zic, ztc, prefix="coco_it"), "av": diag},
        "coco_avg_R": float(coco_it["avg_R"]),
        "av_it_avg_R": float(av_it["avg_R"]),
        "av_at_avg_R": float(av_at["avg_R"]),
        "av_ia_avg_R": float(av_ia["avg_R"]),
    }


def _train_audio_text_phase(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: str,
    ckpt_dir: Path,
    patience: int = 8,
    grad_clip: float = 1.0,
) -> dict:
    model = model.to(device)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if epochs == 0:
        torch.save(model.state_dict(), ckpt_dir / "best.pt")
        return {"best_val_r10": 0.0, "best_epoch": 0, "history": {"train_loss": [], "val_r10": []}}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_val = -1.0
    best_epoch = -1
    no_imp = 0
    hist = {"train_loss": [], "val_r10": []}

    @torch.no_grad()
    def _val_r10() -> float:
        model.eval()
        zs, zt = [], []
        for aud, txt in val_loader:
            zs.append(model.encode_audio(aud.to(device)).cpu())
            zt.append(model.encode_text(txt.to(device)).cpu())
        m = recall_at_k(torch.cat(zs, dim=0), torch.cat(zt, dim=0))
        return float(m["i2t_R@10"])

    for ep in range(epochs):
        model.train()
        running = 0.0
        for aud, txt in train_loader:
            aud = aud.to(device)
            txt = txt.to(device)
            scale = model.logit_scale.exp().clamp(max=100.0) if hasattr(model, "logit_scale") else (1.0 / 0.07)
            za = model.encode_audio(aud)
            zt = model.encode_text(txt)
            loss = infonce_loss(za, zt, scale)
            if hasattr(model, "regularization_loss"):
                reg = model.regularization_loss()
                if reg is not None:
                    loss = loss + reg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            running += float(loss.item())
        sch.step()

        avg_loss = running / max(1, len(train_loader))
        vr10 = _val_r10()
        hist["train_loss"].append(avg_loss)
        hist["val_r10"].append(vr10)
        print(f"AT Epoch {ep+1:03d}/{epochs} loss={avg_loss:.4f} valA2T_R10={vr10:.4f}")

        if vr10 > best_val:
            best_val = vr10
            best_epoch = ep + 1
            no_imp = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"AT early stop at epoch {ep+1}")
                break

    return {"best_val_r10": best_val, "best_epoch": best_epoch, "history": hist}


def _set_trainable_phase(model, method: str, phase: str, phase_order: str) -> None:
    """
    phase in {"phase_a", "phase_b"}. phase_order in {"it_then_at", "at_then_it"}
    """
    if phase_order == "it_then_at":
        if phase == "phase_a":
            # image-text phase first
            if hasattr(model, "set_trainable"):
                try:
                    model.set_trainable(image=True, audio=False, text=True, text_lora=False, logit_scale=True)
                except TypeError:
                    model.set_trainable(image=True, audio=False, text=True, logit_scale=True)
        else:
            # audio-text phase second
            if method == "audio_text_lora_proxy":
                model.set_trainable(image=False, audio=True, text=False, text_lora=True, logit_scale=False)
            else:
                if hasattr(model, "set_trainable"):
                    try:
                        model.set_trainable(image=False, audio=True, text=False, text_lora=False, logit_scale=False)
                    except TypeError:
                        model.set_trainable(image=False, audio=True, text=False, logit_scale=False)
    else:
        # at_then_it
        if phase == "phase_a":
            # audio-text first
            if method == "audio_text_lora_proxy":
                model.set_trainable(image=False, audio=True, text=True, text_lora=False, logit_scale=True)
            else:
                if hasattr(model, "set_trainable"):
                    try:
                        model.set_trainable(image=False, audio=True, text=True, text_lora=False, logit_scale=True)
                    except TypeError:
                        model.set_trainable(image=False, audio=True, text=True, logit_scale=True)
        else:
            # image-text second; freeze text/audio to keep symmetric bridge constraint.
            if method == "audio_text_lora_proxy":
                model.set_trainable(image=True, audio=False, text=False, text_lora=False, logit_scale=False)
            else:
                if hasattr(model, "set_trainable"):
                    try:
                        model.set_trainable(image=True, audio=False, text=False, text_lora=False, logit_scale=False)
                    except TypeError:
                        model.set_trainable(image=True, audio=False, text=False, logit_scale=False)


def _build_model(method: str, m: int, cfg: dict):
    common = {
        "image_dim": int(cfg["vision_dim"]),
        "audio_dim": int(cfg["audio_dim"]),
        "text_dim": int(cfg["text_dim"]),
        "embed_dim": int(m),
    }
    jl_kw = {
        "shared_raw_dim": int(cfg.get("shared_raw_dim", 768)),
        "jl_eps": float(cfg.get("jl_eps", 0.1)),
        "jl_seed": int(cfg.get("jl_seed", 42)),
    }

    if method in {"modular_shared_jl", "joint_shared_jl"}:
        return SharedJLTriModalMahalHead(**common, **jl_kw)
    if method == "modular_separate_jl":
        return SeparateJLTriModalMahalHead(**common, **jl_kw)
    if method == "modular_hybrid_it_jl":
        return HybridITJLTriModalMahalHead(**common, **jl_kw)
    if method == "modular_hybrid_at_jl":
        return HybridATJLTriModalMahalHead(**common, **jl_kw)
    if method in {"audio_linear_probe", "joint_clip_head", "joint_clap_head"}:
        return TriModalCLIPHead(**common)
    if method == "audio_text_lora_proxy":
        return TriModalCLIPTextLoRAHead(
            **common,
            lora_rank=int(cfg.get("text_lora_rank", 16)),
            lora_alpha=float(cfg.get("text_lora_alpha", 1.0)),
        )
    if method == "raw_cosine_baseline":
        # Eval-only path handled in _run_one.
        return None
    raise ValueError(f"Unsupported method: {method}")


def _build_loaders(cfg: dict, coco: KarpathyCache, av: AudioCapsAVCache, output_root: Path):
    loader_kw = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("num_workers", 4)),
        "pin_memory": True,
    }

    phase_a_source = str(cfg.get("phase_a_source", "coco"))
    if phase_a_source == "coco":
        train_indices = coco.split_indices(cfg.get("coco_train_split", "train_restval"))
        max_imgs = cfg.get("coco_max_train_images")
        if max_imgs is not None:
            g = torch.Generator()
            g.manual_seed(int(cfg.get("coco_subsample_seed", 2026)))
            perm = torch.randperm(len(train_indices), generator=g)
            train_indices = [train_indices[i] for i in perm[:int(max_imgs)].tolist()]
        from ..data.datasets import ImageCaptionTrainDataset
        it_train = ImageCaptionTrainDataset(
            img_feats=coco.image_feats,
            txt_feats=coco.text_feats,
            image_indices=train_indices,
            n_captions=coco.n_captions,
        )
        it_train.train(True)
        it_val = coco.make_train_dataset(cfg.get("coco_val_split", "val"), training=False)
    elif phase_a_source == "cc3m":
        cc3m_root = Path(cfg.get("cc3m_cache_root", "/jumbo/lisp/f004ndc/CLIP JST/data/cache/cc3m")).resolve()
        cc3m_adapter_dir = Path(cfg.get("cc3m_adapter_root", output_root / "caches" / "cc3m_adapter")).resolve()
        paths = build_cc3m_adapter(
            cc3m_cache_root=cc3m_root,
            out_dir=cc3m_adapter_dir,
            image_cache_file=str(cfg.get("cc3m_image_cache_file", "image_feats_openai_clip-vit-base-patch32_raw.pt")),
            text_cache_file=str(cfg.get("cc3m_text_cache_file", "text_feats_openai_clip-vit-base-patch32_raw.pt")),
            split_seed=int(cfg.get("cc3m_split_seed", 2026)),
            train_frac=float(cfg.get("cc3m_train_frac", 0.8)),
            val_frac=float(cfg.get("cc3m_val_frac", 0.1)),
        )
        cc3m = CC3MCache.from_paths(paths["image"], paths["text"], paths["meta"])
        it_train = cc3m.make_train_dataset("train")
        it_val = cc3m.make_train_dataset("val")
    else:
        raise ValueError(f"Unsupported phase_a_source: {phase_a_source}")

    phase_b_source = str(cfg.get("phase_b_source", "audiocaps"))
    if phase_b_source == "audiocaps":
        at_train_idx = av.split_indices(cfg.get("av_train_split", "train"))
        at_val_idx = av.split_indices(cfg.get("av_val_split", "validation"))
        at_train_audio = av.audio_feats[at_train_idx]
        at_train_text = av.text_feats[at_train_idx]
        if bool(cfg.get("phase_b_shuffle_captions", False)):
            g = torch.Generator()
            g.manual_seed(int(cfg.get("phase_b_shuffle_seed", 1337)))
            perm = torch.randperm(len(at_train_text), generator=g)
            at_train_text = at_train_text[perm]
        at_train = PairedFeatureDataset(at_train_audio, at_train_text)
        at_val = PairedFeatureDataset(av.audio_feats[at_val_idx], av.text_feats[at_val_idx])
    elif phase_b_source == "wavcaps":
        wav_root = Path(cfg.get("wavcaps_cache_root", output_root / "caches" / "wavcaps")).resolve()
        wav_paths = extract_wavcaps_audio_text_cache(
            out_dir=wav_root,
            dataset_name=str(cfg.get("wavcaps_source", "humanify/AS-WavCaps")),
            clap_model_name=str(cfg["clap_model"]),
            clip_backbone_name=str(cfg["clip_backbone"]),
            target_sampling_rate=int(cfg.get("audiocaps_target_sr", 48_000)),
            max_examples=int(cfg.get("wavcaps_target_examples", 200_000)),
            sampling_policy=str(cfg.get("wavcaps_sampling_policy", "stratified")),
            device=str(cfg["device"]),
            audio_batch_size=int(cfg.get("wavcaps_audio_batch_size", 64)),
            text_batch_size=int(cfg.get("wavcaps_text_batch_size", 256)),
            split_name=str(cfg.get("wavcaps_split", "train")),
            stream=bool(cfg.get("wavcaps_stream", True)),
        )
        wav_audio = torch.load(wav_paths["audio"], map_location="cpu", weights_only=True)
        wav_text = torch.load(wav_paths["text"], map_location="cpu", weights_only=True)
        wav_meta = load_json(wav_paths["meta"])
        if len(wav_audio) != len(wav_text):
            raise RuntimeError("wavcaps feature length mismatch")

        include_sources = [str(x) for x in cfg.get("wavcaps_source_filter_include", [])]
        exclude_sources = [str(x) for x in cfg.get("wavcaps_source_filter_exclude", [])]
        if include_sources or exclude_sources:
            sample_sources = wav_meta.get("sample_sources") if isinstance(wav_meta, dict) else None
            if not isinstance(sample_sources, list) or len(sample_sources) != len(wav_audio):
                raise RuntimeError(
                    "wavcaps source filtering requested but sample_sources metadata is unavailable or mismatched"
                )
            include_set = set(include_sources)
            exclude_set = set(exclude_sources)
            keep = []
            for i, src in enumerate(sample_sources):
                s = str(src)
                if include_set and s not in include_set:
                    continue
                if exclude_set and s in exclude_set:
                    continue
                keep.append(i)
            if not keep:
                raise RuntimeError("wavcaps source filtering kept zero samples")
            idx_t = torch.tensor(keep, dtype=torch.long)
            wav_audio = wav_audio.index_select(0, idx_t)
            wav_text = wav_text.index_select(0, idx_t)
            print(
                f"WavCaps source filtering: kept {len(keep)}/{len(sample_sources)} samples; "
                f"include={include_sources or 'ALL'} exclude={exclude_sources or 'NONE'}"
            )

        n = len(wav_audio)
        g = torch.Generator()
        g.manual_seed(int(cfg.get("wavcaps_split_seed", 2026)))
        perm = torch.randperm(n, generator=g)
        wav_audio = wav_audio[perm]
        wav_text = wav_text[perm]
        n_val_frac = max(1, int(round(n * float(cfg.get("wavcaps_val_frac", 0.1)))))
        max_val = int(cfg.get("wavcaps_val_max_gallery", 500))
        n_val = min(n_val_frac, max_val, n - 1)
        train_audio = wav_audio[:-n_val]
        train_text = wav_text[:-n_val]
        val_audio = wav_audio[-n_val:]
        val_text = wav_text[-n_val:]
        at_train = PairedFeatureDataset(train_audio, train_text)
        at_val = PairedFeatureDataset(val_audio, val_text)
    else:
        raise ValueError(f"Unsupported phase_b_source: {phase_b_source}")

    return (
        DataLoader(it_train, shuffle=True, **loader_kw),
        DataLoader(it_val, shuffle=False, **loader_kw),
        DataLoader(at_train, shuffle=True, **loader_kw),
        DataLoader(at_val, shuffle=False, **loader_kw),
    )


def _train_phase_it(model, train_loader, val_loader, cfg: dict, ckpt_dir: Path):
    return train_bimodal(
        model,
        train_loader,
        val_loader,
        epochs=int(cfg["epochs_phase_a"]),
        lr=float(cfg["lr_phase_a"]),
        device=str(cfg["device"]),
        ckpt_dir=ckpt_dir,
        patience=int(cfg.get("patience_phase_a", 8)),
        warmup_epochs=int(cfg.get("warmup_phase_a", 0)),
    )


def _train_phase_at(model, train_loader, val_loader, cfg: dict, ckpt_dir: Path):
    return _train_audio_text_phase(
        model,
        train_loader,
        val_loader,
        epochs=int(cfg["epochs_phase_b"]),
        lr=float(cfg["lr_phase_b"]),
        device=str(cfg["device"]),
        ckpt_dir=ckpt_dir,
        patience=int(cfg.get("patience_phase_b", 8)),
    )


def _run_one(
    *,
    model,
    method: str,
    seed: int,
    m: int,
    cfg: dict,
    stage_root: Path,
    coco: KarpathyCache,
    av: AudioCapsAVCache,
    it_train_loader: DataLoader,
    it_val_loader: DataLoader,
    at_train_loader: DataLoader,
    at_val_loader: DataLoader,
    phase_order: str,
    reuse_phase_a_from: Path | None,
) -> dict[str, Any]:
    seed_dir = stage_root / f"m{m}" / method / f"seed{seed}"
    eval_p = seed_dir / "eval.json"
    if eval_p.exists():
        return load_json(eval_p)

    if method == "raw_cosine_baseline":
        rec = {
            "seed": seed,
            "method": "raw_cosine_baseline",
            "embed_dim": m,
            "phase_order": phase_order,
            "phase_a_source": cfg.get("phase_a_source", "coco"),
            "phase_b_source": cfg.get("phase_b_source", "audiocaps"),
            **_eval_raw_cosine_all(
                coco,
                av,
                shared_raw_dim=int(cfg.get("shared_raw_dim", 768)),
                coco_split=cfg.get("coco_test_split", "test"),
                av_split=cfg.get("av_test_split", "test"),
            ),
        }
        rec["combined_avg_R"] = float((rec["coco_avg_R"] + rec["av_ia_avg_R"] + rec["av_at_avg_R"]) / 3.0)
        save_json(rec, eval_p)
        return rec

    if phase_order not in {"it_then_at", "at_then_it"}:
        raise ValueError(f"Unsupported phase_order: {phase_order}")

    if phase_order == "it_then_at":
        phase_a_dir = seed_dir / "phase_a"
        if reuse_phase_a_from is not None:
            src = reuse_phase_a_from / f"m{m}" / method / f"seed{seed}" / "phase_a" / "best.pt"
            if not src.exists():
                raise FileNotFoundError(f"Missing reused phase-a checkpoint: {src}")
            model.load_state_dict(torch.load(src, map_location=cfg["device"], weights_only=True), strict=True)
            phase_a_dir.mkdir(parents=True, exist_ok=True)
            if not (phase_a_dir / "best.pt").exists():
                torch.save(model.state_dict(), phase_a_dir / "best.pt")
        else:
            _set_trainable_phase(model, method, "phase_a", phase_order)
            _train_phase_it(model, it_train_loader, it_val_loader, cfg, phase_a_dir)
            model = _load_best(model, phase_a_dir, cfg["device"])

        _set_trainable_phase(model, method, "phase_b", phase_order)
        phase_b_dir = seed_dir / "phase_b"
        _train_phase_at(model, at_train_loader, at_val_loader, cfg, phase_b_dir)
        model = _load_best(model, phase_b_dir, cfg["device"])
    else:
        phase_a_dir = seed_dir / "phase_a"
        _set_trainable_phase(model, method, "phase_a", phase_order)
        _train_phase_at(model, at_train_loader, at_val_loader, cfg, phase_a_dir)
        model = _load_best(model, phase_a_dir, cfg["device"])

        _set_trainable_phase(model, method, "phase_b", phase_order)
        phase_b_dir = seed_dir / "phase_b"
        _train_phase_it(model, it_train_loader, it_val_loader, cfg, phase_b_dir)
        model = _load_best(model, phase_b_dir, cfg["device"])

    coco_it, zi_coco, zt_coco = _eval_coco_it(
        model,
        coco,
        cfg.get("coco_test_split", "test"),
        device=cfg["device"],
        batch_size=int(cfg["eval_batch_size"]),
    )
    av_eval = _eval_av_all(
        model,
        av,
        cfg.get("av_test_split", "test"),
        device=cfg["device"],
        batch_size=int(cfg["eval_batch_size"]),
    )

    rec = {
        "seed": seed,
        "method": method,
        "embed_dim": m,
        "phase_order": phase_order,
        "phase_a_source": cfg.get("phase_a_source", "coco"),
        "phase_b_source": cfg.get("phase_b_source", "audiocaps"),
        "coco_image_text": coco_it,
        "av_image_text": av_eval["image_text"],
        "av_audio_text": av_eval["audio_text"],
        "av_image_audio": av_eval["image_audio"],
        "diagnostics": {
            "coco_it": pair_diagnostics(zi_coco, zt_coco, prefix="coco_it"),
            "av": av_eval["diagnostics"],
        },
        "coco_avg_R": float(coco_it["avg_R"]),
        "av_it_avg_R": float(av_eval["image_text"]["avg_R"]),
        "av_at_avg_R": float(av_eval["audio_text"]["avg_R"]),
        "av_ia_avg_R": float(av_eval["image_audio"]["avg_R"]),
    }
    rec["combined_avg_R"] = float((rec["coco_avg_R"] + rec["av_ia_avg_R"] + rec["av_at_avg_R"]) / 3.0)
    save_json(rec, eval_p)
    return rec


def _merge_seed_rows(dst_rows: list[dict[str, Any]], src_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed = {int(r["seed"]): r for r in dst_rows if "seed" in r}
    for r in src_rows:
        by_seed[int(r["seed"])] = r
    return [by_seed[s] for s in sorted(by_seed)]


def _merge_results_existing(
    *,
    stage_name: str,
    out_path: Path,
    incoming: dict[str, Any],
    baseline_method: str,
) -> dict[str, Any]:
    """
    Merge current shard results with an existing stage result file.
    This makes stage29/30/31/32 shard-safe when multiple workers write into
    the same stage directory.
    """
    if not out_path.exists():
        return incoming
    try:
        existing = load_json(out_path)
    except Exception:
        return incoming
    if existing.get("stage") != stage_name:
        return incoming

    merged = dict(existing)
    merged["methods_requested"] = list(
        dict.fromkeys(list(existing.get("methods_requested", [])) + list(incoming.get("methods_requested", [])))
    )
    merged["phase_a_source"] = incoming.get("phase_a_source", existing.get("phase_a_source"))
    merged["phase_b_source"] = incoming.get("phase_b_source", existing.get("phase_b_source"))
    merged["phase_order"] = incoming.get("phase_order", existing.get("phase_order"))

    raw_old = existing.get("raw", {}) if isinstance(existing.get("raw", {}), dict) else {}
    raw_new = incoming.get("raw", {}) if isinstance(incoming.get("raw", {}), dict) else {}
    raw_merged: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for m_key in sorted(set(raw_old.keys()) | set(raw_new.keys()), key=lambda k: int(str(k)[1:])):
        raw_merged[m_key] = {}
        old_methods = raw_old.get(m_key, {})
        new_methods = raw_new.get(m_key, {})
        for method in sorted(set(old_methods.keys()) | set(new_methods.keys())):
            cur = old_methods.get(method, [])
            nxt = new_methods.get(method, [])
            raw_merged[m_key][method] = _merge_seed_rows(cur, nxt)
    merged["raw"] = raw_merged

    stats: dict[str, Any] = {}
    for m_key, methods in raw_merged.items():
        if not methods:
            continue
        b = baseline_method if baseline_method in methods else next(iter(methods))
        stats[m_key] = build_metric_report(methods, metrics=REPORT_METRICS, baseline_method=b)
    merged["stats"] = stats
    return merged


def _save_results_shard_safe(
    *,
    stage_name: str,
    out_path: Path,
    incoming: dict[str, Any],
    baseline_method: str,
) -> None:
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        merged_results = _merge_results_existing(
            stage_name=stage_name,
            out_path=out_path,
            incoming=incoming,
            baseline_method=baseline_method,
        )
        save_json(merged_results, out_path)
        fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def run_core(cfg: dict, *, stage_name: str) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    coco_cache_dir = Path(cfg["cache_root"]) / "coco"
    coco = KarpathyCache.from_paths(
        coco_cache_dir / cfg["image_cache_file"],
        coco_cache_dir / cfg["text_cache_file"],
        coco_cache_dir / "metadata.json",
    )

    av_dir = Path(cfg["av_cache_root"]).resolve()
    if bool(cfg.get("build_av_cache_if_missing", False)):
        img_p = av_dir / "image_feats_clip_raw.pt"
        aud_p = av_dir / "audio_feats_clap_raw.pt"
        txt_p = av_dir / "text_feats_clip_raw.pt"
        meta_p = av_dir / "metadata.json"
        if not (img_p.exists() and aud_p.exists() and txt_p.exists() and meta_p.exists()):
            extract_audiocaps_av_cache(
                out_dir=av_dir,
                dataset_name=str(cfg["audiocaps_dataset"]),
                clap_model_name=str(cfg["clap_model"]),
                clip_backbone_name=str(cfg["clip_backbone"]),
                device=str(cfg["device"]),
                audio_batch_size=int(cfg.get("av_audio_batch_size", 64)),
                image_batch_size=int(cfg.get("av_image_batch_size", 128)),
                text_batch_size=int(cfg.get("av_text_batch_size", 256)),
                target_sampling_rate=int(cfg.get("audiocaps_target_sr", 48_000)),
                max_examples_per_split=cfg.get("audiocaps_max_examples_per_split"),
                thumbnail_timeout_sec=float(cfg.get("thumbnail_timeout_sec", 10.0)),
                thumbnail_retries=int(cfg.get("thumbnail_retries", 2)),
                thumbnail_backoff_sec=float(cfg.get("thumbnail_backoff_sec", 1.0)),
            )

    av = AudioCapsAVCache.from_paths(
        av_dir / "image_feats_clip_raw.pt",
        av_dir / "audio_feats_clap_raw.pt",
        av_dir / "text_feats_clip_raw.pt",
        av_dir / "metadata.json",
    )

    methods = list(cfg.get("methods", []))
    seeds = [int(s) for s in cfg["seeds"]]
    embed_dims = [int(m) for m in cfg["embed_dims"]]
    phase_order = str(cfg.get("phase_order", "it_then_at"))

    # Optional phase-a checkpoint reuse path (stage31).
    reuse_phase_a_from = cfg.get("reuse_phase_a_from")
    reuse_phase_a_path = Path(reuse_phase_a_from).resolve() if reuse_phase_a_from else None

    try:
        it_train_loader, it_val_loader, at_train_loader, at_val_loader = _build_loaders(cfg, coco, av, output_root)
    except Exception as e:
        fail = {
            "stage": stage_name,
            "status": "failed_loader",
            "error_type": type(e).__name__,
            "error": str(e),
            "phase_a_source": cfg.get("phase_a_source", "coco"),
            "phase_b_source": cfg.get("phase_b_source", "audiocaps"),
        }
        save_json(fail, stage_root / f"{stage_name}_failure.json")
        mark_done(markers / f"{stage_name}.done.json", fail)
        print(f"{stage_name} failed during loader construction: {type(e).__name__}: {e}")
        return

    results = {
        "stage": stage_name,
        "methods_requested": methods,
        "raw": {},
        "stats": {},
        "phase_a_source": cfg.get("phase_a_source", "coco"),
        "phase_b_source": cfg.get("phase_b_source", "audiocaps"),
        "phase_order": phase_order,
    }

    baseline_method = str(cfg.get("baseline_method", methods[0] if methods else "modular_shared_jl"))

    for m in embed_dims:
        m_key = f"m{m}"
        results["raw"][m_key] = {method: [] for method in methods}
        for seed in seeds:
            set_seed(seed)
            for method in methods:
                model = _build_model(method, m, cfg)
                rec = _run_one(
                    model=model,
                    method=method,
                    seed=seed,
                    m=m,
                    cfg=cfg,
                    stage_root=stage_root,
                    coco=coco,
                    av=av,
                    it_train_loader=it_train_loader,
                    it_val_loader=it_val_loader,
                    at_train_loader=at_train_loader,
                    at_val_loader=at_val_loader,
                    phase_order=phase_order,
                    reuse_phase_a_from=reuse_phase_a_path,
                )
                results["raw"][m_key][method].append(rec)
                print(f"{stage_name} {m_key} {method} seed={seed} combined_avg_R={rec['combined_avg_R']:.4f}")

        results["stats"][m_key] = build_metric_report(
            results["raw"][m_key],
            metrics=REPORT_METRICS,
            baseline_method=baseline_method,
        )

    out_path = stage_root / f"{stage_name}_results.json"
    _save_results_shard_safe(
        stage_name=stage_name,
        out_path=out_path,
        incoming=results,
        baseline_method=baseline_method,
    )
    # If a stale failure marker exists from earlier failed attempts, remove it
    # after a successful write to avoid confusing downstream status checks.
    failure_path = stage_root / f"{stage_name}_failure.json"
    if failure_path.exists():
        failure_path.unlink()
    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": stage_name,
            "elapsed_sec": time.time() - start,
            "phase_a_source": cfg.get("phase_a_source", "coco"),
            "phase_b_source": cfg.get("phase_b_source", "audiocaps"),
            "phase_order": phase_order,
        },
    )
    save_json(provenance, stage_root / f"provenance_{stage_name}.json")
    mark_done(markers / f"{stage_name}.done.json", {"elapsed_sec": time.time() - start})
    print(f"{stage_name} complete")


def run(cfg: dict) -> None:
    run_core(cfg, stage_name="stage29_cc3m_phaseA_modular")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data import AudioCapsAVCache, KarpathyCache, extract_audiocaps_av_cache
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
)
from ..training import train_bimodal, train_trimodal
from ..training.losses import infonce_loss


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
        "embeddings": {"image": zi, "audio": za, "text": zt},
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
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    ckpt_dir.mkdir(parents=True, exist_ok=True)
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


def _method_enabled(method: str, embed_dim: int, active: set[str], method_embed_dims: dict[str, set[int]]) -> bool:
    if method not in active:
        return False
    allowed = method_embed_dims.get(method)
    if allowed is None:
        return True
    return int(embed_dim) in allowed


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage20_modular_audio_transitivity"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    if bool(cfg.get("require_gate_pass", False)):
        gate_path = Path(cfg["gate_results_path"])
        gate_obj = load_json(gate_path)
        gk = f"m{int(cfg.get('gate_embed_dim', 256))}"
        gate_pass = bool(gate_obj.get("gate", {}).get(gk, {}).get("pass", False))
        if not gate_pass:
            save_json(
                {
                    "stage": "stage20_modular_audio_transitivity",
                    "skipped": True,
                    "reason": "stage19 gate failed",
                    "gate_file": str(gate_path),
                    "gate_key": gk,
                },
                stage_root / "stage20_results.json",
            )
            mark_done(markers / "stage20_modular_audio_transitivity.done.json", {"skipped": True})
            print("Stage 20 skipped due to gate failure")
            return

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

    seeds = [int(s) for s in cfg["seeds"]]
    embed_dims = [int(m) for m in cfg["embed_dims"]]
    active_methods_list = list(cfg.get("methods", ["modular_shared_jl", "joint_shared_jl", "joint_clip_head"]))
    active_method_set = set(active_methods_list)
    method_embed_dims = {
        k: {int(x) for x in v}
        for k, v in dict(cfg.get("method_embed_dims", {})).items()
    }

    shared_raw_dim = int(cfg.get("shared_raw_dim", 768))

    results = {
        "stage": "stage20_modular_audio_transitivity",
        "methods_requested": active_methods_list,
        "method_embed_dims": {k: sorted(v) for k, v in method_embed_dims.items()},
        "raw": {},
        "stats": {},
    }

    # Build reusable loaders for trainable methods.
    it_train = coco.make_train_dataset(cfg.get("coco_train_split", "train_restval"), training=True)
    it_val = coco.make_train_dataset(cfg.get("coco_val_split", "val"), training=False)

    at_train_idx = av.split_indices(cfg.get("av_train_split", "train"))
    at_val_idx = av.split_indices(cfg.get("av_val_split", "validation"))
    at_train_ds = PairedFeatureDataset(av.audio_feats[at_train_idx], av.text_feats[at_train_idx])
    at_val_ds = PairedFeatureDataset(av.audio_feats[at_val_idx], av.text_feats[at_val_idx])

    loader_kw = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("num_workers", 4)),
        "pin_memory": True,
    }
    it_train_loader = DataLoader(it_train, shuffle=True, **loader_kw)
    it_val_loader = DataLoader(it_val, shuffle=False, **loader_kw)
    at_train_loader = DataLoader(at_train_ds, shuffle=True, **loader_kw)
    at_val_loader = DataLoader(at_val_ds, shuffle=False, **loader_kw)

    for m in embed_dims:
        m_key = f"m{m}"
        methods_this_m = [
            method_name
            for method_name in active_methods_list
            if _method_enabled(method_name, m, active_method_set, method_embed_dims)
        ]
        if not methods_this_m:
            continue

        results["raw"][m_key] = {method_name: [] for method_name in methods_this_m}

        raw_cosine_cache: dict | None = None

        for seed in seeds:
            set_seed(seed)

            # -------- modular shared-JL (phase A then phase B) --------
            if _method_enabled("modular_shared_jl", m, active_method_set, method_embed_dims):
                mod_seed_dir = stage_root / m_key / "modular_shared_jl" / f"seed{seed}"
                mod_eval = mod_seed_dir / "eval.json"
                if mod_eval.exists():
                    results["raw"][m_key]["modular_shared_jl"].append(load_json(mod_eval))
                else:
                    model = SharedJLTriModalMahalHead(
                        image_dim=int(cfg["vision_dim"]),
                        audio_dim=int(cfg["audio_dim"]),
                        text_dim=int(cfg["text_dim"]),
                        embed_dim=m,
                        shared_raw_dim=shared_raw_dim,
                        jl_eps=float(cfg["jl_eps"]),
                        jl_seed=int(cfg["jl_seed"]),
                    )

                    phase_a_dir = mod_seed_dir / "phase_a"
                    train_bimodal(
                        model,
                        it_train_loader,
                        it_val_loader,
                        epochs=int(cfg["epochs_phase_a"]),
                        lr=float(cfg["lr_phase_a"]),
                        device=str(cfg["device"]),
                        ckpt_dir=phase_a_dir,
                        patience=int(cfg.get("patience_phase_a", 8)),
                        warmup_epochs=int(cfg.get("warmup_phase_a", 0)),
                    )
                    model = _load_best(model, phase_a_dir, cfg["device"])

                    model.set_trainable(image=False, audio=True, text=False, logit_scale=True)
                    phase_b_dir = mod_seed_dir / "phase_b"
                    _train_audio_text_phase(
                        model,
                        at_train_loader,
                        at_val_loader,
                        epochs=int(cfg["epochs_phase_b"]),
                        lr=float(cfg["lr_phase_b"]),
                        device=str(cfg["device"]),
                        ckpt_dir=phase_b_dir,
                        patience=int(cfg.get("patience_phase_b", 8)),
                    )
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
                        "method": "modular_shared_jl",
                        "embed_dim": m,
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
                    save_json(rec, mod_eval)
                    results["raw"][m_key]["modular_shared_jl"].append(rec)
                    print(f"stage20 {m_key} modular_shared_jl seed={seed} combined_avg_R={rec['combined_avg_R']:.4f}")

            # -------- modular separate-JL (phase A then phase B) --------
            if _method_enabled("modular_separate_jl", m, active_method_set, method_embed_dims):
                mod_seed_dir = stage_root / m_key / "modular_separate_jl" / f"seed{seed}"
                mod_eval = mod_seed_dir / "eval.json"
                if mod_eval.exists():
                    results["raw"][m_key]["modular_separate_jl"].append(load_json(mod_eval))
                else:
                    model = SeparateJLTriModalMahalHead(
                        image_dim=int(cfg["vision_dim"]),
                        audio_dim=int(cfg["audio_dim"]),
                        text_dim=int(cfg["text_dim"]),
                        embed_dim=m,
                        shared_raw_dim=shared_raw_dim,
                        jl_eps=float(cfg["jl_eps"]),
                        jl_seed=int(cfg["jl_seed"]),
                    )

                    phase_a_dir = mod_seed_dir / "phase_a"
                    train_bimodal(
                        model,
                        it_train_loader,
                        it_val_loader,
                        epochs=int(cfg["epochs_phase_a"]),
                        lr=float(cfg["lr_phase_a"]),
                        device=str(cfg["device"]),
                        ckpt_dir=phase_a_dir,
                        patience=int(cfg.get("patience_phase_a", 8)),
                        warmup_epochs=int(cfg.get("warmup_phase_a", 0)),
                    )
                    model = _load_best(model, phase_a_dir, cfg["device"])

                    model.set_trainable(image=False, audio=True, text=False, logit_scale=True)
                    phase_b_dir = mod_seed_dir / "phase_b"
                    _train_audio_text_phase(
                        model,
                        at_train_loader,
                        at_val_loader,
                        epochs=int(cfg["epochs_phase_b"]),
                        lr=float(cfg["lr_phase_b"]),
                        device=str(cfg["device"]),
                        ckpt_dir=phase_b_dir,
                        patience=int(cfg.get("patience_phase_b", 8)),
                    )
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
                        "method": "modular_separate_jl",
                        "embed_dim": m,
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
                    save_json(rec, mod_eval)
                    results["raw"][m_key]["modular_separate_jl"].append(rec)
                    print(f"stage20 {m_key} modular_separate_jl seed={seed} combined_avg_R={rec['combined_avg_R']:.4f}")

            # -------- modular hybrid-IT-JL (image+text shared, audio separate) --------
            if _method_enabled("modular_hybrid_it_jl", m, active_method_set, method_embed_dims):
                mod_seed_dir = stage_root / m_key / "modular_hybrid_it_jl" / f"seed{seed}"
                mod_eval = mod_seed_dir / "eval.json"
                if mod_eval.exists():
                    results["raw"][m_key]["modular_hybrid_it_jl"].append(load_json(mod_eval))
                else:
                    model = HybridITJLTriModalMahalHead(
                        image_dim=int(cfg["vision_dim"]),
                        audio_dim=int(cfg["audio_dim"]),
                        text_dim=int(cfg["text_dim"]),
                        embed_dim=m,
                        shared_raw_dim=shared_raw_dim,
                        jl_eps=float(cfg["jl_eps"]),
                        jl_seed=int(cfg["jl_seed"]),
                    )

                    phase_a_dir = mod_seed_dir / "phase_a"
                    train_bimodal(
                        model,
                        it_train_loader,
                        it_val_loader,
                        epochs=int(cfg["epochs_phase_a"]),
                        lr=float(cfg["lr_phase_a"]),
                        device=str(cfg["device"]),
                        ckpt_dir=phase_a_dir,
                        patience=int(cfg.get("patience_phase_a", 8)),
                        warmup_epochs=int(cfg.get("warmup_phase_a", 0)),
                    )
                    model = _load_best(model, phase_a_dir, cfg["device"])

                    model.set_trainable(image=False, audio=True, text=False, logit_scale=True)
                    phase_b_dir = mod_seed_dir / "phase_b"
                    _train_audio_text_phase(
                        model,
                        at_train_loader,
                        at_val_loader,
                        epochs=int(cfg["epochs_phase_b"]),
                        lr=float(cfg["lr_phase_b"]),
                        device=str(cfg["device"]),
                        ckpt_dir=phase_b_dir,
                        patience=int(cfg.get("patience_phase_b", 8)),
                    )
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
                        "method": "modular_hybrid_it_jl",
                        "embed_dim": m,
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
                    save_json(rec, mod_eval)
                    results["raw"][m_key]["modular_hybrid_it_jl"].append(rec)
                    print(f"stage20 {m_key} modular_hybrid_it_jl seed={seed} combined_avg_R={rec['combined_avg_R']:.4f}")

            # -------- modular hybrid-AT-JL (audio+text shared, image separate) --------
            if _method_enabled("modular_hybrid_at_jl", m, active_method_set, method_embed_dims):
                mod_seed_dir = stage_root / m_key / "modular_hybrid_at_jl" / f"seed{seed}"
                mod_eval = mod_seed_dir / "eval.json"
                if mod_eval.exists():
                    results["raw"][m_key]["modular_hybrid_at_jl"].append(load_json(mod_eval))
                else:
                    model = HybridATJLTriModalMahalHead(
                        image_dim=int(cfg["vision_dim"]),
                        audio_dim=int(cfg["audio_dim"]),
                        text_dim=int(cfg["text_dim"]),
                        embed_dim=m,
                        shared_raw_dim=shared_raw_dim,
                        jl_eps=float(cfg["jl_eps"]),
                        jl_seed=int(cfg["jl_seed"]),
                    )

                    phase_a_dir = mod_seed_dir / "phase_a"
                    train_bimodal(
                        model,
                        it_train_loader,
                        it_val_loader,
                        epochs=int(cfg["epochs_phase_a"]),
                        lr=float(cfg["lr_phase_a"]),
                        device=str(cfg["device"]),
                        ckpt_dir=phase_a_dir,
                        patience=int(cfg.get("patience_phase_a", 8)),
                        warmup_epochs=int(cfg.get("warmup_phase_a", 0)),
                    )
                    model = _load_best(model, phase_a_dir, cfg["device"])

                    model.set_trainable(image=False, audio=True, text=False, logit_scale=True)
                    phase_b_dir = mod_seed_dir / "phase_b"
                    _train_audio_text_phase(
                        model,
                        at_train_loader,
                        at_val_loader,
                        epochs=int(cfg["epochs_phase_b"]),
                        lr=float(cfg["lr_phase_b"]),
                        device=str(cfg["device"]),
                        ckpt_dir=phase_b_dir,
                        patience=int(cfg.get("patience_phase_b", 8)),
                    )
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
                        "method": "modular_hybrid_at_jl",
                        "embed_dim": m,
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
                    save_json(rec, mod_eval)
                    results["raw"][m_key]["modular_hybrid_at_jl"].append(rec)
                    print(f"stage20 {m_key} modular_hybrid_at_jl seed={seed} combined_avg_R={rec['combined_avg_R']:.4f}")

            # -------- joint shared-JL baseline --------
            if _method_enabled("joint_shared_jl", m, active_method_set, method_embed_dims):
                joint_seed_dir = stage_root / m_key / "joint_shared_jl" / f"seed{seed}"
                joint_eval = joint_seed_dir / "eval.json"
                if joint_eval.exists():
                    results["raw"][m_key]["joint_shared_jl"].append(load_json(joint_eval))
                else:
                    model = SharedJLTriModalMahalHead(
                        image_dim=int(cfg["vision_dim"]),
                        audio_dim=int(cfg["audio_dim"]),
                        text_dim=int(cfg["text_dim"]),
                        embed_dim=m,
                        shared_raw_dim=shared_raw_dim,
                        jl_eps=float(cfg["jl_eps"]),
                        jl_seed=int(cfg["jl_seed"]),
                    )

                    def _val_eval_joint() -> dict:
                        model.eval()
                        it_m, _, _ = _eval_coco_it(model, coco, cfg.get("coco_val_split", "val"), device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
                        at_a = _encode_batches(model.encode_audio, av.audio_feats[at_val_idx], device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
                        at_t = _encode_batches(model.encode_text, av.text_feats[at_val_idx], device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
                        at_m = recall_at_k(at_a, at_t)
                        return {
                            "combined_avg_R": float(0.5 * (it_m["avg_R"] + at_m["avg_R"])),
                            "val_it_avg_R": float(it_m["avg_R"]),
                            "val_at_avg_R": float(at_m["avg_R"]),
                        }

                    train_trimodal(
                        model,
                        it_train_loader,
                        at_train_loader,
                        val_eval_fn=_val_eval_joint,
                        epochs=int(cfg["epochs_joint"]),
                        lr=float(cfg["lr_joint"]),
                        device=str(cfg["device"]),
                        ckpt_dir=joint_seed_dir,
                        patience=int(cfg.get("patience_joint", 8)),
                        warmup_epochs=int(cfg.get("warmup_joint", 0)),
                        eval_every=1,
                    )
                    model = _load_best(model, joint_seed_dir, cfg["device"])

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
                        "method": "joint_shared_jl",
                        "embed_dim": m,
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
                    save_json(rec, joint_eval)
                    results["raw"][m_key]["joint_shared_jl"].append(rec)
                    print(f"stage20 {m_key} joint_shared_jl seed={seed} combined_avg_R={rec['combined_avg_R']:.4f}")

            # -------- joint CLIP-head tri-modal baseline --------
            for joint_method_name in ("joint_clip_head", "joint_clap_head"):
                if not _method_enabled(joint_method_name, m, active_method_set, method_embed_dims):
                    continue
                clip_seed_dir = stage_root / m_key / joint_method_name / f"seed{seed}"
                clip_eval = clip_seed_dir / "eval.json"
                if clip_eval.exists():
                    results["raw"][m_key][joint_method_name].append(load_json(clip_eval))
                else:
                    model = TriModalCLIPHead(
                        image_dim=int(cfg["vision_dim"]),
                        audio_dim=int(cfg["audio_dim"]),
                        text_dim=int(cfg["text_dim"]),
                        embed_dim=m,
                    )

                    def _val_eval_clip() -> dict:
                        model.eval()
                        it_m, _, _ = _eval_coco_it(model, coco, cfg.get("coco_val_split", "val"), device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
                        at_a = _encode_batches(model.encode_audio, av.audio_feats[at_val_idx], device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
                        at_t = _encode_batches(model.encode_text, av.text_feats[at_val_idx], device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
                        at_m = recall_at_k(at_a, at_t)
                        return {
                            "combined_avg_R": float(0.5 * (it_m["avg_R"] + at_m["avg_R"])),
                            "val_it_avg_R": float(it_m["avg_R"]),
                            "val_at_avg_R": float(at_m["avg_R"]),
                        }

                    train_trimodal(
                        model,
                        it_train_loader,
                        at_train_loader,
                        val_eval_fn=_val_eval_clip,
                        epochs=int(cfg["epochs_joint"]),
                        lr=float(cfg["lr_joint"]),
                        device=str(cfg["device"]),
                        ckpt_dir=clip_seed_dir,
                        patience=int(cfg.get("patience_joint", 8)),
                        warmup_epochs=int(cfg.get("warmup_joint", 0)),
                        eval_every=1,
                    )
                    model = _load_best(model, clip_seed_dir, cfg["device"])

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
                        "method": joint_method_name,
                        "embed_dim": m,
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
                    save_json(rec, clip_eval)
                    results["raw"][m_key][joint_method_name].append(rec)
                    print(f"stage20 {m_key} {joint_method_name} seed={seed} combined_avg_R={rec['combined_avg_R']:.4f}")

            # -------- raw cosine baseline (eval only) --------
            if _method_enabled("raw_cosine_baseline", m, active_method_set, method_embed_dims):
                base_seed_dir = stage_root / m_key / "raw_cosine_baseline" / f"seed{seed}"
                base_eval = base_seed_dir / "eval.json"
                if base_eval.exists():
                    results["raw"][m_key]["raw_cosine_baseline"].append(load_json(base_eval))
                else:
                    if raw_cosine_cache is None:
                        raw_cosine_cache = _eval_raw_cosine_all(
                            coco,
                            av,
                            shared_raw_dim=shared_raw_dim,
                            coco_split=cfg.get("coco_test_split", "test"),
                            av_split=cfg.get("av_test_split", "test"),
                        )
                    rec = {
                        "seed": seed,
                        "method": "raw_cosine_baseline",
                        "embed_dim": m,
                        **raw_cosine_cache,
                    }
                    rec["combined_avg_R"] = float((rec["coco_avg_R"] + rec["av_ia_avg_R"] + rec["av_at_avg_R"]) / 3.0)
                    save_json(rec, base_eval)
                    results["raw"][m_key]["raw_cosine_baseline"].append(rec)
                    print(f"stage20 {m_key} raw_cosine_baseline seed={seed} combined_avg_R={rec['combined_avg_R']:.4f}")

        if results["raw"][m_key]:
            baseline = (
                "joint_shared_jl"
                if "joint_shared_jl" in results["raw"][m_key]
                else next(iter(results["raw"][m_key].keys()))
            )
            results["stats"][m_key] = build_metric_report(
                results["raw"][m_key],
                metrics=["combined_avg_R", "coco_avg_R", "av_it_avg_R", "av_at_avg_R", "av_ia_avg_R"],
                baseline_method=baseline,
            )

    save_json(results, stage_root / "stage20_results.json")
    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": "stage20_modular_audio_transitivity",
            "elapsed_sec": time.time() - start,
        },
    )
    save_json(provenance, stage_root / "provenance_stage20.json")
    mark_done(markers / "stage20_modular_audio_transitivity.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 20 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

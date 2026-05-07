from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data import AudioCapsAVCache, KarpathyCache
from ..data.datasets import PairedFeatureDataset
from ..eval.diagnostics import pair_diagnostics
from ..eval.stats import build_metric_report
from ..training.losses import infonce_loss
from .run_stage29_cc3m_phaseA_modular import (
    REPORT_METRICS,
    _build_model,
    _encode_batches,
    _eval_av_all,
    _eval_coco_it,
    _load_best,
    _save_results_shard_safe,
    _set_trainable_phase,
)


def _build_at_loaders(av: AudioCapsAVCache, cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    loader_kw = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("num_workers", 4)),
        "pin_memory": True,
    }
    train_idx = av.split_indices(str(cfg.get("av_train_split", "train")))
    val_idx = av.split_indices(str(cfg.get("av_val_split", "validation")))
    train_ds = PairedFeatureDataset(av.audio_feats[train_idx], av.text_feats[train_idx])
    val_ds = PairedFeatureDataset(av.audio_feats[val_idx], av.text_feats[val_idx])
    return (
        DataLoader(train_ds, shuffle=True, **loader_kw),
        DataLoader(val_ds, shuffle=False, **loader_kw),
    )


@torch.no_grad()
def _compute_image_anchor(
    model,
    av: AudioCapsAVCache,
    *,
    split_name: str,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    idx = av.split_indices(split_name)
    imgs = av.image_feats[idx]
    zi = _encode_batches(model.encode_image, imgs, device=device, batch_size=batch_size)
    return zi.mean(dim=0).to(device)


def _train_phase_b_gap_regularized(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    image_anchor: torch.Tensor,
    epochs: int,
    lr: float,
    device: str,
    ckpt_dir: Path,
    gap_lambda: float,
    patience: int = 8,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    model = model.to(device)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable params in Phase B for gap intervention")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_val = -1.0
    best_epoch = -1
    no_imp = 0
    hist = {"train_loss": [], "train_gap_pen": [], "val_r10": []}

    @torch.no_grad()
    def _val_r10() -> float:
        model.eval()
        zs, zt = [], []
        for aud, txt in val_loader:
            zs.append(model.encode_audio(aud.to(device)).cpu())
            zt.append(model.encode_text(txt.to(device)).cpu())
        from ..eval.retrieval import recall_at_k

        m = recall_at_k(torch.cat(zs, dim=0), torch.cat(zt, dim=0))
        return float(m["i2t_R@10"])

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        running_pen = 0.0
        for aud, txt in train_loader:
            aud = aud.to(device)
            txt = txt.to(device)
            scale = model.logit_scale.exp().clamp(max=100.0) if hasattr(model, "logit_scale") else (1.0 / 0.07)
            za = model.encode_audio(aud)
            zt = model.encode_text(txt)
            loss_nce = infonce_loss(za, zt, scale)
            batch_centroid = za.mean(dim=0)
            gap_pen = torch.nn.functional.mse_loss(batch_centroid, image_anchor)
            loss = loss_nce + gap_lambda * gap_pen
            if hasattr(model, "regularization_loss"):
                reg = model.regularization_loss()
                if reg is not None:
                    loss = loss + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            running_loss += float(loss.item())
            running_pen += float(gap_pen.item())
        sch.step()

        avg_loss = running_loss / max(1, len(train_loader))
        avg_pen = running_pen / max(1, len(train_loader))
        vr10 = _val_r10()
        hist["train_loss"].append(avg_loss)
        hist["train_gap_pen"].append(avg_pen)
        hist["val_r10"].append(vr10)
        print(
            f"AT+GAP Epoch {ep+1:03d}/{epochs} loss={avg_loss:.4f} "
            f"gap_pen={avg_pen:.5f} valA2T_R10={vr10:.4f}"
        )

        if vr10 > best_val:
            best_val = vr10
            best_epoch = ep + 1
            no_imp = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"AT+GAP early stop at epoch {ep+1}")
                break

    return {"best_val_r10": best_val, "best_epoch": best_epoch, "history": hist}


def _run_one(
    *,
    model,
    method: str,
    seed: int,
    m: int,
    cfg: dict[str, Any],
    stage_root: Path,
    coco: KarpathyCache,
    av: AudioCapsAVCache,
    at_train_loader: DataLoader,
    at_val_loader: DataLoader,
    reuse_phase_a_from: Path,
) -> dict[str, Any]:
    seed_dir = stage_root / f"m{m}" / method / f"seed{seed}"
    eval_p = seed_dir / "eval.json"
    if eval_p.exists():
        return load_json(eval_p)

    phase_a_dir = seed_dir / "phase_a"
    phase_b_dir = seed_dir / "phase_b"
    phase_a_dir.mkdir(parents=True, exist_ok=True)
    phase_b_dir.mkdir(parents=True, exist_ok=True)

    src = reuse_phase_a_from / f"m{m}" / method / f"seed{seed}" / "phase_a" / "best.pt"
    if not src.exists():
        raise FileNotFoundError(f"Missing reused phase-a checkpoint: {src}")
    model.load_state_dict(torch.load(src, map_location=str(cfg["device"]), weights_only=True), strict=True)
    model = model.to(str(cfg["device"]))
    if not (phase_a_dir / "best.pt").exists():
        torch.save(model.state_dict(), phase_a_dir / "best.pt")

    image_anchor = _compute_image_anchor(
        model,
        av,
        split_name=str(cfg.get("anchor_split", "train")),
        device=str(cfg["device"]),
        batch_size=int(cfg.get("eval_batch_size", 4096)),
    )

    _set_trainable_phase(model, method, "phase_b", str(cfg.get("phase_order", "it_then_at")))
    tr = _train_phase_b_gap_regularized(
        model,
        at_train_loader,
        at_val_loader,
        image_anchor=image_anchor,
        epochs=int(cfg["epochs_phase_b"]),
        lr=float(cfg["lr_phase_b"]),
        device=str(cfg["device"]),
        ckpt_dir=phase_b_dir,
        gap_lambda=float(cfg.get("centroid_reg_lambda", 0.1)),
        patience=int(cfg.get("patience_phase_b", 8)),
    )
    model = _load_best(model, phase_b_dir, str(cfg["device"]))

    coco_it, zi_coco, zt_coco = _eval_coco_it(
        model,
        coco,
        str(cfg.get("coco_test_split", "test")),
        device=str(cfg["device"]),
        batch_size=int(cfg.get("eval_batch_size", 4096)),
    )
    av_eval = _eval_av_all(
        model,
        av,
        str(cfg.get("av_test_split", "test")),
        device=str(cfg["device"]),
        batch_size=int(cfg.get("eval_batch_size", 4096)),
    )
    rec = {
        "seed": seed,
        "method": method,
        "embed_dim": m,
        "stage_variant": "gap_regularized_phase_b",
        "centroid_reg_lambda": float(cfg.get("centroid_reg_lambda", 0.1)),
        "phase_a_source": "coco",
        "phase_b_source": "audiocaps",
        "phase_order": str(cfg.get("phase_order", "it_then_at")),
        "train_phase_b_best_val_r10": float(tr["best_val_r10"]),
        "train_phase_b_best_epoch": int(tr["best_epoch"]),
        "image_anchor_norm": float(image_anchor.norm().item()),
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


def run(cfg: dict[str, Any]) -> None:
    start = time.time()
    stage_name = "stage66_gap_intervention_pilot"
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    coco_cache_dir = Path(cfg["cache_root"]) / "coco"
    coco = KarpathyCache.from_paths(
        coco_cache_dir / str(cfg["image_cache_file"]),
        coco_cache_dir / str(cfg["text_cache_file"]),
        coco_cache_dir / "metadata.json",
    )
    av_dir = Path(cfg["av_cache_root"]).resolve()
    av = AudioCapsAVCache.from_paths(
        av_dir / "image_feats_clip_raw.pt",
        av_dir / "audio_feats_clap_raw.pt",
        av_dir / "text_feats_clip_raw.pt",
        av_dir / "metadata.json",
    )
    at_train_loader, at_val_loader = _build_at_loaders(av, cfg)

    methods = [str(x) for x in cfg.get("methods", ["modular_shared_jl"])]
    seeds = [int(s) for s in cfg["seeds"]]
    embed_dims = [int(m) for m in cfg["embed_dims"]]
    baseline_method = str(cfg.get("baseline_method", methods[0]))
    reuse_phase_a_from = Path(cfg["reuse_phase_a_from"]).resolve()

    results = {
        "stage": stage_name,
        "methods_requested": methods,
        "raw": {},
        "stats": {},
        "phase_a_source": "coco",
        "phase_b_source": "audiocaps",
        "phase_order": str(cfg.get("phase_order", "it_then_at")),
        "centroid_reg_lambda": float(cfg.get("centroid_reg_lambda", 0.1)),
        "reuse_phase_a_from": str(reuse_phase_a_from),
    }

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
                    at_train_loader=at_train_loader,
                    at_val_loader=at_val_loader,
                    reuse_phase_a_from=reuse_phase_a_from,
                )
                results["raw"][m_key][method].append(rec)
                print(
                    f"{stage_name} {m_key} {method} seed={seed} "
                    f"av_ia={rec['av_ia_avg_R']:.4f} av_at={rec['av_at_avg_R']:.4f} combined={rec['combined_avg_R']:.4f}"
                )

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

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": stage_name,
            "elapsed_sec": float(time.time() - start),
            "centroid_reg_lambda": float(cfg.get("centroid_reg_lambda", 0.1)),
            "reuse_phase_a_from": str(reuse_phase_a_from),
        },
    )
    save_json(provenance, stage_root / f"provenance_{stage_name}.json")
    mark_done(markers / f"{stage_name}.done.json", {"elapsed_sec": float(time.time() - start)})
    print(f"{stage_name} complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

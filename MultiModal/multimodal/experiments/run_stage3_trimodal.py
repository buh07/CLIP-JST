from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data.datasets import AudioCapsCache, KarpathyCache
from ..eval.diagnostics import centroid_distance_matrix, pair_diagnostics
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report
from ..models import (
    TriModalCLIPHead,
    TriModalLearnedSparseJLHead,
    TriModalOrthogonalHead,
    TriModalRandomJLMahalanobisHead,
)
from ..training import train_trimodal


def _load_best(model, ckpt_dir: Path, device: str):
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))
    return model.to(device)


@torch.no_grad()
def _encode_batches(encode_fn, tensor: torch.Tensor, device: str, batch_size: int) -> torch.Tensor:
    outs = []
    for i in range(0, len(tensor), batch_size):
        outs.append(encode_fn(tensor[i:i + batch_size].to(device)).cpu())
    return torch.cat(outs)


@torch.no_grad()
def _eval_image_text(model, cache: KarpathyCache, split_name: str, *, device: str, batch_size: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    img, txt, gt_i2t, gt_t2i = cache.eval_tensors(split_name)
    zi = _encode_batches(model.encode_image, img, device, batch_size)
    zt = _encode_batches(model.encode_text, txt, device, batch_size)
    metrics = recall_at_k(zi, zt, gt_i2t=gt_i2t, gt_t2i=gt_t2i)
    return metrics, zi, zt


@torch.no_grad()
def _eval_audio_text(model, cache: AudioCapsCache, split_name: str, *, device: str, batch_size: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    aud, txt = cache.eval_tensors(split_name)
    za = _encode_batches(model.encode_audio, aud, device, batch_size)
    zt = _encode_batches(model.encode_text, txt, device, batch_size)
    metrics = recall_at_k(za, zt)
    return metrics, za, zt


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    seeds = list(cfg["seeds"])
    embed_dims = list(cfg["embed_dims"])

    # Load Karpathy caches for image-text modalities.
    karpathy_caches: dict[str, KarpathyCache] = {}
    for ds in cfg["image_text_datasets"]:
        ds_dir = Path(cfg["karpathy_cache_root"]) / ds
        karpathy_caches[ds] = KarpathyCache.from_paths(
            ds_dir / cfg["image_cache_file"],
            ds_dir / cfg["text_cache_file"],
            ds_dir / "metadata.json",
        )

    # Load AudioCaps cache.
    audio_cache = AudioCapsCache.from_paths(
        Path(cfg["audiocaps_cache_root"]) / "audio_feats_clap_raw.pt",
        Path(cfg["audiocaps_cache_root"]) / "text_feats_clip_raw.pt",
        Path(cfg["audiocaps_cache_root"]) / "metadata.json",
    )

    results: dict = {"stage": "stage3_trimodal", "raw": {}, "stats": {}}

    for m in embed_dims:
        m_key = f"m{m}"
        results["raw"][m_key] = {}

        for method in ["clip_head", "random_jl_mahal", "orth_jl_trainable", "learned_jl_sparse"]:
            results["raw"][m_key][method] = []

            for seed in seeds:
                set_seed(seed)
                seed_dir = output_root / m_key / method / f"seed{seed}"
                eval_file = seed_dir / "eval.json"
                if eval_file.exists():
                    results["raw"][m_key][method].append(load_json(eval_file))
                    continue

                # Build train/val datasets.
                image_train_parts = []
                image_val_parts = []
                for ds, cache in karpathy_caches.items():
                    if ds == "coco":
                        image_train_parts.append(cache.make_train_dataset("train_restval", training=True))
                        image_val_parts.append(cache.make_train_dataset("val", training=False))
                    else:
                        image_train_parts.append(cache.make_train_dataset("train", training=True))
                        image_val_parts.append(cache.make_train_dataset("val", training=False))

                it_train = ConcatDataset(image_train_parts)
                it_val = ConcatDataset(image_val_parts)
                at_train = audio_cache.make_dataset("train")
                at_val = audio_cache.make_dataset("validation")

                loader_kw = {
                    "batch_size": cfg["batch_size"],
                    "num_workers": cfg.get("num_workers", 4),
                    "pin_memory": True,
                }
                it_train_loader = DataLoader(it_train, shuffle=True, **loader_kw)
                it_val_loader = DataLoader(it_val, shuffle=False, **loader_kw)
                at_train_loader = DataLoader(at_train, shuffle=True, **loader_kw)
                at_val_loader = DataLoader(at_val, shuffle=False, **loader_kw)

                if method == "clip_head":
                    model = TriModalCLIPHead(
                        image_dim=cfg["image_dim"],
                        audio_dim=cfg["audio_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                    )
                elif method == "random_jl_mahal":
                    model = TriModalRandomJLMahalanobisHead(
                        image_dim=cfg["image_dim"],
                        audio_dim=cfg["audio_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        jl_eps=cfg["jl_eps"],
                        jl_seed=cfg["jl_seed"],
                    )
                elif method == "orth_jl_trainable":
                    model = TriModalOrthogonalHead(
                        image_dim=cfg["image_dim"],
                        audio_dim=cfg["audio_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        orth_reg=cfg.get("orth_reg", 1e-3),
                    )
                else:
                    model = TriModalLearnedSparseJLHead(
                        image_dim=cfg["image_dim"],
                        audio_dim=cfg["audio_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        jl_eps=cfg["jl_eps"],
                        jl_seed=cfg["jl_seed"],
                        sparsity_reg=cfg.get("sparsity_reg", 1e-4),
                    )

                def val_eval_fn() -> dict:
                    model.eval()
                    it_i, it_t = [], []
                    for img, txt in it_val_loader:
                        it_i.append(model.encode_image(img.to(cfg["device"])).cpu())
                        it_t.append(model.encode_text(txt.to(cfg["device"])).cpu())
                    it_metrics = recall_at_k(torch.cat(it_i), torch.cat(it_t))

                    at_a, at_t = [], []
                    for aud, txt in at_val_loader:
                        at_a.append(model.encode_audio(aud.to(cfg["device"])).cpu())
                        at_t.append(model.encode_text(txt.to(cfg["device"])).cpu())
                    at_metrics = recall_at_k(torch.cat(at_a), torch.cat(at_t))

                    combined = 0.5 * (it_metrics["avg_R"] + at_metrics["avg_R"])
                    return {
                        "combined_avg_R": combined,
                        "val_it_avg_R": it_metrics["avg_R"],
                        "val_at_avg_R": at_metrics["avg_R"],
                    }

                train_trimodal(
                    model,
                    it_train_loader,
                    at_train_loader,
                    val_eval_fn=val_eval_fn,
                    epochs=cfg["epochs"],
                    lr=cfg["lr"],
                    device=cfg["device"],
                    ckpt_dir=seed_dir,
                    patience=cfg.get("patience", 10),
                    warmup_epochs=cfg.get("warmup_epochs", 0),
                    eval_every=cfg.get("eval_every", 1),
                )

                model = _load_best(model, seed_dir, cfg["device"])

                image_task = {}
                image_embs = []
                text_image_embs = []
                for ds, cache in karpathy_caches.items():
                    split = "test"
                    met, zi, zt = _eval_image_text(
                        model,
                        cache,
                        split,
                        device=cfg["device"],
                        batch_size=cfg["eval_batch_size"],
                    )
                    image_task[ds] = met
                    image_embs.append(zi)
                    text_image_embs.append(zt)

                audio_metrics, za, zta = _eval_audio_text(
                    model,
                    audio_cache,
                    "test",
                    device=cfg["device"],
                    batch_size=cfg["eval_batch_size"],
                )

                img_all = torch.cat(image_embs)
                txt_img_all = torch.cat(text_image_embs)
                txt_all = torch.cat([txt_img_all, zta])

                diag = {}
                diag.update(pair_diagnostics(img_all, txt_img_all, prefix="it"))
                diag.update(pair_diagnostics(za, zta, prefix="at"))
                diag["centroid_distance_matrix"] = centroid_distance_matrix(
                    {
                        "image": img_all,
                        "audio": za,
                        "text": txt_all,
                    }
                )

                avg_it = sum(v["avg_R"] for v in image_task.values()) / len(image_task)
                combined = (avg_it + audio_metrics["avg_R"]) / 2.0

                rec = {
                    "seed": seed,
                    "embed_dim": m,
                    "method": method,
                    "image_text": image_task,
                    "audio_text": audio_metrics,
                    "avg_image_text_avg_R": avg_it,
                    "combined_avg_R": combined,
                    "n_params": model.n_trainable_params(),
                    "diagnostics": diag,
                }
                save_json(rec, eval_file)
                results["raw"][m_key][method].append(rec)
                print(f"trimodal {m_key} {method} seed={seed} combined_avg_R={combined:.4f}")

        # Stats per m.
        flat: dict[str, list[dict]] = {}
        for method, rows in results["raw"][m_key].items():
            flat[method] = []
            for r in rows:
                flat[method].append(
                    {
                        "seed": r["seed"],
                        "combined_avg_R": r["combined_avg_R"],
                        "avg_image_text_avg_R": r["avg_image_text_avg_R"],
                        "audio_text_avg_R": r["audio_text"]["avg_R"],
                    }
                )

        results["stats"][m_key] = build_metric_report(
            flat,
            metrics=["combined_avg_R", "avg_image_text_avg_R", "audio_text_avg_R"],
            baseline_method="clip_head",
        )

    provenance = env_snapshot(Path(cfg["project_root"]), seeds=seeds, extra={
        "stage": "stage3_trimodal",
        "elapsed_sec": time.time() - start,
    })
    save_json(provenance, output_root / "provenance_stage3.json")
    save_json(results, output_root / "trimodal_results.json")
    mark_done(markers / "stage3_trimodal.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 3 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

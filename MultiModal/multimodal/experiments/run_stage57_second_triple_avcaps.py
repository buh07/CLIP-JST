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
from ..data import AudioCapsAVCache, extract_avcaps_av_cache
from ..data.datasets import PairedFeatureDataset
from ..eval.diagnostics import centroid_distance_matrix, pair_diagnostics
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report
from .run_stage29_cc3m_phaseA_modular import (
    _build_model,
    _encode_batches,
    _load_best,
    _set_trainable_phase,
    _train_phase_at,
    _train_phase_it,
)


REPORT_METRICS = ["combined_avg_R", "av_it_avg_R", "av_at_avg_R", "av_ia_avg_R"]


@torch.no_grad()
def _eval_triplet_all(model, cache: AudioCapsAVCache, split_name: str, *, device: str, batch_size: int) -> dict[str, Any]:
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
        "video_text": m_it,
        "audio_text": m_at,
        "video_audio": m_ia,
        "diagnostics": diag,
    }


@torch.no_grad()
def _eval_raw_cosine_all(cache: AudioCapsAVCache, *, shared_raw_dim: int, split_name: str) -> dict[str, Any]:
    def _pad(x: torch.Tensor, d: int) -> torch.Tensor:
        if d == shared_raw_dim:
            return x
        if d > shared_raw_dim:
            raise ValueError(f"raw dim {d} exceeds shared_raw_dim {shared_raw_dim}")
        return F.pad(x, (0, shared_raw_dim - d), value=0.0)

    img, aud, txt = cache.eval_tensors(split_name)
    zi = F.normalize(_pad(img, int(img.shape[1])), dim=-1)
    za = F.normalize(_pad(aud, int(aud.shape[1])), dim=-1)
    zt = F.normalize(_pad(txt, int(txt.shape[1])), dim=-1)

    m_it = recall_at_k(zi, zt)
    m_at = recall_at_k(za, zt)
    m_ia = recall_at_k(zi, za)

    diag = {}
    diag.update(pair_diagnostics(zi, zt, prefix="it"))
    diag.update(pair_diagnostics(za, zt, prefix="at"))
    diag.update(pair_diagnostics(zi, za, prefix="ia"))
    diag["centroid_distance_matrix"] = centroid_distance_matrix({"image": zi, "audio": za, "text": zt})

    return {
        "video_text": m_it,
        "audio_text": m_at,
        "video_audio": m_ia,
        "diagnostics": diag,
    }


def _build_loaders(cache: AudioCapsAVCache, cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    loader_kw = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("num_workers", 4)),
        "pin_memory": True,
    }

    tr_idx = cache.split_indices(cfg.get("av_train_split", "train"))
    va_idx = cache.split_indices(cfg.get("av_val_split", "validation"))

    it_train = PairedFeatureDataset(cache.image_feats[tr_idx], cache.text_feats[tr_idx])
    it_val = PairedFeatureDataset(cache.image_feats[va_idx], cache.text_feats[va_idx])
    at_train = PairedFeatureDataset(cache.audio_feats[tr_idx], cache.text_feats[tr_idx])
    at_val = PairedFeatureDataset(cache.audio_feats[va_idx], cache.text_feats[va_idx])

    return (
        DataLoader(it_train, shuffle=True, **loader_kw),
        DataLoader(it_val, shuffle=False, **loader_kw),
        DataLoader(at_train, shuffle=True, **loader_kw),
        DataLoader(at_val, shuffle=False, **loader_kw),
    )


def _run_one(
    *,
    model,
    method: str,
    seed: int,
    m: int,
    cfg: dict,
    stage_root: Path,
    cache: AudioCapsAVCache,
    it_train_loader: DataLoader,
    it_val_loader: DataLoader,
    at_train_loader: DataLoader,
    at_val_loader: DataLoader,
) -> dict[str, Any]:
    seed_dir = stage_root / f"m{m}" / method / f"seed{seed}"
    eval_p = seed_dir / "eval.json"
    if eval_p.exists():
        return load_json(eval_p)

    if method == "raw_cosine_baseline":
        ev = _eval_raw_cosine_all(
            cache,
            shared_raw_dim=int(cfg.get("shared_raw_dim", 768)),
            split_name=str(cfg.get("av_test_split", "test")),
        )
        rec = {
            "seed": seed,
            "method": method,
            "embed_dim": m,
            "video_text": ev["video_text"],
            "audio_text": ev["audio_text"],
            "video_audio": ev["video_audio"],
            "diagnostics": ev["diagnostics"],
            "av_it_avg_R": float(ev["video_text"]["avg_R"]),
            "av_at_avg_R": float(ev["audio_text"]["avg_R"]),
            "av_ia_avg_R": float(ev["video_audio"]["avg_R"]),
        }
        rec["combined_avg_R"] = float((rec["av_it_avg_R"] + rec["av_at_avg_R"] + rec["av_ia_avg_R"]) / 3.0)
        save_json(rec, eval_p)
        return rec

    set_seed(seed)
    phase_a_dir = seed_dir / "phase_a"
    _set_trainable_phase(model, method, "phase_a", "it_then_at")
    _train_phase_it(model, it_train_loader, it_val_loader, cfg, phase_a_dir)
    model = _load_best(model, phase_a_dir, str(cfg["device"]))

    phase_b_dir = seed_dir / "phase_b"
    _set_trainable_phase(model, method, "phase_b", "it_then_at")
    _train_phase_at(model, at_train_loader, at_val_loader, cfg, phase_b_dir)
    model = _load_best(model, phase_b_dir, str(cfg["device"]))

    ev = _eval_triplet_all(
        model,
        cache,
        str(cfg.get("av_test_split", "test")),
        device=str(cfg["device"]),
        batch_size=int(cfg["eval_batch_size"]),
    )

    rec = {
        "seed": seed,
        "method": method,
        "embed_dim": m,
        "video_text": ev["video_text"],
        "audio_text": ev["audio_text"],
        "video_audio": ev["video_audio"],
        "diagnostics": ev["diagnostics"],
        "av_it_avg_R": float(ev["video_text"]["avg_R"]),
        "av_at_avg_R": float(ev["audio_text"]["avg_R"]),
        "av_ia_avg_R": float(ev["video_audio"]["avg_R"]),
    }
    rec["combined_avg_R"] = float((rec["av_it_avg_R"] + rec["av_at_avg_R"] + rec["av_ia_avg_R"]) / 3.0)
    save_json(rec, eval_p)
    return rec


def _merge_seed_rows(dst_rows: list[dict[str, Any]], src_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed = {int(r["seed"]): r for r in dst_rows if "seed" in r}
    for r in src_rows:
        by_seed[int(r["seed"])] = r
    return [by_seed[s] for s in sorted(by_seed)]


def _merge_results_file(*, stage_root: Path, stage_name: str, incoming: dict[str, Any]) -> None:
    out_path = stage_root / f"{stage_name}_results.json"
    lock_path = stage_root / f".{stage_name}.merge.lock"
    stage_root.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)

        if out_path.exists():
            existing = load_json(out_path)
        else:
            existing = {
                "stage": stage_name,
                "raw": {},
                "stats": {},
                "methods_requested": incoming.get("methods_requested", []),
                "embed_dims_requested": incoming.get("embed_dims_requested", []),
                "seeds_requested": incoming.get("seeds_requested", []),
            }

        for m_key, methods in incoming.get("raw", {}).items():
            dst_m = existing.setdefault("raw", {}).setdefault(m_key, {})
            for method, rows in methods.items():
                dst_rows = dst_m.get(method, [])
                dst_m[method] = _merge_seed_rows(dst_rows, rows)

        existing["stats"] = {}
        baseline = existing.get("baseline_method") or incoming.get("baseline_method")
        for m_key, methods in existing.get("raw", {}).items():
            existing["stats"][m_key] = build_metric_report(
                methods,
                metrics=REPORT_METRICS,
                baseline_method=str(baseline) if baseline else None,
            )

        existing["baseline_method"] = baseline
        existing["elapsed_sec"] = float(existing.get("elapsed_sec", 0.0)) + float(incoming.get("elapsed_sec", 0.0))

        save_json(existing, out_path)
        fcntl.flock(lock_f, fcntl.LOCK_UN)


def run(cfg: dict) -> None:
    start = time.time()
    stage_name = "stage57_second_triple_avcaps"

    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cfg.get("avcaps_cache_root", output_root / "caches" / "avcaps_av")).resolve()
    if bool(cfg.get("build_avcaps_cache_if_missing", True)):
        paths = extract_avcaps_av_cache(
            out_dir=cache_root,
            dataset_name=str(cfg.get("avcaps_dataset", "TUT-ARG/AVCaps")),
            clap_model_name=str(cfg["clap_model"]),
            clip_backbone_name=str(cfg["clip_backbone"]),
            target_sampling_rate=int(cfg.get("avcaps_target_sr", 48_000)),
            device=str(cfg["device"]),
            audio_batch_size=int(cfg.get("avcaps_audio_batch_size", 32)),
            image_batch_size=int(cfg.get("avcaps_image_batch_size", 32)),
            text_batch_size=int(cfg.get("avcaps_text_batch_size", 128)),
            max_examples_per_split=cfg.get("avcaps_max_examples_per_split"),
        )
    else:
        paths = {
            "image": cache_root / "image_feats_clip_raw.pt",
            "audio": cache_root / "audio_feats_clap_raw.pt",
            "text": cache_root / "text_feats_clip_raw.pt",
            "meta": cache_root / "metadata.json",
        }

    cache = AudioCapsAVCache.from_paths(paths["image"], paths["audio"], paths["text"], paths["meta"])
    it_train_loader, it_val_loader, at_train_loader, at_val_loader = _build_loaders(cache, cfg)

    methods = [str(x) for x in cfg.get("methods", ["modular_shared_jl"])]
    seeds = [int(s) for s in cfg["seeds"]]
    embed_dims = [int(m) for m in cfg["embed_dims"]]
    baseline_method = str(cfg.get("baseline_method", methods[0]))

    local = {
        "stage": stage_name,
        "methods_requested": methods,
        "embed_dims_requested": embed_dims,
        "seeds_requested": seeds,
        "baseline_method": baseline_method,
        "raw": {},
        "stats": {},
    }

    for m in embed_dims:
        m_key = f"m{m}"
        local["raw"][m_key] = {method: [] for method in methods}
        for seed in seeds:
            for method in methods:
                model = _build_model(method, m, cfg)
                rec = _run_one(
                    model=model,
                    method=method,
                    seed=seed,
                    m=m,
                    cfg=cfg,
                    stage_root=stage_root,
                    cache=cache,
                    it_train_loader=it_train_loader,
                    it_val_loader=it_val_loader,
                    at_train_loader=at_train_loader,
                    at_val_loader=at_val_loader,
                )
                local["raw"][m_key][method].append(rec)
                print(
                    f"{stage_name} {m_key} {method} seed={seed} "
                    f"vt={rec['av_it_avg_R']:.4f} at={rec['av_at_avg_R']:.4f} va={rec['av_ia_avg_R']:.4f}"
                )

        local["stats"][m_key] = build_metric_report(
            local["raw"][m_key],
            metrics=REPORT_METRICS,
            baseline_method=baseline_method,
        )

    local["elapsed_sec"] = time.time() - start
    _merge_results_file(stage_root=stage_root, stage_name=stage_name, incoming=local)

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={"stage": stage_name, "elapsed_sec": time.time() - start},
    )
    save_json(provenance, stage_root / f"provenance_{stage_name}.json")
    mark_done(markers / f"{stage_name}.done.json", {"elapsed_sec": time.time() - start})
    print(f"{stage_name} complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

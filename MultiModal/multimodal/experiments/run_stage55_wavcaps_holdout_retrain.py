from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data import AudioCapsAVCache, extract_wavcaps_audio_text_cache
from ..data.datasets import PairedFeatureDataset
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report
from .run_stage29_cc3m_phaseA_modular import (
    _build_model,
    _encode_batches,
    _eval_av_all,
    _set_trainable_phase,
    _train_phase_at,
)

REPORT_METRICS = [
    "av_it_avg_R",
    "av_at_avg_R",
    "av_ia_avg_R",
    "wav_holdout_at_avg_R",
]


def _filter_sources(
    wav_audio: torch.Tensor,
    wav_text: torch.Tensor,
    wav_meta: dict[str, Any],
    include_sources: list[str],
    exclude_sources: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not include_sources and not exclude_sources:
        return wav_audio, wav_text
    sample_sources = wav_meta.get("sample_sources") if isinstance(wav_meta, dict) else None
    if not isinstance(sample_sources, list) or len(sample_sources) != len(wav_audio):
        raise RuntimeError("wavcaps source filtering requested but sample_sources metadata is unavailable or mismatched")

    include_set = set(str(x) for x in include_sources)
    exclude_set = set(str(x) for x in exclude_sources)
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
    return wav_audio.index_select(0, idx_t), wav_text.index_select(0, idx_t)


def _build_wavcaps_splits(cfg: dict, output_root: Path):
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

    wav_audio, wav_text = _filter_sources(
        wav_audio,
        wav_text,
        wav_meta,
        [str(x) for x in cfg.get("wavcaps_source_filter_include", [])],
        [str(x) for x in cfg.get("wavcaps_source_filter_exclude", [])],
    )

    subsample_n = cfg.get("wavcaps_subsample_n")
    if subsample_n is not None and int(subsample_n) < len(wav_audio):
        g_sub = torch.Generator()
        g_sub.manual_seed(int(cfg.get("wavcaps_subsample_seed", 9999)))
        sub_perm = torch.randperm(len(wav_audio), generator=g_sub)[:int(subsample_n)]
        wav_audio = wav_audio[sub_perm]
        wav_text = wav_text[sub_perm]

    n = len(wav_audio)
    holdout_n = int(cfg.get("wavcaps_holdout_size", 5000))
    holdout_n = max(1, min(holdout_n, n - 2))
    g_hold = torch.Generator()
    g_hold.manual_seed(int(cfg.get("wavcaps_holdout_seed", 4242)))
    perm_hold = torch.randperm(n, generator=g_hold)
    hold_idx = perm_hold[:holdout_n]
    rem_idx = perm_hold[holdout_n:]

    hold_audio = wav_audio[hold_idx]
    hold_text = wav_text[hold_idx]

    rem_audio = wav_audio[rem_idx]
    rem_text = wav_text[rem_idx]

    g_split = torch.Generator()
    g_split.manual_seed(int(cfg.get("wavcaps_split_seed", 2026)))
    perm = torch.randperm(len(rem_audio), generator=g_split)
    rem_audio = rem_audio[perm]
    rem_text = rem_text[perm]

    n_val_frac = max(1, int(round(len(rem_audio) * float(cfg.get("wavcaps_val_frac", 0.1)))))
    max_val = int(cfg.get("wavcaps_val_max_gallery", 500))
    n_val = min(n_val_frac, max_val, len(rem_audio) - 1)

    train_audio = rem_audio[:-n_val]
    train_text = rem_text[:-n_val]
    val_audio = rem_audio[-n_val:]
    val_text = rem_text[-n_val:]

    return (
        PairedFeatureDataset(train_audio, train_text),
        PairedFeatureDataset(val_audio, val_text),
        hold_audio,
        hold_text,
        {
            "n_total": int(n),
            "n_train": int(len(train_audio)),
            "n_val": int(len(val_audio)),
            "n_holdout": int(len(hold_audio)),
        },
    )


@torch.no_grad()
def _eval_holdout_at(model, hold_audio: torch.Tensor, hold_text: torch.Tensor, *, device: str, batch_size: int) -> dict[str, float]:
    za = _encode_batches(model.encode_audio, hold_audio, device=device, batch_size=batch_size)
    zt = _encode_batches(model.encode_text, hold_text, device=device, batch_size=batch_size)
    met = recall_at_k(za, zt)
    return {k: float(v) for k, v in met.items()}


def _phase_a_ckpt_path(cfg: dict, method: str, m: int, seed: int) -> Path:
    base = Path(str(cfg["reuse_phase_a_from"]))
    stage_name = str(cfg.get("reuse_phase_a_stage_name", "stage44_zero_shot_baseline_control"))
    return base / stage_name / f"m{m}" / method / f"seed{seed}" / "phase_a" / "best.pt"


def _run_one(
    *,
    cfg: dict,
    stage_root: Path,
    av: AudioCapsAVCache,
    at_train_loader: DataLoader,
    at_val_loader: DataLoader,
    hold_audio: torch.Tensor,
    hold_text: torch.Tensor,
    method: str,
    m: int,
    seed: int,
) -> dict[str, Any]:
    seed_dir = stage_root / f"m{m}" / method / f"seed{seed}"
    eval_p = seed_dir / "eval.json"
    if eval_p.exists():
        return load_json(eval_p)

    set_seed(seed)
    model = _build_model(method, m, cfg)
    ckpt = _phase_a_ckpt_path(cfg, method, m, seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing phase-a checkpoint: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=cfg["device"], weights_only=True), strict=True)
    model = model.to(cfg["device"])

    _set_trainable_phase(model, method, "phase_b", str(cfg.get("phase_order", "it_then_at")))

    phase_b_dir = seed_dir / "phase_b"
    train_stats = _train_phase_at(model, at_train_loader, at_val_loader, cfg, phase_b_dir)
    model.load_state_dict(torch.load(phase_b_dir / "best.pt", map_location=cfg["device"], weights_only=True), strict=True)
    model = model.to(cfg["device"]).eval()

    av_eval = _eval_av_all(
        model,
        av,
        str(cfg.get("av_test_split", "test")),
        device=str(cfg["device"]),
        batch_size=int(cfg["eval_batch_size"]),
    )
    hold = _eval_holdout_at(
        model,
        hold_audio,
        hold_text,
        device=str(cfg["device"]),
        batch_size=int(cfg["eval_batch_size"]),
    )

    rec = {
        "seed": int(seed),
        "method": str(method),
        "embed_dim": int(m),
        "phase_a_source": "coco_reuse_stage44",
        "phase_b_source": "wavcaps",
        "condition_name": str(cfg.get("condition_name", "unknown")),
        "train_stats": train_stats,
        "wav_holdout_audio_text": hold,
        "av_image_text": av_eval["image_text"],
        "av_audio_text": av_eval["audio_text"],
        "av_image_audio": av_eval["image_audio"],
        "av_it_avg_R": float(av_eval["image_text"]["avg_R"]),
        "av_at_avg_R": float(av_eval["audio_text"]["avg_R"]),
        "av_ia_avg_R": float(av_eval["image_audio"]["avg_R"]),
        "wav_holdout_at_avg_R": float(hold["avg_R"]),
    }
    save_json(rec, eval_p)
    return rec


def run(cfg: dict) -> None:
    start = time.time()
    stage_name = "stage55_wavcaps_holdout_retrain"

    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    av_dir = Path(cfg["av_cache_root"]).resolve()
    av = AudioCapsAVCache.from_paths(
        av_dir / "image_feats_clip_raw.pt",
        av_dir / "audio_feats_clap_raw.pt",
        av_dir / "text_feats_clip_raw.pt",
        av_dir / "metadata.json",
    )

    at_train_ds, at_val_ds, hold_audio, hold_text, split_meta = _build_wavcaps_splits(cfg, output_root)
    loader_kw = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("num_workers", 4)),
        "pin_memory": True,
    }
    at_train_loader = DataLoader(at_train_ds, shuffle=True, **loader_kw)
    at_val_loader = DataLoader(at_val_ds, shuffle=False, **loader_kw)

    methods = [str(x) for x in cfg.get("methods", ["modular_shared_jl"]) ]
    seeds = [int(s) for s in cfg["seeds"]]
    embed_dims = [int(m) for m in cfg["embed_dims"]]

    results = {
        "stage": stage_name,
        "condition_name": str(cfg.get("condition_name", "unknown")),
        "methods_requested": methods,
        "split_meta": split_meta,
        "raw": {},
        "stats": {},
    }

    baseline_method = str(cfg.get("baseline_method", methods[0]))

    for m in embed_dims:
        m_key = f"m{m}"
        results["raw"][m_key] = {method: [] for method in methods}
        for seed in seeds:
            for method in methods:
                rec = _run_one(
                    cfg=cfg,
                    stage_root=stage_root,
                    av=av,
                    at_train_loader=at_train_loader,
                    at_val_loader=at_val_loader,
                    hold_audio=hold_audio,
                    hold_text=hold_text,
                    method=method,
                    m=m,
                    seed=seed,
                )
                results["raw"][m_key][method].append(rec)
                print(
                    f"{stage_name} {results['condition_name']} {m_key} {method} seed={seed} "
                    f"av_at={rec['av_at_avg_R']:.4f} wav_holdout_at={rec['wav_holdout_at_avg_R']:.4f} av_ia={rec['av_ia_avg_R']:.4f}"
                )

        results["stats"][m_key] = build_metric_report(
            results["raw"][m_key],
            metrics=REPORT_METRICS,
            baseline_method=baseline_method,
        )

    out_path = stage_root / f"{stage_name}_results.json"
    save_json(results, out_path)

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": stage_name,
            "elapsed_sec": time.time() - start,
            "condition_name": results["condition_name"],
            "split_meta": split_meta,
        },
    )
    save_json(provenance, stage_root / f"provenance_{stage_name}.json")
    mark_done(markers / f"{stage_name}.done.json", {"elapsed_sec": time.time() - start, "condition_name": results["condition_name"]})
    print(f"{stage_name} complete ({results['condition_name']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

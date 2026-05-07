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
from ..data import AudioCapsAVCache, KarpathyCache, extract_speechcoco_av_cache
from ..data.datasets import ImageCaptionTrainDataset, PairedFeatureDataset
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
        "image_text": m_it,
        "audio_text": m_at,
        "image_audio": m_ia,
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
        "image_text": m_it,
        "audio_text": m_at,
        "image_audio": m_ia,
        "diagnostics": diag,
    }


def _load_coco_image_ids(coco_meta_path: Path) -> list[int]:
    meta = load_json(coco_meta_path)
    ids = [int(x) for x in meta.get("image_ids", [])]
    if not ids:
        raise RuntimeError(f"Missing image_ids in COCO metadata: {coco_meta_path}")
    return ids


def _filter_indices_by_forbidden_image_ids(
    indices: list[int],
    image_ids: list[int],
    forbidden_ids: set[int],
) -> tuple[list[int], int]:
    out: list[int] = []
    removed = 0
    for i in indices:
        if int(image_ids[int(i)]) in forbidden_ids:
            removed += 1
            continue
        out.append(int(i))
    return out, removed


def _build_loaders(
    cfg: dict,
    coco: KarpathyCache,
    coco_image_ids: list[int],
    cache: AudioCapsAVCache,
    speech_meta: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    loader_kw = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("num_workers", 4)),
        "pin_memory": True,
    }

    strict_disjoint = bool(cfg.get("strict_disjoint", True))

    speech_image_ids = [int(x) for x in speech_meta.get("image_ids", [])]
    if len(speech_image_ids) != len(cache.image_feats):
        raise RuntimeError("SpeechCoco metadata image_ids length mismatch with features")

    phase_b_train_idx = [int(x) for x in cache.split_indices(cfg.get("speech_train_split", "phase_b_train"))]
    phase_b_val_idx = [int(x) for x in cache.split_indices(cfg.get("speech_val_split", "phase_b_val"))]
    eval_idx = [int(x) for x in cache.split_indices(cfg.get("speech_eval_split", "eval_test"))]

    eval_image_ids = {speech_image_ids[i] for i in eval_idx}

    # Sanity check speech split disjointness.
    train_eval_overlap = {speech_image_ids[i] for i in phase_b_train_idx}.intersection(eval_image_ids)
    val_eval_overlap = {speech_image_ids[i] for i in phase_b_val_idx}.intersection(eval_image_ids)
    if strict_disjoint and (train_eval_overlap or val_eval_overlap):
        raise RuntimeError(
            "SpeechCoco strict disjoint violated before training: "
            f"train_eval={len(train_eval_overlap)} val_eval={len(val_eval_overlap)}"
        )

    # COCO Phase-A indices and disjoint filtering.
    coco_train_idx = [int(x) for x in coco.split_indices(cfg.get("coco_train_split", "train_restval"))]
    coco_val_idx = [int(x) for x in coco.split_indices(cfg.get("coco_val_split", "val"))]

    coco_train_idx_f, removed_train = _filter_indices_by_forbidden_image_ids(coco_train_idx, coco_image_ids, eval_image_ids)
    coco_val_idx_f, removed_val = _filter_indices_by_forbidden_image_ids(coco_val_idx, coco_image_ids, eval_image_ids)

    if strict_disjoint:
        if not coco_train_idx_f:
            raise RuntimeError("COCO Phase-A train split is empty after strict disjoint filtering")
        if not coco_val_idx_f:
            raise RuntimeError("COCO Phase-A val split is empty after strict disjoint filtering")

    it_train = ImageCaptionTrainDataset(
        img_feats=coco.image_feats,
        txt_feats=coco.text_feats,
        image_indices=coco_train_idx_f,
        n_captions=coco.n_captions,
    )
    it_train.train(True)

    it_val = ImageCaptionTrainDataset(
        img_feats=coco.image_feats,
        txt_feats=coco.text_feats,
        image_indices=coco_val_idx_f,
        n_captions=coco.n_captions,
    )
    it_val.train(False)

    at_train = PairedFeatureDataset(cache.audio_feats[phase_b_train_idx], cache.text_feats[phase_b_train_idx])
    at_val = PairedFeatureDataset(cache.audio_feats[phase_b_val_idx], cache.text_feats[phase_b_val_idx])

    disjoint_diag = {
        "strict_disjoint": strict_disjoint,
        "speech_eval_examples": int(len(eval_idx)),
        "speech_phaseb_train_examples": int(len(phase_b_train_idx)),
        "speech_phaseb_val_examples": int(len(phase_b_val_idx)),
        "speech_train_eval_overlap_count": int(len(train_eval_overlap)),
        "speech_val_eval_overlap_count": int(len(val_eval_overlap)),
        "coco_train_removed_for_disjoint": int(removed_train),
        "coco_val_removed_for_disjoint": int(removed_val),
        "coco_train_examples_after_filter": int(len(coco_train_idx_f)),
        "coco_val_examples_after_filter": int(len(coco_val_idx_f)),
    }

    return (
        DataLoader(it_train, shuffle=True, **loader_kw),
        DataLoader(it_val, shuffle=False, **loader_kw),
        DataLoader(at_train, shuffle=True, **loader_kw),
        DataLoader(at_val, shuffle=False, **loader_kw),
        disjoint_diag,
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
    disjoint_diag: dict[str, Any],
) -> dict[str, Any]:
    seed_dir = stage_root / f"m{m}" / method / f"seed{seed}"
    eval_p = seed_dir / "eval.json"
    if eval_p.exists():
        return load_json(eval_p)

    if method == "raw_cosine_baseline":
        ev = _eval_raw_cosine_all(
            cache,
            shared_raw_dim=int(cfg.get("shared_raw_dim", 768)),
            split_name=str(cfg.get("speech_eval_split", "eval_test")),
        )
        rec = {
            "seed": seed,
            "method": method,
            "embed_dim": m,
            "image_text": ev["image_text"],
            "audio_text": ev["audio_text"],
            "image_audio": ev["image_audio"],
            "diagnostics": ev["diagnostics"],
            "disjoint_diagnostics": disjoint_diag,
            "av_it_avg_R": float(ev["image_text"]["avg_R"]),
            "av_at_avg_R": float(ev["audio_text"]["avg_R"]),
            "av_ia_avg_R": float(ev["image_audio"]["avg_R"]),
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
        str(cfg.get("speech_eval_split", "eval_test")),
        device=str(cfg["device"]),
        batch_size=int(cfg["eval_batch_size"]),
    )

    rec = {
        "seed": seed,
        "method": method,
        "embed_dim": m,
        "image_text": ev["image_text"],
        "audio_text": ev["audio_text"],
        "image_audio": ev["image_audio"],
        "diagnostics": ev["diagnostics"],
        "disjoint_diagnostics": disjoint_diag,
        "av_it_avg_R": float(ev["image_text"]["avg_R"]),
        "av_at_avg_R": float(ev["audio_text"]["avg_R"]),
        "av_ia_avg_R": float(ev["image_audio"]["avg_R"]),
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
                "disjoint_diagnostics": incoming.get("disjoint_diagnostics", {}),
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
        existing["disjoint_diagnostics"] = incoming.get("disjoint_diagnostics", existing.get("disjoint_diagnostics", {}))
        existing["elapsed_sec"] = float(existing.get("elapsed_sec", 0.0)) + float(incoming.get("elapsed_sec", 0.0))

        save_json(existing, out_path)
        fcntl.flock(lock_f, fcntl.LOCK_UN)


def run(cfg: dict) -> None:
    start = time.time()
    stage_name = "stage69_third_triple_speechcoco"

    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cfg.get("speechcoco_cache_root", output_root / "caches" / "speechcoco_av")).resolve()
    paths = extract_speechcoco_av_cache(
        out_dir=cache_root,
        dataset_name=str(cfg.get("speechcoco_dataset", "mteb/SpeechCoco")),
        clap_model_name=str(cfg["clap_model"]),
        clip_backbone_name=str(cfg["clip_backbone"]),
        target_sampling_rate=int(cfg.get("speechcoco_target_sr", 48_000)),
        device=str(cfg["device"]),
        audio_batch_size=int(cfg.get("speechcoco_audio_batch_size", 32)),
        image_batch_size=int(cfg.get("speechcoco_image_batch_size", 32)),
        text_batch_size=int(cfg.get("speechcoco_text_batch_size", 128)),
        train_max_examples=int(cfg.get("speechcoco_train_max_examples", 120_000)),
        phase_b_val_examples=int(cfg.get("speechcoco_phase_b_val_examples", 20_000)),
        eval_test_examples=int(cfg.get("speechcoco_eval_test_examples", 5_000)),
        sample_seed=int(cfg.get("speechcoco_sample_seed", 69001)),
        eval_seed=int(cfg.get("speechcoco_eval_seed", 69002)),
        phase_b_split_seed=int(cfg.get("phase_b_split_seed", 69003)),
        strict_disjoint=bool(cfg.get("strict_disjoint", True)),
        hf_cache_dir=str(
            cfg.get(
                "speechcoco_hf_cache_dir",
                output_root / "caches" / "hf_speechcoco",
            )
        ),
        force_rebuild=bool(cfg.get("speechcoco_force_rebuild", False)),
    )

    if bool(cfg.get("cache_only", False)):
        done = {
            "stage": stage_name,
            "status": "cache_only_complete",
            "cache_root": str(cache_root),
            "elapsed_sec": float(time.time() - start),
        }
        save_json(done, stage_root / f"{stage_name}_cache_only.json")
        mark_done(markers / f"{stage_name}.done.json", done)
        print(f"{stage_name} cache-only complete")
        return

    cache = AudioCapsAVCache.from_paths(paths["image"], paths["audio"], paths["text"], paths["meta"])
    speech_meta = load_json(paths["meta"])

    coco_cache_dir = Path(cfg["cache_root"]).resolve() / "coco"
    coco = KarpathyCache.from_paths(
        coco_cache_dir / str(cfg["image_cache_file"]),
        coco_cache_dir / str(cfg["text_cache_file"]),
        coco_cache_dir / "metadata.json",
    )
    coco_image_ids = _load_coco_image_ids(coco_cache_dir / "metadata.json")

    it_train_loader, it_val_loader, at_train_loader, at_val_loader, disjoint_diag = _build_loaders(
        cfg,
        coco,
        coco_image_ids,
        cache,
        speech_meta,
    )

    run_units_cfg = cfg.get("run_units")
    if run_units_cfg:
        run_units = [
            (int(u["embed_dim"]), str(u["method"]), int(u["seed"]))
            for u in run_units_cfg
        ]
        methods = sorted({m for _, m, _ in run_units})
        embed_dims = sorted({m for m, _, _ in run_units})
        seeds = sorted({s for _, _, s in run_units})
    else:
        methods = [str(x) for x in cfg.get("methods", ["modular_shared_jl"])]
        seeds = [int(s) for s in cfg["seeds"]]
        embed_dims = [int(m) for m in cfg["embed_dims"]]
        run_units = [(m, method, seed) for m in embed_dims for seed in seeds for method in methods]

    baseline_method = str(cfg.get("baseline_method", methods[0]))

    local = {
        "stage": stage_name,
        "methods_requested": methods,
        "embed_dims_requested": embed_dims,
        "seeds_requested": seeds,
        "baseline_method": baseline_method,
        "disjoint_diagnostics": disjoint_diag,
        "raw": {},
        "stats": {},
    }

    for m, method, seed in run_units:
        m_key = f"m{m}"
        local["raw"].setdefault(m_key, {})
        local["raw"][m_key].setdefault(method, [])
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
            disjoint_diag=disjoint_diag,
        )
        local["raw"][m_key][method].append(rec)
        print(
            f"{stage_name} {m_key} {method} seed={seed} "
            f"it={rec['av_it_avg_R']:.4f} at={rec['av_at_avg_R']:.4f} ia={rec['av_ia_avg_R']:.4f}"
        )

    for m_key in sorted(local["raw"].keys(), key=lambda x: int(x[1:])):
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
        extra={"stage": stage_name, "elapsed_sec": time.time() - start, "disjoint_diagnostics": disjoint_diag},
    )
    save_json(provenance, stage_root / f"provenance_{stage_name}.json")
    mark_done(markers / f"{stage_name}.done.json", {"elapsed_sec": time.time() - start, "disjoint_diagnostics": disjoint_diag})
    print(f"{stage_name} complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

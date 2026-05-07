from __future__ import annotations

import argparse
import fcntl
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data import AudioCapsAVCache, KarpathyCache
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report
from .run_stage29_cc3m_phaseA_modular import _build_model, _encode_batches


def _load_best(model, ckpt_dir: Path, device: str):
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True), strict=True)
    return model.to(device).eval()


def _find_ckpt(stage_root: Path, m: int, method: str, seed: int) -> Path:
    base = stage_root / f"m{m}" / method / f"seed{seed}"
    cands = [base / "phase_b" / "best.pt", base / "joint" / "best.pt", base / "best.pt"]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"No checkpoint found for m={m} method={method} seed={seed} under {stage_root}")


def _margin_stats(sims: torch.Tensor) -> dict[str, float]:
    n = sims.shape[0]
    pos = sims.diag()
    mask = torch.eye(n, device=sims.device, dtype=torch.bool)
    neg = sims.masked_fill(mask, float("-inf")).max(dim=1).values
    margin = pos - neg
    return {
        "pos_mean": float(pos.mean().item()),
        "hardneg_mean": float(neg.mean().item()),
        "margin_mean": float(margin.mean().item()),
        "margin_std": float(margin.std(unbiased=True).item() if n > 1 else 0.0),
        "margin_positive_fraction": float((margin > 0).float().mean().item()),
    }


def _merge_seed_rows(dst_rows: list[dict[str, Any]], src_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed = {int(r["seed"]): r for r in dst_rows if "seed" in r}
    for r in src_rows:
        by_seed[int(r["seed"])] = r
    return [by_seed[s] for s in sorted(by_seed)]


def _merge_results_existing(out_path: Path, incoming: dict[str, Any], metrics: list[str], baseline: str) -> dict[str, Any]:
    if not out_path.exists():
        return incoming
    try:
        existing = load_json(out_path)
    except Exception:
        return incoming
    if existing.get("stage") != incoming.get("stage"):
        return incoming

    merged = dict(existing)
    raw_old = existing.get("raw", {}) if isinstance(existing.get("raw", {}), dict) else {}
    raw_new = incoming.get("raw", {}) if isinstance(incoming.get("raw", {}), dict) else {}
    raw_merged: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for m_key in sorted(set(raw_old.keys()) | set(raw_new.keys()), key=lambda k: int(str(k).lstrip("m"))):
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
        b = baseline if baseline in methods else next(iter(methods))
        stats[m_key] = build_metric_report(methods, metrics=metrics, baseline_method=b)
    merged["stats"] = stats
    return merged


def _save_results_shard_safe(out_path: Path, incoming: dict[str, Any], metrics: list[str], baseline: str) -> None:
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        merged = _merge_results_existing(out_path, incoming, metrics, baseline)
        save_json(merged, out_path)
        fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


@torch.no_grad()
def _eval_one(
    model,
    coco: KarpathyCache,
    av: AudioCapsAVCache,
    *,
    coco_split: str,
    av_split: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    img_c, txt_c, gt_i2t, gt_t2i = coco.eval_tensors(coco_split)
    zi_c = _encode_batches(model.encode_image, img_c, device=device, batch_size=batch_size)
    zt_c = _encode_batches(model.encode_text, txt_c, device=device, batch_size=batch_size)
    coco_it = recall_at_k(zi_c, zt_c, gt_i2t=gt_i2t, gt_t2i=gt_t2i)

    img_a, aud_a, txt_a = av.eval_tensors(av_split)
    zi = _encode_batches(model.encode_image, img_a, device=device, batch_size=batch_size)
    za = _encode_batches(model.encode_audio, aud_a, device=device, batch_size=batch_size)
    zt = _encode_batches(model.encode_text, txt_a, device=device, batch_size=batch_size)

    met_it = recall_at_k(zi, zt)
    met_at = recall_at_k(za, zt)
    met_ia = recall_at_k(zi, za)

    # Modality gaps in embedding space.
    gap_it = float((zi.mean(dim=0) - zt.mean(dim=0)).norm().item())
    gap_at = float((za.mean(dim=0) - zt.mean(dim=0)).norm().item())
    gap_ia = float((zi.mean(dim=0) - za.mean(dim=0)).norm().item())

    sims_ia = zi @ za.T
    sims_it = zi @ zt.T
    sims_at = za @ zt.T

    ia = _margin_stats(sims_ia)
    ai = _margin_stats(sims_ia.T)
    it = _margin_stats(sims_it)
    at = _margin_stats(sims_at)

    out = {
        "coco_avg_R": float(coco_it["avg_R"]),
        "av_it_avg_R": float(met_it["avg_R"]),
        "av_at_avg_R": float(met_at["avg_R"]),
        "av_ia_avg_R": float(met_ia["avg_R"]),
        "combined_avg_R": float((float(coco_it["avg_R"]) + float(met_at["avg_R"]) + float(met_ia["avg_R"])) / 3.0),
        "gap_it_l2": gap_it,
        "gap_at_l2": gap_at,
        "gap_ia_l2": gap_ia,
        "i2a_margin_mean": ia["margin_mean"],
        "i2a_margin_std": ia["margin_std"],
        "i2a_margin_positive_fraction": ia["margin_positive_fraction"],
        "a2i_margin_mean": ai["margin_mean"],
        "a2i_margin_std": ai["margin_std"],
        "a2i_margin_positive_fraction": ai["margin_positive_fraction"],
        "i2t_margin_mean": it["margin_mean"],
        "a2t_margin_mean": at["margin_mean"],
    }
    return out


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage39_modality_gap_linear_vs_jl"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    source_stage_root = Path(cfg["source_stage_root"]).resolve()

    coco_cache_dir = Path(cfg["cache_root"]) / "coco"
    coco = KarpathyCache.from_paths(
        coco_cache_dir / cfg["image_cache_file"],
        coco_cache_dir / cfg["text_cache_file"],
        coco_cache_dir / "metadata.json",
    )
    av_dir = Path(cfg["av_cache_root"]).resolve()
    av = AudioCapsAVCache.from_paths(
        av_dir / "image_feats_clip_raw.pt",
        av_dir / "audio_feats_clap_raw.pt",
        av_dir / "text_feats_clip_raw.pt",
        av_dir / "metadata.json",
    )

    methods = list(cfg.get("methods", ["modular_shared_jl", "audio_linear_probe"]))
    dims = [int(x) for x in cfg.get("embed_dims", [64, 128, 256, 512])]
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]
    baseline = str(cfg.get("baseline_method", methods[0]))

    results = {"stage": "stage39_modality_gap_linear_vs_jl", "raw": {}, "stats": {}}

    metrics = [
        "combined_avg_R",
        "coco_avg_R",
        "av_it_avg_R",
        "av_at_avg_R",
        "av_ia_avg_R",
        "gap_it_l2",
        "gap_at_l2",
        "gap_ia_l2",
        "i2a_margin_mean",
        "i2a_margin_positive_fraction",
        "a2i_margin_mean",
        "a2i_margin_positive_fraction",
    ]

    for m in dims:
        m_key = f"m{m}"
        results["raw"][m_key] = {k: [] for k in methods}
        for seed in seeds:
            set_seed(seed)
            for method in methods:
                seed_dir = stage_root / m_key / method / f"seed{seed}"
                eval_p = seed_dir / "eval.json"
                if eval_p.exists():
                    rec = load_json(eval_p)
                    results["raw"][m_key][method].append(rec)
                    continue

                model = _build_model(method, m, cfg)
                ckpt = _find_ckpt(source_stage_root, m, method, seed)
                model = _load_best(model, ckpt.parent, str(cfg.get("device", "cuda")))

                met = _eval_one(
                    model,
                    coco,
                    av,
                    coco_split=str(cfg.get("coco_test_split", "test")),
                    av_split=str(cfg.get("av_test_split", "test")),
                    device=str(cfg.get("device", "cuda")),
                    batch_size=int(cfg.get("eval_batch_size", 4096)),
                )
                rec = {"seed": seed, "method": method, "embed_dim": m, **met}
                save_json(rec, eval_p)
                results["raw"][m_key][method].append(rec)
                print(
                    f"[stage39] {m_key} {method} seed={seed} "
                    f"av_ia={rec['av_ia_avg_R']:.4f} gap_ia={rec['gap_ia_l2']:.4f} "
                    f"i2a_margin={rec['i2a_margin_mean']:.4f}"
                )

        results["stats"][m_key] = build_metric_report(results["raw"][m_key], metrics=metrics, baseline_method=baseline)

    _save_results_shard_safe(
        stage_root / "stage39_modality_gap_linear_vs_jl_results.json",
        results,
        metrics=metrics,
        baseline=baseline,
    )
    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": "stage39_modality_gap_linear_vs_jl",
            "elapsed_sec": float(time.time() - start),
        },
    )
    save_json(provenance, stage_root / "provenance_stage39.json")
    mark_done(markers / "stage39_modality_gap_linear_vs_jl.done.json", {"elapsed_sec": float(time.time() - start)})
    print("Stage39 complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

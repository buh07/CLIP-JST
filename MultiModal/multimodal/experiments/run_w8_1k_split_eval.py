"""W8: 1K-split evaluation on existing Stage 30 and Stage 44 checkpoints.

The standard AudioCaps test pool has 4411 items (883 unique YouTube clips × 5 captions).
Reviewers may question whether results degrade when we cap to the conventional
883-clip × 1-caption evaluation (the '1K split').

This script:
  - Loads existing Stage 30 (full-supervision) and Stage 44 (zero-shot baseline) checkpoints.
  - Identifies the 883-item 1K subset: first occurrence of each unique youtube_id in test split.
  - Runs retrieval evaluation on both the full 4411-item pool and the 883-item 1K split.
  - Saves comparison to results/reviewer_fixes_suite/w8_1k_split_eval/.

No training; inference only.  Device: GPU specified via CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data import AudioCapsAVCache
from ..eval.retrieval import recall_at_k
from .run_stage29_cc3m_phaseA_modular import _build_model, _encode_batches
from .run_stage39_modality_gap_linear_vs_jl import _find_ckpt


_MM_ROOT = Path(__file__).resolve().parents[2]  # MultiModal/

_STAGE30_ROOT = (
    _MM_ROOT
    / "results/domain_gap_closure_suite/stage30_modular_vs_nonmodular"
)
_STAGE44_ROOT = (
    _MM_ROOT
    / "results/experimental_fixes_suite/stage44_zero_shot_baseline_control"
)
_AV_CACHE_DIR = (
    _MM_ROOT
    / "results/modular_transitivity_followup/caches/audiocaps_av_clap_htsat_unfused"
)
_OUTPUT_DIR = (
    _MM_ROOT
    / "results/reviewer_fixes_suite/w8_1k_split_eval"
)


def _1k_indices(av: AudioCapsAVCache) -> list[int]:
    """Return the first global index for each unique youtube_id in the test split."""
    test_idxs = av.split_to_indices["test"]
    seen: set[str] = set()
    out: list[int] = []
    for i in test_idxs:
        ytid = av.youtube_ids[i]
        if ytid not in seen:
            seen.add(ytid)
            out.append(i)
    out.sort()
    return out


@torch.no_grad()
def _eval_pool(
    model,
    av: AudioCapsAVCache,
    pool_indices: list[int],
    device: str,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate retrieval on a given index pool (all 5 captions per clip)."""
    img = av.image_feats[pool_indices]
    aud = av.audio_feats[pool_indices]
    txt = av.text_feats[pool_indices]
    zi = _encode_batches(model.encode_image, img, device=device, batch_size=batch_size)
    za = _encode_batches(model.encode_audio, aud, device=device, batch_size=batch_size)
    zt = _encode_batches(model.encode_text, txt, device=device, batch_size=batch_size)
    met_at = recall_at_k(za, zt)
    met_ia = recall_at_k(zi, za)
    met_it = recall_at_k(zi, zt)
    return {
        "av_at_avg_R": float(met_at["avg_R"]),
        "av_ia_avg_R": float(met_ia["avg_R"]),
        "av_it_avg_R": float(met_it["avg_R"]),
        "n_items": len(pool_indices),
        **{f"at_{k}": float(v) for k, v in met_at.items() if k != "avg_R"},
        **{f"ia_{k}": float(v) for k, v in met_ia.items() if k != "avg_R"},
    }


@torch.no_grad()
def _eval_raw_cosine(
    av: AudioCapsAVCache,
    pool_indices: list[int],
) -> dict[str, float]:
    """Raw cosine baseline: no projection; CLAP audio vs CLIP text in their native dims."""
    aud = av.audio_feats[pool_indices]
    txt = av.text_feats[pool_indices]
    # Normalize
    aud_n = torch.nn.functional.normalize(aud.float(), dim=-1)
    txt_n = torch.nn.functional.normalize(txt.float(), dim=-1)
    # Pad/project to same dim if needed (just zero-pad smaller to bigger)
    d_a, d_t = aud_n.shape[-1], txt_n.shape[-1]
    if d_a < d_t:
        aud_n = torch.cat([aud_n, torch.zeros(len(aud_n), d_t - d_a)], dim=-1)
    elif d_t < d_a:
        txt_n = torch.cat([txt_n, torch.zeros(len(txt_n), d_a - d_t)], dim=-1)
    met_at = recall_at_k(aud_n, txt_n)
    img = av.image_feats[pool_indices]
    img_n = torch.nn.functional.normalize(img.float(), dim=-1)
    d_i = img_n.shape[-1]
    if d_i < d_a:
        img_n = torch.cat([img_n, torch.zeros(len(img_n), d_a - d_i)], dim=-1)
    elif d_a < d_i:
        aud_n2 = torch.cat([aud_n[:, :d_a], torch.zeros(len(aud_n), d_i - d_a)], dim=-1)
        img_n2 = img_n
        met_ia = recall_at_k(img_n2, aud_n2)
    else:
        met_ia = recall_at_k(img_n, aud_n)
    return {
        "av_at_avg_R": float(met_at["avg_R"]),
        "av_ia_avg_R": float(met_ia["avg_R"]),
        "n_items": len(pool_indices),
    }


def _run_stage(
    stage_root: Path,
    stage_name: str,
    methods: list[str],
    dims: list[int],
    seeds: list[int],
    av: AudioCapsAVCache,
    full_test_idxs: list[int],
    k1_idxs: list[int],
    cfg_common: dict,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    raw: dict[str, dict[str, list[dict]]] = {}
    for m in dims:
        m_key = f"m{m}"
        raw[m_key] = {meth: [] for meth in methods}
        for seed in seeds:
            set_seed(seed)
            for method in methods:
                if method == "raw_cosine_baseline":
                    # No checkpoint needed
                    full = _eval_raw_cosine(av, full_test_idxs)
                    k1 = _eval_raw_cosine(av, k1_idxs)
                    rec = {
                        "method": method,
                        "embed_dim": m,
                        "seed": seed,
                        "full_pool": full,
                        "k1_split": k1,
                    }
                    raw[m_key][method].append(rec)
                    continue

                try:
                    ckpt = _find_ckpt(stage_root, m, method, seed)
                except FileNotFoundError as e:
                    print(f"  SKIP {stage_name} m={m} {method} seed={seed}: {e}")
                    continue

                model = _build_model(method, m, cfg_common)
                model.load_state_dict(
                    torch.load(ckpt, map_location=device, weights_only=True),
                    strict=True,
                )
                model = model.to(device).eval()
                print(f"  Evaluating {stage_name} m={m} {method} seed={seed} ...")

                full_met = _eval_pool(model, av, full_test_idxs, device, batch_size)
                k1_met = _eval_pool(model, av, k1_idxs, device, batch_size)

                rec = {
                    "method": method,
                    "embed_dim": m,
                    "seed": seed,
                    "full_pool": full_met,
                    "k1_split": k1_met,
                }
                raw[m_key][method].append(rec)
                del model
                torch.cuda.empty_cache()
    return raw


def run(cfg: dict) -> None:
    t0 = time.time()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = str(cfg.get("device", "cuda"))
    batch_size = int(cfg.get("eval_batch_size", 2048))

    av = AudioCapsAVCache.from_paths(
        _AV_CACHE_DIR / "image_feats_clip_raw.pt",
        _AV_CACHE_DIR / "audio_feats_clap_raw.pt",
        _AV_CACHE_DIR / "text_feats_clip_raw.pt",
        _AV_CACHE_DIR / "metadata.json",
    )

    full_test_idxs = av.split_to_indices["test"]
    k1_idxs = _1k_indices(av)
    print(f"Full test pool: {len(full_test_idxs)} items")
    print(f"1K split: {len(k1_idxs)} unique clips")

    cfg_common = {
        "vision_dim": int(cfg.get("vision_dim", 768)),
        "audio_dim": int(cfg.get("audio_dim", 512)),
        "text_dim": int(cfg.get("text_dim", 512)),
        "shared_raw_dim": int(cfg.get("shared_raw_dim", 768)),
        "jl_eps": float(cfg.get("jl_eps", 0.1)),
        "jl_seed": int(cfg.get("jl_seed", 42)),
        "text_lora_rank": int(cfg.get("text_lora_rank", 16)),
        "text_lora_alpha": float(cfg.get("text_lora_alpha", 1.0)),
    }

    dims = [int(x) for x in cfg.get("embed_dims", [64, 128, 256, 512])]
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]
    methods_s30 = list(cfg.get("methods_stage30", ["modular_shared_jl", "audio_linear_probe", "audio_text_lora_proxy"]))
    methods_s44 = list(cfg.get("methods_stage44", ["modular_shared_jl", "audio_linear_probe", "raw_cosine_baseline"]))

    print("\n=== Stage 30 (full supervised training) ===")
    stage30_raw = _run_stage(
        _STAGE30_ROOT, "stage30",
        methods_s30, dims, seeds,
        av, full_test_idxs, k1_idxs,
        cfg_common, device, batch_size,
    )

    print("\n=== Stage 44 (zero-shot baseline / WavCaps-trained) ===")
    stage44_raw = _run_stage(
        _STAGE44_ROOT, "stage44",
        methods_s44, dims, seeds,
        av, full_test_idxs, k1_idxs,
        cfg_common, device, batch_size,
    )

    result = {
        "stage": "w8_1k_split_eval",
        "description": "Compare full 4411-item pool vs 883-item 1K split evaluation",
        "n_full_pool": len(full_test_idxs),
        "n_1k_split": len(k1_idxs),
        "stage30": stage30_raw,
        "stage44": stage44_raw,
        "elapsed_sec": time.time() - t0,
    }

    suffix = str(cfg.get("output_suffix", ""))
    fname = f"w8_1k_split_eval_results{suffix}.json"
    out_path = _OUTPUT_DIR / fname
    save_json(result, out_path)
    print(f"\nSaved → {out_path}")

    # Summary table
    print("\n=== Summary: full pool vs 1K split ===")
    for stage_name, stage_raw in [("stage30", stage30_raw), ("stage44", stage44_raw)]:
        for m_key, method_dict in stage_raw.items():
            for method, seed_list in method_dict.items():
                if not seed_list:
                    continue
                av_at_full = [r["full_pool"]["av_at_avg_R"] for r in seed_list]
                av_ia_full = [r["full_pool"]["av_ia_avg_R"] for r in seed_list]
                av_at_k1 = [r["k1_split"]["av_at_avg_R"] for r in seed_list]
                av_ia_k1 = [r["k1_split"]["av_ia_avg_R"] for r in seed_list]
                import statistics
                print(
                    f"  {stage_name} {m_key} {method}: "
                    f"av_at full={statistics.mean(av_at_full):.4f}, 1K={statistics.mean(av_at_k1):.4f} | "
                    f"av_ia full={statistics.mean(av_ia_full):.4f}, 1K={statistics.mean(av_ia_k1):.4f}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg: dict = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    run(cfg)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import re
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json
from ..data.datasets import AudioCapsAVCache


def _extract_caption(ex: dict[str, Any]) -> str:
    for key in ["caption", "text", "description", "prompt", "title"]:
        v = ex.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    conv = ex.get("conversations")
    if isinstance(conv, list):
        for item in conv:
            if isinstance(item, dict):
                v = item.get("value")
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return ""


def _text_metrics(captions: list[str]) -> dict[str, float]:
    tokens = []
    lengths = []
    multi_choice_hits = 0
    acoustic_hits = 0
    acoustic_lex = {
        "music", "speech", "voice", "dog", "cat", "bird", "rain", "water",
        "wind", "car", "engine", "horn", "siren", "laugh", "applause",
        "crowd", "drum", "piano", "guitar", "noise", "sound",
    }
    for c in captions:
        toks = re.findall(r"[a-zA-Z0-9']+", c.lower())
        tokens.extend(toks)
        lengths.append(len(toks))
        cc = c.lower()
        if ("a)" in cc and "b)" in cc) or ("option" in cc and "choose" in cc):
            multi_choice_hits += 1
        if any(w in toks for w in acoustic_lex):
            acoustic_hits += 1

    uniq = len(set(tokens))
    total = max(1, len(tokens))
    arr = np.asarray(lengths, dtype=float) if lengths else np.asarray([0.0], dtype=float)
    return {
        "n_captions": float(len(captions)),
        "mean_tokens": float(arr.mean()),
        "std_tokens": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "median_tokens": float(np.median(arr)),
        "type_token_ratio": float(uniq / total),
        "multi_choice_fraction": float(multi_choice_hits / max(1, len(captions))),
        "acoustic_keyword_fraction": float(acoustic_hits / max(1, len(captions))),
    }


def _clap_text_features(
    model,
    tokenizer,
    texts: list[str],
    *,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if hasattr(x, "text_embeds") and x.text_embeds is not None:
            return x.text_embeds
        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            return x.pooler_output
        if isinstance(x, (tuple, list)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
            return x[0]
        raise RuntimeError(f"Unsupported CLAP text output type: {type(x)}")

    outs: list[torch.Tensor] = []
    for s in range(0, len(texts), batch_size):
        e = min(s + batch_size, len(texts))
        toks = tokenizer(texts[s:e], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            z = model.get_text_features(**toks)
        outs.append(_to_tensor(z).cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty(0, 512)


def _paired_similarity_metrics(audio_feats: torch.Tensor, text_feats: torch.Tensor, seed: int) -> dict[str, float]:
    if len(audio_feats) != len(text_feats):
        raise ValueError("audio/text length mismatch")
    n = len(audio_feats)
    if n == 0:
        return {
            "n": 0.0,
            "pos_mean": 0.0,
            "pos_std": 0.0,
            "neg_mean": 0.0,
            "neg_std": 0.0,
            "margin_mean": 0.0,
            "margin_std": 0.0,
            "margin_p25": 0.0,
            "margin_p50": 0.0,
            "margin_p75": 0.0,
            "margin_positive_fraction": 0.0,
        }
    a = F.normalize(audio_feats.float(), dim=-1)
    t = F.normalize(text_feats.float(), dim=-1)
    pos = (a * t).sum(dim=-1)

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    neg = (a * t[perm]).sum(dim=-1)
    margin = pos - neg
    q = torch.quantile(margin, torch.tensor([0.25, 0.5, 0.75]))
    return {
        "n": float(n),
        "pos_mean": float(pos.mean().item()),
        "pos_std": float(pos.std(unbiased=True).item() if n > 1 else 0.0),
        "neg_mean": float(neg.mean().item()),
        "neg_std": float(neg.std(unbiased=True).item() if n > 1 else 0.0),
        "margin_mean": float(margin.mean().item()),
        "margin_std": float(margin.std(unbiased=True).item() if n > 1 else 0.0),
        "margin_p25": float(q[0].item()),
        "margin_p50": float(q[1].item()),
        "margin_p75": float(q[2].item()),
        "margin_positive_fraction": float((margin > 0).float().mean().item()),
    }


def _sample_indices(indices: list[int], *, max_n: int | None, seed: int) -> list[int]:
    if max_n is None or max_n >= len(indices):
        return list(indices)
    g = np.random.default_rng(seed)
    pick = g.choice(np.asarray(indices), size=max_n, replace=False)
    return [int(x) for x in sorted(pick.tolist())]


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage38_phaseb_quality_analysis"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer, ClapModel

    device = str(cfg.get("device", "cuda"))
    clap_model_name = str(cfg.get("clap_model", "laion/clap-htsat-unfused"))
    seed = int(cfg.get("seed", 2026))
    text_bs = int(cfg.get("text_batch_size", 256))
    clean_source_name = str(cfg.get("wavcaps_clean_source", "WavCaps/WavCaps"))

    # -------- AudioCaps (phase-B baseline data) --------
    av_dir = Path(cfg["audiocaps_av_cache_root"]).resolve()
    av_cache = AudioCapsAVCache.from_paths(
        av_dir / "image_feats_clip_raw.pt",
        av_dir / "audio_feats_clap_raw.pt",
        av_dir / "text_feats_clip_raw.pt",
        av_dir / "metadata.json",
    )
    av_meta = load_json(av_dir / "metadata.json")
    av_caps_all = [str(x) for x in av_meta.get("captions", [])]
    av_train_idx = av_cache.split_indices(str(cfg.get("audiocaps_split", "train")))
    av_idx = _sample_indices(av_train_idx, max_n=cfg.get("audiocaps_max_samples"), seed=seed)
    av_audio = av_cache.audio_feats[av_idx]
    av_caps = [av_caps_all[i] for i in av_idx]

    # -------- WavCaps (phase-B scaling data) --------
    wav_root = Path(cfg["wavcaps_cache_root"]).resolve()
    wav_audio_all = torch.load(wav_root / "audio_feats_clap_raw.pt", map_location="cpu", weights_only=True)
    wav_meta = load_json(wav_root / "metadata.json")
    wav_indices = [int(i) for i in wav_meta.get("split_to_indices", {}).get("train", list(range(len(wav_audio_all))))]
    wav_sources_all = [str(x) for x in wav_meta["sample_sources"]]

    clean_all = [i for i in wav_indices if wav_sources_all[i] == clean_source_name]
    if not clean_all:
        raise RuntimeError(f"No clean-source indices found for source={clean_source_name!r}")

    wav_idx = _sample_indices(wav_indices, max_n=cfg.get("wavcaps_max_samples"), seed=seed + 1)
    wav_clean_idx = _sample_indices(clean_all, max_n=cfg.get("wavcaps_clean_max_samples", cfg.get("wavcaps_max_samples")), seed=seed + 2)

    wav_audio = wav_audio_all[wav_idx]
    wav_clean_audio = wav_audio_all[wav_clean_idx]
    wav_sample_ids = [str(wav_meta["sample_ids"][i]) for i in wav_idx]
    wav_sample_sources = [str(wav_meta["sample_sources"][i]) for i in wav_idx]
    wav_clean_ids = [str(wav_meta["sample_ids"][i]) for i in wav_clean_idx]

    # Build caption map for sampled WavCaps ids from tar shard JSON metadata
    # to avoid torchcodec/datasets audio decode paths.
    target_ids = set(wav_sample_ids) | set(wav_clean_ids)
    wav_cap_map: dict[str, str] = {}
    wav_dataset = str(cfg.get("wavcaps_dataset", "humanify/AS-WavCaps"))
    api = HfApi()
    tar_files = sorted([f for f in api.list_repo_files(wav_dataset, repo_type="dataset") if f.endswith(".tar")])
    for tar_name in tar_files:
        tar_path = hf_hub_download(repo_id=wav_dataset, repo_type="dataset", filename=tar_name)
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".json"):
                    continue
                fobj = tf.extractfile(m)
                if fobj is None:
                    continue
                try:
                    import json
                    ex = json.loads(fobj.read().decode("utf-8"))
                except Exception:
                    continue
                sid = str(ex.get("id", ""))
                if sid in target_ids and sid not in wav_cap_map:
                    wav_cap_map[sid] = _extract_caption(ex)
                    if len(wav_cap_map) >= len(target_ids):
                        break
        if len(wav_cap_map) >= len(target_ids):
            break

    wav_caps = [wav_cap_map.get(sid, "") or "an audio clip" for sid in wav_sample_ids]
    wav_clean_caps = [wav_cap_map.get(sid, "") or "an audio clip" for sid in wav_clean_ids]
    matched = sum(1 for sid in wav_sample_ids if sid in wav_cap_map)
    matched_clean = sum(1 for sid in wav_clean_ids if sid in wav_cap_map)

    # -------- CLAP text embeddings for both datasets --------
    clap = ClapModel.from_pretrained(clap_model_name).to(device).eval()
    tok = AutoTokenizer.from_pretrained(clap_model_name)
    av_text = _clap_text_features(clap, tok, av_caps, device=device, batch_size=text_bs)
    wav_text = _clap_text_features(clap, tok, wav_caps, device=device, batch_size=text_bs)
    wav_clean_text = _clap_text_features(clap, tok, wav_clean_caps, device=device, batch_size=text_bs)

    av_sim = _paired_similarity_metrics(av_audio, av_text, seed=seed)
    wav_sim = _paired_similarity_metrics(wav_audio, wav_text, seed=seed)
    wav_clean_sim = _paired_similarity_metrics(wav_clean_audio, wav_clean_text, seed=seed)

    # Source-stratified WavCaps similarity.
    by_source: dict[str, dict[str, float]] = {}
    for src in sorted(set(wav_sample_sources)):
        idx = [i for i, s in enumerate(wav_sample_sources) if s == src]
        if not idx:
            continue
        by_source[src] = _paired_similarity_metrics(wav_audio[idx], wav_text[idx], seed=seed)

    out = {
        "stage": "stage38_phaseb_quality_analysis",
        "seed": seed,
        "clap_model": clap_model_name,
        "audiocaps": {
            "n_selected": int(len(av_idx)),
            "text_metrics": _text_metrics(av_caps),
            "clap_similarity": av_sim,
        },
        "wavcaps": {
            "n_selected": int(len(wav_idx)),
            "n_caption_matched": int(matched),
            "caption_match_fraction": float(matched / max(1, len(wav_idx))),
            "text_metrics": _text_metrics(wav_caps),
            "clap_similarity": wav_sim,
            "clap_similarity_by_source": by_source,
        },
        "wavcaps_clean_source": {
            "source_name": clean_source_name,
            "n_selected": int(len(wav_clean_idx)),
            "n_caption_matched": int(matched_clean),
            "caption_match_fraction": float(matched_clean / max(1, len(wav_clean_idx))),
            "text_metrics": _text_metrics(wav_clean_caps),
            "clap_similarity": wav_clean_sim,
        },
        "recommended_primary_comparison": {
            "wavcaps_side": "wavcaps_clean_source",
            "audiocaps_side": "audiocaps",
        },
        "secondary_comparison": {
            "wavcaps_side": "wavcaps",
            "audiocaps_side": "audiocaps",
        },
        "delta_wav_minus_audio": {
            "margin_mean": float(wav_sim["margin_mean"] - av_sim["margin_mean"]),
            "pos_mean": float(wav_sim["pos_mean"] - av_sim["pos_mean"]),
            "margin_positive_fraction": float(wav_sim["margin_positive_fraction"] - av_sim["margin_positive_fraction"]),
        },
        "delta_wav_clean_minus_audio": {
            "margin_mean": float(wav_clean_sim["margin_mean"] - av_sim["margin_mean"]),
            "pos_mean": float(wav_clean_sim["pos_mean"] - av_sim["pos_mean"]),
            "margin_positive_fraction": float(wav_clean_sim["margin_positive_fraction"] - av_sim["margin_positive_fraction"]),
        },
        "elapsed_sec": float(time.time() - start),
    }

    save_json(out, stage_root / "stage38_phaseb_quality_analysis_results.json")
    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[seed],
        extra={
            "stage": "stage38_phaseb_quality_analysis",
            "elapsed_sec": float(time.time() - start),
        },
    )
    save_json(provenance, stage_root / "provenance_stage38.json")
    mark_done(markers / "stage38_phaseb_quality_analysis.done.json", {"elapsed_sec": float(time.time() - start)})
    print("Stage38 complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

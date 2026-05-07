"""W7: ImageBind zero-shot baseline on 883-item 1K split.

Stage 37 already ran ImageBind on the full 4,411-item pool; its results are read
from the Stage 37 JSON.  This script supplements those results by running ImageBind
on the 883-item 1K split (first occurrence of each unique youtube_id in the test
split), for direct comparison with W8.

AudioCLIP cannot be installed in this environment (git-LFS issue with its weights
repository), so ImageBind-Huge serves as the jointly-trained image-audio-text baseline.

Media files (audio MP3 + thumbnail JPEG) are read directly from the Stage 37 media
directory by scanning filenames for youtube_id matches — no re-downloading needed.
"""
from __future__ import annotations

import argparse
import io
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from ..common import load_json, save_json
from ..data import AudioCapsAVCache
from ..eval.retrieval import recall_at_k


_MM_ROOT = Path(__file__).resolve().parents[2]  # MultiModal/

_AV_CACHE_DIR = (
    _MM_ROOT
    / "results/modular_transitivity_followup/caches/audiocaps_av_clap_htsat_unfused"
)
_STAGE37_ROOT = (
    _MM_ROOT
    / "results/neurips_strengthen_suite/stage37_imagebind_comparison"
)
_STAGE37_RESULTS = _STAGE37_ROOT / "stage37_imagebind_comparison_results.json"
_STAGE37_AUDIO_DIR = _STAGE37_ROOT / "media" / "test" / "audio"
_STAGE37_IMAGE_DIR = _STAGE37_ROOT / "media" / "test" / "images"
_OUTPUT_DIR = (
    _MM_ROOT
    / "results/reviewer_fixes_suite/w7_imagebind_1k_split"
)


def _build_s37_media_index() -> dict[str, dict[str, str]]:
    """Scan Stage 37 media dir and return {youtube_id: {'audio': path, 'image': path}}."""
    index: dict[str, dict[str, str]] = {}
    for f in _STAGE37_AUDIO_DIR.glob("*.mp3"):
        # Filename: {pos:06d}_{youtube_id}.mp3
        yid = f.stem.split("_", 1)[1]
        if yid not in index:
            index[yid] = {}
        index[yid]["audio"] = str(f)
    for f in _STAGE37_IMAGE_DIR.glob("*.jpg"):
        yid = f.stem.split("_", 1)[1]
        if yid not in index:
            index[yid] = {}
        index[yid]["image"] = str(f)
    return index


def _1k_indices(av: AudioCapsAVCache) -> list[int]:
    """Return global index of first occurrence of each unique youtube_id in test."""
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


def _fetch_thumbnail(youtube_id: str, out_path: Path, timeout_sec: float) -> bool:
    """Download YouTube thumbnail to out_path. Returns True if successful."""
    from PIL import Image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return True
    url = f"https://img.youtube.com/vi/{youtube_id}/0.jpg"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.save(out_path, format="JPEG")
        return True
    except Exception:
        Image.new("RGB", (224, 224), (127, 127, 127)).save(out_path, format="JPEG")
        return False


def run(cfg: dict) -> None:
    t0 = time.time()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = str(cfg.get("device", "cuda"))
    batch_size = int(cfg.get("batch_size", 16))
    timeout_sec = float(cfg.get("thumbnail_timeout_sec", 8.0))
    audio_sample_rate = int(cfg.get("audio_sample_rate", 16_000))
    clips_per_item = int(cfg.get("audio_clips_per_item", 3))
    clip_duration = float(cfg.get("audio_clip_duration_sec", 2.0))

    # Load AV cache metadata
    av = AudioCapsAVCache.from_paths(
        _AV_CACHE_DIR / "image_feats_clip_raw.pt",
        _AV_CACHE_DIR / "audio_feats_clap_raw.pt",
        _AV_CACHE_DIR / "text_feats_clip_raw.pt",
        _AV_CACHE_DIR / "metadata.json",
    )
    k1_idxs = _1k_indices(av)
    print(f"1K split: {len(k1_idxs)} unique clips")

    # Build Stage 37 media index: {youtube_id: {'audio': path, 'image': path}}
    print("Scanning Stage 37 media directory...")
    s37_index = _build_s37_media_index()
    print(f"Stage 37 media index: {len(s37_index)} youtube_ids with media")
    coverage = sum(
        1 for gi in k1_idxs
        if av.youtube_ids[gi] in s37_index
        and "audio" in s37_index[av.youtube_ids[gi]]
        and "image" in s37_index[av.youtube_ids[gi]]
    )
    print(f"1K items with full Stage 37 media: {coverage}/{len(k1_idxs)}")

    # For items missing from Stage 37, download thumbnails and get audio from HF
    missing_yids = [
        av.youtube_ids[gi] for gi in k1_idxs
        if av.youtube_ids[gi] not in s37_index
        or "audio" not in s37_index.get(av.youtube_ids[gi], {})
        or "image" not in s37_index.get(av.youtube_ids[gi], {})
    ]
    if missing_yids:
        print(f"Fetching {len(missing_yids)} missing items from HuggingFace...")
        fallback_dir = _OUTPUT_DIR / "media_fallback"
        fallback_dir.mkdir(parents=True, exist_ok=True)

        from datasets import Audio as HFAudio, load_dataset
        ds = load_dataset("JackyHoCL/AudioCaps-mp3", split="test")
        ds = ds.cast_column("audio", HFAudio(sampling_rate=None, decode=False))
        missing_set = set(missing_yids)
        for row in ds:
            yid = str(row["youtube_id"])
            if yid not in missing_set:
                continue
            aud_p = fallback_dir / f"{yid}.mp3"
            img_p = fallback_dir / f"{yid}.jpg"
            if not aud_p.exists():
                blob = row.get("audio", {}) or {}
                audio_bytes = blob.get("bytes")
                if audio_bytes:
                    aud_p.write_bytes(audio_bytes)
            if not img_p.exists():
                _fetch_thumbnail(yid, img_p, timeout_sec)
            if yid not in s37_index:
                s37_index[yid] = {}
            s37_index[yid]["audio"] = str(aud_p)
            s37_index[yid]["image"] = str(img_p)
            missing_set.discard(yid)
            if not missing_set:
                break

    # Build path lists for 1K split
    image_paths: list[str] = []
    audio_paths: list[str] = []
    captions_list: list[str] = []
    valid_indices: list[int] = []
    all_captions = load_json(_AV_CACHE_DIR / "metadata.json").get("captions", [])

    for gi in k1_idxs:
        yid = av.youtube_ids[gi]
        media = s37_index.get(yid, {})
        aud_p = media.get("audio")
        img_p = media.get("image")
        if not aud_p or not img_p:
            print(f"  WARNING: no media for {yid}, skipping")
            continue
        cap = all_captions[gi] if gi < len(all_captions) else "an audio clip"
        image_paths.append(str(img_p))
        audio_paths.append(str(aud_p))
        captions_list.append(cap if cap else "an audio clip")
        valid_indices.append(gi)

    print(f"Items to encode: {len(image_paths)}/{len(k1_idxs)}")

    # Load ImageBind model
    print("Loading ImageBind-Huge...")
    from imagebind import data as ib_data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    model = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()
    print("ImageBind loaded.")

    # Encode
    all_zi: list[torch.Tensor] = []
    all_za: list[torch.Tensor] = []
    all_zt: list[torch.Tensor] = []

    print(f"\n=== Encoding {len(image_paths)} items ===")
    for s in range(0, len(image_paths), batch_size):
        e = min(s + batch_size, len(image_paths))
        with torch.no_grad():
            inputs = {
                ModalityType.VISION: ib_data.load_and_transform_vision_data(image_paths[s:e], device),
                ModalityType.AUDIO: ib_data.load_and_transform_audio_data(
                    audio_paths[s:e], device,
                    sample_rate=audio_sample_rate,
                    clips_per_video=clips_per_item,
                    clip_duration=clip_duration,
                ),
                ModalityType.TEXT: ib_data.load_and_transform_text(captions_list[s:e], device),
            }
            out = model(inputs)
        all_zi.append(F.normalize(out[ModalityType.VISION], dim=-1).cpu())
        all_za.append(F.normalize(out[ModalityType.AUDIO], dim=-1).cpu())
        all_zt.append(F.normalize(out[ModalityType.TEXT], dim=-1).cpu())
        if e % 100 == 0 or e == len(image_paths):
            print(f"  encoded {e}/{len(image_paths)}")

    zi = torch.cat(all_zi)
    za = torch.cat(all_za)
    zt = torch.cat(all_zt)

    m_it = recall_at_k(zi, zt)
    m_at = recall_at_k(za, zt)
    m_ia = recall_at_k(zi, za)

    k1_met = {
        "av_it_avg_R": float(m_it["avg_R"]),
        "av_at_avg_R": float(m_at["avg_R"]),
        "av_ia_avg_R": float(m_ia["avg_R"]),
        "n_items": len(zi),
        **{f"it_{k}": float(v) for k, v in m_it.items() if k != "avg_R"},
        **{f"at_{k}": float(v) for k, v in m_at.items() if k != "avg_R"},
        **{f"ia_{k}": float(v) for k, v in m_ia.items() if k != "avg_R"},
    }

    print(f"\n1K split: av_at={k1_met['av_at_avg_R']:.4f}, av_ia={k1_met['av_ia_avg_R']:.4f}, av_it={k1_met['av_it_avg_R']:.4f}")

    # Read Stage 37 results for full-pool reference
    full_pool_ref: dict = {}
    if _STAGE37_RESULTS.exists():
        s37 = load_json(_STAGE37_RESULTS)
        full_pool_ref = {
            "source": "stage37_imagebind_comparison_results.json",
            "n_examples": s37.get("n_examples"),
            "av_at_avg_R": s37.get("audio_text", {}).get("avg_R"),
            "av_ia_avg_R": s37.get("image_audio", {}).get("avg_R"),
            "av_it_avg_R": s37.get("image_text", {}).get("avg_R"),
        }
        print(f"Full pool (Stage 37): av_at={full_pool_ref['av_at_avg_R']:.4f}, av_ia={full_pool_ref['av_ia_avg_R']:.4f}")

    result = {
        "stage": "w7_imagebind_1k_split",
        "description": (
            "ImageBind-Huge zero-shot baseline on 883-item 1K split. "
            "Full pool (4411 items) reference from Stage 37 results JSON. "
            "AudioCLIP not available (git-LFS install failure); ImageBind-Huge serves "
            "as jointly-trained image-audio-text baseline."
        ),
        "model": "imagebind_huge",
        "n_1k_split": len(zi),
        "k1_split": k1_met,
        "full_pool_reference": full_pool_ref,
        "elapsed_sec": time.time() - t0,
    }

    out_path = _OUTPUT_DIR / "w7_imagebind_1k_split_results.json"
    save_json(result, out_path)
    print(f"\nSaved → {out_path}")


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

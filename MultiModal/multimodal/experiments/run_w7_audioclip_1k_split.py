from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms as tvt

from ..common import load_json, save_json
from ..data import AudioCapsAVCache
from ..eval.retrieval import recall_at_k


def _load_mono_audio(path: str, target_sr: int) -> np.ndarray:
    """
    Load audio without librosa to avoid numba/coverage import issues.
    Prefers torchaudio, falls back to soundfile+scipy when needed.
    Returns float32 mono waveform in [-1, 1]-like range.
    """
    try:
        import torchaudio  # type: ignore

        wav, sr = torchaudio.load(path)
        if wav.ndim == 2:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != int(target_sr):
            wav = torchaudio.functional.resample(wav, int(sr), int(target_sr))
        return wav.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        import soundfile as sf  # type: ignore
        from scipy.signal import resample_poly  # type: ignore

        wav, sr = sf.read(path, always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if int(sr) != int(target_sr):
            g = np.gcd(int(sr), int(target_sr))
            up = int(target_sr) // g
            down = int(sr) // g
            wav = resample_poly(wav, up, down).astype(np.float32, copy=False)
        return wav


def _first_unique_test_indices(av: AudioCapsAVCache) -> list[int]:
    test_idxs = av.split_to_indices["test"]
    seen: set[str] = set()
    out: list[int] = []
    for i in test_idxs:
        yid = str(av.youtube_ids[i])
        if yid not in seen:
            seen.add(yid)
            out.append(i)
    out.sort()
    return out


def _media_index(audio_dir: Path, image_dir: Path) -> dict[str, dict[str, str]]:
    idx: dict[str, dict[str, str]] = {}
    for p in audio_dir.glob("*.mp3"):
        yid = p.stem.split("_", 1)[1]
        idx.setdefault(yid, {})["audio"] = str(p)
    for p in image_dir.glob("*.jpg"):
        yid = p.stem.split("_", 1)[1]
        idx.setdefault(yid, {})["image"] = str(p)
    return idx


def _chunks(n: int, bs: int):
    for i in range(0, n, bs):
        yield i, min(i + bs, n)


def run(cfg: dict) -> None:
    t0 = time.time()
    device = str(cfg.get("device", "cuda"))
    batch_size = int(cfg.get("batch_size", 16))
    sample_rate = int(cfg.get("audio_sample_rate", 44_100))
    max_items = cfg.get("max_items")

    mm_root = Path(cfg["project_root"]).resolve() / "MultiModal"
    out_dir = Path(cfg["output_root"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    av_cache_dir = Path(cfg["av_cache_root"]).resolve()
    media_root = Path(cfg["media_root"]).resolve()
    audio_dir = media_root / "audio"
    image_dir = media_root / "images"
    audioclip_repo = Path(cfg["audioclip_repo_root"]).resolve()
    audioclip_ckpt = Path(cfg["audioclip_checkpoint"]).resolve()

    av = AudioCapsAVCache.from_paths(
        av_cache_dir / "image_feats_clip_raw.pt",
        av_cache_dir / "audio_feats_clap_raw.pt",
        av_cache_dir / "text_feats_clip_raw.pt",
        av_cache_dir / "metadata.json",
    )
    meta = load_json(av_cache_dir / "metadata.json")
    captions = [str(c) for c in meta.get("captions", [])]

    k1_idxs = _first_unique_test_indices(av)
    idx = _media_index(audio_dir, image_dir)

    valid: list[int] = []
    audio_paths: list[str] = []
    image_paths: list[str] = []
    text_prompts: list[list[str]] = []
    for gi in k1_idxs:
        yid = str(av.youtube_ids[gi])
        media = idx.get(yid, {})
        ap = media.get("audio")
        ip = media.get("image")
        if not ap or not ip:
            continue
        valid.append(int(gi))
        audio_paths.append(ap)
        image_paths.append(ip)
        cap = captions[gi] if gi < len(captions) and captions[gi] else "an audio clip"
        text_prompts.append([cap])
        if max_items is not None and len(valid) >= int(max_items):
            break

    if not valid:
        raise RuntimeError("No overlapping media found for AudioCLIP 1K split evaluation")

    # Load AudioCLIP from local repo checkout.
    if str(audioclip_repo) not in sys.path:
        sys.path.insert(0, str(audioclip_repo))
    from model import AudioCLIP  # type: ignore

    model = AudioCLIP(pretrained=str(audioclip_ckpt)).to(device).eval()

    img_tx = tvt.Compose(
        [
            tvt.Resize(224, interpolation=tvt.InterpolationMode.BICUBIC),
            tvt.CenterCrop(224),
            tvt.ToTensor(),
            tvt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )

    all_za: list[torch.Tensor] = []
    all_zi: list[torch.Tensor] = []
    all_zt: list[torch.Tensor] = []

    with torch.no_grad():
        for s, e in _chunks(len(valid), batch_size):
            batch_audio = []
            for p in audio_paths[s:e]:
                wav = _load_mono_audio(p, target_sr=sample_rate)
                # AudioCLIP training pipeline expects PCM-like magnitude range.
                wav = wav * 32767.0
                batch_audio.append(torch.from_numpy(wav).float())
            max_len = max(int(x.shape[0]) for x in batch_audio)
            audio_t = torch.stack(
                [F.pad(x, (0, max_len - int(x.shape[0])), mode="constant", value=0.0) for x in batch_audio],
                dim=0,
            ).to(device)

            batch_images = []
            for p in image_paths[s:e]:
                im = Image.open(p).convert("RGB")
                batch_images.append(img_tx(im))
            image_t = torch.stack(batch_images, dim=0).to(device)

            texts = text_prompts[s:e]
            ((za, _, _), _), _ = model(audio=audio_t)
            ((_, zi, _), _), _ = model(image=image_t)
            ((_, _, zt), _), _ = model(text=texts)

            all_za.append(F.normalize(za, dim=-1).cpu())
            all_zi.append(F.normalize(zi, dim=-1).cpu())
            all_zt.append(F.normalize(zt, dim=-1).cpu())

            if e % 100 == 0 or e == len(valid):
                print(f"AudioCLIP encode: {e}/{len(valid)}")

    za = torch.cat(all_za, dim=0)
    zi = torch.cat(all_zi, dim=0)
    zt = torch.cat(all_zt, dim=0)

    at = recall_at_k(za, zt)
    ia = recall_at_k(zi, za)
    it = recall_at_k(zi, zt)

    out = {
        "stage": "w7_audioclip_1k_split",
        "description": "AudioCLIP baseline on AudioCaps 1K split media overlap (883 items).",
        "audioclip_repo_root": str(audioclip_repo),
        "audioclip_checkpoint": str(audioclip_ckpt),
        "media_root": str(media_root),
        "n_items": int(len(valid)),
        "av_at_avg_R": float(at["avg_R"]),
        "av_ia_avg_R": float(ia["avg_R"]),
        "av_it_avg_R": float(it["avg_R"]),
        "audio_text": at,
        "image_audio": ia,
        "image_text": it,
        "elapsed_sec": float(time.time() - t0),
    }
    save_json(out, out_dir / "w7_audioclip_1k_split_results.json")
    print(f"W7 AudioCLIP 1K complete: n={len(valid)} av_at={out['av_at_avg_R']:.4f} av_ia={out['av_ia_avg_R']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)


if __name__ == "__main__":
    # Avoid tokenizer parallelism warning spam.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

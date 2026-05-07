from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..common import load_json, save_json
from .audiocaps import _decode_audio_with_ffmpeg, _resample_audio


def _fetch_youtube_thumbnail(
    youtube_id: str,
    *,
    timeout_sec: float,
    retries: int,
    backoff_sec: float,
) -> tuple[Image.Image, bool, str]:
    url = f"https://img.youtube.com/vi/{youtube_id}/0.jpg"
    last_err = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return img, True, "ok"
        except Exception as e:  # network + decode failures
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(backoff_sec * (attempt + 1))

    # Deterministic placeholder image when thumbnail is unavailable.
    placeholder = Image.new("RGB", (224, 224), (127, 127, 127))
    status = f"thumbnail_unavailable: {last_err[:200]}"
    return placeholder, False, status


def extract_audiocaps_av_cache(
    *,
    out_dir: Path,
    dataset_name: str,
    clap_model_name: str,
    clip_backbone_name: str,
    device: str,
    audio_batch_size: int,
    image_batch_size: int,
    text_batch_size: int,
    target_sampling_rate: int = 48_000,
    max_examples_per_split: int | None = None,
    thumbnail_timeout_sec: float = 10.0,
    thumbnail_retries: int = 2,
    thumbnail_backoff_sec: float = 1.0,
    reuse_image_text_from_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Build paired image-audio-text cache for AudioCaps.

    - Image: YouTube thumbnail (0.jpg) from each AudioCaps `youtube_id`
    - Audio: CLAP audio branch raw features
    - Text: CLIP text raw features
    """
    from datasets import Audio, load_dataset
    from transformers import AutoProcessor, CLIPModel, CLIPProcessor, ClapModel

    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / "image_feats_clip_raw.pt"
    audio_out = out_dir / "audio_feats_clap_raw.pt"
    text_out = out_dir / "text_feats_clip_raw.pt"
    meta_out = out_dir / "metadata.json"

    if img_out.exists() and audio_out.exists() and text_out.exists() and meta_out.exists():
        try:
            meta = load_json(meta_out)
            expected = {
                "dataset": dataset_name,
                "audio_encoder": clap_model_name,
                "image_text_encoder": clip_backbone_name,
                "target_sampling_rate": int(target_sampling_rate),
            }
            mismatches = {
                k: {"expected": v, "found": meta.get(k)}
                for k, v in expected.items()
                if meta.get(k) != v
            }
            if not mismatches:
                print(f"AudioCaps AV cache hit at {out_dir}")
                return {"image": img_out, "audio": audio_out, "text": text_out, "meta": meta_out}
            print(f"AudioCaps AV cache metadata mismatch; rebuilding cache: {mismatches}")
        except Exception as e:
            print(f"AudioCaps AV cache metadata unreadable; rebuilding cache: {type(e).__name__}: {e}")

    clap_model = ClapModel.from_pretrained(clap_model_name).to(device).eval()
    clap_processor = AutoProcessor.from_pretrained(clap_model_name)

    clip_model = CLIPModel.from_pretrained(clip_backbone_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_backbone_name)

    reuse_img_feats: torch.Tensor | None = None
    reuse_text_feats: torch.Tensor | None = None
    reuse_meta: dict[str, Any] | None = None
    if reuse_image_text_from_dir is not None:
        src_dir = Path(reuse_image_text_from_dir)
        src_img = src_dir / "image_feats_clip_raw.pt"
        src_txt = src_dir / "text_feats_clip_raw.pt"
        src_meta_p = src_dir / "metadata.json"
        if not (src_img.exists() and src_txt.exists() and src_meta_p.exists()):
            raise FileNotFoundError(
                f"reuse_image_text_from_dir missing required files at {src_dir}"
            )
        reuse_meta = load_json(src_meta_p)
        expected_reuse = {
            "dataset": dataset_name,
            "image_text_encoder": clip_backbone_name,
            "target_sampling_rate": int(target_sampling_rate),
        }
        reuse_mismatches = {
            k: {"expected": v, "found": reuse_meta.get(k)}
            for k, v in expected_reuse.items()
            if reuse_meta.get(k) != v
        }
        if reuse_mismatches:
            raise RuntimeError(f"Reuse cache metadata mismatch: {reuse_mismatches}")
        reuse_img_feats = torch.load(src_img, map_location="cpu", weights_only=True)
        reuse_text_feats = torch.load(src_txt, map_location="cpu", weights_only=True)
        if len(reuse_img_feats) != len(reuse_text_feats):
            raise RuntimeError("Reuse image/text cache length mismatch")
        print(f"Reusing image/text cache from {src_dir}")

    reuse_mode = reuse_img_feats is not None and reuse_text_feats is not None and reuse_meta is not None

    all_img_feats: list[torch.Tensor] = []
    all_audio_feats: list[torch.Tensor] = []
    all_text_feats: list[torch.Tensor] = []

    split_to_indices: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    youtube_ids: list[str] = []
    captions: list[str] = []
    thumbnail_ok: list[bool] = []
    thumbnail_status: list[str] = []

    next_idx = 0
    expected_yids: list[str] | None = None
    expected_caps: list[str] | None = None
    expected_thumb_ok: list[bool] | None = None
    expected_thumb_status: list[str] | None = None
    if reuse_mode:
        expected_yids = [str(x) for x in reuse_meta.get("youtube_ids", [])]
        expected_caps = [str(x).strip() for x in reuse_meta.get("captions", [])]
        expected_thumb_ok = [bool(x) for x in reuse_meta.get("thumbnail_ok", [])]
        expected_thumb_status = [str(x) for x in reuse_meta.get("thumbnail_status", [])]

    def _to_audio_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if hasattr(x, "audio_embeds") and x.audio_embeds is not None:
            return x.audio_embeds
        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            return x.pooler_output
        if isinstance(x, (tuple, list)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
            return x[0]
        raise RuntimeError(f"Unsupported CLAP audio output type: {type(x)}")

    batch_images: list[Image.Image] = []
    batch_audio: list[np.ndarray] = []
    batch_text: list[str] = []
    batch_youtube_ids: list[str] = []
    batch_thumbnail_ok: list[bool] = []
    batch_thumbnail_status: list[str] = []

    def flush_batch(split: str) -> None:
        nonlocal next_idx
        if not batch_audio:
            return

        with torch.no_grad():
            # Audio feats
            a_in = clap_processor(
                audio=batch_audio,
                sampling_rate=target_sampling_rate,
                return_tensors="pt",
                padding=True,
            ).to(device)
            a_out = clap_model.get_audio_features(**a_in)
            a_feat = _to_audio_tensor(a_out).cpu()

            if not reuse_mode:
                # Text feats
                t_in = clip_processor(
                    text=batch_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                t_out = clip_model.text_model(
                    input_ids=t_in["input_ids"],
                    attention_mask=t_in.get("attention_mask"),
                )
                t_feat = t_out.pooler_output.cpu()

                # Image feats
                i_in = clip_processor(images=batch_images, return_tensors="pt").to(device)
                i_out = clip_model.vision_model(pixel_values=i_in["pixel_values"])
                i_feat = i_out.pooler_output.cpu()

        if not reuse_mode and not (len(i_feat) == len(a_feat) == len(t_feat)):
            raise RuntimeError("Batch feature count mismatch in AudioCaps AV extraction")

        if not reuse_mode:
            all_img_feats.append(i_feat)
        all_audio_feats.append(a_feat)
        if not reuse_mode:
            all_text_feats.append(t_feat)

        n = len(batch_audio)
        idxs = list(range(next_idx, next_idx + n))
        split_to_indices[split].extend(idxs)
        next_idx += n

        youtube_ids.extend(batch_youtube_ids)
        captions.extend(batch_text)
        thumbnail_ok.extend(batch_thumbnail_ok)
        thumbnail_status.extend(batch_thumbnail_status)

        batch_images.clear()
        batch_audio.clear()
        batch_text.clear()
        batch_youtube_ids.clear()
        batch_thumbnail_ok.clear()
        batch_thumbnail_status.clear()

    for split in ["train", "validation", "test"]:
        print(f"Preparing AudioCaps AV split={split}")
        ds = load_dataset(dataset_name, split=split)
        ds = ds.cast_column("audio", Audio(sampling_rate=None, decode=False))
        if max_examples_per_split is not None:
            ds = ds.select(range(min(max_examples_per_split, len(ds))))

        for i, ex in enumerate(ds):
            yid = str(ex["youtube_id"])
            cap = str(ex["caption"]).strip()
            exp_idx = next_idx + len(batch_audio)
            if reuse_mode:
                if expected_yids is None or expected_caps is None or expected_thumb_ok is None or expected_thumb_status is None:
                    raise RuntimeError("Reuse metadata arrays are unavailable")
                if exp_idx >= len(expected_yids):
                    raise RuntimeError("Reuse cache has fewer rows than AudioCaps dataset iteration")
                if yid != expected_yids[exp_idx] or cap != expected_caps[exp_idx]:
                    raise RuntimeError(
                        f"Reuse cache ordering mismatch at idx={exp_idx}: "
                        f"dataset=({yid!r}, {cap[:80]!r}) "
                        f"reuse=({expected_yids[exp_idx]!r}, {expected_caps[exp_idx][:80]!r})"
                    )
                ok = expected_thumb_ok[exp_idx]
                status = expected_thumb_status[exp_idx]
            else:
                img, ok, status = _fetch_youtube_thumbnail(
                    yid,
                    timeout_sec=thumbnail_timeout_sec,
                    retries=thumbnail_retries,
                    backoff_sec=thumbnail_backoff_sec,
                )

            wav, sr = _decode_audio_with_ffmpeg(ex["audio"], target_sampling_rate)
            wav48 = _resample_audio(wav, src_sr=sr, tgt_sr=target_sampling_rate)

            if not reuse_mode:
                batch_images.append(img)
            batch_audio.append(wav48)
            batch_text.append(cap)
            batch_youtube_ids.append(yid)
            batch_thumbnail_ok.append(ok)
            batch_thumbnail_status.append(status)

            if len(batch_audio) >= min(audio_batch_size, image_batch_size, text_batch_size):
                flush_batch(split)

            if i > 0 and i % 200 == 0:
                print(
                    f"  {split}: processed {i}/{len(ds)} "
                    f"(thumb_ok={sum(batch_thumbnail_ok)} in current buffer)"
                )

        flush_batch(split)
        print(f"  {split}: total {len(split_to_indices[split])}")

    if reuse_mode:
        img_feats = reuse_img_feats
        text_feats = reuse_text_feats
    else:
        img_feats = torch.cat(all_img_feats)
        text_feats = torch.cat(all_text_feats)
    audio_feats = torch.cat(all_audio_feats)
    if not (len(img_feats) == len(audio_feats) == len(text_feats)):
        raise RuntimeError("Final feature length mismatch in AudioCaps AV extraction")

    torch.save(img_feats, img_out)
    torch.save(audio_feats, audio_out)
    torch.save(text_feats, text_out)

    n_ok = int(sum(1 for x in thumbnail_ok if x))
    n_all = int(len(thumbnail_ok))

    meta: dict[str, Any] = {
        "dataset": dataset_name,
        "protocol": "audiocaps_youtube_thumbnail_av",
        "audio_encoder": clap_model_name,
        "image_text_encoder": clip_backbone_name,
        "target_sampling_rate": int(target_sampling_rate),
        "split_to_indices": split_to_indices,
        "num_pairs": int(len(img_feats)),
        "image_dim": int(img_feats.shape[1]),
        "audio_dim": int(audio_feats.shape[1]),
        "text_dim": int(text_feats.shape[1]),
        "youtube_ids": youtube_ids,
        "captions": captions,
        "thumbnail_ok": thumbnail_ok,
        "thumbnail_status": thumbnail_status,
        "thumbnail_success_count": n_ok,
        "thumbnail_success_rate": float(n_ok / max(1, n_all)),
        "thumbnail_fetch": {
            "timeout_sec": float(thumbnail_timeout_sec),
            "retries": int(thumbnail_retries),
            "backoff_sec": float(thumbnail_backoff_sec),
        },
    }
    save_json(meta, meta_out)
    print(f"Saved AudioCaps AV cache to {out_dir} (thumbnail_ok={n_ok}/{n_all})")

    return {"image": img_out, "audio": audio_out, "text": text_out, "meta": meta_out}

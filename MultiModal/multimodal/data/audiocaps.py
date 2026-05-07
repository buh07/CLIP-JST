from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal

from ..common import save_json


def _resample_audio(wave: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if wave.ndim == 2:
        wave = wave.mean(axis=0)
    if src_sr == tgt_sr:
        return wave.astype(np.float32)
    g = math.gcd(src_sr, tgt_sr)
    up = tgt_sr // g
    down = src_sr // g
    return signal.resample_poly(wave, up, down).astype(np.float32)


def _decode_audio_with_ffmpeg(audio_blob: dict[str, Any], target_sampling_rate: int) -> tuple[np.ndarray, int]:
    """
    Decode an AudioCaps mp3 blob to mono float32 waveform using ffmpeg.

    We intentionally avoid datasets' torchcodec path due runtime incompatibilities
    on this host and decode directly from bytes.
    """
    data = audio_blob.get("bytes")
    path = audio_blob.get("path")
    if data is None:
        if not path:
            raise RuntimeError("Audio blob has neither bytes nor path.")
        with open(path, "rb") as f:
            data = f.read()

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sampling_rate),
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg decode failed (code={proc.returncode}): {err[-500:]}")

    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    if wav.size == 0:
        raise RuntimeError("Decoded waveform is empty.")
    return wav, target_sampling_rate


def extract_audiocaps_clap_cache(
    *,
    out_dir: Path,
    dataset_name: str,
    clap_model_name: str,
    clip_backbone_name: str,
    device: str,
    audio_batch_size: int,
    text_batch_size: int,
    target_sampling_rate: int = 48_000,
    max_examples_per_split: int | None = None,
) -> dict[str, Path]:
    """
    Build audio-text feature cache for AudioCaps.

    Audio encoder: CLAP audio branch.
    Text encoder: CLIP text backbone raw features (shared with image-text pipeline).
    """
    from datasets import Audio, load_dataset
    from transformers import AutoProcessor, CLIPModel, CLIPProcessor, ClapModel

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out = out_dir / "audio_feats_clap_raw.pt"
    text_out = out_dir / "text_feats_clip_raw.pt"
    meta_out = out_dir / "metadata.json"

    if audio_out.exists() and text_out.exists() and meta_out.exists():
        print(f"AudioCaps cache hit at {out_dir}")
        return {"audio": audio_out, "text": text_out, "meta": meta_out}

    clap_model = ClapModel.from_pretrained(clap_model_name).to(device).eval()
    clap_processor = AutoProcessor.from_pretrained(clap_model_name)

    clip_model = CLIPModel.from_pretrained(clip_backbone_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_backbone_name)

    all_audio_feats: list[torch.Tensor] = []
    all_text_feats: list[torch.Tensor] = []
    split_to_indices: dict[str, list[int]] = {"train": [], "validation": [], "test": []}

    next_idx = 0

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

    for split in ["train", "validation", "test"]:
        print(f"Preparing AudioCaps split={split}")
        ds = load_dataset(dataset_name, split=split)
        ds = ds.cast_column("audio", Audio(sampling_rate=None, decode=False))
        if max_examples_per_split is not None:
            ds = ds.select(range(min(max_examples_per_split, len(ds))))

        batch_audio: list[np.ndarray] = []
        batch_text: list[str] = []

        def flush_batch() -> None:
            nonlocal next_idx
            if not batch_audio:
                return
            with torch.no_grad():
                a_in = clap_processor(
                    audio=batch_audio,
                    sampling_rate=target_sampling_rate,
                    return_tensors="pt",
                    padding=True,
                ).to(device)
                a_out = clap_model.get_audio_features(**a_in)
                a_feat = _to_audio_tensor(a_out).cpu()

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

            all_audio_feats.append(a_feat)
            all_text_feats.append(t_feat)
            idxs = list(range(next_idx, next_idx + len(batch_audio)))
            split_to_indices[split].extend(idxs)
            next_idx += len(batch_audio)
            batch_audio.clear()
            batch_text.clear()

        for i, ex in enumerate(ds):
            wav, sr = _decode_audio_with_ffmpeg(ex["audio"], target_sampling_rate)
            wav48 = _resample_audio(wav, src_sr=sr, tgt_sr=target_sampling_rate)
            batch_audio.append(wav48)
            batch_text.append(str(ex["caption"]).strip())

            if len(batch_audio) >= audio_batch_size:
                flush_batch()
            if i % 200 == 0 and i > 0:
                print(f"  {split}: processed {i}/{len(ds)}")

        flush_batch()
        print(f"  {split}: total {len(split_to_indices[split])}")

    audio_feats = torch.cat(all_audio_feats)
    text_feats = torch.cat(all_text_feats)
    if audio_feats.shape[0] != text_feats.shape[0]:
        raise RuntimeError("Audio and text feature counts do not match")

    torch.save(audio_feats, audio_out)
    torch.save(text_feats, text_out)
    meta: dict[str, Any] = {
        "dataset": dataset_name,
        "audio_encoder": clap_model_name,
        "text_encoder": clip_backbone_name,
        "target_sampling_rate": target_sampling_rate,
        "split_to_indices": split_to_indices,
        "num_pairs": int(audio_feats.shape[0]),
        "audio_dim": int(audio_feats.shape[1]),
        "text_dim": int(text_feats.shape[1]),
    }
    save_json(meta, meta_out)
    print(f"Saved AudioCaps cache to {out_dir}")
    return {"audio": audio_out, "text": text_out, "meta": meta_out}

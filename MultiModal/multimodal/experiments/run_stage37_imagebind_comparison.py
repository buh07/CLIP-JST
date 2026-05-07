from __future__ import annotations

import argparse
import io
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from ..common import env_snapshot, load_json, mark_done, save_json
from ..eval.retrieval import recall_at_k


def _rename_direction_keys(metrics: dict[str, Any], src_a: str, src_b: str) -> dict[str, Any]:
    out = dict(metrics)
    for k in (1, 5, 10):
        a_key = f"{src_a}2{src_b}_R@{k}"
        b_key = f"{src_b}2{src_a}_R@{k}"
        # Backward-compatible remap from recall_at_k generic names.
        if a_key not in out and f"i2t_R@{k}" in out:
            out[a_key] = out[f"i2t_R@{k}"]
        if b_key not in out and f"t2i_R@{k}" in out:
            out[b_key] = out[f"t2i_R@{k}"]
    return out


def _fetch_thumbnail_to_path(youtube_id: str, out_path: Path, timeout_sec: float) -> tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return True, "cached"
    url = f"https://img.youtube.com/vi/{youtube_id}/0.jpg"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.save(out_path, format="JPEG")
        return True, "ok"
    except Exception as e:
        # Deterministic fallback placeholder.
        Image.new("RGB", (224, 224), (127, 127, 127)).save(out_path, format="JPEG")
        return False, f"placeholder:{type(e).__name__}"


def _audio_blob_to_path(blob: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    data = blob.get("bytes")
    if data is not None:
        out_path.write_bytes(data)
        return
    path = blob.get("path")
    if path:
        p = Path(path)
        if p.exists():
            shutil.copyfile(p, out_path)
            return
    raise RuntimeError("Unsupported audio blob: missing bytes/path")


def _load_reference_rows(cfg: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in cfg.get("reference_rows", []):
        p = Path(spec["results_path"]).resolve()
        m_key = f"m{int(spec['embed_dim'])}"
        method = str(spec["method"])
        metric = str(spec.get("metric", "av_ia_avg_R"))
        if not p.exists():
            continue
        obj = load_json(p)
        val = (
            obj.get("stats", {})
            .get(m_key, {})
            .get("methods", {})
            .get(method, {})
            .get(metric, {})
            .get("mean")
        )
        if val is None:
            continue
        out[f"{method}:{metric}:{m_key}"] = float(val)
    return out


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage37_imagebind_comparison"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    from datasets import Audio, load_dataset
    from imagebind import data as ib_data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    dataset_name = str(cfg.get("audiocaps_dataset", "JackyHoCL/AudioCaps-mp3"))
    split = str(cfg.get("split", "test"))
    max_examples = cfg.get("max_examples")
    device = str(cfg.get("device", "cuda"))
    batch_size = int(cfg.get("batch_size", 16))
    timeout_sec = float(cfg.get("thumbnail_timeout_sec", 8.0))
    media_root = Path(cfg.get("media_root", stage_root / "media")).resolve()
    audio_root = media_root / split / "audio"
    image_root = media_root / split / "images"

    # Prepare model.
    model = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()

    # Build deterministic local media files for ImageBind loaders.
    ds = load_dataset(dataset_name, split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=None, decode=False))
    if max_examples is not None:
        ds = ds.select(range(min(int(max_examples), len(ds))))

    image_paths: list[str] = []
    audio_paths: list[str] = []
    captions: list[str] = []
    thumb_ok = 0

    for i, ex in enumerate(ds):
        yid = str(ex.get("youtube_id", f"row{i:06d}"))
        caption = str(ex.get("caption", "")).strip() or "an audio clip"

        img_p = image_root / f"{i:06d}_{yid}.jpg"
        aud_p = audio_root / f"{i:06d}_{yid}.mp3"

        ok, _ = _fetch_thumbnail_to_path(yid, img_p, timeout_sec=timeout_sec)
        thumb_ok += int(ok)
        _audio_blob_to_path(ex["audio"], aud_p)

        image_paths.append(str(img_p))
        audio_paths.append(str(aud_p))
        captions.append(caption)

        if i > 0 and i % 500 == 0:
            print(f"[stage37] prepared media {i}/{len(ds)}")

    # Encode all three modalities.
    all_i: list[torch.Tensor] = []
    all_a: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    for s in range(0, len(image_paths), batch_size):
        e = min(s + batch_size, len(image_paths))
        batch_imgs = image_paths[s:e]
        batch_auds = audio_paths[s:e]
        batch_txt = captions[s:e]

        inputs = {
            ModalityType.VISION: ib_data.load_and_transform_vision_data(batch_imgs, device),
            ModalityType.AUDIO: ib_data.load_and_transform_audio_data(
                batch_auds,
                device,
                sample_rate=int(cfg.get("audio_sample_rate", 16_000)),
                clips_per_video=int(cfg.get("audio_clips_per_item", 3)),
                clip_duration=float(cfg.get("audio_clip_duration_sec", 2.0)),
            ),
            ModalityType.TEXT: ib_data.load_and_transform_text(batch_txt, device),
        }
        with torch.no_grad():
            out = model(inputs)
        zi = F.normalize(out[ModalityType.VISION], dim=-1).cpu()
        za = F.normalize(out[ModalityType.AUDIO], dim=-1).cpu()
        zt = F.normalize(out[ModalityType.TEXT], dim=-1).cpu()

        all_i.append(zi)
        all_a.append(za)
        all_t.append(zt)
        print(f"[stage37] encoded {e}/{len(image_paths)}")

    zi = torch.cat(all_i, dim=0)
    za = torch.cat(all_a, dim=0)
    zt = torch.cat(all_t, dim=0)

    m_it = _rename_direction_keys(recall_at_k(zi, zt), "i", "t")
    m_at = _rename_direction_keys(recall_at_k(za, zt), "a", "t")
    m_ia = _rename_direction_keys(recall_at_k(zi, za), "i", "a")
    chance_p1 = 1.0 / float(len(zi))

    res = {
        "stage": "stage37_imagebind_comparison",
        "dataset": dataset_name,
        "split": split,
        "n_examples": int(len(zi)),
        "chance_p1": float(chance_p1),
        "thumbnail_ok_fraction": float(thumb_ok / max(1, len(zi))),
        "image_text": m_it,
        "audio_text": m_at,
        "image_audio": m_ia,
        "avg_three_avg_R": float((m_it["avg_R"] + m_at["avg_R"] + m_ia["avg_R"]) / 3.0),
        "elapsed_sec": float(time.time() - start),
    }

    refs = _load_reference_rows(cfg)
    if refs:
        res["references"] = refs
        res["deltas_vs_references"] = {
            k: float(res["image_audio"]["avg_R"] - v) for k, v in refs.items() if "av_ia_avg_R" in k
        }

    save_json(res, stage_root / "stage37_imagebind_comparison_results.json")
    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={
            "stage": "stage37_imagebind_comparison",
            "dataset": dataset_name,
            "split": split,
            "elapsed_sec": float(time.time() - start),
        },
    )
    save_json(provenance, stage_root / "provenance_stage37.json")
    mark_done(markers / "stage37_imagebind_comparison.done.json", {"elapsed_sec": float(time.time() - start)})
    print("Stage37 complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Audio as HFAudio, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, ClapModel

from ..common import env_snapshot, load_json, mark_done, save_json
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report
from .run_stage29_cc3m_phaseA_modular import _build_model, _encode_batches


REPORT_METRICS = ["clotho_at_avg_R"]


def _load_audio_feature(audio_obj: dict[str, Any], target_sr: int) -> list[float]:
    """
    Load HF Audio feature without relying on torchcodec decoding.
    Accepts decode=False payloads (bytes/path) and resamples to target_sr.
    """
    import numpy as np
    import soundfile as sf  # type: ignore
    from scipy.signal import resample_poly  # type: ignore

    if not isinstance(audio_obj, dict):
        raise RuntimeError("Unexpected Clotho audio payload type")

    audio_bytes = audio_obj.get("bytes")
    audio_path = audio_obj.get("path")
    if audio_bytes is not None:
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    elif audio_path:
        wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    else:
        raise RuntimeError("Clotho audio sample missing both bytes and path")

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if int(sr) != int(target_sr):
        g = np.gcd(int(sr), int(target_sr))
        up = int(target_sr) // g
        down = int(sr) // g
        wav = resample_poly(wav, up, down).astype(np.float32, copy=False)
    return wav.tolist()


def _extract_clotho_cache(
    *,
    out_dir: Path,
    dataset_name: str,
    split_name: str,
    clap_model_name: str,
    clip_backbone_name: str,
    device: str,
    audio_batch_size: int,
    text_batch_size: int,
    target_sampling_rate: int,
    max_examples: int | None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out = out_dir / "audio_feats_clap_raw.pt"
    text_out = out_dir / "text_feats_clip_raw.pt"
    meta_out = out_dir / "metadata.json"

    if audio_out.exists() and text_out.exists() and meta_out.exists():
        return {"audio": audio_out, "text": text_out, "meta": meta_out}

    ds = load_dataset(dataset_name, split=split_name)
    # decode=False avoids torchcodec dependency failures on this environment.
    ds = ds.cast_column("audio", HFAudio(decode=False))
    rows: list[tuple[dict[str, Any], str]] = []
    for ex in ds:
        cap = str(ex.get("caption_1", "")).strip()
        if not cap:
            continue
        wav = _load_audio_feature(ex["audio"], target_sampling_rate)
        rows.append(({"array": wav}, cap))
        if max_examples is not None and len(rows) >= int(max_examples):
            break

    if not rows:
        raise RuntimeError("No Clotho rows with caption_1 found")

    clap_model = ClapModel.from_pretrained(clap_model_name).to(device).eval()
    clap_processor = AutoProcessor.from_pretrained(clap_model_name)
    clip_model = CLIPModel.from_pretrained(clip_backbone_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_backbone_name)

    aud_feats: list[torch.Tensor] = []
    txt_feats: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, len(rows), audio_batch_size):
            batch = rows[i:i + audio_batch_size]
            wavs = [b[0]["array"] for b in batch]
            a_in = clap_processor(
                audio=wavs,
                sampling_rate=target_sampling_rate,
                return_tensors="pt",
                padding=True,
            ).to(device)
            a_out = clap_model.get_audio_features(**a_in)
            if hasattr(a_out, "audio_embeds"):
                a_out = a_out.audio_embeds
            elif hasattr(a_out, "pooler_output"):
                a_out = a_out.pooler_output
            elif isinstance(a_out, tuple):
                a_out = a_out[0]
            aud_feats.append(a_out.cpu())

        for i in range(0, len(rows), text_batch_size):
            caps = [rows[j][1] for j in range(i, min(i + text_batch_size, len(rows)))]
            t_in = clip_processor(text=caps, return_tensors="pt", padding=True, truncation=True).to(device)
            t_out = clip_model.text_model(
                input_ids=t_in["input_ids"],
                attention_mask=t_in["attention_mask"],
            )
            txt_feats.append(t_out.pooler_output.cpu())

    audio = torch.cat(aud_feats, dim=0)
    text = torch.cat(txt_feats, dim=0)
    if len(audio) != len(text):
        raise RuntimeError("Clotho cache feature length mismatch")

    torch.save(audio, audio_out)
    torch.save(text, text_out)
    save_json(
        {
            "dataset": dataset_name,
            "split": split_name,
            "target_sampling_rate": int(target_sampling_rate),
            "n_examples": int(len(audio)),
            "audio_encoder": clap_model_name,
            "text_encoder": clip_backbone_name,
            "caption_field": "caption_1",
        },
        meta_out,
    )
    return {"audio": audio_out, "text": text_out, "meta": meta_out}


def _find_ckpts_from_roots(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        p = root / "stage55_wavcaps_holdout_retrain" / "stage55_wavcaps_holdout_retrain_results.json"
        if not p.exists():
            continue
        obj = load_json(p)
        cond = str(obj.get("condition_name", "unknown"))
        for m_key, methods in obj.get("raw", {}).items():
            m = int(str(m_key).lstrip("m"))
            for method, recs in methods.items():
                for rec in recs:
                    seed = int(rec["seed"])
                    ckpt = root / "stage55_wavcaps_holdout_retrain" / m_key / method / f"seed{seed}" / "phase_b" / "best.pt"
                    if not ckpt.exists():
                        continue
                    rows.append(
                        {
                            "condition_name": cond,
                            "embed_dim": m,
                            "method": str(method),
                            "seed": seed,
                            "ckpt_path": str(ckpt),
                            "source_eval_path": str(
                                root / "stage55_wavcaps_holdout_retrain" / m_key / method / f"seed{seed}" / "eval.json"
                            ),
                        }
                    )
    dedup: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for r in rows:
        dedup[(r["condition_name"], r["embed_dim"], r["method"], r["seed"])] = r
    return [dedup[k] for k in sorted(dedup.keys())]


def _passes_filter(row: dict[str, Any], cfg: dict) -> bool:
    conds = [str(x) for x in cfg.get("conditions_filter", [])]
    dims = {int(x) for x in cfg.get("embed_dims_filter", [])}
    methods = [str(x) for x in cfg.get("methods_filter", [])]
    seeds = {int(x) for x in cfg.get("seeds_filter", [])}
    if conds and row["condition_name"] not in conds:
        return False
    if dims and int(row["embed_dim"]) not in dims:
        return False
    if methods and row["method"] not in methods:
        return False
    if seeds and int(row["seed"]) not in seeds:
        return False
    return True


def run(cfg: dict) -> None:
    start = time.time()
    stage_name = "stage62_clotho_intermediate_eval"

    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cfg.get("clotho_cache_root", output_root / "caches" / "clotho")).resolve()
    paths = _extract_clotho_cache(
        out_dir=cache_root,
        dataset_name=str(cfg.get("clotho_dataset", "gijs/clotho")),
        split_name=str(cfg.get("clotho_split", "test")),
        clap_model_name=str(cfg["clap_model"]),
        clip_backbone_name=str(cfg["clip_backbone"]),
        device=str(cfg["device"]),
        audio_batch_size=int(cfg.get("clotho_audio_batch_size", 64)),
        text_batch_size=int(cfg.get("clotho_text_batch_size", 256)),
        target_sampling_rate=int(cfg.get("clotho_target_sr", 48_000)),
        max_examples=cfg.get("clotho_max_examples"),
    )
    clotho_audio = torch.load(paths["audio"], map_location="cpu", weights_only=True)
    clotho_text = torch.load(paths["text"], map_location="cpu", weights_only=True)

    roots = [Path(p).resolve() for p in cfg.get("stage55_roots", [])]
    all_rows = _find_ckpts_from_roots(roots)
    rows = [r for r in all_rows if _passes_filter(r, cfg)]
    if not rows:
        raise RuntimeError("No stage55 checkpoints matched stage62 filters")

    batch_size = int(cfg.get("eval_batch_size", 4096))
    device = str(cfg.get("device", "cuda"))

    out = {
        "stage": stage_name,
        "clotho_cache": {"audio": str(paths["audio"]), "text": str(paths["text"]), "meta": str(paths["meta"])},
        "raw": {},
        "stats": {},
    }

    for row in rows:
        cond = str(row["condition_name"])
        m = int(row["embed_dim"])
        method = str(row["method"])
        seed = int(row["seed"])
        m_key = f"m{m}"
        out["raw"].setdefault(cond, {}).setdefault(m_key, {}).setdefault(method, [])

        rec_p = stage_root / cond / m_key / method / f"seed{seed}" / "eval.json"
        if rec_p.exists():
            rec = load_json(rec_p)
            out["raw"][cond][m_key][method].append(rec)
            continue

        model = _build_model(method, m, cfg)
        state = torch.load(Path(row["ckpt_path"]), map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()

        za = _encode_batches(model.encode_audio, clotho_audio, device=device, batch_size=batch_size)
        zt = _encode_batches(model.encode_text, clotho_text, device=device, batch_size=batch_size)
        met = recall_at_k(za, zt)

        rec = {
            "condition_name": cond,
            "embed_dim": m,
            "method": method,
            "seed": seed,
            "source_ckpt_path": row["ckpt_path"],
            "source_eval_path": row["source_eval_path"],
            "clotho_audio_text": met,
            "clotho_at_avg_R": float(met["avg_R"]),
        }
        save_json(rec, rec_p)
        out["raw"][cond][m_key][method].append(rec)
        print(
            f"{stage_name} cond={cond} {m_key} {method} seed={seed} "
            f"clotho_at={rec['clotho_at_avg_R']:.4f}"
        )

    for cond, by_m in out["raw"].items():
        out["stats"][cond] = {}
        for m_key, methods in by_m.items():
            baseline = str(cfg.get("baseline_method", next(iter(methods))))
            out["stats"][cond][m_key] = build_metric_report(
                methods,
                metrics=REPORT_METRICS,
                baseline_method=baseline if baseline in methods else next(iter(methods)),
            )

    save_json(out, stage_root / f"{stage_name}_results.json")
    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=sorted({int(r["seed"]) for r in rows}),
        extra={"stage": stage_name, "elapsed_sec": float(time.time() - start), "n_rows": len(rows)},
    )
    save_json(provenance, stage_root / f"provenance_{stage_name}.json")
    mark_done(markers / f"{stage_name}.done.json", {"elapsed_sec": float(time.time() - start), "n_rows": len(rows)})
    print(f"{stage_name} complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, ClapModel

from ..common import env_snapshot, load_json, mark_done, save_json
from ..data import AudioCapsAVCache, KarpathyCache
from ..eval.stats import build_metric_report
from .run_stage29_cc3m_phaseA_modular import _build_model, _encode_batches


DEFAULT_CATEGORIES = [
    "speech", "music", "dog", "cat", "bird", "car", "engine", "siren", "horn", "footsteps",
    "water", "rain", "wind", "thunder", "crowd", "applause", "door", "keyboard", "phone", "construction",
]


def _load_source_rows(cfg: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dim_filter = {int(m) for m in cfg.get("embed_dims", [])}
    method_filter = set(cfg.get("methods", []))
    seed_filter = {int(s) for s in cfg.get("seeds", [])}

    for src in cfg.get("source_experiments", []):
        source_id = str(src.get("source_id", "source"))
        stage_root = Path(src["stage_root"]).resolve()
        stage_name = str(src.get("stage_name", stage_root.name))
        results_file = str(src.get("results_file", f"{stage_name}_results.json"))
        p = stage_root / results_file
        if not p.exists():
            print(f"[stage34] skip missing source results: {p}")
            continue
        obj = load_json(p)
        raw = obj.get("raw", {})
        for m_key, methods in raw.items():
            m = int(str(m_key)[1:])
            if dim_filter and m not in dim_filter:
                continue
            for method, recs in methods.items():
                if method_filter and method not in method_filter:
                    continue
                for rec in recs:
                    seed = int(rec["seed"])
                    if seed_filter and seed not in seed_filter:
                        continue
                    rows.append(
                        {
                            "source_id": source_id,
                            "stage_root": str(stage_root),
                            "stage_name": stage_name,
                            "embed_dim": m,
                            "method": method,
                            "seed": seed,
                        }
                    )
    seen = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        k = (r["source_id"], r["stage_root"], r["stage_name"], r["embed_dim"], r["method"], r["seed"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _find_checkpoint(stage_root: Path, stage_name: str, m: int, method: str, seed: int) -> Path:
    seed_dir = stage_root / f"m{m}" / method / f"seed{seed}"
    cands = [
        seed_dir / "phase_b" / "best.pt",
        seed_dir / "joint" / "best.pt",
        seed_dir / "phase_a" / "best.pt",
        seed_dir / "best.pt",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"No checkpoint found for {stage_name} {seed_dir}")


@torch.no_grad()
def _classify_audio_categories(av_audio: torch.Tensor, prompts: list[str], clap_model_name: str, device: str) -> torch.Tensor:
    processor = AutoProcessor.from_pretrained(clap_model_name)
    model = ClapModel.from_pretrained(clap_model_name).to(device).eval()
    t_in = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    t = model.get_text_features(**t_in)
    if hasattr(t, "text_embeds") and t.text_embeds is not None:
        t = t.text_embeds
    elif hasattr(t, "pooler_output") and t.pooler_output is not None:
        t = t.pooler_output
    elif isinstance(t, (tuple, list)) and len(t) > 0 and isinstance(t[0], torch.Tensor):
        t = t[0]
    elif not isinstance(t, torch.Tensor):
        raise RuntimeError(f"Unsupported CLAP text output type: {type(t)}")
    t = F.normalize(t, dim=-1).cpu()
    a = F.normalize(av_audio.float(), dim=-1)
    return (a @ t.T).argmax(dim=1)


@torch.no_grad()
def _classify_clip_categories(
    image_raw: torch.Tensor,
    text_raw: torch.Tensor,
    prompts: list[str],
    clip_backbone: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = CLIPModel.from_pretrained(clip_backbone).to(device).eval()
    proc = CLIPProcessor.from_pretrained(clip_backbone)

    t_in = proc(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    t_out = model.text_model(input_ids=t_in["input_ids"], attention_mask=t_in.get("attention_mask"))
    prompt_txt = t_out.pooler_output

    ptxt = F.normalize(model.text_projection(prompt_txt), dim=-1).cpu()
    img_proj = F.normalize(image_raw.float() @ model.visual_projection.weight.T.cpu(), dim=-1)
    txt_proj = F.normalize(text_raw.float() @ model.text_projection.weight.T.cpu(), dim=-1)

    img_cat = (img_proj @ ptxt.T).argmax(dim=1)
    txt_cat = (txt_proj @ ptxt.T).argmax(dim=1)
    return img_cat, txt_cat


@torch.no_grad()
def _topk_category_scores(
    query_z: torch.Tensor,
    corpus_z: torch.Tensor,
    query_cat: torch.Tensor,
    corpus_cat: torch.Tensor,
    k_values: list[int],
) -> dict[str, float]:
    max_k = max(k_values)
    sims = query_z @ corpus_z.T
    topk = sims.topk(k=max_k, dim=1).indices
    out: dict[str, float] = {}
    for k in k_values:
        idx = topk[:, :k]
        pred_cat = corpus_cat[idx]
        match = pred_cat.eq(query_cat.unsqueeze(1))
        out[f"p{k}"] = float(match.float().mean().item())
        out[f"hit{k}"] = float(match.any(dim=1).float().mean().item())
    return out


def _chance_hit(k: int, n_cat: int) -> float:
    if n_cat <= 0:
        return 0.0
    return float(1.0 - math.pow(1.0 - 1.0 / float(n_cat), k))


def _merge_rows(existing_rows: list[dict[str, Any]], incoming_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {}
    for r in existing_rows:
        k = (r["source_id"], int(r["embed_dim"]), r["method"], int(r["seed"]))
        by_key[k] = r
    for r in incoming_rows:
        k = (r["source_id"], int(r["embed_dim"]), r["method"], int(r["seed"]))
        by_key[k] = r
    return [by_key[k] for k in sorted(by_key, key=lambda x: (x[0], x[1], x[2], x[3]))]


def _metric_list(k_values: list[int]) -> list[str]:
    metrics: list[str] = []
    dirs = ["i2a", "a2i", "i2t", "a2t"]
    for d in dirs:
        for k in k_values:
            metrics.append(f"{d}_cat_p{k}")
            metrics.append(f"{d}_cat_hit{k}")
    for k in k_values:
        metrics.append(f"avg_cat_p{k}")
        metrics.append(f"avg_cat_hit{k}")
    return metrics


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage34_semantic_topk"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    rows = _load_source_rows(cfg)
    if not rows:
        raise RuntimeError("No source rows found for stage34.")

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

    categories = list(cfg.get("category_prompts", DEFAULT_CATEGORIES))
    prompts = [f"a photo or sound of {c}" for c in categories]
    k_values = sorted({int(k) for k in cfg.get("topk_values", [1, 5, 10])})
    n_cat = len(categories)

    coco_img, coco_txt, _, _ = coco.eval_tensors(cfg.get("coco_test_split", "test"))
    av_img, av_aud, av_txt = av.eval_tensors(cfg.get("av_test_split", "test"))

    audio_cat = _classify_audio_categories(av_aud, prompts, str(cfg["clap_model"]), str(cfg["device"]))
    coco_img_cat, _ = _classify_clip_categories(coco_img, coco_txt[: len(coco_img)], prompts, str(cfg["clip_backbone"]), str(cfg["device"]))
    _, av_txt_cat = _classify_clip_categories(av_img, av_txt, prompts, str(cfg["clip_backbone"]), str(cfg["device"]))

    out_path = stage_root / "stage34_semantic_topk_results.json"
    existing_rows = []
    if out_path.exists():
        try:
            existing_rows = load_json(out_path).get("rows", [])
        except Exception:
            existing_rows = []
    existing_keys = {
        (r["source_id"], int(r["embed_dim"]), r["method"], int(r["seed"]))
        for r in existing_rows
    }

    out_rows: list[dict[str, Any]] = []
    for r in rows:
        key = (r["source_id"], int(r["embed_dim"]), r["method"], int(r["seed"]))
        if key in existing_keys:
            continue
        stage_root_src = Path(r["stage_root"])
        method = r["method"]
        m = int(r["embed_dim"])
        seed = int(r["seed"])
        source_id = str(r["source_id"])
        stage_name = str(r["stage_name"])

        ckpt = _find_checkpoint(stage_root_src, stage_name, m, method, seed)
        model = _build_model(method, m, cfg)
        model.load_state_dict(torch.load(ckpt, map_location=cfg["device"], weights_only=True), strict=True)
        model = model.to(cfg["device"]).eval()

        zi_coco = _encode_batches(model.encode_image, coco_img, device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
        za_av = _encode_batches(model.encode_audio, av_aud, device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
        zt_av = _encode_batches(model.encode_text, av_txt, device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))

        i2a = _topk_category_scores(zi_coco, za_av, coco_img_cat, audio_cat, k_values)
        a2i = _topk_category_scores(za_av, zi_coco, audio_cat, coco_img_cat, k_values)
        i2t = _topk_category_scores(zi_coco, zt_av, coco_img_cat, av_txt_cat, k_values)
        a2t = _topk_category_scores(za_av, zt_av, audio_cat, av_txt_cat, k_values)

        rec: dict[str, Any] = {
            "source_id": source_id,
            "source_stage": stage_name,
            "source_root": str(stage_root_src),
            "method": method,
            "embed_dim": m,
            "seed": seed,
            "n_categories": n_cat,
            "chance_p1": float(1.0 / float(n_cat)),
            "topk_values": k_values,
        }
        for k in k_values:
            rec[f"chance_hit{k}"] = _chance_hit(k, n_cat)
            rec[f"i2a_cat_p{k}"] = i2a[f"p{k}"]
            rec[f"i2a_cat_hit{k}"] = i2a[f"hit{k}"]
            rec[f"a2i_cat_p{k}"] = a2i[f"p{k}"]
            rec[f"a2i_cat_hit{k}"] = a2i[f"hit{k}"]
            rec[f"i2t_cat_p{k}"] = i2t[f"p{k}"]
            rec[f"i2t_cat_hit{k}"] = i2t[f"hit{k}"]
            rec[f"a2t_cat_p{k}"] = a2t[f"p{k}"]
            rec[f"a2t_cat_hit{k}"] = a2t[f"hit{k}"]
            rec[f"avg_cat_p{k}"] = float(
                (rec[f"i2a_cat_p{k}"] + rec[f"a2i_cat_p{k}"] + rec[f"i2t_cat_p{k}"] + rec[f"a2t_cat_p{k}"]) / 4.0
            )
            rec[f"avg_cat_hit{k}"] = float(
                (rec[f"i2a_cat_hit{k}"] + rec[f"a2i_cat_hit{k}"] + rec[f"i2t_cat_hit{k}"] + rec[f"a2t_cat_hit{k}"]) / 4.0
            )
        out_rows.append(rec)
        print(
            f"stage34 {source_id} m{m} {method} seed={seed} "
            f"avg_p1={rec['avg_cat_p1']:.4f} avg_p5={rec.get('avg_cat_p5', rec['avg_cat_p1']):.4f} "
            f"avg_hit10={rec.get('avg_cat_hit10', rec.get('avg_cat_hit1', 0.0)):.4f}"
        )

    merged_rows = _merge_rows(existing_rows, out_rows)

    metrics = _metric_list(k_values)
    raw: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for rec in merged_rows:
        src = rec["source_id"]
        m_key = f"m{int(rec['embed_dim'])}"
        raw.setdefault(src, {}).setdefault(m_key, {}).setdefault(rec["method"], []).append(rec)

    baseline_method = str(cfg.get("baseline_method", "modular_shared_jl"))
    stats: dict[str, Any] = {}
    for src, by_m in raw.items():
        stats[src] = {}
        for m_key, methods in by_m.items():
            b = baseline_method if baseline_method in methods else next(iter(methods))
            stats[src][m_key] = build_metric_report(methods, metrics=metrics, baseline_method=b)
            stats[src][m_key]["baseline_method"] = b

    out = {
        "stage": "stage34_semantic_topk",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "category_prompts": categories,
        "n_categories": n_cat,
        "topk_values": k_values,
        "rows": merged_rows,
        "raw": raw,
        "stats": stats,
        "elapsed_sec": time.time() - start,
    }
    save_json(out, out_path)

    md = [
        "# Stage34 Semantic Top-k Retrieval",
        "",
        f"- Categories: `{n_cat}`",
        f"- Top-k: `{k_values}`",
        f"- Chance P@1: `{1.0/float(n_cat):.4f}`",
        "",
    ]
    for src in sorted(stats.keys()):
        md.append(f"## {src}")
        for m_key in sorted(stats[src].keys(), key=lambda x: int(x[1:])):
            md.append(f"### {m_key}")
            for method, blk in sorted(stats[src][m_key].get("methods", {}).items()):
                p1 = blk.get("avg_cat_p1", {})
                p5 = blk.get("avg_cat_p5", {})
                h10 = blk.get("avg_cat_hit10", {})
                md.append(
                    f"- {method}: "
                    f"avg_cat_p1={p1.get('mean', 0.0):.4f}±{p1.get('std', 0.0):.4f}, "
                    f"avg_cat_p5={p5.get('mean', 0.0):.4f}±{p5.get('std', 0.0):.4f}, "
                    f"avg_cat_hit10={h10.get('mean', 0.0):.4f}±{h10.get('std', 0.0):.4f}"
                )
            md.append("")
    (stage_root / "stage34_semantic_topk_results.md").write_text("\n".join(md), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[int(s) for s in cfg.get("seeds", [])],
        extra={"stage": "stage34_semantic_topk", "elapsed_sec": time.time() - start},
    )
    save_json(provenance, stage_root / "provenance_stage34.json")
    mark_done(markers / "stage34_semantic_topk.done.json", {"elapsed_sec": time.time() - start})
    print("Stage34 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

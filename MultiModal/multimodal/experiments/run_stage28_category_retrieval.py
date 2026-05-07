from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, ClapModel

from ..common import env_snapshot, load_json, mark_done, save_json
from ..data import AudioCapsAVCache, KarpathyCache
from .run_stage29_cc3m_phaseA_modular import _build_model, _encode_batches


DEFAULT_CATEGORIES = [
    "speech", "music", "dog", "cat", "bird", "car", "engine", "siren", "horn", "footsteps",
    "water", "rain", "wind", "thunder", "crowd", "applause", "door", "keyboard", "phone", "construction",
]


def _load_stage20_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        p = root / "stage20_modular_audio_transitivity" / "stage20_results.json"
        if not p.exists():
            continue
        obj = load_json(p)
        for m_key, methods in obj.get("raw", {}).items():
            m = int(m_key[1:])
            for method, recs in methods.items():
                for rec in recs:
                    rows.append(
                        {
                            "root": str(root),
                            "embed_dim": m,
                            "method": method,
                            "seed": int(rec["seed"]),
                        }
                    )
    # de-dup by (root,m,method,seed)
    seen = set()
    out = []
    for r in rows:
        k = (r["root"], r["embed_dim"], r["method"], r["seed"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _find_checkpoint(root: Path, m: int, method: str, seed: int) -> Path:
    seed_dir = root / "stage20_modular_audio_transitivity" / f"m{m}" / method / f"seed{seed}"
    cands = [
        seed_dir / "phase_b" / "best.pt",
        seed_dir / "joint" / "best.pt",
        seed_dir / "best.pt",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"No checkpoint found for {seed_dir}")


@torch.no_grad()
def _top1_cat_acc(query_z: torch.Tensor, corpus_z: torch.Tensor, query_cat: torch.Tensor, corpus_cat: torch.Tensor) -> float:
    sims = query_z @ corpus_z.T
    top1 = sims.argmax(dim=1)
    pred = corpus_cat[top1]
    return float((pred == query_cat).float().mean().item())


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

    # project into CLIP similarity space
    ptxt = F.normalize(model.text_projection(prompt_txt), dim=-1).cpu()
    img_proj = F.normalize(image_raw.float() @ model.visual_projection.weight.T.cpu(), dim=-1)
    txt_proj = F.normalize(text_raw.float() @ model.text_projection.weight.T.cpu(), dim=-1)

    img_cat = (img_proj @ ptxt.T).argmax(dim=1)
    txt_cat = (txt_proj @ ptxt.T).argmax(dim=1)
    return img_cat, txt_cat


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage28_category_retrieval"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    roots = [Path(p).resolve() for p in cfg.get("stage20_source_roots", [])]
    rows = _load_stage20_rows(roots)

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

    coco_img, coco_txt, _, _ = coco.eval_tensors(cfg.get("coco_test_split", "test"))
    av_img, av_aud, av_txt = av.eval_tensors(cfg.get("av_test_split", "test"))

    audio_cat = _classify_audio_categories(av_aud, prompts, str(cfg["clap_model"]), str(cfg["device"]))
    coco_img_cat, _ = _classify_clip_categories(coco_img, coco_txt[: len(coco_img)], prompts, str(cfg["clip_backbone"]), str(cfg["device"]))
    _, av_txt_cat = _classify_clip_categories(av_img, av_txt, prompts, str(cfg["clip_backbone"]), str(cfg["device"]))

    out_rows = []
    chance = 1.0 / float(len(categories))

    for r in rows:
        root = Path(r["root"])
        method = r["method"]
        m = int(r["embed_dim"])
        seed = int(r["seed"])

        ckpt = _find_checkpoint(root, m, method, seed)
        model = _build_model(method, m, cfg)
        model.load_state_dict(torch.load(ckpt, map_location=cfg["device"], weights_only=True), strict=True)
        model = model.to(cfg["device"]).eval()

        zi_coco = _encode_batches(model.encode_image, coco_img, device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
        za_av = _encode_batches(model.encode_audio, av_aud, device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
        zt_av = _encode_batches(model.encode_text, av_txt, device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))

        rec = {
            "root": str(root),
            "method": method,
            "embed_dim": m,
            "seed": seed,
            "chance_p1": chance,
            "i2a_cat_p1": _top1_cat_acc(zi_coco, za_av, coco_img_cat, audio_cat),
            "a2i_cat_p1": _top1_cat_acc(za_av, zi_coco, audio_cat, coco_img_cat),
            "i2t_cat_p1": _top1_cat_acc(zi_coco, zt_av, coco_img_cat, av_txt_cat),
            "a2t_cat_p1": _top1_cat_acc(za_av, zt_av, audio_cat, av_txt_cat),
        }
        rec["avg_cat_p1"] = float((rec["i2a_cat_p1"] + rec["a2i_cat_p1"] + rec["i2t_cat_p1"] + rec["a2t_cat_p1"]) / 4.0)
        out_rows.append(rec)
        print(
            f"stage28 m{m} {method} seed={seed} i2a={rec['i2a_cat_p1']:.4f} a2i={rec['a2i_cat_p1']:.4f} avg={rec['avg_cat_p1']:.4f}"
        )

    # aggregate
    agg: dict[str, dict[str, list[float]]] = {}
    for rec in out_rows:
        m_key = f"m{rec['embed_dim']}"
        agg.setdefault(m_key, {}).setdefault(rec["method"], [])
        agg[m_key][rec["method"]].append(float(rec["avg_cat_p1"]))

    summary: dict[str, Any] = {}
    for m_key, methods in agg.items():
        summary[m_key] = {}
        for method, vals in methods.items():
            t = torch.tensor(vals, dtype=torch.float32)
            mean = float(t.mean().item()) if len(vals) else 0.0
            std = float(t.std(unbiased=True).item()) if len(vals) > 1 else 0.0
            summary[m_key][method] = {"n": len(vals), "mean": mean, "std": std}

    out = {
        "stage": "stage28_category_retrieval",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "category_prompts": categories,
        "n_categories": len(categories),
        "chance_p1": chance,
        "rows": out_rows,
        "summary": summary,
        "elapsed_sec": time.time() - start,
    }
    save_json(out, stage_root / "stage28_category_retrieval_results.json")

    md = [
        "# Stage28 Category Retrieval",
        "",
        f"- Chance precision@1: `{chance:.4f}`",
        f"- Categories: `{len(categories)}`",
        "",
    ]
    for m_key in sorted(summary.keys(), key=lambda x: int(x[1:])):
        md.append(f"## {m_key}")
        for method, blk in sorted(summary[m_key].items()):
            md.append(f"- {method}: avg_cat_p1={blk['mean']:.4f} ± {blk['std']:.4f} (n={blk['n']})")
        md.append("")
    (stage_root / "stage28_category_retrieval_results.md").write_text("\n".join(md), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={"stage": "stage28_category_retrieval", "elapsed_sec": time.time() - start},
    )
    save_json(provenance, stage_root / "provenance_stage28.json")
    mark_done(markers / "stage28_category_retrieval.done.json", {"elapsed_sec": time.time() - start})
    print("Stage28 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

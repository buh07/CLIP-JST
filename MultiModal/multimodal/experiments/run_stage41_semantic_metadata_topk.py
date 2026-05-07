from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json
from ..data import AudioCapsAVCache, KarpathyCache
from ..eval.stats import build_metric_report
from .run_stage29_cc3m_phaseA_modular import _build_model, _encode_batches

DEFAULT_CATEGORIES = [
    "speech", "music", "dog", "cat", "bird", "car", "engine", "siren", "horn", "footsteps",
    "water", "rain", "wind", "thunder", "crowd", "applause", "door", "keyboard", "phone", "construction",
]

KEYWORDS = {
    "speech": [r"\bspeech\b", r"\btalk(?:ing)?\b", r"\bvoice\b", r"\bconversation\b", r"\bpeople talking\b"],
    "music": [r"\bmusic\b", r"\bsong\b", r"\bsinging\b", r"\bmelody\b", r"\bpiano\b", r"\bguitar\b", r"\bdrum\b"],
    "dog": [r"\bdog\b", r"\bpuppy\b", r"\bbark(?:ing)?\b"],
    "cat": [r"\bcat\b", r"\bkitten\b", r"\bmeow(?:ing)?\b"],
    "bird": [r"\bbird\b", r"\bchirp(?:ing)?\b", r"\btweet(?:ing)?\b", r"\bcrow\b"],
    "car": [r"\bcar\b", r"\btraffic\b", r"\bvehicle\b", r"\btruck\b", r"\bbus\b", r"\bmotorcycle\b", r"\btrain\b", r"\bairplane\b", r"\bboat\b"],
    "engine": [r"\bengine\b", r"\bmotor\b", r"\brev(?:ving)?\b", r"\bidling\b"],
    "siren": [r"\bsiren\b", r"\bemergency\b"],
    "horn": [r"\bhorn\b", r"\bhonk(?:ing)?\b"],
    "footsteps": [r"\bfootstep(?:s)?\b", r"\bwalking\b", r"\bsteps\b"],
    "water": [r"\bwater\b", r"\bstream\b", r"\briver\b", r"\bocean\b", r"\bsplash(?:ing)?\b"],
    "rain": [r"\brain(?:ing)?\b", r"\bdrizzle\b"],
    "wind": [r"\bwind\b", r"\bgust\b", r"\bblow(?:ing)?\b"],
    "thunder": [r"\bthunder\b", r"\blightning\b", r"\bstorm\b"],
    "crowd": [r"\bcrowd\b", r"\bcheer(?:ing)?\b", r"\bpeople\b"],
    "applause": [r"\bapplause\b", r"\bclapp(?:ing)?\b", r"\bclaps?\b"],
    "door": [r"\bdoor\b", r"\bknock(?:ing)?\b", r"\bslam(?:ming)?\b"],
    "keyboard": [r"\bkeyboard\b", r"\btyping\b", r"\bkeys\b"],
    "phone": [r"\bphone\b", r"\bring(?:ing|tone)?\b", r"\bcellphone\b"],
    "construction": [r"\bconstruction\b", r"\bhammer(?:ing)?\b", r"\bdrill(?:ing)?\b", r"\bsaw(?:ing)?\b", r"\bjackhammer\b"],
}


def _compile_patterns(categories: list[str]) -> dict[str, list[re.Pattern[str]]]:
    pats: dict[str, list[re.Pattern[str]]] = {}
    for c in categories:
        rules = KEYWORDS.get(c, [rf"\b{re.escape(c)}\b"])
        pats[c] = [re.compile(r, re.IGNORECASE) for r in rules]
    return pats


def _label_text(text: str, categories: list[str], pats: dict[str, list[re.Pattern[str]]]) -> int:
    hits: Counter[str] = Counter()
    t = text or ""
    for c in categories:
        for p in pats[c]:
            if p.search(t):
                hits[c] += 1
    if not hits:
        return -1
    best = sorted(hits.items(), key=lambda kv: (-kv[1], categories.index(kv[0])))[0][0]
    return categories.index(best)


def _load_coco_captions(ann_root: Path) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for name in ["captions_train2017.json", "captions_val2017.json"]:
        p = ann_root / name
        if not p.exists():
            continue
        obj = load_json(p)
        for ann in obj.get("annotations", []):
            img_id = int(ann["image_id"])
            out.setdefault(img_id, []).append(str(ann.get("caption", "")))
    if not out:
        raise RuntimeError(f"No COCO caption annotations found in {ann_root}")
    return out


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
            print(f"[stage41] skip missing source results: {p}")
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
                    rows.append({
                        "source_id": source_id,
                        "stage_root": str(stage_root),
                        "stage_name": stage_name,
                        "embed_dim": m,
                        "method": method,
                        "seed": seed,
                    })
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
    cands = [seed_dir / "phase_b" / "best.pt", seed_dir / "joint" / "best.pt", seed_dir / "phase_a" / "best.pt", seed_dir / "best.pt"]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"No checkpoint found for {stage_name} {seed_dir}")


def _topk_category_scores(
    query_z: torch.Tensor,
    corpus_z: torch.Tensor,
    query_cat: torch.Tensor,
    corpus_cat: torch.Tensor,
    k_values: list[int],
) -> dict[str, float]:
    # filter labeled rows only
    q_mask = query_cat >= 0
    c_mask = corpus_cat >= 0
    qz = query_z[q_mask]
    qcat = query_cat[q_mask]
    cz = corpus_z[c_mask]
    ccat = corpus_cat[c_mask]
    if len(qz) == 0 or len(cz) == 0:
        out: dict[str, float] = {}
        for k in k_values:
            out[f"p{k}"] = 0.0
            out[f"hit{k}"] = 0.0
        return out

    max_k = min(max(k_values), len(cz))
    sims = qz @ cz.T
    topk = sims.topk(k=max_k, dim=1).indices
    out: dict[str, float] = {}
    for k in k_values:
        kk = min(k, max_k)
        idx = topk[:, :kk]
        pred_cat = ccat[idx]
        match = pred_cat.eq(qcat.unsqueeze(1))
        out[f"p{k}"] = float(match.float().mean().item())
        out[f"hit{k}"] = float(match.any(dim=1).float().mean().item())
    return out


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
    stage_root = output_root / "stage41_semantic_metadata_topk"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    rows = _load_source_rows(cfg)
    if not rows:
        raise RuntimeError("No source rows found for stage41.")

    coco_cache_dir = Path(cfg["cache_root"]) / "coco"
    coco = KarpathyCache.from_paths(
        coco_cache_dir / cfg["image_cache_file"],
        coco_cache_dir / cfg["text_cache_file"],
        coco_cache_dir / "metadata.json",
    )
    coco_meta = load_json(coco_cache_dir / "metadata.json")
    image_ids = [int(x) for x in coco_meta.get("image_ids", [])]

    av_dir = Path(cfg["av_cache_root"]).resolve()
    av = AudioCapsAVCache.from_paths(
        av_dir / "image_feats_clip_raw.pt",
        av_dir / "audio_feats_clap_raw.pt",
        av_dir / "text_feats_clip_raw.pt",
        av_dir / "metadata.json",
    )
    av_meta = load_json(av_dir / "metadata.json")
    av_captions = [str(x) for x in av_meta.get("captions", [])]

    categories = list(cfg.get("category_prompts", DEFAULT_CATEGORIES))
    k_values = sorted({int(k) for k in cfg.get("topk_values", [1, 5, 10])})
    patterns = _compile_patterns(categories)

    # Build metadata-grounded labels (independent from CLIP/CLAP embedding similarity)
    ann_root = Path(cfg["coco_annotations_root"]).resolve()
    coco_caps = _load_coco_captions(ann_root)

    coco_eval_indices = coco.split_indices(cfg.get("coco_test_split", "test"))
    av_eval_indices = av.split_indices(cfg.get("av_test_split", "test"))

    coco_img, _, _, _ = coco.eval_tensors(cfg.get("coco_test_split", "test"))
    _, av_aud, av_txt = av.eval_tensors(cfg.get("av_test_split", "test"))

    coco_labels = []
    for idx in coco_eval_indices:
        img_id = image_ids[idx]
        caps = coco_caps.get(img_id, [])
        label = _label_text(" ".join(caps), categories, patterns)
        coco_labels.append(label)
    coco_img_cat = torch.tensor(coco_labels, dtype=torch.long)

    av_labels = []
    for idx in av_eval_indices:
        cap = av_captions[idx] if idx < len(av_captions) else ""
        av_labels.append(_label_text(cap, categories, patterns))
    av_cat = torch.tensor(av_labels, dtype=torch.long)
    av_txt_cat = av_cat.clone()

    out_path = stage_root / "stage41_semantic_metadata_topk_results.json"
    existing_rows = []
    if out_path.exists():
        try:
            existing_rows = load_json(out_path).get("rows", [])
        except Exception:
            existing_rows = []
    existing_keys = {(r["source_id"], int(r["embed_dim"]), r["method"], int(r["seed"])) for r in existing_rows}

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

        i2a = _topk_category_scores(zi_coco, za_av, coco_img_cat, av_cat, k_values)
        a2i = _topk_category_scores(za_av, zi_coco, av_cat, coco_img_cat, k_values)
        i2t = _topk_category_scores(zi_coco, zt_av, coco_img_cat, av_txt_cat, k_values)
        a2t = _topk_category_scores(za_av, zt_av, av_cat, av_txt_cat, k_values)

        rec: dict[str, Any] = {
            "source_id": source_id,
            "embed_dim": m,
            "method": method,
            "seed": seed,
            "labeling": "metadata_caption_keyword",
            "n_query_labeled_i": int((coco_img_cat >= 0).sum().item()),
            "n_query_labeled_a": int((av_cat >= 0).sum().item()),
        }
        for k in k_values:
            rec[f"i2a_cat_p{k}"] = i2a[f"p{k}"]
            rec[f"i2a_cat_hit{k}"] = i2a[f"hit{k}"]
            rec[f"a2i_cat_p{k}"] = a2i[f"p{k}"]
            rec[f"a2i_cat_hit{k}"] = a2i[f"hit{k}"]
            rec[f"i2t_cat_p{k}"] = i2t[f"p{k}"]
            rec[f"i2t_cat_hit{k}"] = i2t[f"hit{k}"]
            rec[f"a2t_cat_p{k}"] = a2t[f"p{k}"]
            rec[f"a2t_cat_hit{k}"] = a2t[f"hit{k}"]
            rec[f"avg_cat_p{k}"] = float((i2a[f"p{k}"] + a2i[f"p{k}"] + i2t[f"p{k}"] + a2t[f"p{k}"]) / 4.0)
            rec[f"avg_cat_hit{k}"] = float((i2a[f"hit{k}"] + a2i[f"hit{k}"] + i2t[f"hit{k}"] + a2t[f"hit{k}"]) / 4.0)

        out_rows.append(rec)
        print(f"[stage41] {source_id} m={m} method={method} seed={seed} done")

    merged = { (r["source_id"], int(r["embed_dim"]), r["method"], int(r["seed"])): r for r in existing_rows }
    for r in out_rows:
        merged[(r["source_id"], int(r["embed_dim"]), r["method"], int(r["seed"]))] = r
    rows_all = [merged[k] for k in sorted(merged)]

    metrics = _metric_list(k_values)
    raw: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for rec in rows_all:
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
        "stage": "stage41_semantic_metadata_topk",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": rows_all,
        "raw": raw,
        "stats": stats,
        "category_prompts": categories,
        "topk_values": k_values,
        "labeling_protocol": "metadata_caption_keyword",
        "elapsed_sec": time.time() - start,
    }
    save_json(out, out_path)

    md = [
        "# Stage41 Semantic Top-k (Metadata-Grounded Labels)",
        "",
        "- Label source: human-written captions (COCO captions + AudioCaps captions) via fixed keyword mapping.",
        "- No CLAP/CLIP prompt-classifier labels are used for ground truth in this stage.",
        "",
    ]
    for src in sorted(stats.keys()):
        md.append(f"## {src}")
        for m_key in sorted(stats[src].keys(), key=lambda x: int(x[1:])):
            md.append(f"### {m_key}")
            for method, blk in sorted(stats[src][m_key].get("methods", {}).items()):
                p1 = blk.get("avg_cat_p1", {})
                md.append(f"- {method}: avg_cat_p1={p1.get('mean', 0.0):.4f}±{p1.get('std', 0.0):.4f}")
            md.append("")
    (stage_root / "stage41_semantic_metadata_topk_results.md").write_text("\n".join(md), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=cfg.get("seeds", []),
        extra={"stage": "stage41_semantic_metadata_topk", "elapsed_sec": time.time() - start},
    )
    save_json(provenance, stage_root / "provenance_stage41.json")
    mark_done(markers / "stage41_semantic_metadata_topk.done.json", {"elapsed_sec": time.time() - start})
    print("Stage41 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

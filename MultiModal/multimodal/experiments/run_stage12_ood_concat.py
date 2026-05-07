from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from ..common import env_snapshot, mark_done, save_json
from ..eval.retrieval import recall_at_k
from ..eval.stats import mean_std_ci
from ..models import CLIPProjectionHead, ConcatJLMahalanobisHead, MaskConcatJLMahalanobisHead, MahalanobisOnlyHead, RandomJLMahalanobisHead
from .run_stage11_mia_lira import _DPWrapperForEval, _build_model, _checkpoint_path


def _load_eval_tensors(
    *,
    cache_root: Path,
    dataset: str,
    image_cache_file: str,
    text_cache_file: str,
    n_captions: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, list[int]] | None, dict[int, list[int]] | None]:
    ds_dir = cache_root / dataset
    img = torch.load(ds_dir / image_cache_file, map_location="cpu", weights_only=True).float()
    txt = torch.load(ds_dir / text_cache_file, map_location="cpu", weights_only=True).float()
    if n_captions <= 1:
        return img, txt, None, None
    if txt.shape[0] != img.shape[0] * n_captions:
        raise ValueError(
            f"{dataset}: expected text rows = image_rows * {n_captions}, "
            f"got {txt.shape[0]} vs {img.shape[0]}*{n_captions}"
        )
    gt_i2t = {
        i: list(range(i * n_captions, (i + 1) * n_captions))
        for i in range(img.shape[0])
    }
    gt_t2i = {j: [j // n_captions] for j in range(txt.shape[0])}
    return img, txt, gt_i2t, gt_t2i


@torch.no_grad()
def _eval_model(
    model,
    *,
    img: torch.Tensor,
    txt: torch.Tensor,
    gt_i2t: dict[int, list[int]] | None,
    gt_t2i: dict[int, list[int]] | None,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    model = model.to(device).eval()
    all_i, all_t = [], []
    for i in range(0, len(img), batch_size):
        all_i.append(model.encode_image(img[i:i + batch_size].to(device)).cpu())
    for i in range(0, len(txt), batch_size):
        all_t.append(model.encode_text(txt[i:i + batch_size].to(device)).cpu())
    zi = torch.cat(all_i, dim=0)
    zt = torch.cat(all_t, dim=0)
    return recall_at_k(zi, zt, gt_i2t=gt_i2t, gt_t2i=gt_t2i)


def _summarize(rows: list[dict], metrics: list[str]) -> dict:
    return {m: mean_std_ci([float(r[m]) for r in rows]) for m in metrics}


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage12_ood_concat"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cfg["ood_cache_root"]).resolve()
    seeds = list(cfg["seeds"])
    methods = list(cfg["methods"])
    datasets = list(cfg["ood_datasets"])
    n_caps = {k: int(v) for k, v in cfg.get("ood_n_captions", {}).items()}
    metrics = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10", "avg_R"]

    results = {
        "stage": "stage12_ood_concat",
        "config": cfg,
        "raw": {},
        "stats": {},
    }

    for dataset in datasets:
        print(f"\n=== OOD dataset: {dataset} ===")
        img, txt, gt_i2t, gt_t2i = _load_eval_tensors(
            cache_root=cache_root,
            dataset=dataset,
            image_cache_file=cfg["image_cache_file"],
            text_cache_file=cfg["text_cache_file"],
            n_captions=n_caps.get(dataset, 1),
        )

        results["raw"][dataset] = {}
        results["stats"][dataset] = {}
        for spec in methods:
            method_id = spec["id"]
            rows = []
            print(f"-- method={method_id}")
            for seed in seeds:
                model = _build_model(spec, cfg)
                ckpt = _checkpoint_path(spec, cfg, seed)
                if not ckpt.exists():
                    raise FileNotFoundError(f"missing checkpoint for method={method_id} seed={seed}: {ckpt}")
                state = torch.load(ckpt, map_location=cfg["device"], weights_only=True)
                model.load_state_dict(state, strict=True)
                m = _eval_model(
                    model,
                    img=img,
                    txt=txt,
                    gt_i2t=gt_i2t,
                    gt_t2i=gt_t2i,
                    device=cfg["device"],
                    batch_size=cfg["eval_batch_size"],
                )
                rec = {"seed": seed, **m}
                rows.append(rec)
                print(f"seed={seed} avg_R={m['avg_R']:.4f}")
            results["raw"][dataset][method_id] = rows
            results["stats"][dataset][method_id] = _summarize(rows, metrics)

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": "stage12_ood_concat",
            "elapsed_sec": time.time() - start,
            "ood_datasets": datasets,
        },
    )
    save_json(provenance, stage_root / "provenance_stage12_ood.json")
    save_json(results, stage_root / "E12_ood_concat_results.json")
    mark_done(markers / "stage12_ood_concat.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 12 (OOD concat) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data.datasets import KarpathyCache
from ..eval.diagnostics import pair_diagnostics
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report
from ..models import MaskConcatJLMahalanobisHead
from ..training import train_bimodal


def _load_best(model, ckpt_dir: Path, device: str):
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))
    return model.to(device)


@torch.no_grad()
def _eval_karpathy(
    model,
    cache: KarpathyCache,
    split_name: str,
    *,
    device: str,
    batch_size: int,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model.eval()
    img, txt, gt_i2t, gt_t2i = cache.eval_tensors(split_name)
    all_i, all_t = [], []

    for i in range(0, len(img), batch_size):
        all_i.append(model.encode_image(img[i:i + batch_size].to(device)).cpu())
    for i in range(0, len(txt), batch_size):
        all_t.append(model.encode_text(txt[i:i + batch_size].to(device)).cpu())

    zi = torch.cat(all_i)
    zt = torch.cat(all_t)
    metrics = recall_at_k(zi, zt, gt_i2t=gt_i2t, gt_t2i=gt_t2i)
    return metrics, zi, zt


def _train_or_skip(model, train_loader, val_loader, *, out_dir: Path, cfg: dict, seed: int):
    result_file = out_dir / "train_result.json"
    if result_file.exists():
        return load_json(result_file)
    res = train_bimodal(
        model,
        train_loader,
        val_loader,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        device=cfg["device"],
        ckpt_dir=out_dir,
        patience=cfg.get("patience", 10),
        warmup_epochs=cfg.get("warmup_epochs", 0),
        mode="standard",
    )
    res["seed"] = seed
    save_json(res, result_file)
    return res


def _fmt_float(x: float) -> str:
    s = f"{x:.6g}"
    return s.replace("-", "m").replace(".", "p")


def _method_name(p: float) -> str:
    return f"mask_concat_p{_fmt_float(p)}"


def _merge_stage6_raw(existing: dict, current: dict) -> dict:
    merged = {}
    m_keys = sorted(set(existing.keys()) | set(current.keys()))
    for m_key in m_keys:
        merged[m_key] = {}
        methods = sorted(set(existing.get(m_key, {}).keys()) | set(current.get(m_key, {}).keys()))
        for method in methods:
            by_key = {}
            for r in existing.get(m_key, {}).get(method, []):
                if method == "clip_head":
                    k = ("clip", int(r["seed"]))
                else:
                    k = ("mask", int(r["model_seed"]), int(r["mask_seed"]))
                by_key[k] = r
            for r in current.get(m_key, {}).get(method, []):
                if method == "clip_head":
                    k = ("clip", int(r["seed"]))
                else:
                    k = ("mask", int(r["model_seed"]), int(r["mask_seed"]))
                by_key[k] = r
            merged[m_key][method] = [by_key[k] for k in sorted(by_key.keys())]
    return merged


def _variance_decomposition(rows: list[dict], metric: str = "avg_R") -> dict[str, float]:
    if not rows:
        return {}
    model_ids = sorted({int(r["model_seed"]) for r in rows})
    mask_ids = sorted({int(r["mask_seed"]) for r in rows})
    y = np.array([float(r[metric]) for r in rows], dtype=float)
    grand = float(y.mean())

    # Two-way additive SS decomposition (without interaction term).
    ss_total = float(((y - grand) ** 2).sum())
    ss_model = 0.0
    for ms in model_ids:
        vals = np.array([float(r[metric]) for r in rows if int(r["model_seed"]) == ms], dtype=float)
        ss_model += float(len(vals) * (vals.mean() - grand) ** 2)
    ss_mask = 0.0
    for ks in mask_ids:
        vals = np.array([float(r[metric]) for r in rows if int(r["mask_seed"]) == ks], dtype=float)
        ss_mask += float(len(vals) * (vals.mean() - grand) ** 2)
    ss_resid = max(0.0, ss_total - ss_model - ss_mask)
    denom = ss_total if ss_total > 0 else 1.0
    return {
        "n_rows": len(rows),
        "ss_total": ss_total,
        "ss_model_seed": ss_model,
        "ss_mask_seed": ss_mask,
        "ss_residual": ss_resid,
        "frac_model_seed": ss_model / denom,
        "frac_mask_seed": ss_mask / denom,
        "frac_residual": ss_resid / denom,
    }


def _inject_clip_head_reference(per_method: dict[str, list[dict]], stage5_dataset_results: dict, m_key: str, seeds: list[int]) -> None:
    if not stage5_dataset_results:
        return
    clip_rows = stage5_dataset_results.get("raw", {}).get(m_key, {}).get("clip_head", [])
    clip_by_seed = {int(r["seed"]): r for r in clip_rows}
    rows = []
    for s in seeds:
        if s in clip_by_seed:
            rows.append(clip_by_seed[s])
    if rows:
        per_method["clip_head"] = rows


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage6_e8b"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    model_seeds = list(cfg["seeds"])
    mask_seeds = list(cfg["mask_seeds"])
    embed_dims = list(cfg["embed_dims"])
    ps = [float(p) for p in cfg["mask_ps"]]

    run_summary: dict = {
        "stage": "stage6_e8b_mask_concat",
        "datasets": {},
        "config": cfg,
    }

    for dataset in cfg["datasets"]:
        print(f"\n=== E8b Dataset: {dataset} ===")
        cache_dir = Path(cfg["cache_root"]) / dataset
        cache = KarpathyCache.from_paths(
            cache_dir / cfg["image_cache_file"],
            cache_dir / cfg["text_cache_file"],
            cache_dir / "metadata.json",
        )

        if dataset == "coco":
            train_split = "train_restval"
            val_split = "val"
            test_split = "test"
        else:
            train_split = "train"
            val_split = "val"
            test_split = "test"

        train_ds = cache.make_train_dataset(train_split, training=True)
        val_ds = cache.make_train_dataset(val_split, training=False)
        loader_kw = {
            "batch_size": cfg["batch_size"],
            "num_workers": cfg.get("num_workers", 4),
            "pin_memory": True,
        }
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

        dataset_out = stage_root / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)
        by_m_method: dict[str, dict[str, list[dict]]] = {f"m{m}": {} for m in embed_dims}

        for model_seed in model_seeds:
            print(f"\n-- model_seed={model_seed} --")
            set_seed(model_seed)
            for m in embed_dims:
                m_key = f"m{m}"
                for p in ps:
                    method = _method_name(p)
                    by_m_method[m_key].setdefault(method, [])
                    for mask_seed in mask_seeds:
                        seed_dir = dataset_out / m_key / method / f"model_seed{model_seed}" / f"mask_seed{mask_seed}"
                        eval_file = seed_dir / "eval.json"
                        if eval_file.exists():
                            by_m_method[m_key][method].append(load_json(eval_file))
                            continue

                        model = MaskConcatJLMahalanobisHead(
                            vision_dim=cfg["vision_dim"],
                            text_dim=cfg["text_dim"],
                            embed_dim=m,
                            alpha=float(cfg.get("alpha", 1.0)),
                            beta=float(cfg.get("beta", 1.0)),
                            shared_raw_dim=cfg.get("shared_raw_dim", 768),
                            p=p,
                            mask_seed=mask_seed,
                            jl_eps=cfg["jl_eps"],
                            jl_seed=cfg["jl_seed"],
                        )
                        _train_or_skip(
                            model,
                            train_loader,
                            val_loader,
                            out_dir=seed_dir,
                            cfg=cfg,
                            seed=model_seed,
                        )
                        model = _load_best(model, seed_dir, cfg["device"])
                        metrics, zi, zt = _eval_karpathy(
                            model,
                            cache,
                            test_split,
                            device=cfg["device"],
                            batch_size=cfg["eval_batch_size"],
                        )
                        diag = pair_diagnostics(zi, zt, prefix="it")
                        rec = {
                            "seed": model_seed,
                            "model_seed": model_seed,
                            "mask_seed": mask_seed,
                            "embed_dim": m,
                            "method": method,
                            "p": p,
                            **model.mask_stats(),
                            **metrics,
                            **diag,
                            "n_params": model.n_trainable_params(),
                        }
                        save_json(rec, eval_file)
                        by_m_method[m_key][method].append(rec)
                        print(
                            f"E8b {dataset} {m_key} p={p} "
                            f"model_seed={model_seed} mask_seed={mask_seed} avg_R={metrics['avg_R']:.4f}"
                        )

        existing_dataset_report = {}
        dataset_json = stage_root / f"E8b_mask_concat_{dataset}.json"
        if dataset_json.exists():
            existing_dataset_report = load_json(dataset_json)

        merged_raw = _merge_stage6_raw(existing_dataset_report.get("raw", {}), by_m_method)

        stage5_ref = {}
        stage5_path = Path(cfg.get("stage5_dataset_results_root", "")) / f"E8a_concat_{dataset}.json"
        if stage5_path.exists():
            stage5_ref = load_json(stage5_path)

        metrics_for_stats = [
            "i2t_R@1",
            "i2t_R@5",
            "i2t_R@10",
            "t2i_R@1",
            "t2i_R@5",
            "t2i_R@10",
            "avg_R",
        ]
        dataset_report = {"raw": merged_raw, "stats": {}, "variance_decomposition": {}}
        for m_key, per_method in merged_raw.items():
            _inject_clip_head_reference(per_method, stage5_ref, m_key, model_seeds)
            dataset_report["stats"][m_key] = build_metric_report(
                per_method,
                metrics=metrics_for_stats,
                baseline_method="clip_head" if "clip_head" in per_method else next(iter(per_method.keys())),
            )
            dataset_report["variance_decomposition"][m_key] = {}
            for method_name, rows in per_method.items():
                if method_name == "clip_head":
                    continue
                dataset_report["variance_decomposition"][m_key][method_name] = _variance_decomposition(rows, metric="avg_R")

        run_summary["datasets"][dataset] = dataset_report
        save_json(dataset_report, dataset_json)
        mark_done(markers / f"stage6_e8b_{dataset}.done.json", {"dataset": dataset})

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=model_seeds,
        extra={
            "stage": "stage6_e8b_mask_concat",
            "mask_seeds": mask_seeds,
            "elapsed_sec": time.time() - start,
        },
    )
    save_json(provenance, stage_root / "provenance_stage6_e8b.json")
    save_json(run_summary, stage_root / "E8b_mask_concat_results.json")
    mark_done(markers / "stage6_e8b.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 6 (E8b) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

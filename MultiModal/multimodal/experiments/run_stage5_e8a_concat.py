from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data.datasets import KarpathyCache
from ..eval.diagnostics import pair_diagnostics
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report
from ..models import (
    CLIPProjectionHead,
    ConcatJLMahalanobisHead,
    MahalanobisOnlyHead,
    RandomJLMahalanobisHead,
)
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
        x = img[i:i + batch_size].to(device)
        all_i.append(model.encode_image(x).cpu())

    for i in range(0, len(txt), batch_size):
        x = txt[i:i + batch_size].to(device)
        all_t.append(model.encode_text(x).cpu())

    zi = torch.cat(all_i)
    zt = torch.cat(all_t)
    metrics = recall_at_k(zi, zt, gt_i2t=gt_i2t, gt_t2i=gt_t2i)
    return metrics, zi, zt


def _train_or_skip(
    model,
    train_loader,
    val_loader,
    *,
    seed: int,
    out_dir: Path,
    cfg: dict,
):
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
    # Stable compact method-id formatting, e.g. 0.1 -> 0p1
    s = f"{x:.6g}"
    return s.replace("-", "m").replace(".", "p")


def _concat_name(alpha: float, beta: float) -> str:
    return f"concat_a{_fmt_float(alpha)}_b{_fmt_float(beta)}"


def _merge_stage5_raw(existing: dict, current: dict) -> dict:
    merged = {}
    m_keys = sorted(set(existing.keys()) | set(current.keys()))
    for m_key in m_keys:
        merged[m_key] = {}
        methods = sorted(set(existing.get(m_key, {}).keys()) | set(current.get(m_key, {}).keys()))
        for method in methods:
            by_seed = {}
            for r in existing.get(m_key, {}).get(method, []):
                by_seed[int(r["seed"])] = r
            for r in current.get(m_key, {}).get(method, []):
                by_seed[int(r["seed"])] = r
            merged[m_key][method] = [by_seed[s] for s in sorted(by_seed.keys())]
    return merged


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage5_e8a"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    seeds = list(cfg["seeds"])
    embed_dims = list(cfg["embed_dims"])
    alpha_beta_grid = [(float(a), float(b)) for a, b in cfg["alpha_beta_grid"]]

    run_summary: dict = {
        "stage": "stage5_e8a_concat",
        "datasets": {},
        "config": cfg,
    }

    for dataset in cfg["datasets"]:
        print(f"\n=== E8a Dataset: {dataset} ===")
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

        for seed in seeds:
            print(f"\n-- seed={seed} --")
            set_seed(seed)
            for m in embed_dims:
                m_key = f"m{m}"
                methods = {
                    "clip_head": CLIPProjectionHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                    ),
                    "random_jl_mahal": RandomJLMahalanobisHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        jl_eps=cfg["jl_eps"],
                        jl_seed=cfg["jl_seed"],
                    ),
                    "mahal_only_rfull": MahalanobisOnlyHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        shared_raw_dim=cfg.get("shared_raw_dim", 768),
                    ),
                }
                method_meta: dict[str, dict] = {
                    "clip_head": {},
                    "random_jl_mahal": {},
                    "mahal_only_rfull": {},
                }
                for alpha, beta in alpha_beta_grid:
                    name = _concat_name(alpha, beta)
                    methods[name] = ConcatJLMahalanobisHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        alpha=alpha,
                        beta=beta,
                        shared_raw_dim=cfg.get("shared_raw_dim", 768),
                        jl_eps=cfg["jl_eps"],
                        jl_seed=cfg["jl_seed"],
                    )
                    method_meta[name] = {"alpha": alpha, "beta": beta}

                for method_name, model in methods.items():
                    by_m_method[m_key].setdefault(method_name, [])
                    seed_dir = dataset_out / m_key / method_name / f"seed{seed}"
                    eval_file = seed_dir / "eval.json"
                    if eval_file.exists():
                        by_m_method[m_key][method_name].append(load_json(eval_file))
                        continue

                    _train_or_skip(
                        model,
                        train_loader,
                        val_loader,
                        seed=seed,
                        out_dir=seed_dir,
                        cfg=cfg,
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
                        "seed": seed,
                        "embed_dim": m,
                        "method": method_name,
                        **method_meta[method_name],
                        **metrics,
                        **diag,
                        "n_params": model.n_trainable_params(),
                    }
                    save_json(rec, eval_file)
                    by_m_method[m_key][method_name].append(rec)
                    print(f"E8a {dataset} {m_key} {method_name} seed={seed} avg_R={metrics['avg_R']:.4f}")

        existing_dataset_report = {}
        dataset_json = stage_root / f"E8a_concat_{dataset}.json"
        if dataset_json.exists():
            existing_dataset_report = load_json(dataset_json)

        merged_raw = _merge_stage5_raw(existing_dataset_report.get("raw", {}), by_m_method)

        metrics_for_stats = [
            "i2t_R@1",
            "i2t_R@5",
            "i2t_R@10",
            "t2i_R@1",
            "t2i_R@5",
            "t2i_R@10",
            "avg_R",
        ]
        dataset_report = {"raw": merged_raw, "stats": {}}
        for m_key, per_method in merged_raw.items():
            dataset_report["stats"][m_key] = build_metric_report(
                per_method,
                metrics=metrics_for_stats,
                baseline_method="clip_head",
            )

        run_summary["datasets"][dataset] = dataset_report
        save_json(dataset_report, dataset_json)
        mark_done(markers / f"stage5_e8a_{dataset}.done.json", {"dataset": dataset})

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": "stage5_e8a_concat",
            "elapsed_sec": time.time() - start,
        },
    )
    save_json(provenance, stage_root / "provenance_stage5_e8a.json")
    save_json(run_summary, stage_root / "E8a_concat_results.json")
    # Combined marker in case this invocation runs all datasets.
    done_datasets = sorted(run_summary["datasets"].keys())
    mark_done(markers / "stage5_e8a.done.json", {"datasets": done_datasets, "elapsed_sec": time.time() - start})
    print("Stage 5 (E8a) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

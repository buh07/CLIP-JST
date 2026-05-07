from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data.datasets import KarpathyCache, PairedFeatureDataset
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report
from ..models import SharedJLTriModalMahalHead
from ..training import train_bimodal


def _subset_mean_text(cache: KarpathyCache, image_indices: list[int], groups: list[list[int]]) -> list[torch.Tensor]:
    txt = cache.text_feats
    n_cap = cache.n_captions
    out: list[list[torch.Tensor]] = [[] for _ in groups]
    for idx in image_indices:
        base = idx * n_cap
        for gi, g in enumerate(groups):
            feats = [txt[base + int(k)] for k in g]
            out[gi].append(torch.stack(feats, dim=0).mean(dim=0))
    return [torch.stack(x, dim=0) for x in out]


def _clone_state_dict_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


@torch.no_grad()
def _encode_pair(model: SharedJLTriModalMahalHead, left: torch.Tensor, text: torch.Tensor, *, device: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    model = model.to(device).eval()
    zl, zt = [], []
    for i in range(0, len(left), batch_size):
        zl.append(model.encode_image(left[i:i + batch_size].to(device)).cpu())
        zt.append(model.encode_text(text[i:i + batch_size].to(device)).cpu())
    return torch.cat(zl, dim=0), torch.cat(zt, dim=0)


@torch.no_grad()
def _encode_left_only(model: SharedJLTriModalMahalHead, left: torch.Tensor, *, device: str, batch_size: int) -> torch.Tensor:
    model = model.to(device).eval()
    zl = []
    for i in range(0, len(left), batch_size):
        zl.append(model.encode_image(left[i:i + batch_size].to(device)).cpu())
    return torch.cat(zl, dim=0)


def _train_single(
    *,
    train_x: torch.Tensor,
    train_t: torch.Tensor,
    val_x: torch.Tensor,
    val_t: torch.Tensor,
    seed_dir: Path,
    cfg: dict,
    m: int,
    init_state: dict[str, torch.Tensor] | None,
    freeze_text: bool,
) -> SharedJLTriModalMahalHead:
    model = SharedJLTriModalMahalHead(
        image_dim=int(train_x.shape[1]),
        audio_dim=int(train_x.shape[1]),
        text_dim=int(train_t.shape[1]),
        embed_dim=m,
        shared_raw_dim=int(cfg.get("shared_raw_dim", 768)),
        jl_eps=float(cfg["jl_eps"]),
        jl_seed=int(cfg["jl_seed"]),
    )
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)

    model.set_trainable(image=True, audio=False, text=not freeze_text, logit_scale=True)

    train_loader = DataLoader(
        PairedFeatureDataset(train_x, train_t),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        PairedFeatureDataset(val_x, val_t),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    train_bimodal(
        model,
        train_loader,
        val_loader,
        epochs=int(cfg["epochs"]),
        lr=float(cfg["lr"]),
        device=str(cfg["device"]),
        ckpt_dir=seed_dir,
        patience=int(cfg.get("patience", 10)),
        warmup_epochs=int(cfg.get("warmup_epochs", 0)),
    )
    model.load_state_dict(torch.load(seed_dir / "best.pt", map_location=cfg["device"], weights_only=True))
    return model


def _method_report(
    *,
    method: str,
    seed: int,
    embed_dim: int,
    cross_pair_metrics: dict[str, dict[str, float]],
    pseudo_text_metrics: dict[str, dict[str, float]],
) -> dict:
    pair_avg = sum(float(v["avg_R"]) for v in cross_pair_metrics.values()) / max(1, len(cross_pair_metrics))
    txt_avg = sum(float(v["avg_R"]) for v in pseudo_text_metrics.values()) / max(1, len(pseudo_text_metrics))
    return {
        "seed": int(seed),
        "method": method,
        "embed_dim": int(embed_dim),
        "cross_pair_metrics": cross_pair_metrics,
        "pseudo_text_metrics": pseudo_text_metrics,
        "pair_avg_R": float(pair_avg),
        "pseudo_text_avg_R": float(txt_avg),
    }


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage19_pseudomodality"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cfg["cache_root"]) / "coco"
    cache = KarpathyCache.from_paths(
        cache_dir / cfg["image_cache_file"],
        cache_dir / cfg["text_cache_file"],
        cache_dir / "metadata.json",
    )

    # Three disjoint pseudo-modality caption subsets.
    groups = [list(map(int, g)) for g in cfg.get("pseudo_caption_groups", [[0, 1], [2, 3], [4]])]
    hub_group = list(map(int, cfg.get("text_hub_group", [0])))

    train_split = cfg.get("train_split", "train_restval")
    val_split = cfg.get("val_split", "val")
    test_split = cfg.get("test_split", "test")

    tr_idx = cache.split_indices(train_split)
    va_idx = cache.split_indices(val_split)
    te_idx = cache.split_indices(test_split)

    pseudo_tr = _subset_mean_text(cache, tr_idx, groups)
    pseudo_va = _subset_mean_text(cache, va_idx, groups)
    pseudo_te = _subset_mean_text(cache, te_idx, groups)

    text_tr = _subset_mean_text(cache, tr_idx, [hub_group])[0]
    text_va = _subset_mean_text(cache, va_idx, [hub_group])[0]
    text_te = _subset_mean_text(cache, te_idx, [hub_group])[0]

    seeds = [int(s) for s in cfg["seeds"]]
    embed_dims = [int(m) for m in cfg["embed_dims"]]

    results = {"stage": "stage19_pseudomodality_transitivity", "raw": {}, "stats": {}, "gate": {}}

    for m in embed_dims:
        m_key = f"m{m}"
        results["raw"][m_key] = {"modular_shared_jl": [], "joint_shared_jl_baseline": []}

        for seed in seeds:
            set_seed(seed)

            # -------- Modular (train pseudo0 + freeze text hub, then pseudo1/2 independently) --------
            mod_seed_dir = stage_root / m_key / "modular_shared_jl" / f"seed{seed}"
            mod_eval = mod_seed_dir / "eval.json"
            if mod_eval.exists():
                results["raw"][m_key]["modular_shared_jl"].append(load_json(mod_eval))
            else:
                model0 = _train_single(
                    train_x=pseudo_tr[0],
                    train_t=text_tr,
                    val_x=pseudo_va[0],
                    val_t=text_va,
                    seed_dir=mod_seed_dir / "pseudo0",
                    cfg=cfg,
                    m=m,
                    init_state=None,
                    freeze_text=False,
                )
                base_state = _clone_state_dict_cpu(model0.state_dict())

                model1 = _train_single(
                    train_x=pseudo_tr[1],
                    train_t=text_tr,
                    val_x=pseudo_va[1],
                    val_t=text_va,
                    seed_dir=mod_seed_dir / "pseudo1",
                    cfg=cfg,
                    m=m,
                    init_state=base_state,
                    freeze_text=True,
                )
                model2 = _train_single(
                    train_x=pseudo_tr[2],
                    train_t=text_tr,
                    val_x=pseudo_va[2],
                    val_t=text_va,
                    seed_dir=mod_seed_dir / "pseudo2",
                    cfg=cfg,
                    m=m,
                    init_state=base_state,
                    freeze_text=True,
                )

                models = [model0, model1, model2]
                z_groups = [
                    _encode_left_only(md, pseudo_te[i], device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
                    for i, md in enumerate(models)
                ]

                cross = {}
                for i, j in combinations(range(3), 2):
                    cross[f"p{i}_p{j}"] = recall_at_k(z_groups[i], z_groups[j])

                pt = {}
                for i, md in enumerate(models):
                    zi, zt = _encode_pair(md, pseudo_te[i], text_te, device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
                    pt[f"p{i}_t"] = recall_at_k(zi, zt)

                rec_mod = _method_report(
                    method="modular_shared_jl",
                    seed=seed,
                    embed_dim=m,
                    cross_pair_metrics=cross,
                    pseudo_text_metrics=pt,
                )
                save_json(rec_mod, mod_eval)
                results["raw"][m_key]["modular_shared_jl"].append(rec_mod)
                print(f"stage19 {m_key} modular seed={seed} pair_avg_R={rec_mod['pair_avg_R']:.4f}")

            # -------- Joint baseline (single shared model over all pseudo groups) --------
            joint_seed_dir = stage_root / m_key / "joint_shared_jl_baseline" / f"seed{seed}"
            joint_eval = joint_seed_dir / "eval.json"
            if joint_eval.exists():
                results["raw"][m_key]["joint_shared_jl_baseline"].append(load_json(joint_eval))
            else:
                tr_x = torch.cat(pseudo_tr, dim=0)
                va_x = torch.cat(pseudo_va, dim=0)
                tr_t = torch.cat([text_tr, text_tr, text_tr], dim=0)
                va_t = torch.cat([text_va, text_va, text_va], dim=0)

                joint_model = _train_single(
                    train_x=tr_x,
                    train_t=tr_t,
                    val_x=va_x,
                    val_t=va_t,
                    seed_dir=joint_seed_dir,
                    cfg=cfg,
                    m=m,
                    init_state=None,
                    freeze_text=False,
                )

                z_groups = [
                    _encode_left_only(joint_model, pseudo_te[i], device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
                    for i in range(3)
                ]
                cross = {}
                for i, j in combinations(range(3), 2):
                    cross[f"p{i}_p{j}"] = recall_at_k(z_groups[i], z_groups[j])

                pt = {}
                for i in range(3):
                    zi, zt = _encode_pair(joint_model, pseudo_te[i], text_te, device=cfg["device"], batch_size=int(cfg["eval_batch_size"]))
                    pt[f"p{i}_t"] = recall_at_k(zi, zt)

                rec_joint = _method_report(
                    method="joint_shared_jl_baseline",
                    seed=seed,
                    embed_dim=m,
                    cross_pair_metrics=cross,
                    pseudo_text_metrics=pt,
                )
                save_json(rec_joint, joint_eval)
                results["raw"][m_key]["joint_shared_jl_baseline"].append(rec_joint)
                print(f"stage19 {m_key} joint seed={seed} pair_avg_R={rec_joint['pair_avg_R']:.4f}")

        results["stats"][m_key] = build_metric_report(
            results["raw"][m_key],
            metrics=["pair_avg_R", "pseudo_text_avg_R"],
            baseline_method="joint_shared_jl_baseline",
        )

        mod_mean = results["stats"][m_key]["methods"]["modular_shared_jl"]["pair_avg_R"]["mean"]
        joint_mean = results["stats"][m_key]["methods"]["joint_shared_jl_baseline"]["pair_avg_R"]["mean"]
        rel = float(mod_mean / max(1e-12, joint_mean))
        thr = float(cfg.get("gate_relative_threshold", 0.90))
        results["gate"][m_key] = {
            "modular_pair_avg_R_mean": float(mod_mean),
            "joint_pair_avg_R_mean": float(joint_mean),
            "relative_ratio": rel,
            "threshold": thr,
            "pass": bool(rel >= thr),
        }

    gate_m = int(cfg.get("gate_embed_dim", 256))
    gate_key = f"m{gate_m}"
    final_gate = results["gate"].get(gate_key, {})
    results["gate_decision"] = {
        "gate_embed_dim": gate_m,
        "gate_key": gate_key,
        "pass": bool(final_gate.get("pass", False)),
        "relative_ratio": float(final_gate.get("relative_ratio", 0.0)),
        "threshold": float(final_gate.get("threshold", cfg.get("gate_relative_threshold", 0.90))),
    }

    save_json(results, stage_root / "stage19_pseudomodality_results.json")
    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": "stage19_pseudomodality_transitivity",
            "elapsed_sec": time.time() - start,
            "pseudo_caption_groups": groups,
            "text_hub_group": hub_group,
        },
    )
    save_json(provenance, stage_root / "provenance_stage19.json")
    mark_done(markers / "stage19_pseudomodality_transitivity.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 19 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

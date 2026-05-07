from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..common import env_snapshot, mark_done, save_json, set_seed
from ..data.datasets import KarpathyCache
from ..eval.mia import auc_from_roc, fit_gaussian, lira_log_likelihood_ratio, roc_curve_from_scores, tpr_at_fpr
from ..eval.stats import mean_std_ci
from ..models import CLIPProjectionHead, ConcatJLMahalanobisHead, MaskConcatJLMahalanobisHead, MahalanobisOnlyHead, RandomJLMahalanobisHead


class _DPWrapperForEval(nn.Module):
    """
    Evaluation wrapper matching stage8 DP training module structure.
    """

    def __init__(self, vision_dim: int, text_dim: int, shared_raw_dim: int = 768):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.head = MahalanobisOnlyHead(
            vision_dim=vision_dim,
            text_dim=text_dim,
            shared_raw_dim=shared_raw_dim,
        )
        self.head.logit_scale.requires_grad_(False)

    @property
    def logit_scale(self):
        return self.head.logit_scale

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return self.head.encode_image(v)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return self.head.encode_text(t)


def _build_model(spec: dict, cfg: dict):
    kind = spec["kind"]
    m = int(cfg["embed_dim"])
    if kind == "clip_head":
        return CLIPProjectionHead(cfg["vision_dim"], cfg["text_dim"], m)
    if kind == "random_jl_mahal":
        return RandomJLMahalanobisHead(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
            jl_eps=cfg["jl_eps"],
            jl_seed=cfg["jl_seed"],
        )
    if kind == "concat":
        return ConcatJLMahalanobisHead(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
            alpha=float(spec.get("alpha", 1.0)),
            beta=float(spec.get("beta", 1.0)),
            shared_raw_dim=cfg.get("shared_raw_dim", 768),
            jl_eps=cfg["jl_eps"],
            jl_seed=cfg["jl_seed"],
        )
    if kind == "mask_concat":
        return MaskConcatJLMahalanobisHead(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
            alpha=float(spec.get("alpha", 1.0)),
            beta=float(spec.get("beta", 1.0)),
            shared_raw_dim=cfg.get("shared_raw_dim", 768),
            p=float(spec["p"]),
            mask_seed=int(spec["mask_seed"]),
            jl_eps=cfg["jl_eps"],
            jl_seed=cfg["jl_seed"],
        )
    if kind == "dpsgd":
        return _DPWrapperForEval(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            shared_raw_dim=cfg.get("shared_raw_dim", 768),
        )
    raise ValueError(f"unknown method kind: {kind}")


def _checkpoint_path(spec: dict, cfg: dict, seed: int) -> Path:
    family = spec["family"]
    m = int(cfg["embed_dim"])
    if family == "stage5":
        return (
            Path(cfg["stage5_root"]).resolve()
            / "coco"
            / f"m{m}"
            / spec["method_name"]
            / f"seed{seed}"
            / "best.pt"
        )
    if family == "stage6":
        return (
            Path(cfg["stage6_root"]).resolve()
            / "coco"
            / f"m{m}"
            / spec["method_name"]
            / f"model_seed{seed}"
            / f"mask_seed{int(spec['mask_seed'])}"
            / "best.pt"
        )
    if family == "stage8":
        return (
            Path(cfg["stage8_root"]).resolve()
            / spec["epsilon_key"]
            / f"seed{seed}"
            / "best.pt"
        )
    raise ValueError(f"unknown family: {family}")


@torch.no_grad()
def _pair_scores(model, img: torch.Tensor, txt: torch.Tensor, *, device: str, batch_size: int) -> np.ndarray:
    model = model.to(device).eval()
    out = []
    scale = (
        float(model.logit_scale.exp().clamp(max=100.0).item())
        if hasattr(model, "logit_scale")
        else (1.0 / 0.07)
    )
    for i in range(0, len(img), batch_size):
        vi = img[i:i + batch_size].to(device)
        ti = txt[i:i + batch_size].to(device)
        zi = model.encode_image(vi)
        zt = model.encode_text(ti)
        s = (zi * zt).sum(dim=1) * scale
        out.append(s.cpu().numpy())
    return np.concatenate(out, axis=0)


def _load_state(model: nn.Module, ckpt_path: Path, device: str) -> nn.Module:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage11_mia_lira"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.get("sampling_seed", 0)))
    cache_dir = Path(cfg["cache_root"]) / "coco"
    cache = KarpathyCache.from_paths(
        cache_dir / cfg["image_cache_file"],
        cache_dir / cfg["text_cache_file"],
        cache_dir / "metadata.json",
    )
    n_cap = int(cache.n_captions)
    cap_idx = int(cfg.get("caption_index", 0))
    if cap_idx < 0 or cap_idx >= n_cap:
        raise ValueError(f"caption_index must be in [0,{n_cap-1}]")

    members = cache.split_indices(cfg.get("member_split", "train_restval"))
    nonmembers = cache.split_indices(cfg.get("nonmember_split", "test"))
    n_eval = int(cfg.get("n_eval_per_class", 5000))
    if n_eval > len(nonmembers):
        n_eval = len(nonmembers)
    if n_eval > len(members):
        n_eval = len(members)
    if n_eval <= 0:
        raise ValueError("n_eval_per_class must be > 0 and <= split sizes")

    rng = np.random.default_rng(int(cfg.get("sampling_seed", 0)))
    member_sel = sorted(rng.choice(np.asarray(members), size=n_eval, replace=False).tolist())
    nonmember_sel = sorted(rng.choice(np.asarray(nonmembers), size=n_eval, replace=False).tolist())

    member_img = cache.image_feats[member_sel]
    member_txt = cache.text_feats[[idx * n_cap + cap_idx for idx in member_sel]]
    nonmember_img = cache.image_feats[nonmember_sel]
    nonmember_txt = cache.text_feats[[idx * n_cap + cap_idx for idx in nonmember_sel]]

    seeds = list(cfg["seeds"])
    batch_size = int(cfg.get("eval_batch_size", 4096))
    methods = list(cfg["methods"])

    raw: dict[str, dict] = {}
    stats: dict[str, dict] = {}

    for spec in methods:
        method_id = spec["id"]
        per_seed_scores = {}
        print(f"\n=== LiRA method: {method_id} ===")

        for seed in seeds:
            model = _build_model(spec, cfg)
            ckpt = _checkpoint_path(spec, cfg, seed)
            model = _load_state(model, ckpt, cfg["device"])
            mem_scores = _pair_scores(model, member_img, member_txt, device=cfg["device"], batch_size=batch_size)
            non_scores = _pair_scores(model, nonmember_img, nonmember_txt, device=cfg["device"], batch_size=batch_size)
            per_seed_scores[seed] = {"member": mem_scores, "nonmember": non_scores}
            print(
                f"seed={seed} member_score_mean={mem_scores.mean():.4f} "
                f"nonmember_score_mean={non_scores.mean():.4f}"
            )

        per_target = []
        for target_seed in seeds:
            shadow = [s for s in seeds if s != target_seed]
            if not shadow:
                continue
            in_shadow = np.concatenate([per_seed_scores[s]["member"] for s in shadow], axis=0)
            out_shadow = np.concatenate([per_seed_scores[s]["nonmember"] for s in shadow], axis=0)
            in_mu, in_std = fit_gaussian(in_shadow)
            out_mu, out_std = fit_gaussian(out_shadow)

            t_in = per_seed_scores[target_seed]["member"]
            t_out = per_seed_scores[target_seed]["nonmember"]
            labels = np.concatenate([np.ones_like(t_in), np.zeros_like(t_out)], axis=0)
            target_scores = np.concatenate([t_in, t_out], axis=0)
            llr = lira_log_likelihood_ratio(
                target_scores,
                in_mu=in_mu,
                in_std=in_std,
                out_mu=out_mu,
                out_std=out_std,
            )
            fpr, tpr, _thr = roc_curve_from_scores(labels, llr)
            auc = auc_from_roc(fpr, tpr)
            tpr_1pct = tpr_at_fpr(fpr, tpr, target_fpr=0.01)
            tpr_01pct = tpr_at_fpr(fpr, tpr, target_fpr=0.001)
            rec = {
                "target_seed": target_seed,
                "shadow_seeds": shadow,
                "in_mu": in_mu,
                "in_std": in_std,
                "out_mu": out_mu,
                "out_std": out_std,
                "auc": float(auc),
                "tpr_at_fpr_1pct": float(tpr_1pct),
                "tpr_at_fpr_0p1pct": float(tpr_01pct),
            }
            per_target.append(rec)
            print(
                f"target_seed={target_seed} auc={auc:.4f} "
                f"TPR@1%FPR={tpr_1pct:.4f} TPR@0.1%FPR={tpr_01pct:.4f}"
            )

        raw[method_id] = {
            "method_spec": spec,
            "per_target": per_target,
        }
        stats[method_id] = {
            "auc": mean_std_ci([float(r["auc"]) for r in per_target]),
            "tpr_at_fpr_1pct": mean_std_ci([float(r["tpr_at_fpr_1pct"]) for r in per_target]),
            "tpr_at_fpr_0p1pct": mean_std_ci([float(r["tpr_at_fpr_0p1pct"]) for r in per_target]),
        }

    summary = {
        "stage": "stage11_mia_lira",
        "config": cfg,
        "sample_selection": {
            "n_eval_per_class": int(n_eval),
            "member_split": cfg.get("member_split", "train_restval"),
            "nonmember_split": cfg.get("nonmember_split", "test"),
            "caption_index": cap_idx,
        },
        "raw": raw,
        "stats": stats,
    }

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": "stage11_mia_lira",
            "elapsed_sec": time.time() - start,
            "n_eval_per_class": int(n_eval),
        },
    )
    save_json(provenance, stage_root / "provenance_stage11_mia_lira.json")
    save_json(summary, stage_root / "E11_mia_lira_results.json")
    mark_done(markers / "stage11_mia_lira.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 11 (MIA LiRA) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

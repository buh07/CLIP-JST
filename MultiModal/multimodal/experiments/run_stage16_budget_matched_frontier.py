from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json
from ..data.datasets import KarpathyCache
from ..eval.privacy import iterative_feature_inversion_attack, mlp_inversion_attack, reconstruction_metrics
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report, mean_std_ci
from ..models.jl_ops import kane_nelson_jl
from .run_stage13_federated_comparison import build_model_from_spec, checkpoint_path


DEFAULT_METRICS = [
    "i2t_R@1",
    "i2t_R@5",
    "i2t_R@10",
    "t2i_R@1",
    "t2i_R@5",
    "t2i_R@10",
    "avg_R",
    "linear_rel_error",
    "mlp_rel_error",
    "iterative_rel_error",
    "comm_mb_model_update",
    "embedding_bytes_per_vector",
    "embedding_bytes_per_pair",
]


class BudgetAdapter:
    """
    Fixed random budget adapter:
    - identity for d == budget_dim
    - fixed random JL (Kane-Nelson) for d > budget_dim
    """

    def __init__(
        self,
        *,
        budget_dim: int,
        projector_seed: int,
        projector_eps: float,
    ):
        self.budget_dim = int(budget_dim)
        self.projector_seed = int(projector_seed)
        self.projector_eps = float(projector_eps)
        self._proj_cpu: dict[int, torch.Tensor] = {}
        self._proj_device: dict[tuple[int, str], torch.Tensor] = {}

    def _seed_for_dim(self, in_dim: int) -> int:
        # Stable per-input-dimension seed schedule.
        return int(self.projector_seed + 10007 * in_dim)

    def _cpu_projector(self, in_dim: int) -> torch.Tensor:
        if in_dim in self._proj_cpu:
            return self._proj_cpu[in_dim]
        if in_dim == self.budget_dim:
            p = torch.eye(self.budget_dim, dtype=torch.float32)
        elif in_dim > self.budget_dim:
            seed = self._seed_for_dim(in_dim)
            p = torch.tensor(
                kane_nelson_jl(in_dim, self.budget_dim, eps=self.projector_eps, seed=seed).toarray(),
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"input dim {in_dim} < budget dim {self.budget_dim} is unsupported")
        self._proj_cpu[in_dim] = p
        return p

    def _device_projector(self, in_dim: int, device: torch.device) -> torch.Tensor:
        key = (in_dim, str(device))
        if key in self._proj_device:
            return self._proj_device[key]
        p = self._cpu_projector(in_dim).to(device)
        self._proj_device[key] = p
        return p

    def project(self, z: torch.Tensor) -> torch.Tensor:
        in_dim = int(z.shape[1])
        if in_dim == self.budget_dim:
            out = z
        else:
            p = self._device_projector(in_dim, z.device)
            out = z @ p.T
        return F.normalize(out, dim=-1)

    def metadata(self, in_dim: int) -> dict:
        if in_dim == self.budget_dim:
            return {
                "budget_dim": self.budget_dim,
                "projector_type": "identity",
                "projector_seed": None,
                "projector_eps": None,
                "pre_dim": in_dim,
                "post_dim": self.budget_dim,
                "projection_applied": False,
            }
        return {
            "budget_dim": self.budget_dim,
            "projector_type": "kane_nelson_fixed_random",
            "projector_seed": self._seed_for_dim(in_dim),
            "projector_eps": self.projector_eps,
            "pre_dim": in_dim,
            "post_dim": self.budget_dim,
            "projection_applied": True,
        }


class BudgetWrappedImageModel(nn.Module):
    """
    Wrap model.encode_image with the budget adapter for iterative attacks.
    """

    def __init__(self, base_model: nn.Module, adapter: BudgetAdapter):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter

    @property
    def logit_scale(self):
        return self.base_model.logit_scale

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        z = self.base_model.encode_image(v)
        return self.adapter.project(z)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        z = self.base_model.encode_text(t)
        return self.adapter.project(z)


@torch.no_grad()
def _encode_images_budget(
    model: nn.Module,
    x: torch.Tensor,
    *,
    adapter: BudgetAdapter,
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    model = model.to(device).eval()
    outs = []
    pre_dim: int | None = None
    for i in range(0, len(x), batch_size):
        z = model.encode_image(x[i:i + batch_size].to(device))
        if pre_dim is None:
            pre_dim = int(z.shape[1])
        outs.append(adapter.project(z).cpu())
    return torch.cat(outs, dim=0), int(pre_dim if pre_dim is not None else adapter.budget_dim)


@torch.no_grad()
def _encode_texts_budget(
    model: nn.Module,
    x: torch.Tensor,
    *,
    adapter: BudgetAdapter,
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    model = model.to(device).eval()
    outs = []
    pre_dim: int | None = None
    for i in range(0, len(x), batch_size):
        z = model.encode_text(x[i:i + batch_size].to(device))
        if pre_dim is None:
            pre_dim = int(z.shape[1])
        outs.append(adapter.project(z).cpu())
    return torch.cat(outs, dim=0), int(pre_dim if pre_dim is not None else adapter.budget_dim)


def _linear_probe_reconstruction(
    z: torch.Tensor,
    x: torch.Tensor,
    *,
    train_frac: float,
    seed: int,
) -> dict[str, float]:
    z = z.float().cpu()
    x = x.float().cpu()
    n = len(z)
    n_train = max(1, int(train_frac * n))
    n_test = n - n_train
    if n_test <= 0:
        n_train = n - 1
        n_test = 1
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr = perm[:n_train]
    te = perm[n_train:]
    b = torch.linalg.pinv(z[tr]) @ x[tr]
    x_hat = z[te] @ b
    return reconstruction_metrics(x_hat, x[te])


@torch.no_grad()
def _eval_karpathy_budget(
    model: nn.Module,
    cache: KarpathyCache,
    split_name: str,
    *,
    adapter: BudgetAdapter,
    device: str,
    batch_size: int,
) -> tuple[dict[str, float], dict]:
    img, txt, gt_i2t, gt_t2i = cache.eval_tensors(split_name)
    zi, pre_i = _encode_images_budget(model, img, adapter=adapter, device=device, batch_size=batch_size)
    zt, pre_t = _encode_texts_budget(model, txt, adapter=adapter, device=device, batch_size=batch_size)
    m = recall_at_k(zi, zt, gt_i2t=gt_i2t, gt_t2i=gt_t2i)
    meta = {
        "image_adapter": adapter.metadata(pre_i),
        "text_adapter": adapter.metadata(pre_t),
    }
    return m, meta


def _partition_offset(partition_id: str) -> int:
    return int(sum((i + 1) * ord(c) for i, c in enumerate(partition_id)))


def _sample_local_indices(n_total: int, n_sample: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_sample, replace=False)
    return torch.tensor(idx, dtype=torch.long)


def _load_stage13_comm_index(path: Path) -> dict[tuple[str, str, int], float]:
    if not path.exists():
        return {}
    obj = load_json(path)
    out: dict[tuple[str, str, int], float] = {}
    for part, methods in obj.get("raw", {}).items():
        for method, rows in methods.items():
            for r in rows:
                key = (part, method, int(r["seed"]))
                if "comm_mb" in r:
                    out[key] = float(r["comm_mb"])
    return out


def _aggregate_draws(draw_rows: list[dict]) -> dict:
    def _series(k: str) -> list[float]:
        return [float(r[k]) for r in draw_rows]

    return {
        "n_draws": int(len(draw_rows)),
        "linear_rel_error": mean_std_ci(_series("linear_rel_error")),
        "mlp_rel_error": mean_std_ci(_series("mlp_rel_error")),
        "iterative_rel_error": mean_std_ci(_series("iterative_rel_error")),
    }


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage16_budget_matched"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage13_root = Path(cfg["stage13_root"]).resolve() / "stage13_federated"
    if not stage13_root.exists():
        raise FileNotFoundError(f"missing stage13 federated root: {stage13_root}")
    stage13_comm = _load_stage13_comm_index(stage13_root / "E13_federated_results.json")

    cache_dir = Path(cfg["cache_root"]) / "coco"
    cache = KarpathyCache.from_paths(
        cache_dir / cfg["image_cache_file"],
        cache_dir / cfg["text_cache_file"],
        cache_dir / "metadata.json",
    )

    probe_split = cfg.get("privacy_split", "train_restval")
    probe_idx = cache.split_indices(probe_split)
    x_probe = cache.image_feats[probe_idx].float()
    n_total_probe = int(len(x_probe))
    max_probe = int(cfg.get("max_probe_samples", 0))
    n_probe_sample = n_total_probe if max_probe <= 0 else min(n_total_probe, max_probe)
    n_draws = int(cfg.get("n_probe_draws", 5))
    draw_seed_base = int(cfg.get("draw_seed_base", 20260502))

    seeds = [int(s) for s in cfg["seeds"]]
    methods = list(cfg["methods"])
    partitions = list(cfg["partitions"])

    method_filter = set(cfg.get("method_filter", []))
    if method_filter:
        methods = [m for m in methods if m["id"] in method_filter]
    partition_filter = set(cfg.get("partition_filter", []))
    if partition_filter:
        partitions = [p for p in partitions if p in partition_filter]

    adapter = BudgetAdapter(
        budget_dim=int(cfg.get("budget_dim", 256)),
        projector_seed=int(cfg.get("budget_projector_seed", 12345)),
        projector_eps=float(cfg.get("budget_projector_eps", 0.1)),
    )

    result_path = stage_root / "E16_budget_matched_results.json"
    if result_path.exists():
        results = load_json(result_path)
        results["stage"] = "stage16_budget_matched_frontier"
        results["config"] = cfg
        results.setdefault("raw", {})
        results.setdefault("stats", {})
    else:
        results = {
            "stage": "stage16_budget_matched_frontier",
            "config": cfg,
            "raw": {},
            "stats": {},
        }

    metrics_for_stats = list(cfg.get("metrics_for_stats", DEFAULT_METRICS))
    baseline_method = cfg.get("baseline_method", "mask_concat")

    embedding_bytes_per_vector = int(adapter.budget_dim * 4)
    embedding_bytes_per_pair = int(2 * embedding_bytes_per_vector)

    for partition_id in partitions:
        print(f"\n=== stage16 partition={partition_id} ===")
        results["raw"].setdefault(partition_id, {})

        draw_idx_by_seed: dict[int, dict[int, torch.Tensor]] = {}
        part_off = _partition_offset(partition_id)
        for seed in seeds:
            by_draw = {}
            for draw_id in range(n_draws):
                draw_seed = int(draw_seed_base + part_off * 100000 + seed * 1000 + draw_id)
                by_draw[draw_id] = _sample_local_indices(n_total_probe, n_probe_sample, draw_seed)
            draw_idx_by_seed[seed] = by_draw

        for spec in methods:
            method_id = spec["id"]
            print(f"-- method={method_id}")
            seed_rows = []
            for seed in seeds:
                run_dir = stage_root / partition_id / method_id / f"seed{seed}"
                draw_dir = run_dir / "draws"
                draw_dir.mkdir(parents=True, exist_ok=True)
                eval_file = run_dir / "eval.json"
                retrieval_file = run_dir / "retrieval_budget.json"
                if eval_file.exists():
                    seed_rows.append(load_json(eval_file))
                    continue

                model = build_model_from_spec(spec, cfg)
                ckpt = checkpoint_path(stage13_root, partition_id, method_id, seed)
                if not ckpt.exists():
                    raise FileNotFoundError(f"missing stage13 checkpoint: {ckpt}")
                state = torch.load(ckpt, map_location=cfg["device"], weights_only=True)
                model.load_state_dict(state, strict=True)

                if retrieval_file.exists():
                    retrieval_obj = load_json(retrieval_file)
                    retrieval_metrics = retrieval_obj["metrics"]
                    adapter_meta = retrieval_obj["adapter_meta"]
                else:
                    retrieval_metrics, adapter_meta = _eval_karpathy_budget(
                        model,
                        cache,
                        cfg.get("test_split", "test"),
                        adapter=adapter,
                        device=cfg["device"],
                        batch_size=int(cfg.get("eval_batch_size", 4096)),
                    )
                    save_json(
                        {"metrics": retrieval_metrics, "adapter_meta": adapter_meta},
                        retrieval_file,
                    )

                draw_rows = []
                for draw_id in range(n_draws):
                    draw_file = draw_dir / f"draw{draw_id}.json"
                    if draw_file.exists():
                        draw_rows.append(load_json(draw_file))
                        continue

                    draw_seed = int(draw_seed_base + part_off * 100000 + seed * 1000 + draw_id)
                    idx_local = draw_idx_by_seed[seed][draw_id]
                    x_draw = x_probe[idx_local].float()

                    z_draw, pre_dim = _encode_images_budget(
                        model,
                        x_draw,
                        adapter=adapter,
                        device=cfg["device"],
                        batch_size=int(cfg.get("eval_batch_size", 4096)),
                    )
                    lin = _linear_probe_reconstruction(
                        z_draw,
                        x_draw,
                        train_frac=float(cfg.get("attack_train_frac", 0.8)),
                        seed=draw_seed,
                    )
                    mlp, _mlp_hat, _mlp_true = mlp_inversion_attack(
                        z_draw,
                        x_draw,
                        train_frac=float(cfg.get("attack_train_frac", 0.8)),
                        hidden_dim=int(cfg.get("attack_hidden_dim", 1024)),
                        epochs=int(cfg.get("attack_epochs", 60)),
                        lr=float(cfg.get("attack_lr", 1e-3)),
                        batch_size=int(cfg.get("attack_batch_size", 1024)),
                        seed=draw_seed,
                        device=cfg["device"],
                    )
                    wrapped = BudgetWrappedImageModel(model, adapter)
                    iterative, _it_hat, _it_true = iterative_feature_inversion_attack(
                        wrapped,
                        x_draw,
                        train_frac=float(cfg.get("attack_train_frac", 0.8)),
                        steps=int(cfg.get("iterative_steps", 180)),
                        lr=float(cfg.get("iterative_lr", 0.05)),
                        prior_weight=float(cfg.get("iterative_prior_weight", 1e-3)),
                        batch_size=int(cfg.get("iterative_batch_size", 128)),
                        seed=draw_seed,
                        device=cfg["device"],
                    )

                    draw_rec = {
                        "draw_id": int(draw_id),
                        "draw_seed": int(draw_seed),
                        "n_probe": int(len(x_draw)),
                        "adapter_meta_image": adapter.metadata(pre_dim),
                        "linear_probe": lin,
                        "mlp_inverter": mlp,
                        "iterative_inverter": iterative,
                        "linear_rel_error": float(lin["mean_relative_reconstruction_error"]),
                        "mlp_rel_error": float(mlp["mean_relative_reconstruction_error"]),
                        "iterative_rel_error": float(iterative["mean_relative_reconstruction_error"]),
                    }
                    save_json(draw_rec, draw_file)
                    draw_rows.append(draw_rec)
                    print(
                        f"partition={partition_id} method={method_id} seed={seed} draw={draw_id} "
                        f"lin={draw_rec['linear_rel_error']:.4f} "
                        f"mlp={draw_rec['mlp_rel_error']:.4f} iter={draw_rec['iterative_rel_error']:.4f}"
                    )

                draw_rows = sorted(draw_rows, key=lambda r: int(r["draw_id"]))
                draw_agg = _aggregate_draws(draw_rows)
                comm_key = (partition_id, method_id, int(seed))
                comm_model_mb = stage13_comm.get(comm_key)

                rec = {
                    "seed": int(seed),
                    "partition": partition_id,
                    "method": method_id,
                    "n_probe_total": int(n_total_probe),
                    "n_probe_per_draw": int(n_probe_sample),
                    "n_probe_draws": int(n_draws),
                    "budget_adapter": {
                        "budget_dim": int(adapter.budget_dim),
                        "projector_seed_base": int(adapter.projector_seed),
                        "projector_eps": float(adapter.projector_eps),
                        "image": adapter_meta["image_adapter"],
                        "text": adapter_meta["text_adapter"],
                    },
                    "embedding_bytes_per_vector": int(embedding_bytes_per_vector),
                    "embedding_bytes_per_pair": int(embedding_bytes_per_pair),
                    "retrieval": retrieval_metrics,
                    **retrieval_metrics,
                    "draw_metrics": draw_rows,
                    "draw_agg": draw_agg,
                    "linear_rel_error": float(draw_agg["linear_rel_error"]["mean"]),
                    "mlp_rel_error": float(draw_agg["mlp_rel_error"]["mean"]),
                    "iterative_rel_error": float(draw_agg["iterative_rel_error"]["mean"]),
                }
                if comm_model_mb is not None:
                    rec["comm_mb_model_update"] = float(comm_model_mb)
                save_json(rec, eval_file)
                seed_rows.append(rec)
                print(
                    f"DONE partition={partition_id} method={method_id} seed={seed} "
                    f"avg_R={rec['avg_R']:.4f} mlp={rec['mlp_rel_error']:.4f}"
                )

            results["raw"][partition_id][method_id] = seed_rows

        results["stats"][partition_id] = build_metric_report(
            results["raw"][partition_id],
            metrics=metrics_for_stats,
            baseline_method=baseline_method,
        )
        save_json(results, result_path)

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": "stage16_budget_matched_frontier",
            "elapsed_sec": time.time() - start,
            "n_probe_total": int(n_total_probe),
            "n_probe_per_draw": int(n_probe_sample),
            "n_draws": int(n_draws),
            "budget_dim": int(adapter.budget_dim),
        },
    )
    save_json(provenance, stage_root / "provenance_stage16_budget_matched.json")
    save_json(results, result_path)
    mark_done(markers / "stage16_budget_matched.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 16 (budget-matched frontier) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

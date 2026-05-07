from __future__ import annotations

import argparse
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data.datasets import KarpathyCache
from ..eval.privacy import mlp_inversion_attack, reconstruction_metrics
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report
from ..models import (
    CLIPProjectionHead,
    FedCLIPProxyHead,
    FedCLIPPretrainedAdapterHead,
    FedMVPProxyHead,
    FedMVPPretrainedPromptHead,
    MahalanobisBottleneckHead,
    MahalanobisOnlyHead,
    MaskConcatBudgetMatchedHead,
    MaskConcatJLMahalanobisHead,
    RandomJLMahalanobisHead,
)
from ..training.losses import infonce_loss


def build_model_from_spec(spec: dict, cfg: dict):
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
    if kind == "mask_concat":
        return MaskConcatJLMahalanobisHead(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
            alpha=float(spec.get("alpha", 1.0)),
            beta=float(spec.get("beta", 1.0)),
            p=float(spec.get("p", 0.5)),
            mask_seed=int(spec.get("mask_seed", 0)),
            shared_raw_dim=int(cfg.get("shared_raw_dim", 768)),
            jl_eps=cfg["jl_eps"],
            jl_seed=cfg["jl_seed"],
        )
    if kind == "mask_concat_budget":
        return MaskConcatBudgetMatchedHead(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
            alpha=float(spec.get("alpha", 1.0)),
            beta=float(spec.get("beta", 1.0)),
            p=float(spec.get("p", 0.5)),
            mask_seed=int(spec.get("mask_seed", 0)),
            shared_raw_dim=int(cfg.get("shared_raw_dim", 768)),
            jl_eps=float(cfg["jl_eps"]),
            jl_seed=int(cfg["jl_seed"]),
            budget_dim=int(spec.get("budget_dim", m)),
            budget_jl_seed=int(spec.get("budget_jl_seed", 31415)),
        )
    if kind == "fedclip_proxy":
        return FedCLIPProxyHead(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
            adapter_rank=int(spec.get("adapter_rank", 16)),
            init_seed=int(cfg.get("jl_seed", 42)),
        )
    if kind == "fedclip_pretrained":
        return FedCLIPPretrainedAdapterHead(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
            adapter_rank=int(spec.get("adapter_rank", 16)),
            pretrained_model_name=str(spec.get("pretrained_model_name", cfg.get("pretrained_clip_model", "openai/clip-vit-base-patch32"))),
            jl_eps=float(cfg["jl_eps"]),
            jl_seed=int(spec.get("pretrained_proj_seed", cfg.get("jl_seed", 42))),
        )
    if kind == "fedmvp_proxy":
        return FedMVPProxyHead(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
            init_seed=int(cfg.get("jl_seed", 42)),
            prompt_scale=float(spec.get("prompt_scale", 0.05)),
        )
    if kind == "fedmvp_pretrained":
        return FedMVPPretrainedPromptHead(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
            prompt_scale=float(spec.get("prompt_scale", 0.05)),
            pretrained_model_name=str(spec.get("pretrained_model_name", cfg.get("pretrained_clip_model", "openai/clip-vit-base-patch32"))),
            jl_eps=float(cfg["jl_eps"]),
            jl_seed=int(spec.get("pretrained_proj_seed", cfg.get("jl_seed", 42))),
        )
    if kind == "mahal_only_rfull":
        return MahalanobisOnlyHead(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            shared_raw_dim=int(cfg.get("shared_raw_dim", 768)),
        )
    if kind == "mahal_only_bottleneck":
        return MahalanobisBottleneckHead(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
        )
    raise ValueError(f"unknown method kind: {kind}")


def checkpoint_path(output_root: Path, partition_id: str, method_id: str, seed: int) -> Path:
    return output_root / partition_id / method_id / f"seed{seed}" / "best.pt"


def _load_coco_primary_labels(meta_image_ids: list[int], ann_root: Path) -> list[int]:
    by_img: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for name in ["instances_train2017.json", "instances_val2017.json"]:
        p = ann_root / name
        if not p.exists():
            raise FileNotFoundError(f"missing COCO annotation file: {p}")
        obj = load_json(p)
        for ann in obj.get("annotations", []):
            img_id = int(ann["image_id"])
            cat_id = int(ann["category_id"])
            by_img[img_id][cat_id] += 1

    labels = []
    for img_id in meta_image_ids:
        hist = by_img.get(int(img_id), {})
        if not hist:
            labels.append(-1)
            continue
        top_cat = max(hist.items(), key=lambda kv: kv[1])[0]
        labels.append(int(top_cat))
    return labels


def _iid_partition(indices: list[int], n_clients: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(arr)
    chunks = np.array_split(arr, n_clients)
    return [c.astype(np.int64).tolist() for c in chunks]


def _dirichlet_partition(
    indices: list[int],
    labels: list[int],
    n_clients: int,
    alpha: float,
    seed: int,
    min_size: int,
) -> list[list[int]]:
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    idx = np.asarray(indices, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    if len(idx) != len(y):
        raise ValueError("indices and labels length mismatch")

    valid = y >= 0
    idx_valid = idx[valid]
    y_valid = y[valid]
    idx_missing = idx[~valid]

    classes = np.unique(y_valid)
    rng = np.random.default_rng(seed)

    for attempt in range(25):
        clients = [[] for _ in range(n_clients)]

        for c in classes:
            c_idx = idx_valid[y_valid == c].copy()
            rng.shuffle(c_idx)
            probs = rng.dirichlet(np.full(n_clients, alpha, dtype=np.float64))
            counts = np.floor(probs * len(c_idx)).astype(int)
            counts[-1] = len(c_idx) - int(counts[:-1].sum())
            splits = np.split(c_idx, np.cumsum(counts)[:-1])
            for i, part in enumerate(splits):
                if part.size:
                    clients[i].extend(part.tolist())

        if idx_missing.size:
            rng.shuffle(idx_missing)
            miss_chunks = np.array_split(idx_missing, n_clients)
            for i, part in enumerate(miss_chunks):
                if part.size:
                    clients[i].extend(part.tolist())

        sizes = [len(c) for c in clients]
        if min(sizes) >= min_size:
            return clients

    # Fallback: enforce minimum by moving samples from largest clients.
    clients = _iid_partition(indices, n_clients=n_clients, seed=seed)
    return clients


@torch.no_grad()
def _eval_karpathy(model, cache: KarpathyCache, split_name: str, *, device: str, batch_size: int) -> dict[str, float]:
    model = model.to(device).eval()
    img, txt, gt_i2t, gt_t2i = cache.eval_tensors(split_name)
    all_i, all_t = [], []
    for i in range(0, len(img), batch_size):
        all_i.append(model.encode_image(img[i:i + batch_size].to(device)).cpu())
    for i in range(0, len(txt), batch_size):
        all_t.append(model.encode_text(txt[i:i + batch_size].to(device)).cpu())
    zi = torch.cat(all_i, dim=0)
    zt = torch.cat(all_t, dim=0)
    return recall_at_k(zi, zt, gt_i2t=gt_i2t, gt_t2i=gt_t2i)


def _split_client_train_val(indices: list[int], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    if len(indices) <= 2:
        return indices, indices
    rng = np.random.default_rng(seed)
    arr = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(arr)
    n_val = max(1, int(len(arr) * val_frac))
    val = arr[:n_val].tolist()
    trn = arr[n_val:].tolist()
    if not trn:
        trn = val[:]
    return trn, val


def _train_local_client(
    model,
    cache: KarpathyCache,
    indices: list[int],
    *,
    seed: int,
    cfg: dict,
) -> tuple[dict[str, torch.Tensor], int]:
    device = cfg["device"]
    model = deepcopy(model).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=1e-4)

    trn_idx, _val_idx = _split_client_train_val(indices, val_frac=float(cfg.get("client_val_frac", 0.1)), seed=seed)
    ds = cache.make_train_dataset([], training=True)
    ds.indices = trn_idx
    loader = DataLoader(
        ds,
        batch_size=min(int(cfg["batch_size"]), max(1, len(trn_idx))),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=True,
    )

    local_epochs = int(cfg.get("local_epochs", 1))
    for _ in range(local_epochs):
        model.train()
        for img, txt in loader:
            img = img.to(device)
            txt = txt.to(device)
            zi, zt = model(img, txt)
            scale = model.logit_scale.exp().clamp(max=100.0) if hasattr(model, "logit_scale") else 1.0 / 0.07
            loss = infonce_loss(zi, zt, scale)
            if hasattr(model, "regularization_loss"):
                reg = model.regularization_loss()
                if reg is not None:
                    loss = loss + reg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(cfg.get("grad_clip", 1.0)))
            opt.step()
            if hasattr(model, "project_parameters"):
                model.project_parameters()

    out_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return out_state, len(trn_idx)


def _fedavg_weighted(global_state: dict[str, torch.Tensor], client_states: list[dict[str, torch.Tensor]], weights: list[int], trainable_keys: list[str]) -> dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in global_state.items()}
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum().clamp(min=1e-8)
    for key in trainable_keys:
        stacks = []
        for st in client_states:
            if key not in st:
                break
            stacks.append(st[key].float())
        if len(stacks) != len(client_states):
            continue
        t = torch.stack(stacks, dim=0)
        mix = (t * w.view(-1, *([1] * (t.ndim - 1)))).sum(dim=0)
        out[key] = mix.to(out[key].dtype)
    return out


def _encode_images(model, x: torch.Tensor, *, device: str, batch_size: int) -> torch.Tensor:
    model = model.to(device).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            outs.append(model.encode_image(x[i:i + batch_size].to(device)).cpu())
    return torch.cat(outs, dim=0)


def _linear_probe_reconstruction(z: torch.Tensor, x: torch.Tensor, *, train_frac: float = 0.8, seed: int = 0) -> dict[str, float]:
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


def _run_inversion_probe(model, cache: KarpathyCache, cfg: dict, seed: int) -> dict:
    probe_split = cfg.get("privacy_split", "train_restval")
    idx = cache.split_indices(probe_split)
    x = cache.image_feats[idx].float()
    max_probe = int(cfg.get("max_probe_samples", 0))
    if max_probe > 0 and len(x) > max_probe:
        g = torch.Generator().manual_seed(int(seed) + int(cfg.get("probe_sample_seed_offset", 982451653)))
        sel = torch.randperm(len(x), generator=g)[:max_probe]
        x = x[sel]

    z = _encode_images(model, x, device=cfg["device"], batch_size=int(cfg.get("eval_batch_size", 4096)))
    lin = _linear_probe_reconstruction(
        z,
        x,
        train_frac=float(cfg.get("attack_train_frac", 0.8)),
        seed=seed,
    )
    mlp, _xh, _xt = mlp_inversion_attack(
        z,
        x,
        train_frac=float(cfg.get("attack_train_frac", 0.8)),
        hidden_dim=int(cfg.get("attack_hidden_dim", 1024)),
        epochs=int(cfg.get("attack_epochs", 80)),
        lr=float(cfg.get("attack_lr", 1e-3)),
        batch_size=int(cfg.get("attack_batch_size", 1024)),
        seed=seed,
        device=cfg["device"],
    )
    return {
        "n_probe": int(len(x)),
        "linear_probe": lin,
        "mlp_inverter": mlp,
    }


def _partition_from_scheme(
    scheme: dict,
    train_indices: list[int],
    train_labels: list[int],
    *,
    n_clients: int,
    seed: int,
    min_size: int,
) -> list[list[int]]:
    stype = scheme["type"]
    if stype == "iid":
        return _iid_partition(train_indices, n_clients=n_clients, seed=seed)
    if stype == "dirichlet":
        return _dirichlet_partition(
            train_indices,
            train_labels,
            n_clients=n_clients,
            alpha=float(scheme["alpha"]),
            seed=seed,
            min_size=min_size,
        )
    raise ValueError(f"unknown partition type: {stype}")


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage13_federated"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)
    write_provenance = bool(cfg.get("write_provenance", True))
    write_summary = bool(cfg.get("write_summary", True))
    write_done_marker = bool(cfg.get("write_done_marker", True))

    cache_dir = Path(cfg["cache_root"]) / "coco"
    cache = KarpathyCache.from_paths(
        cache_dir / cfg["image_cache_file"],
        cache_dir / cfg["text_cache_file"],
        cache_dir / "metadata.json",
    )
    train_indices = cache.split_indices(cfg.get("train_split", "train_restval"))

    meta = load_json(cache_dir / "metadata.json")
    image_ids = [int(x) for x in meta["image_ids"]]
    coco_labels = _load_coco_primary_labels(
        image_ids,
        Path(cfg["coco_ann_root"]).resolve(),
    )
    train_labels = [int(coco_labels[i]) for i in train_indices]

    seeds = list(cfg["seeds"])
    methods = list(cfg["methods"])
    schemes = list(cfg["partitions"])
    method_filter = set(cfg.get("method_filter", []))
    partition_filter = set(cfg.get("partition_filter", []))

    if method_filter:
        methods = [m for m in methods if m["id"] in method_filter]
    if partition_filter:
        schemes = [s for s in schemes if s["id"] in partition_filter]

    metrics_for_stats = [
        "i2t_R@1", "i2t_R@5", "i2t_R@10",
        "t2i_R@1", "t2i_R@5", "t2i_R@10",
        "avg_R", "mlp_rel_error", "linear_rel_error", "comm_mb", "rounds",
    ]

    results = {
        "stage": "stage13_federated",
        "config": cfg,
        "raw": {},
        "stats": {},
    }

    for scheme in schemes:
        partition_id = scheme["id"]
        results["raw"][partition_id] = {}
        print(f"\n=== partition={partition_id} ({scheme['type']}) ===")

        for method in methods:
            method_id = method["id"]
            rows = []
            print(f"-- method={method_id}")
            for seed in seeds:
                run_dir = stage_root / partition_id / method_id / f"seed{seed}"
                eval_file = run_dir / "eval.json"
                ckpt_file = checkpoint_path(stage_root, partition_id, method_id, seed)
                run_dir.mkdir(parents=True, exist_ok=True)
                if eval_file.exists():
                    rows.append(load_json(eval_file))
                    continue

                set_seed(seed)
                model = build_model_from_spec(method, cfg)
                model = model.to(cfg["device"])
                with torch.no_grad():
                    probe_feat = cache.image_feats[train_indices[0]:train_indices[0] + 1].to(cfg["device"])
                    out_dim = int(model.encode_image(probe_feat).shape[1])

                clients = _partition_from_scheme(
                    scheme,
                    train_indices,
                    train_labels,
                    n_clients=int(cfg["n_clients"]),
                    seed=seed,
                    min_size=int(cfg.get("min_client_size", 64)),
                )
                sizes = [len(c) for c in clients]
                print(
                    f"seed={seed} clients={len(clients)} min={min(sizes)} "
                    f"median={int(np.median(sizes))} max={max(sizes)}"
                )

                trainable_keys = [k for k, p in model.named_parameters() if p.requires_grad]
                global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                n_rounds = int(cfg["rounds"])
                per_round = []
                param_count = int(sum(model.state_dict()[k].numel() for k in trainable_keys if k in model.state_dict()))
                bytes_per_round = float(param_count * 4 * int(cfg["n_clients"]) * 2)

                for r in range(n_rounds):
                    client_states = []
                    client_weights = []
                    for cid, cidx in enumerate(clients):
                        if not cidx:
                            continue
                        local_model = build_model_from_spec(method, cfg)
                        local_model.load_state_dict(global_state, strict=True)
                        st, n_seen = _train_local_client(
                            local_model,
                            cache,
                            cidx,
                            seed=seed * 1000 + r * 100 + cid,
                            cfg=cfg,
                        )
                        client_states.append(st)
                        client_weights.append(n_seen)

                    if client_states:
                        global_state = _fedavg_weighted(global_state, client_states, client_weights, trainable_keys)

                    model.load_state_dict(global_state, strict=True)
                    val_metrics = _eval_karpathy(
                        model,
                        cache,
                        cfg.get("val_split", "val"),
                        device=cfg["device"],
                        batch_size=int(cfg.get("eval_batch_size", 4096)),
                    )
                    per_round.append(
                        {
                            "round": r + 1,
                            "val_avg_R": float(val_metrics["avg_R"]),
                            "comm_mb_cumulative": bytes_per_round * float(r + 1) / (1024.0 * 1024.0),
                        }
                    )
                    print(
                        f"partition={partition_id} method={method_id} seed={seed} "
                        f"round={r+1}/{n_rounds} val_avg_R={val_metrics['avg_R']:.4f}"
                    )

                model.load_state_dict(global_state, strict=True)
                ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(global_state, ckpt_file)

                test_metrics = _eval_karpathy(
                    model,
                    cache,
                    cfg.get("test_split", "test"),
                    device=cfg["device"],
                    batch_size=int(cfg.get("eval_batch_size", 4096)),
                )
                privacy = _run_inversion_probe(model, cache, cfg, seed)

                rec = {
                    "seed": int(seed),
                    "partition": partition_id,
                    "method": method_id,
                    "embedding_output_dim": int(out_dim),
                    "embed_dim_target": int(cfg["embed_dim"]),
                    "embed_dim_matched": bool(out_dim == int(cfg["embed_dim"])),
                    "n_clients": int(cfg["n_clients"]),
                    "rounds": int(n_rounds),
                    "n_trainable_params": int(param_count),
                    "comm_mb": float(bytes_per_round * n_rounds / (1024.0 * 1024.0)),
                    "client_size": {
                        "min": int(min(sizes)),
                        "median": int(np.median(sizes)),
                        "max": int(max(sizes)),
                    },
                    "round_trace": per_round,
                    **test_metrics,
                    "mlp_rel_error": float(privacy["mlp_inverter"]["mean_relative_reconstruction_error"]),
                    "linear_rel_error": float(privacy["linear_probe"]["mean_relative_reconstruction_error"]),
                    "privacy": privacy,
                }
                save_json(rec, eval_file)
                rows.append(rec)
                print(
                    f"DONE partition={partition_id} method={method_id} seed={seed} "
                    f"test_avg_R={test_metrics['avg_R']:.4f} mlp_rel={rec['mlp_rel_error']:.4f}"
                )

            results["raw"][partition_id][method_id] = rows

        results["stats"][partition_id] = build_metric_report(
            results["raw"][partition_id],
            metrics=metrics_for_stats,
            baseline_method=cfg.get("baseline_method", "mask_concat"),
        )

    if write_provenance:
        provenance = env_snapshot(
            Path(cfg["project_root"]),
            seeds=seeds,
            extra={
                "stage": "stage13_federated",
                "elapsed_sec": time.time() - start,
                "n_clients": int(cfg["n_clients"]),
                "rounds": int(cfg["rounds"]),
            },
        )
        save_json(provenance, stage_root / "provenance_stage13.json")
    if write_summary:
        save_json(results, stage_root / "E13_federated_results.json")
    if write_done_marker:
        mark_done(markers / "stage13_federated.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 13 (federated comparison) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

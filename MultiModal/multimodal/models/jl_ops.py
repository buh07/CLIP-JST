from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def kane_nelson_jl(in_dim: int, out_dim: int, eps: float = 0.1, seed: int = 42) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    s = int(np.ceil(1.0 / eps))
    if out_dim < s:
        raise ValueError(f"out_dim={out_dim} must be >= ceil(1/eps)={s}")
    scale = 1.0 / np.sqrt(s)

    rows, cols, vals = [], [], []
    for col in range(in_dim):
        row_idx = rng.choice(out_dim, size=s, replace=False)
        signs = rng.choice([-1.0, 1.0], size=s)
        rows.extend(row_idx.tolist())
        cols.extend([col] * s)
        vals.extend((signs * scale).tolist())

    return sp.csr_matrix((vals, (rows, cols)), shape=(out_dim, in_dim), dtype=np.float32)


class DenseSparseJL(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.1, seed: int = 42):
        super().__init__()
        phi = kane_nelson_jl(in_dim, out_dim, eps=eps, seed=seed)
        self.register_buffer("Phi", torch.tensor(phi.toarray(), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Phi.T

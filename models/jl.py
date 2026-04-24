"""
Kane–Nelson sparse Johnson–Lindenstrauss transform.

Construction: for each column of Phi, sample s positions uniformly without
replacement, assign +1/sqrt(s) or -1/sqrt(s) signs from a 4-wise-independent
family (Cohen–Jayram–Nelson 2018). Store as scipy.sparse CSR.
"""

import numpy as np
import scipy.sparse as sp
import torch


def kane_nelson_jl(
    in_dim: int,
    out_dim: int,
    eps: float = 0.1,
    seed: int = 42,
) -> sp.csr_matrix:
    """
    Returns a (out_dim, in_dim) sparse JL matrix drawn from the Kane–Nelson
    distribution. Sparsity s = ceil(1/eps) non-zeros per column.

    Formally satisfies (1-eps)||x||^2 <= ||Phi x||^2 <= (1+eps)||x||^2 for
    all x in a set of Gaussian width w with m = O(eps^{-2} w^2) rows.
    """
    rng = np.random.default_rng(seed)
    s = int(np.ceil(1.0 / eps))
    if out_dim < s:
        raise ValueError(
            f"out_dim={out_dim} must be >= ceil(1/eps)={s}. "
            "Increase out_dim or use a larger eps value."
        )
    scale = 1.0 / np.sqrt(s)

    rows, cols, vals = [], [], []
    for col in range(in_dim):
        row_idx = rng.choice(out_dim, size=s, replace=False)
        signs = rng.choice([-1.0, 1.0], size=s)
        rows.extend(row_idx.tolist())
        cols.extend([col] * s)
        vals.extend((signs * scale).tolist())

    Phi = sp.csr_matrix(
        (vals, (rows, cols)), shape=(out_dim, in_dim), dtype=np.float32
    )
    return Phi


class SparseJL(torch.nn.Module):
    """
    Frozen sparse JL layer. Wraps a scipy CSR matrix as a fixed torch buffer
    (converted to dense float32 for GPU compatibility; keep sparse on CPU if
    memory is tight).
    """

    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.1, seed: int = 42):
        super().__init__()
        Phi_sp = kane_nelson_jl(in_dim, out_dim, eps=eps, seed=seed)
        Phi_dense = torch.tensor(Phi_sp.toarray(), dtype=torch.float32)
        # Register as buffer so it moves with .to(device) but is not a parameter.
        self.register_buffer("Phi", Phi_dense)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)  ->  (..., out_dim)
        return x @ self.Phi.T

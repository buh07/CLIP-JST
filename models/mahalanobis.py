"""
Learned per-modality Mahalanobis heads.

Two variants:
  - FullMahalanobis: full m×m Cholesky-parameterized PSD factor.
    f(x) = L x,  M = L^T L.  Output dim = m.
  - LowRankMahalanobis: rank-r factorization.
    f(x) = L^T (L x),  M = L^T L in R^{m×m}, rank r.  Output dim = m.
    Parameter count: 2*r*m vs m*(m+1)/2 for full.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# softplus_inverse(1.0) = log(exp(1) - 1) ≈ 0.5413
# Ensures softplus(diag_init) == 1.0, so initial Mahalanobis is truly identity.
_SOFTPLUS_INV_ONE = math.log(math.exp(1.0) - 1.0)


class FullMahalanobis(nn.Module):
    """
    Full m×m Mahalanobis head.  f(x) = L x  where L is lower-triangular with
    positive diagonal (Cholesky factor of M = L L^T, i.e. M^{1/2} = L).
    Initialized to identity so the initial pipeline is pure JL.
    """

    def __init__(self, dim: int):
        super().__init__()
        rows, cols = torch.tril_indices(dim, dim)
        n_params = rows.shape[0]
        init = torch.zeros(n_params)
        diag_mask = rows == cols
        # softplus(x) == 1 requires x = log(exp(1) - 1) ≈ 0.5413.
        init[diag_mask] = _SOFTPLUS_INV_ONE
        self.tril_params = nn.Parameter(init)
        self.dim = dim
        # Register index tensors as buffers so they follow .to(device) calls.
        self.register_buffer("_tril_rows", rows)
        self.register_buffer("_tril_cols", cols)
        self.register_buffer("_diag_mask", diag_mask)

    def _L(self) -> torch.Tensor:
        L = torch.zeros(self.dim, self.dim, device=self.tril_params.device,
                        dtype=self.tril_params.dtype)
        L[self._tril_rows, self._tril_cols] = self.tril_params
        # Enforce positive diagonal via softplus so M is strictly PD.
        diag_idx = self._tril_rows[self._diag_mask]
        L[diag_idx, diag_idx] = F.softplus(self.tril_params[self._diag_mask])
        return L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)  ->  (..., dim)
        return x @ self._L().T


class LowRankMahalanobis(nn.Module):
    """
    Rank-r Mahalanobis head.

    M = L^T L  where L in R^{r×m}, so M in R^{m×m} has rank at most r.
    f(x) = M^{1/2} x  is approximated as  L^T (L x),  keeping output dim = m.
    Parameter count: r * m per modality (vs m*(m+1)/2 for full).

    Initialized so L = [I_r | 0_{r×(m-r)}], giving M ≈ diag(1,...,1,0,...,0).
    This starts the pipeline in pure-JL mode for the first r dimensions.
    """

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.L = nn.Parameter(torch.zeros(rank, dim))
        with torch.no_grad():
            r = min(rank, dim)
            self.L[:r, :r] = torch.eye(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim) -> (..., dim)  via  L^T (L x)
        return (x @ self.L.T) @ self.L

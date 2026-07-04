from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import torch

from .ld import scipy_to_torch_sparse
from .model import predict_factorized, predict_precomputed


@dataclass
class BenchmarkResult:
    representation: str
    seconds_per_eval: float
    sparse_nnz: int
    estimated_sparse_mb: float


def _csr_memory_mb(matrix: sp.spmatrix) -> float:
    matrix = matrix.tocsr()
    return (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 1024**2


def benchmark_representations(
    A: sp.csr_matrix,
    R2: sp.csr_matrix,
    n_genes: int,
    n_mechanisms: int,
    n_repeats: int = 25,
    device: str = "cpu",
) -> list[BenchmarkResult]:
    """Compare precomputed T = R2 @ A with factorized R2 and A prediction."""

    B = torch.rand((n_genes, n_mechanisms), dtype=torch.float32, device=device) * 1e-6
    tau_raw = torch.tensor(-20.0, dtype=torch.float32, device=device)
    ld_score = torch.as_tensor(np.asarray(R2.sum(axis=1)).ravel(), dtype=torch.float32, device=device)

    A_t = scipy_to_torch_sparse(A.astype(np.float32), device=device)
    R2_t = scipy_to_torch_sparse(R2.astype(np.float32), device=device)
    T = (R2 @ A).tocsr().astype(np.float32)
    T_t = scipy_to_torch_sparse(T, device=device)

    for _ in range(3):
        predict_factorized(B, tau_raw, A_t, R2_t, ld_score, 1.0)
        predict_precomputed(B, tau_raw, T_t, ld_score, 1.0)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(n_repeats):
        predict_factorized(B, tau_raw, A_t, R2_t, ld_score, 1.0)
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    fact_seconds = (perf_counter() - start) / n_repeats

    start = perf_counter()
    for _ in range(n_repeats):
        predict_precomputed(B, tau_raw, T_t, ld_score, 1.0)
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    pre_seconds = (perf_counter() - start) / n_repeats

    return [
        BenchmarkResult(
            "factorized",
            fact_seconds,
            int(A.nnz + R2.nnz),
            _csr_memory_mb(A) + _csr_memory_mb(R2),
        ),
        BenchmarkResult("precomputed_T", pre_seconds, int(T.nnz), _csr_memory_mb(T)),
    ]

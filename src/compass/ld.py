from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_positional_ld(
    variants: pd.DataFrame,
    window_bp: int = 1_000_000,
    decay_bp: float = 100_000.0,
    min_r2: float = 1e-4,
    include_diagonal: bool = True,
) -> sp.csr_matrix:
    """Build a sparse positive LD proxy from genomic distance.

    This is a fallback for development and small public-data tests. For real
    inference, replace this matrix with reference-panel squared correlations.
    The interface matches the PolyFun/LDSC operation: LD scores are R2 @ annot.
    """

    variants = variants.sort_values(["chrom", "pos", "variant_idx"]).reset_index(drop=True)
    n = variants.shape[0]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for _, group in variants.groupby("chrom", sort=False):
        idx = group["variant_idx"].to_numpy(np.int64)
        pos = group["pos"].to_numpy(np.int64)
        for local_i, global_i in enumerate(idx):
            lo = np.searchsorted(pos, pos[local_i] - window_bp, side="left")
            hi = np.searchsorted(pos, pos[local_i] + window_bp, side="right")
            js = idx[lo:hi]
            dist = np.abs(pos[lo:hi] - pos[local_i]).astype(float)
            r2 = np.exp(-dist / decay_bp)
            if not include_diagonal:
                keep = js != global_i
                js = js[keep]
                r2 = r2[keep]
            keep = r2 >= min_r2
            if keep.any():
                rows.append(np.full(int(keep.sum()), global_i, dtype=np.int64))
                cols.append(js[keep].astype(np.int64))
                vals.append(r2[keep].astype(np.float32))
    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        val = np.concatenate(vals)
    else:
        row = col = np.array([], dtype=np.int64)
        val = np.array([], dtype=np.float32)
    return sp.coo_matrix((val, (row, col)), shape=(n, n), dtype=np.float32).tocsr()


def annotation_triples_to_csr(triples: pd.DataFrame, n_variants: int, n_genes: int, n_mechanisms: int) -> sp.csr_matrix:
    """Flatten variant-gene-mechanism triples into A[variant, gene * L + mechanism]."""

    row = triples["variant_idx"].to_numpy(np.int64)
    col = (
        triples["gene_idx"].to_numpy(np.int64) * n_mechanisms
        + triples["mechanism_idx"].to_numpy(np.int64)
    )
    val = triples["value"].to_numpy(np.float32)
    return sp.coo_matrix((val, (row, col)), shape=(n_variants, n_genes * n_mechanisms)).tocsr()


def scipy_to_torch_sparse(matrix: sp.spmatrix, device: str = "cpu"):
    import torch

    coo = matrix.tocoo()
    indices = np.vstack([coo.row, coo.col])
    return torch.sparse_coo_tensor(
        torch.as_tensor(indices, dtype=torch.long, device=device),
        torch.as_tensor(coo.data, dtype=torch.float32, device=device),
        size=coo.shape,
        device=device,
    ).coalesce()

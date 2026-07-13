"""Simulation primitives for validating COMPASS against ABC/LD-derived truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .model import LdChromosomeBlock


@dataclass(frozen=True)
class SparseContextTruth:
    """Direct per-variant heritability generated from sparse ABC contexts."""

    direct_h2: np.ndarray
    context_h2: np.ndarray
    selected_counts: np.ndarray
    selected_variant_counts: np.ndarray


def context_variant_scores(
    annotation: sp.csr_matrix,
    n_genes: int,
    n_mechanisms: int,
    mechanism_indices: list[int],
) -> np.ndarray:
    """Sum ABC scores over linked genes for each requested context and variant."""

    if annotation.shape[1] != n_genes * n_mechanisms:
        raise ValueError("annotation shape does not match gene/mechanism dimensions")
    if any(index < 0 or index >= n_mechanisms for index in mechanism_indices):
        raise ValueError("mechanism index is out of range")
    scores = np.empty((annotation.shape[0], len(mechanism_indices)), dtype=np.float32)
    for output_index, mechanism_index in enumerate(mechanism_indices):
        columns = np.arange(mechanism_index, annotation.shape[1], n_mechanisms, dtype=np.int64)
        scores[:, output_index] = np.asarray(annotation[:, columns].sum(axis=1)).ravel().astype(np.float32)
    return scores


def generate_sparse_context_truth(
    context_scores: np.ndarray,
    context_h2: np.ndarray,
    causal_fraction: float,
    seed: int,
) -> SparseContextTruth:
    """Select sparse causal variants and normalize each context to its target h2.

    Each nonzero context samples independently from variants with positive ABC
    score. This retains real cross-context overlap; a variant selected by both
    causal contexts receives the sum of its direct contributions.
    """

    scores = np.asarray(context_scores, dtype=np.float64)
    targets = np.asarray(context_h2, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("context_scores must be two-dimensional")
    if scores.shape[1] != targets.size:
        raise ValueError("context_scores and context_h2 disagree on context count")
    if not 0 < causal_fraction <= 1:
        raise ValueError("causal_fraction must be in (0, 1]")
    if np.any(targets < 0):
        raise ValueError("context_h2 must be non-negative")

    rng = np.random.default_rng(seed)
    direct_h2 = np.zeros(scores.shape[0], dtype=np.float64)
    selected_counts = np.zeros(targets.size, dtype=np.int64)
    selected_variant_counts = np.zeros(targets.size, dtype=np.int64)
    for context_index, target_h2 in enumerate(targets):
        if target_h2 == 0:
            continue
        eligible = np.flatnonzero(scores[:, context_index] > 0)
        if eligible.size == 0:
            raise ValueError(f"causal context {context_index} has no ABC-positive variants")
        n_selected = max(1, int(np.floor(causal_fraction * eligible.size)))
        selected = rng.choice(eligible, size=n_selected, replace=False)
        weights = scores[selected, context_index]
        direct_h2[selected] += target_h2 * weights / weights.sum()
        selected_counts[context_index] = n_selected
        selected_variant_counts[context_index] = np.unique(selected).size

    return SparseContextTruth(
        direct_h2=direct_h2.astype(np.float32),
        context_h2=targets.astype(np.float32),
        selected_counts=selected_counts,
        selected_variant_counts=selected_variant_counts,
    )


def ld_propagate(blocks: list[LdChromosomeBlock], vector: np.ndarray) -> np.ndarray:
    """Apply chromosome-level R2 blocks without materializing a genome-wide matrix."""

    values = np.asarray(vector, dtype=np.float32)
    propagated = np.zeros(values.size, dtype=np.float32)
    for block in blocks:
        rows = np.asarray(block.rows, dtype=np.int64)
        matrix = block.R2.tocsr()
        if matrix.dtype == np.float16:
            matrix = sp.csr_matrix(
                (matrix.data.astype(np.float32), matrix.indices, matrix.indptr),
                shape=matrix.shape,
            )
        propagated[rows] = matrix @ values[rows]
    return propagated


def simulate_noncentral_chisq(
    ld_propagated_h2: np.ndarray,
    n_eff: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw independent model-matched chi-square GWAS statistics.

    The noncentrality is ``n_eff * R2 @ h_direct``. The real UKBB R2 operator
    therefore determines LD tagging while the draw supplies per-variant GWAS
    sampling noise without a separate approximation to correlated Z noise.
    """

    if n_eff <= 0:
        raise ValueError("n_eff must be positive")
    noncentrality = np.maximum(np.asarray(ld_propagated_h2, dtype=np.float64) * float(n_eff), 0.0)
    chisq = np.random.default_rng(seed).noncentral_chisquare(df=1.0, nonc=noncentrality).astype(np.float32)
    return chisq, noncentrality.astype(np.float32)


def normalized_context_mass(B: np.ndarray, mechanism_indices: list[int]) -> np.ndarray:
    """Return fitted coefficient mass normalized over the requested contexts."""

    values = np.asarray(B, dtype=np.float64)
    mass = values[:, mechanism_indices].sum(axis=0)
    total = mass.sum()
    return (mass / total if total > 0 else np.zeros_like(mass)).astype(np.float64)


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Correlation with a defined result for degenerate vectors."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("vectors must have equal length")
    if x.size == 0 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

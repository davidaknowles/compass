"""Model-matched S-LDSC baseline for gene-aggregated ABC annotations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class SldscResult:
    """Weighted least-squares S-LDSC estimates for one annotation design."""

    annotation_names: list[str]
    coefficients: np.ndarray
    implied_h2: np.ndarray
    weighted_mse: float
    condition_number: float
    n_variants: int

    def as_dict(self) -> dict:
        return {
            "annotation_names": self.annotation_names,
            "coefficients": self.coefficients.tolist(),
            "implied_h2": self.implied_h2.tolist(),
            "weighted_mse": self.weighted_mse,
            "condition_number": self.condition_number,
            "n_variants": self.n_variants,
        }


def fit_sldsc_wls(
    ld_blocks,
    annotation_scores: np.ndarray,
    annotation_names: list[str],
    chisq: np.ndarray,
    n_samples: float | np.ndarray,
    selected_annotations: list[int] | None = None,
) -> SldscResult:
    """Fit the infinitesimal ABC/S-LDSC regression described in ``main.tex``.

    The regression is ``chi2 - 1 = N * (sum_l theta_l R2 A_l +
    tau R2 1)``. ``R2 1`` is the residual non-mediated LD score. Parameters
    are unconstrained, as in standard joint S-LDSC regression, and use the
    same ``1 / max(chi2, 1)`` stabilizing weights as COMPASS.
    """

    scores = np.asarray(annotation_scores, dtype=np.float64)
    y = np.asarray(chisq, dtype=np.float64) - 1.0
    if scores.ndim != 2 or scores.shape[0] != y.size:
        raise ValueError("annotation_scores must have one row per chi-square statistic")
    if scores.shape[1] != len(annotation_names):
        raise ValueError("annotation_names must match annotation_scores columns")
    if np.ndim(n_samples) == 0:
        sample_sizes = None
        scalar_n = float(n_samples)
        if scalar_n <= 0:
            raise ValueError("n_samples must be positive")
    else:
        sample_sizes = np.asarray(n_samples, dtype=np.float64)
        if sample_sizes.shape != y.shape or np.any(sample_sizes <= 0):
            raise ValueError("n_samples must be positive and have one value per variant")
        scalar_n = None
    selected = list(range(scores.shape[1])) if selected_annotations is None else list(selected_annotations)
    if not selected:
        raise ValueError("at least one annotation is required")
    if any(index < 0 or index >= scores.shape[1] for index in selected):
        raise ValueError("selected annotation index is out of range")

    n_columns = len(selected) + 1  # plus non-mediated LD score
    xtwx = np.zeros((n_columns, n_columns), dtype=np.float64)
    xtwy = np.zeros(n_columns, dtype=np.float64)
    weighted_y2 = 0.0
    annotation_totals = np.zeros(len(selected), dtype=np.float64)
    total_weight = 0.0
    for block in ld_blocks:
        rows = np.asarray(block.rows, dtype=np.int64)
        matrix = block.R2.tocsr()
        if matrix.dtype == np.float16:
            matrix = sp.csr_matrix(
                (matrix.data.astype(np.float32), matrix.indices, matrix.indptr), shape=matrix.shape
            )
        annotation_ld_scores = matrix @ scores[rows][:, selected]
        residual_ld_score = np.asarray(matrix.sum(axis=1)).ravel()
        design = np.column_stack((annotation_ld_scores, residual_ld_score))
        n_block = scalar_n if sample_sizes is None else sample_sizes[rows]
        design *= n_block[:, None] if np.ndim(n_block) else n_block
        y_block = y[rows]
        weights = 1.0 / np.maximum(np.asarray(chisq, dtype=np.float64)[rows], 1.0)
        weighted_design = design * weights[:, None]
        xtwx += design.T @ weighted_design
        xtwy += design.T @ (weights * y_block)
        weighted_y2 += float(np.sum(weights * np.square(y_block)))
        total_weight += float(weights.sum())
        annotation_totals += scores[rows][:, selected].sum(axis=0)

    scale = np.sqrt(np.maximum(np.diag(xtwx), np.finfo(float).tiny))
    scaled_normal = xtwx / np.outer(scale, scale)
    scaled_rhs = xtwy / scale
    scaled_coefficients = np.linalg.pinv(scaled_normal, rcond=1e-10) @ scaled_rhs
    coefficients = scaled_coefficients / scale
    weighted_sse = weighted_y2 - 2.0 * coefficients @ xtwy + coefficients @ xtwx @ coefficients
    implied_h2 = np.concatenate((coefficients[:-1] * annotation_totals, [coefficients[-1] * y.size]))
    return SldscResult(
        annotation_names=[annotation_names[index] for index in selected] + ["residual_ld_score"],
        coefficients=coefficients,
        implied_h2=implied_h2,
        weighted_mse=float(weighted_sse / max(total_weight, 1.0)),
        condition_number=float(np.linalg.cond(scaled_normal)),
        n_variants=int(y.size),
    )


def fit_sldsc_panel(
    ld_blocks,
    annotation_scores: np.ndarray,
    annotation_names: list[str],
    chisq: np.ndarray,
    n_samples: float | np.ndarray,
) -> dict[str, dict]:
    """Fit one S-LDSC model per context and a joint five-context model."""

    univariate = {
        annotation_names[index]: fit_sldsc_wls(
            ld_blocks, annotation_scores, annotation_names, chisq, n_samples, selected_annotations=[index]
        ).as_dict()
        for index in range(len(annotation_names))
    }
    joint = fit_sldsc_wls(ld_blocks, annotation_scores, annotation_names, chisq, n_samples).as_dict()
    return {"univariate": univariate, "joint": joint}

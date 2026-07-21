from __future__ import annotations

import math

import pandas as pd


def summarize_hierarchical_run(
    metadata: dict,
    contributions: pd.DataFrame,
    expected_context: str,
) -> dict:
    """Summarize CV performance and the leading biological context of a fit."""

    raw_scores = metadata.get("cv_scores")
    if not raw_scores:
        raise ValueError("hierarchical result has no CV scores")
    scores = {float(key): float(value) for key, value in raw_scores.items()}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("CV scores must be finite")
    best_lambda = float(metadata["best_lambda"])
    if best_lambda not in scores:
        raise ValueError("best lambda is absent from CV scores")

    required = {"context", "total_h2", "fraction"}
    if not required.issubset(contributions.columns):
        raise ValueError(f"context contribution table requires columns: {sorted(required)}")
    biological = contributions.loc[contributions["context"] != "intercept"].copy()
    if biological.empty:
        raise ValueError("context contribution table has no biological contexts")
    biological["total_h2"] = pd.to_numeric(biological["total_h2"], errors="raise")
    leader = biological.loc[biological["total_h2"].idxmax()]
    biological_total = float(biological["total_h2"].sum())
    context_only_lambda = max(scores)
    context_only_score = scores[context_only_lambda]
    best_score = scores[best_lambda]
    improvement = context_only_score - best_score
    return {
        "best_lambda": best_lambda,
        "best_cv_mse": best_score,
        "context_only_lambda": context_only_lambda,
        "context_only_cv_mse": context_only_score,
        "cv_mse_improvement": improvement,
        "cv_mse_relative_improvement": (
            improvement / context_only_score if context_only_score != 0 else math.nan
        ),
        "leading_context": str(leader["context"]),
        "leading_fraction_all": float(leader["fraction"]),
        "leading_fraction_biological": (
            float(leader["total_h2"]) / biological_total if biological_total > 0 else math.nan
        ),
        "expected_context": expected_context,
        "expected_context_is_leading": str(leader["context"]) == expected_context,
    }

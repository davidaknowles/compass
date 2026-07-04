"""COMPASS prototype implementation."""

from .data import load_top_assoc_annotations, load_gwas_sumstats, make_training_table
from .model import (
    CompassDataset,
    FitResult,
    fit_nuclear_norm_path,
    fit_rank1_path,
    predict_precomputed,
    predict_factorized,
)

__all__ = [
    "CompassDataset",
    "FitResult",
    "fit_nuclear_norm_path",
    "fit_rank1_path",
    "load_gwas_sumstats",
    "load_top_assoc_annotations",
    "make_training_table",
    "predict_factorized",
    "predict_precomputed",
]

"""COMPASS prototype implementation."""

from .data import load_top_assoc_annotations, load_gwas_sumstats, make_training_table

_MODEL_EXPORTS = {
    "CompassDataset",
    "FitResult",
    "fit_nuclear_norm_path",
    "fit_rank1_path",
    "predict_factorized",
    "predict_precomputed",
}

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


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

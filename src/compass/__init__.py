"""COMPASS prototype implementation."""

from .data import load_abc_annotations, load_gwas_sumstats, load_top_assoc_annotations, make_training_table

_MODEL_EXPORTS = {
    "CompassDataset",
    "FitResult",
    "LdChromosomeBlock",
    "fit_nuclear_norm_path",
    "fit_rank1_path",
    "predict_factorized",
}

__all__ = [
    "CompassDataset",
    "FitResult",
    "LdChromosomeBlock",
    "fit_nuclear_norm_path",
    "fit_rank1_path",
    "load_abc_annotations",
    "load_gwas_sumstats",
    "load_top_assoc_annotations",
    "make_training_table",
    "predict_factorized",
]


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

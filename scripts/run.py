#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

from compass.data import (
    cv_cache_path,
    load_abc_annotations,
    load_cv_cache,
    load_gwas_sumstats,
    load_open_chromatin_tss_annotations,
    load_peak_context_annotations,
    load_top_assoc_annotations,
    make_training_table,
    write_cv_cache,
)
from compass.ld import (
    annotation_triples_to_csr,
    build_ukbb_ld_r2_by_chromosome,
    filter_variants_to_ukbb_ld,
    make_ld_component_cv_groups,
)
from compass.model import (
    CompassDataset,
    LdChromosomeBlock,
    aggregate_context_annotations,
    context_heritability_components,
    fit_hierarchical_nuclear_path,
    fit_nuclear_norm_path,
    fit_rank1_path,
)


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"
DEFAULT_ABC_NAME = "AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz"
DEFAULT_ABC_CELL_TYPES = (
    "astrocyte-ENCODE,"
    "bipolar_neuron_from_iPSC-ENCODE,"
    "H1_Derived_Neuronal_Progenitor_Cultured_Cells-Roadmap,"
    "CD14-positive_monocyte-ENCODE,"
    "CD14-positive_monocytes-Roadmap,"
    "THP-1_macrophage-VanBortle2017"
)
CONTROL_ABC_CELL_TYPES = (
    "white_adipose-Loft2014,"
    "gastrocnemius_medialis-ENCODE,"
    "uterus-ENCODE"
)
ABC_CONTEXT_PANELS = {
    "ad_proximal": DEFAULT_ABC_CELL_TYPES,
    "ad_with_controls": f"{DEFAULT_ABC_CELL_TYPES},{CONTROL_ABC_CELL_TYPES}",
}
PEAK_ASSAY_FILES = {
    "ATAC": {
        "astrocyte": "LHX2_optimal_peak_IDR_ENCODE.ATAC.bed",
        "microglia": "PU1_optimal_peak_IDR_ENCODE.ATAC.bed",
        "neuron": "NeuN_optimal_peak_IDR_ENCODE.ATAC.bed",
        "oligodendrocyte": "Olig2_optimal_peak_IDR_ENCODE.ATAC.bed",
    },
    "H3K27ac": {
        "astrocyte": "LHX2_optimal_peak.H3K27.bed",
        "microglia": "PU1_optimal_peak.H3K27.bed",
        "neuron": "NeuN_optimal_peak.H3K27.bed",
        "oligodendrocyte": "Olig2_optimal_peak.H3K27.bed",
    },
    "H3K4me3": {
        "astrocyte": "LHX2_optimal_peak.H3K4me3.bed",
        "microglia": "PU1_optimal_peak.H3K4me3.bed",
        "neuron": "NeuN_optimal_peak.H3K4me3.bed",
        "oligodendrocyte": "Olig2_optimal_peak.H3K4me3.bed",
    },
}


def _parse_lambdas(value: str) -> list[float]:
    lambdas = [float(x) for x in value.split(",") if x.strip()]
    if not lambdas:
        raise argparse.ArgumentTypeError("at least one lambda is required")
    return lambdas


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@contextmanager
def _timed(label: str):
    start = perf_counter()
    print(f"[setup] start {label}", flush=True)
    yield
    print(f"[setup] done {label}: {perf_counter() - start:.2f}s", flush=True)


def _cache_key(
    args,
    gwas_path: Path,
    abc_path: Path,
    open_chromatin_tss_path: Path,
    open_chromatin_peaks_root: Path,
) -> str:
    intercept = "intercept" if not args.no_intercept else "nointercept"
    n_samples = "gwas" if args.n_samples is None else f"n{args.n_samples:g}"
    if args.annotation_source == "abc":
        if args.abc_cell_types.lower() == "all":
            cell_types = "all"
        else:
            cell_types = hashlib.sha1(args.abc_cell_types.encode("utf-8")).hexdigest()[:12]
        annotation_path = hashlib.sha1(str(abc_path.resolve()).encode("utf-8")).hexdigest()[:12]
        source = (
            f"abc.allrows.{args.abc_score_column}.min{args.abc_min_score:g}."
            f"r2ge{args.ld_r2_cutoff:g}.chromfp16.{cell_types}.{annotation_path}"
        )
    elif args.annotation_source == "open_chromatin":
        source_paths = (
            f"{open_chromatin_tss_path.resolve()}|{open_chromatin_peaks_root.resolve()}|{args.peak_assay}"
        )
        source_hash = hashlib.sha1(source_paths.encode("utf-8")).hexdigest()[:12]
        context_suffix = ".flatctx" if getattr(args, "context_annotation_source", "gene_aggregate") == "peak" else ""
        source = (
            f"peaktss.{args.peak_assay}.tsswin{args.open_chromatin_tss_window:g}."
            f"{source_hash}{context_suffix}"
        )
    else:
        source = f"topassoc.{args.annotation_value}"
    return f"{source}.{intercept}.{gwas_path.name}.{n_samples}"


def _frame_path(path: Path, extension: str) -> Path:
    return Path(f"{path}{extension}")


def _read_frame(path: Path) -> pd.DataFrame:
    parquet = _frame_path(path, ".parquet")
    pickle = _frame_path(path, ".pkl")
    legacy_parquet = path.with_suffix(".parquet")
    legacy_pickle = path.with_suffix(".pkl")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if pickle.exists():
        return pd.read_pickle(pickle)
    if legacy_parquet.exists():
        return pd.read_parquet(legacy_parquet)
    return pd.read_pickle(legacy_pickle)


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(_frame_path(path, ".parquet"), index=False)
    except ImportError:
        df.to_pickle(_frame_path(path, ".pkl"))


def _load_gwas_cached(gwas_path: Path, cache_dir: Path, rebuild: bool) -> pd.DataFrame:
    cache = cache_dir / f"{gwas_path.name}.normalized"
    if not rebuild and (
        _frame_path(cache, ".parquet").exists()
        or _frame_path(cache, ".pkl").exists()
        or cache.with_suffix(".parquet").exists()
        or cache.with_suffix(".pkl").exists()
    ):
        with _timed("load cached GWAS"):
            return _read_frame(cache)
    with _timed("parse GWAS"):
        gwas = load_gwas_sumstats(gwas_path)
    with _timed("write GWAS cache"):
        _write_frame(gwas, cache)
    return gwas


def _cache_paths(cache_dir: Path, key: str) -> dict[str, Path]:
    prefix = cache_dir / key
    return {
        "A": prefix.with_suffix(".A.npz"),
        "R2": prefix.with_suffix(".R2.npz"),
        "R2_dir": Path(f"{prefix}.R2.chroms"),
        "arrays": prefix.with_suffix(".arrays.npz"),
        "genes": prefix.with_suffix(".genes"),
        "mechanisms": prefix.with_suffix(".mechanisms.json"),
        "ld_diagnostics": prefix.with_suffix(".ld_diagnostics"),
        "metadata": prefix.with_suffix(".metadata.json"),
    }


def _legacy_abc_cache_paths(cache_dir: Path, args, gwas_path: Path) -> dict[str, Path] | None:
    """Reuse pre-component-CV caches without copying their large LD archives."""

    if args.annotation_source != "abc":
        return None
    intercept = "intercept" if not args.no_intercept else "nointercept"
    if args.abc_cell_types.lower() == "all":
        cell_types = "all"
    else:
        cell_types = hashlib.sha1(args.abc_cell_types.encode("utf-8")).hexdigest()[:12]
    pattern = (
        f"abc.allrows.{args.abc_score_column}.min{args.abc_min_score:g}.folds*.gap*."
        f"r2ge{args.ld_r2_cutoff:g}.chromfp16.{cell_types}.{intercept}.{gwas_path.name}*.A.npz"
    )
    matches = sorted(cache_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        return None
    prefix = matches[0].name.removesuffix(".A.npz")
    legacy_prefix = cache_dir / prefix
    return {
        "A": Path(f"{legacy_prefix}.A.npz"),
        "R2": Path(f"{legacy_prefix}.R2.npz"),
        "R2_dir": Path(f"{legacy_prefix}.gwas.R2.chroms"),
        "arrays": Path(f"{legacy_prefix}.arrays.npz"),
        "genes": Path(f"{legacy_prefix}.genes"),
        "mechanisms": Path(f"{legacy_prefix}.mechanisms.json"),
        "ld_diagnostics": Path(f"{legacy_prefix}.ld_diagnostics"),
        "metadata": Path(f"{legacy_prefix}.metadata.json"),
    }


def _dataset_cache_exists(paths: dict[str, Path]) -> bool:
    genes_exists = _frame_path(paths["genes"], ".parquet").exists() or _frame_path(paths["genes"], ".pkl").exists()
    diagnostics_exists = _frame_path(paths["ld_diagnostics"], ".parquet").exists() or _frame_path(
        paths["ld_diagnostics"], ".pkl"
    ).exists()
    r2_exists = paths["R2"].exists() or (
        paths["R2_dir"].exists()
        and any((paths["R2_dir"] / name).exists() for name in ("manifest.uncompressed.json", "manifest.json"))
    )
    return (
        paths["A"].exists()
        and r2_exists
        and paths["arrays"].exists()
        and paths["mechanisms"].exists()
        and genes_exists
        and diagnostics_exists
    )


def _ld_manifest_path(r2_dir: Path) -> Path:
    for name in ("manifest.uncompressed.json", "manifest.json"):
        path = r2_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No LD manifest found in {r2_dir}")


def _load_ld_blocks_from_dir(r2_dir: Path) -> list[LdChromosomeBlock]:
    with open(_ld_manifest_path(r2_dir), encoding="utf-8") as handle:
        manifest = json.load(handle)
    reference_dir = manifest.get("reference_dir")
    if reference_dir is not None:
        reference = Path(reference_dir).expanduser()
        if not reference.is_absolute():
            reference = (r2_dir / reference).resolve()
        if reference.resolve() == r2_dir.resolve():
            raise ValueError(f"LD cache {r2_dir} references itself")
        return _load_ld_blocks_from_dir(reference)
    return [
        LdChromosomeBlock(
            chrom=int(item["chrom"]),
            rows=np.load(r2_dir / item["rows"], allow_pickle=False),
            R2=_load_csr_npz(r2_dir / item["matrix"]),
        )
        for item in manifest["blocks"]
    ]


def _ld_blocks_match_chromosome_rows(ld_blocks: list[LdChromosomeBlock], chrom: np.ndarray) -> bool:
    if sum(block.rows.size for block in ld_blocks) != chrom.size:
        return False
    for block in ld_blocks:
        expected_rows = np.flatnonzero(chrom == block.chrom)
        if not np.array_equal(np.asarray(block.rows, dtype=np.int64), expected_rows):
            return False
    return True


def _load_ld_chromosome_rows(r2_dir: Path) -> list[tuple[int, np.ndarray]]:
    """Load only chromosome row indices, without materializing LD matrices."""

    with open(_ld_manifest_path(r2_dir), encoding="utf-8") as handle:
        manifest = json.load(handle)
    reference_dir = manifest.get("reference_dir")
    if reference_dir is not None:
        reference = Path(reference_dir).expanduser()
        if not reference.is_absolute():
            reference = (r2_dir / reference).resolve()
        if reference.resolve() == r2_dir.resolve():
            raise ValueError(f"LD cache {r2_dir} references itself")
        return _load_ld_chromosome_rows(reference)
    return [
        (int(item["chrom"]), np.load(r2_dir / item["rows"], allow_pickle=False))
        for item in manifest["blocks"]
    ]


def _ld_chromosome_rows_match(rows_by_chrom: list[tuple[int, np.ndarray]], chrom: np.ndarray) -> bool:
    if sum(rows.size for _, rows in rows_by_chrom) != chrom.size:
        return False
    for chromosome, rows in rows_by_chrom:
        if not np.array_equal(np.asarray(rows, dtype=np.int64), np.flatnonzero(chrom == chromosome)):
            return False
    return True


def _ld_diagnostics_for_dir(r2_dir: Path) -> pd.DataFrame:
    name = r2_dir.name
    for suffix in (".gwas.R2.chroms", ".R2.chroms"):
        if name.endswith(suffix):
            diagnostics = r2_dir.parent / f"{name.removesuffix(suffix)}.ld_diagnostics"
            if _frame_path(diagnostics, ".parquet").exists() or _frame_path(diagnostics, ".pkl").exists():
                return _read_frame(diagnostics)
    return pd.DataFrame()


def _find_compatible_allrow_ld(
    cache_dir: Path,
    chrom: np.ndarray,
    r2_cutoff: float,
) -> tuple[list[LdChromosomeBlock], pd.DataFrame, Path] | None:
    """Load a verified all-row chromosome LD archive for a new ABC context panel."""

    pattern = f"abc.allrows*.r2ge{r2_cutoff:g}.chromfp16.*.R2.chroms"
    candidates = sorted(cache_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            rows_by_chrom = _load_ld_chromosome_rows(candidate)
        except (FileNotFoundError, KeyError, OSError, ValueError):
            continue
        if _ld_chromosome_rows_match(rows_by_chrom, chrom):
            blocks = _load_ld_blocks_from_dir(candidate)
            return blocks, _ld_diagnostics_for_dir(candidate), candidate.resolve()
    return None


def _load_dataset_cache(paths: dict[str, Path]):
    with _timed("load dataset cache"):
        A = sp.load_npz(paths["A"])
        R2 = sp.load_npz(paths["R2"]) if paths["R2"].exists() else None
        ld_blocks = None
        if R2 is None:
            ld_blocks = _load_ld_blocks_from_dir(paths["R2_dir"])
        arrays = np.load(paths["arrays"], allow_pickle=False)
        genes = _read_frame(paths["genes"])
        ld_diagnostics = _read_frame(paths["ld_diagnostics"])
        with open(paths["mechanisms"], encoding="utf-8") as handle:
            mechanisms = json.load(handle)
    dataset = CompassDataset(
        A=A,
        chisq=arrays["chisq"],
        chrom=arrays["chrom"],
        n_samples=arrays["n_samples"],
        position=arrays["position"] if "position" in arrays and arrays["position"].size else None,
        R2=R2,
        ld_blocks=ld_blocks,
        cv_groups=arrays["cv_groups"] if "cv_groups" in arrays and np.any(arrays["cv_groups"] >= 0) else None,
        cv_score_groups=(
            arrays["cv_score_groups"]
            if "cv_score_groups" in arrays and np.any(arrays["cv_score_groups"] >= 0)
            else None
        ),
    )
    return dataset, genes, mechanisms, ld_diagnostics, str(arrays["n_samples_source"])


def _save_csr_npz(path: Path, matrix: sp.csr_matrix, data_dtype=np.float16) -> None:
    matrix = matrix.tocsr()
    np.savez(
        path,
        data=matrix.data.astype(data_dtype),
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape, dtype=np.int64),
    )


def _load_csr_npz(path: Path) -> sp.csr_matrix:
    if path.name.endswith(".scipy.npz"):
        return sp.load_npz(path)
    arrays = np.load(path, allow_pickle=False)
    return sp.csr_matrix(
        (
            arrays["data"],
            arrays["indices"],
            arrays["indptr"],
        ),
        shape=tuple(arrays["shape"]),
    )


def _write_dataset_cache(
    paths: dict[str, Path],
    dataset: CompassDataset,
    genes: pd.DataFrame,
    mechanisms: list[str],
    ld_diagnostics: pd.DataFrame,
    n_samples_source: str,
    ld_reference_dir: Path | None = None,
) -> None:
    with _timed("write dataset cache"):
        sp.save_npz(paths["A"], dataset.A)
        if dataset.ld_blocks is None:
            sp.save_npz(paths["R2"], dataset.R2)
        else:
            paths["R2_dir"].mkdir(parents=True, exist_ok=True)
            if ld_reference_dir is not None:
                manifest = {
                    "representation": "chromosome_reference",
                    "reference_dir": os.path.relpath(ld_reference_dir.resolve(), paths["R2_dir"].resolve()),
                }
            else:
                manifest = {"representation": "chromosome", "blocks": []}
                for block in dataset.ld_blocks:
                    matrix_name = f"chr{block.chrom}.R2.fp16.npz"
                    rows_name = f"chr{block.chrom}.rows.npy"
                    _save_csr_npz(paths["R2_dir"] / matrix_name, block.R2, data_dtype=np.float16)
                    np.save(paths["R2_dir"] / rows_name, np.asarray(block.rows, dtype=np.int64))
                    manifest["blocks"].append({"chrom": int(block.chrom), "matrix": matrix_name, "rows": rows_name})
            with open(paths["R2_dir"] / "manifest.json", "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
        np.savez_compressed(
            paths["arrays"],
            chisq=dataset.chisq,
            chrom=dataset.chrom,
            position=(
                np.asarray(dataset.position, dtype=np.int64)
                if dataset.position is not None
                else np.asarray([], dtype=np.int64)
            ),
            cv_groups=(
                np.asarray(dataset.cv_groups, dtype=np.int64)
                if dataset.cv_groups is not None
                else np.full(dataset.n_variants, -1, dtype=np.int64)
            ),
            cv_score_groups=(
                np.asarray(dataset.cv_score_groups, dtype=np.int64)
                if dataset.cv_score_groups is not None
                else np.full(dataset.n_variants, -1, dtype=np.int64)
            ),
            n_samples=np.asarray(dataset.n_samples, dtype=np.float32),
            n_samples_source=np.asarray(n_samples_source),
        )
        _write_frame(genes, paths["genes"])
        _write_frame(ld_diagnostics, paths["ld_diagnostics"])
        with open(paths["mechanisms"], "w", encoding="utf-8") as handle:
            json.dump(mechanisms, handle)


def _load_initial_B(path: str | Path, genes: pd.DataFrame, mechanisms: list[str]) -> np.ndarray:
    initial = pd.read_csv(Path(path).expanduser(), sep="\t", index_col=0)
    initial.index = initial.index.astype(str)
    initial.columns = initial.columns.astype(str)
    expected_genes = genes["gene"].astype(str)
    missing_mechanisms = set(mechanisms).difference(initial.columns)
    if missing_mechanisms:
        raise ValueError(f"Initial B is missing mechanisms: {sorted(missing_mechanisms)}")
    return (
        initial.reindex(index=expected_genes, columns=mechanisms, fill_value=0.0)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _load_context_effects(
    path: str | Path, mechanisms: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(Path(path).expanduser(), sep="\t")
    required = {"context", "effect"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Context effect table requires columns: {sorted(required)}")
    if frame["context"].duplicated().any():
        raise ValueError("Context effect names must be unique")
    indexed = frame.set_index(frame["context"].astype(str))
    missing = set(mechanisms).difference(indexed.index)
    if missing:
        raise ValueError(f"Context effect table is missing mechanisms: {sorted(missing)}")
    effects = pd.to_numeric(indexed.reindex(mechanisms)["effect"], errors="raise").to_numpy(np.float32)
    if np.any(effects < 0):
        raise ValueError("Fixed context effects must be non-negative")
    if "standard_error" in indexed:
        standard_errors = pd.to_numeric(
            indexed.reindex(mechanisms)["standard_error"], errors="coerce"
        ).to_numpy(np.float32)
    else:
        standard_errors = np.full(len(mechanisms), np.nan, dtype=np.float32)
    return effects, standard_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run genome-wide COMPASS with UKBB LD.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--top-assoc-dir", default=None)
    parser.add_argument("--annotation-source", default="abc", choices=["abc", "open_chromatin", "top_assoc"])
    parser.add_argument("--abc-path", default=None)
    parser.add_argument("--abc-cell-types", default=None)
    parser.add_argument("--abc-context-panel", default="ad_proximal", choices=sorted(ABC_CONTEXT_PANELS))
    parser.add_argument("--abc-score-column", default="ABC.Score")
    parser.add_argument("--abc-min-score", type=float, default=0.015)
    parser.add_argument("--open-chromatin-tss", default=None)
    parser.add_argument("--open-chromatin-peaks-root", default=None)
    parser.add_argument("--open-chromatin-tss-window", type=int, default=100_000)
    parser.add_argument("--peak-assay", default="ATAC", choices=sorted(PEAK_ASSAY_FILES))
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--cv-r2-threshold", type=float, default=0.01)
    parser.add_argument("--gwas", default=None)
    parser.add_argument("--ld-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--annotation-value", default="z2", choices=["z2", "abs_z", "beta2", "neglog10p"])
    parser.add_argument("--no-intercept", action="store_true")
    parser.add_argument("--n-samples", type=float, default=None)
    parser.add_argument(
        "--lambdas",
        type=_parse_lambdas,
        default=_parse_lambdas("1e3,3e2,1e2,3e1,1e1,3,1,3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4"),
    )
    parser.add_argument("--max-lambda-extensions", type=int, default=4)
    parser.add_argument("--lambda-extension-factor", type=float, default=3.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-8)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--objective-relative-tol", type=float, default=1e-5)
    parser.add_argument("--objective-window", type=int, default=10)
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument(
        "--cv-checkpoint",
        default=None,
        help="Hierarchical CV checkpoint path (defaults to RUN_NAME.hierarchical_cv_checkpoint.npz)",
    )
    parser.add_argument("--method", default="nuclear", choices=["nuclear", "hierarchical", "rank1"])
    parser.add_argument("--context-annotation", default="binary", choices=["binary", "sum"])
    parser.add_argument(
        "--context-annotation-source",
        default="gene_aggregate",
        choices=["gene_aggregate", "peak"],
    )
    parser.add_argument("--context-effects-tsv", default=None)
    parser.add_argument(
        "--context-effects-mode",
        default="scaled_frozen",
        choices=["fixed", "scaled", "scaled_frozen"],
    )
    parser.add_argument(
        "--regression-weighting",
        default="auto",
        choices=["auto", "uniform", "observed_chisq"],
        help="auto preserves legacy weights except for the response-independent hierarchical fit",
    )
    parser.add_argument("--init-b-tsv", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--svd-method", default="auto", choices=["auto", "exact", "randomized"])
    parser.add_argument("--svd-rank", type=int, default=None)
    parser.add_argument("--svd-oversamples", type=int, default=5)
    parser.add_argument("--svd-n-iter", type=int, default=2)
    parser.add_argument("--ld-chunk-nnz", type=int, default=150_000_000)
    parser.add_argument("--ld-jobs", type=int, default=8)
    parser.add_argument("--ld-r2-cutoff", type=float, default=0.01)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if args.annotation_source == "abc":
        if args.abc_cell_types is None:
            args.abc_cell_types = ABC_CONTEXT_PANELS[args.abc_context_panel]
        elif args.abc_context_panel != "ad_proximal":
            parser.error("--abc-cell-types cannot be combined with a non-default --abc-context-panel")

    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    if not 0 < args.cv_r2_threshold <= 1:
        parser.error("--cv-r2-threshold must be in (0, 1]")
    if args.cv_r2_threshold < args.ld_r2_cutoff:
        parser.error("--cv-r2-threshold cannot be below --ld-r2-cutoff")
    if args.open_chromatin_tss_window < 0:
        parser.error("--open-chromatin-tss-window must be non-negative")
    if args.objective_relative_tol < 0:
        parser.error("--objective-relative-tol must be non-negative")
    if args.objective_window < 1:
        parser.error("--objective-window must be positive")
    if args.method == "rank1" and args.init_b_tsv is None:
        parser.error("--method rank1 requires --init-b-tsv")
    if args.context_effects_tsv is not None and args.method != "hierarchical":
        parser.error("--context-effects-tsv requires --method hierarchical")

    data_root = Path(args.data_root).expanduser()
    top_assoc_dir = Path(args.top_assoc_dir).expanduser() if args.top_assoc_dir else data_root / "raw" / "zenodo_top_assoc"
    abc_path = Path(args.abc_path).expanduser() if args.abc_path else data_root / "raw" / "abc" / DEFAULT_ABC_NAME
    open_chromatin_tss_path = (
        Path(args.open_chromatin_tss).expanduser()
        if args.open_chromatin_tss
        else data_root / "raw" / "abc" / "glass_brain_v2.expressed_tss.hg19.tsv.gz"
    )
    open_chromatin_peaks_root = (
        Path(args.open_chromatin_peaks_root).expanduser()
        if args.open_chromatin_peaks_root
        else data_root / "raw" / "brain-cell-type-peak-files"
    )
    peak_files = {
        cell_type: open_chromatin_peaks_root / args.peak_assay / filename
        for cell_type, filename in PEAK_ASSAY_FILES[args.peak_assay].items()
    }
    if args.context_annotation_source == "peak" and args.annotation_source != "open_chromatin":
        parser.error("--context-annotation-source peak requires --annotation-source open_chromatin")
    gwas_path = Path(args.gwas).expanduser() if args.gwas else data_root / "raw" / "ad_gwas_2026" / "GCST90704647.hg19.tsv.gz"
    ld_dir = Path(args.ld_dir).expanduser() if args.ld_dir else data_root / "raw" / "ukbb_ld"
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_root / "results"
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else data_root / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"compass-{stamp}"
    prefix = out_dir / run_name
    cv_checkpoint_path = (
        Path(args.cv_checkpoint).expanduser()
        if args.cv_checkpoint
        else Path(f"{prefix}.hierarchical_cv_checkpoint.npz")
    )

    if args.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    key = _cache_key(args, gwas_path, abc_path, open_chromatin_tss_path, open_chromatin_peaks_root)
    paths = _cache_paths(cache_dir, key)
    ld_reference_dir = None
    if not args.rebuild_cache and not _dataset_cache_exists(paths):
        legacy_paths = _legacy_abc_cache_paths(cache_dir, args, gwas_path)
        if legacy_paths is not None and _dataset_cache_exists(legacy_paths):
            print(f"[setup] reuse compatible pre-component-CV cache {legacy_paths['A'].stem}", flush=True)
            paths = legacy_paths
    if not args.rebuild_cache and _dataset_cache_exists(paths):
        dataset, genes, mechanisms, ld_diagnostics, n_samples_source = _load_dataset_cache(paths)
    else:
        gwas = _load_gwas_cached(gwas_path, cache_dir, args.rebuild_cache)
        with _timed("load annotations"):
            if args.annotation_source == "abc":
                ann = load_abc_annotations(
                    abc_path,
                    gwas,
                    score_column=args.abc_score_column,
                    min_score=args.abc_min_score,
                    cell_types=args.abc_cell_types,
                    add_intercept=not args.no_intercept,
                )
            elif args.annotation_source == "open_chromatin":
                missing_peak_files = [str(path) for path in peak_files.values() if not path.is_file()]
                if missing_peak_files:
                    raise FileNotFoundError(f"Missing ATAC peak files: {missing_peak_files}")
                ann = load_open_chromatin_tss_annotations(
                    peak_files,
                    open_chromatin_tss_path,
                    gwas,
                    tss_window=args.open_chromatin_tss_window,
                    add_intercept=not args.no_intercept,
                )
            else:
                ann = load_top_assoc_annotations(
                    top_assoc_dir,
                    annotation_value=args.annotation_value,
                    add_intercept=not args.no_intercept,
                )
        if args.annotation_source in {"abc", "open_chromatin"}:
            with _timed(f"filter annotations to UKBB LD panel with {args.ld_jobs} jobs"):
                ann_variants = filter_variants_to_ukbb_ld(
                    ann.variants,
                    str(ld_dir),
                    n_jobs=args.ld_jobs,
                    progress_every=25,
                )
        else:
            ann_variants = ann.variants
        with _timed("align GWAS"):
            training = make_training_table(ann_variants, gwas)
            training = training.drop_duplicates("variant_idx").sort_values("variant_idx").reset_index(drop=True)
            if training.empty:
                raise ValueError("No annotation variants aligned to GWAS summary statistics")

        filtered_variant_idx = training["variant_idx"].to_numpy(np.int64)
        variants = ann_variants.iloc[filtered_variant_idx].copy().reset_index(drop=True)
        source_variant_idx = (
            variants["source_variant_idx"].to_numpy(np.int64)
            if "source_variant_idx" in variants.columns
            else filtered_variant_idx
        )
        variants["variant_idx"] = np.arange(variants.shape[0], dtype=np.int64)
        old_to_new = pd.DataFrame(
            {"variant_idx": source_variant_idx, "new_variant_idx": variants["variant_idx"].to_numpy(np.int64)}
        )
        triples = ann.triples.merge(old_to_new, on="variant_idx", how="inner")
        triples = triples.drop(columns=["variant_idx"]).rename(columns={"new_variant_idx": "variant_idx"})

        with _timed("build annotation matrix"):
            A = annotation_triples_to_csr(
                triples,
                n_variants=variants.shape[0],
                n_genes=ann.genes.shape[0],
                n_mechanisms=len(ann.mechanisms),
            )
        reusable_ld = None
        if args.annotation_source in {"abc", "open_chromatin"} and not args.rebuild_cache:
            with _timed("find compatible all-row LD cache"):
                reusable_ld = _find_compatible_allrow_ld(
                    cache_dir,
                    variants["chrom"].to_numpy(np.int64),
                    args.ld_r2_cutoff,
                )
        if reusable_ld is not None:
            ld_blocks, ld_diagnostics, ld_reference_dir = reusable_ld
            print(f"[setup] reuse verified all-row LD archive {ld_reference_dir}", flush=True)
        else:
            with _timed(f"build chromosome UKBB LD R2 with {args.ld_jobs} jobs"):
                ld_block_dicts, ld_diagnostics = build_ukbb_ld_r2_by_chromosome(
                    variants,
                    str(ld_dir),
                    n_jobs=args.ld_jobs,
                    progress_every=25,
                    r2_cutoff=args.ld_r2_cutoff,
                    dtype=np.float32,
                )
                ld_blocks = [
                    LdChromosomeBlock(chrom=int(block["chrom"]), rows=block["rows"], R2=block["R2"])
                    for block in ld_block_dicts
                ]

        if args.n_samples is not None:
            n_samples: float | np.ndarray = float(args.n_samples)
            n_samples_source = "argument"
        elif "n" in training.columns and training["n"].notna().all():
            n_samples = training["n"].to_numpy(np.float32)
            n_samples_source = "gwas"
        else:
            raise ValueError("Known sample sizes are required: pass --n-samples or use a GWAS file with Nsum/Neff/N")

        dataset = CompassDataset(
            A=A,
            chisq=training["chisq"].to_numpy(np.float32),
            chrom=variants["chrom"].to_numpy(np.int64),
            n_samples=n_samples,
            position=variants["pos"].to_numpy(np.int64),
            ld_blocks=ld_blocks,
        )
        genes = ann.genes
        mechanisms = ann.mechanisms

    cv_metadata = None
    if not args.no_cv:
        if dataset.ld_blocks is None:
            raise ValueError("LD-component CV requires chromosome-level LD blocks")
        cv_cache = cv_cache_path(paths["arrays"], args.cv_folds, args.cv_r2_threshold)
        cached_cv = None if args.rebuild_cache else load_cv_cache(cv_cache, dataset.n_variants)
        if cached_cv is None:
            with _timed("build global LD-component CV folds"):
                cv_groups, cv_score_groups, cv_metadata = make_ld_component_cv_groups(
                    dataset.ld_blocks,
                    dataset.A,
                    len(mechanisms),
                    n_folds=args.cv_folds,
                    r2_threshold=args.cv_r2_threshold,
                )
            with _timed("write LD-component CV cache"):
                write_cv_cache(cv_cache, cv_groups, cv_score_groups, cv_metadata)
        else:
            with _timed("load LD-component CV cache"):
                cv_groups, cv_score_groups, cv_metadata = cached_cv
        dataset.cv_groups = cv_groups
        dataset.cv_score_groups = cv_score_groups

    if args.method == "hierarchical":
        weighting = "uniform" if args.regression_weighting == "auto" else args.regression_weighting
        if weighting == "uniform":
            dataset.sample_weight = np.ones(dataset.n_variants, dtype=np.float32)
        with _timed(f"build {args.context_annotation} context annotations"):
            if args.context_annotation_source == "peak":
                if dataset.position is None:
                    raise ValueError("Flat peak contexts require a cache with variant positions")
                dataset.context_annotations = load_peak_context_annotations(
                    peak_files,
                    dataset.chrom,
                    dataset.position,
                    mechanisms,
                )
            else:
                dataset.context_annotations = aggregate_context_annotations(
                    dataset.A,
                    genes.shape[0],
                    len(mechanisms),
                    mode=args.context_annotation,
                )

    if not _dataset_cache_exists(paths):
        _write_dataset_cache(
            paths,
            dataset,
            genes,
            mechanisms,
            ld_diagnostics,
            n_samples_source,
            ld_reference_dir=ld_reference_dir,
        )

    print(
        f"[setup] dataset variants={dataset.n_variants} params={dataset.n_params} "
        f"A_nnz={dataset.A.nnz} R2_nnz="
        f"{dataset.R2.nnz if dataset.R2 is not None else sum(block.R2.nnz for block in dataset.ld_blocks)}",
        flush=True,
    )
    if cv_metadata is not None:
        print(
            f"[setup] LD-component CV components={cv_metadata['cv_components']} "
            f"largest_component={cv_metadata['cv_largest_component']} "
            f"fold_rows={cv_metadata['cv_fold_rows']} "
            f"score_rows={cv_metadata['cv_score_rows']} "
            f"score_rows_by_fold={cv_metadata['cv_score_rows_by_fold']}",
            flush=True,
        )
    if args.setup_only:
        print("[setup] setup-only complete", flush=True)
        return

    with _timed("fit"):
        if args.method == "nuclear":
            fit = fit_nuclear_norm_path(
                dataset,
                n_genes=genes.shape[0],
                n_mechanisms=len(mechanisms),
                lambdas=args.lambdas,
                max_lambda_extensions=args.max_lambda_extensions,
                lambda_extension_factor=args.lambda_extension_factor,
                cv=not args.no_cv,
                lr=args.lr,
                max_iter=args.max_iter,
                progress_every=args.progress_every,
                tol=args.tol,
                device=device,
                model_dtype=args.model_dtype,
                svd_method=args.svd_method,
                svd_rank=args.svd_rank,
                svd_oversamples=args.svd_oversamples,
                svd_n_iter=args.svd_n_iter,
                ld_chunk_nnz=args.ld_chunk_nnz,
            )
        elif args.method == "hierarchical":
            fixed_context_effects = None
            fixed_context_effect_se = None
            if args.context_effects_tsv is not None:
                fixed_context_effects, fixed_context_effect_se = _load_context_effects(
                    args.context_effects_tsv, mechanisms
                )
            fit = fit_hierarchical_nuclear_path(
                dataset,
                n_genes=genes.shape[0],
                n_mechanisms=len(mechanisms),
                lambdas=args.lambdas,
                cv=not args.no_cv,
                fixed_context_effects=fixed_context_effects,
                fixed_context_effect_se=fixed_context_effect_se,
                scale_fixed_context_effects=(
                    fixed_context_effects is not None
                    and args.context_effects_mode in {"scaled", "scaled_frozen"}
                ),
                freeze_scaled_context_effects=(args.context_effects_mode == "scaled_frozen"),
                cv_checkpoint_path=cv_checkpoint_path if not args.no_cv else None,
                max_lambda_extensions=args.max_lambda_extensions,
                lambda_extension_factor=args.lambda_extension_factor,
                lr=args.lr,
                max_iter=args.max_iter,
                progress_every=args.progress_every,
                tol=args.tol,
                objective_relative_tol=args.objective_relative_tol,
                objective_window=args.objective_window,
                device=device,
                model_dtype=args.model_dtype,
                svd_method=args.svd_method,
                svd_rank=args.svd_rank,
                svd_oversamples=args.svd_oversamples,
                svd_n_iter=args.svd_n_iter,
                ld_chunk_nnz=args.ld_chunk_nnz,
            )
        else:
            fit = fit_rank1_path(
                dataset,
                n_genes=genes.shape[0],
                n_mechanisms=len(mechanisms),
                lambdas=args.lambdas,
                cv=not args.no_cv,
                initial_B=_load_initial_B(args.init_b_tsv, genes, mechanisms),
                lr=args.lr,
                max_iter=args.max_iter,
                tol=args.tol,
                device=device,
                model_dtype=args.model_dtype,
                ld_chunk_nnz=args.ld_chunk_nnz,
                progress_every=args.progress_every,
                progress_label="full",
            )

    result_arrays = {
        "B": fit.B,
        "tau": np.asarray(fit.tau, dtype=np.float32),
        "losses": np.asarray(fit.losses, dtype=np.float32),
        "lambdas": np.asarray(fit.lambdas, dtype=np.float32),
        "best_lambda": np.asarray(fit.best_lambda, dtype=np.float32),
    }
    if fit.context_effects is not None:
        result_arrays["context_effects"] = fit.context_effects
        result_arrays["context_effect_se"] = fit.context_effect_se
    np.savez_compressed(f"{prefix}.npz", **result_arrays)
    pd.DataFrame(fit.B, index=genes["gene"], columns=mechanisms).to_csv(f"{prefix}.B.tsv", sep="\t")
    if fit.context_effects is not None:
        context_summary = context_heritability_components(
            dataset.A,
            fit.B,
            dataset.context_annotations,
            fit.context_effects,
        )
        pd.DataFrame(
            {
                "context": mechanisms,
                "effect": fit.context_effects,
                "standard_error": fit.context_effect_se,
                "z": np.divide(
                    fit.context_effects,
                    fit.context_effect_se,
                    out=np.zeros_like(fit.context_effects),
                    where=fit.context_effect_se > 0,
                ),
                "annotation_count": context_summary["annotation_count"],
                "implied_h2": context_summary["global_h2"],
            }
        ).to_csv(f"{prefix}.context_effects.tsv", sep="\t", index=False)
        pd.DataFrame(
            {
                "context": mechanisms,
                **context_summary,
            }
        ).to_csv(f"{prefix}.context_contributions.tsv", sep="\t", index=False)
    ld_diagnostics.to_csv(f"{prefix}.ld_diagnostics.tsv", sep="\t", index=False)
    metadata = {
        "method": fit.method,
        "context_annotation": args.context_annotation if args.method == "hierarchical" else None,
        "context_annotation_source": (
            args.context_annotation_source if args.method == "hierarchical" else None
        ),
        "context_effects": fit.context_effects,
        "context_effect_se": fit.context_effect_se,
        "context_effects_tsv": (
            str(Path(args.context_effects_tsv).expanduser()) if args.context_effects_tsv else None
        ),
        "context_effects_mode": args.context_effects_mode if args.context_effects_tsv else None,
        "context_annotation_counts": (
            context_summary["annotation_count"]
            if fit.context_effects is not None
            else None
        ),
        "context_heritability_components": context_summary if fit.context_effects is not None else None,
        "regression_weighting": (
            "uniform"
            if args.method == "hierarchical" and args.regression_weighting == "auto"
            else args.regression_weighting
        ),
        "init_b_tsv": str(Path(args.init_b_tsv).expanduser()) if args.init_b_tsv else None,
        "best_lambda": fit.best_lambda,
        "cv_scores": fit.cv_scores,
        "metadata": fit.metadata,
        "n_variants": dataset.n_variants,
        "n_genes": genes.shape[0],
        "n_mechanisms": len(mechanisms),
        "n_samples_source": n_samples_source,
        "device": device,
        "model_dtype": args.model_dtype,
        "ld_dtype": "float16",
        "ld_chunk_nnz": args.ld_chunk_nnz,
        "max_lambda_extensions": args.max_lambda_extensions,
        "lambda_extension_factor": args.lambda_extension_factor,
        "svd_method": args.svd_method,
        "svd_rank": args.svd_rank,
        "ld_jobs": args.ld_jobs,
        "ld_r2_cutoff": args.ld_r2_cutoff,
        "ld_representation": "chromosome_fp16",
        "cv_component_metadata": cv_metadata,
        "cache_key": key,
        "cv": not args.no_cv,
        "cv_checkpoint": (
            str(cv_checkpoint_path)
            if args.method == "hierarchical" and not args.no_cv
            else None
        ),
        "annotation_source": args.annotation_source,
        "abc_path": str(abc_path) if args.annotation_source == "abc" else None,
        "abc_cell_types": args.abc_cell_types if args.annotation_source == "abc" else None,
        "abc_context_panel": args.abc_context_panel if args.annotation_source == "abc" else None,
        "abc_score_column": args.abc_score_column if args.annotation_source == "abc" else None,
        "abc_min_score": args.abc_min_score if args.annotation_source == "abc" else None,
        "open_chromatin_tss": str(open_chromatin_tss_path) if args.annotation_source == "open_chromatin" else None,
        "open_chromatin_peaks_root": (
            str(open_chromatin_peaks_root) if args.annotation_source == "open_chromatin" else None
        ),
        "open_chromatin_tss_window": (
            args.open_chromatin_tss_window if args.annotation_source == "open_chromatin" else None
        ),
        "peak_assay": args.peak_assay if args.annotation_source == "open_chromatin" else None,
        "cv_folds": args.cv_folds if not args.no_cv else None,
        "cv_r2_threshold": args.cv_r2_threshold if not args.no_cv else None,
    }
    with open(f"{prefix}.metadata.json", "w", encoding="utf-8") as handle:
        json.dump(_json_safe(metadata), handle, indent=2, sort_keys=True)
    print(f"wrote {prefix}.npz")


if __name__ == "__main__":
    main()

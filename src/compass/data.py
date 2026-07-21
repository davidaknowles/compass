from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp


CELLTYPE_NAMES = {
    "Ast": "astrocyte",
    "End": "endothelial",
    "Ext": "excitatory_neuron",
    "IN": "inhibitory_neuron",
    "MG": "microglia",
    "OD": "oligodendrocyte",
    "OPC": "opc",
}


def cv_cache_path(arrays_path: str | Path, n_folds: int, r2_threshold: float) -> Path:
    """Return the deterministic sidecar path for LD-component fold assignments."""

    return Path(f"{Path(arrays_path)}.cv{n_folds}.r2ge{r2_threshold:g}.npz")


def load_cv_cache(path: str | Path, n_variants: int) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Load a validated LD-component fold cache, or return None if unusable."""

    cache_path = Path(path)
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            groups = archive["cv_groups"].astype(np.int64, copy=True)
            score_groups = archive["cv_score_groups"].astype(np.int64, copy=True)
            metadata = json.loads(str(archive["metadata_json"].item()))
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None
    if groups.shape != (n_variants,) or score_groups.shape != (n_variants,):
        return None
    return groups, score_groups, metadata


def write_cv_cache(
    path: str | Path,
    cv_groups: np.ndarray,
    cv_score_groups: np.ndarray,
    metadata: dict,
) -> None:
    """Atomically persist LD-component fold assignments and diagnostics."""

    cache_path = Path(path)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cv_groups=np.asarray(cv_groups, dtype=np.int64),
            cv_score_groups=np.asarray(cv_score_groups, dtype=np.int64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    os.replace(temporary, cache_path)


@dataclass(frozen=True)
class AnnotationData:
    variants: pd.DataFrame
    genes: pd.DataFrame
    mechanisms: list[str]
    triples: pd.DataFrame


def _variant_id(chr_values: pd.Series, pos_values: pd.Series) -> pd.Series:
    return "chr" + chr_values.astype(str) + ":" + pos_values.astype(str)


def _normalize_chrom(chrom: pd.Series) -> pd.Series:
    return pd.to_numeric(chrom.astype(str).str.replace("chr", "", regex=False), errors="coerce")


def load_top_assoc_annotations(
    top_assoc_dir: str | Path,
    annotation_value: str = "z2",
    add_intercept: bool = True,
    max_genes: int | None = None,
    chromosomes: Iterable[int] | None = None,
) -> AnnotationData:
    """Load Zenodo SingleBrain top associations as sparse variant-gene annotations.

    The top-association files contain one lead cis-eQTL per gene and cell type.
    This loader converts them into non-negative annotation triples
    ``variant_idx, gene_idx, mechanism_idx, value``.
    """

    top_assoc_dir = Path(top_assoc_dir).expanduser()
    files = sorted(top_assoc_dir.glob("*_top_assoc.tsv.gz"))
    if not files:
        raise FileNotFoundError(f"No *_top_assoc.tsv.gz files found in {top_assoc_dir}")

    frames: list[pd.DataFrame] = []
    keep_chroms = None if chromosomes is None else {int(c) for c in chromosomes}
    for file in files:
        code = file.name.split("_top_assoc.tsv.gz")[0]
        mechanism = CELLTYPE_NAMES.get(code, code)
        usecols = ["feature", "CHR", "POS", "MarkerID", "BETA", "SE", "Tstat", "p.value", "N"]
        df = pd.read_csv(file, sep="\t", usecols=usecols)
        df = df.rename(columns={"feature": "gene", "CHR": "chrom", "POS": "pos"})
        df["mechanism"] = mechanism
        df["chrom"] = _normalize_chrom(df["chrom"])
        df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
        df = df[df["chrom"].between(1, 22) & df["pos"].notna()].copy()
        df["chrom"] = df["chrom"].astype(int)
        df["pos"] = df["pos"].astype(int)
        if keep_chroms is not None:
            df = df[df["chrom"].isin(keep_chroms)]
        frames.append(df)

    assoc = pd.concat(frames, ignore_index=True)
    assoc = assoc.dropna(subset=["gene", "chrom", "pos", "BETA", "SE", "Tstat"])
    assoc["variant_id"] = _variant_id(assoc["chrom"], assoc["pos"])

    if max_genes is not None:
        gene_order = (
            assoc.assign(abs_t=assoc["Tstat"].abs())
            .groupby("gene", as_index=False)["abs_t"]
            .max()
            .sort_values("abs_t", ascending=False)
            .head(max_genes)["gene"]
        )
        assoc = assoc[assoc["gene"].isin(set(gene_order))]

    if annotation_value == "z2":
        assoc["value"] = assoc["Tstat"].astype(float).pow(2)
    elif annotation_value == "abs_z":
        assoc["value"] = assoc["Tstat"].astype(float).abs()
    elif annotation_value == "beta2":
        assoc["value"] = assoc["BETA"].astype(float).pow(2)
    elif annotation_value == "neglog10p":
        p = assoc["p.value"].astype(float).clip(lower=np.finfo(float).tiny)
        assoc["value"] = -np.log10(p)
    else:
        raise ValueError(f"Unknown annotation_value: {annotation_value}")

    assoc["value"] = assoc["value"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assoc = assoc[assoc["value"] > 0].copy()

    genes = pd.DataFrame({"gene": sorted(assoc["gene"].unique())})
    genes["gene_idx"] = np.arange(len(genes), dtype=np.int64)
    variants = (
        assoc[["variant_id", "chrom", "pos", "MarkerID"]]
        .drop_duplicates("variant_id")
        .sort_values(["chrom", "pos", "variant_id"])
        .reset_index(drop=True)
    )
    variants["variant_idx"] = np.arange(len(variants), dtype=np.int64)

    mechanisms = sorted(assoc["mechanism"].unique())
    if add_intercept:
        mechanisms = ["intercept"] + mechanisms
    mechanism_to_idx = {m: i for i, m in enumerate(mechanisms)}

    triples = assoc.merge(genes, on="gene", how="inner").merge(
        variants[["variant_id", "variant_idx"]], on="variant_id", how="inner"
    )
    triples["mechanism_idx"] = triples["mechanism"].map(mechanism_to_idx).astype(np.int64)
    triples = triples[["variant_idx", "gene_idx", "mechanism_idx", "value"]]

    if add_intercept:
        intercept = (
            assoc[["variant_id", "gene"]]
            .drop_duplicates()
            .merge(genes, on="gene", how="inner")
            .merge(variants[["variant_id", "variant_idx"]], on="variant_id", how="inner")
        )
        intercept = intercept[["variant_idx", "gene_idx"]].copy()
        intercept["mechanism_idx"] = mechanism_to_idx["intercept"]
        intercept["value"] = 1.0
        triples = pd.concat([triples, intercept], ignore_index=True)

    triples = triples.groupby(["variant_idx", "gene_idx", "mechanism_idx"], as_index=False)[
        "value"
    ].max()
    return AnnotationData(
        variants=variants.reset_index(drop=True),
        genes=genes.reset_index(drop=True),
        mechanisms=mechanisms,
        triples=triples.reset_index(drop=True),
    )


def _parse_cell_types(value: str | Iterable[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in {"", "all"}:
            return None
        return {x.strip() for x in value.split(",") if x.strip()}
    return {str(x).strip() for x in value if str(x).strip()}


def load_abc_annotations(
    abc_path: str | Path,
    gwas: pd.DataFrame,
    score_column: str = "ABC.Score",
    min_score: float = 0.015,
    cell_types: str | Iterable[str] | None = None,
    exclude_self_promoter: bool = False,
    add_intercept: bool = True,
    chunksize: int = 200_000,
) -> AnnotationData:
    """Load public ABC enhancer-gene links as sparse variant-gene annotations.

    By default, variants are all usable autosomal GWAS variants. Variants that
    fall inside an ABC candidate regulatory element (CRE) receive sparse
    CRE-gene-context annotations; other variants remain as all-zero rows so LD
    tagging and residual LD-score regression still use the genome-wide row set.
    Cross-validation groups are assigned later from global LD components after
    the complete all-variant chromosome LD representation has been assembled.
    """

    abc_path = Path(abc_path).expanduser()
    if "chrom" not in gwas.columns or "pos" not in gwas.columns:
        raise ValueError("ABC annotations require GWAS chromosome and position columns")

    keep_cell_types = _parse_cell_types(cell_types)
    gwas_variants = gwas.dropna(subset=["chrom", "pos"]).copy()
    gwas_variants["chrom"] = _normalize_chrom(gwas_variants["chrom"])
    gwas_variants["pos"] = pd.to_numeric(gwas_variants["pos"], errors="coerce")
    gwas_variants = gwas_variants[gwas_variants["chrom"].between(1, 22) & gwas_variants["pos"].notna()].copy()
    gwas_variants["chrom"] = gwas_variants["chrom"].astype(int)
    gwas_variants["pos"] = gwas_variants["pos"].astype(int)
    if "variant_id" not in gwas_variants.columns:
        gwas_variants["variant_id"] = _variant_id(gwas_variants["chrom"], gwas_variants["pos"])
    if "snp" in gwas_variants.columns:
        gwas_variants["MarkerID"] = gwas_variants["snp"].astype(str)
    else:
        gwas_variants["MarkerID"] = gwas_variants["variant_id"].astype(str)
    gwas_variants = gwas_variants.drop_duplicates("variant_id").sort_values(["chrom", "pos", "variant_id"])

    chrom_lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, group in gwas_variants.groupby("chrom", sort=False):
        ids = group["variant_id"].astype(str).to_numpy()
        positions = group["pos"].to_numpy(np.int64)
        order = np.argsort(positions, kind="mergesort")
        chrom_lookup[int(chrom)] = (positions[order], ids[order])

    gene_to_idx: dict[str, int] = {}
    mechanism_to_idx: dict[str, int] = {}
    mechanisms: list[str] = []
    if add_intercept:
        mechanism_to_idx["intercept"] = 0
        mechanisms.append("intercept")
    raw_records: list[tuple[str, int, int, float]] = []
    usecols = ["chr", "start", "end", "TargetGene", score_column, "CellType"]
    if exclude_self_promoter:
        usecols.append("isSelfPromoter")

    compression = "gzip" if abc_path.suffix == ".gz" else None
    reader = pd.read_csv(abc_path, sep="\t", usecols=usecols, compression=compression, chunksize=chunksize)
    for chunk in reader:
        chunk = chunk.rename(columns={"chr": "chrom", score_column: "score"})
        chunk["chrom"] = _normalize_chrom(chunk["chrom"])
        chunk["start"] = pd.to_numeric(chunk["start"], errors="coerce")
        chunk["end"] = pd.to_numeric(chunk["end"], errors="coerce")
        chunk["score"] = pd.to_numeric(chunk["score"], errors="coerce")
        chunk = chunk.dropna(subset=["chrom", "start", "end", "TargetGene", "score", "CellType"])
        chunk = chunk[chunk["chrom"].between(1, 22) & (chunk["score"] >= min_score)].copy()
        if exclude_self_promoter:
            chunk = chunk[~chunk["isSelfPromoter"].astype(str).str.lower().eq("true")].copy()
        if keep_cell_types is not None:
            chunk = chunk[chunk["CellType"].isin(keep_cell_types)].copy()
        if chunk.empty:
            continue
        chunk["chrom"] = chunk["chrom"].astype(int)
        chunk["start"] = chunk["start"].astype(int)
        chunk["end"] = chunk["end"].astype(int)
        for row in chunk.itertuples(index=False):
            chrom = int(row.chrom)
            if chrom not in chrom_lookup:
                continue
            gene = str(row.TargetGene)
            mechanism = str(row.CellType)
            gene_idx = gene_to_idx.setdefault(gene, len(gene_to_idx))
            if mechanism not in mechanism_to_idx:
                mechanism_to_idx[mechanism] = len(mechanisms)
                mechanisms.append(mechanism)
            mechanism_idx = mechanism_to_idx[mechanism]
            positions, variant_ids = chrom_lookup[chrom]
            lo = int(np.searchsorted(positions, int(row.start), side="left"))
            hi = int(np.searchsorted(positions, int(row.end), side="right"))
            if hi <= lo:
                continue
            raw_records.extend(
                (str(variant_id), gene_idx, mechanism_idx, float(row.score))
                for variant_id in variant_ids[lo:hi]
            )
            if add_intercept:
                raw_records.extend(
                    (str(variant_id), gene_idx, mechanism_to_idx["intercept"], 1.0)
                    for variant_id in variant_ids[lo:hi]
                )

    if not raw_records:
        raise ValueError(f"No ABC CREs in {abc_path} overlapped GWAS variants")

    variants = gwas_variants[["variant_id", "chrom", "pos", "MarkerID"]].copy()
    variants = variants.sort_values(["chrom", "pos", "variant_id"]).reset_index(drop=True)
    variants["variant_idx"] = np.arange(variants.shape[0], dtype=np.int64)
    variant_to_idx = dict(zip(variants["variant_id"].astype(str), variants["variant_idx"]))

    triples = pd.DataFrame(raw_records, columns=["variant_id", "gene_idx", "mechanism_idx", "value"])
    triples["variant_idx"] = triples["variant_id"].map(variant_to_idx).astype(np.int64)
    triples = (
        triples[["variant_idx", "gene_idx", "mechanism_idx", "value"]]
        .groupby(["variant_idx", "gene_idx", "mechanism_idx"], as_index=False)["value"]
        .max()
    )

    genes = pd.DataFrame(
        sorted(gene_to_idx.items(), key=lambda item: item[1]),
        columns=["gene", "gene_idx"],
    )
    return AnnotationData(
        variants=variants.reset_index(drop=True),
        genes=genes.reset_index(drop=True),
        mechanisms=mechanisms,
        triples=triples.reset_index(drop=True),
    )


def _read_bed_intervals(path: str | Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    peaks = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    peaks["chrom"] = _normalize_chrom(peaks["chrom"])
    peaks["start"] = pd.to_numeric(peaks["start"], errors="coerce")
    peaks["end"] = pd.to_numeric(peaks["end"], errors="coerce")
    peaks = peaks.dropna().astype({"chrom": np.int64, "start": np.int64, "end": np.int64})
    peaks = peaks[peaks["chrom"].between(1, 22) & peaks["end"].gt(peaks["start"])]
    intervals: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, group in peaks.groupby("chrom", sort=False):
        ordered = group.sort_values(["start", "end"], kind="mergesort")
        starts = ordered["start"].to_numpy(np.int64)
        intervals[int(chrom)] = (starts, np.maximum.accumulate(ordered["end"].to_numpy(np.int64)))
    return intervals


def _positions_in_bed_intervals(
    positions: np.ndarray,
    intervals: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if intervals is None or positions.size == 0:
        return np.zeros(positions.size, dtype=bool)
    starts, max_ends = intervals
    previous = np.searchsorted(starts, positions, side="right") - 1
    valid = previous >= 0
    result = np.zeros(positions.size, dtype=bool)
    result[valid] = positions[valid] <= max_ends[previous[valid]]
    return result


def load_peak_context_annotations(
    peak_files: dict[str, str | Path],
    chrom: np.ndarray,
    position: np.ndarray,
    mechanisms: list[str],
) -> sp.csr_matrix:
    """Build flat binary peak annotations aligned to an existing variant universe."""

    chrom = np.asarray(chrom, dtype=np.int64)
    position = np.asarray(position, dtype=np.int64)
    if chrom.shape != position.shape:
        raise ValueError("chrom and position must have the same shape")
    unknown = set(peak_files).difference(mechanisms)
    if unknown:
        raise ValueError(f"Peak contexts are absent from mechanisms: {sorted(unknown)}")
    intervals = {name: _read_bed_intervals(path) for name, path in peak_files.items()}
    values = np.zeros((chrom.size, len(mechanisms)), dtype=np.float32)
    for mechanism, by_chromosome in intervals.items():
        column = mechanisms.index(mechanism)
        for chromosome in range(1, 23):
            rows = np.flatnonzero(chrom == chromosome)
            values[rows, column] = _positions_in_bed_intervals(
                position[rows], by_chromosome.get(chromosome)
            )
    if "intercept" in mechanisms:
        context_columns = [mechanisms.index(name) for name in peak_files]
        values[:, mechanisms.index("intercept")] = values[:, context_columns].max(axis=1)
    return sp.csr_matrix(values)


def load_open_chromatin_tss_annotations(
    peak_files: dict[str, str | Path],
    tss_path: str | Path,
    gwas: pd.DataFrame,
    tss_window: int = 100_000,
    add_intercept: bool = True,
) -> AnnotationData:
    """Link ATAC-overlapping variants to expressed cell-specific genes by TSS distance.

    Every usable GWAS variant is retained. A variant receives a binary
    gene-context annotation only when it lies in the corresponding ATAC peak
    and the gene's cell-specific expressed TSS is within ``tss_window`` bases.
    """

    if not peak_files:
        raise ValueError("peak_files must not be empty")
    if tss_window < 0:
        raise ValueError("tss_window must be non-negative")
    if "chrom" not in gwas.columns or "pos" not in gwas.columns:
        raise ValueError("Open-chromatin annotations require GWAS chromosome and position columns")

    variants = gwas.dropna(subset=["chrom", "pos"]).copy()
    variants["chrom"] = _normalize_chrom(variants["chrom"])
    variants["pos"] = pd.to_numeric(variants["pos"], errors="coerce")
    variants = variants[variants["chrom"].between(1, 22) & variants["pos"].notna()].copy()
    variants["chrom"] = variants["chrom"].astype(int)
    variants["pos"] = variants["pos"].astype(int)
    if "variant_id" not in variants.columns:
        variants["variant_id"] = _variant_id(variants["chrom"], variants["pos"])
    if "snp" in variants.columns:
        variants["MarkerID"] = variants["snp"].astype(str)
    else:
        variants["MarkerID"] = variants["variant_id"].astype(str)
    variants = (
        variants[["variant_id", "chrom", "pos", "MarkerID"]]
        .drop_duplicates("variant_id")
        .sort_values(["chrom", "pos", "variant_id"])
        .reset_index(drop=True)
    )
    variants["variant_idx"] = np.arange(variants.shape[0], dtype=np.int64)

    mechanisms = list(peak_files)
    tss = pd.read_csv(tss_path, sep="\t", compression="infer", usecols=["chrom", "tss", "gene", "CellType"])
    tss["chrom"] = _normalize_chrom(tss["chrom"])
    tss["tss"] = pd.to_numeric(tss["tss"], errors="coerce")
    tss = tss.dropna(subset=["chrom", "tss", "gene", "CellType"])
    tss = tss[tss["chrom"].between(1, 22) & tss["CellType"].isin(mechanisms)].copy()
    tss["chrom"] = tss["chrom"].astype(int)
    tss["tss"] = tss["tss"].astype(int)
    tss = tss[tss["tss"].gt(0)].drop_duplicates(["chrom", "tss", "gene", "CellType"])
    missing_mechanisms = set(mechanisms).difference(tss["CellType"])
    if missing_mechanisms:
        raise ValueError(f"No expressed TSS records for requested contexts: {sorted(missing_mechanisms)}")

    genes = pd.DataFrame({"gene": sorted(tss["gene"].astype(str).unique())})
    genes["gene_idx"] = np.arange(genes.shape[0], dtype=np.int64)
    gene_to_idx = dict(zip(genes["gene"], genes["gene_idx"]))
    tss["gene_idx"] = tss["gene"].astype(str).map(gene_to_idx).astype(np.int64)
    mechanism_to_idx = {name: index + int(add_intercept) for index, name in enumerate(mechanisms)}

    triples_by_block: list[pd.DataFrame] = []
    for mechanism, peak_path in peak_files.items():
        intervals = _read_bed_intervals(peak_path)
        mechanism_idx = mechanism_to_idx[mechanism]
        cell_tss = tss[tss["CellType"].eq(mechanism)]
        for chrom, var_block in variants.groupby("chrom", sort=False):
            positions = var_block["pos"].to_numpy(np.int64)
            contained = _positions_in_bed_intervals(positions, intervals.get(int(chrom)))
            if not contained.any():
                continue
            peak_positions = positions[contained]
            peak_variant_idx = var_block["variant_idx"].to_numpy(np.int64)[contained]
            tss_block = cell_tss[cell_tss["chrom"].eq(int(chrom))].sort_values("tss", kind="mergesort")
            if tss_block.empty:
                continue
            variant_parts: list[np.ndarray] = []
            gene_parts: list[np.ndarray] = []
            for row in tss_block[["tss", "gene_idx"]].itertuples(index=False):
                lo = np.searchsorted(peak_positions, int(row.tss) - tss_window, side="left")
                hi = np.searchsorted(peak_positions, int(row.tss) + tss_window, side="right")
                if hi > lo:
                    count = hi - lo
                    variant_parts.append(peak_variant_idx[lo:hi])
                    gene_parts.append(np.full(count, int(row.gene_idx), dtype=np.int64))
            if variant_parts:
                variant_idx = np.concatenate(variant_parts)
                triples_by_block.append(
                    pd.DataFrame(
                        {
                            "variant_idx": variant_idx,
                            "gene_idx": np.concatenate(gene_parts),
                            "mechanism_idx": np.full(variant_idx.size, mechanism_idx, dtype=np.int64),
                            "value": np.ones(variant_idx.size, dtype=np.float32),
                        }
                    )
                )
    if not triples_by_block:
        raise ValueError("No ATAC-overlapping variants had an expressed TSS within the requested window")
    triples = pd.concat(triples_by_block, ignore_index=True)
    if add_intercept:
        intercept = triples[["variant_idx", "gene_idx"]].drop_duplicates().copy()
        intercept["mechanism_idx"] = 0
        intercept["value"] = np.float32(1.0)
        triples = pd.concat([triples, intercept], ignore_index=True)
    triples = triples.groupby(["variant_idx", "gene_idx", "mechanism_idx"], as_index=False)["value"].max()
    return AnnotationData(variants=variants, genes=genes, mechanisms=(['intercept'] if add_intercept else []) + mechanisms, triples=triples)


def load_gwas_sumstats(path: str | Path) -> pd.DataFrame:
    """Load GWAS summary statistics with flexible column normalization."""

    path = Path(path).expanduser()
    compression = "gzip" if path.suffix == ".gz" else None
    suffixes = path.suffixes
    if ".tsv" in suffixes or ".txt" in suffixes:
        df = pd.read_csv(path, sep="\t", compression=compression)
    elif ".csv" in suffixes:
        df = pd.read_csv(path, sep=",", compression=compression)
    else:
        df = pd.read_csv(path, sep=None, engine="python", compression=compression)
    lower = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for name in names:
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    chrom_col = pick("chromosome", "chr", "chrom", "CHR")
    pos_col = pick("position", "bp", "pos", "BP")
    snp_col = pick("snp", "rsid", "variant", "markerid", "MarkerID", "id")
    z_col = pick("z", "zscore", "z_stat", "stat")
    beta_col = pick("beta", "effect", "b")
    se_col = pick("se", "stderr", "standard_error")
    p_col = pick("p", "pvalue", "p.value", "pval")
    n_col = pick("n", "nsum", "neff", "samplesize", "n_complete_samples")

    out = pd.DataFrame(index=df.index)
    if chrom_col is not None and pos_col is not None:
        out["chrom"] = pd.to_numeric(df[chrom_col], errors="coerce").astype("Int64")
        out["pos"] = pd.to_numeric(df[pos_col], errors="coerce").astype("Int64")
        out["variant_id"] = _variant_id(out["chrom"], out["pos"])
    if snp_col is not None:
        out["snp"] = df[snp_col].astype(str)

    if z_col is not None:
        z = pd.to_numeric(df[z_col], errors="coerce")
    elif beta_col is not None and se_col is not None:
        z = pd.to_numeric(df[beta_col], errors="coerce") / pd.to_numeric(df[se_col], errors="coerce")
    elif p_col is not None:
        from scipy.stats import norm

        p = pd.to_numeric(df[p_col], errors="coerce").clip(np.finfo(float).tiny, 1.0)
        z = pd.Series(norm.isf(p / 2.0), index=df.index)
    else:
        raise ValueError("Could not infer Z statistics from GWAS file")

    out["z"] = z
    out["chisq"] = z.pow(2)
    if n_col is not None:
        out["n"] = pd.to_numeric(df[n_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["chisq"])
    return out.reset_index(drop=True)


def make_training_table(variants: pd.DataFrame, gwas: pd.DataFrame) -> pd.DataFrame:
    """Align annotation variants to GWAS chi-square values."""

    if "snp" in gwas.columns:
        candidates = variants.rename(columns={"MarkerID": "snp"})
        if "snp" in candidates.columns:
            wanted = set(candidates["snp"].astype(str))
            gwas_snp = gwas[gwas["snp"].astype(str).isin(wanted)]
            cols = ["snp", "chisq"] + (["n"] if "n" in gwas_snp else [])
            merged = candidates.merge(gwas_snp[cols].drop_duplicates("snp"), on="snp", how="inner")
            if not merged.empty:
                return merged

    if "variant_id" in gwas.columns and "variant_id" in variants.columns:
        wanted = set(variants["variant_id"].astype(str))
        gwas_pos = gwas[gwas["variant_id"].astype(str).isin(wanted)]
        cols = ["variant_id", "chisq"] + (["n"] if "n" in gwas_pos else [])
        merged = variants.merge(gwas_pos[cols].drop_duplicates("variant_id"), on="variant_id", how="inner")
        if not merged.empty:
            return merged

    raise ValueError("GWAS file must contain chromosome/position or SNP IDs for alignment")

"""Helpers for BaselineLD-adjusted official LDSC analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_abc_annotations, load_gwas_sumstats
from .ld import annotation_triples_to_csr
from .simulation import context_variant_scores


ABC_COLUMN_NAMES = {
    "bipolar_neuron_from_iPSC-ENCODE": "ABC_bipolar_neuron_iPSC",
    "CD14-positive_monocyte-ENCODE": "ABC_CD14_monocyte",
    "white_adipose-Loft2014": "ABC_white_adipose",
    "gastrocnemius_medialis-ENCODE": "ABC_gastrocnemius",
    "uterus-ENCODE": "ABC_uterus",
}


def read_bim(path: str | Path) -> pd.DataFrame:
    """Read a PLINK BIM file using LDSC-compatible column names."""

    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
        dtype={"CHR": np.int64, "SNP": str, "CM": np.float64, "BP": np.int64, "A1": str, "A2": str},
    )
    if frame["SNP"].duplicated().any():
        raise ValueError(f"PLINK BIM file has duplicate SNP IDs: {path}")
    return frame


def write_abc_annotations(
    bim_by_chrom: dict[int, pd.DataFrame],
    abc_path: str | Path,
    contexts: list[str],
    output_dir: str | Path,
    min_score: float = 0.015,
) -> dict[str, int]:
    """Write continuous ABC annotations in the exact PLINK BIM row order.

    ABC CRE scores are summed over target genes independently for each cell
    context. The output keeps every 1000 Genomes SNP, including those with no
    ABC link, so its rows can be passed directly to ``ldsc.py --l2``.
    """

    if set(contexts) != set(ABC_COLUMN_NAMES):
        raise ValueError("contexts must match the configured five-context ABC panel")
    return write_continuous_abc_annotations(
        bim_by_chrom,
        abc_path,
        contexts,
        output_dir,
        column_names=[ABC_COLUMN_NAMES[context] for context in contexts],
        min_score=min_score,
        prefix="abc_baselineld",
    )


def write_continuous_abc_annotations(
    bim_by_chrom: dict[int, pd.DataFrame],
    abc_path: str | Path,
    contexts: list[str],
    output_dir: str | Path,
    column_names: list[str],
    min_score: float = 0.015,
    prefix: str = "abc_baselineld",
    exclude_self_promoter: bool = False,
    binary: bool = False,
) -> dict[str, int]:
    """Write continuous gene-summed ABC annotations in exact BIM row order."""

    if len(contexts) != len(column_names) or not contexts:
        raise ValueError("contexts and column_names must be non-empty and have equal length")
    if len(set(contexts)) != len(contexts) or len(set(column_names)) != len(column_names):
        raise ValueError("contexts and column_names must be unique")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_bim = pd.concat(
        [
            frame.assign(chrom=int(chrom), pos=frame["BP"].astype(np.int64), snp=frame["SNP"].astype(str))
            for chrom, frame in sorted(bim_by_chrom.items())
        ],
        ignore_index=True,
    )
    gwas = all_bim[["chrom", "pos", "snp"]].copy()
    gwas["variant_id"] = "chr" + gwas["chrom"].astype(str) + ":" + gwas["pos"].astype(str)
    annotation = load_abc_annotations(
        abc_path,
        gwas,
        min_score=min_score,
        cell_types=contexts,
        exclude_self_promoter=exclude_self_promoter,
        add_intercept=False,
    )
    missing_contexts = set(contexts).difference(annotation.mechanisms)
    if missing_contexts:
        raise ValueError(f"ABC data did not yield requested contexts: {sorted(missing_contexts)}")
    matrix = annotation_triples_to_csr(
        annotation.triples,
        n_variants=annotation.variants.shape[0],
        n_genes=annotation.genes.shape[0],
        n_mechanisms=len(annotation.mechanisms),
    )
    indices = [annotation.mechanisms.index(context) for context in contexts]
    scores = context_variant_scores(matrix, annotation.genes.shape[0], len(annotation.mechanisms), indices)
    if binary:
        scores = (scores > 0).astype(np.float32)
    score_frame = annotation.variants[["variant_id"]].copy()
    score_frame["chrom"] = score_frame["variant_id"].str.extract(r"^chr(\d+):", expand=False).astype(np.int64)
    score_frame["pos"] = score_frame["variant_id"].str.extract(r":(\d+)$", expand=False).astype(np.int64)
    score_columns = column_names
    score_frame[score_columns] = scores
    if score_frame.duplicated(["chrom", "pos"]).any():
        raise ValueError("ABC variant coordinates must be unique after annotation loading")

    counts: dict[str, int] = {}
    for chrom, bim in sorted(bim_by_chrom.items()):
        frame = bim.merge(
            score_frame[score_frame["chrom"].eq(chrom)][["pos", *score_columns]],
            left_on="BP",
            right_on="pos",
            how="left",
            validate="many_to_one",
        )
        if frame.shape[0] != bim.shape[0]:
            raise ValueError(f"ABC annotation row count changed on chromosome {chrom}")
        frame[score_columns] = frame[score_columns].fillna(0.0).astype(np.float32)
        frame = frame[["CHR", "SNP", "CM", "BP", *score_columns]]
        chromosome_prefix = output_dir / f"{prefix}.{chrom}"
        frame.to_csv(f"{chromosome_prefix}.annot.gz", sep="\t", index=False, compression="gzip")
        counts[str(chrom)] = int(frame.shape[0])
    return counts


def _read_peak_intervals(path: str | Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read hg19 BED intervals as per-chromosome starts and running end maxima."""

    peaks = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    peaks["chrom"] = pd.to_numeric(peaks["chrom"].astype(str).str.replace("chr", "", regex=False), errors="coerce")
    peaks["start"] = pd.to_numeric(peaks["start"], errors="coerce")
    peaks["end"] = pd.to_numeric(peaks["end"], errors="coerce")
    peaks = peaks.dropna().astype({"chrom": np.int64, "start": np.int64, "end": np.int64})
    peaks = peaks[peaks["chrom"].between(1, 22) & peaks["end"].gt(peaks["start"])]
    if peaks.empty:
        raise ValueError(f"No autosomal intervals found in {path}")
    intervals: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, group in peaks.groupby("chrom", sort=False):
        ordered = group.sort_values(["start", "end"], kind="mergesort")
        starts = ordered["start"].to_numpy(np.int64)
        # A running maximum supports overlapping and nested BED intervals.
        max_ends = np.maximum.accumulate(ordered["end"].to_numpy(np.int64))
        intervals[int(chrom)] = (starts, max_ends)
    return intervals


def _positions_in_intervals(
    positions: np.ndarray,
    intervals: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Return membership in 0-based half-open BED intervals for 1-based SNP positions."""

    result = np.zeros(positions.size, dtype=np.float32)
    if intervals is None or positions.size == 0:
        return result
    starts, max_ends = intervals
    previous = np.searchsorted(starts, positions, side="right") - 1
    valid = previous >= 0
    # A 1-based genomic position is in BED [start, end) exactly when
    # start < position <= end.
    result[valid] = (positions[valid] <= max_ends[previous[valid]]).astype(np.float32)
    return result


def write_peak_annotations(
    bim_by_chrom: dict[int, pd.DataFrame],
    peak_files: dict[str, str | Path],
    output_dir: str | Path,
    prefix: str = "peaks_baselineld",
) -> dict[str, int]:
    """Write binary ATAC/ChIP-seq peak annotations in exact BIM row order."""

    if not peak_files:
        raise ValueError("peak_files must not be empty")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(peak_files)
    intervals = {name: _read_peak_intervals(path) for name, path in peak_files.items()}
    counts: dict[str, int] = {}
    for chrom, bim in sorted(bim_by_chrom.items()):
        positions = bim["BP"].to_numpy(np.int64)
        values = np.column_stack(
            [_positions_in_intervals(positions, intervals[name].get(chrom)) for name in names]
        )
        frame = pd.concat((bim[["CHR", "SNP", "CM", "BP"]].reset_index(drop=True), pd.DataFrame(values, columns=names)), axis=1)
        frame.to_csv(output_dir / f"{prefix}.{chrom}.annot.gz", sep="\t", index=False, compression="gzip")
        counts[str(chrom)] = int(frame.shape[0])
    return counts


def write_hapmap3_sumstats(
    sumstats_path: str | Path,
    output_path: str | Path,
    bim_by_chrom: dict[int, pd.DataFrame] | None = None,
) -> int:
    """Export autosomal Z statistics and known sample sizes for LDSC.

    When reference BIMs are supplied, their SNP IDs replace source IDs using
    unique chromosome-position matches. This is required after coordinate
    liftover because source coordinate-based IDs still describe the old build.
    """

    source = load_gwas_sumstats(sumstats_path)
    if "n" not in source:
        raise ValueError("LDSC sumstats require known per-variant sample sizes")
    autosomal = source["chrom"].between(1, 22) if "chrom" in source else pd.Series(True, index=source.index)
    valid = source.loc[autosomal & source["z"].notna() & source["n"].gt(0)].copy()
    if bim_by_chrom is not None:
        if "chrom" not in valid or "pos" not in valid:
            raise ValueError("Reference SNP matching requires chromosome and position")
        reference = pd.concat(
            [bim[["CHR", "BP", "SNP"]] for _chrom, bim in sorted(bim_by_chrom.items())],
            ignore_index=True,
        )
        reference = reference.drop_duplicates(["CHR", "BP"], keep=False)
        valid = valid.merge(
            reference,
            left_on=["chrom", "pos"],
            right_on=["CHR", "BP"],
            how="inner",
            validate="many_to_one",
        )
        result = valid[["SNP", "n", "z"]]
    else:
        if "snp" not in valid:
            raise ValueError("LDSC sumstats require SNP IDs when reference BIMs are absent")
        result = valid.loc[valid["snp"].notna(), ["snp", "n", "z"]].rename(columns={"snp": "SNP"})
    result = result.rename(columns={"n": "N", "z": "Z"}).drop_duplicates("SNP")
    if result.empty:
        raise ValueError("No valid autosomal summary statistics with known sample sizes")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False, compression="gzip")
    return int(result.shape[0])

"""Helpers for BaselineLD-adjusted official LDSC analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_abc_annotations
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
    score_frame = annotation.variants[["variant_id"]].copy()
    score_frame["chrom"] = score_frame["variant_id"].str.extract(r"^chr(\d+):", expand=False).astype(np.int64)
    score_frame["pos"] = score_frame["variant_id"].str.extract(r":(\d+)$", expand=False).astype(np.int64)
    score_columns = [ABC_COLUMN_NAMES[context] for context in contexts]
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
        prefix = output_dir / f"abc_baselineld.{chrom}"
        frame.to_csv(f"{prefix}.annot.gz", sep="\t", index=False, compression="gzip")
        counts[str(chrom)] = int(frame.shape[0])
    return counts


def write_hapmap3_sumstats(
    ad_sumstats_path: str | Path,
    output_path: str | Path,
) -> int:
    """Export native AD Z statistics and known effective sample sizes for LDSC."""

    source = pd.read_csv(
        ad_sumstats_path,
        sep="\t",
        compression="gzip",
        usecols=["CHR", "SNP", "Z", "Neff"],
    )
    source["CHR"] = pd.to_numeric(source["CHR"], errors="coerce")
    source["Z"] = pd.to_numeric(source["Z"], errors="coerce")
    source["N"] = pd.to_numeric(source["Neff"], errors="coerce")
    result = source.loc[
        source["CHR"].between(1, 22) & source["SNP"].notna() & source["Z"].notna() & source["N"].gt(0),
        ["SNP", "N", "Z"],
    ].drop_duplicates("SNP")
    if result.empty:
        raise ValueError("No valid autosomal AD summary statistics with known Neff")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False, compression="gzip")
    return int(result.shape[0])

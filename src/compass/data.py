from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CELLTYPE_NAMES = {
    "Ast": "astrocyte",
    "End": "endothelial",
    "Ext": "excitatory_neuron",
    "IN": "inhibitory_neuron",
    "MG": "microglia",
    "OD": "oligodendrocyte",
    "OPC": "opc",
}


@dataclass(frozen=True)
class AnnotationData:
    variants: pd.DataFrame
    genes: pd.DataFrame
    mechanisms: list[str]
    triples: pd.DataFrame


def _variant_id(chr_values: pd.Series, pos_values: pd.Series) -> pd.Series:
    return "chr" + chr_values.astype(str) + ":" + pos_values.astype(str)


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
        df["chrom"] = pd.to_numeric(
            df["chrom"].astype(str).str.replace("chr", "", regex=False), errors="coerce"
        )
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


def load_gwas_sumstats(path: str | Path) -> pd.DataFrame:
    """Load GWAS summary statistics with flexible column normalization."""

    path = Path(path).expanduser()
    compression = "gzip" if path.suffix == ".gz" else None
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

    if "variant_id" in gwas.columns and "variant_id" in variants.columns:
        wanted = set(variants["variant_id"].astype(str))
        gwas_pos = gwas[gwas["variant_id"].astype(str).isin(wanted)]
        cols = ["variant_id", "chisq"] + (["n"] if "n" in gwas_pos else [])
        merged = variants.merge(gwas_pos[cols].drop_duplicates("variant_id"), on="variant_id", how="inner")
        if not merged.empty:
            return merged

    if "snp" in gwas.columns:
        candidates = variants.rename(columns={"MarkerID": "snp"})
        if "snp" in candidates.columns:
            wanted = set(candidates["snp"].astype(str))
            gwas_snp = gwas[gwas["snp"].astype(str).isin(wanted)]
            cols = ["snp", "chisq"] + (["n"] if "n" in gwas_snp else [])
            merged = candidates.merge(gwas_snp[cols].drop_duplicates("snp"), on="snp", how="inner")
            if not merged.empty:
                return merged

    raise ValueError("GWAS file must contain chromosome/position or SNP IDs for alignment")

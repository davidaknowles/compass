"""Preparation and loading of cell-type predicted-eQTL annotations."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .data import AnnotationData


PREDICTED_EQTL_CELLS = {
    "Ast": "astrocyte",
    "Exc": "excitatory_neuron",
    "Inh": "inhibitory_neuron",
    "Mic": "microglia",
    "OPC": "opc",
    "Oli": "oligodendrocyte",
}


def _read_source_chromosome(path: Path, chrom: int, min_score: float) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    table = dataset.to_table(
        columns=["pos", "gene_id", "pred_prob"],
        filter=(ds.field("chr") == f"chr{chrom}") & (ds.field("pred_prob") >= min_score),
    )
    return table.to_pandas().rename(columns={"gene_id": "gene", "pred_prob": "score"})


def _liftover_positions(
    positions: np.ndarray,
    chrom: int,
    chain_path: Path,
    liftover_executable: str,
) -> pd.DataFrame:
    """Lift unique one-base positions, retaining only unique destination mappings."""

    positions = np.unique(np.asarray(positions, dtype=np.int64))
    with tempfile.TemporaryDirectory(prefix=f"pred-eqtl-chr{chrom}-") as directory:
        directory = Path(directory)
        source = directory / "source.bed"
        mapped = directory / "mapped.bed"
        unmapped = directory / "unmapped.bed"
        with source.open("w") as handle:
            for index, position in enumerate(positions):
                handle.write(f"chr{chrom}\t{position - 1}\t{position}\t{index}\n")
        subprocess.run(
            [liftover_executable, str(source), str(chain_path), str(mapped), str(unmapped)],
            check=True,
        )
        if mapped.stat().st_size == 0:
            return pd.DataFrame(columns=["source_pos", "pos"])
        lifted = pd.read_csv(
            mapped,
            sep="\t",
            header=None,
            usecols=[0, 1, 2, 3],
            names=["destination_chrom", "start", "end", "source_index"],
        )
    lifted["source_index"] = pd.to_numeric(lifted["source_index"], errors="coerce")
    lifted = lifted.dropna(subset=["source_index"])
    lifted["source_index"] = lifted["source_index"].astype(np.int64)
    lifted = lifted[lifted["destination_chrom"].eq(f"chr{chrom}")]
    lifted = lifted[lifted["end"].sub(lifted["start"]).eq(1)]
    lifted = lifted.drop_duplicates("source_index", keep=False)
    lifted["source_pos"] = positions[lifted["source_index"].to_numpy(np.int64)]
    lifted["pos"] = lifted["end"].astype(np.int64)
    return lifted[["source_pos", "pos"]]


def prepare_predicted_eqtl_annotations(
    source_root: str | Path,
    output_root: str | Path,
    chain_path: str | Path,
    min_score: float = 0.9,
    liftover_executable: str = "liftOver",
    chromosomes: Iterable[int] = range(1, 23),
) -> dict:
    """Filter GRCh38 predictions, lift them to hg19, and write chromosome partitions."""

    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be in [0, 1]")
    source_root = Path(source_root).expanduser()
    output_root = Path(output_root).expanduser()
    chain_path = Path(chain_path).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    for chrom in chromosomes:
        frames: list[pd.DataFrame] = []
        for source_cell, mechanism in PREDICTED_EQTL_CELLS.items():
            source = source_root / f"{source_cell}_mega_eQTL" / "predictions_parquet_catboost_GPN"
            if not source.exists():
                raise FileNotFoundError(source)
            frame = _read_source_chromosome(source, int(chrom), min_score)
            if not frame.empty:
                frame["mechanism"] = mechanism
                frames.append(frame)
        links = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        raw_count = int(links.shape[0])
        if raw_count:
            links["pos"] = pd.to_numeric(links["pos"], errors="coerce")
            links["score"] = pd.to_numeric(links["score"], errors="coerce")
            links = links.dropna(subset=["pos", "gene", "score"])
            links["pos"] = links["pos"].astype(np.int64)
            mapping = _liftover_positions(
                links["pos"].to_numpy(np.int64), int(chrom), chain_path, liftover_executable
            )
            links = links.rename(columns={"pos": "source_pos"}).merge(
                mapping, on="source_pos", how="inner", validate="many_to_one"
            )
            links["chrom"] = np.int16(chrom)
            links["score"] = links["score"].astype(np.float32)
            links = links.groupby(
                ["chrom", "pos", "gene", "mechanism"], as_index=False, observed=True
            )["score"].max()
        else:
            links = pd.DataFrame(columns=["chrom", "pos", "gene", "mechanism", "score"])
        partition = output_root / f"chrom={chrom}"
        partition.mkdir(parents=True, exist_ok=True)
        links.to_parquet(partition / "links.parquet", index=False, compression="zstd")
        counts[str(chrom)] = {"source_links": raw_count, "lifted_links": int(links.shape[0])}
        print(
            f"chromosome {chrom}: {raw_count:,} filtered source links, "
            f"{links.shape[0]:,} unique lifted links",
            flush=True,
        )
    manifest = {
        "coordinate_build": "hg19",
        "source_coordinate_build": "hg38",
        "min_score": min_score,
        "cells": PREDICTED_EQTL_CELLS,
        "counts": counts,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def read_prepared_chromosome(path: str | Path, chrom: int, min_score: float = 0.9) -> pd.DataFrame:
    """Read one prepared hg19 partition and apply a reproducible score threshold."""

    path = Path(path).expanduser()
    partition = path / f"chrom={chrom}"
    if not partition.exists():
        return pd.DataFrame(columns=["chrom", "pos", "gene", "mechanism", "score"])
    frame = pd.read_parquet(partition, filters=[("score", ">=", min_score)])
    return frame[frame["score"].ge(min_score)].copy()


def load_predicted_eqtl_annotations(
    prepared_path: str | Path,
    gwas: pd.DataFrame,
    min_score: float = 0.9,
    add_intercept: bool = True,
) -> AnnotationData:
    """Load sparse predicted-eQTL links while retaining all usable GWAS variants."""

    if "chrom" not in gwas or "pos" not in gwas:
        raise ValueError("Predicted-eQTL annotations require GWAS chromosome and position columns")
    variants = gwas.dropna(subset=["chrom", "pos"]).copy()
    variants["chrom"] = pd.to_numeric(
        variants["chrom"].astype(str).str.replace("chr", "", regex=False), errors="coerce"
    )
    variants["pos"] = pd.to_numeric(variants["pos"], errors="coerce")
    variants = variants[variants["chrom"].between(1, 22) & variants["pos"].notna()].copy()
    variants[["chrom", "pos"]] = variants[["chrom", "pos"]].astype(np.int64)
    variants["variant_id"] = "chr" + variants["chrom"].astype(str) + ":" + variants["pos"].astype(str)
    variants["MarkerID"] = (
        variants["snp"].astype(str) if "snp" in variants else variants["variant_id"].astype(str)
    )
    variants = variants[["variant_id", "chrom", "pos", "MarkerID"]].drop_duplicates("variant_id")
    variants = variants.sort_values(["chrom", "pos", "variant_id"]).reset_index(drop=True)
    variants["variant_idx"] = np.arange(variants.shape[0], dtype=np.int64)

    mechanisms = list(PREDICTED_EQTL_CELLS.values())
    mechanism_offset = int(add_intercept)
    mechanism_to_idx = {name: i + mechanism_offset for i, name in enumerate(mechanisms)}
    gene_to_idx: dict[str, int] = {}
    triple_parts: list[pd.DataFrame] = []
    for chrom, chromosome_variants in variants.groupby("chrom", sort=False):
        links = read_prepared_chromosome(prepared_path, int(chrom), min_score)
        if links.empty:
            continue
        links = links[links["mechanism"].isin(mechanism_to_idx)].copy()
        links = links.merge(
            chromosome_variants[["pos", "variant_idx"]], on="pos", how="inner", validate="many_to_one"
        )
        if links.empty:
            continue
        new_genes = sorted(set(links["gene"].astype(str)).difference(gene_to_idx))
        gene_to_idx.update({gene: len(gene_to_idx) + i for i, gene in enumerate(new_genes)})
        part = pd.DataFrame(
            {
                "variant_idx": links["variant_idx"].to_numpy(np.int64),
                "gene_idx": links["gene"].astype(str).map(gene_to_idx).to_numpy(np.int64),
                "mechanism_idx": links["mechanism"].map(mechanism_to_idx).to_numpy(np.int64),
                "value": links["score"].to_numpy(np.float32),
            }
        )
        triple_parts.append(part)
    if not triple_parts:
        raise ValueError("No predicted-eQTL links matched GWAS coordinates")
    triples = pd.concat(triple_parts, ignore_index=True)
    if add_intercept:
        intercept = triples[["variant_idx", "gene_idx"]].drop_duplicates().copy()
        intercept["mechanism_idx"] = 0
        intercept["value"] = np.float32(1.0)
        triples = pd.concat([triples, intercept], ignore_index=True)
    genes = pd.DataFrame({"gene": list(gene_to_idx), "gene_idx": list(gene_to_idx.values())})
    return AnnotationData(
        variants=variants,
        genes=genes.sort_values("gene_idx").reset_index(drop=True),
        mechanisms=(["intercept"] if add_intercept else []) + mechanisms,
        triples=triples,
    )


def write_predicted_eqtl_annotations(
    bim_by_chrom: dict[int, pd.DataFrame],
    prepared_path: str | Path,
    output_dir: str | Path,
    min_score: float = 0.9,
    prefix: str = "predicted_eqtl_baselineld",
) -> dict[str, int]:
    """Write gene-summed predicted-eQTL scores in exact PLINK BIM row order."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mechanisms = list(PREDICTED_EQTL_CELLS.values())
    columns = [f"PredEQTL_{name}" for name in mechanisms]
    counts: dict[str, int] = {}
    for chrom, bim in sorted(bim_by_chrom.items()):
        links = read_prepared_chromosome(prepared_path, chrom, min_score)
        if links.empty:
            scores = pd.DataFrame(columns=["pos", *columns])
        else:
            scores = links.groupby(["pos", "mechanism"], observed=True)["score"].sum().unstack(fill_value=0)
            scores = scores.reindex(columns=mechanisms, fill_value=0.0)
            scores.columns = columns
            scores = scores.reset_index()
        frame = bim.merge(scores, left_on="BP", right_on="pos", how="left", validate="many_to_one")
        frame[columns] = frame[columns].fillna(0.0).astype(np.float32)
        frame = frame[["CHR", "SNP", "CM", "BP", *columns]]
        frame.to_csv(output_dir / f"{prefix}.{chrom}.annot.gz", sep="\t", index=False, compression="gzip")
        counts[str(chrom)] = int(frame.shape[0])
    return counts

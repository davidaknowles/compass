#!/usr/bin/env python
"""Build official LDSC inputs for gene-summed five-context ABC annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from compass.data import load_gwas_sumstats, make_training_table
from compass.ld import filter_variants_to_ukbb_ld, read_ukbb_ld_metadata, ukbb_ld_block_stems
from compass.official_ldsc import write_ldsc_reference
from compass.simulation import context_variant_scores
from run import _cache_key, _cache_paths, _dataset_cache_exists, _load_dataset_cache


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"
CONTEXTS = [
    "bipolar_neuron_from_iPSC-ENCODE",
    "CD14-positive_monocyte-ENCODE",
    "white_adipose-Loft2014",
    "gastrocnemius_medialis-ENCODE",
    "uterus-ENCODE",
]


def _real_cache_paths(data_root: Path):
    gwas_path = data_root / "raw" / "ad_gwas" / "AD_sumstats_Jansenetal_2019sept.txt.gz"
    args = SimpleNamespace(
        annotation_source="abc",
        no_intercept=False,
        n_samples=None,
        abc_cell_types=",".join(CONTEXTS),
        abc_score_column="ABC.Score",
        abc_min_score=0.015,
        ld_r2_cutoff=0.01,
    )
    return _cache_paths(data_root / "cache", _cache_key(args, gwas_path)), gwas_path


def _reconstruct_variants(data_root: Path, gwas_path: Path) -> pd.DataFrame:
    """Reproduce run.py's all-row UKBB-aligned ordering with metadata fields."""

    gwas = load_gwas_sumstats(gwas_path)
    variants = gwas.dropna(subset=["chrom", "pos"]).copy()
    variants = variants[variants["chrom"].between(1, 22)].copy()
    variants["chrom"] = variants["chrom"].astype(int)
    variants["pos"] = variants["pos"].astype(int)
    variants["MarkerID"] = variants["snp"].astype(str) if "snp" in variants else variants["variant_id"].astype(str)
    variants = variants.drop_duplicates("variant_id").sort_values(["chrom", "pos", "variant_id"]).reset_index(drop=True)
    # load_abc_annotations retains MarkerID rather than the source GWAS SNP
    # column. Match that shape so make_training_table can rename it safely.
    variants = variants[["variant_id", "chrom", "pos", "MarkerID"]].copy()
    variants["variant_idx"] = np.arange(variants.shape[0], dtype=np.int64)
    ld_dir = data_root / "raw" / "ukbb_ld"
    ann_variants = filter_variants_to_ukbb_ld(variants, str(ld_dir), n_jobs=8, progress_every=50)
    training = make_training_table(ann_variants, gwas)
    training = training.drop_duplicates("variant_idx").sort_values("variant_idx").reset_index(drop=True)
    filtered = training["variant_idx"].to_numpy(np.int64)
    out = ann_variants.iloc[filtered].copy().reset_index(drop=True)
    out["variant_idx"] = np.arange(out.shape[0], dtype=np.int64)
    return out


def _attach_ukbb_snp_identity(variants: pd.DataFrame, ld_dir: Path) -> pd.DataFrame:
    """Attach UKBB rsid and alleles using run.py's SNP-first matching rule."""

    result = variants.copy()
    result["SNP"] = pd.NA
    result["A1"] = pd.NA
    result["A2"] = pd.NA
    for chrom, start, end, stem in ukbb_ld_block_stems():
        block = result[
            result["chrom"].eq(chrom) & result["pos"].ge(start) & result["pos"].lt(end)
        ]
        if block.empty:
            continue
        meta = read_ukbb_ld_metadata(str(ld_dir), stem)
        by_marker = dict(zip(block["MarkerID"].astype(str), block["variant_idx"]))
        by_position = dict(zip(block["variant_id"].astype(str), block["variant_idx"]))
        target = meta["SNP"].astype(str).map(by_marker)
        target = target.where(target.notna(), meta["variant_id"].map(by_position))
        hit = meta.loc[target.notna(), ["SNP", "A1", "A2"]].copy()
        hit["variant_idx"] = target.loc[target.notna()].to_numpy(np.int64)
        hit = hit.drop_duplicates("variant_idx")
        result.loc[hit["variant_idx"], ["SNP", "A1", "A2"]] = hit[["SNP", "A1", "A2"]].to_numpy()
    if result[["SNP", "A1", "A2"]].isna().any().any():
        raise ValueError("Could not recover UKBB rsid/alleles for every LDSC variant")
    result["CHR"] = result["chrom"].astype(np.int64)
    result["BP"] = result["pos"].astype(np.int64)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_root / "results" / "official_sldsc_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, gwas_path = _real_cache_paths(data_root)
    if not _dataset_cache_exists(paths):
        raise FileNotFoundError("Build the five-context real-data cache before creating official LDSC inputs")
    dataset, genes, mechanisms, _, _ = _load_dataset_cache(paths)
    variants = _reconstruct_variants(data_root, gwas_path)
    if variants.shape[0] != dataset.n_variants or not np.array_equal(variants["chrom"].to_numpy(np.int64), dataset.chrom):
        raise ValueError("Reconstructed variant order does not match the cached COMPASS dataset")
    variants = _attach_ukbb_snp_identity(variants, data_root / "raw" / "ukbb_ld")
    indices = [mechanisms.index(context) for context in CONTEXTS]
    scores = context_variant_scores(dataset.A, genes.shape[0], len(mechanisms), indices)
    manifest = write_ldsc_reference(out_dir, dataset.ld_blocks, variants, scores, CONTEXTS)
    sumstats = variants[["SNP", "A1", "A2"]].copy()
    sumstats["N"] = np.asarray(dataset.n_samples, dtype=np.float64)
    sumstats["Z"] = np.sqrt(np.maximum(dataset.chisq, 0.0))
    sumstats_path = out_dir / "ad.sumstats.gz"
    sumstats.to_csv(sumstats_path, sep="\t", index=False, compression="gzip")
    manifest.update(
        {
            "sumstats": str(sumstats_path),
            "n_variants": int(dataset.n_variants),
            "contexts": CONTEXTS,
            "ld_r2_cutoff": 0.01,
        }
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

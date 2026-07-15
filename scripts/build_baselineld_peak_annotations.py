#!/usr/bin/env python
"""Build binary brain cell-type ATAC and histone peak annotations for LDSC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from compass.baselineld import read_bim, write_hapmap3_sumstats, write_peak_annotations


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"
PEAK_SPECS = {
    "ATAC_LHX2": ("ATAC", "LHX2_optimal_peak_IDR_ENCODE.ATAC.bed"),
    "ATAC_NeuN": ("ATAC", "NeuN_optimal_peak_IDR_ENCODE.ATAC.bed"),
    "ATAC_Olig2": ("ATAC", "Olig2_optimal_peak_IDR_ENCODE.ATAC.bed"),
    "ATAC_PU1": ("ATAC", "PU1_optimal_peak_IDR_ENCODE.ATAC.bed"),
    "H3K27ac_LHX2": ("H3K27ac", "LHX2_optimal_peak.H3K27.bed"),
    "H3K27ac_NeuN": ("H3K27ac", "NeuN_optimal_peak.H3K27.bed"),
    "H3K27ac_Olig2": ("H3K27ac", "Olig2_optimal_peak.H3K27.bed"),
    "H3K27ac_PU1": ("H3K27ac", "PU1_optimal_peak.H3K27.bed"),
    "H3K4me3_LHX2": ("H3K4me3", "LHX2_optimal_peak.H3K4me3.bed"),
    "H3K4me3_NeuN": ("H3K4me3", "NeuN_optimal_peak.H3K4me3.bed"),
    "H3K4me3_Olig2": ("H3K4me3", "Olig2_optimal_peak.H3K4me3.bed"),
    "H3K4me3_PU1": ("H3K4me3", "PU1_optimal_peak.H3K4me3.bed"),
}


def _one_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name} below {root}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--peaks-root", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    reference_root = Path(args.reference_root).expanduser() if args.reference_root else data_root / "raw" / "ldsc_1000g"
    peaks_root = Path(args.peaks_root).expanduser() if args.peaks_root else data_root / "raw" / "brain-cell-type-peak-files"
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_root / "results" / "official_sldsc_peaks_inputs"
    bims: dict[int, pd.DataFrame] = {}
    bfile_prefixes: dict[str, str] = {}
    regression_snp_counts: dict[str, int] = {}
    for chrom in range(1, 23):
        bim_path = _one_file(reference_root, f"1000G.EUR.QC.{chrom}.bim")
        bims[chrom] = read_bim(bim_path)
        bfile_prefixes[str(chrom)] = str(bim_path.with_suffix(""))
        baseline_ldscore = _one_file(reference_root, f"baselineLD.{chrom}.l2.ldscore.gz")
        regression = pd.read_csv(baseline_ldscore, sep="\t", usecols=["SNP"])
        snp_path = out_dir / "regression_snps" / f"{chrom}.snps"
        snp_path.parent.mkdir(parents=True, exist_ok=True)
        regression["SNP"].to_csv(snp_path, index=False, header=False)
        regression_snp_counts[str(chrom)] = int(regression.shape[0])

    peak_files = {name: peaks_root / directory / filename for name, (directory, filename) in PEAK_SPECS.items()}
    missing = [str(path) for path in peak_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing peak files: {missing}")
    annotation_counts = write_peak_annotations(bims, peak_files, out_dir / "annotation")
    n_sumstats = write_hapmap3_sumstats(
        data_root / "raw" / "ad_gwas" / "AD_sumstats_Jansenetal_2019sept.txt.gz",
        out_dir / "ad.sumstats.gz",
    )
    manifest = {
        "annotations": list(peak_files),
        "peak_files": {name: str(path) for name, path in peak_files.items()},
        "annotation_prefix": str(out_dir / "annotation" / "peaks_baselineld."),
        "bfile_prefixes": bfile_prefixes,
        "regression_snp_counts": regression_snp_counts,
        "annotation_row_counts": annotation_counts,
        "sumstats": str(out_dir / "ad.sumstats.gz"),
        "n_sumstats": n_sumstats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

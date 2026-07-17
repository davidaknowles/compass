#!/usr/bin/env python
"""Build five continuous ABC annotations for a standard 1000 Genomes LDSC panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from compass.baselineld import read_bim, write_abc_annotations, write_hapmap3_sumstats


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"
CONTEXTS = [
    "bipolar_neuron_from_iPSC-ENCODE",
    "CD14-positive_monocyte-ENCODE",
    "white_adipose-Loft2014",
    "gastrocnemius_medialis-ENCODE",
    "uterus-ENCODE",
]


def _one_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name} below {root}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--gwas", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    reference_root = Path(args.reference_root).expanduser() if args.reference_root else data_root / "raw" / "ldsc_1000g"
    gwas_path = Path(args.gwas).expanduser() if args.gwas else data_root / "raw" / "ad_gwas_2026" / "GCST90704647.hg19.tsv.gz"
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_root / "results" / "official_sldsc_baselineld_inputs"
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

    annotation_counts = write_abc_annotations(
        bims,
        data_root / "raw" / "abc" / "AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz",
        CONTEXTS,
        out_dir / "annotation",
    )
    n_sumstats = write_hapmap3_sumstats(
        gwas_path,
        out_dir / "sumstats.gz",
    )
    manifest = {
        "contexts": CONTEXTS,
        "annotation_prefix": str(out_dir / "annotation" / "abc_baselineld."),
        "bfile_prefixes": bfile_prefixes,
        "regression_snp_counts": regression_snp_counts,
        "annotation_row_counts": annotation_counts,
        "sumstats": str(out_dir / "sumstats.gz"),
        "gwas": str(gwas_path),
        "n_sumstats": n_sumstats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

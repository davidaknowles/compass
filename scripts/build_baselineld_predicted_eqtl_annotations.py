#!/usr/bin/env python
"""Build BaselineLD inputs from prepared predicted-eQTL links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from compass.baselineld import read_bim, write_hapmap3_sumstats
from compass.predicted_eqtl import PREDICTED_EQTL_CELLS, write_predicted_eqtl_annotations


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"


def _one_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name} below {root}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--predicted-eqtl-path", default=None)
    parser.add_argument("--gwas", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-score", type=float, default=0.9)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    reference_root = (
        Path(args.reference_root).expanduser()
        if args.reference_root
        else data_root / "raw" / "ldsc_1000g"
    )
    predicted_eqtl_path = (
        Path(args.predicted_eqtl_path).expanduser()
        if args.predicted_eqtl_path
        else data_root / "raw" / "predicted_eqtl" / "links.hg19.min0.9"
    )
    out_dir = Path(args.out_dir).expanduser()
    bims: dict[int, pd.DataFrame] = {}
    bfile_prefixes: dict[str, str] = {}
    regression_snp_counts: dict[str, int] = {}
    for chrom in range(1, 23):
        bim_path = _one_file(reference_root, f"1000G.EUR.QC.{chrom}.bim")
        bims[chrom] = read_bim(bim_path)
        bfile_prefixes[str(chrom)] = str(bim_path.with_suffix(""))
        regression = pd.read_csv(
            _one_file(reference_root, f"baselineLD.{chrom}.l2.ldscore.gz"),
            sep="\t",
            usecols=["SNP"],
        )
        snp_path = out_dir / "regression_snps" / f"{chrom}.snps"
        snp_path.parent.mkdir(parents=True, exist_ok=True)
        regression["SNP"].to_csv(snp_path, index=False, header=False)
        regression_snp_counts[str(chrom)] = int(regression.shape[0])

    annotation_counts = write_predicted_eqtl_annotations(
        bims,
        predicted_eqtl_path,
        out_dir / "annotation",
        min_score=args.min_score,
    )
    n_sumstats = write_hapmap3_sumstats(args.gwas, out_dir / "sumstats.gz", bims)
    contexts = list(PREDICTED_EQTL_CELLS.values())
    manifest = {
        "contexts": contexts,
        "annotation_columns": [f"PredEQTL_{name}" for name in contexts],
        "min_score": args.min_score,
        "annotation_prefix": str(out_dir / "annotation" / "predicted_eqtl_baselineld."),
        "bfile_prefixes": bfile_prefixes,
        "regression_snp_counts": regression_snp_counts,
        "annotation_row_counts": annotation_counts,
        "sumstats": str(out_dir / "sumstats.gz"),
        "gwas": str(Path(args.gwas).expanduser()),
        "n_sumstats": n_sumstats,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

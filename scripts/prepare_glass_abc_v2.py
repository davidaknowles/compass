#!/usr/bin/env python
"""Lift Glass Lab brain ABC v2 predictions from GRCh38 to hg19."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compass.glass_abc import GLASS_ABC_V2_CELLS, prepare_glass_abc_v2, prepare_glass_expressed_tss


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"
DEFAULT_SOURCE_ROOT = Path(
    "/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/annotations/brain_all/"
    "glass_lab_fastq/processed_files/ABC_data"
)
DEFAULT_CHAIN = Path("/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/liftover/hg38ToHg19.over.chain.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--chain", default=str(DEFAULT_CHAIN))
    parser.add_argument("--liftover", default="liftOver")
    parser.add_argument("--out", default=None)
    parser.add_argument("--tss-out", default=None)
    parser.add_argument("--cells", default=",".join(GLASS_ABC_V2_CELLS))
    args = parser.parse_args()

    cells = tuple(cell.strip() for cell in args.cells.split(",") if cell.strip())
    if not cells:
        parser.error("--cells must name at least one cell type")
    output = (
        Path(args.out).expanduser()
        if args.out
        else DEFAULT_DATA_ROOT / "raw" / "abc" / "glass_brain_v2.hg19.tsv.gz"
    )
    manifest = prepare_glass_abc_v2(args.source_root, output, args.chain, args.liftover, cells)
    print(json.dumps(manifest, sort_keys=True))
    tss_output = (
        Path(args.tss_out).expanduser()
        if args.tss_out
        else DEFAULT_DATA_ROOT / "raw" / "abc" / "glass_brain_v2.expressed_tss.hg19.tsv.gz"
    )
    tss_manifest = prepare_glass_expressed_tss(args.source_root, tss_output, args.chain, args.liftover, cells)
    print(json.dumps(tss_manifest, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Prepare high-confidence predicted-eQTL links for genome-wide analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

from compass.predicted_eqtl import prepare_predicted_eqtl_annotations


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-score", type=float, default=0.9)
    parser.add_argument("--liftover", default="liftOver")
    args = parser.parse_args()
    output = (
        Path(args.output).expanduser()
        if args.output
        else DEFAULT_DATA_ROOT / "raw" / "predicted_eqtl" / f"links.hg19.min{args.min_score:g}"
    )
    prepare_predicted_eqtl_annotations(
        args.source_root,
        output,
        args.chain,
        min_score=args.min_score,
        liftover_executable=args.liftover,
    )


if __name__ == "__main__":
    main()

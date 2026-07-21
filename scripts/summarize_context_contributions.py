#!/usr/bin/env python
"""Write aggregate context contributions for an existing hierarchical fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from compass.model import context_heritability_from_masses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_prefix", help="Result prefix without .metadata.json or .npz")
    parser.add_argument("--data-root", type=Path, default=Path.home() / "knowles_lab/data/compass")
    args = parser.parse_args()

    prefix = Path(args.run_prefix).expanduser()
    with Path(f"{prefix}.metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with np.load(f"{prefix}.npz", allow_pickle=False) as fit:
        B = fit["B"]
        context_effects = fit["context_effects"]

    cache_prefix = args.data_root.expanduser() / "cache" / metadata["cache_key"]
    mechanisms_path = cache_prefix.with_suffix(".mechanisms.json")
    annotation_path = cache_prefix.with_suffix(".A.npz")
    with mechanisms_path.open(encoding="utf-8") as handle:
        mechanisms = json.load(handle)
    annotation_mass = np.asarray(sp.load_npz(annotation_path).sum(axis=0)).ravel().reshape(B.shape)
    summary = context_heritability_from_masses(
        annotation_mass,
        B,
        np.asarray(metadata["context_annotation_counts"]),
        context_effects,
    )
    output = pd.DataFrame({"context": mechanisms, **summary})
    output.to_csv(f"{prefix}.context_contributions.tsv", sep="\t", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()

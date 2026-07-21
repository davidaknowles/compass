#!/usr/bin/env python
"""Merge completed single-fold hierarchical CV checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser(), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    merged = _load(args.base)
    folds = np.asarray(merged["folds"], dtype=np.int64)
    fold_to_row = {int(fold): row for row, fold in enumerate(folds)}
    contract_fields = (
        "version",
        "lambdas",
        "fixed_context_effects",
        "scale_fixed_context_effects",
        "freeze_scaled_context_effects",
    )
    for shard_path in args.shard:
        shard = _load(shard_path)
        for field in contract_fields:
            if not np.array_equal(shard[field], merged[field]):
                raise ValueError(f"{shard_path}: incompatible {field}")
        for shard_row, fold in enumerate(np.asarray(shard["folds"], dtype=np.int64)):
            if int(fold) not in fold_to_row:
                raise ValueError(f"{shard_path}: fold {fold} is absent from base checkpoint")
            if int(shard["next_lambda_index"][shard_row]) != len(merged["lambdas"]):
                raise ValueError(f"{shard_path}: fold {fold} is incomplete")
            row = fold_to_row[int(fold)]
            for field in ("B", "context_effects", "tau", "scores", "next_lambda_index"):
                merged[field][row] = shard[field][shard_row]

    incomplete = folds[merged["next_lambda_index"] != len(merged["lambdas"])]
    if incomplete.size:
        raise ValueError(f"merged checkpoint has incomplete folds: {incomplete.tolist()}")
    if not np.isfinite(merged["scores"]).all():
        raise ValueError("merged checkpoint contains non-finite CV scores")

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **merged)
    temporary.replace(output)
    means = merged["scores"].mean(axis=0)
    print(f"merged folds={folds.tolist()}")
    for value, score in zip(merged["lambdas"], means, strict=True):
        print(f"lambda={value:g} mean_cv_mse={score:.9g}")


if __name__ == "__main__":
    main()

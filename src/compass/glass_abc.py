"""Preparation helpers for Glass Lab brain-cell ABC predictions."""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


GLASS_ABC_V2_CELLS = ("astrocyte", "microglia", "neuron", "oligodendrocyte")
_ABC_FILENAME = "EnhancerPredictionsAllPutative.ForVariantOverlap.shrunk150bp.tsv.gz"
_REQUIRED_COLUMNS = ["chr", "start", "end", "TargetGene", "ABC.Score", "CellType"]


def glass_abc_v2_paths(source_root: str | Path, cells: tuple[str, ...] = GLASS_ABC_V2_CELLS) -> dict[str, Path]:
    """Return the expected v2 prediction file for every requested cell type."""

    root = Path(source_root)
    paths = {
        cell: root / f"ABC_results_{cell}_v2" / cell / "Predictions" / _ABC_FILENAME for cell in cells
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Glass Lab ABC v2 predictions: " + ", ".join(missing))
    return paths


def prepare_glass_abc_v2(
    source_root: str | Path,
    output_path: str | Path,
    chain_path: str | Path,
    liftover: str = "liftOver",
    cells: tuple[str, ...] = GLASS_ABC_V2_CELLS,
) -> dict[str, object]:
    """Lift GRCh38 Glass ABC intervals to hg19 and combine selected cell types."""

    output_path = Path(output_path)
    chain_path = Path(chain_path)
    if not chain_path.is_file():
        raise FileNotFoundError(f"LiftOver chain file not found: {chain_path}")
    executable = shutil.which(liftover)
    if executable is None:
        raise FileNotFoundError(f"Could not find liftOver executable: {liftover}")

    source_paths = glass_abc_v2_paths(source_root, cells)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    temp_output = output_path.with_name(f".{output_path.name}.tmp")
    with tempfile.TemporaryDirectory(prefix="glass_abc_liftover_", dir=output_path.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        with gzip.open(temp_output, "wt", encoding="utf-8", newline="") as destination:
            destination.write("\t".join(_REQUIRED_COLUMNS) + "\n")
            for cell, source_path in source_paths.items():
                source_bed = temp_dir / f"{cell}.grch38.bed"
                mapped_bed = temp_dir / f"{cell}.hg19.bed"
                unmapped_bed = temp_dir / f"{cell}.unmapped.bed"
                with gzip.open(source_path, "rt", encoding="utf-8", newline="") as source, source_bed.open(
                    "w", encoding="utf-8", newline=""
                ) as bed:
                    header = source.readline().rstrip("\r\n").split("\t")
                    missing = set(_REQUIRED_COLUMNS).difference(header)
                    if missing:
                        raise ValueError(f"{source_path} is missing required columns: {sorted(missing)}")
                    input_rows = 0
                    for line in source:
                        if line.strip():
                            bed.write(line)
                            input_rows += 1
                subprocess.run(
                    [
                        executable,
                        "-bedPlus=3",
                        "-tab",
                        str(source_bed),
                        str(chain_path),
                        str(mapped_bed),
                        str(unmapped_bed),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                if mapped_bed.exists() and mapped_bed.stat().st_size:
                    frame = pd.read_csv(mapped_bed, sep="\t", header=None, names=header, usecols=_REQUIRED_COLUMNS)
                    frame["CellType"] = cell
                    frame.to_csv(destination, sep="\t", index=False, header=False)
                    mapped_rows = int(frame.shape[0])
                else:
                    mapped_rows = 0
                counts[cell] = {
                    "input_rows": input_rows,
                    "mapped_rows": mapped_rows,
                    "unmapped_rows": input_rows - mapped_rows,
                }
    temp_output.replace(output_path)
    manifest = {
        "source_build": "GRCh38",
        "target_build": "hg19",
        "chain_path": str(chain_path),
        "cells": list(cells),
        "output": str(output_path),
        "counts": counts,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest

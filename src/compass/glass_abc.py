"""Preparation helpers for Glass Lab brain-cell ABC predictions."""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


GLASS_ABC_V2_CELLS = ("astrocyte", "microglia", "neuron", "oligodendrocyte")
_ABC_FILENAME = "EnhancerPredictionsAllPutative.ForVariantOverlap.shrunk150bp.tsv.gz"
_REQUIRED_COLUMNS = ["chr", "start", "end", "TargetGene", "ABC.Score", "CellType", "class", "isSelfPromoter"]


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
                    frame[_REQUIRED_COLUMNS].to_csv(destination, sep="\t", index=False, header=False)
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


def prepare_glass_expressed_tss(
    source_root: str | Path,
    output_path: str | Path,
    chain_path: str | Path,
    liftover: str = "liftOver",
    cells: tuple[str, ...] = GLASS_ABC_V2_CELLS,
) -> dict[str, object]:
    """Lift cell-specific expressed-gene TSS positions from GRCh38 to hg19."""

    output_path = Path(output_path)
    chain_path = Path(chain_path)
    if not chain_path.is_file():
        raise FileNotFoundError(f"LiftOver chain file not found: {chain_path}")
    executable = shutil.which(liftover)
    if executable is None:
        raise FileNotFoundError(f"Could not find liftOver executable: {liftover}")

    frames: list[pd.DataFrame] = []
    for cell, source_path in glass_abc_v2_paths(source_root, cells).items():
        frame = pd.read_csv(
            source_path,
            sep="\t",
            usecols=["chr", "TargetGene", "TargetGeneTSS", "TargetGeneIsExpressed"],
        ).drop_duplicates()
        frame = frame[frame["TargetGeneIsExpressed"].astype(str).str.lower().eq("true")].copy()
        frame["CellType"] = cell
        frames.append(frame)
    tss = pd.concat(frames, ignore_index=True).rename(
        columns={"chr": "chrom", "TargetGene": "gene", "TargetGeneTSS": "tss"}
    )
    tss["tss"] = pd.to_numeric(tss["tss"], errors="coerce")
    tss = tss.dropna(subset=["chrom", "gene", "tss"]).copy()
    tss["chrom"] = tss["chrom"].astype(str)
    tss["tss"] = tss["tss"].astype(int)
    tss = tss[tss["tss"].gt(0)].drop_duplicates(["chrom", "tss", "gene", "CellType"]).reset_index(drop=True)
    tss["record_id"] = np.arange(tss.shape[0], dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="glass_tss_liftover_", dir=output_path.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_bed = temp_dir / "expressed_tss.grch38.bed"
        mapped_bed = temp_dir / "expressed_tss.hg19.bed"
        unmapped_bed = temp_dir / "expressed_tss.unmapped.bed"
        bed = pd.DataFrame(
            {
                0: tss["chrom"],
                1: tss["tss"] - 1,
                2: tss["tss"],
                3: tss["record_id"],
            }
        )
        bed.to_csv(source_bed, sep="\t", index=False, header=False)
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
        mapped = pd.read_csv(mapped_bed, sep="\t", header=None, names=["chrom", "start", "end", "record_id"])
    mapped["record_id"] = pd.to_numeric(mapped["record_id"], errors="raise").astype(np.int64)
    result = mapped.merge(tss[["record_id", "gene", "CellType"]], on="record_id", how="inner", validate="one_to_one")
    result["tss"] = result["end"].astype(np.int64)
    result = result[["chrom", "tss", "gene", "CellType"]].sort_values(["chrom", "tss", "gene", "CellType"])
    result.to_csv(output_path, sep="\t", index=False, compression="gzip")
    counts = result.groupby("CellType").size().to_dict()
    manifest = {
        "source_build": "GRCh38",
        "target_build": "hg19",
        "chain_path": str(chain_path),
        "cells": list(cells),
        "output": str(output_path),
        "mapped_expressed_tss": {str(key): int(value) for key, value in counts.items()},
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest

#!/usr/bin/env python
"""Create an uncompressed LD cache manifest without replacing existing archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"


def _find_cache_prefix(cache_dir: Path) -> Path:
    manifests = sorted(
        cache_dir.glob("abc.allrows*.gwas.R2.chroms/manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise FileNotFoundError(f"No chromosome LD cache manifests found under {cache_dir}")
    r2_dir = manifests[0].parent
    return cache_dir / r2_dir.name.removesuffix(".gwas.R2.chroms")


def _write_uncompressed_matrix(source: Path, target: Path) -> None:
    with np.load(source, allow_pickle=False) as arrays:
        np.savez(
            target,
            data=arrays["data"],
            indices=arrays["indices"],
            indptr=arrays["indptr"],
            shape=arrays["shape"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create uncompressed copies of cached chromosome LD matrices.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cache-prefix", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    cache_prefix = Path(args.cache_prefix).expanduser() if args.cache_prefix else _find_cache_prefix(data_root / "cache")
    r2_dir = Path(str(cache_prefix) + ".gwas.R2.chroms")
    source_manifest = r2_dir / "manifest.json"
    target_manifest = r2_dir / "manifest.uncompressed.json"
    if target_manifest.exists():
        raise FileExistsError(f"Uncompressed manifest already exists: {target_manifest}")

    with open(source_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    target_blocks = []
    start = perf_counter()
    for item in manifest["blocks"]:
        source = r2_dir / item["matrix"]
        target_name = f"{source.stem}.uncompressed.npz"
        target = r2_dir / target_name
        if not target.exists():
            _write_uncompressed_matrix(source, target)
        target_blocks.append({"chrom": item["chrom"], "matrix": target_name, "rows": item["rows"]})
        print(f"[cache] wrote chromosome {item['chrom']}", flush=True)

    with open(target_manifest, "x", encoding="utf-8") as handle:
        json.dump(
            {"representation": "chromosome", "storage": "npz_uncompressed", "blocks": target_blocks},
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"[cache] wrote {target_manifest} in {perf_counter() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()

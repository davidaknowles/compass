#!/usr/bin/env python
"""Summarize LD connected components across r-squared thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numba
import numpy as np
import pandas as pd
from plotnine import aes, geom_line, geom_point, ggplot, labs, scale_x_log10, theme_bw


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"
DEFAULT_THRESHOLDS = "0.01,0.015,0.02,0.03,0.05,0.075,0.1"


def _parse_float_list(value: str) -> np.ndarray:
    values = np.asarray(sorted({float(item) for item in value.split(",") if item.strip()}), dtype=np.float32)
    if values.size == 0 or np.any(values <= 0) or np.any(values > 1):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated values in (0, 1]")
    return values


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values or any(value < 1 or value > 22 for value in values):
        raise argparse.ArgumentTypeError("chromosomes must be comma-separated autosomes")
    return values


def _find_cache_prefix(cache_dir: Path) -> Path:
    manifests = sorted(cache_dir.glob("abc.allrows*.gwas.R2.chroms/manifest.json"), key=lambda path: path.stat().st_mtime)
    if not manifests:
        raise FileNotFoundError(f"No chromosome LD cache manifest found under {cache_dir}")
    r2_dir = manifests[-1].parent
    return cache_dir / r2_dir.name.removesuffix(".gwas.R2.chroms")


@numba.njit(cache=True)
def _find(parent: np.ndarray, index: int) -> int:
    root = index
    while parent[root] != root:
        root = parent[root]
    while parent[index] != index:
        next_index = parent[index]
        parent[index] = root
        index = next_index
    return root


@numba.njit(cache=True)
def _union(parent: np.ndarray, sizes: np.ndarray, left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return
    if sizes[left_root] < sizes[right_root]:
        left_root, right_root = right_root, left_root
    parent[right_root] = left_root
    sizes[left_root] += sizes[right_root]


@numba.njit(cache=True)
def _union_thresholds(
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    thresholds: np.ndarray,
    parents: np.ndarray,
    sizes: np.ndarray,
    edge_counts: np.ndarray,
) -> None:
    """Union each undirected cached edge into every threshold graph it satisfies."""

    n_thresholds = thresholds.shape[0]
    n_rows = indptr.shape[0] - 1
    for row in range(n_rows):
        for offset in range(indptr[row], indptr[row + 1]):
            column = indices[offset]
            if column <= row:
                continue
            value = values[offset]
            for threshold_index in range(n_thresholds):
                if value < thresholds[threshold_index]:
                    break
                edge_counts[threshold_index] += 1
                _union(parents[threshold_index], sizes[threshold_index], row, column)


@numba.njit(cache=True)
def _compress(parent: np.ndarray) -> None:
    for index in range(parent.shape[0]):
        parent[index] = _find(parent, index)


def _component_sizes(parent: np.ndarray) -> np.ndarray:
    _compress(parent)
    counts = np.bincount(parent, minlength=parent.shape[0])
    return counts[counts > 0].astype(np.int32, copy=False)


def _size_summary(component_sizes: np.ndarray) -> dict[str, float | int]:
    return {
        "components": int(component_sizes.size),
        "singletons": int(np.count_nonzero(component_sizes == 1)),
        "largest_component": int(component_sizes.max(initial=0)),
        "component_size_median": float(np.quantile(component_sizes, 0.5)),
        "component_size_p90": float(np.quantile(component_sizes, 0.9)),
        "component_size_p99": float(np.quantile(component_sizes, 0.99)),
    }


def _load_manifest(r2_dir: Path) -> dict:
    for name in ("manifest.uncompressed.json", "manifest.json"):
        path = r2_dir / name
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError(f"No manifest found in {r2_dir}")


def _histogram_rows(threshold: float, component_sizes: np.ndarray) -> list[dict]:
    exponents = np.floor(np.log2(component_sizes)).astype(np.int16)
    values, counts = np.unique(exponents, return_counts=True)
    rows = []
    for exponent, count in zip(values, counts):
        low = 1 << int(exponent)
        high = (1 << (int(exponent) + 1)) - 1
        rows.append(
            {
                "r2_threshold": threshold,
                "size_bin": str(low) if low == high else f"{low}-{high}",
                "size_bin_lower": low,
                "components": int(count),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute chromosome-blocked LD connected components.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cache-prefix", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, type=_parse_float_list)
    parser.add_argument("--chromosomes", default=None, type=_parse_int_list)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    thresholds = args.thresholds
    data_root = Path(args.data_root).expanduser()
    cache_prefix = Path(args.cache_prefix).expanduser() if args.cache_prefix else _find_cache_prefix(data_root / "cache")
    r2_dir = Path(f"{cache_prefix}.gwas.R2.chroms")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_root / "results" / "ld_component_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(r2_dir)
    entries = [entry for entry in manifest["blocks"] if args.chromosomes is None or int(entry["chrom"]) in args.chromosomes]
    if not entries:
        raise ValueError("No chromosome LD blocks selected")

    global_sizes: dict[float, list[np.ndarray]] = {float(threshold): [] for threshold in thresholds}
    total_variants = np.zeros(thresholds.size, dtype=np.int64)
    total_edges = np.zeros(thresholds.size, dtype=np.int64)
    cached_edges = 0
    per_chrom_rows: list[dict] = []
    start = perf_counter()

    for entry_index, entry in enumerate(entries, start=1):
        chrom = int(entry["chrom"])
        chrom_start = perf_counter()
        with np.load(r2_dir / entry["matrix"], allow_pickle=False) as arrays:
            indptr = arrays["indptr"].astype(np.int64, copy=False)
            indices = arrays["indices"].astype(np.int64, copy=False)
            values = arrays["data"].astype(np.float32, copy=False)
            n_rows = int(arrays["shape"][0])

        parents = np.tile(np.arange(n_rows, dtype=np.int32), (thresholds.size, 1))
        sizes = np.ones((thresholds.size, n_rows), dtype=np.int32)
        edge_counts = np.zeros(thresholds.size, dtype=np.int64)
        _union_thresholds(indptr, indices, values, thresholds, parents, sizes, edge_counts)
        chrom_cached_edges = int(np.count_nonzero(indices > np.repeat(np.arange(n_rows), np.diff(indptr))))
        cached_edges += chrom_cached_edges

        for threshold_index, threshold in enumerate(thresholds):
            component_sizes = _component_sizes(parents[threshold_index])
            summary = _size_summary(component_sizes)
            summary.update(
                {
                    "chrom": chrom,
                    "r2_threshold": float(threshold),
                    "variants": n_rows,
                    "undirected_edges": int(edge_counts[threshold_index]),
                    "edge_fraction_of_cached": edge_counts[threshold_index] / max(chrom_cached_edges, 1),
                    "singleton_fraction": summary["singletons"] / max(n_rows, 1),
                    "largest_component_fraction": summary["largest_component"] / max(n_rows, 1),
                    "elapsed_seconds": perf_counter() - chrom_start,
                }
            )
            per_chrom_rows.append(summary)
            global_sizes[float(threshold)].append(component_sizes)
            total_variants[threshold_index] += n_rows
            total_edges[threshold_index] += edge_counts[threshold_index]

        print(
            f"[components] chromosome {chrom} ({entry_index}/{len(entries)}) rows={n_rows} "
            f"elapsed={perf_counter() - chrom_start:.1f}s",
            flush=True,
        )

    per_chrom = pd.DataFrame(per_chrom_rows)
    global_rows: list[dict] = []
    histogram_rows: list[dict] = []
    for threshold_index, threshold in enumerate(thresholds):
        component_sizes = np.concatenate(global_sizes[float(threshold)])
        summary = _size_summary(component_sizes)
        summary.update(
            {
                "r2_threshold": float(threshold),
                "variants": int(total_variants[threshold_index]),
                "undirected_edges": int(total_edges[threshold_index]),
                "edge_fraction_of_cached": total_edges[threshold_index] / max(cached_edges, 1),
                "singleton_fraction": summary["singletons"] / max(total_variants[threshold_index], 1),
                "largest_component_fraction": summary["largest_component"] / max(total_variants[threshold_index], 1),
                "elapsed_seconds": perf_counter() - start,
            }
        )
        global_rows.append(summary)
        histogram_rows.extend(_histogram_rows(float(threshold), component_sizes))

    global_summary = pd.DataFrame(global_rows)
    histograms = pd.DataFrame(histogram_rows)
    per_chrom.to_csv(out_dir / "components_by_chromosome.tsv", sep="\t", index=False)
    global_summary.to_csv(out_dir / "components_genomewide.tsv", sep="\t", index=False)
    histograms.to_csv(out_dir / "component_size_histogram.tsv", sep="\t", index=False)

    if not args.no_plots:
        plot = (
            ggplot(global_summary, aes("r2_threshold", "components"))
            + geom_line()
            + geom_point()
            + scale_x_log10()
            + labs(x="LD threshold (r-squared)", y="Genome-wide components")
            + theme_bw()
        )
        plot.save(out_dir / "components_by_threshold.png", width=6, height=4, dpi=180)
        plot = (
            ggplot(global_summary, aes("r2_threshold", "largest_component_fraction"))
            + geom_line()
            + geom_point()
            + scale_x_log10()
            + labs(x="LD threshold (r-squared)", y="Largest component fraction")
            + theme_bw()
        )
        plot.save(out_dir / "largest_component_fraction.png", width=6, height=4, dpi=180)

    print(f"[components] complete elapsed={perf_counter() - start:.1f}s out={out_dir}", flush=True)


if __name__ == "__main__":
    main()

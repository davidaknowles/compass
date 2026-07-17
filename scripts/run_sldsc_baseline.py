#!/usr/bin/env python
"""Run the infinitesimal gene-aggregated ABC S-LDSC baseline on AD and simulations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compass.simulation import context_variant_scores
from compass.sldsc import fit_sldsc_panel
from run import _cache_key, _cache_paths, _dataset_cache_exists, _load_dataset_cache


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"
DEFAULT_CONTEXTS = [
    "bipolar_neuron_from_iPSC-ENCODE",
    "CD14-positive_monocyte-ENCODE",
    "white_adipose-Loft2014",
    "gastrocnemius_medialis-ENCODE",
    "uterus-ENCODE",
]


def _cache_paths_for_real_data(data_root: Path, cell_types: str):
    gwas_path = data_root / "raw" / "ad_gwas_2026" / "GCST90704647.hg19.tsv.gz"
    args = SimpleNamespace(
        annotation_source="abc",
        no_intercept=False,
        n_samples=None,
        abc_cell_types=cell_types,
        abc_score_column="ABC.Score",
        abc_min_score=0.015,
        ld_r2_cutoff=0.01,
    )
    return _cache_paths(data_root / "cache", _cache_key(args, gwas_path))


def _prepare_dataset(data_root: Path, cell_types: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/run.py",
            "--data-root",
            str(data_root),
            "--abc-cell-types",
            cell_types,
            "--setup-only",
        ],
        check=True,
    )


def _write(path: Path, value: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--simulation-dir", default=None)
    parser.add_argument("--max-seed", type=int, default=2)
    parser.add_argument("--contexts", default=",".join(DEFAULT_CONTEXTS))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    contexts = [value.strip() for value in args.contexts.split(",") if value.strip()]
    if len(contexts) != 5:
        parser.error("the baseline is configured for the five-context recovery panel")
    data_root = Path(args.data_root).expanduser()
    results_dir = Path(args.results_dir).expanduser() if args.results_dir else data_root / "results" / "abc_recovery_simulation"
    results_dir.mkdir(parents=True, exist_ok=True)
    cell_types = ",".join(contexts)
    paths = _cache_paths_for_real_data(data_root, cell_types)
    if not _dataset_cache_exists(paths):
        _prepare_dataset(data_root, cell_types)
    dataset, genes, mechanisms, _, _ = _load_dataset_cache(paths)
    if args.prepare_only:
        print(f"prepared {paths['A']}")
        return
    indices = [mechanisms.index(context) for context in contexts]
    scores = context_variant_scores(dataset.A, genes.shape[0], len(mechanisms), indices)
    real = fit_sldsc_panel(dataset.ld_blocks, scores, contexts, dataset.chisq, dataset.n_samples)
    _write(results_dir / "sldsc_real_ad.json", {"contexts": contexts, "model": real})

    if args.simulation_dir is None:
        return
    simulation_dir = Path(args.simulation_dir).expanduser()
    summary_rows = []
    for metadata_path in sorted(simulation_dir.glob("n*-seed*.metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if int(metadata["seed"]) > args.max_seed:
            continue
        truth_path = simulation_dir / f"{metadata['run_id']}.truth.npz"
        if not truth_path.exists():
            continue
        arrays = np.load(truth_path, allow_pickle=False)
        model = fit_sldsc_panel(dataset.ld_blocks, scores, contexts, arrays["chisq"], float(metadata["n_eff"]))
        _write(simulation_dir / f"{metadata['run_id']}.sldsc.json", {"contexts": contexts, "model": model})
        joint = model["joint"]
        row = {"run_id": metadata["run_id"], "seed": metadata["seed"], "n_eff": metadata["n_eff"]}
        for context, value in zip(joint["annotation_names"], joint["implied_h2"]):
            row[f"joint_implied_h2_{context}"] = value
        summary_rows.append(row)
    if summary_rows:
        import pandas as pd

        pd.DataFrame(summary_rows).sort_values(["n_eff", "seed"]).to_csv(
            simulation_dir / "sldsc_simulation_summary.tsv", sep="\t", index=False
        )


if __name__ == "__main__":
    main()

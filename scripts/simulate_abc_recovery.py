#!/usr/bin/env python
"""Run one ABC/UKBB-LD COMPASS recovery simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from compass.ld import make_ld_component_cv_groups
from compass.model import CompassDataset, fit_nuclear_norm_path, fit_rank1_path
from compass.simulation import (
    context_variant_scores,
    generate_sparse_context_truth,
    ld_propagate,
    normalized_context_mass,
    pearson_correlation,
    simulate_noncentral_chisq,
)

from run import _cache_key, _cache_paths, _dataset_cache_exists, _load_dataset_cache


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"
SIMULATION_CONTEXTS = [
    "bipolar_neuron_from_iPSC-ENCODE",
    "CD14-positive_monocyte-ENCODE",
    "white_adipose-Loft2014",
    "gastrocnemius_medialis-ENCODE",
    "uterus-ENCODE",
]
DEFAULT_CONTEXT_H2 = [0.14, 0.06, 0.0, 0.0, 0.0]
DEFAULT_LAMBDAS = "1e3,3e2,1e2,3e1,1e1,3,1,3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4"


def _parse_csv(value: str, cast):
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _dataset_paths(data_root: Path, cell_types: str):
    gwas_path = data_root / "raw" / "ad_gwas_2026" / "GCST90704647.hg19.tsv.gz"
    args = SimpleNamespace(
        annotation_source="abc",
        no_intercept=False,
        # The cached annotation/LD dataset is independent of simulation sample
        # size; simulations replace n_samples after loading.
        n_samples=100_000.0,
        abc_cell_types=cell_types,
        abc_score_column="ABC.Score",
        abc_min_score=0.015,
        ld_r2_cutoff=0.01,
    )
    return _cache_paths(data_root / "cache", _cache_key(args, gwas_path))


def _prepare_dataset(data_root: Path, cell_types: str) -> None:
    command = [
        sys.executable,
        "scripts/run.py",
        "--data-root",
        str(data_root),
        "--abc-cell-types",
        cell_types,
        "--n-samples",
        "100000",
        "--setup-only",
    ]
    subprocess.run(command, check=True)


def _cv_cache_path(results_dir: Path) -> Path:
    return results_dir / "dataset_cv_groups.npz"


def _load_or_make_cv_groups(dataset, n_mechanisms: int, results_dir: Path, n_folds: int, threshold: float):
    path = _cv_cache_path(results_dir)
    if path.exists():
        arrays = np.load(path, allow_pickle=False)
        return arrays["cv_groups"], arrays["cv_score_groups"], json.loads(str(arrays["metadata"]))
    groups, score_groups, metadata = make_ld_component_cv_groups(
        dataset.ld_blocks,
        dataset.A,
        n_mechanisms,
        n_folds=n_folds,
        r2_threshold=threshold,
    )
    np.savez_compressed(
        path,
        cv_groups=groups,
        cv_score_groups=score_groups,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return groups, score_groups, metadata


def _write_result(prefix: Path, truth, chisq, noncentrality, nuclear, rank1, summary: dict, mechanisms: list[str]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        f"{prefix}.truth.npz",
        direct_h2=truth.direct_h2,
        context_h2=truth.context_h2,
        selected_counts=truth.selected_counts,
        selected_variant_counts=truth.selected_variant_counts,
        chisq=chisq,
        noncentrality=noncentrality,
    )
    np.savez_compressed(
        f"{prefix}.nuclear.npz",
        B=nuclear.B,
        tau=np.asarray(nuclear.tau, dtype=np.float32),
        losses=np.asarray(nuclear.losses, dtype=np.float32),
        lambdas=np.asarray(nuclear.lambdas, dtype=np.float32),
        best_lambda=np.asarray(nuclear.best_lambda, dtype=np.float32),
    )
    np.savez_compressed(
        f"{prefix}.rank1.npz",
        B=rank1.B,
        tau=np.asarray(rank1.tau, dtype=np.float32),
        losses=np.asarray(rank1.losses, dtype=np.float32),
        lambdas=np.asarray(rank1.lambdas, dtype=np.float32),
        best_lambda=np.asarray(rank1.best_lambda, dtype=np.float32),
    )
    pd.DataFrame(nuclear.B, columns=mechanisms).to_csv(f"{prefix}.nuclear.B.tsv", sep="\t", index=False)
    pd.DataFrame(rank1.B, columns=mechanisms).to_csv(f"{prefix}.rank1.B.tsv", sep="\t", index=False)
    with open(f"{prefix}.metadata.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--n-eff", type=float, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contexts", default=",".join(SIMULATION_CONTEXTS))
    parser.add_argument("--context-h2", default=",".join(str(value) for value in DEFAULT_CONTEXT_H2))
    parser.add_argument("--causal-fraction", type=float, default=0.1)
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--cv-r2-threshold", type=float, default=0.01)
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--max-lambda-extensions", type=int, default=4)
    parser.add_argument("--lambda-extension-factor", type=float, default=3.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-8)
    parser.add_argument("--rank1-lr", type=float, default=1e-3)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--ld-chunk-nnz", type=int, default=150_000_000)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    contexts = _parse_csv(args.contexts, str)
    context_h2 = np.asarray(_parse_csv(args.context_h2, float), dtype=np.float32)
    if len(contexts) != 5 or context_h2.size != 5:
        parser.error("the recovery experiment requires exactly five contexts and five context-h2 values")
    if not np.isclose(context_h2.sum(), 0.20):
        parser.error("context-h2 must sum to the requested total heritability of 0.20")
    if np.count_nonzero(context_h2) != 2 or not np.isclose(context_h2[0], 0.14) or not np.isclose(context_h2[1], 0.06):
        parser.error("the recovery experiment requires 0.14,0.06,0,0,0 context heritability")

    data_root = Path(args.data_root).expanduser()
    results_dir = Path(args.results_dir).expanduser() if args.results_dir else data_root / "results" / "abc_recovery_simulation"
    results_dir.mkdir(parents=True, exist_ok=True)
    cell_types = ",".join(contexts)
    paths = _dataset_paths(data_root, cell_types)
    if not _dataset_cache_exists(paths):
        _prepare_dataset(data_root, cell_types)
    dataset, genes, mechanisms, _, _ = _load_dataset_cache(paths)
    if args.prepare_only:
        _load_or_make_cv_groups(dataset, len(mechanisms), results_dir, args.cv_folds, args.cv_r2_threshold)
        print(f"prepared {paths['A']}")
        return

    missing = [context for context in contexts if context not in mechanisms]
    if missing:
        raise ValueError(f"requested contexts are absent from the ABC cache: {missing}")
    mechanism_indices = [mechanisms.index(context) for context in contexts]
    cv_groups, cv_score_groups, cv_metadata = _load_or_make_cv_groups(
        dataset, len(mechanisms), results_dir, args.cv_folds, args.cv_r2_threshold
    )
    scores = context_variant_scores(dataset.A, genes.shape[0], len(mechanisms), mechanism_indices)
    truth = generate_sparse_context_truth(scores, context_h2, args.causal_fraction, args.seed)
    ld_truth = ld_propagate(dataset.ld_blocks, truth.direct_h2)
    chisq, noncentrality = simulate_noncentral_chisq(ld_truth, args.n_eff, args.seed + 10_000_000)
    simulated = CompassDataset(
        A=dataset.A,
        chisq=chisq,
        chrom=dataset.chrom,
        n_samples=float(args.n_eff),
        ld_blocks=dataset.ld_blocks,
        cv_groups=cv_groups,
        cv_score_groups=cv_score_groups,
    )
    lambdas = _parse_csv(args.lambdas, float)
    nuclear = fit_nuclear_norm_path(
        simulated,
        n_genes=genes.shape[0],
        n_mechanisms=len(mechanisms),
        lambdas=lambdas,
        max_lambda_extensions=args.max_lambda_extensions,
        lambda_extension_factor=args.lambda_extension_factor,
        cv=True,
        lr=args.lr,
        max_iter=args.max_iter,
        tol=args.tol,
        device=args.device,
        model_dtype="float32",
        ld_chunk_nnz=args.ld_chunk_nnz,
        progress_every=10,
    )
    rank1 = fit_rank1_path(
        simulated,
        n_genes=genes.shape[0],
        n_mechanisms=len(mechanisms),
        lambdas=[nuclear.best_lambda],
        cv=False,
        initial_B=nuclear.B,
        lr=args.rank1_lr,
        max_iter=args.max_iter,
        tol=args.tol,
        device=args.device,
        model_dtype="float32",
        ld_chunk_nnz=args.ld_chunk_nnz,
        progress_every=10,
    )
    annotation_prediction = dataset.A @ nuclear.B.reshape(-1)
    ld_prediction = ld_propagate(dataset.ld_blocks, annotation_prediction)
    truth_mass = context_h2 / context_h2.sum()
    nuclear_mass = normalized_context_mass(nuclear.B, mechanism_indices)
    rank1_mass = normalized_context_mass(rank1.B, mechanism_indices)
    top_two = np.argsort(nuclear_mass)[-2:]
    run_id = f"n{int(args.n_eff)}-seed{args.seed:02d}"
    summary = {
        "run_id": run_id,
        "seed": args.seed,
        "n_eff": args.n_eff,
        "contexts": contexts,
        "true_context_h2": context_h2.tolist(),
        "true_context_mass": truth_mass.tolist(),
        "causal_fraction": args.causal_fraction,
        "direct_h2_total": float(truth.direct_h2.sum()),
        "selected_counts": truth.selected_counts.tolist(),
        "nuclear_best_lambda": nuclear.best_lambda,
        "nuclear_cv_scores": nuclear.cv_scores,
        "nuclear_tau": nuclear.tau,
        "nuclear_context_mass": nuclear_mass.tolist(),
        "rank1_tau": rank1.tau,
        "rank1_context_mass": rank1_mass.tolist(),
        "nuclear_top_two_exact": bool(set(top_two.tolist()) == {0, 1}),
        "nuclear_causal_control_gap": float(nuclear_mass[:2].min() - nuclear_mass[2:].max()),
        "nuclear_7030_l1_error": float(np.abs(nuclear_mass[:2] - truth_mass[:2]).sum()),
        "ld_signal_correlation": pearson_correlation(ld_truth, ld_prediction),
        "cv_metadata": cv_metadata,
    }
    _write_result(results_dir / run_id, truth, chisq, noncentrality, nuclear, rank1, summary, mechanisms)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

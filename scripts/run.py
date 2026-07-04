#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from compass.data import load_gwas_sumstats, load_top_assoc_annotations, make_training_table
from compass.ld import annotation_triples_to_csr, build_ukbb_ld_r2
from compass.model import CompassDataset, fit_nuclear_norm_path


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"


def _parse_lambdas(value: str) -> list[float]:
    lambdas = [float(x) for x in value.split(",") if x.strip()]
    if not lambdas:
        raise argparse.ArgumentTypeError("at least one lambda is required")
    return lambdas


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run genome-wide COMPASS with UKBB LD.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--top-assoc-dir", default=None)
    parser.add_argument("--gwas", default=None)
    parser.add_argument("--ld-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--annotation-value", default="z2", choices=["z2", "abs_z", "beta2", "neglog10p"])
    parser.add_argument("--no-intercept", action="store_true")
    parser.add_argument("--n-samples", type=float, default=None)
    parser.add_argument("--lambdas", type=_parse_lambdas, default=_parse_lambdas("1e-2,1e-3,1e-4"))
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--use-precomputed", action="store_true")
    parser.add_argument("--svd-method", default="auto", choices=["auto", "exact", "randomized"])
    parser.add_argument("--svd-rank", type=int, default=None)
    parser.add_argument("--svd-oversamples", type=int, default=5)
    parser.add_argument("--svd-n-iter", type=int, default=2)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    top_assoc_dir = Path(args.top_assoc_dir).expanduser() if args.top_assoc_dir else data_root / "raw" / "zenodo_top_assoc"
    gwas_path = Path(args.gwas).expanduser() if args.gwas else data_root / "raw" / "ad_gwas" / "AD_sumstats_Jansenetal_2019sept.txt.gz"
    ld_dir = Path(args.ld_dir).expanduser() if args.ld_dir else data_root / "raw" / "ukbb_ld"
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ann = load_top_assoc_annotations(
        top_assoc_dir,
        annotation_value=args.annotation_value,
        add_intercept=not args.no_intercept,
    )
    gwas = load_gwas_sumstats(gwas_path)
    training = make_training_table(ann.variants, gwas)
    training = training.drop_duplicates("variant_idx").sort_values("variant_idx").reset_index(drop=True)
    if training.empty:
        raise ValueError("No annotation variants aligned to GWAS summary statistics")

    old_variant_idx = training["variant_idx"].to_numpy(np.int64)
    variants = ann.variants.iloc[old_variant_idx].copy().reset_index(drop=True)
    variants["variant_idx"] = np.arange(variants.shape[0], dtype=np.int64)
    old_to_new = pd.DataFrame(
        {"variant_idx": old_variant_idx, "new_variant_idx": variants["variant_idx"].to_numpy(np.int64)}
    )
    triples = ann.triples.merge(old_to_new, on="variant_idx", how="inner")
    triples = triples.drop(columns=["variant_idx"]).rename(columns={"new_variant_idx": "variant_idx"})

    A = annotation_triples_to_csr(
        triples,
        n_variants=variants.shape[0],
        n_genes=ann.genes.shape[0],
        n_mechanisms=len(ann.mechanisms),
    )
    R2, ld_diagnostics = build_ukbb_ld_r2(variants, str(ld_dir))

    if args.n_samples is not None:
        n_samples: float | np.ndarray = float(args.n_samples)
        n_samples_source = "argument"
    elif "n" in training.columns and training["n"].notna().all():
        n_samples = training["n"].to_numpy(np.float32)
        n_samples_source = "gwas"
    else:
        raise ValueError("Known sample sizes are required: pass --n-samples or use a GWAS file with Nsum/Neff/N")

    dataset = CompassDataset(
        A=A,
        R2=R2,
        chisq=training["chisq"].to_numpy(np.float32),
        chrom=variants["chrom"].to_numpy(np.int64),
        n_samples=n_samples,
    )

    fit = fit_nuclear_norm_path(
        dataset,
        n_genes=ann.genes.shape[0],
        n_mechanisms=len(ann.mechanisms),
        lambdas=args.lambdas,
        cv=args.cv,
        lr=args.lr,
        max_iter=args.max_iter,
        tol=args.tol,
        device=device,
        use_precomputed=args.use_precomputed,
        svd_method=args.svd_method,
        svd_rank=args.svd_rank,
        svd_oversamples=args.svd_oversamples,
        svd_n_iter=args.svd_n_iter,
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"compass-{stamp}"
    prefix = out_dir / run_name

    np.savez_compressed(
        f"{prefix}.npz",
        B=fit.B,
        tau=np.asarray(fit.tau, dtype=np.float32),
        losses=np.asarray(fit.losses, dtype=np.float32),
        lambdas=np.asarray(fit.lambdas, dtype=np.float32),
        best_lambda=np.asarray(fit.best_lambda, dtype=np.float32),
    )
    pd.DataFrame(fit.B, index=ann.genes["gene"], columns=ann.mechanisms).to_csv(f"{prefix}.B.tsv", sep="\t")
    ld_diagnostics.to_csv(f"{prefix}.ld_diagnostics.tsv", sep="\t", index=False)
    metadata = {
        "method": fit.method,
        "best_lambda": fit.best_lambda,
        "cv_scores": fit.cv_scores,
        "metadata": fit.metadata,
        "n_variants": dataset.n_variants,
        "n_genes": ann.genes.shape[0],
        "n_mechanisms": len(ann.mechanisms),
        "n_samples_source": n_samples_source,
        "device": device,
        "use_precomputed": args.use_precomputed,
        "svd_method": args.svd_method,
        "svd_rank": args.svd_rank,
    }
    with open(f"{prefix}.metadata.json", "w", encoding="utf-8") as handle:
        json.dump(_json_safe(metadata), handle, indent=2, sort_keys=True)
    print(f"wrote {prefix}.npz")


if __name__ == "__main__":
    main()

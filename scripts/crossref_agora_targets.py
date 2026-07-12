#!/usr/bin/env python
"""Compare COMPASS gene rankings with Agora nominated AD targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def _strip_version(gene_id: str) -> str:
    return str(gene_id).split(".", 1)[0]


def _load_targets(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as handle:
        items = json.load(handle).get("items", [])
    targets = pd.DataFrame(items)
    required = {"ensembl_gene_id", "hgnc_symbol", "total_nominations"}
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Agora target export is missing fields: {sorted(missing)}")
    return targets[["ensembl_gene_id", "hgnc_symbol", "total_nominations"]].drop_duplicates("ensembl_gene_id")


def _rankings(B: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rankings = {"global": B.sum(axis=1)}
    rankings.update({str(column): B[column] for column in B.columns if column != "intercept"})
    out = {}
    for name, score in rankings.items():
        frame = pd.DataFrame({"gene_id": score.index.astype(str), "score": score.to_numpy(float)})
        frame["ensembl_gene_id"] = frame["gene_id"].map(_strip_version)
        out[name] = frame.sort_values(["score", "ensembl_gene_id"], ascending=[False, True]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-reference COMPASS rankings with Agora nominated AD targets.")
    parser.add_argument("--b-tsv", required=True)
    parser.add_argument("--agora-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", default="10,100,500,1000")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    top_k = sorted({int(value) for value in args.top_k.split(",") if value.strip()})
    if not top_k or any(value < 1 for value in top_k):
        raise ValueError("--top-k must contain positive integers")

    B = pd.read_csv(Path(args.b_tsv).expanduser(), sep="\t", index_col=0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    targets = _load_targets(Path(args.agora_json).expanduser())
    target_ids = set(targets["ensembl_gene_id"].astype(str))
    targets.to_csv(out_dir / "agora_nominated_targets.tsv", sep="\t", index=False)

    summary_rows = []
    for analysis, ranking in _rankings(B).items():
        ranking["agora_nominated"] = ranking["ensembl_gene_id"].isin(target_ids)
        ranking = ranking.merge(targets, on="ensembl_gene_id", how="left")
        ranking.insert(0, "rank", np.arange(1, ranking.shape[0] + 1))
        ranking.to_csv(out_dir / f"{analysis}.agora_ranked_genes.tsv", sep="\t", index=False)

        eligible = ranking["ensembl_gene_id"].isin(target_ids)
        population = int(ranking.shape[0])
        target_total = int(eligible.sum())
        for k in top_k:
            subset = ranking.head(min(k, population))
            hits = int(subset["agora_nominated"].sum())
            draws = int(subset.shape[0])
            summary_rows.append(
                {
                    "analysis": analysis,
                    "top_k": k,
                    "tested_genes": draws,
                    "agora_targets_in_background": target_total,
                    "agora_hits": hits,
                    "expected_hits": draws * target_total / population,
                    "fold_enrichment": (hits / draws) / (target_total / population) if target_total else np.nan,
                    "hypergeom_p_value": float(hypergeom.sf(hits - 1, population, target_total, draws)),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(["analysis", "top_k"])
    summary.to_csv(out_dir / "agora_overlap_summary.tsv", sep="\t", index=False)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "b_tsv": str(Path(args.b_tsv).expanduser()),
                "agora_json": str(Path(args.agora_json).expanduser()),
                "top_k": top_k,
                "n_compass_genes": int(B.shape[0]),
                "n_agora_targets": int(targets.shape[0]),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

from compass.model import gene_deviation_heritability_from_masses


DEFAULT_GTF = Path.home() / "knowles_lab" / "data" / "ADSP_reguloML" / "fasta_files" / "gencode.v38.annotation.gtf"
DEFAULT_SOURCES = ["GO:BP", "REAC", "WP", "KEGG", "CORUM", "HPA"]
GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


def _strip_version(gene_id: str) -> str:
    return str(gene_id).split(".", 1)[0]


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def load_gene_symbols(gtf_path: Path, wanted: set[str]) -> pd.DataFrame:
    if not gtf_path.exists():
        return pd.DataFrame({"ensembl": sorted(wanted), "symbol": sorted(wanted)})
    gene_id_re = re.compile(r'gene_id "([^"]+)"')
    gene_name_re = re.compile(r'gene_name "([^"]+)"')
    rows: list[tuple[str, str]] = []
    with _open_text(gtf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            gene_id_match = gene_id_re.search(fields[8])
            if gene_id_match is None:
                continue
            ensembl = _strip_version(gene_id_match.group(1))
            if ensembl not in wanted:
                continue
            gene_name_match = gene_name_re.search(fields[8])
            symbol = gene_name_match.group(1) if gene_name_match is not None else ensembl
            rows.append((ensembl, symbol))
    if not rows:
        return pd.DataFrame({"ensembl": sorted(wanted), "symbol": sorted(wanted)})
    return pd.DataFrame(rows, columns=["ensembl", "symbol"]).drop_duplicates("ensembl")


def build_rankings(
    B: pd.DataFrame,
    symbols: pd.DataFrame,
    annotation_mass: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    score_matrix = (
        B
        if annotation_mass is None
        else pd.DataFrame(
            gene_deviation_heritability_from_masses(annotation_mass, B.to_numpy()),
            index=B.index,
            columns=B.columns,
        )
    )
    score_frames: dict[str, pd.Series] = {
        "global": score_matrix.sum(axis=1),
    }
    coefficient_frames: dict[str, pd.Series] = {
        "global": B.sum(axis=1),
    }
    for mechanism in score_matrix.columns:
        if mechanism == "intercept":
            continue
        score_frames[str(mechanism)] = score_matrix[mechanism]
        coefficient_frames[str(mechanism)] = B[mechanism]

    symbol_map = dict(zip(symbols["ensembl"], symbols["symbol"]))
    rankings: dict[str, pd.DataFrame] = {}
    for analysis, scores in score_frames.items():
        frame = pd.DataFrame({"gene_id": scores.index.astype(str), "score": scores.to_numpy(dtype=float)})
        frame["coefficient"] = coefficient_frames[analysis].to_numpy(dtype=float)
        frame["ensembl"] = frame["gene_id"].map(_strip_version)
        frame["symbol"] = frame["ensembl"].map(symbol_map).fillna(frame["ensembl"])
        frame = frame.sort_values(["score", "ensembl"], ascending=[False, True]).reset_index(drop=True)
        frame["rank"] = np.arange(1, frame.shape[0] + 1)
        rankings[analysis] = frame[
            ["rank", "gene_id", "ensembl", "symbol", "score", "coefficient"]
        ]
    return rankings


def run_gprofiler(
    ranked_genes: list[str],
    background: list[str],
    sources: list[str],
    timeout: int,
    retries: int,
) -> pd.DataFrame:
    if not ranked_genes:
        return pd.DataFrame()
    payload = {
        "organism": "hsapiens",
        "query": ranked_genes,
        "sources": sources,
        "user_threshold": 1.0,
        "all_results": True,
        "ordered": True,
        "no_evidences": True,
        "domain_scope": "custom",
        "background": background,
    }
    for attempt in range(retries + 1):
        try:
            response = requests.post(GPROFILER_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries:
                raise
            sleep(2**attempt)
    data = response.json()
    result = data.get("result", [])
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ordered g:Profiler enrichment for COMPASS-ranked genes.")
    parser.add_argument("--b-tsv", required=True, help="COMPASS gene-by-mechanism B table")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--annotation-npz",
        default=None,
        help="Sparse annotation matrix; when supplied, rank by fitted contribution rather than coefficient",
    )
    parser.add_argument("--gtf", default=str(DEFAULT_GTF))
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--top-genes", type=int, default=100)
    parser.add_argument(
        "--all-positive",
        action="store_true",
        help="Query every positive-score gene in ranked order; --top-genes still controls summary files",
    )
    parser.add_argument(
        "--cumulative-score-fraction",
        type=float,
        default=None,
        help="Query the smallest leading set reaching this fraction of positive fitted contribution",
    )
    parser.add_argument("--top-terms", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()
    if args.cumulative_score_fraction is not None and not (
        0 < args.cumulative_score_fraction <= 1
    ):
        parser.error("--cumulative-score-fraction must be in (0, 1]")
    if args.all_positive and args.cumulative_score_fraction is not None:
        parser.error("--all-positive and --cumulative-score-fraction are mutually exclusive")

    b_path = Path(args.b_tsv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [source.strip() for source in args.sources.split(",") if source.strip()]

    B = pd.read_csv(b_path, sep="\t", index_col=0)
    B = B.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    annotation_mass = None
    if args.annotation_npz is not None:
        annotation = sp.load_npz(Path(args.annotation_npz).expanduser())
        if annotation.shape[1] != B.size:
            raise ValueError("annotation columns do not match the gene-by-context coefficient table")
        annotation_mass = np.asarray(annotation.sum(axis=0)).ravel().reshape(B.shape)
    background = sorted({_strip_version(gene_id) for gene_id in B.index.astype(str)})
    symbols = load_gene_symbols(Path(args.gtf).expanduser(), set(background))
    rankings = build_rankings(B, symbols, annotation_mass)

    all_terms: list[pd.DataFrame] = []
    summary_rows: list[pd.DataFrame] = []
    query_sizes: dict[str, int] = {}
    for analysis, ranking in rankings.items():
        ranking.to_csv(out_dir / f"{analysis}.ranked_genes.tsv", sep="\t", index=False)
        ranking.head(args.top_genes).to_csv(out_dir / f"{analysis}.top_genes.tsv", sep="\t", index=False)
        positive_query = ranking.loc[ranking["score"] > 0, "ensembl"].drop_duplicates()
        if args.all_positive:
            query = positive_query.tolist()
        elif args.cumulative_score_fraction is not None:
            positive_ranking = ranking.loc[ranking["score"] > 0].drop_duplicates("ensembl")
            cumulative = positive_ranking["score"].cumsum().to_numpy()
            target = args.cumulative_score_fraction * float(positive_ranking["score"].sum())
            query_size = int(np.searchsorted(cumulative, target, side="left") + 1)
            query = positive_ranking["ensembl"].head(query_size).tolist()
        else:
            query = positive_query.head(args.top_genes).tolist()
        query_sizes[analysis] = len(query)
        terms = run_gprofiler(query, background, sources, args.timeout, args.retries)
        if terms.empty:
            terms = pd.DataFrame({"analysis": [analysis], "note": ["no ranked genes or no enrichment results"]})
        else:
            terms.insert(0, "analysis", analysis)
        terms.to_csv(out_dir / f"{analysis}.gprofiler.tsv", sep="\t", index=False)
        all_terms.append(terms)
        if "p_value" in terms.columns:
            top = terms.sort_values("p_value").head(args.top_terms).copy()
            summary_rows.append(top)
        sleep(args.sleep)

    all_gprofiler = pd.concat(all_terms, ignore_index=True, sort=False)
    all_gprofiler.to_csv(out_dir / "all_gprofiler.tsv", sep="\t", index=False)
    if "p_value" in all_gprofiler.columns:
        significant = all_gprofiler.loc[
            pd.to_numeric(all_gprofiler["p_value"], errors="coerce") < 0.05
        ]
        significant.to_csv(out_dir / "significant_terms.tsv", sep="\t", index=False)
    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True, sort=False)
        keep = [
            col
            for col in ["analysis", "source", "native", "name", "p_value", "term_size", "query_size", "intersection_size"]
            if col in summary.columns
        ]
        summary[keep].to_csv(out_dir / "top_terms.tsv", sep="\t", index=False)

    metadata = {
        "b_tsv": str(b_path),
        "annotation_npz": (
            str(Path(args.annotation_npz).expanduser()) if args.annotation_npz is not None else None
        ),
        "ranking_score": "deviation_contribution" if annotation_mass is not None else "coefficient",
        "gtf": str(Path(args.gtf).expanduser()),
        "sources": sources,
        "background_size": len(background),
        "gprofiler_query_size": (
            "all_positive"
            if args.all_positive
            else "cumulative_score_fraction"
            if args.cumulative_score_fraction is not None
            else args.top_genes
        ),
        "cumulative_score_fraction": args.cumulative_score_fraction,
        "query_sizes": query_sizes,
        "analyses": sorted(rankings),
        "gprofiler_url": GPROFILER_URL,
        "multiple_testing_correction": "g_SCS",
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

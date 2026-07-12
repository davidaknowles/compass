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


def build_rankings(B: pd.DataFrame, symbols: pd.DataFrame) -> dict[str, pd.DataFrame]:
    score_frames: dict[str, pd.Series] = {
        "global": B.sum(axis=1),
    }
    for mechanism in B.columns:
        if mechanism == "intercept":
            continue
        score_frames[str(mechanism)] = B[mechanism]

    symbol_map = dict(zip(symbols["ensembl"], symbols["symbol"]))
    rankings: dict[str, pd.DataFrame] = {}
    for analysis, scores in score_frames.items():
        frame = pd.DataFrame({"gene_id": scores.index.astype(str), "score": scores.to_numpy(dtype=float)})
        frame["ensembl"] = frame["gene_id"].map(_strip_version)
        frame["symbol"] = frame["ensembl"].map(symbol_map).fillna(frame["ensembl"])
        frame = frame.sort_values(["score", "ensembl"], ascending=[False, True]).reset_index(drop=True)
        frame["rank"] = np.arange(1, frame.shape[0] + 1)
        rankings[analysis] = frame[["rank", "gene_id", "ensembl", "symbol", "score"]]
    return rankings


def run_gprofiler(
    ranked_genes: list[str],
    background: list[str],
    sources: list[str],
    timeout: int,
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
    response = requests.post(GPROFILER_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    result = data.get("result", [])
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ordered g:Profiler enrichment for COMPASS-ranked genes.")
    parser.add_argument("--b-tsv", required=True, help="COMPASS gene-by-mechanism B table")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gtf", default=str(DEFAULT_GTF))
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--top-genes", type=int, default=100)
    parser.add_argument("--top-terms", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    b_path = Path(args.b_tsv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [source.strip() for source in args.sources.split(",") if source.strip()]

    B = pd.read_csv(b_path, sep="\t", index_col=0)
    B = B.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    background = sorted({_strip_version(gene_id) for gene_id in B.index.astype(str)})
    symbols = load_gene_symbols(Path(args.gtf).expanduser(), set(background))
    rankings = build_rankings(B, symbols)

    all_terms: list[pd.DataFrame] = []
    summary_rows: list[pd.DataFrame] = []
    for analysis, ranking in rankings.items():
        ranking.to_csv(out_dir / f"{analysis}.ranked_genes.tsv", sep="\t", index=False)
        ranking.head(args.top_genes).to_csv(out_dir / f"{analysis}.top_genes.tsv", sep="\t", index=False)
        query = ranking.loc[ranking["score"] > 0, "ensembl"].drop_duplicates().head(args.top_genes).tolist()
        terms = run_gprofiler(query, background, sources, args.timeout)
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
        "gtf": str(Path(args.gtf).expanduser()),
        "sources": sources,
        "background_size": len(background),
        "gprofiler_query_size": args.top_genes,
        "analyses": sorted(rankings),
        "gprofiler_url": GPROFILER_URL,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

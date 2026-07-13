#!/usr/bin/env python
"""Aggregate ABC/UKBB-LD recovery simulations and render plotnine diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from plotnine import aes, facet_wrap, geom_boxplot, geom_hline, ggplot, labs, theme_bw


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir).expanduser()
    metadata_paths = sorted(results_dir.glob("n*-seed*.metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No per-seed metadata found in {results_dir}")

    records = [json.loads(path.read_text()) for path in metadata_paths]
    rows = []
    mass_rows = []
    for record in records:
        row = {key: value for key, value in record.items() if not isinstance(value, (list, dict))}
        for index, context in enumerate(record["contexts"]):
            slug = _slug(context)
            row[f"truth_mass_{slug}"] = record["true_context_mass"][index]
            row[f"nuclear_mass_{slug}"] = record["nuclear_context_mass"][index]
            row[f"rank1_mass_{slug}"] = record["rank1_context_mass"][index]
            mass_rows.extend(
                [
                    {
                        "run_id": record["run_id"],
                        "n_eff": record["n_eff"],
                        "context": context,
                        "method": "nuclear",
                        "mass": record["nuclear_context_mass"][index],
                        "truth_mass": record["true_context_mass"][index],
                        "causal": index < 2,
                    },
                    {
                        "run_id": record["run_id"],
                        "n_eff": record["n_eff"],
                        "context": context,
                        "method": "rank1",
                        "mass": record["rank1_context_mass"][index],
                        "truth_mass": record["true_context_mass"][index],
                        "causal": index < 2,
                    },
                ]
            )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["n_eff", "seed"])
    context_mass = pd.DataFrame(mass_rows)
    aggregate = (
        summary.groupby("n_eff", as_index=False)
        .agg(
            simulations=("run_id", "size"),
            top_two_recovery_rate=("nuclear_top_two_exact", "mean"),
            causal_control_gap_mean=("nuclear_causal_control_gap", "mean"),
            causal_control_gap_sd=("nuclear_causal_control_gap", "std"),
            split_l1_error_mean=("nuclear_7030_l1_error", "mean"),
            split_l1_error_sd=("nuclear_7030_l1_error", "std"),
            ld_signal_correlation_mean=("ld_signal_correlation", "mean"),
            ld_signal_correlation_sd=("ld_signal_correlation", "std"),
            tau_mean=("nuclear_tau", "mean"),
            tau_sd=("nuclear_tau", "std"),
        )
        .sort_values("n_eff")
    )
    summary.to_csv(results_dir / "per_simulation_summary.tsv", sep="\t", index=False)
    context_mass.to_csv(results_dir / "context_mass_long.tsv", sep="\t", index=False)
    aggregate.to_csv(results_dir / "aggregate_summary.tsv", sep="\t", index=False)

    figures = results_dir / "figures"
    figures.mkdir(exist_ok=True)
    for method in ("nuclear", "rank1"):
        plot_data = context_mass.query("method == @method")
        plot = (
            ggplot(plot_data, aes("context", "mass", fill="causal"))
            + geom_boxplot()
            + geom_hline(aes(yintercept="truth_mass"), data=plot_data.drop_duplicates(["n_eff", "context"]), linetype="dashed")
            + facet_wrap("~n_eff")
            + labs(x="ABC context", y="Normalized fitted coefficient mass", title=f"{method} context recovery")
            + theme_bw()
        )
        plot.save(figures / f"{method}_context_mass.png", width=13, height=6, dpi=180)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Audit final hierarchical COMPASS fits across traits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from compass.reporting import summarize_hierarchical_run


def _mapping(values: list[str], label: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} values must use NAME=VALUE")
        name, item = value.split("=", 1)
        if not name or not item or name in result:
            raise ValueError(f"invalid or duplicate {label} value: {value}")
        result[name] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="TRAIT=RESULT_PREFIX")
    parser.add_argument("--expected", action="append", required=True, help="TRAIT=CONTEXT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = _mapping(args.run, "run")
    expected = _mapping(args.expected, "expected")
    if runs.keys() != expected.keys():
        parser.error("--run and --expected must name the same traits")

    rows = []
    for trait, raw_prefix in runs.items():
        prefix = Path(raw_prefix).expanduser()
        with Path(f"{prefix}.metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        contributions = pd.read_csv(f"{prefix}.context_contributions.tsv", sep="\t")
        rows.append(
            {
                "trait": trait,
                **summarize_hierarchical_run(metadata, contributions, expected[trait]),
            }
        )
    result = pd.DataFrame(rows)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output.expanduser(), sep="\t", index=False)
    print(result.to_string(index=False))
    if not result["expected_context_is_leading"].all():
        raise SystemExit("At least one fit failed its expected-context ranking gate")


if __name__ == "__main__":
    main()

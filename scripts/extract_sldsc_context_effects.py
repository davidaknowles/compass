#!/usr/bin/env python
"""Extract non-negative global context effects from an official S-LDSC fit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ANNOTATION_MAPS = {
    "glass_binary": {
        "astrocyte": "ABC_glass_astrocyteL2_0",
        "microglia": "ABC_glass_microgliaL2_0",
        "neuron": "ABC_glass_neuronL2_0",
        "oligodendrocyte": "ABC_glass_oligodendrocyteL2_0",
    },
    "h3k27ac": {
        "astrocyte": "H3K27ac_LHX2L2_0",
        "microglia": "H3K27ac_PU1L2_0",
        "neuron": "H3K27ac_NeuNL2_0",
        "oligodendrocyte": "H3K27ac_Olig2L2_0",
    },
    "predicted_eqtl": {
        "astrocyte": "PredEQTL_astrocyteL2_0",
        "excitatory_neuron": "PredEQTL_excitatory_neuronL2_0",
        "inhibitory_neuron": "PredEQTL_inhibitory_neuronL2_0",
        "microglia": "PredEQTL_microgliaL2_0",
        "opc": "PredEQTL_opcL2_0",
        "oligodendrocyte": "PredEQTL_oligodendrocyteL2_0",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("output")
    parser.add_argument("--panel", required=True, choices=sorted(ANNOTATION_MAPS))
    args = parser.parse_args()

    results = pd.read_csv(Path(args.results).expanduser(), sep="\t").set_index("Category")
    rows = [{"context": "intercept", "effect": 0.0, "standard_error": 0.0}]
    for context, category in ANNOTATION_MAPS[args.panel].items():
        if category not in results.index:
            raise ValueError(f"Missing S-LDSC category: {category}")
        source = results.loc[category]
        rows.append(
            {
                "context": context,
                "effect": max(0.0, float(source["Coefficient"])),
                "standard_error": float(source["Coefficient_std_error"]),
            }
        )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

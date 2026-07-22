#!/usr/bin/env python
"""Stream selected public GWAS formats into the COMPASS hg19 schema."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

from download_neuropsychiatric_gwas import SOURCES


SCHEMAS = {
    "pd": {"chrom": "chromosome", "pos": "base_pair_location", "a1": "effect_allele", "a2": "other_allele", "beta": "beta", "se": "standard_error"},
    "bipolar": {"chrom": "CHR", "pos": "BP", "snp": "SNP", "a1": "A1", "a2": "A2", "odds_ratio": "OR", "se": "SE", "n_half": "Neff_half"},
    "mdd": {"chrom": "CHR", "pos": "BP", "snp": "SNP", "a1": "A1", "a2": "A2", "odds_ratio": "OR", "se": "SE", "n": "Neff"},
    "als": {"chrom": "chromosome", "pos": "base_pair_location", "snp": "rsid", "a1": "effect_allele", "a2": "other_allele", "beta": "beta", "se": "standard_error", "n": "N_effective"},
    "anxiety": {"chrom": "CHR", "pos": "BP", "snp": "SNP", "a1": "A1", "a2": "A2", "beta": "BETA", "se": "SE"},
}


def effective_sample_size(trait: str) -> float | None:
    value = SOURCES[trait].get("sample_size")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and "cases" in value and "controls" in value:
        cases = float(value["cases"])
        controls = float(value["controls"])
        return 4.0 * cases * controls / (cases + controls)
    return None


def normalize(trait: str, source_path: Path, output_path: Path) -> dict[str, object]:
    schema = SCHEMAS[trait]
    fallback_n = effective_sample_size(trait)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.partial")
    source_count = output_count = 0
    n_min = math.inf
    n_max = -math.inf
    with gzip.open(source_path, "rt", encoding="utf-8") as source, gzip.open(
        temporary, "wt", encoding="utf-8"
    ) as destination:
        header = source.readline().strip().split()
        index = {name: position for position, name in enumerate(header)}
        missing = {column for column in schema.values() if column not in index}
        if missing:
            raise ValueError(f"{trait}: missing source columns {sorted(missing)}")
        destination.write("CHR\tBP\tSNP\tA1\tA2\tBETA\tSE\tN\n")
        for line in source:
            source_count += 1
            fields = line.split()
            try:
                chrom = int(fields[index[schema["chrom"]]])
                position = int(fields[index[schema["pos"]]])
                if not 1 <= chrom <= 22:
                    continue
                if "beta" in schema:
                    beta = float(fields[index[schema["beta"]]])
                else:
                    odds_ratio = float(fields[index[schema["odds_ratio"]]])
                    if odds_ratio <= 0:
                        continue
                    beta = math.log(odds_ratio)
                se = float(fields[index[schema["se"]]])
                if "n" in schema:
                    n = float(fields[index[schema["n"]]])
                elif "n_half" in schema:
                    n = 2.0 * float(fields[index[schema["n_half"]]])
                elif fallback_n is not None:
                    n = fallback_n
                else:
                    raise ValueError(f"{trait}: no known sample size")
                if not math.isfinite(beta) or not math.isfinite(se) or se <= 0 or not math.isfinite(n) or n <= 0:
                    continue
            except (IndexError, ValueError):
                continue
            snp = fields[index[schema["snp"]]] if "snp" in schema else ""
            a1 = fields[index[schema["a1"]]]
            a2 = fields[index[schema["a2"]]]
            destination.write(f"{chrom}\t{position}\t{snp}\t{a1}\t{a2}\t{beta:.12g}\t{se:.12g}\t{n:.12g}\n")
            output_count += 1
            n_min = min(n_min, n)
            n_max = max(n_max, n)
    if output_count == 0:
        raise ValueError(f"{trait}: no valid autosomal records")
    temporary.replace(output_path)
    manifest = {
        "trait": trait,
        "source": str(source_path),
        "output": str(output_path),
        "genome_build": "GRCh37/hg19",
        "source_records": source_count,
        "output_records": output_count,
        "n_effective_min": n_min,
        "n_effective_max": n_max,
        "sample_size_source": "variant" if "n" in schema or "n_half" in schema else "study",
    }
    output_path.with_suffix("").with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(Path.home() / "knowles_lab" / "data" / "compass"))
    parser.add_argument("traits", nargs="*", choices=sorted(SOURCES), default=None)
    args = parser.parse_args()

    root = Path(args.data_root).expanduser() / "raw" / "neuropsychiatric_gwas"
    for trait in args.traits or sorted(SOURCES):
        source = root / trait / str(SOURCES[trait]["filename"])
        output = root / trait / f"{trait}.hg19.tsv.gz"
        manifest = normalize(trait, source, output)
        print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

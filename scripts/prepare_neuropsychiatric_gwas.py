#!/usr/bin/env python
"""Stream selected public GWAS formats into the COMPASS hg19 schema."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from download_neuropsychiatric_gwas import SOURCES


DEFAULT_CHAIN = Path(
    "/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/liftover/hg38ToHg19.over.chain.gz"
)


SCHEMAS = {
    "pd": {"chrom": "chromosome", "pos": "base_pair_position", "snp": "SNP_ID", "a1": "effect_allele", "a2": "other_allele", "beta": "beta", "se": "standard_error"},
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


def normalize(
    trait: str,
    source_path: Path,
    output_path: Path,
    chain_path: Path = DEFAULT_CHAIN,
    liftover: str = "liftOver",
) -> dict[str, object]:
    schema = SCHEMAS[trait]
    fallback_n = effective_sample_size(trait)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.partial")
    source_count = output_count = 0
    n_min = math.inf
    n_max = -math.inf
    requires_liftover = SOURCES[trait].get("genome_build") == "GRCh38"
    liftover_executable = shutil.which(liftover) if requires_liftover else None
    if requires_liftover and (liftover_executable is None or not chain_path.is_file()):
        raise FileNotFoundError("hg38ToHg19 chain and liftOver executable are required")

    temp_context = tempfile.TemporaryDirectory(prefix=f"{trait}_liftover_", dir=output_path.parent)
    temp_dir = Path(temp_context.name)
    source_bed = temp_dir / "source.hg38.bed"
    mapped_bed = temp_dir / "mapped.hg19.bed"
    unmapped_bed = temp_dir / "unmapped.hg38.bed"
    raw_destination = source_bed.open("w") if requires_liftover else gzip.open(
        temporary, "wt", encoding="utf-8"
    )
    with gzip.open(source_path, "rt", encoding="utf-8") as source, raw_destination as destination:
        header = source.readline().strip().split()
        index = {name: position for position, name in enumerate(header)}
        missing = {column for column in schema.values() if column not in index}
        if missing:
            raise ValueError(f"{trait}: missing source columns {sorted(missing)}")
        if not requires_liftover:
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
            payload = f"{snp}|{a1}|{a2}|{beta:.12g}|{se:.12g}|{n:.12g}"
            if requires_liftover:
                destination.write(f"chr{chrom}\t{position - 1}\t{position}\t{payload}\n")
            else:
                destination.write(f"{chrom}\t{position}\t{snp}\t{a1}\t{a2}\t{beta:.12g}\t{se:.12g}\t{n:.12g}\n")
                output_count += 1
                n_min = min(n_min, n)
                n_max = max(n_max, n)
    unmapped_count = 0
    if requires_liftover:
        subprocess.run(
            [liftover_executable, str(source_bed), str(chain_path), str(mapped_bed), str(unmapped_bed)],
            check=True,
        )
        with mapped_bed.open() as source, gzip.open(temporary, "wt", encoding="utf-8") as destination:
            destination.write("CHR\tBP\tSNP\tA1\tA2\tBETA\tSE\tN\n")
            for line in source:
                chrom, _start, position, payload = line.rstrip("\n").split("\t")
                snp, a1, a2, beta, se, n = payload.split("|")
                destination.write(f"{chrom[3:]}\t{position}\t{snp}\t{a1}\t{a2}\t{beta}\t{se}\t{n}\n")
                n_value = float(n)
                output_count += 1
                n_min = min(n_min, n_value)
                n_max = max(n_max, n_value)
        unmapped_count = sum(1 for line in unmapped_bed.open() if line and not line.startswith("#"))
    temp_context.cleanup()
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
        "genome_build_source": SOURCES[trait].get("genome_build", "GRCh37/hg19"),
        "unmapped_records": unmapped_count,
    }
    output_path.with_suffix("").with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(Path.home() / "knowles_lab" / "data" / "compass"))
    parser.add_argument("--chain", default=str(DEFAULT_CHAIN))
    parser.add_argument("--liftover", default="liftOver")
    parser.add_argument("traits", nargs="*", choices=sorted(SOURCES), default=None)
    args = parser.parse_args()

    root = Path(args.data_root).expanduser() / "raw" / "neuropsychiatric_gwas"
    for trait in args.traits or sorted(SOURCES):
        source = root / trait / str(SOURCES[trait]["filename"])
        output = root / trait / f"{trait}.hg19.tsv.gz"
        manifest = normalize(
            trait, source, output, Path(args.chain).expanduser(), args.liftover
        )
        print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

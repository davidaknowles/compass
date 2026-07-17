#!/usr/bin/env python
"""Normalize the GCST90704647 no-proxy AD GWAS onto hg19."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"
DEFAULT_CHAIN = Path(
    "/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/liftover/hg38ToHg19.over.chain.gz"
)


def _source_to_bed(source_path: Path, bed_path: Path) -> int:
    required = {
        "chromosome",
        "base_pair_location",
        "effect_allele",
        "other_allele",
        "beta",
        "standard_error",
        "rs_id",
        "Neff_total",
        "N_cases",
        "N_controls",
    }
    count = 0
    with gzip.open(source_path, "rt", newline="") as source, bed_path.open("w") as destination:
        reader = csv.DictReader(source, delimiter="\t")
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required GCST90704647 columns: {sorted(missing)}")
        for row in reader:
            chrom = row["chromosome"]
            if not chrom.isdigit() or not 1 <= int(chrom) <= 22:
                continue
            position = int(row["base_pair_location"])
            rsid = row["rs_id"] or row.get("variant_id", "")
            payload = "|".join(
                [
                    rsid,
                    row["effect_allele"],
                    row["other_allele"],
                    row["beta"],
                    row["standard_error"],
                    row["Neff_total"],
                    row["N_cases"],
                    row["N_controls"],
                ]
            )
            destination.write(f"chr{chrom}\t{position - 1}\t{position}\t{payload}\n")
            count += 1
    return count


def _write_normalized(mapped_bed: Path, output_path: Path) -> tuple[int, float, float]:
    count = 0
    n_min = math.inf
    n_max = -math.inf
    with mapped_bed.open() as source, gzip.open(output_path, "wt", newline="") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(["CHR", "BP", "SNP", "A1", "A2", "BETA", "SE", "N", "N_CASES", "N_CONTROLS"])
        for line in source:
            chrom, _start, end, payload = line.rstrip("\n").split("\t")
            rsid, effect, other, beta, se, n_effective, n_cases, n_controls = payload.split("|")
            if not chrom.startswith("chr") or not chrom[3:].isdigit():
                continue
            try:
                n = float(n_effective)
                beta_value = float(beta)
                se_value = float(se)
            except ValueError:
                continue
            if not n > 0 or not math.isfinite(beta_value) or not se_value > 0:
                continue
            writer.writerow([chrom[3:], end, rsid, effect, other, beta, se, n_effective, n_cases, n_controls])
            n_min = min(n_min, n)
            n_max = max(n_max, n)
            count += 1
    if count == 0:
        raise ValueError("No valid GCST90704647 records remained after liftover")
    return count, n_min, n_max


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_DATA_ROOT / "raw" / "ad_gwas_2026" / "GCST90704647.tsv.gz"))
    parser.add_argument("--output", default=str(DEFAULT_DATA_ROOT / "raw" / "ad_gwas_2026" / "GCST90704647.hg19.tsv.gz"))
    parser.add_argument("--chain", default=str(DEFAULT_CHAIN))
    parser.add_argument("--liftover", default="liftOver")
    args = parser.parse_args()

    source_path = Path(args.source).expanduser()
    output_path = Path(args.output).expanduser()
    chain_path = Path(args.chain).expanduser()
    liftover = shutil.which(args.liftover)
    if not source_path.is_file() or not chain_path.is_file() or liftover is None:
        raise FileNotFoundError("GCST90704647, hg38ToHg19 chain, and liftOver executable are required")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ad_2026_liftover_", dir=output_path.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_bed = temp_dir / "source.hg38.bed"
        mapped_bed = temp_dir / "mapped.hg19.bed"
        unmapped_bed = temp_dir / "unmapped.hg38.bed"
        source_count = _source_to_bed(source_path, source_bed)
        subprocess.run([liftover, str(source_bed), str(chain_path), str(mapped_bed), str(unmapped_bed)], check=True)
        output_count, n_min, n_max = _write_normalized(mapped_bed, output_path)
        unmapped_count = sum(1 for line in unmapped_bed.open() if line and not line.startswith("#"))

    manifest = {
        "source": str(source_path),
        "source_accession": "GCST90704647",
        "proxy_samples_excluded": True,
        "genome_build_source": "GRCh38",
        "genome_build_output": "hg19",
        "source_records": source_count,
        "lifted_records": output_count,
        "unmapped_records": unmapped_count,
        "n_effective_min": n_min,
        "n_effective_max": n_max,
    }
    manifest_path = output_path.with_suffix("").with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

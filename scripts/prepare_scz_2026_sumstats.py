#!/usr/bin/env python
"""Normalize the Bigdeli et al. 2026 EUR schizophrenia BCF onto hg19."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"
DEFAULT_CHAIN = Path(
    "/gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/liftover/hg38ToHg19.over.chain.gz"
)


def _bcf_to_bed(bcf_path: Path, bed_path: Path) -> int:
    query = "%CHROM\\t%POS\\t%ID\\t%REF\\t%ALT[\\t%ES\\t%SE\\t%LP\\t%NS\\t%NE]\\n"
    process = subprocess.Popen(
        ["bcftools", "query", "-f", query, str(bcf_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    count = 0
    with bed_path.open("w") as destination:
        for line in process.stdout:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 10:
                raise ValueError(f"Expected 10 BCF fields, got {len(fields)}: {line[:200]!r}")
            chrom, pos, rsid, ref, alt, beta, se, logp, n_total, n_effective = fields
            position = int(pos)
            # Kent liftOver only accepts BED4 without interpreting later
            # fields as BED scores. Keep the association payload in the name.
            payload = "|".join([rsid, ref, alt, beta, se, logp, n_total, n_effective])
            destination.write("\t".join([chrom, str(position - 1), str(position), payload]) + "\n")
            count += 1
    stderr = process.stderr.read() if process.stderr is not None else ""
    if process.wait() != 0:
        raise RuntimeError(f"bcftools query failed: {stderr}")
    return count


def _write_normalized(mapped_bed: Path, output_path: Path) -> tuple[int, list[float]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    sample_sizes: list[float] = []
    with mapped_bed.open() as source, gzip.open(output_path, "wt", newline="") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(["CHR", "BP", "SNP", "A1", "A2", "BETA", "SE", "LOG10P", "N", "N_TOTAL"])
        for line in source:
            chrom, _start, end, payload = line.rstrip("\n").split("\t")
            rsid, ref, alt, beta, se, logp, n_total, n_effective = payload.split("|")
            if not chrom.startswith("chr") or not chrom[3:].isdigit():
                continue
            n = float(n_effective)
            if not n > 0:
                n = float(n_total)
            if not n > 0:
                continue
            writer.writerow([chrom[3:], end, rsid, alt, ref, beta, se, logp, n, n_total])
            sample_sizes.append(n)
            count += 1
    return count, sample_sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf", default=str(DEFAULT_DATA_ROOT / "raw" / "scz_gwas_2026" / "SCZ_EUR_autosomes.bcf"))
    parser.add_argument("--output", default=str(DEFAULT_DATA_ROOT / "raw" / "scz_gwas_2026" / "SCZ_EUR_2026.hg19.tsv.gz"))
    parser.add_argument("--chain", default=str(DEFAULT_CHAIN))
    parser.add_argument("--liftover", default="liftOver")
    args = parser.parse_args()

    bcf_path = Path(args.bcf).expanduser()
    output_path = Path(args.output).expanduser()
    chain_path = Path(args.chain).expanduser()
    liftover = shutil.which(args.liftover)
    if not bcf_path.is_file() or not chain_path.is_file() or liftover is None:
        raise FileNotFoundError("BCF input, hg38ToHg19 chain, and liftOver executable are required")

    with tempfile.TemporaryDirectory(prefix="scz_2026_liftover_", dir=output_path.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_bed = temp_dir / "source.hg38.bed"
        mapped_bed = temp_dir / "mapped.hg19.bed"
        unmapped_bed = temp_dir / "unmapped.hg38.bed"
        source_count = _bcf_to_bed(bcf_path, source_bed)
        subprocess.run([liftover, str(source_bed), str(chain_path), str(mapped_bed), str(unmapped_bed)], check=True)
        output_count, sample_sizes = _write_normalized(mapped_bed, output_path)
        unmapped_count = sum(1 for line in unmapped_bed.open() if line and not line.startswith("#"))

    manifest = {
        "source_bcf": str(bcf_path),
        "source_entity": "syn66321648",
        "genome_build_source": "GRCh38",
        "genome_build_output": "hg19",
        "source_records": source_count,
        "lifted_records": output_count,
        "unmapped_records": unmapped_count,
        "n_effective_min": min(sample_sizes),
        "n_effective_max": max(sample_sizes),
    }
    manifest_path = output_path.with_suffix("").with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

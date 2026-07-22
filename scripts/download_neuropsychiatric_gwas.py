#!/usr/bin/env python
"""Download the selected public European neuropsychiatric GWAS releases."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path


SOURCES = {
    "pd": {
        "filename": "GCST009324.tsv.gz",
        "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST009001-GCST010000/GCST009324/GCST009324.tsv.gz",
        "citation": "Nalls et al. 2019, PMID 31701892",
        "population": "European",
        "sample_size": {"cases": 15056, "controls": 12637, "proxies": 0},
    },
    "bipolar": {
        "filename": "bip2024_eur_no23andMe.gz",
        "url": "https://ndownloader.figshare.com/files/49760772",
        "citation": "O'Connell et al. 2025, DOI 10.1038/s41586-024-08468-9",
        "population": "European",
    },
    "mdd": {
        "filename": "daner_pgc_mdd_no23andMe_eur_hg19_v3.49.24.11.neff.gz",
        "url": "https://ndownloader.figshare.com/files/52371878",
        "citation": "Adams et al. 2025, DOI 10.1016/j.cell.2024.12.002",
        "population": "European",
    },
    "als": {
        "filename": "GCST90027164_buildGRCh37.tsv.gz",
        "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90027001-GCST90028000/GCST90027164/GCST90027164_buildGRCh37.tsv.gz",
        "citation": "van Rheenen et al. 2021, PMID 34873335",
        "population": "European",
        "sample_size": {"cases": 27205, "controls": 110881},
    },
    "anxiety": {
        "filename": "ANX_EUR.txt.gz",
        "url": "https://zenodo.org/api/records/13135834/files/ANX_EUR.txt.gz/content",
        "citation": "Friligkou et al. 2024, DOI 10.1038/s41588-024-01908-2",
        "population": "European",
        "sample_size": 1096458,
    },
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"exists {destination}", flush=True)
        return
    temporary = destination.with_name(f".{destination.name}.partial")
    request = urllib.request.Request(url, headers={"User-Agent": "compass-gwas-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=300) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    temporary.replace(destination)
    print(f"downloaded {destination} ({destination.stat().st_size} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(Path.home() / "knowles_lab" / "data" / "compass"))
    parser.add_argument("traits", nargs="*", choices=sorted(SOURCES), default=None)
    args = parser.parse_args()

    root = Path(args.data_root).expanduser() / "raw" / "neuropsychiatric_gwas"
    selected = args.traits or sorted(SOURCES)
    manifest: dict[str, dict[str, object]] = {}
    for trait in selected:
        metadata = SOURCES[trait]
        destination = root / trait / str(metadata["filename"])
        download(str(metadata["url"]), destination)
        manifest[trait] = {**metadata, "path": str(destination), "bytes": destination.stat().st_size}
    manifest_path = root / "sources.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

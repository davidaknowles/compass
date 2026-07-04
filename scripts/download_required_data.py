#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve


AD_GWAS_URL = (
    "https://vu.data.surf.nl/index.php/s/l7aiRr1UEgdoJfZ/download"
    "?path=%2F&files=AD_sumstats_Jansenetal_2019sept.txt.gz"
)

ZENODO_RECORD = "15860973"
TOP_ASSOC_FILES = [
    "Ast_top_assoc.tsv.gz",
    "End_top_assoc.tsv.gz",
    "Ext_top_assoc.tsv.gz",
    "IN_top_assoc.tsv.gz",
    "MG_top_assoc.tsv.gz",
    "OD_top_assoc.tsv.gz",
    "OPC_top_assoc.tsv.gz",
]
UKBB_LD_PREFIX = "https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD"
REGION_LENGTH = 3_000_000
GRCH37_AUTOSOME_LENGTHS = {
    1: 249_250_621,
    2: 243_199_373,
    3: 198_022_430,
    4: 191_154_276,
    5: 180_915_260,
    6: 171_115_067,
    7: 159_138_663,
    8: 146_364_022,
    9: 141_213_431,
    10: 135_534_747,
    11: 135_006_516,
    12: 133_851_895,
    13: 115_169_878,
    14: 107_349_540,
    15: 102_531_392,
    16: 90_354_753,
    17: 81_195_210,
    18: 78_077_248,
    19: 59_128_983,
    20: 63_025_520,
    21: 48_129_895,
    22: 51_304_566,
}

DEFAULT_DATA_ROOT = Path.home() / "knowles_lab" / "data" / "compass"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists {dest}")
        return
    print(f"download {url} -> {dest}")
    urlretrieve(url, dest)


def download_with_optional_suffix2(url: str, dest: Path, required: bool = True) -> bool:
    try:
        download(url, dest)
        return True
    except HTTPError:
        try:
            download(url + "2", dest)
            return True
        except HTTPError:
            if required:
                raise
            print(f"missing {url}")
            return False


def iter_ukbb_ld_blocks(chrom: int | None = None):
    chroms = [chrom] if chrom is not None else sorted(GRCH37_AUTOSOME_LENGTHS)
    for chr_num in chroms:
        chrom_length = GRCH37_AUTOSOME_LENGTHS[chr_num]
        for region_start in range(1, chrom_length + 1, REGION_LENGTH):
            region_end = region_start + REGION_LENGTH
            stem = f"chr{chr_num}_{region_start}_{region_end}"
            yield chr_num, region_start, region_end, stem


def _run_parallel_downloads(tasks: list[tuple[str, Path]], jobs: int) -> None:
    if not tasks:
        return
    if jobs <= 1:
        for url, dest in tasks:
            download_with_optional_suffix2(url, dest)
        return
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(download_with_optional_suffix2, url, dest) for url, dest in tasks]
        for future in as_completed(futures):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--top-assoc-dir", default=None)
    parser.add_argument("--ad-out", default=None)
    parser.add_argument("--ld-dir", default=None)
    parser.add_argument("--download-top-assoc", action="store_true")
    parser.add_argument("--download-ad", action="store_true")
    parser.add_argument("--download-ld-metadata", action="store_true")
    parser.add_argument("--download-ld-npz", action="store_true")
    parser.add_argument("--chrom", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    top_assoc_dir = Path(args.top_assoc_dir).expanduser() if args.top_assoc_dir else data_root / "raw" / "zenodo_top_assoc"
    ad_out = Path(args.ad_out).expanduser() if args.ad_out else data_root / "raw" / "ad_gwas" / "AD_sumstats_Jansenetal_2019sept.txt.gz"
    ld_dir = Path(args.ld_dir).expanduser() if args.ld_dir else data_root / "raw" / "ukbb_ld"

    if args.download_top_assoc:
        for file_name in TOP_ASSOC_FILES:
            url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{file_name}?download=1"
            download(url, top_assoc_dir / file_name)

    if args.download_ad:
        download(AD_GWAS_URL, ad_out)

    blocks = list(iter_ukbb_ld_blocks(args.chrom))
    print(f"UKBB LD blocks: {len(blocks)}")
    tasks: list[tuple[str, Path]] = []
    for _, _, _, stem in blocks:
        if args.download_ld_metadata:
            tasks.append((f"{UKBB_LD_PREFIX}/{stem}.gz", ld_dir / f"{stem}.gz"))
        if args.download_ld_npz:
            tasks.append((f"{UKBB_LD_PREFIX}/{stem}.npz", ld_dir / f"{stem}.npz"))
    _run_parallel_downloads(tasks, args.jobs)


if __name__ == "__main__":
    main()

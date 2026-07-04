#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve

import pandas as pd


AD_GWAS_URL = (
    "https://vu.data.surf.nl/index.php/s/l7aiRr1UEgdoJfZ/download"
    "?path=%2F&files=AD_sumstats_Jansenetal_2019sept.txt.gz"
)

DEFAULT_DATA_ROOT = Path.home() / "data" / "compass" / "data"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists {dest}")
        return
    print(f"download {url} -> {dest}")
    urlretrieve(url, dest)


def download_with_optional_suffix2(url: str, dest: Path) -> bool:
    try:
        download(url, dest)
        return True
    except HTTPError:
        try:
            download(url + "2", dest)
            return True
        except HTTPError:
            print(f"missing {url}")
            return False


def build_ld_manifest(top_assoc_dir: Path, out: Path) -> pd.DataFrame:
    frames = []
    for file in sorted(top_assoc_dir.glob("*_top_assoc.tsv.gz")):
        df = pd.read_csv(file, sep="\t", usecols=["CHR", "POS"])
        chrom = pd.to_numeric(df["CHR"].astype(str).str.replace("chr", "", regex=False), errors="coerce")
        pos = pd.to_numeric(df["POS"], errors="coerce")
        keep = chrom.between(1, 22) & pos.notna()
        frames.append(pd.DataFrame({"CHR": chrom[keep].astype(int), "POS": pos[keep].astype(int)}))
    variants = pd.concat(frames, ignore_index=True).drop_duplicates()
    variants["region_start"] = ((variants.POS - 1) // 3_000_000) * 3_000_000 + 1
    blocks = variants[["CHR", "region_start"]].drop_duplicates().sort_values(["CHR", "region_start"])
    blocks["region_end"] = blocks["region_start"] + 3_000_000
    prefix = "https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD"
    stem = (
        "chr"
        + blocks.CHR.astype(str)
        + "_"
        + blocks.region_start.astype(str)
        + "_"
        + blocks.region_end.astype(str)
    )
    blocks["npz_url"] = prefix + "/" + stem + ".npz"
    blocks["gz_url"] = prefix + "/" + stem + ".gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(out, sep="\t", index=False)
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--top-assoc-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--ad-out", default=None)
    parser.add_argument("--ld-dir", default=None)
    parser.add_argument("--download-ad", action="store_true")
    parser.add_argument("--download-ld-metadata", action="store_true")
    parser.add_argument("--download-ld-npz", action="store_true")
    parser.add_argument("--chrom", type=int, default=None)
    parser.add_argument("--start", type=int, default=None, help="Only download blocks with region_start >= this")
    parser.add_argument("--end", type=int, default=None, help="Only download blocks with region_start <= this")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    top_assoc_dir = Path(args.top_assoc_dir).expanduser() if args.top_assoc_dir else data_root / "raw" / "zenodo_top_assoc"
    manifest = Path(args.manifest).expanduser() if args.manifest else data_root / "raw" / "required_ukbb_ld_blocks.tsv"
    ad_out = Path(args.ad_out).expanduser() if args.ad_out else data_root / "raw" / "ad_gwas" / "AD_sumstats_Jansenetal_2019sept.txt.gz"
    ld_dir = Path(args.ld_dir).expanduser() if args.ld_dir else data_root / "raw" / "ukbb_ld"

    if args.download_ad:
        download(AD_GWAS_URL, ad_out)

    if manifest.exists():
        blocks = pd.read_csv(manifest, sep="\t")
    else:
        blocks = build_ld_manifest(top_assoc_dir, manifest)
    print(f"LD manifest: {manifest} ({len(blocks)} blocks)")

    if args.chrom is not None:
        blocks = blocks[blocks.CHR == args.chrom]
    if args.start is not None:
        blocks = blocks[blocks.region_start >= args.start]
    if args.end is not None:
        blocks = blocks[blocks.region_start <= args.end]

    for row in blocks.itertuples(index=False):
        stem = f"chr{row.CHR}_{row.region_start}_{row.region_end}"
        if args.download_ld_metadata:
            download_with_optional_suffix2(row.gz_url, ld_dir / f"{stem}.gz")
        if args.download_ld_npz:
            download_with_optional_suffix2(row.npz_url, ld_dir / f"{stem}.npz")


if __name__ == "__main__":
    main()

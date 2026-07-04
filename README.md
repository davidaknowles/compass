# COMPASS prototype

This repository contains a Torch prototype of the model described in `docs/main.tex`:
non-negative gene-by-mechanism mediated heritability regression with a nuclear-norm
relaxation, plus a rank-1 alternating baseline.

## Installation

Use a Python environment with Torch installed:

```bash
git clone git@github.com:davidaknowles/compass.git
cd compass
source <torch-venv>/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

The current code uses `torch`, `numpy`, `pandas`, `scipy`, `matplotlib`, and `seaborn`.

## Data Layout

Data should live outside the repo under:

```bash
~/data/compass/data
```

Current downloaded files:

```text
~/data/compass/data/raw/zenodo_top_assoc/*.tsv.gz
~/data/compass/data/raw/ad_gwas/AD_sumstats_Jansenetal_2019sept.txt.gz
~/data/compass/data/raw/required_ukbb_ld_blocks.tsv
~/data/compass/data/raw/ukbb_ld/
```

The AD GWAS file is the updated Jansen et al. 2019 Alzheimer summary statistics
downloaded from the CNCR/SURF data share. The eQTL top-association files are from
Zenodo record `15860973`.

## Data Sources

The downloader uses these public sources:

- SingleBrain top-association eQTL files: Zenodo record `15860973`, `https://zenodo.org/records/15860973`
- Alzheimer GWAS summary statistics: CNCR/SURF share, `https://vu.data.surf.nl/index.php/s/l7aiRr1UEgdoJfZ/download?path=%2F&files=AD_sumstats_Jansenetal_2019sept.txt.gz`
- UK Biobank LD matrices used by PolyFun: Broad/Alkes S3 prefix, `https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD/`

## Download Commands

Download or refresh the AD GWAS:

```bash
python scripts/download_required_data.py --download-ad
```

Build the UKBB LD block manifest from the Zenodo top-association files:

```bash
python scripts/download_required_data.py
```

Download UKBB LD metadata sidecars:

```bash
python scripts/download_required_data.py --download-ld-metadata
```

Download a targeted UKBB LD matrix block, for example the APOE-region blocks:

```bash
python scripts/download_required_data.py \
  --download-ld-npz --download-ld-metadata \
  --chrom 19 --start 42000001 --end 48000001
```

Do not download all `--download-ld-npz` blocks blindly. The top-association variants
touch 906 UKBB 3 Mb blocks, and individual `.npz` files can exceed 1 GB.

## Usage

Load top eQTL annotations with an intercept mechanism:

```python
from compass.data import load_top_assoc_annotations

ann = load_top_assoc_annotations(
    "~/data/compass/data/raw/zenodo_top_assoc",
    annotation_value="z2",
    add_intercept=True,
)
```

Build sparse annotations and a positional LD fallback:

```python
from compass.ld import annotation_triples_to_csr, build_positional_ld

A = annotation_triples_to_csr(
    ann.triples,
    n_variants=len(ann.variants),
    n_genes=len(ann.genes),
    n_mechanisms=len(ann.mechanisms),
)
R2 = build_positional_ld(ann.variants)
```

Fit a small nuclear-norm path:

```python
from compass.model import CompassDataset, fit_nuclear_norm_path

dataset = CompassDataset(
    A=A,
    R2=R2,
    chisq=chisq,
    chrom=ann.variants["chrom"].to_numpy(),
    n_samples=360_000,
)

fit = fit_nuclear_norm_path(
    dataset,
    n_genes=len(ann.genes),
    n_mechanisms=len(ann.mechanisms),
    lambdas=[1e-2, 1e-3, 1e-4],
    max_iter=200,
    lr=1e-2,
    device="cpu",
)
```

Compare precomputed `T = R2 @ A` against the factorized representation:

```python
from compass.benchmark import benchmark_representations

benchmark_representations(A, R2, len(ann.genes), len(ann.mechanisms))
```

## Notes

`build_positional_ld` is only a development fallback. For inference, replace it with
reference-panel squared correlations loaded from the UKBB LD `.npz` blocks.

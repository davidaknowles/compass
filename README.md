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
~/knowles_lab/data/compass
```

Current downloaded files:

```text
~/knowles_lab/data/compass/raw/zenodo_top_assoc/*.tsv.gz
~/knowles_lab/data/compass/raw/abc/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz
~/knowles_lab/data/compass/raw/ad_gwas_2026/GCST90704647.hg19.tsv.gz
~/knowles_lab/data/compass/raw/ukbb_ld/
~/knowles_lab/data/compass/results/
```

The AD GWAS is the 2026 European consensus meta-analysis GCST90704647, with
proxy samples excluded. The downloaded GRCh38 GWAS-SSF file is normalized to
hg19 by `scripts/prepare_ad_2026_sumstats.py`. The eQTL top-association files
are from Zenodo record `15860973`. The default annotation source is public ABC
enhancer-gene scores from Nasser et al. 2021.

Glass Lab v2 brain-cell ABC predictions can be prepared for the hg19 GWAS and
LD references with `scripts/prepare_glass_abc_v2.py`. The prepared panel
contains astrocyte, microglia, neuron, and oligodendrocyte annotations and is
used by `scripts/slurm/run_glass_abc_compass_b6k.sbatch` and the matching
BaselineLD-adjusted S-LDSC scripts.

## Data Sources

The downloader uses these public sources:

- SingleBrain top-association eQTL files: Zenodo record `15860973`, `https://zenodo.org/records/15860973`
- ABC enhancer-gene predictions in 131 biosamples: Engreitz Lab/Nasser et al. 2021, `https://mitra.stanford.edu/engreitz/oak/public/Nasser2021/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz`
- Alzheimer GWAS summary statistics: GWAS Catalog GCST90704647, `https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90704001-GCST90705000/GCST90704647/`
- UK Biobank LD matrices used by PolyFun: Broad/Alkes S3 prefix, `https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD/`

## Download Commands

Download or refresh all required public inputs, including every autosomal UKBB
LD `.npz` block and `.gz` metadata sidecar:

```bash
python scripts/download_required_data.py \
  --download-top-assoc \
  --download-abc \
  --download-ad \
  --download-ld-metadata \
  --download-ld-npz
```

Normalize the downloaded GRCh38 AD statistics to the hg19 build used by the LD
references:

```bash
module load Kent_tools/461-GCC-12.2.0
python scripts/prepare_ad_2026_sumstats.py
```

Submit the same full download through Slurm:

```bash
sbatch scripts/slurm/download_all_data.sbatch
```

## Usage

Run the default genome-wide workflow:

```bash
python scripts/run.py
```

The default annotation source is ABC CRE-to-gene predictions. For AD, the
default ABC contexts include the brain-labelled biosamples in the public table
plus myeloid proxy contexts for microglia:
`astrocyte-ENCODE`, `bipolar_neuron_from_iPSC-ENCODE`,
`H1_Derived_Neuronal_Progenitor_Cultured_Cells-Roadmap`,
`CD14-positive_monocyte-ENCODE`, `CD14-positive_monocytes-Roadmap`, and
`THP-1_macrophage-VanBortle2017`. Use all 131 ABC biosamples with:

```bash
python scripts/run.py --abc-cell-types all
```

Use the AD-proximal panel plus three peripheral controls (white adipose,
gastrocnemius muscle, and uterus) with:

```bash
python scripts/run.py --abc-context-panel ad_with_controls
```

The run requires known sample sizes. The normalized AD GWAS contains
per-variant `Neff_total`, loaded as `N`; otherwise pass a scalar sample size explicitly:

```bash
python scripts/run.py --gwas path/to/sumstats.tsv.gz --n-samples 360000
```

Load top eQTL annotations with an intercept mechanism:

```python
from compass.data import load_top_assoc_annotations

ann = load_top_assoc_annotations(
    "~/knowles_lab/data/compass/raw/zenodo_top_assoc",
    annotation_value="z2",
    add_intercept=True,
)
```

Build sparse annotations and UKBB LD squared correlations:

```python
from compass.ld import annotation_triples_to_csr, build_ukbb_ld_r2

A = annotation_triples_to_csr(
    ann.triples,
    n_variants=len(ann.variants),
    n_genes=len(ann.genes),
    n_mechanisms=len(ann.mechanisms),
)
R2, ld_diagnostics = build_ukbb_ld_r2(
    ann.variants,
    "~/knowles_lab/data/compass/raw/ukbb_ld",
)
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
    lambdas=[1e3, 3e2, 1e2, 3e1, 1e1, 3, 1, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
    cv=True,
    max_iter=200,
    lr=1e-8,
    device="cpu",
)
```

By default `scripts/run.py` chooses the nuclear-norm penalty by CRE-structured
cross-validation. Different LD-separated CRE groups for the same gene are placed
in different folds where possible. Whole-chromosome CV was removed because it
holds out mostly unseen genes and is invalid for selecting gene-level penalties.
The `--no-cv` flag is available only for debugging.

ABC runs use all GWAS variants represented in the UKBB LD reference as regression
rows. Variants that do not overlap an ABC CRE have all-zero annotation rows but
remain in the model for LD tagging and residual LD-score estimation.

LD is represented as chromosome-level sparse blocks. Values are thresholded at
`r2 >= 0.01`, stored as fp16, and applied in the fit as torch sparse fp16
matrices. Non-LD model tensors default to fp32; compare fp16 model tensors with:

```bash
python scripts/run.py --model-dtype float16
```

Run ordered gene-set enrichment on a completed fit:

```bash
python scripts/run_gsea.py \
  --b-tsv ~/knowles_lab/data/compass/results/<run-name>.B.tsv \
  --out-dir ~/knowles_lab/data/compass/results/<run-name>.gsea
```

Run the ABC/UKBB-LD recovery simulation study (three independent seeds at each
of `N_eff=100000` and `200000`):

```bash
bash scripts/slurm/submit_abc_recovery_simulation.sh
```

The study uses bipolar neuron and CD14 monocyte ABC annotations as causal
contexts (70:30 of total `h2=0.20`) and adipose, muscle, and uterus as null
controls. It selects 10% of ABC-positive variants per causal context, generates
model-matched noncentral chi-square statistics with the real UKBB `R2` blocks,
and runs the same LD-component CV used for the real-data fit.

Compare precomputed `T = R2 @ A` against the factorized representation:

```python
from compass.benchmark import benchmark_representations

benchmark_representations(A, R2, len(ann.genes), len(ann.mechanisms))
```

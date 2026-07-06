# LAB NOTEBOOK

## 2026-07-05 genome-wide AD COMPASS run

### Model components

- COMPASS fits GWAS chi-square statistics with LD-smoothed variant-gene-context annotations.
- Primary fitted parameter is a non-negative gene-by-context matrix `B`.
- The fitted nuclear-norm objective uses non-negative proximal updates and a residual LD-score term `tau`.
- Default runs now select `lambda` by full leave-one-chromosome cross-validation and then refit on all chromosomes.
- The strict rank-1 alternating optimizer is retained as a baseline only.

### Data

- Data root: `/gpfs/commons/home/daknowles/knowles_lab/data/compass`.
- AD GWAS: `raw/ad_gwas/AD_sumstats_Jansenetal_2019sept.txt.gz`.
- SingleBrain top eQTL annotations: `raw/zenodo_top_assoc/*_top_assoc.tsv.gz`.
- UKBB LD: all downloaded autosomal `.npz` blocks under `raw/ukbb_ld`, with metadata sidecars.
- Dataset cache: `cache/z2.intercept.AD_sumstats_Jansenetal_2019sept.txt.gz.gwas.*`.
- Cached design: 76,101 variants, 15,363 genes, 8 contexts, 122,904 parameters, 166,253 annotation nonzeros, 7,829,001 LD nonzeros.

### Implementation details

- `scripts/run.py` is the genome-wide entry point.
- Default lambda grid is now `1e3,3e2,1e2,3e1,1e1,3,1,3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4`.
- `src/compass/model.py` records `cv_method=leave_one_chromosome` and the held-out chromosome list in fit metadata.
- `scripts/run_gsea.py` produces global and cell-type ranked gene tables and ordered g:Profiler enrichment with all assayed genes as background.
- GENCODE v38 at `/gpfs/commons/home/daknowles/knowles_lab/data/ADSP_reguloML/fasta_files/gencode.v38.annotation.gtf` was used for ENSG-to-symbol mapping.

### Lambda selection findings

- Narrow CV run `compass-chromcv` selected the largest tested lambda, `0.1`.
- Higher grid `compass-chromcv-higherlam` selected the largest tested lambda, `100`, and produced only 4 positive genes with total `B` sum `8.096e-06`.
- Final broad grid `compass-chromcv-veryhighlam` tested `1e5,3e4,1e4,3e3,1e3,3e2,1e2`.
- Final leave-one-chromosome CV scores were identical for `lambda >= 300` at `163.960192995`; `lambda=100` was slightly worse at `163.960196300`.
- The selected final fit used `lambda=100000` and had zero positive gene-context entries, zero positive genes, and total mediated score `0`.

### Alternating optimizer

- Rank-1 alternating chromosome-CV run `compass-rank1-chromcv` gave held-out errors around `1e16`, far worse than the nuclear-norm fits near `164`.
- The rank-1 solution was dense across genes and is not useful for this AD analysis.

### GSEA

- Primary GSEA was run on `compass-chromcv-veryhighlam.B.tsv` into `results/compass-chromcv-veryhighlam.gsea`.
- Because the CV-selected `B` matrix was all zero, global and cell-type positive gene lists were empty.
- Sensitivity GSEA was run on the adjacent weaker `lambda=100` fit in `results/compass-chromcv-higherlam.gsea`.
- The weaker fit's four positive global genes were `ZNF222`, `ZBTB11-AS1`, `CCDC153`, and `LCN10`, but g:Profiler returned no meaningful enrichment (`p=1.0` top terms).

### Interpretation

- The final grid brackets the optimum by reaching a broad null mediated plateau.
- Under the current top-eQTL annotation design, weighting, and objective, chromosome-held-out prediction favors removing the mediated component.
- The current analysis therefore does not support robust global or cell-type-specific AD gene discoveries.

## 2026-07-05/06 validation correction and ABC annotation migration

### Validation issue

- Whole-chromosome CV is invalid for selecting `lambda` in the gene-indexed COMPASS model.
- When a chromosome is held out, most held-out genes have no training annotations, so the validation task becomes extrapolation to unseen genes.
- The previous chromosome-CV null result should be treated as a validation diagnostic, not a biological AD conclusion.

### New annotation source

- Downloaded public ABC predictions:
  `/gpfs/commons/home/daknowles/knowles_lab/data/compass/raw/abc/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz`.
- Source URL:
  `https://mitra.stanford.edu/engreitz/oak/public/Nasser2021/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz`.
- The file is the Nasser et al. 2021 all-biosample ABC table containing element-gene links with ABC score at least 0.015.
- `scripts/download_required_data.py --download-abc` now reproduces the download.

### New CV design

- `src/compass/data.py::load_abc_annotations` intersects GWAS variants with ABC CRE intervals.
- Annotation values are ABC scores for variant-overlapping CRE-gene-biosample links.
- Nearby CREs for the same gene are collapsed into coarse LD-distance clusters using `--cre-ld-gap`, default `1,000,000` bp.
- CRE clusters for the same gene are assigned across `--cre-folds`, default `5`.
- Variant rows inherit a CRE fold when their overlapping CRE-gene links agree; ambiguous variants are allowed in training but not used as held-out validation rows.
- `src/compass/model.py` now uses grouped variant CV (`cv_method=cre_ld_group`) and no longer calls chromosome-heldout CV from the fit path.

### Default AD ABC contexts

- The public table has three brain-labelled biosamples by name:
  `astrocyte-ENCODE`,
  `bipolar_neuron_from_iPSC-ENCODE`,
  `H1_Derived_Neuronal_Progenitor_Cultured_Cells-Roadmap`.
- These are now the default `--abc-cell-types` for AD.
- Passing `--abc-cell-types all` uses all 131 public ABC biosamples.

### Setup test

- Completed a setup-only run for the three brain-labelled ABC biosamples.
- Cached GWAS load took about 11 seconds.
- ABC overlap construction took about 134 seconds.
- GWAS alignment took about 3 seconds and annotation matrix construction took about 0.02 seconds.
- LD assembly over 910 blocks took about 1933 seconds with 8 jobs.
- Final cached ABC design: 171,788 variants, 67,196 gene-context parameters, 1,291,460 annotation nonzeros, and 40,455,388 LD nonzeros.
- A small synthetic smoke test exercised `fit_nuclear_norm_path` with `cv_method=cre_ld_group`; it produced fold scores for folds `[0, 1]` and selected among the provided lambdas.

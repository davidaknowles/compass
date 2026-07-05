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

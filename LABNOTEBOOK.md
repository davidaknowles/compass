# LAB NOTEBOOK

## 2026-07-20 hierarchical context effects

- The original nuclear penalty acts on the complete gene-by-context matrix and therefore shrinks both context-wide and gene-specific effects. Added a hierarchical decomposition with an unpenalized global context layer and a non-negative nuclear-regularized gene-deviation matrix. At large deviation penalty, the model reduces to one shared coefficient per context, matching the gene-aggregated S-LDSC assumption.
- The legacy `1 / observed chi-square` regression weight is response-dependent and drove the non-negative global coefficients to zero. Hierarchical fits now use response-independent uniform weights by default; legacy fits retain their previous weighting unless explicitly changed.
- Global context effects are estimated jointly with the residual LD-score coefficient by non-negative weighted least squares. Chromosome leave-one-out estimates provide context standard errors. Binary gene-aggregated annotations and flat peak annotations are both supported.
- In unadjusted high-penalty pilots, flat H3K27ac context effects prioritized neurons for SCZ 2026. For AD 2026, the microglial coefficient matched the BaselineLD S-LDSC magnitude, but neuronal and oligodendrocyte coefficients remained spuriously positive. This isolates BaselineLD nuisance adjustment as necessary for AD context specificity.
- Added a calibrated two-stage form of the hierarchical model: official BaselineLD-adjusted S-LDSC estimates the global context profile, COMPASS re-estimates one non-negative profile scale in its own LD reference, and gene-specific deviations are fit conditional on the scaled profile. Negative S-LDSC coefficients are clipped to zero because global context effects represent non-negative heritability contributions. This avoids transferring reference-specific coefficient magnitudes while preserving the baseline-adjusted cell-type contrast. The AD scale check prioritized microglia (effect `4.45e-7`, chromosome-jackknife `z=2.03`); full LD-component-CV deviation fits for AD and SCZ are pending.
- Official BaselineLD-adjusted S-LDSC gives coherent trait-specific positive controls. For AD 2026, PU1 H3K27ac has 13.0-fold enrichment and coefficient `z=2.97`; binary Glass microglial ABC has 27.1-fold enrichment and coefficient `z=3.22`. For SCZ 2026, NeuN H3K27ac has 5.71-fold enrichment and coefficient `z=7.28`, NeuN H3K4me3 has coefficient `z=4.63`, and binary Glass neuronal ABC has coefficient `z=3.74`.
- Uncalibrated flat-H3K27ac COMPASS independently ranks neurons first for SCZ (effect `3.65e-7`, `z=2.46`) but also assigns an astrocyte effect (`z=2.34`). For AD, the analogous uncalibrated fit is nonspecific: neuron and oligodendrocyte effects exceed microglia and all three have similar uncertainty. Therefore the AD calibrated result is explicitly a conditional compatibility result whose cell-type ordering comes from S-LDSC, not an independent COMPASS discovery.
- Submitted full ten-fold hierarchical fits as jobs `19283870` (AD) and `19283871` (SCZ). These fit nuclear-regularized gene deviations conditional on the scaled H3K27ac S-LDSC profile. The completion audit must compare the selected CV score with the high-penalty/no-deviation limit and inspect deviation mass before claiming added COMPASS resolution.
- The initial full jobs revealed that scaled context effects were reinitialized to the unscaled profile at every lambda instead of retaining the preceding lambda's fitted scale. This added a stale-scale gradient update without changing the converged model. Corrected the warm start and added diagnostics and regression coverage.
- Hierarchical CV now writes an atomic checkpoint after every completed fold/lambda pair. It stores fold-specific `B`, context effects, residual coefficients, scores, and next-lambda indices, validates the fold/grid/profile/model contract on load, and resumes incomplete folds. This protects the full grid from scheduler time limits without reducing optimization iterations or omitting weak penalties.
- The production Slurm entry point is idempotent: it exits when both final metadata and coefficient archives exist, and otherwise resumes the automatic CV checkpoint. Dependent continuation jobs can therefore be queued with `afterany`; successful primary runs make them no-ops, while timed-out runs continue from the last completed fold/lambda pair.
- Replaced the pre-checkpoint full jobs with checkpoint-enabled primaries `19285062` (AD) and `19285063` (SCZ). Queued two `afterany` continuations per trait: `19285097` and `19285098` for AD, and `19285099` and `19285100` for SCZ. The first AD checkpoint reproduces the calibrated ordering in fold 0: microglia `4.61e-7`, oligodendrocyte `4.57e-8`, neuron `2.06e-8`, and astrocyte zero.
- Near the deviation activation boundary, tiny coefficients repeatedly entered and left the non-negative support: the coefficient-relative update remained near 0.07 even after the objective changed by only `1.6e-5` over ten iterations. Added a windowed relative-objective stopping criterion (`1e-5` over ten iterations) alongside the existing parameter and stagnation criteria. This is specific to the hierarchical optimizer and records its convergence reason; it avoids treating unstable relative changes of near-zero coefficients as substantive loss improvement.
- Continuation profiling showed that deterministic LD-component fold construction could take about 29 minutes for the SCZ design because the original dataset cache was written before folds were assigned and was not updated afterward. Added an atomic sidecar fold cache keyed by the dataset cache, fold count, and LD threshold. It stores validated fold labels, score labels, and diagnostics so future continuations skip the full LD graph traversal.
- The joint scaled-profile path remained allocation-nonidentifiable at weak penalties: in AD fold 0, the explicit context scale went to zero while the gene-deviation matrix became diffuse across contexts. Replaced this with a sequential hierarchy. Each training fold estimates the shared profile scale once at the context-only largest lambda, freezes that context vector, and then cross-validates only the gene deviations and residual term. The full-data refit does the same. Checkpoints include this mode in their compatibility contract, preventing invalid joint-scale states from being resumed.
- Archived the incompatible joint-scale checkpoints. Jobs `19285098` (AD) and `19285100` (SCZ) were also rejected after checkpoint inspection showed that their submitted batch-script snapshots still requested joint scaling. The verified frozen-scale replacements are `19286105` (AD) and `19286107` (SCZ), with continuations `19286106` and `19286108`; their submitted scripts explicitly request `scaled_frozen`. The prior joint-scale CV scores are diagnostic only and must not be reported as final model-selection results.
- Added a context-contribution summary that separates direct heritability from the frozen global layer and the gene-deviation matrix, then reports their total and normalized fraction by context. Final ranking audits use this total, so diffuse deviations cannot be hidden by reporting only the calibrated global coefficient. A standalone postprocessor generates the same table for fits started before this output was introduced, without reloading LD or refitting.
- Added second idempotent continuations `19286197` (AD) and `19286198` (SCZ) after the existing continuation jobs. Dependent CPU jobs `19286199` and `19286200` generate the final context-contribution tables after the respective chains finish.
- In AD fold 0, the first deviation-bearing fit (`lambda=100`) increased held-out MSE from the context-only plateau `46.05195` to `46.05761`. Its aggregate contribution remained microglia-dominant: 70.3% microglia, 11.0% oligodendrocyte, 9.1% neuron, 2.2% astrocyte, and 7.4% gene-shared intercept. This is an interim single-fold diagnostic; final selection requires all ten folds and the complete grid.
- Full-genome continuation profiling showed that the original `1e-8` proximal step was unnecessarily conservative for the frozen-context objective. At `lambda=30`, `lr=1e-7` reached objective `313.62` in 20 iterations, versus 180 iterations at `lr=1e-8`, and continued monotonically to `311.71` at iteration 100. Its held-out fold-0 MSE was `42.2674`, substantially below the context-only `46.0520`. Microglia remained the largest aggregate contribution at 46.4% of all modeled mass (59.6% after excluding the gene-shared intercept). Hierarchical runs now default to `lr=1e-7`; other methods retain `1e-8`.
- The converged production fit at AD fold-0 `lambda=30` improved held-out MSE further to `42.1771`. Aggregate contribution remained 46.4% microglia, 12.9% neuron, 12.0% oligodendrocyte, 6.7% astrocyte, and 22.1% gene-shared intercept. Thus the first strongly predictive deviation model preserves microglia as the leading biological context.
- A follow-up `lr=3e-7` profile accelerated AD `lambda=10` by another factor of about three and gave provisional held-out MSE `37.8414`; microglia comprised 52.0% of aggregate mass. However, the next `lambda=3` objective increased after the first update, demonstrating that one fixed step is not suitable across the complete path.
- Added objective-based hierarchical step adaptation. The first prototype rejected any uphill update, but full-genome tracing showed that AD `lambda=10` alternates transient uphill and larger downhill updates while improving rapidly. Backoff now requires five consecutive objectives above the best value; it then restores the best coefficient/context/residual state and halves the step. This retains the fast `3e-7` trajectory while controlling sustained instability. The provisional AD `lambda=3` held-out MSE was `38.5172`, worse than `lambda=10`, while microglia remained the leading context at 51.2%. Hierarchical fits start at `3e-7` and adapt downward; other methods retain `1e-8`.
- The first SCZ deviation-bearing checkpoint (`lambda=100`) improved fold-0 held-out MSE from the context-only `9.31676` to `9.21994`. Aggregate contribution remained neuron-dominant: 59.8% neuron, 10.5% astrocyte, 10.3% oligodendrocyte, 4.0% microglia, and 15.4% gene-shared intercept.
- Replaced the fixed-step production allocations after preserving their atomic state. Adaptive AD jobs `19286610`, `19286611`, and `19286612` start at `lr=3e-7`, followed by context-summary job `19286613`. SCZ L40S warm-up `19286310` preserved the completed `lambda=100` checkpoint but was cancelled during the slower, incomplete `lambda=30` fit to release account GPU capacity; adaptive B6K continuations are `19286614`, `19286615`, and `19286616`, followed by summary job `19286617`. Superseded fixed-step continuations were cancelled. All active submitted scripts request frozen scaling and adaptive `lr=3e-7`.
- Added an executable final audit that combines each trait's selected CV penalty and context-contribution table, quantifies improvement over the context-only limit, and fails unless AD is led by microglia and SCZ by neurons. Job `19286642` runs this gate after both contribution-summary jobs and writes the cross-trait audit table.
- The completed pilot folds bracket the useful deviation penalty before the weak-penalty tail. AD fold 0 selected `lambda=30` (MSE `42.1771` versus `46.0520` at the context-only limit), while SCZ fold 0 selected `lambda=100` (MSE `9.21994` versus `9.31676`). Values below `lambda=10` were uniformly poor in both fold-0 paths; AD fold 1 already favored the context-only limit and deteriorated from MSE `16.0404` to `87.3643` at `lambda=30`. The production grid was therefore pruned to the eight pre-specified values from `1e6` through `10`, preserving the completed fold scores and retaining all ten LD-component folds. Grid extension is disabled for this run because the retained lower boundary did not win either completed path.
- Archived the original twenty-penalty checkpoints and resumed from compact checkpoints containing the retained grid. AD jobs `19287430`, `19287431`, and `19287432` resume after two completed folds; SCZ jobs `19287433`, `19287434`, and `19287435` resume fold 1 at `lambda=30`. Contribution jobs `19287436` and `19287437` follow the respective chains, and cross-trait ranking audit `19287438` runs after both summaries.

## 2026-07-17 AD GWAS correction

- All AD analyses completed before this entry used the Jansen et al. 2019 summary statistics and must not be described as analyses of the 2026 consensus GWAS.
- The corrected default is GWAS Catalog accession GCST90704647, a European Alzheimer disease and related dementias meta-analysis of 1,119,443 participants. Its metadata states that proxy samples were excluded.
- The source is GWAS-SSF on GRCh38 and provides variant-specific `Neff_total`. `scripts/prepare_ad_2026_sumstats.py` maps coordinates to hg19 and exports `Neff_total` as `N` for COMPASS and S-LDSC.
- New analyses use separate `ad2026` result names. The Jansen 2019 files and results are retained as historical outputs.

### Initial submission correction

- The first AD 2026 and SCZ 2026 COMPASS setup jobs exceeded 128 GB while `_find_compatible_allrow_ld` loaded complete LD matrices for each candidate before checking row compatibility.
- Compatibility checks now load only chromosome row-index arrays, then materialize one matching LD archive. Setup requests 192 GB to accommodate the normalized GWAS, annotation tables, and selected LD archive concurrently.
- The first S-LDSC build jobs failed because braces inside a Bash `${parameter:?message}` usage string terminated parameter expansion early. Argument validation now uses an explicit argument-count check.
- The first dependent COMPASS GPU fits exposed the same brace-parsing defect in their target argument. Their target validation now uses the same explicit argument-count check.
- All jobs depending on those failed submissions are obsolete and were replaced with corrected dependency chains.

## 2026-07-17 SCZ 2026 EUR analysis setup

### GWAS input

- Downloaded the EUR-only schizophrenia meta-analysis BCF from the Bigdeli et al. 2026 Synapse release (`syn66321648`, `SCZ_EUR_autosomes.bcf`).
- The source BCF is GRCh38. `scripts/prepare_scz_2026_sumstats.py` lifts single-variant coordinates to hg19 with the hg38-to-hg19 chain, retains rsID, effect allele, beta, SE, and effective N, and writes `raw/scz_gwas_2026/SCZ_EUR_2026.hg19.tsv.gz`.
- The source has 12,991,921 autosomal records; 12,931,677 lifted to hg19. Effective sample size is retained per variant rather than replaced by a study-wide constant.

### Analysis plan

- `scripts/slurm/setup_scz2026_compass.sbatch` creates the all-variant UKBB-LD cache once. `scripts/slurm/run_scz2026_compass_b6k.sbatch` runs COMPASS with Glass ABC, expressed-gene open-chromatin-to-TSS links, public ABC, and the public-ABC control panel.
- `scripts/slurm/run_scz2026_sldsc.sbatch` builds, computes LD scores, and runs official BaselineLD-adjusted S-LDSC for continuous, binary, and distal Glass ABC, brain ATAC/histone peaks, and the public ABC panel.
- BaselineLD builders now accept `--gwas` and normalize arbitrary supported GWAS schemas through `load_gwas_sumstats`; all LDSC inputs require known per-variant sample sizes.

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
- Global UK Biobank LD components at `r2 >= 0.01` are assigned to ten folds while balancing all regression rows and spreading a gene's components across folds.
- Every variant row inherits its component fold and is excluded from that fold's training loss; non-CRE rows therefore cannot leak correlated GWAS signal across the split.
- Only held-out CRE rows with at least one linked gene represented in a training component are scored. Unsupported gene effects remain zero in the fold fit, which restricts the mediated prediction to supported genes.
- `src/compass/model.py` uses this global component CV (`cv_method=ld_component`) and no longer calls chromosome-heldout CV from the fit path.

### Default AD ABC contexts

- The public table has three brain-labelled biosamples by name:
  `astrocyte-ENCODE`,
  `bipolar_neuron_from_iPSC-ENCODE`,
  `H1_Derived_Neuronal_Progenitor_Cultured_Cells-Roadmap`.
- The public table has no obvious microglia biosample. Added myeloid proxy contexts:
  `CD14-positive_monocyte-ENCODE`,
  `CD14-positive_monocytes-Roadmap`,
  `THP-1_macrophage-VanBortle2017`.
- These six contexts are now the default `--abc-cell-types` for AD.
- Passing `--abc-cell-types all` uses all 131 public ABC biosamples.

### All-row LD correction

- The first ABC pilot only modeled variants overlapping selected ABC CREs.
- That design is invalid for LD accounting because non-CRE SNPs that tag CRE SNPs through LD are absent from the regression rows.
- Corrected design:
  all GWAS variants represented in the UKBB LD metadata are model rows;
  variants outside selected ABC CREs have all-zero annotation rows;
  CRE fold labels are used only to define validation rows, while ambiguous or non-CRE rows can remain in training/background.
- Added `filter_variants_to_ukbb_ld` so the all-row universe is the GWAS/UKBB-LD intersection, not all 13.37M imputed GWAS rows.
- The previous result `compass-abc-brain-crecv` is an annotated-only pilot and should not be interpreted as a final model fit.

### LD precision and representation

- Full all-row genome-wide `R2` materialization failed with OOM at 192 GB.
- Sampled five UKBB LD blocks: `r2 >= 0.01` keeps about 2.95% of stored entries, a useful sparsity increase.
- LD is now built as chromosome-level sparse blocks with `r2_cutoff=0.01`.
- LD values are stored in fp16 custom CSR archives and converted to torch sparse fp16 tensors for fitting.
- SciPy is only used as a construction/cache container because SciPy sparse does not support fp16 arithmetic.
- `--model-dtype float32|float16` controls non-LD model tensors for comparison; LD remains fp16 in both cases.
- A synthetic two-chromosome smoke test ran both `model_dtype=float32` and `model_dtype=float16`; both completed CRE-group CV and reported `ld_dtype=float16`.

### Full all-row setup and fit status

- Full all-row ABC/UKBB setup job `18707724` completed and wrote the dataset cache.
- Cached design summary: 12,362,312 variants, 128,772 gene-context parameters, 1,818,567 annotation nonzeros, and 10,866,690,906 chromosome-level `R2` nonzeros.
- Setup timing: annotations about 195 seconds, UKBB panel filtering about 244 seconds, chromosome LD construction about 3006 seconds, and cache write about 1863 seconds.
- The dependent fp32 and fp16 model fit jobs failed with CUDA OOM while coalescing all chromosome LD tensors on GPU.
- The next fit implementation should keep LD cached on disk/CPU and move one chromosome block to GPU at a time during each objective pass.
- Removed the obsolete `--use-precomputed` path and the precomputed-vs-factorized benchmark because LD should always be applied as fp16 sparse `R2` blocks.

### Streaming chromosome LD during fitting

- Updated the Torch fit loops so chromosome LD is no longer preloaded onto GPU for all chromosomes.
- Each objective pass now converts only the active chromosome `R2` block to a torch fp16 sparse tensor, evaluates that block, immediately backpropagates its normalized loss contribution, then releases the block graph before moving to the next chromosome.
- On CUDA, LD conversion uses torch sparse CSR to avoid COO `coalesce()` memory overhead; CPU smoke tests continue to use COO because fp16 CSR matmul is not implemented on CPU in the current Torch build.
- CRE-fold CV keeps the full chromosome LD operators in memory. Held-out CRE-fold rows have zero training weight, while their predictions are scored through the same full-LD GPU path; this avoids LD submatrix copies and preserves all variant rows for LD accounting.
- Full streaming jobs `18708903` and `18708904` still failed on L40S: fp32 OOMed during backward for a whole chromosome block, and fp16 model dtype exposed that CUDA sparse COO annotation matmul is not implemented for half.
- Added `--ld-chunk-nnz` so each chromosome is evaluated in row chunks while retaining chromosome-level CPU/cache organization.
- Annotation sparse matmul now remains fp32; fp16 model dtype stores/updates model parameters in half but casts coefficients to fp32 for the annotation multiply, while LD remains fp16.
- L40S chunk smoke job `18709273` completed for both `model_dtype=float32` and `model_dtype=float16` with `ld_gpu_layout=csr`.
- L40S int32 CSR smoke job `18709337` confirmed fp16 CSR sparse matmul and backward work with int32 `crow_indices` and `col_indices`.
- Added int32 CUDA CSR indices for LD chunks and timing metadata for LD chunk conversion and eval/backward time.
- Model-path L40S smoke job `18709338` completed with `ld_gpu_layout=csr`, `ld_index_dtype=int32`, and populated LD chunk timing fields.
- Added `scripts/benchmark_ld_chunks.py` to benchmark cached chromosome LD row-chunk sizes on GPU.
- On chromosome 22, chunk sizes 25M, 50M, 100M, and 200M processed 102,321,592 LD nonzeros with peak GPU memory from about 1.5 GB to 6.1 GB; 200M was fastest in this small chromosome benchmark.
- On chromosome 6, chunk sizes 100M, 150M, and 200M processed 907,481,458 LD nonzeros with peak GPU memory about 6.0 GB, 8.9 GB, and 11.9 GB respectively; 150M had the fastest total time in the benchmark.
- Updated the default `--ld-chunk-nnz` to 150M.

### Obsolete annotated-only setup test

- Completed a setup-only run for the three brain-labelled ABC biosamples before correcting the row universe.
- Cached GWAS load took about 11 seconds.
- ABC overlap construction took about 134 seconds.
- GWAS alignment took about 3 seconds and annotation matrix construction took about 0.02 seconds.
- LD assembly over 910 blocks took about 1933 seconds with 8 jobs.
- Final cached ABC design: 171,788 variants, 67,196 gene-context parameters, 1,291,460 annotation nonzeros, and 40,455,388 LD nonzeros.
- This cache is annotated-only and obsolete for inference.
- A small synthetic smoke test exercised the superseded per-gene fold scheme. It is retained only as historical validation of the previous implementation.

### GPU profiling and fit stabilization

- Cancelled the previous fp32 and fp16 full fits after 3.6 days: both were still in an unlogged CV loop and would not finish within the requested allocation.
- `cProfile` on a large chromosome showed that SciPy CSR row slicing and compressed-cache loading, rather than CUDA sparse kernels or SVD, dominated avoidable runtime.
- CUDA LD chunks now use direct views of their parent CSR `indices` and `data` arrays; only the short rebased row-pointer array is copied. Host LD values remain fp16 until transfer.
- On the representative large chromosome with 150M-nonzero chunks, this reduced a forward/backward pass from 4.14 s to 0.79 s at the same 8.9 GB peak GPU allocation.
- The full-cache startup time fell from about 142 s to 30 s after adding uncompressed CSR archives and an opt-in manifest that leaves the original compressed cache untouched.
- The residual non-mediated LD coefficient was previously initialized through `softplus(1e-8)`, which made its gradient effectively zero. It is now updated exactly as a non-negative weighted least-squares coordinate during each proximal iteration, without an additional genome-wide pass.
- With the exact residual update and `lr=1e-8`, a 100-iteration full-genome trace reduced the data loss monotonically from 1.01523 to 0.77576 and reached relative coefficient change `9.86e-3`. The default tolerance is now `1e-2`, retaining 500 iterations as a safety cap.
- A full iteration after fit setup takes about 6.4 s on the profiled GPU. CUDA sparse backward and fp16 CSR transfer are now the principal recurring costs; custom CUDA SpMV and fp8 LD are not justified by this profile.
- A full five-fold, one-iteration CRE-CV smoke test completed without copying train/test LD matrices and included the final full-data step. The previous fold-subsetting implementation was removed after it exceeded host memory.
- FP16 model parameters are not suitable for this path: at `lambda=100`, fp16 quantization immediately selected an all-zero coefficient matrix, while BF16 followed the same nonzero trajectory as fp32. LD values remain fp16 for every model dtype.
- Added `bfloat16` as a model-parameter option. BF16 outputs are converted to fp32 only at the NumPy result boundary.
- Convergence norms are now evaluated in fp32 so a zero fp16 coefficient matrix does not produce `0/0` from an underflowed epsilon.
- Fit selection and plateau stopping now use the full penalized objective (weighted data loss plus nuclear-norm penalty), not data loss alone. A BF16 `lambda=100` trace reached the plateau stop at iteration 40 rather than the previous 500-step cap.
- The completed FP32 and BF16 fits both selected the lower lambda boundary (`1e-4`), so that grid does not identify an interior CV optimum. Nuclear-path CV now automatically continues from each fold's warm start at up to four successively smaller lambdas (factor 3 by default) whenever the lower boundary wins.

## 2026-07-11 LD-component CV diagnostic

### Implementation

- Added `scripts/diagnose_ld_components.py`, which computes exact connected components independently by chromosome from the cached UK Biobank LD graph.
- The diagnostic uses a Numba disjoint-set union implementation and evaluates every requested threshold in a single scan of each chromosome's sparse CSR matrix.
- It writes genome-wide and per-chromosome summaries, component-size histograms, and plotnine threshold plots. `scripts/slurm/diagnose_ld_components.sbatch` runs the complete diagnostic.

### Genome-wide results

- The all-autosome sweep completed in 368 seconds of analysis time, with 19.5 GB peak resident memory.
- At `r2 >= 0.01`, the graph has 3,354 components across 12,362,312 variants. The largest component contains 34,901 variants (0.282% of rows), and 5.427 billion undirected edges are retained.
- Increasing the threshold to `0.015`, `0.02`, `0.03`, `0.05`, `0.075`, and `0.10` yields 5,841, 9,001, 17,719, 43,661, 87,808, and 140,412 components, respectively.
- The largest component remains small across this range (34,343--34,901 variants). At `r2 >= 0.01`, the 99th percentile component size is 18,248 variants; at `r2 >= 0.10`, it is 45 variants, with a small number of much larger components remaining.

### Consequence for CV

- `r2 >= 0.01` is a viable candidate for global LD-component CV: it preserves separation for every cached LD edge while leaving thousands of indivisible units for ten-fold balancing.
- The selected implementation uses ten folds at `r2 >= 0.01`; the per-gene, distance-binned CRE-fold code has been removed.

### Ten-fold implementation validation

- `make_ld_component_cv_groups` assigns every all-variant regression row to one chromosome-local LD component, then greedily balances components across ten global folds while spreading components linked to the same gene.
- The full-cache setup test completed successfully with 3,354 components, a largest component of 34,901 variants, and fold row totals of 1,236,231 or 1,236,232 each.
- There are 104,411 held-out CRE scoring rows after requiring at least one linked gene to have an annotation in another fold. The ten fold-specific score counts range from 8,975 to 11,558.
- Cache loading took 170 seconds and component/fold construction took 269 seconds on the validation CPU allocation, with approximately 75 GB peak resident memory.

### Ten-fold AD fit

- The FP32 ten-fold LD-component CV fit completed successfully after 10.6 hours, with 72.4 GB peak resident memory.
- CV selected the interior penalty `lambda=100`: mean held-out MSE was `2.98058882`, compared with `2.98074335` at `lambda=300` and `1000`, and `2.98737117` at `lambda=30`.
- The selected full-data fit has `tau=0`, 2,181 positive genes, and 15,267 positive gene-context entries. Its total coefficient sum is `8.58e-05`.
- The coefficient matrix is numerically rank one: its leading singular value is `8.63e-07`, while the remaining singular values are approximately `1e-14`. All seven contexts are positive for the same 2,181 genes, with nearly identical cross-gene coefficient ratios.
- This fit identifies a weak aggregate mediated component but does not provide stable gene- or cell-type-specific prioritization. GSEA on this rank-one, near-uniform solution would not be biologically interpretable.

### Factorized follow-up

- The rank-1 follow-up is initialized from the leading non-negative SVD factors of the selected nuclear-norm solution and uses the same selected penalty (`lambda=100`).
- The rank-1 residual coefficient now uses the same exact non-negative weighted least-squares coordinate update as the nuclear fit.
- Agora's public nominated-target export (`/api/v1/genes/nominated`) contains 967 targets in the current download. Post-fit analysis reports target overlap for global and context-specific rankings alongside ordered g:Profiler enrichment.

### Rank-one fit and post-fit analysis

- The rank-one refit at `lambda=100`, initialized from the nuclear solution, completed in 164 seconds. Its final objective was `0.782008`, compared with the nuclear fit's `0.782004`.
- Rank-one and nuclear coefficients have Pearson correlation `0.999984` and relative Frobenius difference `0.0139`; both are numerically rank one with `tau=0`.
- For ordered g:Profiler, queries were limited to the top 100 positive ranked genes. The resulting terms are dominated by broad HPA tissue signatures, not AD- or brain-specific pathways; no mechanistic interpretation is supported.
- Agora cross-reference uses HGNC-symbol matching when ABC target labels are symbols. Of 18,396 assayed genes, 865 overlap the 967 Agora nominated targets. At top 1,000 global genes, the nuclear fit has 68 targets (`p=0.00139`) and the rank-one fit has 69 (`p=0.000877`) versus about 47 expected. At top 100, both have 6 targets (`p=0.33`). These are unadjusted, exploratory overlaps.

### Peripheral-control sensitivity analysis

- Added the `ad_with_controls` ABC context panel: the six AD-proximal contexts plus `white_adipose-Loft2014`, `gastrocnemius_medialis-ENCODE`, and `uterus-ENCODE`.
- New context-panel caches may reference a verified existing all-row LD archive instead of rebuilding or copying its chromosome FP16 blocks. Reuse requires exact agreement of every chromosome's dense row indices with the newly constructed all-variant row order.
- The expanded-panel setup smoke test completed in 11.9 minutes and reused the existing LD archive. It produced 192,460 parameters, 2,825,623 annotation entries, and 165,905 eligible ten-fold validation rows; fold row counts remained balanced to within one variant.
- The expanded nuclear fit completed in 10.7 hours with 72.0 GB peak resident memory. CV selected the interior `lambda=100` (MSE `2.912379`); `lambda=300` and `1000` both had MSE `2.912998`.
- The nuclear fit remained numerically rank one (leading singular value `1.25e-6`; remaining singular values about `1e-14`) with `tau=0`. The rank-one refit initialized from that solution had coefficient correlation `0.999982` and relative Frobenius difference `0.0128`.
- Coefficient mass was not selectively concentrated in the AD-proximal contexts: intercept 14.02%, bipolar neuron 11.06%, uterine control 10.56%, neuronal progenitor 10.36%, muscle control 10.36%, CD14 monocyte 9.97%, CD14 Roadmap 9.46%, astrocyte 8.76%, adipose control 8.60%, and THP-1 macrophage 6.85%. The controls therefore fail the intended negative-control expectation under the current model.
- All context-specific rankings were nearly identical. GSEA remained dominated by broad tissue or non-AD pathway terms. The expanded gene background contains 878 Agora targets; at top 1,000, both fits contained 64 targets versus 45.6 expected (`p=0.00389`), while top-100 overlap was 7 (`p=0.17`) for nuclear and 8 (`p=0.086`) for rank one. These are exploratory, unadjusted comparisons.

## 2026-07-13 ABC/LD recovery simulation design

- `src/compass/simulation.py` constructs sparse direct variant heritability from real ABC scores, propagates it with the cached chromosome-level UKBB `R2` blocks, and draws model-matched noncentral chi-square statistics.
- The primary experiment has five ABC contexts: bipolar neuron (70% of total heritability), CD14 monocyte (30%), and adipose, gastrocnemius muscle, and uterus controls (0%). Total direct heritability is 0.20; each nonzero context independently selects 10% of its ABC-positive variants and distributes its assigned heritability in proportion to summed ABC score.
- Three seeds are run at each `N_eff=100000` and `200000`. Each seed uses the production ten-fold LD-component CV and an initialized rank-one refit. Recovery summaries include causal-context ranking, the 70:30 fitted mass split, control leakage, selected lambda, residual `tau`, and correlation of LD-propagated simulated and fitted signals.
- `src/compass/sldsc.py` implements the infinitesimal Gaussian comparison in `docs/main.tex`: ABC scores are summed over genes, LD-scored with the same UKBB `R2` blocks, and fit both one-context-at-a-time and jointly with a residual LD-score annotation. The baseline runs on the real AD statistics and every retained simulated replicate.
- All three retained simulations completed at each sample size. The nuclear model recovered the two causal contexts as the top two in 2/3 replicates at `N_eff=100000` and 3/3 at `N_eff=200000`, but normalized causal mass was only about 46--47% in both settings; null controls retained 53--54%.
- The joint infinitesimal baseline assigned positive implied annotated heritability to the two causal contexts and near-zero or negative estimates to controls. Its total implied heritability is not calibrated because the truth is deliberately sparse whereas the baseline assumes infinitesimal annotation-wide effects; its signed residual LD-score estimate was negative.
- On the real AD summary statistics, the same joint baseline has small, mixed signed context coefficients and a negative residual term. It should be treated as a comparative annotation ranking rather than a calibrated AD heritability decomposition until standard LDSC weighting and intercept treatment are added.

### Official LDSC follow-up

- Added an official LDSC input writer that exports the gene-summed ABC annotations, an all-SNP baseline annotation, the matching UKBB `R2` LD scores, regression-weight LD scores, annotation totals, and AD summary statistics in the formats consumed by `ldsc.py --h2`.
- The official engine provides LDSC's native free intercept, iterative heteroskedastic weights, and block-jackknife uncertainty. This comparison is matched to the current UKBB panel and `r2 >= 0.01` representation; it is not a baseline-LD-adjusted analysis because compatible baseline-LD annotations have not yet been added.
- The current UKBB metadata do not provide a compatible allele-frequency file for overlap-annotation enrichment output. The official fit therefore reports its native joint coefficients, intercept, and block-jackknife errors without requesting the optional overlap-enrichment table; no synthetic frequencies are introduced.
- Official LDSC completed on 12,362,217 post-filter SNPs. Total observed-scale heritability was `0.016 (SE 0.002)`, with intercept `1.0155 (SE 0.0062)`. The joint bipolar-neuron estimate was `0.0028 (SE 0.0012; coefficient 1.21e-7; z about 2.3)`, CD14 monocyte was `0.0015 (SE 0.0010)`, and the three peripheral controls were not individually resolved. This is a matched UKBB-reference comparison, not a baseline-LD-adjusted published-style S-LDSC result.

### BaselineLD-adjusted official LDSC workflow

- Added a conventional official-LDSC workflow using the public 1000 Genomes Phase 3 EUR reference, BaselineLD v2.2 features, HapMap3 regression weights, and matching allele frequencies. The reference files are acquired reproducibly from the public Zenodo mirror of the LDSC distribution.
- Five continuous gene-summed ABC annotations are rebuilt in the exact 1000G BIM row order. Their LD scores are computed per chromosome with the official `ldsc.py --l2` implementation and a 1 cM LD window. All BaselineLD regression-SNP rows are retained, while LDSC's `M_5_50` totals use MAF above 0.05.
- The final `--h2` call combines the new ABC LD-score prefix with BaselineLD, uses official weights and frequencies, enables overlap-annotation accounting and coefficient output, and retains the native AD Z statistics with per-SNP known effective sample sizes.
- For LDSC's overlap-annotation accounting, the custom annotation files are linked alongside their corresponding custom LD-score files. This matches LDSC's prefix convention while keeping the reusable ABC annotation builder and its LD-score outputs separate.
- The BaselineLD-adjusted official LDSC run completed on 1,185,548 matched regression SNPs. Observed-scale $h^2$ was `0.0152 (SE 0.0022)`, with intercept `1.0159 (SE 0.0143)`. None of the five conditional ABC coefficients was resolved: bipolar neuron was `3.56e-8 (SE 5.91e-8; z=0.60)`, CD14 monocyte was `-1.01e-8 (SE 1.19e-7; z=-0.08)`, and the three controls were also imprecise. The unadjusted UKBB-reference bipolar result is therefore not robust to conventional BaselineLD adjustment.
- The legacy Python 2 LDSC source requires `bitarray`. Its prior extension contained AVX-512 instructions and failed in the genotype-reading `--l2` path on the available CPU nodes. `scripts/setup_portable_ldsc_py2.sh` rebuilds that dependency with a generic x86-64 target; the real chromosome-22 smoke calculation then completed in 23 seconds and wrote 15,633 regression-SNP LD scores.

### Brain peak S-LDSC follow-up

- Cloned `nottalexi/brain-cell-type-peak-files` at commit `c09bfe9` using SSH. Its hg19 peak calls cover NeuN neurons, PU1 microglia, Olig2 oligodendrocytes, and LHX2 astrocytes for ATAC-seq, H3K27ac, and H3K4me3.
- Added a separate binary-peak BaselineLD workflow. It fits all 12 assay-by-cell-type annotations jointly, preserving the exact 1000G BIM row order and the existing HapMap3/BaselineLD overlap-annotation procedure. It intentionally replaces, rather than combines with, the ABC annotations.
- The joint peak/BaselineLD fit completed on 1,185,548 matched regression SNPs with observed-scale `h2=0.0151 (SE 0.0022)` and intercept `1.0160 (SE 0.0146)`. PU1 microglial H3K27ac was the only custom annotation with a nominal conditional coefficient: `3.26e-8 (SE 1.47e-8; z=2.22; p=0.026)`. It covered 2.72% of SNPs, had 14.7-fold enrichment (`SE 3.82; p=1.36e-4`), and accounted for 40.0% of fitted h2 (`SE 10.4%`). Its 12-test Benjamini--Hochberg coefficient q-value is about 0.31; this is follow-up evidence, not a corrected discovery. The remaining 11 assay-by-cell-type annotations were unresolved.

## 2026-07-15 Glass Lab v2 brain ABC follow-up

- The Glass Lab v2 files provide four matched brain-cell predictions: astrocyte, microglia, neuron, and oligodendrocyte.
- Their prediction coordinates are GRCh38, whereas the AD GWAS, UKBB LD, and 1000 Genomes BaselineLD reference used here are hg19. `src/compass/glass_abc.py` therefore lifts the BED intervals through the GRCh38-to-hg19 chain before either method consumes them.
- `scripts/prepare_glass_abc_v2.py` emits a compact common ABC table containing only interval, target-gene, score, and cell-type fields. COMPASS and BaselineLD-adjusted S-LDSC use this identical table, eliminating an annotation-definition difference between the analyses.

### Annotation diagnostic

- The BaselineLD-adjusted continuous ABC analysis was null for all four Glass contexts: absolute coefficient z-scores were at most 1.59. The matched H3K27ac peak analysis had nominal PU1 microglial enrichment, so the discrepancy is not explained by reference panel, GWAS, or baseline-LD configuration.
- ABC support covers 0.59--0.73% of 1000 Genomes reference SNPs, compared with 2.80--5.29% for matched H3K27ac peaks. ABC-positive SNPs are nevertheless strongly enriched in their matched H3K27ac peaks (13.5--25.9-fold), which supports the coordinate conversion and biological relevance of the linked elements.
- Self-promoter records account for 86.4--86.9% of continuous ABC annotation mass because they receive ABC score one. BaselineLD already contains promoter annotations, so their conditional coefficient has limited opportunity to explain peak-associated AD signal.
- The distal ABC links are cell-specific rather than highly collinear: pairwise distal support Jaccard indices are approximately 0.001 and score correlations are near zero. Poor cell-type specificity is therefore not the primary explanation.
- Added S-LDSC sensitivity modes that respectively exclude self-promoters and binarize the ABC support. These isolate promoter dominance from score and multi-gene weighting; results remain pending.

## 2026-07-15 ATAC-to-expressed-TSS COMPASS sensitivity

- Added an all-row open-chromatin annotation source for COMPASS. It links each variant in the matched ATAC peak set to every expressed gene in the same cell type with a TSS within 100 kb.
- The per-cell expressed-gene sets come from the Glass ABC model's `TargetGeneIsExpressed` field, and their GRCh38 TSS positions are lifted to hg19 before overlap with the ATAC peaks and GWAS/LD reference.
- Contexts are LHX2 astrocyte, PU1 microglia, NeuN neuron, and Olig2 oligodendrocyte ATAC peaks. Variants outside peaks remain explicit all-zero model rows for correct LD accounting.

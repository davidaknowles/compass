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
- A small synthetic smoke test exercised `fit_nuclear_norm_path` with `cv_method=cre_ld_group`; it produced fold scores for folds `[0, 1]` and selected among the provided lambdas.

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

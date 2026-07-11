# COMPASS Sparse GPU Performance TODO

## Current full jobs

- [x] Cancel fp32 and fp16 model fits `18709275` and `18709276`: both were still in the unlogged CV loop after 3.6 days and could not finish before their time limits.
- [x] Cancel pre-objective-stop fp32 job `18756779`: it was spending 500 iterations on high-lambda plateaus.
- [x] Cancel fp16 job `18757013`: fp16 model-state underflow caused a non-finite convergence metric at high lambda.
- [x] Submit validated BF16 masked-row CRE-CV fit `18760205` with penalized-objective plateau stopping; it entered fold 0 cleanly.
- [x] Submit corrected fp32 comparison fit `18769161` after a local full-genome high-lambda test stopped at iteration 40.
- [x] Submit extended-grid FP32 fit `18773141`; it entered fold 0 with the automatic lower-bound extension code.

## Near-term engineering

- [x] Test whether torch CUDA sparse CSR supports int32 `crow_indices` and `col_indices`: L40S smoke job `18709337` passed for fp16 CSR matmul and backward with int32 and int64 indices.
- [x] Add an optional int32-index CUDA sparse conversion path and make it the default for LD: model-path L40S smoke job `18709338` reported `ld_index_dtype=int32`.
- [x] Add lightweight per-fit timing metadata for LD chunk count and time spent in LD sparse matmul/backward: model-path L40S smoke job `18709338` reported chunk counts and conversion/eval-backward seconds.
- [x] Tune `--ld-chunk-nnz` using short GPU smoke jobs before changing full-fit defaults: chr22 and chr6 benchmarks support `150,000,000` as the default speed/memory tradeoff.

## Later optimization

- [x] Profile representative and full-genome CUDA fits with cProfile: avoidable SciPy CSR row slicing dominated the old inner loop; CUDA sparse backward and CSR transfer are now the recurring costs.
- [x] Remove per-chunk SciPy submatrix copies using parent-array CSR row views and preserve fp16 LD values on the host.
- [x] Replace the frozen softplus residual coefficient with an exact non-negative weighted least-squares update per proximal iteration.
- [x] Migrate the current chromosome LD cache to non-destructive uncompressed archives and prefer them through `manifest.uncompressed.json`.
- [x] Replace copied train/test LD fold subsets with masked-row CRE CV over the full LD operator; the five-fold all-variant smoke test completed without OOM.
- [x] Keep torch CSR rather than a custom CUDA kernel or fp8: after the zero-copy change, sparse backward is the largest recurring cost and transfer is second; neither justifies the added implementation risk yet.

## LD-Component Cross-Validation

- [x] Measure connected components across `r2` thresholds `0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10` from the cached all-variant LD graph. The full sweep completed in 368 seconds with 19.5 GB peak RSS.
- [ ] Select `rho_CV` and replace distance-binned, per-gene CRE folds with global LD-component fold labels. At `r2 >= 0.01`, 3,354 components cover 12.36 million variants and the largest component is 0.282% of rows.

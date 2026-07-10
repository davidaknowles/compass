# COMPASS Sparse GPU Performance TODO

## Current full jobs

- [x] Cancel fp32 and fp16 model fits `18709275` and `18709276`: both were still in the unlogged CV loop after 3.6 days and could not finish before their time limits.
- [x] Submit the corrected full fp32 masked-row CRE-CV fit as job `18756779`; it loaded the uncompressed cache and entered fold 0 cleanly.
- [x] Submit matched fp16 model-parameter fit `18757013`; LD remains fp16 in both this and the fp32 comparison run.

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

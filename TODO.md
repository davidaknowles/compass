# COMPASS Sparse GPU Performance TODO

## Current full jobs

- [x] Monitor fp32 model fit job `18709275`: running in fit loop after cache load.
- [x] Monitor fp16 model fit job `18709276`: running in fit loop after cache load.
- [ ] Compare fp32 and fp16 fit outputs once both complete.

## Near-term engineering

- [x] Test whether torch CUDA sparse CSR supports int32 `crow_indices` and `col_indices`: L40S smoke job `18709337` passed for fp16 CSR matmul and backward with int32 and int64 indices.
- [x] Add an optional int32-index CUDA sparse conversion path and make it the default for LD: model-path L40S smoke job `18709338` reported `ld_index_dtype=int32`.
- [x] Add lightweight per-fit timing metadata for LD chunk count and time spent in LD sparse matmul/backward: model-path L40S smoke job `18709338` reported chunk counts and conversion/eval-backward seconds.
- [ ] Tune `--ld-chunk-nnz` using short GPU smoke jobs before changing full-fit defaults.

## Later optimization

- [ ] Profile full or representative chromosome chunks to identify whether runtime is LD SpMV, SVD/prox, CPU cache loading, or data transfer.
- [ ] Consider a custom CUDA LD SpMV kernel only if torch/cuSPARSE CSR remains the bottleneck after int32 indices and chunk tuning.
- [ ] Do not pursue fp8 LD values until profiling shows fp16 value bandwidth, rather than sparse indices or atomics, is the limiting factor.

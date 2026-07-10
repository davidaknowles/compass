#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from compass.ld import iter_csr_row_ranges, scipy_csr_rows_to_torch_sparse, scipy_to_torch_sparse


DEFAULT_DATA_ROOT = Path("/gpfs/commons/home/daknowles/knowles_lab/data/compass")


def _load_csr_npz(path: Path) -> sp.csr_matrix:
    if path.name.endswith(".scipy.npz"):
        return sp.load_npz(path)
    arrays = np.load(path, allow_pickle=False)
    return sp.csr_matrix(
        (arrays["data"], arrays["indices"], arrays["indptr"]),
        shape=tuple(arrays["shape"]),
    )


def _find_cache_prefix(cache_dir: Path) -> Path:
    manifests = sorted(
        cache_dir.glob("abc.allrows*.gwas.R2.chroms/manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise FileNotFoundError(f"No chromosome LD cache manifests found under {cache_dir}")
    r2_dir = manifests[0].parent
    return cache_dir / r2_dir.name.removesuffix(".gwas.R2.chroms")


def _chrom_entry(r2_dir: Path, chrom: int) -> dict:
    with open(r2_dir / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for item in manifest["blocks"]:
        if int(item["chrom"]) == int(chrom):
            return item
    raise ValueError(f"chromosome {chrom} not present in {r2_dir}")


def benchmark_chunk_size(
    A_block: sp.csr_matrix,
    R2: sp.csr_matrix,
    chisq: torch.Tensor,
    n_samples: torch.Tensor,
    chunk_nnz: int,
    device: str,
) -> dict:
    A_t = scipy_to_torch_sparse(A_block, device=device, dtype=torch.float32)
    B_flat = (torch.rand(A_block.shape[1], dtype=torch.float32, device=device) * 1e-6).requires_grad_(True)
    tau_raw = torch.tensor(-20.0, dtype=torch.float32, device=device, requires_grad=True)
    ld_score = torch.as_tensor(np.asarray(R2.sum(axis=1)).ravel(), dtype=torch.float32, device=device)
    weight = 1.0 / torch.clamp(chisq, min=1.0)

    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    convert_seconds = 0.0
    eval_backward_seconds = 0.0
    chunks = 0
    processed_nnz = 0
    start_total = perf_counter()
    for start, end in iter_csr_row_ranges(R2, chunk_nnz):
        chunks += 1
        processed_nnz += int(R2.indptr[end] - R2.indptr[start])
        start_convert = perf_counter()
        if torch.device(device).type == "cuda":
            R2_t = scipy_csr_rows_to_torch_sparse(
                R2,
                start,
                end,
                device=device,
                dtype=torch.float16,
                index_dtype=torch.int32,
            )
        else:
            R2_t = scipy_to_torch_sparse(
                R2[start:end],
                device=device,
                dtype=torch.float32,
                layout="coo",
            )
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        convert_seconds += perf_counter() - start_convert

        start_eval = perf_counter()
        mediated = torch.sparse.mm(A_t, B_flat.unsqueeze(1)).squeeze(1)
        smoothed = torch.sparse.mm(R2_t, mediated.to(R2_t.dtype).unsqueeze(1)).squeeze(1).float()
        pred = 1.0 + n_samples[start:end] * (smoothed + torch.nn.functional.softplus(tau_raw) * ld_score[start:end])
        residual = chisq[start:end] - pred
        loss = torch.sum(weight[start:end] * residual.square()) / max(R2.shape[0], 1)
        loss.backward()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        eval_backward_seconds += perf_counter() - start_eval
        del R2_t, mediated, smoothed, pred, residual, loss
    if torch.device(device).type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_mb = float("nan")
    return {
        "chunk_nnz": int(chunk_nnz),
        "chunks": chunks,
        "processed_nnz": processed_nnz,
        "convert_seconds": convert_seconds,
        "eval_backward_seconds": eval_backward_seconds,
        "total_seconds": perf_counter() - start_total,
        "peak_cuda_mb": peak_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CUDA LD row chunk sizes on a cached chromosome.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cache-prefix", default=None)
    parser.add_argument("--chrom", type=int, default=22)
    parser.add_argument("--chunk-nnz", default="25000000,50000000,100000000,200000000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    cache_prefix = Path(args.cache_prefix).expanduser() if args.cache_prefix else _find_cache_prefix(data_root / "cache")
    r2_dir = Path(str(cache_prefix) + ".gwas.R2.chroms")
    item = _chrom_entry(r2_dir, args.chrom)

    A = sp.load_npz(str(cache_prefix) + ".A.npz").tocsr().astype(np.float32)
    arrays = np.load(str(cache_prefix) + ".arrays.npz", allow_pickle=False)
    rows = np.load(r2_dir / item["rows"], allow_pickle=False)
    R2 = _load_csr_npz(r2_dir / item["matrix"]).astype(np.float32, copy=False)
    A_block = A[rows]
    chisq = torch.as_tensor(arrays["chisq"][rows], dtype=torch.float32, device=args.device)
    n_samples_array = arrays["n_samples"]
    if np.ndim(n_samples_array) == 0:
        n_samples = torch.full((rows.size,), float(n_samples_array), dtype=torch.float32, device=args.device)
    else:
        n_samples = torch.as_tensor(n_samples_array[rows], dtype=torch.float32, device=args.device)

    results = [
        benchmark_chunk_size(A_block, R2, chisq, n_samples, int(chunk), args.device)
        for chunk in args.chunk_nnz.split(",")
    ]
    table = pd.DataFrame(results)
    print(table.to_csv(sep="\t", index=False), end="")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    main()

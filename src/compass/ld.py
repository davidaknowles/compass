from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


REGION_LENGTH = 3_000_000
UKBB_LD_N = 337_545
GRCH37_AUTOSOME_LENGTHS = {
    1: 249_250_621,
    2: 243_199_373,
    3: 198_022_430,
    4: 191_154_276,
    5: 180_915_260,
    6: 171_115_067,
    7: 159_138_663,
    8: 146_364_022,
    9: 141_213_431,
    10: 135_534_747,
    11: 135_006_516,
    12: 133_851_895,
    13: 115_169_878,
    14: 107_349_540,
    15: 102_531_392,
    16: 90_354_753,
    17: 81_195_210,
    18: 78_077_248,
    19: 59_128_983,
    20: 63_025_520,
    21: 48_129_895,
    22: 51_304_566,
}


def ukbb_ld_block_stems(chromosomes: list[int] | None = None) -> list[tuple[int, int, int, str]]:
    chroms = sorted(GRCH37_AUTOSOME_LENGTHS) if chromosomes is None else [int(c) for c in chromosomes]
    blocks = []
    for chrom in chroms:
        for region_start in range(1, GRCH37_AUTOSOME_LENGTHS[chrom] + 1, REGION_LENGTH):
            region_end = region_start + REGION_LENGTH
            blocks.append((chrom, region_start, region_end, f"chr{chrom}_{region_start}_{region_end}"))
    return blocks


def _normalize_ld_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.rename(
        columns={
            "rsid": "SNP",
            "chromosome": "CHR",
            "position": "BP",
            "allele1": "A1",
            "allele2": "A2",
        },
        errors="ignore",
    )
    required = {"SNP", "CHR", "BP", "A1", "A2"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"LD metadata missing columns: {sorted(missing)}")
    meta = meta.copy()
    meta["CHR"] = pd.to_numeric(meta["CHR"], errors="raise").astype(np.int64)
    meta["BP"] = pd.to_numeric(meta["BP"], errors="raise").astype(np.int64)
    meta["variant_id"] = "chr" + meta["CHR"].astype(str) + ":" + meta["BP"].astype(str)
    meta["ld_row"] = np.arange(meta.shape[0], dtype=np.int64)
    return meta


def read_ukbb_ld_metadata(ld_dir: str, stem: str) -> pd.DataFrame:
    path = pd.io.common.stringify_path(ld_dir)
    meta_path = f"{path.rstrip('/')}/{stem}.gz"
    return _normalize_ld_metadata(pd.read_csv(meta_path, sep=r"\s+"))


def load_ukbb_ld_block_r2(
    ld_dir: str,
    stem: str,
    local_rows: np.ndarray | None = None,
    unbiased_n: int | None = None,
    r2_cutoff: float = 0.0,
    dtype: np.dtype = np.float32,
) -> sp.csr_matrix:
    """Load a UKBB LD block as sparse squared correlations.

    PolyFun stores triangular correlations. This returns a symmetric selected
    submatrix with diagonal one, then squares nonzero correlations.
    """

    path = pd.io.common.stringify_path(ld_dir)
    matrix = sp.load_npz(f"{path.rstrip('/')}/{stem}.npz").tocsr().astype(np.float32)
    if local_rows is not None:
        local_rows = np.asarray(local_rows, dtype=np.int64)
        matrix = matrix[local_rows][:, local_rows].tocsr()
    if r2_cutoff > 0:
        matrix.data[np.square(matrix.data) < float(r2_cutoff)] = 0.0
        matrix.eliminate_zeros()
    matrix = (matrix + matrix.T).tocsr()
    matrix.setdiag(1.0)
    matrix.eliminate_zeros()
    matrix.data = np.square(matrix.data, dtype=np.float32)
    if unbiased_n is not None:
        matrix.data *= (unbiased_n - 1) / (unbiased_n - 2)
        matrix.data -= 1 / (unbiased_n - 2)
        matrix.data = np.maximum(matrix.data, 0.0)
        matrix.eliminate_zeros()
    if r2_cutoff > 0:
        matrix.data[matrix.data < float(r2_cutoff)] = 0.0
        matrix.eliminate_zeros()
        matrix.setdiag(1.0)
        matrix.eliminate_zeros()
    matrix.data = matrix.data.astype(dtype, copy=False)
    return matrix


def _process_ukbb_ld_block(args):
    chrom, start, end, stem, block_variants, ld_dir, unbiased_n, r2_cutoff, dtype = args
    diagnostic = {
        "chrom": chrom,
        "region_start": start,
        "region_end": end,
        "stem": stem,
        "annotation_variants": int(block_variants.shape[0]),
        "matched_variants": 0,
        "missing_ld_files": False,
    }
    stem_path = Path(pd.io.common.stringify_path(ld_dir)) / stem
    if not stem_path.with_suffix(".gz").exists() or not stem_path.with_suffix(".npz").exists():
        diagnostic["missing_ld_files"] = True
        return diagnostic, None, None, None, np.array([], dtype=np.int64)

    meta = read_ukbb_ld_metadata(ld_dir, stem)
    by_variant_id = dict(zip(block_variants["variant_id"], block_variants["variant_idx"]))
    marker = block_variants.get("MarkerID")
    by_marker = {} if marker is None else dict(zip(marker.astype(str), block_variants["variant_idx"]))
    meta["target_variant_idx"] = meta["SNP"].astype(str).map(by_marker)
    missing_snp_match = meta["target_variant_idx"].isna()
    meta.loc[missing_snp_match, "target_variant_idx"] = meta.loc[missing_snp_match, "variant_id"].map(by_variant_id)
    hit = meta.dropna(subset=["target_variant_idx"]).copy()
    hit = hit.drop_duplicates("target_variant_idx")
    diagnostic["matched_variants"] = int(hit.shape[0])
    if hit.empty:
        return diagnostic, None, None, None, np.array([], dtype=np.int64)

    local_rows = hit["ld_row"].to_numpy(np.int64)
    global_rows = hit["target_variant_idx"].to_numpy(np.int64)
    if global_rows.size == 1:
        return (
            diagnostic,
            global_rows,
            global_rows,
            np.ones(1, dtype=np.float32),
            global_rows,
        )
    block_r2 = load_ukbb_ld_block_r2(
        ld_dir,
        stem,
        local_rows=local_rows,
        unbiased_n=unbiased_n,
        r2_cutoff=r2_cutoff,
        dtype=dtype,
    )
    coo = block_r2.tocoo()
    return (
        diagnostic,
        global_rows[coo.row],
        global_rows[coo.col],
        coo.data.astype(dtype, copy=False),
        global_rows,
    )


def _matched_ukbb_ld_variants_for_block(args):
    chrom, start, end, stem, block_variants, ld_dir = args
    stem_path = Path(pd.io.common.stringify_path(ld_dir)) / stem
    if not stem_path.with_suffix(".gz").exists():
        return np.array([], dtype=np.int64)
    meta = read_ukbb_ld_metadata(ld_dir, stem)
    by_variant_id = dict(zip(block_variants["variant_id"], block_variants["variant_idx"]))
    marker = block_variants.get("MarkerID")
    by_marker = {} if marker is None else dict(zip(marker.astype(str), block_variants["variant_idx"]))
    meta["target_variant_idx"] = meta["SNP"].astype(str).map(by_marker)
    missing_snp_match = meta["target_variant_idx"].isna()
    meta.loc[missing_snp_match, "target_variant_idx"] = meta.loc[missing_snp_match, "variant_id"].map(by_variant_id)
    hit = meta.dropna(subset=["target_variant_idx"]).drop_duplicates("target_variant_idx")
    return hit["target_variant_idx"].to_numpy(np.int64)


def filter_variants_to_ukbb_ld(
    variants: pd.DataFrame,
    ld_dir: str,
    chromosomes: list[int] | None = None,
    n_jobs: int = 1,
    progress_every: int = 0,
) -> pd.DataFrame:
    """Return variants represented in the UKBB LD metadata.

    Matching follows the same SNP-ID-first, chromosome-position-second logic as
    LD matrix assembly. The output preserves the input columns and resets
    ``variant_idx`` to dense row indices.
    """

    variants = variants.copy()
    variants["chrom"] = variants["chrom"].astype(np.int64)
    variants["pos"] = variants["pos"].astype(np.int64)
    if "variant_id" not in variants:
        variants["variant_id"] = "chr" + variants["chrom"].astype(str) + ":" + variants["pos"].astype(str)
    if "variant_idx" not in variants:
        variants["variant_idx"] = np.arange(variants.shape[0], dtype=np.int64)

    tasks = []
    for chrom, start, end, stem in ukbb_ld_block_stems(chromosomes):
        in_block = variants["chrom"].eq(chrom) & variants["pos"].ge(start) & variants["pos"].lt(end)
        block_variants = variants.loc[in_block]
        if not block_variants.empty:
            tasks.append((chrom, start, end, stem, block_variants.copy(), ld_dir))

    if n_jobs > 1 and len(tasks) > 1:
        matched_parts = []
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_matched_ukbb_ld_variants_for_block, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), start=1):
                matched_parts.append(future.result())
                if progress_every and (i % progress_every == 0 or i == len(futures)):
                    print(f"[setup] matched LD panel blocks {i}/{len(futures)}", flush=True)
    else:
        matched_parts = []
        for i, task in enumerate(tasks, start=1):
            matched_parts.append(_matched_ukbb_ld_variants_for_block(task))
            if progress_every and (i % progress_every == 0 or i == len(tasks)):
                print(f"[setup] matched LD panel blocks {i}/{len(tasks)}", flush=True)

    if matched_parts:
        matched_idx = np.unique(np.concatenate([part for part in matched_parts if part.size]))
    else:
        matched_idx = np.array([], dtype=np.int64)
    out = variants[variants["variant_idx"].isin(matched_idx)].copy()
    out["source_variant_idx"] = out["variant_idx"].to_numpy(np.int64)
    out = out.sort_values(["chrom", "pos", "variant_id"]).reset_index(drop=True)
    out["variant_idx"] = np.arange(out.shape[0], dtype=np.int64)
    return out


def build_ukbb_ld_r2(
    variants: pd.DataFrame,
    ld_dir: str,
    chromosomes: list[int] | None = None,
    add_identity_for_missing: bool = True,
    unbiased_n: int | None = None,
    r2_cutoff: float = 0.0,
    dtype: np.dtype = np.float32,
    n_jobs: int = 1,
    progress_every: int = 0,
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """Assemble UKBB LD R2 over COMPASS annotation variants.

    Variants are matched by SNP ID first and chromosome/position second.
    The returned diagnostics table records per-block match counts and missing
    files fail immediately.
    """

    n = variants.shape[0]
    variants = variants.copy()
    variants["chrom"] = variants["chrom"].astype(np.int64)
    variants["pos"] = variants["pos"].astype(np.int64)
    if "variant_id" not in variants:
        variants["variant_id"] = "chr" + variants["chrom"].astype(str) + ":" + variants["pos"].astype(str)
    blocks = ukbb_ld_block_stems(chromosomes)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    matched = np.zeros(n, dtype=bool)
    diagnostics: list[dict] = []
    tasks = []

    for chrom, start, end, stem in blocks:
        in_block = variants["chrom"].eq(chrom) & variants["pos"].ge(start) & variants["pos"].lt(end)
        block_variants = variants.loc[in_block]
        if block_variants.empty:
            continue
        tasks.append((chrom, start, end, stem, block_variants.copy(), ld_dir, unbiased_n, r2_cutoff, dtype))

    if n_jobs > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_process_ukbb_ld_block, task) for task in tasks]
            results = []
            for i, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if progress_every and (i % progress_every == 0 or i == len(futures)):
                    print(f"[setup] processed LD blocks {i}/{len(futures)}", flush=True)
    else:
        results = []
        for i, task in enumerate(tasks, start=1):
            results.append(_process_ukbb_ld_block(task))
            if progress_every and (i % progress_every == 0 or i == len(tasks)):
                print(f"[setup] processed LD blocks {i}/{len(tasks)}", flush=True)

    for diagnostic, row, col, val, matched_idx in results:
        diagnostics.append(diagnostic)
        if row is not None and col is not None and val is not None:
            rows.append(row)
            cols.append(col)
            vals.append(val)
        if matched_idx.size:
            matched[matched_idx] = True

    if add_identity_for_missing:
        missing = np.flatnonzero(~matched)
        if missing.size:
            rows.append(missing.astype(np.int64))
            cols.append(missing.astype(np.int64))
            vals.append(np.ones(missing.size, dtype=dtype))

    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        val = np.concatenate(vals)
    else:
        row = col = np.array([], dtype=np.int64)
        val = np.array([], dtype=dtype)
    r2 = sp.coo_matrix((val, (row, col)), shape=(n, n), dtype=dtype).tocsr()
    r2.sum_duplicates()
    diag = r2.diagonal()
    missing_diag = np.flatnonzero(diag == 0)
    if missing_diag.size:
        r2[missing_diag, missing_diag] = 1.0
    return r2, pd.DataFrame(diagnostics)


def build_ukbb_ld_r2_by_chromosome(
    variants: pd.DataFrame,
    ld_dir: str,
    chromosomes: list[int] | None = None,
    add_identity_for_missing: bool = True,
    unbiased_n: int | None = None,
    r2_cutoff: float = 0.01,
    dtype: np.dtype = np.float32,
    n_jobs: int = 1,
    progress_every: int = 0,
) -> tuple[list[dict], pd.DataFrame]:
    """Assemble thresholded UKBB LD R2 as chromosome-level sparse blocks."""

    variants = variants.copy()
    variants["chrom"] = variants["chrom"].astype(np.int64)
    variants["pos"] = variants["pos"].astype(np.int64)
    if "variant_id" not in variants:
        variants["variant_id"] = "chr" + variants["chrom"].astype(str) + ":" + variants["pos"].astype(str)
    if "variant_idx" not in variants:
        variants["variant_idx"] = np.arange(variants.shape[0], dtype=np.int64)

    chroms = sorted(variants["chrom"].unique().astype(int).tolist()) if chromosomes is None else [int(c) for c in chromosomes]
    all_blocks: list[dict] = []
    all_diagnostics: list[pd.DataFrame] = []
    for chrom in chroms:
        chrom_variants = variants.loc[variants["chrom"].eq(chrom)].copy()
        if chrom_variants.empty:
            continue
        chrom_rows = chrom_variants["variant_idx"].to_numpy(np.int64)
        local_lookup = np.full(int(variants["variant_idx"].max()) + 1, -1, dtype=np.int64)
        local_lookup[chrom_rows] = np.arange(chrom_rows.size, dtype=np.int64)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        matched = np.zeros(chrom_rows.size, dtype=bool)
        diagnostics: list[dict] = []
        tasks = []
        for block_chrom, start, end, stem in ukbb_ld_block_stems([chrom]):
            in_block = chrom_variants["pos"].ge(start) & chrom_variants["pos"].lt(end)
            block_variants = chrom_variants.loc[in_block]
            if not block_variants.empty:
                tasks.append((block_chrom, start, end, stem, block_variants.copy(), ld_dir, unbiased_n, r2_cutoff, dtype))

        if n_jobs > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                futures = [pool.submit(_process_ukbb_ld_block, task) for task in tasks]
                results = []
                for i, future in enumerate(as_completed(futures), start=1):
                    results.append(future.result())
                    if progress_every and (i % progress_every == 0 or i == len(futures)):
                        print(f"[setup] processed chr{chrom} LD blocks {i}/{len(futures)}", flush=True)
        else:
            results = []
            for i, task in enumerate(tasks, start=1):
                results.append(_process_ukbb_ld_block(task))
                if progress_every and (i % progress_every == 0 or i == len(tasks)):
                    print(f"[setup] processed chr{chrom} LD blocks {i}/{len(tasks)}", flush=True)

        for diagnostic, row, col, val, matched_idx in results:
            diagnostics.append(diagnostic)
            if row is not None and col is not None and val is not None:
                rows.append(local_lookup[row])
                cols.append(local_lookup[col])
                vals.append(val.astype(dtype, copy=False))
            if matched_idx.size:
                matched_local = local_lookup[matched_idx]
                matched_local = matched_local[matched_local >= 0]
                matched[matched_local] = True

        if add_identity_for_missing:
            missing = np.flatnonzero(~matched)
            if missing.size:
                rows.append(missing.astype(np.int64))
                cols.append(missing.astype(np.int64))
                vals.append(np.ones(missing.size, dtype=dtype))

        if rows:
            row = np.concatenate(rows)
            col = np.concatenate(cols)
            val = np.concatenate(vals)
        else:
            row = col = np.array([], dtype=np.int64)
            val = np.array([], dtype=dtype)
        r2 = sp.coo_matrix((val, (row, col)), shape=(chrom_rows.size, chrom_rows.size), dtype=dtype).tocsr()
        r2.sum_duplicates()
        diag = r2.diagonal()
        missing_diag = np.flatnonzero(diag == 0)
        if missing_diag.size:
            r2[missing_diag, missing_diag] = np.asarray(1.0, dtype=dtype)
        all_blocks.append({"chrom": chrom, "rows": chrom_rows, "R2": r2})
        all_diagnostics.append(pd.DataFrame(diagnostics))
    diagnostics_df = pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame()
    return all_blocks, diagnostics_df


def annotation_triples_to_csr(triples: pd.DataFrame, n_variants: int, n_genes: int, n_mechanisms: int) -> sp.csr_matrix:
    """Flatten variant-gene-mechanism triples into A[variant, gene * L + mechanism]."""

    row = triples["variant_idx"].to_numpy(np.int64)
    col = (
        triples["gene_idx"].to_numpy(np.int64) * n_mechanisms
        + triples["mechanism_idx"].to_numpy(np.int64)
    )
    val = triples["value"].to_numpy(np.float32)
    return sp.coo_matrix((val, (row, col)), shape=(n_variants, n_genes * n_mechanisms)).tocsr()


def scipy_to_torch_sparse(
    matrix: sp.spmatrix,
    device: str = "cpu",
    dtype=None,
    layout: str = "coo",
    index_dtype=None,
):
    import torch

    if dtype is None:
        dtype = torch.float32
    if index_dtype is None:
        index_dtype = torch.long
    if layout == "csr":
        csr = matrix.tocsr()
        return torch.sparse_csr_tensor(
            torch.as_tensor(csr.indptr, dtype=index_dtype, device=device),
            torch.as_tensor(csr.indices, dtype=index_dtype, device=device),
            torch.as_tensor(csr.data, dtype=dtype, device=device),
            size=csr.shape,
            device=device,
        )
    if layout == "coo":
        coo = matrix.tocoo()
        indices = np.vstack([coo.row, coo.col])
        return torch.sparse_coo_tensor(
            torch.as_tensor(indices, dtype=torch.long, device=device),
            torch.as_tensor(coo.data, dtype=dtype, device=device),
            size=coo.shape,
            device=device,
        ).coalesce()
    raise ValueError(f"Unknown torch sparse layout: {layout}")

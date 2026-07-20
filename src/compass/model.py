from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.optimize import nnls
import torch

from .ld import iter_csr_row_ranges, scipy_csr_rows_to_torch_sparse, scipy_to_torch_sparse


@dataclass
class LdChromosomeBlock:
    chrom: int
    rows: np.ndarray
    R2: sp.csr_matrix


@dataclass
class CompassDataset:
    A: sp.csr_matrix
    chisq: np.ndarray
    chrom: np.ndarray
    n_samples: float | np.ndarray
    position: np.ndarray | None = None
    R2: sp.csr_matrix | None = None
    ld_blocks: list[LdChromosomeBlock] | None = None
    sample_weight: np.ndarray | None = None
    cv_groups: np.ndarray | None = None
    cv_score_groups: np.ndarray | None = None
    context_annotations: sp.csr_matrix | None = None

    @property
    def n_variants(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_params(self) -> int:
        return int(self.A.shape[1])


@dataclass
class FitResult:
    method: str
    lambdas: list[float]
    cv_scores: dict[float, float] | None
    best_lambda: float
    B: np.ndarray
    tau: float
    losses: list[float]
    metadata: dict
    context_effects: np.ndarray | None = None
    context_effect_se: np.ndarray | None = None


def _weights(chisq: torch.Tensor, weights: np.ndarray | None, device: str) -> torch.Tensor:
    if weights is None:
        # Simple LDSC-like stabilizer: downweight extremely large chi-square values.
        return (1.0 / torch.clamp(chisq, min=1.0)).to(device)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _flatten_B(B: torch.Tensor) -> torch.Tensor:
    return B.reshape(-1)


def aggregate_context_annotations(
    annotation: sp.csr_matrix,
    n_genes: int,
    n_mechanisms: int,
    mode: str = "binary",
    excluded_mechanisms: set[int] | None = None,
) -> sp.csr_matrix:
    """Aggregate gene-level annotations into one direct annotation per context."""

    if annotation.shape[1] != n_genes * n_mechanisms:
        raise ValueError("annotation shape does not match gene/mechanism dimensions")
    if mode not in {"binary", "sum"}:
        raise ValueError("context annotation mode must be 'binary' or 'sum'")
    excluded = set() if excluded_mechanisms is None else set(excluded_mechanisms)
    columns = []
    for mechanism in range(n_mechanisms):
        if mechanism in excluded:
            columns.append(sp.csr_matrix((annotation.shape[0], 1), dtype=np.float32))
            continue
        gene_columns = np.arange(mechanism, annotation.shape[1], n_mechanisms, dtype=np.int64)
        values = np.asarray(annotation[:, gene_columns].sum(axis=1)).ravel().astype(np.float32)
        if mode == "binary":
            values = (values > 0).astype(np.float32)
        columns.append(sp.csr_matrix(values[:, None]))
    return sp.hstack(columns, format="csr", dtype=np.float32)


def _relative_change(next_value: torch.Tensor, current_value: torch.Tensor) -> torch.Tensor:
    """Compute a stable relative update norm even when model parameters are fp16."""

    return torch.linalg.vector_norm((next_value - current_value).float()) / (
        torch.linalg.vector_norm(current_value.float()) + 1e-8
    )


def _samples_tensor(n_samples: float | np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(n_samples, dtype=torch.float32, device=device)


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {name}")


def predict_factorized(
    B: torch.Tensor,
    tau: torch.Tensor,
    A_t: torch.Tensor,
    R2_t: torch.Tensor,
    ld_score: torch.Tensor,
    n_samples: float | torch.Tensor,
) -> torch.Tensor:
    mediated = torch.sparse.mm(A_t, _flatten_B(B).to(A_t.dtype).unsqueeze(1)).squeeze(1)
    smoothed = torch.sparse.mm(R2_t, mediated.to(R2_t.dtype).unsqueeze(1)).squeeze(1).float()
    return 1.0 + n_samples * (smoothed + tau.float() * ld_score)


def _iter_ld_blocks(dataset: CompassDataset):
    if dataset.ld_blocks is not None:
        yield from dataset.ld_blocks
        return
    if dataset.R2 is None:
        raise ValueError("CompassDataset requires R2 or ld_blocks")
    yield LdChromosomeBlock(
        chrom=-1,
        rows=np.arange(dataset.n_variants, dtype=np.int64),
        R2=dataset.R2,
    )


def _materialize_block_r2(block: LdChromosomeBlock) -> sp.csr_matrix:
    return block.R2.tocsr()


def _ld_torch_layout(device: str) -> str:
    return "csr" if torch.device(device).type == "cuda" else "coo"


def _ld_index_dtype(device: str):
    return torch.int32 if torch.device(device).type == "cuda" else torch.long


def _prepare_fit_blocks(
    dataset: CompassDataset,
    device: str,
) -> list[dict]:
    A = dataset.A.astype(np.float32)
    block_specs = []
    for block in _iter_ld_blocks(dataset):
        rows = np.asarray(block.rows, dtype=np.int64)
        R2_block = _materialize_block_r2(block)
        if torch.device(device).type != "cuda" and R2_block.dtype == np.float16:
            # SciPy cannot slice or multiply CSR matrices with float16 values.
            # Keep the production CUDA path fp16, but make the CPU fallback valid.
            R2_block = sp.csr_matrix(
                (R2_block.data.astype(np.float32), R2_block.indices, R2_block.indptr),
                shape=R2_block.shape,
            )
        A_block = A[rows]
        chisq_block = torch.as_tensor(dataset.chisq[rows], dtype=torch.float32, device=device)
        weight_block = _weights(
            chisq_block,
            None if dataset.sample_weight is None else dataset.sample_weight[rows],
            device,
        )
        n_samples_block = _samples_tensor(
            np.asarray(dataset.n_samples)[rows] if np.ndim(dataset.n_samples) > 0 else dataset.n_samples,
            device,
        )
        block_spec = {
                "chrom": block.chrom,
                "rows": rows,
                "A_t": scipy_to_torch_sparse(A_block, device=device, dtype=torch.float32),
                "R2_cpu": R2_block,
                "R2_nnz": int(R2_block.nnz),
                "chisq": chisq_block,
                "weight": weight_block,
                "ld_score": torch.as_tensor(np.asarray(R2_block.sum(axis=1)).ravel(), dtype=torch.float32, device=device),
                "n_samples": n_samples_block,
                "cv_groups": (
                    None
                    if dataset.cv_groups is None
                    else torch.as_tensor(dataset.cv_groups[rows], dtype=torch.int64, device=device)
                ),
                "cv_score_groups": (
                    None
                    if dataset.cv_score_groups is None
                    else torch.as_tensor(dataset.cv_score_groups[rows], dtype=torch.int64, device=device)
                ),
            }
        if dataset.context_annotations is not None:
            context_block = dataset.context_annotations[rows].astype(np.float32)
            if context_block.shape[1] == 0:
                raise ValueError("context_annotations must have at least one column")
            if torch.device(device).type == "cuda":
                context_t = torch.as_tensor(context_block.toarray(), dtype=torch.float16, device=device)
                context_ld_chunks = []
                for start, end in iter_csr_row_ranges(R2_block, 150_000_000):
                    R2_t = scipy_csr_rows_to_torch_sparse(
                        R2_block,
                        start,
                        end,
                        device=device,
                        dtype=torch.float16,
                        index_dtype=_ld_index_dtype(device),
                    )
                    context_ld_chunks.append(torch.sparse.mm(R2_t, context_t).float())
                    del R2_t
                block_spec["context_ld"] = torch.cat(context_ld_chunks, dim=0)
                del context_t, context_ld_chunks
            else:
                block_spec["context_ld"] = torch.as_tensor(
                    R2_block @ context_block.toarray(), dtype=torch.float32, device=device
                )
        block_specs.append(block_spec)
    return block_specs


def _block_r2_cpu(block_spec: dict) -> sp.csr_matrix:
    return block_spec["R2_cpu"]


def _slice_vector(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if x.ndim == 0:
        return x
    return x[start:end]


def _backward_data_loss(
    block_specs: list[dict],
    B: torch.Tensor,
    tau: torch.Tensor,
    device: str,
    total_den: int,
    ld_chunk_nnz: int | None,
    stats: dict[str, float],
) -> tuple[float, float, float]:
    loss_value = 0.0
    tau_numerator = 0.0
    tau_denominator = 0.0
    ld_layout = _ld_torch_layout(device)
    ld_index_dtype = _ld_index_dtype(device)
    for block in block_specs:
        R2_cpu = _block_r2_cpu(block)
        for start, end in iter_csr_row_ranges(R2_cpu, ld_chunk_nnz):
            stats["ld_chunks"] += 1
            stats["ld_chunk_nnz_total"] += int(R2_cpu.indptr[end] - R2_cpu.indptr[start])
            chunk_start = perf_counter()
            if ld_layout == "csr":
                R2_t = scipy_csr_rows_to_torch_sparse(
                    R2_cpu,
                    start,
                    end,
                    device=device,
                    dtype=torch.float16,
                    index_dtype=ld_index_dtype,
                )
            else:
                R2_t = scipy_to_torch_sparse(
                    R2_cpu[start:end],
                    device=device,
                    dtype=torch.float32,
                    layout=ld_layout,
                    index_dtype=ld_index_dtype,
                )
            stats["ld_sparse_convert_seconds"] += perf_counter() - chunk_start
            eval_start = perf_counter()
            pred = predict_factorized(
                B,
                tau,
                block["A_t"],
                R2_t,
                block["ld_score"][start:end],
                _slice_vector(block["n_samples"], start, end),
            )
            residual = block["chisq"][start:end] - pred
            weight = block["weight"][start:end]
            ld_term = _slice_vector(block["n_samples"], start, end) * block["ld_score"][start:end]
            loss_block = torch.sum(weight * residual.square()) / max(total_den, 1)
            loss_value += float(loss_block.detach().cpu())
            # The residual LD term is a non-negative scalar weighted least-squares
            # coefficient. Accumulate its exact coordinate update while evaluating
            # the data loss, so selecting tau requires no extra genome-wide pass.
            tau_numerator += float(torch.sum(weight * ld_term * (residual + ld_term * tau)).detach().cpu())
            tau_denominator += float(torch.sum(weight * ld_term.square()).detach().cpu())
            loss_block.backward()
            if torch.device(device).type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            stats["ld_eval_backward_seconds"] += perf_counter() - eval_start
            del R2_t, pred, residual, loss_block, ld_term
        del R2_cpu
    return loss_value, tau_numerator, tau_denominator


def _solve_nonnegative_normal(normal: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a small non-negative least-squares problem from normal equations."""

    normal = np.asarray(normal, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    scale = np.sqrt(np.maximum(np.diag(normal), np.finfo(np.float64).tiny))
    scaled_normal = normal / np.outer(scale, scale)
    scaled_rhs = rhs / scale
    scaled_normal = 0.5 * (scaled_normal + scaled_normal.T)
    eigenvalues, eigenvectors = np.linalg.eigh(scaled_normal)
    keep = eigenvalues > max(float(eigenvalues.max(initial=0.0)) * 1e-10, 1e-12)
    if not np.any(keep):
        return np.zeros_like(rhs)
    roots = np.sqrt(eigenvalues[keep])
    design = roots[:, None] * eigenvectors[:, keep].T
    target = (eigenvectors[:, keep].T @ scaled_rhs) / roots
    scaled_solution, _ = nnls(design, target)
    return scaled_solution / scale


def _backward_hierarchical_data_loss(
    block_specs: list[dict],
    B: torch.Tensor,
    context_effects: torch.Tensor,
    tau: torch.Tensor,
    device: str,
    total_den: int,
    ld_chunk_nnz: int | None,
    stats: dict[str, float],
    fixed_context_effects: bool = False,
    context_profile: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict]:
    """Evaluate gene deviations and update context/residual effects by NNLS."""

    loss_value = 0.0
    n_contexts = int(context_effects.numel())
    normal = np.zeros((n_contexts + 1, n_contexts + 1), dtype=np.float64)
    rhs = np.zeros(n_contexts + 1, dtype=np.float64)
    block_normals = []
    block_rhs = []
    current = torch.cat((context_effects.float(), tau.float().reshape(1)))
    ld_layout = _ld_torch_layout(device)
    ld_index_dtype = _ld_index_dtype(device)
    for block in block_specs:
        if "context_ld" not in block:
            raise ValueError("hierarchical fitting requires context annotations")
        R2_cpu = _block_r2_cpu(block)
        block_normal = np.zeros_like(normal)
        block_target = np.zeros_like(rhs)
        mediated = torch.sparse.mm(
            block["A_t"], _flatten_B(B).to(block["A_t"].dtype).unsqueeze(1)
        ).squeeze(1)
        row_ranges = list(iter_csr_row_ranges(R2_cpu, ld_chunk_nnz))
        for chunk_index, (start, end) in enumerate(row_ranges):
            stats["ld_chunks"] += 1
            stats["ld_chunk_nnz_total"] += int(R2_cpu.indptr[end] - R2_cpu.indptr[start])
            chunk_start = perf_counter()
            if ld_layout == "csr":
                R2_t = scipy_csr_rows_to_torch_sparse(
                    R2_cpu,
                    start,
                    end,
                    device=device,
                    dtype=torch.float16,
                    index_dtype=ld_index_dtype,
                )
            else:
                R2_t = scipy_to_torch_sparse(
                    R2_cpu[start:end],
                    device=device,
                    dtype=torch.float32,
                    layout=ld_layout,
                    index_dtype=ld_index_dtype,
                )
            stats["ld_sparse_convert_seconds"] += perf_counter() - chunk_start
            eval_start = perf_counter()
            smoothed = torch.sparse.mm(
                R2_t, mediated.to(R2_t.dtype).unsqueeze(1)
            ).squeeze(1).float()
            n_samples = _slice_vector(block["n_samples"], start, end)
            context_design = block["context_ld"][start:end] * (
                n_samples if n_samples.ndim == 0 else n_samples.unsqueeze(1)
            )
            ld_term = n_samples * block["ld_score"][start:end]
            design = torch.cat((context_design, ld_term.unsqueeze(1)), dim=1)
            gene_prediction = 1.0 + n_samples * smoothed
            pred = gene_prediction + design @ current
            residual = block["chisq"][start:end] - pred
            weight = block["weight"][start:end]
            loss_block = torch.sum(weight * residual.square()) / max(total_den, 1)
            loss_value += float(loss_block.detach().cpu())
            target_without_context = residual + design @ current
            weighted_design = design * weight.unsqueeze(1)
            chunk_normal = (design.T @ weighted_design).detach().cpu().numpy()
            chunk_target = (design.T @ (weight * target_without_context)).detach().cpu().numpy()
            normal += chunk_normal
            rhs += chunk_target
            block_normal += chunk_normal
            block_target += chunk_target
            loss_block.backward(retain_graph=chunk_index + 1 < len(row_ranges))
            if torch.device(device).type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            stats["ld_eval_backward_seconds"] += perf_counter() - eval_start
            del R2_t, smoothed, context_design, ld_term, design, pred, residual, loss_block
        del mediated, R2_cpu, row_ranges
        block_normals.append(block_normal)
        block_rhs.append(block_target)
    if context_profile is not None:
        profile = np.asarray(context_profile, dtype=np.float64)
        context_normal = normal[:-1, :-1]
        context_residual = normal[:-1, -1]
        profile_normal = np.array(
            [
                [profile @ context_normal @ profile, profile @ context_residual],
                [context_residual @ profile, normal[-1, -1]],
            ]
        )
        profile_rhs = np.array([profile @ rhs[:-1], rhs[-1]])
        scale_tau = _solve_nonnegative_normal(profile_normal, profile_rhs)
        next_effects = np.concatenate((profile * scale_tau[0], [scale_tau[1]]))
    elif fixed_context_effects:
        fixed = context_effects.detach().float().cpu().numpy().astype(np.float64)
        tau_numerator = rhs[-1] - normal[-1, :-1] @ fixed
        tau_next = max(0.0, tau_numerator / normal[-1, -1]) if normal[-1, -1] > 0 else 0.0
        next_effects = np.concatenate((fixed, [tau_next]))
    else:
        next_effects = _solve_nonnegative_normal(normal, rhs)
    current_np = current.detach().cpu().numpy()
    old_quadratic = float(current_np @ normal @ current_np - 2 * current_np @ rhs)
    new_quadratic = float(next_effects @ normal @ next_effects - 2 * next_effects @ rhs)
    loss_value += (new_quadratic - old_quadratic) / max(total_den, 1)
    if context_profile is not None:
        leave_one_out = []
        profile = np.asarray(context_profile, dtype=np.float64)
        for block_normal, block_target in zip(block_normals, block_rhs):
            leave_normal = normal - block_normal
            leave_rhs = rhs - block_target
            context_normal = leave_normal[:-1, :-1]
            context_residual = leave_normal[:-1, -1]
            profile_normal = np.array(
                [
                    [profile @ context_normal @ profile, profile @ context_residual],
                    [context_residual @ profile, leave_normal[-1, -1]],
                ]
            )
            profile_rhs = np.array([profile @ leave_rhs[:-1], leave_rhs[-1]])
            scale_tau = _solve_nonnegative_normal(profile_normal, profile_rhs)
            leave_one_out.append(np.concatenate((profile * scale_tau[0], [scale_tau[1]])))
        leave_one_out = np.vstack(leave_one_out)
    elif fixed_context_effects:
        leave_one_out = []
        fixed = next_effects[:-1]
        for block_normal, block_target in zip(block_normals, block_rhs):
            leave_normal = normal - block_normal
            leave_rhs = rhs - block_target
            numerator = leave_rhs[-1] - leave_normal[-1, :-1] @ fixed
            leave_tau = max(0.0, numerator / leave_normal[-1, -1]) if leave_normal[-1, -1] > 0 else 0.0
            leave_one_out.append(np.concatenate((fixed, [leave_tau])))
        leave_one_out = np.vstack(leave_one_out)
    else:
        leave_one_out = np.vstack(
            [
                _solve_nonnegative_normal(normal - block_normal, rhs - block_target)
                for block_normal, block_target in zip(block_normals, block_rhs)
            ]
        )
    jackknife_mean = leave_one_out.mean(axis=0)
    jackknife_se = np.sqrt(
        (leave_one_out.shape[0] - 1)
        / leave_one_out.shape[0]
        * np.square(leave_one_out - jackknife_mean).sum(axis=0)
    )
    diagnostics = {
        "context_effect_se": jackknife_se[:-1],
        "tau_se": float(jackknife_se[-1]),
        "jackknife_blocks": int(leave_one_out.shape[0]),
    }
    return loss_value, next_effects, diagnostics


def _evaluate_fit_mse(
    block_specs: list[dict],
    B: np.ndarray,
    tau: float,
    device: str,
    ld_chunk_nnz: int | None,
    fold: int | None = None,
    context_effects: np.ndarray | None = None,
) -> float:
    """Evaluate an LD-component fold without rebuilding LD subsets."""

    B_t = torch.as_tensor(B, dtype=torch.float32, device=device)
    tau_t = torch.tensor(float(tau), dtype=torch.float32, device=device)
    context_t = None if context_effects is None else torch.as_tensor(context_effects, dtype=torch.float32, device=device)
    total_squared_error = 0.0
    total_rows = 0
    ld_layout = _ld_torch_layout(device)
    ld_index_dtype = _ld_index_dtype(device)
    with torch.no_grad():
        for block in block_specs:
            R2_cpu = _block_r2_cpu(block)
            mediated = torch.sparse.mm(
                block["A_t"], _flatten_B(B_t).to(block["A_t"].dtype).unsqueeze(1)
            ).squeeze(1)
            for start, end in iter_csr_row_ranges(R2_cpu, ld_chunk_nnz):
                if ld_layout == "csr":
                    R2_t = scipy_csr_rows_to_torch_sparse(
                        R2_cpu,
                        start,
                        end,
                        device=device,
                        dtype=torch.float16,
                        index_dtype=ld_index_dtype,
                    )
                else:
                    R2_t = scipy_to_torch_sparse(
                        R2_cpu[start:end],
                        device=device,
                        dtype=torch.float32,
                        layout=ld_layout,
                        index_dtype=ld_index_dtype,
                    )
                smoothed = torch.sparse.mm(R2_t, mediated.to(R2_t.dtype).unsqueeze(1)).squeeze(1).float()
                pred = 1.0 + _slice_vector(block["n_samples"], start, end) * (
                    smoothed + tau_t * block["ld_score"][start:end]
                )
                if context_t is not None:
                    pred = pred + _slice_vector(block["n_samples"], start, end) * (
                        block["context_ld"][start:end] @ context_t
                    )
                residual = block["chisq"][start:end] - pred
                if fold is None:
                    total_squared_error += float(torch.sum(residual.square()).cpu())
                    total_rows += end - start
                else:
                    score_groups = block["cv_score_groups"]
                    if score_groups is None:
                        raise ValueError("LD-component CV requires cv_score_groups in prepared blocks")
                    held_out = score_groups[start:end].eq(fold)
                    total_squared_error += float(torch.sum(residual.square()[held_out]).cpu())
                    total_rows += int(held_out.sum().cpu())
                del R2_t, smoothed, pred, residual
            del mediated
    return total_squared_error / max(total_rows, 1)


def _randomized_svd(
    B: torch.Tensor,
    rank: int,
    n_oversamples: int = 5,
    n_iter: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = max(1, min(rank, min(B.shape)))
    q = min(min(B.shape), rank + max(0, n_oversamples))
    omega = torch.randn((B.shape[1], q), dtype=B.dtype, device=B.device)
    y = B @ omega
    for _ in range(n_iter):
        y = B @ (B.T @ y)
    q_mat, _ = torch.linalg.qr(y, mode="reduced")
    small = q_mat.T @ B
    u_small, s, vh = torch.linalg.svd(small, full_matrices=False)
    u = q_mat @ u_small
    return u[:, :rank], s[:rank], vh[:rank]


def _svt_from_gram(B: torch.Tensor, threshold: float) -> torch.Tensor:
    gram = B.T @ B
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = torch.clamp(eigenvalues[order], min=0.0)
    eigenvectors = eigenvectors[:, order]
    singular_values = torch.sqrt(eigenvalues)
    shrink = torch.clamp(singular_values - threshold, min=0.0)
    scale = torch.where(singular_values > 0, shrink / singular_values, torch.zeros_like(shrink))
    return (B @ eigenvectors * scale.unsqueeze(0)) @ eigenvectors.T


def nuclear_prox_nonnegative(
    B: torch.Tensor,
    threshold: float,
    svd_method: str = "auto",
    svd_rank: int | None = None,
    svd_oversamples: int = 5,
    svd_n_iter: int = 2,
) -> torch.Tensor:
    with torch.no_grad():
        B = torch.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
        min_dim = min(B.shape)
        use_randomized = svd_method == "randomized" or (
            svd_method == "auto" and min_dim > 64 and svd_rank is not None and svd_rank < min_dim
        )
        if svd_method not in {"auto", "exact", "randomized"}:
            raise ValueError(f"Unknown svd_method: {svd_method}")
        if not use_randomized and B.shape[0] >= B.shape[1]:
            out = _svt_from_gram(B, threshold)
            return torch.clamp(torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
        elif use_randomized:
            U, S, Vh = _randomized_svd(
                B,
                rank=min_dim if svd_rank is None else svd_rank,
                n_oversamples=svd_oversamples,
                n_iter=svd_n_iter,
            )
        else:
            U, S, Vh = torch.linalg.svd(B, full_matrices=False)
        S = torch.clamp(S - threshold, min=0.0)
        out = (U * S.unsqueeze(0)) @ Vh
        return torch.clamp(torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)


def fit_nuclear_norm(
    dataset: CompassDataset,
    n_genes: int,
    n_mechanisms: int,
    lambda_value: float,
    init_B: np.ndarray | None = None,
    init_tau: float = 1e-8,
    lr: float = 1e-2,
    max_iter: int = 500,
    tol: float = 1e-6,
    device: str = "cpu",
    svd_method: str = "auto",
    svd_rank: int | None = None,
    svd_oversamples: int = 5,
    svd_n_iter: int = 2,
    grad_clip: float | None = 1.0,
    model_dtype: str = "float32",
    ld_chunk_nnz: int | None = 150_000_000,
    progress_every: int = 0,
    progress_label: str = "fit",
    prepared_blocks: list[dict] | None = None,
    objective_improve_tol: float = 1e-6,
    stagnation_patience: int = 25,
) -> tuple[np.ndarray, float, list[float], dict]:
    """Fit the convex non-negative nuclear-norm COMPASS relaxation."""

    train_dtype = _torch_dtype(model_dtype)
    block_specs = _prepare_fit_blocks(dataset, device) if prepared_blocks is None else prepared_blocks
    total_den = int(sum(int(torch.count_nonzero(block["weight"]).item()) for block in block_specs))

    if init_B is None:
        B = torch.zeros((n_genes, n_mechanisms), dtype=train_dtype, device=device, requires_grad=True)
    else:
        B = torch.as_tensor(init_B, dtype=train_dtype, device=device).clone().requires_grad_(True)
    tau = torch.tensor(max(float(init_tau), 0.0), dtype=train_dtype, device=device)

    losses: list[float] = []
    start = perf_counter()
    ld_stats = {
        "ld_chunks": 0,
        "ld_chunk_nnz_total": 0,
        "ld_sparse_convert_seconds": 0.0,
        "ld_eval_backward_seconds": 0.0,
    }
    best_objective = np.inf
    best_B = B.detach().clone()
    best_tau = tau.detach().clone()
    stalled_iterations = 0
    for it in range(max_iter):
        if B.grad is not None:
            B.grad.zero_()
        data_loss, tau_numerator, tau_denominator = _backward_data_loss(
            block_specs, B, tau, device, total_den, ld_chunk_nnz, ld_stats
        )
        tau_next = max(0.0, tau_numerator / tau_denominator) if tau_denominator > 0 else 0.0
        data_loss -= tau_denominator * (float(tau.detach().cpu()) - tau_next) ** 2 / max(total_den, 1)
        regularization = float(lambda_value * torch.linalg.matrix_norm(B.float(), ord="nuc").detach().cpu())
        objective = data_loss + regularization
        if np.isfinite(objective) and objective < best_objective - objective_improve_tol:
            best_objective = objective
            best_B = B.detach().clone()
            best_tau = torch.as_tensor(tau_next, dtype=train_dtype, device=device)
            stalled_iterations = 0
        else:
            stalled_iterations += 1
        with torch.no_grad():
            b_grad = torch.nan_to_num(B.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if grad_clip is not None:
                b_grad = torch.clamp(b_grad, min=-grad_clip, max=grad_clip)
            B_next = B - lr * b_grad
            B_next = nuclear_prox_nonnegative(
                B_next.float(),
                lr * lambda_value,
                svd_method=svd_method,
                svd_rank=svd_rank,
                svd_oversamples=svd_oversamples,
                svd_n_iter=svd_n_iter,
            ).to(train_dtype)
            tau.fill_(tau_next)
            delta = _relative_change(B_next, B)
            B.copy_(B_next)
        losses.append(objective)
        if progress_every and (it == 0 or (it + 1) % progress_every == 0):
            print(
                f"[fit] {progress_label} lambda={lambda_value:g} iteration={it + 1} "
                f"objective={objective:.6g} relative_change={float(delta.detach().cpu()):.3g} "
                f"stalled={stalled_iterations}",
                flush=True,
            )
        if it > 10 and (float(delta.detach().cpu()) < tol or stalled_iterations >= stagnation_patience):
            break
    with torch.no_grad():
        B.copy_(best_B)
        tau.copy_(best_tau)
    metadata = {
        "iterations": len(losses),
        "seconds": perf_counter() - start,
        "ld_blocks": len(block_specs),
        "ld_gpu_layout": _ld_torch_layout(device),
        "ld_index_dtype": str(_ld_index_dtype(device)).replace("torch.", ""),
        "ld_nnz": int(sum(block["R2_nnz"] for block in block_specs)),
        "ld_chunk_nnz": ld_chunk_nnz,
        **ld_stats,
        "svd_method": svd_method,
        "svd_rank": svd_rank,
        "grad_clip": grad_clip,
        "best_objective": best_objective,
        "tau_update": "exact_weighted_least_squares",
        "objective_improve_tol": objective_improve_tol,
        "stagnation_patience": stagnation_patience,
        "model_dtype": model_dtype,
        "ld_dtype": "float16",
    }
    return (
        B.detach().float().cpu().numpy(),
        float(tau.detach().cpu()),
        losses,
        metadata,
    )


def fit_hierarchical_nuclear(
    dataset: CompassDataset,
    n_genes: int,
    n_mechanisms: int,
    lambda_value: float,
    init_B: np.ndarray | None = None,
    init_context_effects: np.ndarray | None = None,
    fixed_context_effects: np.ndarray | None = None,
    fixed_context_effect_se: np.ndarray | None = None,
    scale_fixed_context_effects: bool = False,
    init_tau: float = 1e-8,
    lr: float = 1e-2,
    max_iter: int = 500,
    tol: float = 1e-6,
    device: str = "cpu",
    svd_method: str = "auto",
    svd_rank: int | None = None,
    svd_oversamples: int = 5,
    svd_n_iter: int = 2,
    grad_clip: float | None = 1.0,
    model_dtype: str = "float32",
    ld_chunk_nnz: int | None = 150_000_000,
    progress_every: int = 0,
    progress_label: str = "fit",
    prepared_blocks: list[dict] | None = None,
    objective_improve_tol: float = 1e-6,
    objective_relative_tol: float = 1e-5,
    objective_window: int = 10,
    stagnation_patience: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[float], dict]:
    """Fit unpenalized context effects plus nuclear-penalized gene deviations."""

    if dataset.context_annotations is None:
        raise ValueError("hierarchical fitting requires dataset.context_annotations")
    if dataset.context_annotations.shape != (dataset.n_variants, n_mechanisms):
        raise ValueError("context_annotations must have one column per mechanism")
    train_dtype = _torch_dtype(model_dtype)
    block_specs = _prepare_fit_blocks(dataset, device) if prepared_blocks is None else prepared_blocks
    total_den = int(sum(int(torch.count_nonzero(block["weight"]).item()) for block in block_specs))
    if init_B is None:
        B = torch.zeros((n_genes, n_mechanisms), dtype=train_dtype, device=device, requires_grad=True)
    else:
        B = torch.as_tensor(init_B, dtype=train_dtype, device=device).clone().requires_grad_(True)
    if fixed_context_effects is not None and not scale_fixed_context_effects:
        fixed_context_effects = np.asarray(fixed_context_effects, dtype=np.float32)
        if fixed_context_effects.shape != (n_mechanisms,) or np.any(fixed_context_effects < 0):
            raise ValueError("fixed_context_effects must be a non-negative vector with one value per mechanism")
        context_effects = torch.as_tensor(
            fixed_context_effects, dtype=train_dtype, device=device
        ).clone()
    elif init_context_effects is None:
        initial_context = (
            np.asarray(fixed_context_effects, dtype=np.float32)
            if scale_fixed_context_effects
            else np.zeros(n_mechanisms, dtype=np.float32)
        )
        if initial_context.shape != (n_mechanisms,) or np.any(initial_context < 0):
            raise ValueError("fixed_context_effects must be a non-negative vector with one value per mechanism")
        context_effects = torch.as_tensor(initial_context, dtype=train_dtype, device=device).clone()
    else:
        context_effects = torch.as_tensor(
            init_context_effects, dtype=train_dtype, device=device
        ).clone()
    initial_context_effects_used = context_effects.detach().float().cpu().numpy().copy()
    tau = torch.tensor(max(float(init_tau), 0.0), dtype=train_dtype, device=device)

    losses: list[float] = []
    start = perf_counter()
    ld_stats = {
        "ld_chunks": 0,
        "ld_chunk_nnz_total": 0,
        "ld_sparse_convert_seconds": 0.0,
        "ld_eval_backward_seconds": 0.0,
    }
    best_objective = np.inf
    best_B = B.detach().clone()
    best_context_effects = context_effects.detach().clone()
    best_context_effect_se = (
        np.asarray(fixed_context_effect_se, dtype=np.float64).copy()
        if fixed_context_effect_se is not None
        else np.full(n_mechanisms, np.nan, dtype=np.float64)
    )
    best_tau = tau.detach().clone()
    stalled_iterations = 0
    convergence_reason = "max_iterations"
    for it in range(max_iter):
        if B.grad is not None:
            B.grad.zero_()
        data_loss, effects_next_np, context_diagnostics = _backward_hierarchical_data_loss(
            block_specs,
            B,
            context_effects,
            tau,
            device,
            total_den,
            ld_chunk_nnz,
            ld_stats,
            fixed_context_effects=fixed_context_effects is not None and not scale_fixed_context_effects,
            context_profile=fixed_context_effects if scale_fixed_context_effects else None,
        )
        context_next = torch.as_tensor(
            effects_next_np[:-1], dtype=train_dtype, device=device
        )
        tau_next = torch.as_tensor(effects_next_np[-1], dtype=train_dtype, device=device)
        regularization = float(
            lambda_value * torch.linalg.matrix_norm(B.float(), ord="nuc").detach().cpu()
        )
        objective = data_loss + regularization
        if np.isfinite(objective) and objective < best_objective - objective_improve_tol:
            best_objective = objective
            best_B = B.detach().clone()
            best_context_effects = context_next.detach().clone()
            if fixed_context_effect_se is None:
                best_context_effect_se = context_diagnostics["context_effect_se"].copy()
            elif scale_fixed_context_effects:
                profile = np.asarray(fixed_context_effects, dtype=np.float64)
                positive = profile > 0
                scale = float(np.median(effects_next_np[:-1][positive] / profile[positive])) if np.any(positive) else 0.0
                best_context_effect_se = np.sqrt(
                    np.square(scale * np.asarray(fixed_context_effect_se, dtype=np.float64))
                    + np.square(context_diagnostics["context_effect_se"])
                )
            best_tau = tau_next.detach().clone()
            stalled_iterations = 0
        else:
            stalled_iterations += 1
        with torch.no_grad():
            b_grad = torch.nan_to_num(B.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if grad_clip is not None:
                b_grad = torch.clamp(b_grad, min=-grad_clip, max=grad_clip)
            B_next = nuclear_prox_nonnegative(
                (B - lr * b_grad).float(),
                lr * lambda_value,
                svd_method=svd_method,
                svd_rank=svd_rank,
                svd_oversamples=svd_oversamples,
                svd_n_iter=svd_n_iter,
            ).to(train_dtype)
            b_delta = _relative_change(B_next, B)
            context_delta = _relative_change(context_next, context_effects)
            delta = torch.maximum(b_delta, context_delta)
            B.copy_(B_next)
            context_effects.copy_(context_next)
            tau.copy_(tau_next)
        losses.append(objective)
        objective_converged = False
        if len(losses) > objective_window:
            previous_objective = losses[-objective_window - 1]
            objective_converged = (
                abs(previous_objective - objective)
                / max(abs(previous_objective), 1.0)
                < objective_relative_tol
            )
        if progress_every and (it == 0 or (it + 1) % progress_every == 0):
            print(
                f"[fit] {progress_label} lambda={lambda_value:g} iteration={it + 1} "
                f"objective={objective:.6g} relative_change={float(delta.detach().cpu()):.3g} "
                f"stalled={stalled_iterations}",
                flush=True,
            )
        if it > 10 and (
            float(delta.detach().cpu()) < tol
            or objective_converged
            or stalled_iterations >= stagnation_patience
        ):
            if objective_converged:
                convergence_reason = "relative_objective"
            elif stalled_iterations >= stagnation_patience:
                convergence_reason = "objective_stagnation"
            else:
                convergence_reason = "relative_parameters"
            break
    with torch.no_grad():
        B.copy_(best_B)
        context_effects.copy_(best_context_effects)
        tau.copy_(best_tau)
    metadata = {
        "iterations": len(losses),
        "seconds": perf_counter() - start,
        "ld_blocks": len(block_specs),
        "ld_gpu_layout": _ld_torch_layout(device),
        "ld_index_dtype": str(_ld_index_dtype(device)).replace("torch.", ""),
        "ld_nnz": int(sum(block["R2_nnz"] for block in block_specs)),
        "ld_chunk_nnz": ld_chunk_nnz,
        **ld_stats,
        "svd_method": svd_method,
        "svd_rank": svd_rank,
        "grad_clip": grad_clip,
        "best_objective": best_objective,
        "context_update": "joint_nonnegative_weighted_least_squares",
        "context_effects_fixed": fixed_context_effects is not None and not scale_fixed_context_effects,
        "context_effects_scaled": scale_fixed_context_effects,
        "initial_context_effects": initial_context_effects_used,
        "context_effect_se": best_context_effect_se,
        "context_effect_z": np.divide(
            best_context_effects.detach().float().cpu().numpy(),
            best_context_effect_se,
            out=np.zeros_like(best_context_effect_se),
            where=best_context_effect_se > 0,
        ),
        "jackknife_blocks": context_diagnostics["jackknife_blocks"],
        "objective_improve_tol": objective_improve_tol,
        "objective_relative_tol": objective_relative_tol,
        "objective_window": objective_window,
        "convergence_reason": convergence_reason,
        "stagnation_patience": stagnation_patience,
        "model_dtype": model_dtype,
        "ld_dtype": "float16",
    }
    return (
        B.detach().float().cpu().numpy(),
        context_effects.detach().float().cpu().numpy(),
        best_context_effect_se.astype(np.float32),
        float(tau.detach().cpu()),
        losses,
        metadata,
    )


def _normalize_simplex(raw: torch.Tensor, constrain: bool) -> torch.Tensor:
    if constrain:
        return torch.softmax(raw, dim=0)
    return torch.nn.functional.softplus(raw)


def rank1_factors_from_matrix(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return non-negative rank-1 factors initialized from a coefficient matrix."""

    B = np.maximum(np.asarray(B, dtype=np.float64), 0.0)
    if B.ndim != 2:
        raise ValueError("B must be a two-dimensional matrix")
    left, singular_values, right = np.linalg.svd(B, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0:
        return np.zeros(B.shape[0], dtype=np.float32), np.zeros(B.shape[1], dtype=np.float32)
    left_factor = left[:, 0]
    right_factor = right[0]
    if left_factor.sum() < 0:
        left_factor = -left_factor
        right_factor = -right_factor
    scale = np.sqrt(singular_values[0])
    return (
        np.maximum(left_factor * scale, 0.0).astype(np.float32),
        np.maximum(right_factor * scale, 0.0).astype(np.float32),
    )


def fit_rank1_alt(
    dataset: CompassDataset,
    n_genes: int,
    n_mechanisms: int,
    lambda_value: float,
    init_s: np.ndarray | None = None,
    init_w: np.ndarray | None = None,
    init_tau: float = 1e-8,
    lr: float = 1e-2,
    max_iter: int = 500,
    tol: float = 1e-6,
    constrain_w_simplex: bool = False,
    device: str = "cpu",
    model_dtype: str = "float32",
    ld_chunk_nnz: int | None = 150_000_000,
    progress_every: int = 0,
    progress_label: str = "rank1",
) -> tuple[np.ndarray, float, list[float], dict]:
    """Fit rank-1 B = s w' with alternating Torch updates for s and w."""

    train_dtype = _torch_dtype(model_dtype)
    block_specs = _prepare_fit_blocks(dataset, device)
    total_den = int(sum(int(torch.count_nonzero(block["weight"]).item()) for block in block_specs))

    s_raw0 = np.full(n_genes, -8.0, dtype=np.float32) if init_s is None else np.log(np.expm1(np.maximum(init_s, 1e-12)))
    if init_w is None:
        w_raw0 = np.zeros(n_mechanisms, dtype=np.float32)
    elif constrain_w_simplex:
        w_raw0 = np.log(np.maximum(init_w, 1e-12))
    else:
        w_raw0 = np.log(np.expm1(np.maximum(init_w, 1e-12)))
    s_raw = torch.as_tensor(s_raw0, dtype=train_dtype, device=device).clone().requires_grad_(True)
    w_raw = torch.as_tensor(w_raw0, dtype=train_dtype, device=device).clone().requires_grad_(True)
    tau = torch.tensor(max(float(init_tau), 0.0), dtype=train_dtype, device=device)

    opt_s = torch.optim.Adam([s_raw], lr=lr)
    opt_w = torch.optim.Adam([w_raw], lr=lr)
    losses: list[float] = []
    start = perf_counter()
    ld_stats = {
        "ld_chunks": 0,
        "ld_chunk_nnz_total": 0,
        "ld_sparse_convert_seconds": 0.0,
        "ld_eval_backward_seconds": 0.0,
    }
    last_B = None

    for it in range(max_iter):
        opt = opt_s if it % 2 == 0 else opt_w
        opt.zero_grad()
        data_loss = 0.0
        tau_numerator = 0.0
        tau_denominator = 0.0
        ld_layout = _ld_torch_layout(device)
        ld_index_dtype = _ld_index_dtype(device)
        for block in block_specs:
            R2_cpu = _block_r2_cpu(block)
            for start, end in iter_csr_row_ranges(R2_cpu, ld_chunk_nnz):
                ld_stats["ld_chunks"] += 1
                ld_stats["ld_chunk_nnz_total"] += int(R2_cpu.indptr[end] - R2_cpu.indptr[start])
                s = torch.nn.functional.softplus(s_raw)
                w = _normalize_simplex(w_raw, constrain_w_simplex)
                B = torch.outer(s, w)
                chunk_start = perf_counter()
                if ld_layout == "csr":
                    R2_t = scipy_csr_rows_to_torch_sparse(
                        R2_cpu,
                        start,
                        end,
                        device=device,
                        dtype=torch.float16,
                        index_dtype=ld_index_dtype,
                    )
                else:
                    R2_t = scipy_to_torch_sparse(
                        R2_cpu[start:end],
                        device=device,
                        dtype=torch.float32,
                        layout=ld_layout,
                        index_dtype=ld_index_dtype,
                    )
                ld_stats["ld_sparse_convert_seconds"] += perf_counter() - chunk_start
                eval_start = perf_counter()
                pred = predict_factorized(
                    B,
                    tau,
                    block["A_t"],
                    R2_t,
                    block["ld_score"][start:end],
                    _slice_vector(block["n_samples"], start, end),
                )
                residual = block["chisq"][start:end] - pred
                weight = block["weight"][start:end]
                ld_term = _slice_vector(block["n_samples"], start, end) * block["ld_score"][start:end]
                loss_block = torch.sum(weight * residual.square()) / max(total_den, 1)
                data_loss += float(loss_block.detach().cpu())
                tau_numerator += float(torch.sum(weight * ld_term * (residual + ld_term * tau)).detach().cpu())
                tau_denominator += float(torch.sum(weight * ld_term.square()).detach().cpu())
                loss_block.backward()
                if torch.device(device).type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                ld_stats["ld_eval_backward_seconds"] += perf_counter() - eval_start
                del R2_t, pred, residual, loss_block, B, ld_term
            del R2_cpu
        s = torch.nn.functional.softplus(s_raw)
        w = _normalize_simplex(w_raw, constrain_w_simplex)
        B = torch.outer(s, w)
        reg = lambda_value * torch.linalg.norm(B, ord="nuc")
        reg.backward()
        opt.step()
        tau_next = max(0.0, tau_numerator / tau_denominator) if tau_denominator > 0 else 0.0
        data_loss -= tau_denominator * (float(tau.detach().cpu()) - tau_next) ** 2 / max(total_den, 1)
        with torch.no_grad():
            tau.fill_(tau_next)
            B_now = torch.outer(torch.nn.functional.softplus(s_raw), _normalize_simplex(w_raw, constrain_w_simplex))
            delta = _relative_change(B_now, last_B) if last_B is not None else torch.tensor(float("inf"), device=device)
        objective = data_loss + float(reg.detach().cpu())
        losses.append(objective)
        if progress_every and (it == 0 or (it + 1) % progress_every == 0):
            print(
                f"[fit] {progress_label} lambda={lambda_value:g} iteration={it + 1} "
                f"objective={objective:.6g} relative_change={float(delta.detach().cpu()):.3g}",
                flush=True,
            )
        with torch.no_grad():
            if last_B is not None:
                if it > 20 and float(delta.detach().cpu()) < tol:
                    break
            last_B = B_now.detach().clone()

    B_final = torch.outer(torch.nn.functional.softplus(s_raw), _normalize_simplex(w_raw, constrain_w_simplex))
    metadata = {
        "iterations": len(losses),
        "seconds": perf_counter() - start,
        "constrain_w_simplex": constrain_w_simplex,
        "ld_blocks": len(block_specs),
        "ld_gpu_layout": _ld_torch_layout(device),
        "ld_index_dtype": str(_ld_index_dtype(device)).replace("torch.", ""),
        "ld_nnz": int(sum(block["R2_nnz"] for block in block_specs)),
        "ld_chunk_nnz": ld_chunk_nnz,
        **ld_stats,
        "model_dtype": model_dtype,
        "ld_dtype": "float16",
    }
    return (
        B_final.detach().float().cpu().numpy(),
        float(tau.detach().cpu()),
        losses,
        metadata,
    )


def grouped_variant_cv(
    dataset: CompassDataset,
    n_genes: int,
    n_mechanisms: int,
    lambdas: list[float],
    method: str = "nuclear",
    max_iter: int = 300,
    lr: float = 1e-2,
    device: str = "cpu",
    folds: list[int] | None = None,
    prepared_blocks: list[dict] | None = None,
    max_lambda_extensions: int = 4,
    lambda_extension_factor: float = 3.0,
    **kwargs,
) -> dict[float, float]:
    if dataset.cv_groups is None:
        raise ValueError("LD-component CV requires dataset.cv_groups")
    if dataset.cv_score_groups is None:
        raise ValueError("LD-component CV requires dataset.cv_score_groups")
    groups = np.asarray(dataset.cv_groups)
    available = sorted(int(x) for x in np.unique(groups) if int(x) >= 0)
    if not available:
        raise ValueError("LD-component CV requires at least one non-negative CV group")
    if max_lambda_extensions < 0:
        raise ValueError("max_lambda_extensions must be non-negative")
    if lambda_extension_factor <= 1.0:
        raise ValueError("lambda_extension_factor must be greater than one")
    ordered = sorted({float(lam) for lam in lambdas}, reverse=True)
    scores: dict[float, list[float]] = {lam: [] for lam in ordered}
    selected_folds = available if folds is None else [int(fold) for fold in folds]
    block_specs = _prepare_fit_blocks(dataset, device) if prepared_blocks is None else prepared_blocks
    fold_states: dict[int, tuple[np.ndarray, float]] = {}
    for fold in selected_folds:
        if not np.any(groups != fold) or not np.any(groups == fold):
            continue
        train_blocks = []
        for block in block_specs:
            block_groups = block["cv_groups"]
            if block_groups is None:
                raise ValueError("LD-component CV requires cv_groups in prepared blocks")
            train_weight = block["weight"] * block_groups.ne(fold).to(block["weight"].dtype)
            train_blocks.append({**block, "weight": train_weight})
        train_ds = None
        if method == "rank1":
            base_sample_weight = np.ones(dataset.n_variants, dtype=np.float32)
            if dataset.sample_weight is not None:
                base_sample_weight = np.asarray(dataset.sample_weight, dtype=np.float32)
            train_ds = CompassDataset(
                A=dataset.A,
                chisq=dataset.chisq,
                chrom=dataset.chrom,
                n_samples=dataset.n_samples,
                R2=dataset.R2,
                ld_blocks=dataset.ld_blocks,
                sample_weight=base_sample_weight * (groups != fold),
                cv_groups=dataset.cv_groups,
                cv_score_groups=dataset.cv_score_groups,
            )
        init_B = None
        init_tau = 1e-8
        init_s = None
        init_w = None
        for lam in ordered:
            if method == "nuclear":
                B, tau, _, _ = fit_nuclear_norm(
                    dataset,
                    n_genes,
                    n_mechanisms,
                    lam,
                    init_B=init_B,
                    init_tau=init_tau,
                    lr=lr,
                    max_iter=max_iter,
                    device=device,
                    progress_label=f"cv-fold={fold}",
                    prepared_blocks=train_blocks,
                    **kwargs,
                )
                init_B = B
            elif method == "rank1":
                assert train_ds is not None
                B, tau, _, _ = fit_rank1_alt(
                    train_ds,
                    n_genes,
                    n_mechanisms,
                    lam,
                    init_s=init_s,
                    init_w=init_w,
                    init_tau=init_tau,
                    lr=lr,
                    max_iter=max_iter,
                    device=device,
                    **kwargs,
                )
                init_s = B.sum(axis=1)
                init_w = B.sum(axis=0)
                if init_w.sum() > 0:
                    init_w = init_w / init_w.sum()
            else:
                raise ValueError(f"Unknown method: {method}")
            init_tau = tau
            score = _evaluate_fit_mse(block_specs, B, tau, device, kwargs.get("ld_chunk_nnz"), fold=fold)
            scores[lam].append(score)
        if method == "nuclear":
            assert init_B is not None
            fold_states[fold] = (init_B, init_tau)
        del train_blocks

    # A lower-bound CV selection leaves the regularization optimum unidentified.
    # Continue each fold from its endpoint rather than refitting the existing path.
    extensions = 0
    while method == "nuclear" and extensions < max_lambda_extensions:
        mean_scores = {lam: float(np.mean(values)) for lam, values in scores.items() if values}
        if not mean_scores:
            break
        smallest = min(mean_scores)
        if min(mean_scores, key=mean_scores.get) != smallest:
            break
        next_lambda = smallest / lambda_extension_factor
        if next_lambda in scores:
            break
        scores[next_lambda] = []
        for fold in selected_folds:
            if fold not in fold_states:
                continue
            train_blocks = []
            for block in block_specs:
                block_groups = block["cv_groups"]
                if block_groups is None:
                    raise ValueError("LD-component CV requires cv_groups in prepared blocks")
                train_weight = block["weight"] * block_groups.ne(fold).to(block["weight"].dtype)
                train_blocks.append({**block, "weight": train_weight})
            init_B, init_tau = fold_states[fold]
            B, tau, _, _ = fit_nuclear_norm(
                dataset,
                n_genes,
                n_mechanisms,
                next_lambda,
                init_B=init_B,
                init_tau=init_tau,
                lr=lr,
                max_iter=max_iter,
                device=device,
                progress_label=f"cv-fold={fold}",
                prepared_blocks=train_blocks,
                **kwargs,
            )
            score = _evaluate_fit_mse(block_specs, B, tau, device, kwargs.get("ld_chunk_nnz"), fold=fold)
            scores[next_lambda].append(score)
            fold_states[fold] = (B, tau)
            del train_blocks
        extensions += 1
    return {lam: float(np.mean(vals)) for lam, vals in scores.items() if vals}


def fit_nuclear_norm_path(
    dataset: CompassDataset,
    n_genes: int,
    n_mechanisms: int,
    lambdas: list[float],
    cv: bool = True,
    max_lambda_extensions: int = 4,
    lambda_extension_factor: float = 3.0,
    **kwargs,
) -> FitResult:
    ordered = sorted([float(l) for l in lambdas], reverse=True)
    prepared_blocks = _prepare_fit_blocks(dataset, kwargs.get("device", "cpu"))
    cv_scores = (
        grouped_variant_cv(
            dataset,
            n_genes,
            n_mechanisms,
            ordered,
            method="nuclear",
            prepared_blocks=prepared_blocks,
            max_lambda_extensions=max_lambda_extensions,
            lambda_extension_factor=lambda_extension_factor,
            **kwargs,
        )
        if cv
        else None
    )
    if cv_scores:
        ordered = sorted(cv_scores, reverse=True)
    best = min(cv_scores, key=cv_scores.get) if cv_scores else ordered[-1]
    init_B = None
    init_tau = 1e-8
    losses_all: list[float] = []
    cv_folds = None
    if cv and dataset.cv_groups is not None:
        cv_folds = sorted(int(x) for x in np.unique(dataset.cv_groups) if int(x) >= 0)
    metadata = {
        "cv_method": "ld_component" if cv else None,
        "cv_folds": cv_folds,
        "tau_update": "exact_weighted_least_squares",
        "lambda_extensions": max(0, len(ordered) - len({float(lam) for lam in lambdas})),
        "lambda_extension_factor": lambda_extension_factor,
    }
    B = None
    tau = init_tau
    for lam in ordered:
        B, tau, losses, meta = fit_nuclear_norm(
            dataset,
            n_genes,
            n_mechanisms,
            lam,
            init_B=init_B,
            init_tau=init_tau,
            progress_label="full",
            prepared_blocks=prepared_blocks,
            **kwargs,
        )
        losses_all.extend(losses)
        metadata[lam] = meta
        init_B = B
        init_tau = tau
        if lam == best:
            break
    assert B is not None
    return FitResult("nuclear", ordered, cv_scores, float(best), B, tau, losses_all, metadata)


def _save_hierarchical_cv_checkpoint(path: str | Path, checkpoint: dict[str, np.ndarray]) -> None:
    """Atomically persist resumable hierarchical CV state."""

    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **checkpoint)
    os.replace(temporary, checkpoint_path)


def _load_or_initialize_hierarchical_cv_checkpoint(
    path: str | Path,
    base_lambdas: list[float],
    cv_folds: list[int],
    n_genes: int,
    n_mechanisms: int,
    fixed_context_effects: np.ndarray | None,
    scale_fixed_context_effects: bool,
) -> dict[str, np.ndarray]:
    """Load compatible fold states or initialize an empty checkpoint."""

    checkpoint_path = Path(path).expanduser()
    profile = (
        np.asarray(fixed_context_effects, dtype=np.float32)
        if fixed_context_effects is not None
        else np.empty(0, dtype=np.float32)
    )
    if checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            checkpoint = {key: archive[key].copy() for key in archive.files}
        required = {
            "version",
            "lambdas",
            "folds",
            "B",
            "context_effects",
            "tau",
            "scores",
            "next_lambda_index",
            "fixed_context_effects",
            "scale_fixed_context_effects",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"Hierarchical CV checkpoint is missing fields: {sorted(missing)}")
        loaded_lambdas = np.asarray(checkpoint["lambdas"], dtype=np.float64)
        expected_prefix = np.asarray(base_lambdas, dtype=np.float64)
        compatible = (
            int(np.asarray(checkpoint["version"]).item()) == 1
            and loaded_lambdas.size >= expected_prefix.size
            and np.array_equal(loaded_lambdas[: expected_prefix.size], expected_prefix)
            and np.array_equal(checkpoint["folds"], np.asarray(cv_folds, dtype=np.int64))
            and checkpoint["B"].shape == (len(cv_folds), n_genes, n_mechanisms)
            and checkpoint["context_effects"].shape == (len(cv_folds), n_mechanisms)
            and checkpoint["scores"].shape == (len(cv_folds), loaded_lambdas.size)
            and np.array_equal(checkpoint["fixed_context_effects"], profile)
            and bool(np.asarray(checkpoint["scale_fixed_context_effects"]).item())
            == bool(scale_fixed_context_effects)
        )
        if not compatible:
            raise ValueError(
                "Hierarchical CV checkpoint is incompatible with the requested folds, "
                "lambda grid, model dimensions, or context profile"
            )
        print(f"[fit] resuming hierarchical CV from {checkpoint_path}", flush=True)
        return checkpoint

    checkpoint = {
        "version": np.asarray(1, dtype=np.int64),
        "lambdas": np.asarray(base_lambdas, dtype=np.float64),
        "folds": np.asarray(cv_folds, dtype=np.int64),
        "B": np.zeros((len(cv_folds), n_genes, n_mechanisms), dtype=np.float32),
        "context_effects": np.zeros((len(cv_folds), n_mechanisms), dtype=np.float32),
        "tau": np.full(len(cv_folds), 1e-8, dtype=np.float32),
        "scores": np.full((len(cv_folds), len(base_lambdas)), np.nan, dtype=np.float64),
        "next_lambda_index": np.zeros(len(cv_folds), dtype=np.int64),
        "fixed_context_effects": profile,
        "scale_fixed_context_effects": np.asarray(scale_fixed_context_effects, dtype=np.bool_),
    }
    _save_hierarchical_cv_checkpoint(checkpoint_path, checkpoint)
    return checkpoint


def fit_hierarchical_nuclear_path(
    dataset: CompassDataset,
    n_genes: int,
    n_mechanisms: int,
    lambdas: list[float],
    cv: bool = True,
    fixed_context_effects: np.ndarray | None = None,
    fixed_context_effect_se: np.ndarray | None = None,
    scale_fixed_context_effects: bool = False,
    cv_checkpoint_path: str | Path | None = None,
    max_lambda_extensions: int = 4,
    lambda_extension_factor: float = 3.0,
    **kwargs,
) -> FitResult:
    """Fit a context-main-effect model with nuclear-regularized gene deviations."""

    if dataset.context_annotations is None:
        raise ValueError("hierarchical fitting requires context annotations")
    ordered = sorted({float(value) for value in lambdas}, reverse=True)
    prepared_blocks = _prepare_fit_blocks(dataset, kwargs.get("device", "cpu"))
    cv_scores: dict[float, float] | None = None
    cv_folds = None
    if cv:
        if dataset.cv_groups is None or dataset.cv_score_groups is None:
            raise ValueError("LD-component CV groups are required")
        groups = np.asarray(dataset.cv_groups)
        cv_folds = sorted(int(value) for value in np.unique(groups) if int(value) >= 0)
        base_lambdas = ordered.copy()
        checkpoint = None
        if cv_checkpoint_path is not None:
            checkpoint = _load_or_initialize_hierarchical_cv_checkpoint(
                cv_checkpoint_path,
                base_lambdas,
                cv_folds,
                n_genes,
                n_mechanisms,
                fixed_context_effects,
                scale_fixed_context_effects,
            )
            ordered = checkpoint["lambdas"].tolist()
        fold_scores: dict[float, list[float]] = {value: [] for value in ordered}
        if checkpoint is not None:
            for lambda_index, value in enumerate(ordered):
                fold_scores[value] = checkpoint["scores"][:, lambda_index][
                    np.isfinite(checkpoint["scores"][:, lambda_index])
                ].tolist()
        fold_states: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        for fold_index, fold in enumerate(cv_folds):
            start_index = (
                int(checkpoint["next_lambda_index"][fold_index])
                if checkpoint is not None
                else 0
            )
            if start_index:
                fold_states[fold] = (
                    checkpoint["B"][fold_index].copy(),
                    checkpoint["context_effects"][fold_index].copy(),
                    float(checkpoint["tau"][fold_index]),
                )
            if start_index >= len(ordered):
                continue
            train_blocks = []
            for block in prepared_blocks:
                block_groups = block["cv_groups"]
                train_weight = block["weight"] * block_groups.ne(fold).to(block["weight"].dtype)
                train_blocks.append({**block, "weight": train_weight})
            if start_index:
                init_B, init_context, init_tau = fold_states[fold]
            else:
                init_B = None
                init_context = None
                init_tau = 1e-8
            for lambda_index in range(start_index, len(ordered)):
                lambda_value = ordered[lambda_index]
                B, context_effects, _, tau, _, _ = fit_hierarchical_nuclear(
                    dataset,
                    n_genes,
                    n_mechanisms,
                    lambda_value,
                    init_B=init_B,
                    init_context_effects=init_context,
                    init_tau=init_tau,
                    fixed_context_effects=fixed_context_effects,
                    fixed_context_effect_se=fixed_context_effect_se,
                    scale_fixed_context_effects=scale_fixed_context_effects,
                    progress_label=f"cv-fold={fold}",
                    prepared_blocks=train_blocks,
                    **kwargs,
                )
                score = _evaluate_fit_mse(
                    prepared_blocks,
                    B,
                    tau,
                    kwargs.get("device", "cpu"),
                    kwargs.get("ld_chunk_nnz"),
                    fold=fold,
                    context_effects=context_effects,
                )
                fold_scores[lambda_value].append(score)
                init_B = B
                init_context = context_effects
                init_tau = tau
                fold_states[fold] = (B, context_effects, tau)
                if checkpoint is not None:
                    checkpoint["B"][fold_index] = B
                    checkpoint["context_effects"][fold_index] = context_effects
                    checkpoint["tau"][fold_index] = tau
                    checkpoint["scores"][fold_index, lambda_index] = score
                    checkpoint["next_lambda_index"][fold_index] = lambda_index + 1
                    _save_hierarchical_cv_checkpoint(cv_checkpoint_path, checkpoint)
            fold_states[fold] = (init_B, init_context, init_tau)
            del train_blocks
        cv_scores = {value: float(np.mean(scores)) for value, scores in fold_scores.items()}
        extensions = max(0, len(ordered) - len(base_lambdas))
        while (
            extensions < max_lambda_extensions
            and min(cv_scores, key=cv_scores.get) == min(cv_scores)
        ):
            next_lambda = min(cv_scores) / lambda_extension_factor
            if next_lambda in cv_scores:
                break
            if checkpoint is not None:
                checkpoint["lambdas"] = np.append(checkpoint["lambdas"], next_lambda)
                checkpoint["scores"] = np.column_stack(
                    (checkpoint["scores"], np.full(len(cv_folds), np.nan, dtype=np.float64))
                )
                ordered = checkpoint["lambdas"].tolist()
            extension_scores = []
            for fold_index, fold in enumerate(cv_folds):
                train_blocks = []
                for block in prepared_blocks:
                    block_groups = block["cv_groups"]
                    train_weight = block["weight"] * block_groups.ne(fold).to(block["weight"].dtype)
                    train_blocks.append({**block, "weight": train_weight})
                init_B, init_context, init_tau = fold_states[fold]
                B, context_effects, _, tau, _, _ = fit_hierarchical_nuclear(
                    dataset,
                    n_genes,
                    n_mechanisms,
                    next_lambda,
                    init_B=init_B,
                    init_context_effects=init_context,
                    init_tau=init_tau,
                    fixed_context_effects=fixed_context_effects,
                    fixed_context_effect_se=fixed_context_effect_se,
                    scale_fixed_context_effects=scale_fixed_context_effects,
                    progress_label=f"cv-fold={fold}",
                    prepared_blocks=train_blocks,
                    **kwargs,
                )
                extension_scores.append(
                    _evaluate_fit_mse(
                        prepared_blocks,
                        B,
                        tau,
                        kwargs.get("device", "cpu"),
                        kwargs.get("ld_chunk_nnz"),
                        fold=fold,
                        context_effects=context_effects,
                    )
                )
                fold_states[fold] = (B, context_effects, tau)
                if checkpoint is not None:
                    checkpoint["B"][fold_index] = B
                    checkpoint["context_effects"][fold_index] = context_effects
                    checkpoint["tau"][fold_index] = tau
                    checkpoint["scores"][fold_index, -1] = extension_scores[-1]
                    checkpoint["next_lambda_index"][fold_index] = len(ordered)
                    _save_hierarchical_cv_checkpoint(cv_checkpoint_path, checkpoint)
                del train_blocks
            cv_scores[next_lambda] = float(np.mean(extension_scores))
            if checkpoint is None:
                ordered.append(next_lambda)
                ordered.sort(reverse=True)
            extensions += 1
    best = min(cv_scores, key=cv_scores.get) if cv_scores else ordered[-1]
    metadata = {
        "cv_method": "ld_component" if cv else None,
        "cv_folds": cv_folds,
        "context_update": "joint_nonnegative_weighted_least_squares",
        "lambda_extensions": max(0, len(ordered) - len({float(value) for value in lambdas})),
        "lambda_extension_factor": lambda_extension_factor,
        "cv_checkpoint_path": str(cv_checkpoint_path) if cv_checkpoint_path is not None else None,
    }
    init_B = None
    init_context = None
    init_tau = 1e-8
    losses_all: list[float] = []
    B = None
    context_effects = None
    context_effect_se = None
    tau = init_tau
    for lambda_value in ordered:
        B, context_effects, context_effect_se, tau, losses, fit_metadata = fit_hierarchical_nuclear(
            dataset,
            n_genes,
            n_mechanisms,
            lambda_value,
            init_B=init_B,
            init_context_effects=init_context,
            init_tau=init_tau,
            fixed_context_effects=fixed_context_effects,
            fixed_context_effect_se=fixed_context_effect_se,
            scale_fixed_context_effects=scale_fixed_context_effects,
            progress_label="full",
            prepared_blocks=prepared_blocks,
            **kwargs,
        )
        losses_all.extend(losses)
        metadata[lambda_value] = fit_metadata
        init_B = B
        init_context = context_effects
        init_tau = tau
        if lambda_value == best:
            break
    assert B is not None and context_effects is not None and context_effect_se is not None
    return FitResult(
        "hierarchical",
        ordered,
        cv_scores,
        float(best),
        B,
        tau,
        losses_all,
        metadata,
        context_effects=context_effects,
        context_effect_se=context_effect_se,
    )


def fit_rank1_path(
    dataset: CompassDataset,
    n_genes: int,
    n_mechanisms: int,
    lambdas: list[float],
    cv: bool = True,
    initial_B: np.ndarray | None = None,
    **kwargs,
) -> FitResult:
    ordered = sorted([float(l) for l in lambdas], reverse=True)
    cv_scores = grouped_variant_cv(dataset, n_genes, n_mechanisms, ordered, method="rank1", **kwargs) if cv else None
    best = min(cv_scores, key=cv_scores.get) if cv_scores else ordered[-1]
    init_s = None
    init_w = None
    if initial_B is not None:
        expected_shape = (n_genes, n_mechanisms)
        if np.asarray(initial_B).shape != expected_shape:
            raise ValueError(f"initial_B must have shape {expected_shape}")
        init_s, init_w = rank1_factors_from_matrix(initial_B)
    init_tau = 1e-8
    losses_all: list[float] = []
    cv_folds = None
    if cv and dataset.cv_groups is not None:
        cv_folds = sorted(int(x) for x in np.unique(dataset.cv_groups) if int(x) >= 0)
    metadata = {
        "cv_method": "ld_component" if cv else None,
        "cv_folds": cv_folds,
        "initialization": "nuclear_rank1_svd" if initial_B is not None else "default",
    }
    B = None
    tau = init_tau
    for lam in ordered:
        B, tau, losses, meta = fit_rank1_alt(
            dataset,
            n_genes,
            n_mechanisms,
            lam,
            init_s=init_s,
            init_w=init_w,
            init_tau=init_tau,
            **kwargs,
        )
        losses_all.extend(losses)
        metadata[lam] = meta
        init_s = B.sum(axis=1)
        init_w = B.sum(axis=0)
        if init_w.sum() > 0:
            init_w = init_w / init_w.sum()
        init_tau = tau
        if lam == best:
            break
    assert B is not None
    return FitResult("rank1", ordered, cv_scores, float(best), B, tau, losses_all, metadata)

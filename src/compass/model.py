from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp
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
    R2: sp.csr_matrix | None = None
    ld_blocks: list[LdChromosomeBlock] | None = None
    sample_weight: np.ndarray | None = None
    cv_groups: np.ndarray | None = None
    cv_score_groups: np.ndarray | None = None

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


def _weights(chisq: torch.Tensor, weights: np.ndarray | None, device: str) -> torch.Tensor:
    if weights is None:
        # Simple LDSC-like stabilizer: downweight extremely large chi-square values.
        return (1.0 / torch.clamp(chisq, min=1.0)).to(device)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _flatten_B(B: torch.Tensor) -> torch.Tensor:
    return B.reshape(-1)


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
        block_specs.append(
            {
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
        )
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


def _evaluate_fit_mse(
    block_specs: list[dict],
    B: np.ndarray,
    tau: float,
    device: str,
    ld_chunk_nnz: int | None,
    fold: int | None = None,
) -> float:
    """Evaluate an LD-component fold without rebuilding LD subsets."""

    B_t = torch.as_tensor(B, dtype=torch.float32, device=device)
    tau_t = torch.tensor(float(tau), dtype=torch.float32, device=device)
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

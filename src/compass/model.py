from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import torch

from .ld import scipy_to_torch_sparse


@dataclass
class CompassDataset:
    A: sp.csr_matrix
    R2: sp.csr_matrix
    chisq: np.ndarray
    chrom: np.ndarray
    n_samples: float | np.ndarray
    sample_weight: np.ndarray | None = None

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


def _samples_tensor(n_samples: float | np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(n_samples, dtype=torch.float32, device=device)


def predict_factorized(
    B: torch.Tensor,
    tau_raw: torch.Tensor,
    A_t: torch.Tensor,
    R2_t: torch.Tensor,
    ld_score: torch.Tensor,
    n_samples: float | torch.Tensor,
) -> torch.Tensor:
    mediated = torch.sparse.mm(A_t, _flatten_B(B).unsqueeze(1)).squeeze(1)
    smoothed = torch.sparse.mm(R2_t, mediated.unsqueeze(1)).squeeze(1)
    tau = torch.nn.functional.softplus(tau_raw)
    return 1.0 + n_samples * (smoothed + tau * ld_score)


def predict_precomputed(
    B: torch.Tensor,
    tau_raw: torch.Tensor,
    T_t: torch.Tensor,
    ld_score: torch.Tensor,
    n_samples: float | torch.Tensor,
) -> torch.Tensor:
    smoothed = torch.sparse.mm(T_t, _flatten_B(B).unsqueeze(1)).squeeze(1)
    tau = torch.nn.functional.softplus(tau_raw)
    return 1.0 + n_samples * (smoothed + tau * ld_score)


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
    use_precomputed: bool = True,
    device: str = "cpu",
    svd_method: str = "auto",
    svd_rank: int | None = None,
    svd_oversamples: int = 5,
    svd_n_iter: int = 2,
    grad_clip: float | None = 1.0,
) -> tuple[np.ndarray, float, list[float], dict]:
    """Fit the convex non-negative nuclear-norm COMPASS relaxation."""

    A = dataset.A.astype(np.float32)
    R2 = dataset.R2.astype(np.float32)
    T = (R2 @ A).tocsr() if use_precomputed else None
    A_t = scipy_to_torch_sparse(A, device=device) if not use_precomputed else None
    R2_t = scipy_to_torch_sparse(R2, device=device) if not use_precomputed else None
    T_t = scipy_to_torch_sparse(T, device=device) if use_precomputed else None
    chisq = torch.as_tensor(dataset.chisq, dtype=torch.float32, device=device)
    weight = _weights(chisq, dataset.sample_weight, device)
    ld_score = torch.as_tensor(np.asarray(R2.sum(axis=1)).ravel(), dtype=torch.float32, device=device)
    n_samples = _samples_tensor(dataset.n_samples, device)

    if init_B is None:
        B = torch.zeros((n_genes, n_mechanisms), dtype=torch.float32, device=device, requires_grad=True)
    else:
        B = torch.as_tensor(init_B, dtype=torch.float32, device=device).clone().requires_grad_(True)
    tau_raw = torch.tensor(float(np.log(np.expm1(max(init_tau, 1e-12)))), dtype=torch.float32, device=device, requires_grad=True)

    losses: list[float] = []
    start = perf_counter()
    for it in range(max_iter):
        if B.grad is not None:
            B.grad.zero_()
        if tau_raw.grad is not None:
            tau_raw.grad.zero_()
        if use_precomputed:
            pred = predict_precomputed(B, tau_raw, T_t, ld_score, n_samples)
        else:
            pred = predict_factorized(B, tau_raw, A_t, R2_t, ld_score, n_samples)
        residual = chisq - pred
        loss = torch.mean(weight * residual.square())
        loss.backward()
        with torch.no_grad():
            b_grad = torch.nan_to_num(B.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if grad_clip is not None:
                b_grad = torch.clamp(b_grad, min=-grad_clip, max=grad_clip)
            tau_grad = torch.nan_to_num(tau_raw.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if grad_clip is not None:
                tau_grad = torch.clamp(tau_grad, min=-grad_clip, max=grad_clip)
            B_next = B - lr * b_grad
            B_next = nuclear_prox_nonnegative(
                B_next,
                lr * lambda_value,
                svd_method=svd_method,
                svd_rank=svd_rank,
                svd_oversamples=svd_oversamples,
                svd_n_iter=svd_n_iter,
            )
            tau_raw -= lr * tau_grad
            delta = torch.linalg.norm(B_next - B) / (torch.linalg.norm(B) + 1e-8)
            B.copy_(B_next)
        losses.append(float(loss.detach().cpu()))
        if it > 10 and float(delta.detach().cpu()) < tol:
            break
    metadata = {
        "iterations": len(losses),
        "seconds": perf_counter() - start,
        "use_precomputed": use_precomputed,
        "T_nnz": None if T is None else int(T.nnz),
        "svd_method": svd_method,
        "svd_rank": svd_rank,
        "grad_clip": grad_clip,
    }
    return (
        B.detach().cpu().numpy(),
        float(torch.nn.functional.softplus(tau_raw).detach().cpu()),
        losses,
        metadata,
    )


def _normalize_simplex(raw: torch.Tensor, constrain: bool) -> torch.Tensor:
    if constrain:
        return torch.softmax(raw, dim=0)
    return torch.nn.functional.softplus(raw)


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
    use_precomputed: bool = True,
    device: str = "cpu",
) -> tuple[np.ndarray, float, list[float], dict]:
    """Fit rank-1 B = s w' with alternating Torch updates for s and w."""

    A = dataset.A.astype(np.float32)
    R2 = dataset.R2.astype(np.float32)
    T = (R2 @ A).tocsr() if use_precomputed else None
    A_t = scipy_to_torch_sparse(A, device=device) if not use_precomputed else None
    R2_t = scipy_to_torch_sparse(R2, device=device) if not use_precomputed else None
    T_t = scipy_to_torch_sparse(T, device=device) if use_precomputed else None
    chisq = torch.as_tensor(dataset.chisq, dtype=torch.float32, device=device)
    weight = _weights(chisq, dataset.sample_weight, device)
    ld_score = torch.as_tensor(np.asarray(R2.sum(axis=1)).ravel(), dtype=torch.float32, device=device)
    n_samples = _samples_tensor(dataset.n_samples, device)

    s_raw0 = np.full(n_genes, -8.0, dtype=np.float32) if init_s is None else np.log(np.expm1(np.maximum(init_s, 1e-12)))
    if init_w is None:
        w_raw0 = np.zeros(n_mechanisms, dtype=np.float32)
    elif constrain_w_simplex:
        w_raw0 = np.log(np.maximum(init_w, 1e-12))
    else:
        w_raw0 = np.log(np.expm1(np.maximum(init_w, 1e-12)))
    s_raw = torch.as_tensor(s_raw0, dtype=torch.float32, device=device).clone().requires_grad_(True)
    w_raw = torch.as_tensor(w_raw0, dtype=torch.float32, device=device).clone().requires_grad_(True)
    tau_raw = torch.tensor(float(np.log(np.expm1(max(init_tau, 1e-12)))), dtype=torch.float32, device=device, requires_grad=True)

    opt_s = torch.optim.Adam([s_raw, tau_raw], lr=lr)
    opt_w = torch.optim.Adam([w_raw, tau_raw], lr=lr)
    losses: list[float] = []
    start = perf_counter()
    last_B = None

    for it in range(max_iter):
        opt = opt_s if it % 2 == 0 else opt_w
        opt.zero_grad()
        s = torch.nn.functional.softplus(s_raw)
        w = _normalize_simplex(w_raw, constrain_w_simplex)
        B = torch.outer(s, w)
        if use_precomputed:
            pred = predict_precomputed(B, tau_raw, T_t, ld_score, n_samples)
        else:
            pred = predict_factorized(B, tau_raw, A_t, R2_t, ld_score, n_samples)
        residual = chisq - pred
        loss = torch.mean(weight * residual.square()) + lambda_value * torch.linalg.norm(B, ord="nuc")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            B_now = torch.outer(torch.nn.functional.softplus(s_raw), _normalize_simplex(w_raw, constrain_w_simplex))
            if last_B is not None:
                delta = torch.linalg.norm(B_now - last_B) / (torch.linalg.norm(last_B) + 1e-8)
                if it > 20 and float(delta.detach().cpu()) < tol:
                    break
            last_B = B_now.detach().clone()

    B_final = torch.outer(torch.nn.functional.softplus(s_raw), _normalize_simplex(w_raw, constrain_w_simplex))
    metadata = {
        "iterations": len(losses),
        "seconds": perf_counter() - start,
        "use_precomputed": use_precomputed,
        "constrain_w_simplex": constrain_w_simplex,
        "T_nnz": None if T is None else int(T.nnz),
    }
    return (
        B_final.detach().cpu().numpy(),
        float(torch.nn.functional.softplus(tau_raw).detach().cpu()),
        losses,
        metadata,
    )


def _subset_dataset(dataset: CompassDataset, keep: np.ndarray) -> CompassDataset:
    keep = np.asarray(keep, dtype=bool)
    n_samples = dataset.n_samples
    if np.ndim(n_samples) > 0:
        n_samples = np.asarray(n_samples)[keep]
    return CompassDataset(
        A=dataset.A[keep],
        R2=dataset.R2[keep][:, keep],
        chisq=dataset.chisq[keep],
        chrom=dataset.chrom[keep],
        n_samples=n_samples,
        sample_weight=None if dataset.sample_weight is None else dataset.sample_weight[keep],
    )


def leave_one_chrom_cv(
    dataset: CompassDataset,
    n_genes: int,
    n_mechanisms: int,
    lambdas: list[float],
    method: str = "nuclear",
    max_iter: int = 300,
    lr: float = 1e-2,
    device: str = "cpu",
    folds: list[int] | None = None,
    **kwargs,
) -> dict[float, float]:
    scores: dict[float, list[float]] = {float(lam): [] for lam in lambdas}
    chroms = sorted(np.unique(dataset.chrom).astype(int).tolist()) if folds is None else folds
    for chrom in chroms:
        train = dataset.chrom != chrom
        test = dataset.chrom == chrom
        if train.sum() == 0 or test.sum() == 0:
            continue
        train_ds = _subset_dataset(dataset, train)
        test_ds = _subset_dataset(dataset, test)
        init_B = None
        init_tau = 1e-8
        init_s = None
        init_w = None
        for lam in sorted([float(x) for x in lambdas], reverse=True):
            if method == "nuclear":
                B, tau, _, _ = fit_nuclear_norm(
                    train_ds,
                    n_genes,
                    n_mechanisms,
                    lam,
                    init_B=init_B,
                    init_tau=init_tau,
                    lr=lr,
                    max_iter=max_iter,
                    device=device,
                    **kwargs,
                )
                init_B = B
            elif method == "rank1":
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
            pred = _predict_numpy(test_ds, B, tau)
            score = float(np.average((test_ds.chisq - pred) ** 2))
            scores[lam].append(score)
    return {lam: float(np.mean(vals)) for lam, vals in scores.items() if vals}


def _predict_numpy(dataset: CompassDataset, B: np.ndarray, tau: float) -> np.ndarray:
    ld_score = np.asarray(dataset.R2.sum(axis=1)).ravel()
    smoothed = dataset.R2 @ (dataset.A @ B.reshape(-1))
    return 1.0 + np.asarray(dataset.n_samples) * (np.asarray(smoothed).ravel() + tau * ld_score)


def fit_nuclear_norm_path(
    dataset: CompassDataset,
    n_genes: int,
    n_mechanisms: int,
    lambdas: list[float],
    cv: bool = True,
    **kwargs,
) -> FitResult:
    ordered = sorted([float(l) for l in lambdas], reverse=True)
    cv_scores = leave_one_chrom_cv(dataset, n_genes, n_mechanisms, ordered, method="nuclear", **kwargs) if cv else None
    best = min(cv_scores, key=cv_scores.get) if cv_scores else ordered[-1]
    init_B = None
    init_tau = 1e-8
    losses_all: list[float] = []
    metadata = {}
    B = None
    tau = init_tau
    for lam in ordered:
        B, tau, losses, meta = fit_nuclear_norm(
            dataset, n_genes, n_mechanisms, lam, init_B=init_B, init_tau=init_tau, **kwargs
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
    **kwargs,
) -> FitResult:
    ordered = sorted([float(l) for l in lambdas], reverse=True)
    cv_scores = leave_one_chrom_cv(dataset, n_genes, n_mechanisms, ordered, method="rank1", **kwargs) if cv else None
    best = min(cv_scores, key=cv_scores.get) if cv_scores else ordered[-1]
    init_s = None
    init_w = None
    init_tau = 1e-8
    losses_all: list[float] = []
    metadata = {}
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

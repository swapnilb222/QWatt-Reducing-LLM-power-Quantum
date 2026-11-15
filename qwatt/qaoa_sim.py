"""Exact numpy statevector QAOA (plan 6.5-6.6). Used offline to train angles and to cross-check
Quantum Rings counts (total variation distance).

State index k <-> bits x_i = (k >> i) & 1 (qubit i = bit i, same as Qiskit's integer convention).
    |psi> = prod_l  exp(-i beta_l sum_i X_i) exp(-i gamma_l H_C)  |+>^n
    H_C   = sum_i a_i Z_i + sum_{i<j} b_ij Z_i Z_j       (normalized Ising, z_i = 1 - 2 x_i)
H_C is diagonal, so exp(-i gamma H_C) is an elementwise phase. The mixer is RX(2 beta) on every qubit.
These are exactly the gates of qaoa_qr.build_circuit: RZ(2 gamma a_i), RZZ(2 gamma b_ij), RX(2 beta).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from qwatt import qubo


def ising_diagonal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """E_k = sum_i a_i z_i + sum_{i<j} b_ij z_i z_j for every basis state k."""
    Z = 1.0 - 2.0 * qubo.all_bits(len(a)).astype(np.float64)
    return Z @ a + 0.5 * np.einsum("mi,mi->m", Z @ b, Z)


def _mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(n):
        v = psi.reshape(-1, 2, 2 ** q)          # axis 1 = qubit q
        v0, v1 = v[:, 0, :].copy(), v[:, 1, :]
        v[:, 0, :] = c * v0 + s * v1
        v[:, 1, :] = s * v0 + c * v1
    return psi


def statevector(diag: np.ndarray, gammas, betas) -> np.ndarray:
    n = int(np.log2(diag.size))
    psi = np.full(diag.size, 1 / np.sqrt(diag.size), dtype=np.complex128)
    for g, bt in zip(gammas, betas):
        psi *= np.exp(-1j * g * diag)
        psi = _mixer(psi, bt, n)
    return psi


def probabilities(diag: np.ndarray, gammas, betas) -> np.ndarray:
    return np.abs(statevector(diag, gammas, betas)) ** 2


def expectation(diag: np.ndarray, gammas, betas) -> float:
    return float(probabilities(diag, gammas, betas) @ diag)


def top_mask(costs: np.ndarray, frac: float = 0.01) -> np.ndarray:
    """Boolean mask of the best `frac` of all bitstrings by cost (S_top in plan 6.9)."""
    m = max(1, int(np.ceil(frac * costs.size)))
    idx = np.argpartition(costs, m - 1)[:m]
    mask = np.zeros(costs.size, dtype=bool)
    mask[idx] = True
    return mask


def amplification(probs: np.ndarray, costs: np.ndarray, frac: float = 0.01) -> float:
    """A = P(S_top) / (|S_top| / 2^n). A = 1 means no better than uniform sampling."""
    mask = top_mask(costs, frac)
    return float(probs[mask].sum() / (mask.sum() / costs.size))


def train_instance(diag: np.ndarray, p: int = 1, grid: int = 24, seed: int = 0) -> tuple[list, list, float]:
    """Coarse grid over (gamma, beta) for p=1 (beta in [0, pi), gamma in (0, 2]), then Nelder-Mead.
    For p>1 the p=1 optimum seeds a linear ramp."""
    gs = np.linspace(0.05, 2.0, grid)
    bs = np.linspace(0.0, np.pi, grid, endpoint=False)
    best = (np.inf, 0.4, 2.5)
    for g in gs:
        for b in bs:
            e = expectation(diag, [g], [b])
            if e < best[0]:
                best = (e, g, b)
    x0 = [best[1], best[2]] if p == 1 else list(np.linspace(best[1] / p, best[1], p)) + list(np.linspace(best[2], best[2] / p, p))
    res = minimize(lambda v: expectation(diag, v[:p], v[p:]), x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400 * p})
    g, b = list(map(float, res.x[:p])), [float(v % np.pi) for v in res.x[p:]]
    return g, b, float(res.fun)


def p1_marginals(a: np.ndarray, b: np.ndarray, gamma: float, beta: float) -> np.ndarray:
    """Exact p=1 QAOA single-qubit marginals at ANY n (closed form, verified against the statevector):
        <Z_u> = sin(2 beta) sin(2 gamma a_u) prod_v cos(2 gamma b_uv)
    Returns P(x_u = 1) = (1 - <Z_u>) / 2. Used to predict how many sentences a shot keeps at n = 100-200."""
    ez = np.sin(2 * beta) * np.sin(2 * gamma * a) * np.prod(np.cos(2 * gamma * b), axis=1)
    return (1 - ez) / 2

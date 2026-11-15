"""Classical selectors on the same QUBO. All are budget-feasible: sum_i t_i x_i <= B.

Each solver takes (h, J, t, budget, ...) and returns a 0/1 numpy vector x (x_i = keep sentence i).
Cost is always evaluated by qwatt.qubo.
"""
from __future__ import annotations

import math

import numpy as np

from qwatt import qubo


def topk(relevance: np.ndarray, t: np.ndarray, budget: float) -> np.ndarray:
    """A1: highest relevance first; skip a sentence that does not fit, keep going. Ignores pairs."""
    x = np.zeros(len(t), dtype=np.uint8)
    used = 0.0
    for i in np.argsort(-relevance, kind="stable"):
        if used + t[i] <= budget:
            x[i] = 1
            used += t[i]
    return x


def greedy(h: np.ndarray, J: np.ndarray, t: np.ndarray, budget: float) -> np.ndarray:
    """A2: repeatedly add the sentence with the most negative marginal cost h_i + sum_{j in S} J_ij,
    among those that still fit. Stop when no addition lowers the cost."""
    n = len(h)
    x = np.zeros(n, dtype=np.uint8)
    used = 0.0
    while True:
        gain = h + J @ x
        cand = [i for i in range(n) if not x[i] and used + t[i] <= budget and gain[i] < 0]
        if not cand:
            return x
        i = min(cand, key=lambda k: gain[k])
        x[i] = 1
        used += t[i]


def brute_force(h: np.ndarray, J: np.ndarray, t: np.ndarray, budget: float,
                costs: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """A4: exact budget-feasible optimum over all 2^n bitstrings (vectorized numpy; ~10-20 ms at n=16).

    Returns (x_opt, info) with info = {k_opt, c_opt, c_worst_feasible, n_feasible, c_opt_unconstrained,
    k_opt_unconstrained} — what the solver-quality metrics need.
    """
    n = len(h)
    if costs is None:
        costs = qubo.all_costs(h, J)
    tok = qubo.all_bits(n) @ t
    feas = tok <= budget
    cf = np.where(feas, costs, np.inf)
    k = int(np.argmin(cf))
    k_u = int(np.argmin(costs))
    info = {"k_opt": k, "c_opt": float(costs[k]), "c_worst_feasible": float(np.max(costs[feas])),
            "n_feasible": int(feas.sum()), "c_opt_unconstrained": float(costs[k_u]), "k_opt_unconstrained": k_u,
            "unconstrained_opt_feasible": bool(feas[k_u])}
    return qubo.int_to_bits(k, n), info


def anneal(h: np.ndarray, J: np.ndarray, t: np.ndarray, budget: float, sweeps: int = 300, restarts: int = 4,
           t_hi: float = 1.0, t_lo: float = 0.005, seed: int = 0) -> np.ndarray:
    """A5: simulated annealing with single bit flips; moves that break the budget are rejected,
    so every visited state is feasible. Geometric temperature schedule, best state kept."""
    rng = np.random.default_rng(seed)
    n = len(h)
    best_x, best_c = np.zeros(n, dtype=np.uint8), 0.0
    temps = t_hi * (t_lo / t_hi) ** (np.arange(sweeps) / max(1, sweeps - 1))
    for _ in range(restarts):
        x = np.zeros(n, dtype=np.uint8)
        field = np.zeros(n)  # J @ x
        c, used = 0.0, 0.0
        for T in temps:
            for i in rng.permutation(n):
                if x[i]:
                    d, dt = -(h[i] + field[i]), -t[i]
                else:
                    d, dt = h[i] + field[i], t[i]
                    if used + dt > budget:
                        continue
                if d <= 0 or rng.random() < math.exp(-d / T):
                    x[i] ^= 1
                    field += J[:, i] * (1 if x[i] else -1)
                    c += d
                    used += dt
                    if c < best_c - 1e-12:
                        best_c, best_x = c, x.copy()
    return best_x


def random_k(n: int, k: int, seed: int = 0) -> dict[str, int]:
    """A6 control: K uniform random bitstrings as {bits: count} (char i == qubit i), decoded like QAOA shots."""
    rng = np.random.default_rng(seed)
    if n > 30:  # long-context sizes: draw the bits directly (integers would overflow int64)
        bits = rng.integers(0, 2, size=(k, n), dtype=np.uint8)
        out: dict[str, int] = {}
        for row in bits:
            key = qubo.bits_to_str(row)
            out[key] = out.get(key, 0) + 1
        return out
    ks = rng.integers(0, 2 ** n, size=k)
    vals, cnt = np.unique(ks, return_counts=True)
    return {qubo.bits_to_str(qubo.int_to_bits(v, n)): int(c) for v, c in zip(vals, cnt)}

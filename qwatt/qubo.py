"""The QUBO (plan 6.3-6.5, 6.7). The ONLY place that evaluates cost.

    C(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j
    h_i  = -r_i^p + mu * tau_i              tau_i = t_i / tbar,  mu = kappa * eps_in * tbar
                                          (so the rent of sentence i is kappa * eps_in * t_i:
                                           kappa x measured joules of its tokens)
    J_ij = lam * red_ij - nu * br_ij

r^p (p = settings.extra["r_power"], tuned on dev, default 1) makes relevance utility convex: with a budget,
a plain sum of min-max relevances rewards many short mediocre sentences; squaring keeps [0,1] and the
ranking (so top-k is unaffected) but makes one strong sentence worth more than two weak ones.

Representation: h is (n,), J is a symmetric (n,n) matrix with zero diagonal; each pair i<j counted once,
i.e. C(x) = h.x + 0.5 * x^T J x.

kappa and calibration. kappa is "relevance per joule". It is tuned on dev as the dimensionless product
rho = kappa * eps_in (relevance per token). hyperparams.json stores rho and eps_in_ref (the eps_in in force
during tuning). When calibration later changes eps_in, `effective_kappa` returns rho / eps_in, i.e. kappa is
re-expressed in the new joule units so the tuned trade-off is preserved. Re-running tune_qubo.py after
calibration is cheap and is the clean fix.

Bit convention used everywhere: integer k <-> bits x_i = (k >> i) & 1, i.e. qubit i == bit i == sentence i.
"""
from __future__ import annotations

import math

import numpy as np

from qwatt.types import Pool


# ---------------------------------------------------------------- build
def effective_kappa(settings, eps_in: float) -> float:
    rho = (settings.extra or {}).get("rho")
    return float(rho) / eps_in if rho is not None else float(settings.kappa)


def build_qubo(pool: Pool, settings, eps_in: float) -> tuple[np.ndarray, np.ndarray]:
    t = pool.tokens
    tbar = float(t.mean())
    mu = effective_kappa(settings, eps_in) * eps_in * tbar
    r_power = float((settings.extra or {}).get("r_power", 1.0))
    h = -pool.relevance ** r_power + mu * (t / tbar)
    J = settings.lam * pool.red - settings.nu * pool.br
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)
    return h, J


def edges(J: np.ndarray, tol: float = 1e-12) -> list[tuple[int, int, float]]:
    n = J.shape[0]
    return [(i, j, float(J[i, j])) for i in range(n) for j in range(i + 1, n) if abs(J[i, j]) > tol]


# ---------------------------------------------------------------- cost
def qubo_cost(x, h: np.ndarray, J: np.ndarray):
    """Cost of one bit vector (n,) or a batch (m, n). Returns float or (m,) array."""
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        return float(X @ h + 0.5 * X @ J @ X)
    return X @ h + 0.5 * np.einsum("mi,ij,mj->m", X, J, X)


def tokens_of(x, t: np.ndarray):
    X = np.asarray(x, dtype=float)
    return X @ t


_ALL_BITS: dict[int, np.ndarray] = {}


def all_bits(n: int) -> np.ndarray:
    """(2^n, n) uint8 matrix, row k = bits of integer k (bit i = column i)."""
    if n not in _ALL_BITS:
        k = np.arange(2 ** n, dtype=np.int64)
        _ALL_BITS[n] = ((k[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    return _ALL_BITS[n]


def all_costs(h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """C(x) for every x in {0,1}^n, indexed by integer k. Vectorized; ~10 ms at n=16."""
    X = all_bits(len(h)).astype(np.float64)
    return X @ h + 0.5 * np.einsum("mi,mi->m", X @ J, X)


def bits_to_str(x) -> str:
    """Bit vector -> string, char i == qubit i (left to right = qubit 0..n-1)."""
    return "".join("1" if b else "0" for b in np.asarray(x).astype(int))


def str_to_bits(s: str) -> np.ndarray:
    return np.array([c == "1" for c in s], dtype=np.uint8)


def int_to_bits(k: int, n: int) -> np.ndarray:
    return ((int(k) >> np.arange(n)) & 1).astype(np.uint8)


def bits_to_int(x) -> int:
    return int(sum(int(b) << i for i, b in enumerate(np.asarray(x).astype(int))))


# ---------------------------------------------------------------- Ising (6.4) and normalization (6.5)
def to_ising(h: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """x_i = (1 - z_i)/2 :  C = c0 + sum_i a_i z_i + sum_{i<j} b_ij z_i z_j.

    Returns (a, b, c0) with b symmetric, zero diagonal (pair counted once in the sum over i<j).
    """
    Ju = np.triu(J, 1)
    b = J / 4.0
    a = -h / 2.0 - J.sum(axis=1) / 4.0
    c0 = h.sum() / 2.0 + Ju.sum() / 4.0
    return a, b, float(c0)


def ising_energy(z, a: np.ndarray, b: np.ndarray, c0: float = 0.0):
    Z = np.asarray(z, dtype=float)
    if Z.ndim == 1:
        return float(c0 + Z @ a + 0.5 * Z @ b @ Z)
    return c0 + Z @ a + 0.5 * np.einsum("mi,ij,mj->m", Z, b, Z)


def rms_scale(a: np.ndarray, b: np.ndarray) -> float:
    """s = sqrt(mean(b_ij^2 over edges) + mean(a_i^2)). Divide a, b by s before building the circuit."""
    iu = np.triu_indices(len(a), 1)
    bw = b[iu]
    bw = bw[np.abs(bw) > 1e-12]
    s = math.sqrt((float(np.mean(bw ** 2)) if bw.size else 0.0) + float(np.mean(a ** 2)))
    return s if s > 0 else 1.0


def normalized_ising(h: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    a, b, c0 = to_ising(h, J)
    s = rms_scale(a, b)
    return a / s, b / s, c0, s


# ---------------------------------------------------------------- decoding (6.7)
def make_samples(counts_bits: dict[str, int], h: np.ndarray, J: np.ndarray, t: np.ndarray, budget: float) -> list[dict]:
    """{bits (char i == qubit i): count} -> contract sample dicts, sorted by cost."""
    out = []
    for bits, c in counts_bits.items():
        x = str_to_bits(bits)
        tok = int(round(float(x @ t)))
        out.append({"bits": bits, "count": int(c), "cost": round(qubo_cost(x, h, J), 6), "tokens": tok,
                    "feasible": bool(tok <= budget)})
    out.sort(key=lambda d: (d["cost"], -d["count"]))
    return out


def decode_budget(samples: list[dict], budget: float, h: np.ndarray | None = None,
                  t: np.ndarray | None = None) -> tuple[str, bool]:
    """Best feasible sample for `budget` (no new circuit). Returns (bits, feasible_fallback).

    Feasibility is re-evaluated from each sample's 'tokens', so any budget works (the dashboard dial).
    If no sample fits: take the lowest-cost sample and drop its lowest-value sentences until it fits.
    "Value" of a kept sentence = -h_i (relevance minus rent); if h is not given, drop the longest first.
    """
    feas = [s for s in samples if s["tokens"] <= budget]
    if feas:
        return min(feas, key=lambda s: s["cost"])["bits"], False
    if t is None:  # no per-sentence tokens available: least-token sample (still flagged as fallback)
        return min(samples, key=lambda s: (s["tokens"], s["cost"]))["bits"], True
    best = min(samples, key=lambda s: s["cost"])
    x = str_to_bits(best["bits"]).astype(int)
    t = np.asarray(t, dtype=float)
    value = -np.asarray(h, dtype=float) if h is not None else -t
    while x @ t > budget and x.any():
        on = np.flatnonzero(x)
        x[on[np.argmin(value[on])]] = 0
    return bits_to_str(x), True


# ================================================================ long-context mode (additive, pivot 04:35)
def budget_matched_rho(relevance_pow: np.ndarray, t: np.ndarray, budget: float, fill: float = 1.0) -> float:
    """Rent per token at which the SEPARABLE part alone (keep i iff r_i^p > rho * t_i) fits fill * budget.

    This is the Lagrange multiplier of the budget for the diagonal problem: it puts the Hamiltonian's low-energy
    states at roughly the budget size, so the circuit's shots are not wasted far outside it. O(n log n), no solve.
    """
    v = relevance_pow / t
    order = np.argsort(-v)
    cum = np.cumsum(t[order])
    k = int(np.searchsorted(cum, fill * budget, side="right"))   # items 0..k-1 fit
    return float(v[order[k]]) * (1 + 1e-9) if k < len(v) else 0.0   # tiny margin: item k itself stays out


def build_qubo_lc(pool: Pool, settings, eps_in: float, budget: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Same objective as build_qubo; the rent per token rho = kappa * eps_in is either the tuned constant
    (rent_mode="fixed") or max(rho_floor, budget-matched rho) (rent_mode="budget_matched", default)."""
    ex = settings.extra or {}
    t = pool.tokens
    r_pow = pool.relevance ** float(ex.get("r_power", 1.0))
    budget = settings.budget_tokens if budget is None else budget
    rho = float(ex.get("rho", 0.0))
    if ex.get("rent_mode", "fixed") == "budget_matched":
        rho = max(rho, budget_matched_rho(r_pow, t, budget, float(ex.get("fill", 1.0))))
    h = -r_pow + rho * t
    J = settings.lam * pool.red - settings.nu * pool.br
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)
    tbar = float(t.mean())
    return h, J, {"rho": rho, "kappa": rho / eps_in, "mu": rho * tbar, "eps_in": eps_in,
                  "rent_mode": ex.get("rent_mode", "fixed")}


def repair(x, h: np.ndarray, J: np.ndarray, t: np.ndarray, budget: float) -> np.ndarray:
    """Drop the kept sentence with the worst marginal cost (h_i + sum_j J_ij x_j, largest first) until the
    selection fits the budget. Deterministic, classical, O(k n)."""
    x = np.asarray(x).astype(np.int8).copy()
    while x @ t > budget and x.any():
        on = np.flatnonzero(x)
        marg = h[on] + J[on] @ x
        x[on[np.argmax(marg)]] = 0
    return x.astype(np.uint8)


def local_polish(x, h: np.ndarray, J: np.ndarray, t: np.ndarray, budget: float, max_rounds: int = 50) -> np.ndarray:
    """1-flip local search that keeps feasibility (reported separately, never silently)."""
    x = np.asarray(x).astype(np.int8).copy()
    for _ in range(max_rounds):
        field = h + J @ x
        delta = np.where(x == 1, -field, field)           # cost change of flipping each bit
        fits = np.where(x == 1, True, x @ t + t <= budget)
        delta = np.where(fits, delta, np.inf)
        i = int(np.argmin(delta))
        if delta[i] >= -1e-12:
            break
        x[i] ^= 1
    return x.astype(np.uint8)


def decode_budget_lc(samples: list[dict], budget: float, h, J, t, n_repair: int = 64,
                     repair_mode: str = "fallback") -> tuple[str, dict]:
    """Best feasible selection from the shots (costs = ORIGINAL QUBO).
    repair_mode="fallback" (default): best raw feasible shot; only if NO shot fits, repair the n_repair lowest-cost
    shots to the budget. repair_mode="always": also compare repaired shots against raw feasible ones.
    Returns (bits, info) with info["source"] in {"raw_feasible", "repaired", "empty"}."""
    h, t = np.asarray(h, float), np.asarray(t, float)
    best_bits, best_cost, source = None, np.inf, None
    feas = [s for s in samples if s["tokens"] <= budget]
    if feas:
        f = min(feas, key=lambda s: s["cost"])
        best_bits, best_cost, source = f["bits"], f["cost"], "raw_feasible"
        if repair_mode == "fallback":
            return best_bits, {"source": source, "n_raw_feasible": len(feas), "cost": float(best_cost)}
    for s in sorted(samples, key=lambda s: s["cost"])[:n_repair]:
        if s["tokens"] <= budget:
            continue
        x = repair(str_to_bits(s["bits"]), h, J, t, budget)
        c = qubo_cost(x, h, J)
        if c < best_cost - 1e-12:
            best_bits, best_cost, source = bits_to_str(x), c, "repaired"
    if best_bits is None:
        best_bits, best_cost, source = "0" * len(h), 0.0, "empty"
    return best_bits, {"source": source, "n_raw_feasible": len(feas), "cost": float(best_cost)}


def cost_histograms(costs_a: np.ndarray, w_a: np.ndarray, costs_b: np.ndarray, w_b: np.ndarray, bins: int = 30) -> dict:
    """Two weighted cost histograms over the SAME bins (QAOA shots vs uniform random bitstrings)."""
    lo, hi = float(min(costs_a.min(), costs_b.min())), float(max(costs_a.max(), costs_b.max()))
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(costs_a, bins=edges, weights=w_a)
    hb, _ = np.histogram(costs_b, bins=edges, weights=w_b)
    return {"edges": [float(e) for e in edges], "qaoa": [float(v) for v in ha / max(1, w_a.sum())],
            "random": [float(v) for v in hb / max(1, w_b.sum())]}


def calibrate_circuit_rent(h: np.ndarray, J: np.ndarray, t: np.ndarray, relevance_pow: np.ndarray, budget: float,
                           gamma: float, beta: float, rho0: float = 0.0, target: float | None = None) -> dict:
    """Budget-calibrated circuit rent: raise the energy price in the CIRCUIT Hamiltonian until the circuit's
    typical measurement fits the budget.

    Circuit Hamiltonian: h_c = h + mu_c (a price per kept sentence, i.e. the joule price of an average sentence;
    a per-token price saturates under RMS normalization because the fields become proportional to t_i and a single
    p=1 layer cannot bias them all), RMS-normalized, then scaled by `scale` (equivalently gamma_eff = gamma * scale;
    beta unchanged). For each (mu_c, scale) the EXACT p=1 marginals (qaoa_sim.p1_marginals) give a shot's token count
    as a Poisson-binomial: E[T] = sum t_i P_i, Var = sum t_i^2 P_i (1-P_i), P(T <= B) ~ Phi((B + 0.5 - E)/sd).
    Score = P(T <= B) * E[sum_i r_i^p x_i] (fits the budget AND carries relevance); best score wins.
    With `target` (e.g. 0.8): maximize E[sum r^p x] subject to E[T] <= target * B (bigger shots, more relevance).
    Decoding still scores every shot with the ORIGINAL QUBO."""
    from math import erf, sqrt

    from qwatt.qaoa_sim import p1_marginals

    best = None
    for mu_c in np.geomspace(0.01, 5.0, 30):
        a, b, _, s0 = normalized_ising(h + mu_c, J)
        for scale in np.geomspace(0.3, 3.0, 20):
            P = p1_marginals(a * scale, b * scale, gamma, beta)
            mean, var = float(P @ t), float((t ** 2) @ (P * (1 - P)))
            p_feas = 0.5 * (1 + erf((budget + 0.5 - mean) / sqrt(2 * max(var, 1e-9))))
            rel = float(P @ relevance_pow)
            score = p_feas * rel if target is None else (rel if mean <= target * budget else -mean)
            if best is None or score > best["score"]:
                best = {"mu_circuit": float(mu_c), "scale": float(scale), "gamma_effective": float(gamma * scale),
                        "predicted_shot_tokens": mean, "predicted_feasible_frac": p_feas, "score": score,
                        "a": a * scale, "b": b * scale}
    return best

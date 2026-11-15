"""Long-context mode: local-only couplings, 200-bit decode order, pool sanity, p=1 closed-form marginals."""
import numpy as np
import pytest

from qwatt import longdoc, qaoa_qr, qaoa_sim, qubo


def _pool():
    if not longdoc.SUBSET_JSON.exists():
        pytest.skip("no LC subset")
    s = longdoc.lc_settings()
    return longdoc.get_lc_pool(longdoc.load_lc_split("lc_dev")[0]["id"], s), s


def test_pool_sanity_and_local_edges():
    pool, s = _pool()
    W = pool.meta["W"]
    assert pool.n <= longdoc.MAX_SENTS and pool.n == len(pool.meta["paragraph_of"])
    assert pool.n_gold_total >= 1 and any(x.is_gold for x in pool.sentences)
    assert all(x.idx == i for i, x in enumerate(pool.sentences))           # document order == qubit order
    h, J, info = qubo.build_qubo_lc(pool, s, 0.06)
    i, j = np.nonzero(J)
    assert len(i) > 0 and np.all(np.abs(i - j) <= W)                       # couplings are local only
    assert np.allclose(J, J.T) and np.all(np.diag(J) == 0)
    assert info["rho"] > 0
    assert pool.full_tokens > pool.tokens.sum() * 0.9


def test_decode_200_bits_order():
    n = 200
    key_qiskit = "".join("1" if q == 7 else "0" for q in reversed(range(n)))   # qubit 0 rightmost in Qiskit
    bits = qaoa_qr.decode_key(key_qiskit)
    assert len(bits) == n and bits[7] == "1" and bits.count("1") == 1
    x = qubo.str_to_bits(bits)
    assert x[7] == 1 and qubo.bits_to_str(x) == bits


def test_repair_and_decode_lc_respect_budget():
    rng = np.random.default_rng(0)
    n = 60
    h = rng.normal(size=n)
    J = np.zeros((n, n))
    for k in range(n - 1):
        J[k, k + 1] = J[k + 1, k] = rng.normal() * 0.3
    t = rng.integers(10, 50, size=n).astype(float)
    counts = {qubo.bits_to_str(rng.integers(0, 2, n)): 1 for _ in range(50)}
    samples = qubo.make_samples(counts, h, J, t, 200)
    bits, info = qubo.decode_budget_lc(samples, 200, h, J, t)
    assert qubo.str_to_bits(bits) @ t <= 200 and info["source"] in ("raw_feasible", "repaired")
    xp = qubo.local_polish(qubo.str_to_bits(bits), h, J, t, 200)
    assert xp @ t <= 200 and qubo.qubo_cost(xp, h, J) <= qubo.qubo_cost(qubo.str_to_bits(bits), h, J) + 1e-12


def test_p1_marginals_closed_form():
    rng = np.random.default_rng(1)
    n = 8
    a = rng.normal(size=n)
    b = np.triu(rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.4), 1)
    b = b + b.T
    p = qaoa_sim.probabilities(qaoa_sim.ising_diagonal(a, b), [0.37], [2.2])
    exact = p @ qubo.all_bits(n)
    assert np.max(np.abs(exact - qaoa_sim.p1_marginals(a, b, 0.37, 2.2))) < 1e-10


def test_budget_matched_rho_fits():
    r = np.linspace(1, 0, 50) ** 2
    t = np.full(50, 20.0)
    rho = qubo.budget_matched_rho(r, t, 200)
    assert t[r > rho * t].sum() <= 200


def test_circuit_rent_calibration_fits_budget():
    """Calibrated circuit Hamiltonian: predicted shot size (exact p=1 marginals) fits the budget on a real LC pool."""
    pool, s = _pool()
    h, J, info = qubo.build_qubo_lc(pool, s, 0.06)
    cal = qubo.calibrate_circuit_rent(h, J, pool.tokens, pool.relevance ** 2, s.budget_tokens, 1.0, 2.39)
    assert cal["predicted_shot_tokens"] < s.budget_tokens and cal["predicted_feasible_frac"] > 0.5
    P = qaoa_sim.p1_marginals(cal["a"], cal["b"], 1.0, 2.39)
    assert abs(float(P @ pool.tokens) - cal["predicted_shot_tokens"]) < 1e-6

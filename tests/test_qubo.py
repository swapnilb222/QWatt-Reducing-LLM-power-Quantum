import numpy as np

from qwatt import qubo


def _random_instance(n=10, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=n)
    J = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.4)
    J = np.triu(J, 1)
    J = J + J.T
    return h, J


def test_ising_equals_qubo_all_bitstrings():
    h, J = _random_instance(12, seed=1)
    a, b, c0 = qubo.to_ising(h, J)
    X = qubo.all_bits(12).astype(float)
    Z = 1 - 2 * X
    c_qubo = qubo.all_costs(h, J)
    c_ising = qubo.ising_energy(Z, a, b, c0)
    assert np.max(np.abs(c_qubo - c_ising)) < 1e-10


def test_all_costs_matches_pointwise():
    h, J = _random_instance(8, seed=2)
    allc = qubo.all_costs(h, J)
    for k in [0, 1, 5, 77, 255]:
        x = qubo.int_to_bits(k, 8)
        direct = sum(h[i] * x[i] for i in range(8)) + sum(J[i, j] * x[i] * x[j] for i in range(8) for j in range(i + 1, 8))
        assert abs(allc[k] - direct) < 1e-12
        assert abs(qubo.qubo_cost(x, h, J) - direct) < 1e-12
        assert qubo.bits_to_int(x) == k


def test_rms_normalization_scale_invariant():
    h, J = _random_instance(10, seed=3)
    a1, b1, _, _ = qubo.normalized_ising(h, J)
    a2, b2, _, _ = qubo.normalized_ising(7.5 * h, 7.5 * J)
    assert np.allclose(a1, a2) and np.allclose(b1, b2)


def test_decode_budget_and_fallback():
    h = np.array([-1.0, -0.5, -0.2])
    J = np.zeros((3, 3))
    t = np.array([10.0, 10.0, 10.0])
    samples = qubo.make_samples({"111": 5, "110": 3, "001": 1}, h, J, t, budget=20)
    bits, fb = qubo.decode_budget(samples, 20, h, t)
    assert bits == "110" and not fb
    bits, fb = qubo.decode_budget(samples, 10, h, t)
    assert bits == "001" and not fb  # only feasible sample
    bits, fb = qubo.decode_budget([s for s in samples if s["bits"] == "111"], 10, h, t)
    assert fb and bits == "100"  # dropped the two lowest-value sentences

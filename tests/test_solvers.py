import itertools

import numpy as np

from qwatt import dispatch, qubo, solvers_classical as sc


def _inst(n=10, seed=0):
    rng = np.random.default_rng(seed)
    h = -rng.random(n) + 0.2
    J = np.triu(rng.normal(scale=0.4, size=(n, n)) * (rng.random((n, n)) < 0.3), 1)
    J = J + J.T
    t = rng.integers(5, 40, size=n).astype(float)
    return h, J, t


def test_brute_force_matches_enumeration():
    h, J, t = _inst(9, seed=1)
    B = 80
    x, info = sc.brute_force(h, J, t, B)
    best = min((qubo.qubo_cost(np.array(b), h, J), b) for b in itertools.product([0, 1], repeat=9)
               if np.array(b) @ t <= B)
    assert abs(info["c_opt"] - best[0]) < 1e-12
    assert x @ t <= B
    assert abs(qubo.qubo_cost(x, h, J) - best[0]) < 1e-12


def test_heuristics_feasible_and_not_better_than_exact():
    for seed in range(5):
        h, J, t = _inst(12, seed=seed)
        B = 90
        _, info = sc.brute_force(h, J, t, B)
        for x in (sc.greedy(h, J, t, B), sc.anneal(h, J, t, B, seed=seed), sc.topk(-h, t, B)):
            assert x @ t <= B
            assert qubo.qubo_cost(x, h, J) >= info["c_opt"] - 1e-12


def test_anneal_finds_optimum_on_small_instances():
    hits = 0
    for seed in range(5):
        h, J, t = _inst(10, seed=seed)
        _, info = sc.brute_force(h, J, t, 100)
        hits += abs(qubo.qubo_cost(sc.anneal(h, J, t, 100, seed=seed), h, J) - info["c_opt"]) < 1e-9
    assert hits >= 4


def test_dispatch_closed_forms():
    h = np.array([-1.0, -0.5])
    t = np.array([10.0, 10.0])
    assert dispatch.decide(h, np.zeros((2, 2)), t, 100)[0] == "closed_form_fits"
    h2 = np.array([-1.0, 0.5])
    route, x = dispatch.decide(h2, np.zeros((2, 2)), t, 100)
    assert route == "closed_form_separable" and list(x) == [1, 0]
    J = np.array([[0, 0.3], [0.3, 0]])
    assert dispatch.decide(h, J, t, 100)[0] == "circuit"


def test_random_k_counts():
    c = sc.random_k(8, 500, seed=0)
    assert sum(c.values()) == 500 and all(len(k) == 8 for k in c)

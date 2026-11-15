"""Solver quality on real TEST instances (plan 6.9, gate G3). Writes results/solver_quality.json.

Per instance (same frozen QUBO, same budget):
  qaoa_qr   QAOA p=1, stored angles, ONE execution on Quantum Rings (QrBackendV2), best feasible shot
  exact     brute force over 2^16 (the oracle)
  anneal    simulated annealing (budget-feasible moves)
  greedy    greedy marginal gain
  random_k  K = shots uniform random bitstrings, decoded exactly like the QAOA shots
Metrics: approximation ratio over feasible solutions, optimum hit, amplification of the best 1% of all 2^n
bitstrings (from the shots, and from the exact numpy statevector), wall time.

Cross-check (first N_TVD instances): QR counts vs exact numpy probabilities, total variation distance.
At n=16 with 8192 shots over 65,536 states the empirical TVD is dominated by sampling noise, so we report
the noise floor (TVD of an ideal multinomial sample of the same size), plus TVD on the Hamming-weight
distribution and the max per-qubit marginal error, and a clean full-distribution TVD on an 8-qubit
sub-instance (256 states) where 8192 shots resolve the distribution.

Run:  .venv/Scripts/python scripts/eval_solvers.py [n_instances] [split]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from qwatt import config, data, qaoa_qr, qaoa_sim, qubo, solvers_classical as sc  # noqa: E402
from qwatt.selectors import _qid_seed, load_angles  # noqa: E402

N_TVD = 5


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def counts_to_probs(counts: dict[str, int], n: int) -> np.ndarray:
    p = np.zeros(2 ** n)
    for bits, c in counts.items():
        p[qubo.bits_to_int(qubo.str_to_bits(bits))] += c
    return p / p.sum()


def hamming_dist(p: np.ndarray, n: int) -> np.ndarray:
    w = qubo.all_bits(n).sum(axis=1)
    return np.bincount(w, weights=p, minlength=n + 1)


def marginals(p: np.ndarray, n: int) -> np.ndarray:
    return p @ qubo.all_bits(n)


def cross_check(a, b, angles, shots, rng) -> dict:
    n = len(a)
    diag = qaoa_sim.ising_diagonal(a, b)
    p_exact = qaoa_sim.probabilities(diag, angles["gamma"], angles["beta"])
    counts, _ = qaoa_qr.run_qaoa(a, b, angles["gamma"], angles["beta"], shots=shots)
    p_qr = counts_to_probs(counts, n)
    p_ideal = rng.multinomial(shots, p_exact) / shots
    return {"n_qubits": n, "shots": shots, "tvd_qr_vs_exact": tvd(p_qr, p_exact),
            "tvd_noise_floor": tvd(p_ideal, p_exact),
            "tvd_hamming_qr": tvd(hamming_dist(p_qr, n), hamming_dist(p_exact, n)),
            "tvd_hamming_floor": tvd(hamming_dist(p_ideal, n), hamming_dist(p_exact, n)),
            "max_marginal_err_qr": float(np.max(np.abs(marginals(p_qr, n) - marginals(p_exact, n)))),
            "max_marginal_err_floor": float(np.max(np.abs(marginals(p_ideal, n) - marginals(p_exact, n))))}


def main(n_inst: int = 20, split: str = "test") -> None:
    settings = config.load_settings()
    eps_in = config.eps_in_default()
    B, K = settings.budget_tokens, settings.shots
    angles = load_angles()
    rng = np.random.default_rng(settings.seed)
    qaoa_qr.get_backend(settings.n_pool)  # warm the backend once (provider construction is not per query)
    rows, tvds = [], []
    for k, it in enumerate(data.load_split(split)[:n_inst]):
        pool = data.get_pool(it["_id"], settings)
        h, J = qubo.build_qubo(pool, settings, eps_in)
        t = pool.tokens
        n = pool.n

        t0 = time.perf_counter()
        costs = qubo.all_costs(h, J)
        x_opt, info = sc.brute_force(h, J, t, B, costs=costs)
        w_exact = time.perf_counter() - t0
        c_opt, c_worst = info["c_opt"], info["c_worst_feasible"]
        ar = (lambda c: float((c_worst - c) / (c_worst - c_opt)) if c_worst - c_opt > 1e-12 else 1.0)
        hit = (lambda c: bool(abs(c - c_opt) < 1e-9))
        top = qaoa_sim.top_mask(costs, 0.01)

        def shot_metrics(counts, wall):
            samples = qubo.make_samples(counts, h, J, t, B)
            bits, fb = qubo.decode_budget(samples, B, h, t)
            c = qubo.qubo_cost(qubo.str_to_bits(bits), h, J)
            p = counts_to_probs(counts, n)
            return {"cost": c, "approx_ratio": ar(c), "opt_hit": hit(c), "wall_s": wall, "fallback": fb,
                    "amplification": float(p[top].sum() / (top.sum() / costs.size)),
                    "feasible_shot_frac": float(sum(s["count"] for s in samples if s["feasible"]) / K),
                    "n_distinct": len(samples)}

        a, b, c0, s = qubo.normalized_ising(h, J)
        t0 = time.perf_counter()
        counts, cinfo = qaoa_qr.run_qaoa(a, b, angles["gamma"], angles["beta"], shots=K)
        w_q = time.perf_counter() - t0
        row = {"qid": pool.qid, "n_qubits": n, "n_edges": cinfo["n_edges"], "depth": cinfo["depth"],
               "gate_count": cinfo["gate_count"], "c_opt": c_opt, "c_worst_feasible": c_worst,
               "n_feasible": info["n_feasible"], "unconstrained_opt_feasible": info["unconstrained_opt_feasible"]}
        row["qaoa_qr"] = shot_metrics(counts, w_q)
        row["qaoa_qr"]["circuit_wall_s"] = cinfo["circuit_wall_s"]

        p_exact = qaoa_sim.probabilities(qaoa_sim.ising_diagonal(a, b), angles["gamma"], angles["beta"])
        p_opt = float(p_exact[info["k_opt"]])
        row["qaoa_exact_sim"] = {"amplification": qaoa_sim.amplification(p_exact, costs), "p_feasible_opt": p_opt,
                                 "p_opt_uniform": 1 / costs.size, "opt_amplification": p_opt * costs.size,
                                 "p_hit_in_K_shots": float(1 - (1 - p_opt) ** K)}
        row["exact"] = {"cost": c_opt, "approx_ratio": 1.0, "opt_hit": True, "wall_s": w_exact}
        for name, fn in (("anneal", lambda: sc.anneal(h, J, t, B, seed=settings.seed)), ("greedy", lambda: sc.greedy(h, J, t, B))):
            t0 = time.perf_counter()
            x = fn()
            w = time.perf_counter() - t0
            c = qubo.qubo_cost(x, h, J)
            row[name] = {"cost": c, "approx_ratio": ar(c), "opt_hit": hit(c), "wall_s": w}
        t0 = time.perf_counter()
        rk = sc.random_k(n, K, seed=_qid_seed(pool.qid, settings.seed))
        row["random_k"] = shot_metrics(rk, time.perf_counter() - t0)
        rows.append(row)
        q = row["qaoa_qr"]
        print(f"{k + 1:2d} {pool.qid} edges={row['n_edges']:2d} QR: AR={q['approx_ratio']:.3f} hit={q['opt_hit']} "
              f"amp={q['amplification']:.1f} wall={q['wall_s']:.2f}s | randK AR={row['random_k']['approx_ratio']:.3f} "
              f"hit={row['random_k']['opt_hit']} | anneal hit={row['anneal']['opt_hit']} greedy hit={row['greedy']['opt_hit']}",
              flush=True)
        if k < N_TVD:
            cc = cross_check(a, b, angles, K, rng)
            m = 8
            cc8 = cross_check(a[:m], b[:m, :m], angles, K, rng)
            tvds.append({"qid": pool.qid, "n16": cc, "n8_subinstance": cc8})
            print(f"   TVD n16 qr={cc['tvd_qr_vs_exact']:.3f} floor={cc['tvd_noise_floor']:.3f} hamming qr={cc['tvd_hamming_qr']:.4f} "
                  f"floor={cc['tvd_hamming_floor']:.4f} | n8 qr={cc8['tvd_qr_vs_exact']:.4f} floor={cc8['tvd_noise_floor']:.4f}", flush=True)

    solvers = ["qaoa_qr", "exact", "anneal", "greedy", "random_k"]
    summ = {}
    for sname in solvers:
        v = [r[sname] for r in rows]
        summ[sname] = {"approx_ratio_mean": float(np.mean([x["approx_ratio"] for x in v])),
                       "approx_ratio_min": float(np.min([x["approx_ratio"] for x in v])),
                       "opt_hit_rate": float(np.mean([x["opt_hit"] for x in v])),
                       "wall_s_mean": float(np.mean([x["wall_s"] for x in v])),
                       "wall_s_max": float(np.max([x["wall_s"] for x in v]))}
        if "amplification" in v[0]:
            summ[sname]["amplification_median"] = float(np.median([x["amplification"] for x in v]))
            summ[sname]["amplification_mean"] = float(np.mean([x["amplification"] for x in v]))
            summ[sname]["fallback_rate"] = float(np.mean([x["fallback"] for x in v]))
            summ[sname]["feasible_shot_frac_mean"] = float(np.mean([x["feasible_shot_frac"] for x in v]))
    es = [r["qaoa_exact_sim"] for r in rows]
    summ["qaoa_exact_sim"] = {"amplification_median": float(np.median([x["amplification"] for x in es])),
                              "opt_amplification_median": float(np.median([x["opt_amplification"] for x in es])),
                              "p_hit_in_K_shots_mean": float(np.mean([x["p_hit_in_K_shots"] for x in es]))}
    summ["circuit"] = {"edges_mean": float(np.mean([r["n_edges"] for r in rows])),
                       "edges_max": int(np.max([r["n_edges"] for r in rows])),
                       "depth_mean": float(np.mean([r["depth"] for r in rows])),
                       "gate_count_mean": float(np.mean([r["gate_count"] for r in rows])),
                       "circuit_wall_s_mean": float(np.mean([r["qaoa_qr"]["circuit_wall_s"] for r in rows]))}
    g3 = {"opt_hit_rate_qr": summ["qaoa_qr"]["opt_hit_rate"], "opt_hit_rate_random_k": summ["random_k"]["opt_hit_rate"],
          "amplification_median_qr": summ["qaoa_qr"]["amplification_median"],
          "selector_wall_max_s": summ["qaoa_qr"]["wall_s_max"]}
    g3["pass"] = bool(g3["opt_hit_rate_qr"] > 0.5 and g3["amplification_median_qr"] > 5 and g3["selector_wall_max_s"] < 5)
    out = {"split": split, "n": len(rows), "budget_tokens": B, "shots": K, "angles": angles,
           "backend": qaoa_qr.get_backend(settings.n_pool).name, "summary": summ, "G3": g3, "tvd": tvds,
           "per_instance": rows, "settings_hash": settings.hash()}
    config.SOLVER_QUALITY_JSON.write_text(json.dumps(out, indent=1))
    print(json.dumps({"summary": summ, "G3": g3}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 20, a[1] if len(a) > 1 else "test")

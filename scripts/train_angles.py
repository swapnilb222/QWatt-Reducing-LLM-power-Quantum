"""Train QAOA angles once, offline, on ~20 dev instances (plan 6.6) with the exact numpy statevector.

Per instance: build the frozen QUBO, map to Ising, RMS-normalize, minimize <H_C> over (gamma, beta)
(grid + Nelder-Mead). Median angles over instances -> results/angles.json. Then the median ("transferred")
angles are evaluated on every training instance: expectation gap, amplification of the top 1%, probability of
the budget-feasible optimum.

Run:  .venv/Scripts/python scripts/train_angles.py [p] [n_train]
      .venv/Scripts/python scripts/train_angles.py p2exp      # p=2 check (G3 remedy), appended to angles.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from qwatt import config, data, qaoa_sim, qubo  # noqa: E402
from qwatt.solvers_classical import brute_force  # noqa: E402


def main(p: int = 1, n_train: int = 20) -> None:
    settings = config.load_settings()
    eps_in = config.eps_in_default()
    items = data.load_split("dev")[:n_train]
    per, inst = [], []
    t0 = time.perf_counter()
    for it in items:
        pool = data.get_pool(it["_id"], settings)
        h, J = qubo.build_qubo(pool, settings, eps_in)
        a, b, c0, s = qubo.normalized_ising(h, J)
        diag = qaoa_sim.ising_diagonal(a, b)
        g, bt, e = qaoa_sim.train_instance(diag, p=p, grid=16)
        costs = qubo.all_costs(h, J)
        x_opt, info = brute_force(h, J, pool.tokens, settings.budget_tokens, costs=costs)
        inst.append((pool.qid, diag, costs, info))
        per.append({"qid": pool.qid, "gamma": g, "beta": bt, "exp_opt": e, "e_min": float(diag.min()),
                    "e_mean_uniform": float(diag.mean()), "n_edges": len(qubo.edges(J)), "scale": s})
        print(f"{pool.qid}: gamma={np.round(g, 3).tolist()} beta={np.round(bt, 3).tolist()} <H>={e:.3f} "
              f"min={diag.min():.3f} edges={per[-1]['n_edges']}")
    G = np.array([r["gamma"] for r in per])
    Bt = np.array([r["beta"] for r in per])
    gamma_med = np.median(G, axis=0).tolist()
    beta_med = np.median(Bt, axis=0).tolist()

    amps, p_opt, gaps, amps_own = [], [], [], []
    for (qid, diag, costs, info), r in zip(inst, per):
        probs = qaoa_sim.probabilities(diag, gamma_med, beta_med)
        e_med = float(probs @ diag)
        amps.append(qaoa_sim.amplification(probs, costs))
        amps_own.append(qaoa_sim.amplification(qaoa_sim.probabilities(diag, r["gamma"], r["beta"]), costs))
        p_opt.append(float(probs[info["k_opt"]]))
        # fraction of the (uniform -> optimum) energy gap closed by transferred vs own angles
        gaps.append((r["e_mean_uniform"] - e_med) / max(1e-12, r["e_mean_uniform"] - r["exp_opt"]))
        r.update({"exp_median_angles": e_med, "amplification_median_angles": amps[-1],
                  "amplification_own_angles": amps_own[-1], "p_feasible_opt": p_opt[-1],
                  "p_feasible_opt_uniform": 1 / len(diag),
                  "gap_closed_by_transfer": gaps[-1]})
    n = int(np.log2(inst[0][1].size))
    out = {"p": p, "gamma": gamma_med, "beta": beta_med, "n_train": len(per), "normalization": "rms",
           "objective": "min <H_C> (normalized Ising), exact numpy statevector",
           "split": "dev", "n_qubits": n, "train_wall_s": time.perf_counter() - t0,
           "summary": {"amplification_top1pct_median": float(np.median(amps)),
                       "amplification_top1pct_mean": float(np.mean(amps)),
                       "amplification_own_angles_median": float(np.median(amps_own)),
                       "p_feasible_opt_median": float(np.median(p_opt)), "uniform_p": 1 / 2 ** n,
                       "expected_hits_in_1024_shots_median": float(np.median(p_opt) * 1024),
                       "transfer_gap_closed_median": float(np.median(gaps)),
                       "gamma_iqr": np.percentile(G, [25, 75], axis=0).tolist(),
                       "beta_iqr": np.percentile(Bt, [25, 75], axis=0).tolist()},
           "per_instance": per}
    config.ANGLES_JSON.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_instance"}, indent=1))


def p2_experiment(n_train: int = 10, n_eval: int = 20, shots: int = 8192) -> dict:
    """Plan G3 remedy check: p=1 median angles vs p=1 / p=2 angles trained JOINTLY on n_train dev instances
    (objective: mean normalized energy (<H>-mean)/(mean-min)). Evaluated on n_eval dev instances.
    Appends the result to angles.json under "experiments" (does not change the shipped angles)."""
    from scipy.optimize import minimize

    settings = config.load_settings()
    eps_in = config.eps_in_default()
    ang = json.loads(config.ANGLES_JSON.read_text())
    inst = []
    for it in data.load_split("dev")[:n_eval]:
        pool = data.get_pool(it["_id"], settings)
        h, J = qubo.build_qubo(pool, settings, eps_in)
        a, b, _, _ = qubo.normalized_ising(h, J)
        d = qaoa_sim.ising_diagonal(a, b)
        costs = qubo.all_costs(h, J)
        _, info = brute_force(h, J, pool.tokens, settings.budget_tokens, costs=costs)
        inst.append((d, costs, info["k_opt"], d.mean(), d.min()))

    def obj(v, p):
        return float(np.mean([(qaoa_sim.expectation(d, v[:p], v[p:]) - m) / (m - mn) for d, _, _, m, mn in inst[:n_train]]))

    def report(g, bt):
        probs = [qaoa_sim.probabilities(d, g, bt) for d, *_ in inst]
        po = [pr[k] for pr, (_, _, k, _, _) in zip(probs, inst)]
        return {"gamma": [float(x) for x in g], "beta": [float(x) for x in bt],
                "p_feasible_opt_median": float(np.median(po)),
                "p_hit_in_shots_mean": float(np.mean([1 - (1 - x) ** shots for x in po])),
                "amplification_median": float(np.median([qaoa_sim.amplification(pr, c) for pr, (_, c, *_) in zip(probs, inst)]))}

    out = {"shots": shots, "n_train": n_train, "n_eval": n_eval, "p1_median": report(ang["gamma"], ang["beta"])}
    r1 = minimize(obj, [ang["gamma"][0], ang["beta"][0]], args=(1,), method="Nelder-Mead", options={"maxiter": 120})
    out["p1_joint"] = report(r1.x[:1], r1.x[1:])
    g0, b0 = r1.x
    x0 = [g0 * 0.5, g0, b0, b0 * 0.5 if b0 < np.pi / 2 else (b0 + np.pi) / 2]
    r2 = minimize(obj, x0, args=(2,), method="Nelder-Mead", options={"maxiter": 300})
    out["p2_joint"] = report(r2.x[:2], r2.x[2:])
    ang.setdefault("experiments", {})["p2_joint"] = out
    config.ANGLES_JSON.write_text(json.dumps(ang, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["p2exp"]:
        p2_experiment()
    else:
        args = [int(a) for a in sys.argv[1:]]
        main(*args)

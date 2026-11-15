"""Train p=1 angles for the long-context Hamiltonian on SMALL WINDOWS (exact numpy), transfer to the full document.

For each lc_dev question: build the full-document QUBO (rent matched on the whole document), cut a window of
WIN consecutive sentences (half the windows contain the most relevant sentence, half are random), RMS-normalize the
window's Ising, minimize <H_C> with the exact statevector. Median angles -> results/lc_angles.json.

Transfer checks:
  * n = 18 windows: <H> at the median angles vs each window's own optimum (fraction of the gap closed).
  * full documents (n ~ 120-200): exact p=1 single-qubit marginals (closed form, qaoa_sim.p1_marginals) give the
    expected tokens per shot vs the budget, at the median angles and at a gamma sweep.

Run:  .venv/Scripts/python scripts/train_lc_angles.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from qwatt import config, longdoc, qaoa_sim, qubo  # noqa: E402

WIN = 16
N_CHECK = 6
N_CHECK_QUBITS = 18


def windows(pool, rng, n):
    top = int(np.argmax(pool.relevance))
    s1 = int(np.clip(top - n // 2, 0, pool.n - n))
    s2 = int(rng.integers(0, pool.n - n + 1))
    return [s1, s2]


def norm_window(h, J, start, n):
    hw, Jw = h[start:start + n], J[start:start + n, start:start + n]
    a, b, _, _ = qubo.normalized_ising(hw, Jw)
    return a, b


def main() -> None:
    s = longdoc.lc_settings()
    eps_in = config.eps_in_default()
    rng = np.random.default_rng(s.seed)
    qs = longdoc.load_lc_split("lc_dev")
    t0 = time.perf_counter()
    per, full = [], []
    for k, q in enumerate(qs[:20]):
        pool = longdoc.get_lc_pool(q["id"], s)
        h, J, _ = qubo.build_qubo_lc(pool, s, eps_in, s.budget_tokens)
        full.append((pool, h, J))
        st = windows(pool, rng, WIN)[k % 2]
        a, b = norm_window(h, J, st, WIN)
        g, bt, e = qaoa_sim.train_instance(qaoa_sim.ising_diagonal(a, b), p=1, grid=16)
        per.append({"qid": q["id"], "start": st, "gamma": g, "beta": bt, "exp_opt": e})
        print(f"{q['id']} win@{st}: gamma={g[0]:.3f} beta={bt[0]:.3f}", flush=True)
    gamma = [float(np.median([r["gamma"][0] for r in per]))]
    beta = [float(np.median([r["beta"][0] for r in per]))]

    # transfer check at n = 20 (exact)
    chk = []
    for pool, h, J in full[:N_CHECK // 2]:
        for st in windows(pool, rng, N_CHECK_QUBITS):
            a, b = norm_window(h, J, st, N_CHECK_QUBITS)
            d = qaoa_sim.ising_diagonal(a, b)
            g_o, b_o, e_o = qaoa_sim.train_instance(d, p=1, grid=8)
            e_t = qaoa_sim.expectation(d, gamma, beta)
            chk.append({"gap_closed": float((d.mean() - e_t) / (d.mean() - e_o)), "own_gamma": g_o[0], "own_beta": b_o[0]})
    # full-document shot size from exact p=1 marginals
    shot = {}
    for gm in sorted({gamma[0], 0.2, 0.4, 0.6, 0.8, 1.0, 1.2}):
        v = []
        for pool, h, J in full:
            a, b, _, _ = qubo.normalized_ising(h, J)
            v.append(float(qaoa_sim.p1_marginals(a, b, gm, beta[0]) @ pool.tokens))
        shot[f"{gm:.3f}"] = {"shot_tokens_median": float(np.median(v)), "over_budget_x": float(np.median(v) / s.budget_tokens)}
    out = {"p": 1, "gamma": gamma, "beta": beta, "normalization": "rms (per instance, full document at runtime)",
           "trained_on": f"lc_dev windows of {WIN} sentences (exact numpy statevector)", "n_train": len(per),
           "transfer_n18": {"gap_closed_median": float(np.median([c["gap_closed"] for c in chk])),
                            "gap_closed_min": float(np.min([c["gap_closed"] for c in chk])), "n": len(chk)},
           "full_doc_shot_tokens": shot, "budget_tokens": s.budget_tokens, "per_instance": per, "checks": chk,
           "wall_s": time.perf_counter() - t0}
    longdoc.LC_ANGLES_JSON.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_instance", "checks")}, indent=1))


if __name__ == "__main__":
    main()

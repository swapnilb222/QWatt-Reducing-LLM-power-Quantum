"""Tune lam, nu, rho (= kappa * eps_in), r_power, theta, B on the TUNE split (dev + 170 more non-test questions)
with the exact solver (gate G2). The 30-question dev split alone overfit (dev +6 pts, test below top-k).

For each instance the cost of every bitstring is a linear combination of four precomputed vectors:
    C = -X.r + rho * X.t + lam * R_theta(X) - nu * Br(X)
so a full grid costs a few seconds. Objective: mean supporting-fact recall of the budget-feasible exact
optimum. Ties (within TIE_TOL) are broken by how often the UNCONSTRAINED optimum is already within budget
(that is what QAOA samples, so it matters for the circuit), then by smaller parameters.

Writes results/hyperparams.json (frozen) and results/tune_qubo.json (full grid summary).
Run:  .venv/Scripts/python scripts/tune_qubo.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from qwatt import config, data, qubo  # noqa: E402
from qwatt.embed import EMBED_BACKEND  # noqa: E402
from qwatt.solvers_classical import topk  # noqa: E402

TUNE_SPLIT = "tune"                          # dev (30) + 170 more non-test questions; 30 alone overfit
LAMS = [0.0, 0.5, 1.0]
NUS = [0.0, 0.1, 0.25]
# rho = relevance per token (rent of a 30-token sentence = 30*rho). Floor at 0.0025: the energy rent is part of the
# design (and the Lagrangian relaxation of the budget). rho=0 scored ~1 pt higher on tune but then the Hamiltonian's
# ground state almost never respects the budget (0.5% of questions), which starves the circuit of feasible shots.
RHOS = [0.0025, 0.005, 0.0075, 0.01]
THETAS = [0.2, 0.3]
R_POWERS = [2, 3]                            # r_power=1 was never competitive in the earlier, wider grid
TIE_TOL = 0.01                               # within 1 recall point of the best -> prefer budget-respecting ground state
RECALL_FLOOR = 0.79                          # v2: B=240 (recall 0.796, +3 pts over top-k) chosen over B=320 (tie with top-k); decided on tune split only
# 8192 shots: the budget-feasible optimum has median probability ~3e-4 under the trained p=1 circuit
# (results/angles.json), so 1024 shots would contain it only ~30% of the time; 8192 costs ~0.6 s on QR.
SHOTS = 2048                                 # v2: measured ~5.6 J per 1000 shots; 2048 matched 4096 on selection quality
BUDGETS = [160, 240, 320]


def pair_sum(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """sum_{i<j} M_ij x_i x_j for every row of X (M symmetric, zero diagonal)."""
    return 0.5 * np.einsum("mi,mi->m", X @ M.astype(np.float32), X)


def main() -> None:
    settings = config.load_settings()
    pools = [data.get_pool(it["_id"], settings) for it in data.load_split(TUNE_SPLIT)]
    n = pools[0].n
    X = qubo.all_bits(n).astype(np.float32)
    pre = []
    for p in pools:
        sim = np.array(p.meta["sim"])
        gold = np.array([s.is_gold for s in p.sentences], dtype=float)
        titles = np.array([s.title for s in p.sentences])
        gold_titles = p.meta["gold_titles"]
        cover = np.stack([X @ ((titles == gt) & (gold > 0)).astype(float) > 0 for gt in gold_titles], 1).all(1)
        pre.append({
            "Xr": {rp: X @ (p.relevance ** rp).astype(np.float32) for rp in R_POWERS}, "Xt": X @ p.tokens.astype(np.float32), "Br": pair_sum(X, p.br),
            "R": {th: pair_sum(X, data.red_from_sim(sim, th)) for th in THETAS},
            "rec": (X @ gold) / max(1, p.n_gold_total), "both": cover,
            "topk": {B: topk(p.relevance, p.tokens, B) for B in BUDGETS}, "gold": gold, "pool": p,
        })

    rows = []
    for B, th, lam, nu, rho, rp in itertools.product(BUDGETS, THETAS, LAMS, NUS, RHOS, R_POWERS):
        rec, both, ufeas, ntok = [], [], [], []
        for q in pre:
            c = -q["Xr"][rp] + rho * q["Xt"] + lam * q["R"][th] - nu * q["Br"]
            feas = q["Xt"] <= B
            k = int(np.argmin(np.where(feas, c, np.inf)))
            ku = int(np.argmin(c))
            rec.append(q["rec"][k]); both.append(q["both"][k]); ufeas.append(feas[ku]); ntok.append(q["Xt"][k])
        rows.append({"B": B, "theta": th, "lam": lam, "nu": nu, "rho": rho, "r_power": rp, "sf_recall": float(np.mean(rec)),
                     "both_gold": float(np.mean(both)), "uncon_feasible": float(np.mean(ufeas)),
                     "tokens": float(np.mean(ntok))})

    per_B = {}
    for B in BUDGETS:
        tk = [data.sf_recall(q["pool"], np.flatnonzero(q["topk"][B])) for q in pre]
        tkb = [data.both_gold(q["pool"], np.flatnonzero(q["topk"][B])) for q in pre]
        cand = [r for r in rows if r["B"] == B]
        top = max(r["sf_recall"] for r in cand)
        near = [r for r in cand if r["sf_recall"] >= top - TIE_TOL]
        best = max(near, key=lambda r: (r["uncon_feasible"], r["both_gold"], -r["lam"] - r["nu"] - r["rho"] - r["r_power"]))
        per_B[B] = {"topk_sf_recall": float(np.mean(tk)), "topk_both_gold": float(np.mean(tkb)), "best": best,
                    }
        print(f"B={B}: topk={np.mean(tk):.3f} (both {np.mean(tkb):.2f})  exact_best={best}")

    # Budget rule: among budgets whose exact recall clears a quality floor (answers must stay correct for the
    # A3-vs-A0 energy claim), take the largest margin of exact over top-k; ties -> smaller budget (fewer joules).
    ok = [B for B in BUDGETS if per_B[B]["best"]["sf_recall"] >= RECALL_FLOOR] or BUDGETS
    B_star = max(ok, key=lambda B: (round(per_B[B]["best"]["sf_recall"] - per_B[B]["topk_sf_recall"], 3), -B))
    best = per_B[B_star]["best"]
    eps_ref = config.eps_in_default()
    hp = {"lam": best["lam"], "nu": best["nu"], "red_threshold": best["theta"], "budget_tokens": B_star,
          "n_pool": settings.n_pool, "kappa": best["rho"] / eps_ref, "shots": SHOTS,
          "extra": {"rho": best["rho"], "r_power": best["r_power"], "lead_bonus": data.LEAD_BONUS, "eps_in_ref": eps_ref, "embed_backend": EMBED_BACKEND,
                    "kappa_note": "kappa = rho/eps_in; qwatt.qubo.effective_kappa re-normalizes when eps_in changes",
                    "tuned_on": TUNE_SPLIT, "budget_rule": f"max exact-topk margin s.t. exact recall >= {RECALL_FLOOR}", "n_tune": len(pools)}}
    config.HYPERPARAMS_JSON.write_text(json.dumps(hp, indent=1))
    g2 = best["sf_recall"] >= per_B[B_star]["topk_sf_recall"]
    (config.RESULTS_DIR / "tune_qubo.json").write_text(json.dumps(
        {"chosen": hp, "G2_pass_tune": bool(g2), "per_budget": {str(k): v for k, v in per_B.items()},
         "rows": rows, "grid": {"r_power": R_POWERS, "lam": LAMS, "nu": NUS, "rho": RHOS, "theta": THETAS, "B": BUDGETS}, "n_tune": len(pools), "split": TUNE_SPLIT},
        indent=1))
    print(f"chosen B={B_star}: exact={best['sf_recall']:.3f} vs topk={per_B[B_star]['topk_sf_recall']:.3f}  "
          f"G2(tune)={'PASS' if g2 else 'FAIL'}  -> {config.HYPERPARAMS_JSON}")


if __name__ == "__main__":
    main()

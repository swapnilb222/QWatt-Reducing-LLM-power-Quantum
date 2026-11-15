"""Tune the long-context QUBO on lc_dev: budget B and weights to keep the gold sentence with B as small as possible.

No brute force at n ~ 140, so the QUBO optimum is proxied by greedy + feasible 1-flip polish (fast), and the
chosen point is re-checked with the full classical reference (annealing + greedy, polished).

Rent: rho = max(rho_floor, budget-matched rho at `fill`) relevance per token (kappa = rho / eps_in). A strong rent
matters for the circuit: with a weak rent the irrelevant sentences have near-zero fields, p=1 QAOA leaves them
~50/50 and every shot is far over budget. `ground_tokens` = tokens of {i : h_i < 0}, a cheap view of where the
Hamiltonian's low-energy states sit relative to B.

Rule: smallest B whose best proxy gold recall >= RECALL_TARGET; at that B the weights with the best recall, ties
(within 0.034 = one question) broken by the smallest predicted QAOA shot size / B: expected tokens of a shot from the
exact p=1 marginals (qaoa_sim.p1_marginals, closed form at any n) -> shots closest to the budget, least repair.
Writes results/lc_hyperparams.json and results/lc_tune.json.

Run:  .venv/Scripts/python scripts/tune_lc.py
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from qwatt import config, longdoc, qaoa_sim, qubo, solvers_classical as sc  # noqa: E402
from qwatt.embed import EMBED_BACKEND  # noqa: E402
from qwatt.selectors import lc_anneal, load_lc_angles  # noqa: E402

TUNE_SPLIT = "lc_tune"        # lc_dev + 70 q from 7 more non-test docs
BUDGETS = [300, 400]            # 150/200 cost ~8-10 pts of recall on lc_tune (first lc_tune run)
R_POWERS = [2, 3, 4]         # high powers ~ ranking (top-k-like) while keeping the couplings
FILLS = [0.5, 1.0]
RHO_FLOORS = [0.001, 0.0025, 0.005, 0.01]
N_STAGE2 = 6                    # top proxy settings re-scored with the annealing reference (closer to the true optimum)
LAMS = [0.0, 1.0]
NUS = [0.0, 0.03, 0.1, 0.3]     # a strong coherence reward outweighs weak relevance at the true optimum
RECALL_TARGET = 0.95
# Couplings must be ON (lam > 0 and nu > 0): with J = 0 the problem is separable, dispatch solves it in closed form and
# no circuit runs. Gold-sentence recall cannot reward coherence (SQuAD answers live in one sentence); the coherence
# term exists to keep the antecedent of pronoun-initial sentences for the LLM. We accept it only within TIE of the
# best recall and report the uncoupled best alongside.
TIE = 0.034
ANG = load_lc_angles()          # p=1 angles used for the analytic shot-size prediction


def topk_recall(pool, B):
    x = sc.topk(pool.relevance, pool.tokens, B)
    return longdoc.gold_recall(pool, np.flatnonzero(x))


def main() -> None:
    base = longdoc.lc_settings()
    eps_in = config.eps_in_default()
    pools = [longdoc.get_lc_pool(q["id"], base) for q in longdoc.load_lc_split(TUNE_SPLIT)]
    t0 = time.perf_counter()
    rows = []
    for B, rp, fill, rho, lam, nu in itertools.product(BUDGETS, R_POWERS, FILLS, RHO_FLOORS, LAMS, NUS):
        s = copy.deepcopy(base)
        s.lam, s.nu, s.budget_tokens = lam, nu, B
        s.extra = {**s.extra, "r_power": rp, "fill": fill, "rho": rho, "rent_mode": "budget_matched"}
        rec, gt, ntok, rhos, shot = [], [], [], [], []
        for p in pools:
            h, J, info = qubo.build_qubo_lc(p, s, eps_in, B)
            x = qubo.local_polish(sc.greedy(h, J, p.tokens, B), h, J, p.tokens, B)
            rec.append(longdoc.gold_recall(p, np.flatnonzero(x)))
            gt.append(float(p.tokens[h < 0].sum()))
            ntok.append(float(x @ p.tokens))
            rhos.append(info["rho"])
            a, b, _, _ = qubo.normalized_ising(h, J)
            shot.append(float(qaoa_sim.p1_marginals(a, b, ANG["gamma"][0], ANG["beta"][0]) @ p.tokens))
        rows.append({"B": B, "r_power": rp, "fill": fill, "rho_floor": rho, "lam": lam, "nu": nu,
                     "gold_recall": float(np.mean(rec)), "ground_tokens_over_B": float(np.median(gt) / B),
                     "tokens": float(np.mean(ntok)), "rho_median": float(np.median(rhos)),
                     "qaoa_shot_tokens_over_B": float(np.median(shot) / B)})
    per_B = {}
    for B in BUDGETS:
        cand = [r for r in rows if r["B"] == B]
        top = max(r["gold_recall"] for r in cand)
        best = min([r for r in cand if r["gold_recall"] >= top - TIE and r["lam"] > 0 and r["nu"] > 0] or cand,
                   key=lambda r: (r["qaoa_shot_tokens_over_B"], -r["gold_recall"]))
        per_B[B] = {"topk_gold_recall": float(np.mean([topk_recall(p, B) for p in pools])), "best": best,
                    "best_recall": top}
        print(f"B={B}: topk={per_B[B]['topk_gold_recall']:.3f} best_recall={top:.3f} chosen={best}", flush=True)
    ok = [B for B in BUDGETS if per_B[B]["best"]["gold_recall"] >= RECALL_TARGET] or [max(BUDGETS)]
    B_star = min(ok)
    # Stage 2: greedy is only a proxy; the circuit + repair lands near the low-cost end, so re-score the best coupled
    # settings with the annealing reference and keep the one whose (near-)optimal selections keep the gold most often.
    cand = []
    for nu in [v for v in NUS if v > 0]:            # best proxy setting per coherence weight, plus runner-up r_power
        rs = sorted([r for r in rows if r["B"] == B_star and r["lam"] > 0 and r["nu"] == nu],
                    key=lambda r: (-r["gold_recall"], r["qaoa_shot_tokens_over_B"]))
        cand += rs[:2]
    for r in cand:
        s = copy.deepcopy(base)
        s.lam, s.nu, s.budget_tokens = r["lam"], r["nu"], B_star
        s.extra = {**s.extra, "r_power": r["r_power"], "fill": r["fill"], "rho": r["rho_floor"], "rent_mode": "budget_matched"}
        ref = []
        for p in pools:
            h, J, _ = qubo.build_qubo_lc(p, s, eps_in, B_star)
            ref.append(longdoc.gold_recall(p, np.flatnonzero(lc_anneal(h, J, p.tokens, B_star, seed=s.seed))))
        r["reference_gold_recall"] = float(np.mean(ref))
        print(f"stage2 {r}", flush=True)
    b = max(cand, key=lambda r: (r["reference_gold_recall"], -r["qaoa_shot_tokens_over_B"]))
    ref = [b["reference_gold_recall"]]
    hp = {"lam": b["lam"], "nu": b["nu"], "red_threshold": base.red_threshold, "budget_tokens": B_star,
          "shots": 256, "p": 1,
          "extra": {"rho": b["rho_floor"], "r_power": b["r_power"], "fill": b["fill"], "rent_mode": "budget_matched",
                    "W": base.extra.get("W", longdoc.W_DEFAULT), "embed_backend": EMBED_BACKEND, "eps_in_ref": eps_in,
                    "tuned_on": TUNE_SPLIT, "n_tune": len(pools), "proxy": "greedy + 1-flip polish",
                    "rule": f"smallest B with proxy gold recall >= {RECALL_TARGET}; weights = best annealing-reference "
                            f"recall among the top {N_STAGE2} coupled proxy settings",
                    "shots_rule": "lc_dev gold_kept equal at 256/512/1024 shots; 256 = lowest select energy (scaling.json)"}}
    longdoc.LC_HYPERPARAMS_JSON.write_text(json.dumps(hp, indent=1))
    out = {"chosen": hp, "reference_gold_recall_at_chosen": float(np.mean(ref)),
           "per_budget": {str(k): v for k, v in per_B.items()}, "rows": rows, "wall_s": time.perf_counter() - t0}
    (config.RESULTS_DIR / "lc_tune.json").write_text(json.dumps(out, indent=1))
    print(f"chosen B={B_star}: proxy recall={b['gold_recall']:.3f} reference recall={np.mean(ref):.3f} "
          f"topk={per_B[B_star]['topk_gold_recall']:.3f} -> {longdoc.LC_HYPERPARAMS_JSON}")


if __name__ == "__main__":
    main()

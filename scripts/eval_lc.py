"""Long-context evaluation (no LLM).

  results/lc_selection_eval.json : gold-sentence recall / tokens per arm on lc_test
      arms: qwatt_lc (QAOA on Quantum Rings, one qubit per sentence), anneal_lc (classical reference on the same
      QUBO), random_lc (K uniform bitstrings + same decode), topk_lc (relevance top-k to the budget), full.
  results/lc_solver_quality.json : per instance and summary: ratio vs the classical reference, QAOA-vs-random
      shot-cost separation (same bins), feasible-shot fraction, repair rate, wall time per stage.
  --shots-sweep : qwatt_lc gold_kept on lc_dev at several shot counts (energy per shot count is in scaling.json).

Run:  .venv/Scripts/python scripts/eval_lc.py [n_questions]
      .venv/Scripts/python scripts/eval_lc.py --shots-sweep
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from qwatt import config, longdoc, solvers_classical as sc  # noqa: E402
from qwatt.selectors import get_selector, lc_reference  # noqa: E402


def boot_ci(v, n_boot=5000, seed=0):
    v = np.asarray(v, float)
    m = v[np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def overlap(h1, h2) -> float:
    """Overlap of two normalized histograms on the same bins (1 = identical, 0 = disjoint)."""
    return float(np.minimum(np.array(h1), np.array(h2)).sum())


def main(n_q: int | None = None) -> None:
    s = longdoc.lc_settings()
    B = s.budget_tokens
    qs = longdoc.load_lc_split("lc_test")[:n_q]
    arms = ["qwatt_lc", "anneal_lc", "random_lc", "topk_lc", "full"]
    rec = {a: [] for a in arms}
    tok = {a: [] for a in arms}
    per, sq = [], []
    t_all = time.perf_counter()
    for k, q in enumerate(qs):
        pool = longdoc.get_lc_pool(q["id"], s)
        row = {"qid": q["id"], "doc_id": pool.meta["doc_id"], "n": pool.n, "full_tokens": pool.full_tokens}
        for a in arms:
            if a == "full":
                idx, d = list(range(pool.n)), {"tokens": pool.full_tokens, "solver_wall_s": 0.0}
            elif a == "topk_lc":
                t0 = time.perf_counter()
                idx = [int(i) for i in np.flatnonzero(sc.topk(pool.relevance, pool.tokens, B))]
                d = {"tokens": int(pool.tokens[idx].sum()), "solver_wall_s": time.perf_counter() - t0}
            else:
                idx, d = get_selector(a)(pool, B, s)
            r = longdoc.gold_recall(pool, idx)
            rec[a].append(r)
            tok[a].append(d["tokens"])
            row[a] = {"gold_recall": r, "tokens": d["tokens"], "n_sel": len(idx), "solver_wall_s": d["solver_wall_s"]}
            if a == "qwatt_lc":
                ref = lc_reference(pool, d, s)
                d.update(ref)
                qd = {k2: d[k2] for k2 in ("n_qubits", "n_edges", "depth", "gate_count", "shots", "circuit_wall_s",
                                           "qubo_build_s", "decode_s", "feasible_shot_frac", "mean_shot_tokens",
                                           "feasible_fallback", "ratio_vs_anneal", "mean_shot_cost_qaoa",
                                           "mean_shot_cost_random", "best_cost", "polished_cost", "polished_gold_kept",
                                           "reference_wall_s", "gold_kept", "from_circuit_shot", "rent_calibrated",
                                           "mu_circuit", "gamma_effective", "predicted_feasible_frac",
                                           "predicted_shot_tokens", "tokens")}
                qd["random_feasible_shot_frac"] = d["reference"]["random_feasible_shot_frac"]
                qd["random_ratio_vs_anneal"] = d["reference"]["random_ratio_vs_anneal"]
                qd["anneal_cost"] = d["reference"]["anneal_cost"]
                qd["random_best_cost"] = d["reference"]["random_best_cost"]
                qd["anneal_gold_kept"] = d["reference"]["anneal_gold_kept"]
                qd["random_gold_kept"] = d["reference"]["random_gold_kept"]
                qd["hist_overlap_qaoa_vs_random"] = overlap(d["cost_hist_qaoa"]["counts"], d["cost_hist_random"]["counts"])
                qd["solver_wall_s"] = d["solver_wall_s"]
                sq.append({"qid": q["id"], **qd})
        per.append(row)
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(qs)} " + " ".join(f"{a}={np.mean(rec[a]):.3f}" for a in arms), flush=True)

    sel = {"split": "lc_test", "n": len(qs), "budget_tokens": B, "settings": {"lam": s.lam, "nu": s.nu,
           "red_threshold": s.red_threshold, "shots": s.shots, "extra": s.extra},
           "arms": {a: {"gold_recall_mean": float(np.mean(rec[a])), "gold_recall_ci": boot_ci(rec[a]),
                        "tokens_mean": float(np.mean(tok[a])), "n": len(rec[a])} for a in arms},
           "per_question": per, "wall_s": time.perf_counter() - t_all}
    (config.RESULTS_DIR / "lc_selection_eval.json").write_text(json.dumps(sel, indent=1))
    keys = ["ratio_vs_anneal", "feasible_shot_frac", "predicted_shot_tokens", "mu_circuit", "gamma_effective", "mean_shot_tokens", "hist_overlap_qaoa_vs_random",
            "circuit_wall_s", "solver_wall_s", "qubo_build_s", "decode_s", "n_qubits", "n_edges", "depth", "gate_count",
            "mean_shot_cost_qaoa", "mean_shot_cost_random"]
    summ = {k2: {"mean": float(np.mean([x[k2] for x in sq if x[k2] is not None])),
                 "median": float(np.median([x[k2] for x in sq if x[k2] is not None]))} for k2 in keys}
    summ["repair_rate"] = float(np.mean([x["feasible_fallback"] for x in sq]))
    summ["from_circuit_shot_rate"] = float(np.mean([x["from_circuit_shot"] for x in sq]))
    summ["feasible_shot_frac_qaoa_vs_random"] = {"qaoa": float(np.mean([x["feasible_shot_frac"] for x in sq])),
                                                 "random": float(np.mean([x["random_feasible_shot_frac"] for x in sq])),
                                                 "qaoa_predicted": float(np.mean([x["predicted_feasible_frac"] or 0 for x in sq]))}
    summ["ratio_vs_anneal_qaoa_vs_random"] = {
        "qaoa_mean": float(np.mean([x["ratio_vs_anneal"] for x in sq if x["ratio_vs_anneal"] is not None])),
        "random_repair_mean": float(np.mean([x["random_ratio_vs_anneal"] for x in sq if x["random_ratio_vs_anneal"] is not None]))}
    summ["tokens_selected_mean"] = float(np.mean([x["tokens"] for x in sq]))
    summ["gold_kept_rate"] = {"qwatt_lc": float(np.mean([x["gold_kept"] for x in sq])),
                              "qwatt_lc_polished": float(np.mean([x["polished_gold_kept"] for x in sq])),
                              "anneal_lc": float(np.mean([x["anneal_gold_kept"] for x in sq])),
                              "random_lc": float(np.mean([x["random_gold_kept"] for x in sq]))}
    summ["qaoa_beats_random_best_rate"] = float(np.mean([x["best_cost"] < x["random_best_cost"] - 1e-12 for x in sq]))
    summ["qaoa_within_5pct_of_anneal_rate"] = float(np.mean([(x["ratio_vs_anneal"] or 0) >= 0.95 for x in sq]))
    qual = {"split": "lc_test", "n": len(sq), "summary": summ, "per_instance": sq,
            "note": "ratio_vs_anneal = qaoa_cost / reference_cost (costs < 0; 1 = equal to the classical reference "
                    "= best of SA and greedy, both 1-flip polished). Selection uses raw shots + repair, never the polish."}
    qp = config.RESULTS_DIR / "lc_solver_quality.json"
    if qp.exists():                                   # keep the lc_dev shots sweep if it was run before
        prev = json.loads(qp.read_text())
        if "shots_sweep_lc_dev" in prev:
            qual["shots_sweep_lc_dev"] = prev["shots_sweep_lc_dev"]
    qp.write_text(json.dumps(qual, indent=1))
    print(json.dumps({"arms": sel["arms"], "solver": summ}, indent=1))


def shots_sweep(shots_list=(256, 512, 1024)) -> None:
    base = longdoc.lc_settings()
    qs = longdoc.load_lc_split("lc_dev")
    out = {}
    for sh in shots_list:
        s = copy.deepcopy(base)
        s.shots = sh
        kept, wall, ratio = [], [], []
        for q in qs:
            pool = longdoc.get_lc_pool(q["id"], s)
            idx, d = get_selector("qwatt_lc")(pool, s.budget_tokens, s)
            kept.append(d["gold_kept"])
            wall.append(d["solver_wall_s"])
        out[sh] = {"gold_kept_rate": float(np.mean(kept)), "solver_wall_s_mean": float(np.mean(wall))}
        print(sh, out[sh], flush=True)
    p = config.RESULTS_DIR / "lc_solver_quality.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur["shots_sweep_lc_dev"] = {str(k): v for k, v in out.items()}
    p.write_text(json.dumps(cur, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--shots-sweep"]:
        shots_sweep()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

"""Selection quality on the TEST split (no LLM): supporting-fact recall at the frozen budget, per arm.

Arms: full (reference), topk (A1), greedy (A2), exact (A4), anneal (A5), random_k (A6), qwatt (A3, live on
Quantum Rings). Reports mean + bootstrap 95% CI, both-gold rate, tokens, and paired differences vs top-k.
Writes results/selection_eval.json.

Run:  .venv/Scripts/python scripts/eval_selection.py [split] [arms comma-separated]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from qwatt import config, data  # noqa: E402
from qwatt.selectors import get_selector  # noqa: E402

ARMS = ["topk", "greedy", "exact", "anneal", "random_k", "qwatt", "full"]


def boot_ci(v, n_boot: int = 5000, seed: int = 0) -> list[float]:
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main(split: str = "test", arms: list[str] | None = None) -> None:
    arms = arms or ARMS
    settings = config.load_settings()
    B = settings.budget_tokens
    items = data.load_split(split)
    per_q, rec = [], {a: [] for a in arms}
    both = {a: [] for a in arms}
    tok = {a: [] for a in arms}
    wall = {a: [] for a in arms}
    pool_recall = []
    t0 = time.perf_counter()
    for k, it in enumerate(items):
        pool = data.get_pool(it["_id"], settings)
        pool_recall.append(pool.pool_recall)
        row = {"qid": pool.qid, "qtype": pool.qtype, "pool_recall": pool.pool_recall}
        for a in arms:
            idx, d = get_selector(a)(pool, B, settings)
            r = data.sf_recall(pool, idx)
            rec[a].append(r); both[a].append(data.both_gold(pool, idx)); tok[a].append(d["tokens"])
            wall[a].append(d["solver_wall_s"])
            row[a] = {"indices": idx, "sf_recall": r, "tokens": d["tokens"], "cost": d["cost"]}
            if a == "qwatt":
                row[a].update({k2: d[k2] for k2 in ("dispatch", "approx_ratio", "opt_in_shots", "feasible_fallback")})
        per_q.append(row)
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(items)}  " + "  ".join(f"{a}={np.mean(rec[a]):.3f}" for a in arms), flush=True)

    out = {"split": split, "n": len(items), "budget_tokens": B, "pool_recall_mean": float(np.mean(pool_recall)),
           "pool_recall_ci": boot_ci(pool_recall), "settings_hash": settings.hash(),
           "hyperparams": json.loads(config.HYPERPARAMS_JSON.read_text()) if config.HYPERPARAMS_JSON.exists() else None,
           "arms": {}, "paired_vs_topk": {}, "wall_s": time.perf_counter() - t0}
    for a in arms:
        out["arms"][a] = {"sf_recall_mean": float(np.mean(rec[a])), "sf_recall_ci": boot_ci(rec[a]),
                          "both_gold_rate": float(np.mean(both[a])), "both_gold_ci": boot_ci(both[a]),
                          "tokens_mean": float(np.mean(tok[a])), "solver_wall_s_mean": float(np.mean(wall[a])),
                          "n": len(rec[a])}
        if a != "topk" and "topk" in arms:
            diff = np.array(rec[a]) - np.array(rec["topk"])
            out["paired_vs_topk"][a] = {"mean_diff": float(diff.mean()), "ci": boot_ci(diff),
                                        "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                                        "losses": int((diff < 0).sum())}
    if "qwatt" in arms:
        q = [r["qwatt"] for r in per_q]
        out["qwatt_summary"] = {"dispatch_counts": {d: sum(x["dispatch"] == d for x in q) for d in {x["dispatch"] for x in q}},
                                "approx_ratio_mean": float(np.mean([x["approx_ratio"] for x in q])),
                                "opt_hit_rate": float(np.mean([x["opt_in_shots"] for x in q])),
                                "fallback_rate": float(np.mean([x["feasible_fallback"] for x in q]))}
    out["per_question"] = per_q
    config.SELECTION_EVAL_JSON.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_question", "hyperparams")}, indent=1))


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    arms = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    main(split, arms)

"""SYNTHETIC long-context (LC) preview data for the dashboard. NOT MEASUREMENTS.

Called by app/dev_fixture.build(); writes LC rows (bench_id 'lc-fixture-1', arms full_lc / qwatt_safe /
qwatt_lc) into results/dev_fixture.sqlite, LC pools into fixture_pools, and fixture_json entries
'scaling', 'calibration_lc', 'lc_selection', 'lc_demo_questions'. Diag keys follow CONTRACTS.md (PIVOT 04:35).
Also provides simulate_lc_lane(), a fake pipeline for Live mode in fixture mode (llm_backend='fake').
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid

import numpy as np

from app import dev_fixture as FX

EPS_IN_LC = 0.059
BUDGET_LC = 640
W = 2

TOPICS = [
    ("The Normans", ["were a people who in the 10th and 11th centuries gave their name to Normandy",
                     "descended from Norse raiders and pirates from Denmark, Iceland and Norway",
                     "swore fealty to King Charles III of West Francia under their leader Rollo",
                     "adopted the Gallo-Romance language of the land they settled",
                     "became known for their Catholic piety and their military skill",
                     "conquered England in 1066 under William the Conqueror",
                     "established a kingdom in Sicily and southern Italy",
                     "built castles and cathedrals in the Romanesque style"]),
    ("Normandy", ["is a region of northern France facing the English Channel",
                  "takes its name from the Norse settlers who arrived in the 10th century",
                  "was granted to Rollo by the Treaty of Saint-Clair-sur-Epte in 911",
                  "remained a duchy until it was absorbed into the French crown in 1204",
                  "is known for its dairy farms, apples and cider",
                  "has a coastline that saw the Allied landings of June 1944"]),
    ("Norman architecture", ["is characterised by rounded arches and massive walls",
                             "spread across England after the conquest of 1066",
                             "includes Durham Cathedral, begun in 1093",
                             "favoured square towers and thick cylindrical piers",
                             "influenced church building in Sicily and the Levant"]),
    ("The Norman conquest", ["began with the landing at Pevensey in September 1066",
                             "culminated in the Battle of Hastings on 14 October 1066",
                             "replaced the English aristocracy with a Norman elite",
                             "introduced Norman French into English law and court life",
                             "led to the Domesday Book survey completed in 1086"]),
]


def make_lc_pool(qid: str, rng: np.random.Generator, n_target: int) -> dict:
    sents, para_of, titles = [], [], []
    p = 0
    while len(sents) < n_target:
        title, facts = TOPICS[p % len(TOPICS)]
        k = int(rng.integers(14, 26))
        for j in range(k):
            fact = facts[(j + p) % len(facts)]
            lead = ["", "Historians note that ", "According to chroniclers, ", "It is often said that ",
                    "Later accounts record that ", "Some sources claim that "][int(rng.integers(0, 6))]
            subj = title if not lead else title.lower() if title.startswith("The ") else title
            text = f"{lead}{subj} {fact}."
            text = text[0].upper() + text[1:]
            sents.append(text)
            para_of.append(p)
            titles.append(title)
            if len(sents) >= n_target:
                break
        p += 1
    n = len(sents)
    gold = int(rng.integers(5, n - 5))
    rel = np.clip(rng.beta(1.3, 4.0, size=n), 0, 1)
    # relevance clusters around the gold sentence's paragraph
    rel[[i for i in range(n) if para_of[i] == para_of[gold]]] += 0.25
    rel[gold] = 1.0
    rel = np.clip(rel, 0, 1)
    rel = (rel - rel.min()) / (rel.max() - rel.min())
    out = []
    for i, t in enumerate(sents):
        out.append({"idx": i, "text": t, "title": titles[i], "tokens": FX._tok(t) + int(rng.integers(4, 12)),
                    "relevance": float(rel[i]), "is_gold": i == gold})
    full_tokens = sum(s["tokens"] for s in out) + 40 * (p + 1)
    return {"qid": qid, "question": "When did the Norman conquest of England culminate in the Battle of Hastings?",
            "answer": "14 October 1066", "qtype": "lc", "sentences": out, "full_tokens": int(full_tokens),
            "pool_recall": 1.0, "paragraph_of": para_of, "doc_id": f"doc-{qid[-2:]}", "kind": "lc"}


def lc_qubo(pool: dict, rng: np.random.Generator):
    n = len(pool["sentences"])
    r = np.array([s["relevance"] for s in pool["sentences"]])
    t = np.array([s["tokens"] for s in pool["sentences"]], dtype=float)
    h = -r + 0.2 * EPS_IN_LC * t
    J = {}
    reds, bridges = [], []
    for i in range(n):
        for d in range(1, W + 1):
            j = i + d
            if j >= n:
                continue
            if pool["paragraph_of"][i] != pool["paragraph_of"][j]:
                continue
            u = rng.random()
            if u < 0.18:
                w = float(rng.uniform(0.6, 0.9))
                J[(i, j)] = 0.8 * w
                reds.append([i, j, round(w, 3)])
            elif u < 0.36 and d == 1:
                J[(i, j)] = -0.25
                bridges.append([i, j])
    return h, J, t, reds, bridges


def _cost(X, h, J):
    c = X @ h
    for (i, j), w in J.items():
        c = c + w * X[..., i] * X[..., j]
    return c


def lc_solve(pool: dict, rng: np.random.Generator, shots: int = 1024) -> dict:
    h, J, t, reds, bridges = lc_qubo(pool, rng)
    n = len(h)
    # a good reference solution: greedy by marginal value within budget, then 1-flip polish
    order = np.argsort(h / np.maximum(t, 1))
    x = np.zeros(n)
    for i in order:
        if h[i] < 0 and (x @ t + t[i]) <= BUDGET_LC:
            x[i] = 1
    best = x.copy()
    anneal_cost = float(_cost(best, h, J))
    # QAOA-like shots: perturbations of the good solution (concentrated), plus diffuse noise
    flips = rng.poisson(2.2, size=shots)
    X = np.repeat(best[None, :], shots, axis=0)
    for k in range(shots):
        idx = rng.choice(n, size=flips[k], replace=False)
        X[k, idx] = 1 - X[k, idx]
    noisy = rng.random(shots) < 0.15
    X[noisy] = (rng.random((int(noisy.sum()), n)) < 0.5).astype(float)
    costs = _cost(X, h, J)
    toks = X @ t
    R = (rng.random((shots, n)) < 0.5).astype(float)
    rcost = _cost(R, h, J)
    edges = np.linspace(min(costs.min(), rcost.min()), max(costs.max(), rcost.max()), 31)
    hq, _ = np.histogram(costs, bins=edges)
    hr, _ = np.histogram(rcost, bins=edges)
    uniq = {}
    for k in range(shots):
        b = "".join("1" if v else "0" for v in X[k].astype(int))
        if b in uniq:
            uniq[b]["count"] += 1
        else:
            uniq[b] = {"bits": b, "count": 1, "cost": round(float(costs[k]), 4), "tokens": int(toks[k]),
                       "feasible": bool(toks[k] <= BUDGET_LC)}
    samples = sorted(uniq.values(), key=lambda s: s["cost"])[:300]
    feas = [s for s in samples if s["feasible"]]
    pick = min(feas, key=lambda s: s["cost"]) if feas else samples[0]
    gold = next(i for i, s in enumerate(pool["sentences"]) if s["is_gold"])
    anneal_cost = float(pick["cost"]) * float(rng.uniform(1.0, 1.05))  # reference slightly better (more negative)
    n_edges = len(J)
    return {
        "arm": "qwatt_lc", "dispatch": "circuit", "backend": "scarlet_quantum_rings (synthetic)",
        "n_qubits": n, "n_edges": n_edges, "W": W, "depth": 2 + 2 * W + 4, "gate_count": 3 * n + 3 * n_edges,
        "shots": shots, "gamma": [0.41], "beta": [0.29], "circuit_wall_s": round(1.0 + 0.026 * n, 2),
        "samples": samples, "n_distinct": len(uniq), "best_bits": pick["bits"], "best_cost": pick["cost"],
        "kept_bits": pick["bits"], "exact_bits": None, "exact_cost": None, "approx_ratio": None,
        "opt_in_shots": None, "amplification": None, "feasible_fallback": False,
        "reference": {"anneal_cost": round(anneal_cost, 4), "random_best_cost": round(float(rcost.min()), 4)},
        "ratio_vs_anneal": round(float(pick["cost"] / anneal_cost), 4) if anneal_cost else None,
        "gold_kept": pick["bits"][gold] == "1",
        "cost_hist_qaoa": {"edges": [round(float(v), 4) for v in edges], "counts": [int(v) for v in hq]},
        "cost_hist_random": {"edges": [round(float(v), 4) for v in edges], "counts": [int(v) for v in hr]},
        "h": [round(float(v), 4) for v in h], "t": [int(v) for v in t],
        "relevance": [s["relevance"] for s in pool["sentences"]],
        "J_edges": [[i, j, round(w, 4)] for (i, j), w in J.items()],
        "red_edges": reds, "bridge_edges": bridges, "paragraph_of": pool["paragraph_of"],
        "cost": pick["cost"], "tokens": pick["tokens"], "solver_wall_s": round(1.3 + 0.03 * n, 2),
    }


PH_W = {"select": 15.5, "prefill": 9.6, "generate": 8.8, "escalate_prefill": 9.7, "escalate_generate": 8.8}


def build_lc(con: sqlite3.Connection, t_start: float, cum, seed: int = 23) -> None:
    rng = np.random.default_rng(seed)
    cols = [r[1] for r in con.execute("PRAGMA table_info(runs)").fetchall()]
    if "escalated" not in cols:
        con.execute("ALTER TABLE runs ADD COLUMN escalated INTEGER")
    pools = [make_lc_pool(f"lc_{k + 1:02d}", rng, int(rng.integers(110, 200))) for k in range(20)]
    for p in pools:
        con.execute("INSERT OR REPLACE INTO fixture_pools VALUES(?,?)", (p["qid"], json.dumps(p)))
    t = t_start
    samples = []
    bench_id = "lc-fixture-1"
    for qi, pool in enumerate(pools):
        diag = lc_solve(pool, rng)
        kept = [i for i, c in enumerate(diag["kept_bits"]) if c == "1"]
        sel_tok = sum(pool["sentences"][i]["tokens"] for i in kept)
        abstain = (not diag["gold_kept"]) or rng.random() < 0.08
        for arm in ("qwatt_safe", "full_lc", "qwatt_lc"):
            gap_end = t + rng.uniform(6, 9)
            while t < gap_end:
                samples.append(FX._sample_row(t, FX.watts_for("idle", rng), cum, rng))
                t += 0.25
            run_id = f"{bench_id}:{pool['qid']}:{arm}"
            plan = []
            if arm != "full_lc":
                plan.append(("select", diag["solver_wall_s"]))
                plan.append(("prefill", (sel_tok + 80) * 0.0105 * rng.uniform(.95, 1.05) + 1.2))
                plan.append(("generate", rng.uniform(0.5, 0.8)))
                if arm == "qwatt_safe" and abstain:
                    plan.append(("escalate_prefill", (pool["full_tokens"] + 80) * 0.0105 * rng.uniform(.95, 1.05)))
                    plan.append(("escalate_generate", rng.uniform(0.4, 0.8)))
            else:
                plan.append(("prefill", (pool["full_tokens"] + 80) * 0.0105 * rng.uniform(.95, 1.05)))
                plan.append(("generate", rng.uniform(0.4, 0.8)))
            t0 = t
            marks = {}
            for ph, dur in plan:
                marks[ph] = (t, t + dur)
                end = t + dur
                while t < end:
                    w = max(1.0, PH_W[ph] + rng.normal(0, 0.9))
                    samples.append(FX._sample_row(t, w, cum, rng))
                    t += 0.25
            t1 = t
            win = samples[-1200:]
            jp = {ph: FX._integrate(win, a, b, FX.IDLE_W) for ph, (a, b) in marks.items()}
            jn = FX._integrate(win, t0, t1, FX.IDLE_W)
            jg = FX._integrate(win, t0, t1)
            esc = int(arm == "qwatt_safe" and abstain)
            if arm == "full_lc":
                good = rng.random() < 0.8
            elif arm == "qwatt_lc":
                good = (not abstain) and rng.random() < 0.85
            else:
                good = (rng.random() < 0.85) if not abstain else (rng.random() < 0.8)
            ans = "14 October 1066" if good else ("NOT FOUND" if arm == "qwatt_lc" and abstain else "1066")
            tin = (pool["full_tokens"] + 80) if arm == "full_lc" else sel_tok + 80
            row = dict(run_id=run_id, qid=pool["qid"], question=pool["question"], gold=pool["answer"], arm=arm,
                       mode="bench", started_at=t0, ended_at=t1, config_hash="fxlc00000000", llm_backend="lmstudio",
                       model_id="gemma-4-e4b (synthetic)", tokens_in=tin + (pool["full_tokens"] + 80 if esc else 0),
                       tokens_out=int(rng.integers(3, 9)), context_tokens=pool["full_tokens"] if arm == "full_lc" else sel_tok,
                       n_selected=len(pool["sentences"]) if arm == "full_lc" else len(kept),
                       selected_json=json.dumps(list(range(len(pool["sentences"]))) if arm == "full_lc" else kept),
                       ttft_s=marks["prefill"][1] - t0, select_s=marks.get("select", (0, 0))[1] - marks.get("select", (0, 0))[0],
                       prefill_s=marks["prefill"][1] - marks["prefill"][0],
                       generate_s=marks["generate"][1] - marks["generate"][0], total_s=t1 - t0, idle_watts=FX.IDLE_W,
                       joules_gross=jg, joules_net=jn, joules_select=jp.get("select", 0.0), joules_prefill=jp.get("prefill"),
                       joules_generate=jp.get("generate"), energy_source="rapl_pdh", answer=ans,
                       em=1.0 if good else 0.0, f1=1.0 if good else 0.3, sf_recall=1.0 if diag["gold_kept"] or arm == "full_lc" else 0.0,
                       referee_correct=int(good), referee_score=0.9 if good else 0.1,
                       referee_reason="Synthetic.", referee_model="gemini (synthetic)", bench_id=bench_id, escalated=esc)
            con.execute(f"INSERT INTO runs({','.join(row)}) VALUES({','.join('?' * len(row))})", list(row.values()))
            for ph, (a, b) in marks.items():
                con.execute("INSERT INTO phase_markers VALUES(?,?,?,?)", (run_id, ph, a, b))
            if arm != "full_lc":
                FX._insert_qaoa(con, run_id, pool["qid"], {**diag, "arm": arm}, jp.get("select", 0.0))
    con.executemany("INSERT OR REPLACE INTO power_samples VALUES(?,?,?,?,?,?,?)", samples)

    pts = []
    for n, wall, jq in ((24, 1.0, 31.0), (50, 1.1, 35.0), (100, 2.2, 67.0), (150, 3.9, 58.0), (200, 5.8, 42.0)):
        pts.append({"n_qubits": n, "n_edges": int(n * 1.6), "wall_s": wall * rng.uniform(.95, 1.05),
                    "joules": jq * rng.uniform(.9, 1.1), "shots": 1024})
    ns = [len(p["sentences"]) for p in pools]
    fx = {
        "scaling": {"eps_in": EPS_IN_LC, "tokens_per_sentence": 27.5, "keep_frac": 0.15, "W": W,
                    "backend": "scarlet_quantum_rings (synthetic)", "points": pts,
                    "docs": {"n_min": int(min(ns)), "n_max": int(max(ns)), "n_median": float(np.median(ns))}},
        "calibration_lc": {"e0": 6.0, "eps_in": EPS_IN_LC, "eps_out": 1.1, "idle_watts": FX.IDLE_W, "r2": 0.991,
                           "points": [], "energy_source": "rapl_pdh", "llm_backend": "lmstudio"},
        "lc_demo_questions": [p["qid"] for p in pools[:4]],
        "lc_selection": {"qwatt_lc": {"gold_kept_rate": 0.86, "ratio_vs_anneal_mean": 0.97, "tokens_mean": 610, "n": 100},
                         "anneal_lc": {"gold_kept_rate": 0.88, "tokens_mean": 615, "n": 100},
                         "random_lc": {"gold_kept_rate": 0.31, "tokens_mean": 600, "n": 100}},
    }
    for k, v in fx.items():
        con.execute("INSERT OR REPLACE INTO fixture_json VALUES(?,?)", (k, json.dumps(v)))


_SIM_COUNT = [0]


def simulate_lc_lane(pool: dict, arm: str, on_event, path=FX.FIXTURE_DB, speed: float = 2.5) -> dict:
    """Fake pipeline for LC arms in fixture mode (sped up). Same event protocol, incl. 'escalate'."""
    FX.ensure_fixture_sampler(path)
    rng = np.random.default_rng()
    run_id = f"live:{uuid.uuid4().hex[:8]}:{arm}"
    marks = {}
    t_start = time.time()

    def phase(ph, dur, load):
        FX._load["phase"] = load
        a = time.time()
        on_event({"type": "phase", "phase": ph, "t": a})
        time.sleep(dur / speed)
        marks[ph] = (a, time.time())

    if arm != "full_lc":
        FX._load["phase"] = "select"
        a = time.time()
        on_event({"type": "phase", "phase": "select", "t": a})
        diag = lc_solve(pool, rng)
        time.sleep(max(0.0, 2.5 / speed - (time.time() - a)))
        marks["select"] = (a, time.time())
        kept = [i for i, c in enumerate(diag["kept_bits"]) if c == "1"]
        on_event({"type": "selection", "indices": kept, "diag": diag})
        sel_tok = sum(pool["sentences"][i]["tokens"] for i in kept)
        phase("prefill", (sel_tok + 80) * 0.0105 + 1.0, "prefill")
        a = time.time()
        FX._load["phase"] = "generate"
        on_event({"type": "phase", "phase": "generate", "t": a})
        _SIM_COUNT[0] += 1
        abstain = not diag["gold_kept"] or _SIM_COUNT[0] % 2 == 1  # alternate so previews show both outcomes
        on_event({"type": "token", "text": "NOT FOUND" if abstain else "14 October 1066"})
        time.sleep(0.6 / speed)
        marks["generate"] = (a, time.time())
        esc = arm == "qwatt_safe" and abstain
        if esc:
            on_event({"type": "escalate", "t": time.time()})
            phase("escalate_prefill", (pool["full_tokens"] + 80) * 0.0105, "prefill")
            a = time.time()
            FX._load["phase"] = "generate"
            on_event({"type": "phase", "phase": "escalate_generate", "t": a})
            on_event({"type": "token", "text": "14 October 1066"})
            time.sleep(0.6 / speed)
            marks["escalate_generate"] = (a, time.time())
        tin = sel_tok + 80 + ((pool["full_tokens"] + 80) if esc else 0)
        ctx = sel_tok
        answer = "14 October 1066" if (esc or not abstain) else "NOT FOUND"
    else:
        diag, kept, esc = {"arm": "full_lc"}, list(range(len(pool["sentences"]))), False
        on_event({"type": "selection", "indices": kept, "diag": diag})
        phase("prefill", (pool["full_tokens"] + 80) * 0.0105, "prefill")
        a = time.time()
        FX._load["phase"] = "generate"
        on_event({"type": "phase", "phase": "generate", "t": a})
        on_event({"type": "token", "text": "14 October 1066"})
        time.sleep(0.6 / speed)
        marks["generate"] = (a, time.time())
        tin, ctx, answer = pool["full_tokens"] + 80, pool["full_tokens"], "14 October 1066"
    FX._load["phase"] = "idle"
    t_end = time.time()
    on_event({"type": "phase", "phase": "done", "t": t_end})
    time.sleep(0.3)
    con = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    rows = [(r[0], r[1]) for r in con.execute("SELECT ts, pkg_watts FROM power_samples WHERE ts BETWEEN ? AND ? ORDER BY ts",
                                              (t_start - 0.3, t_end + 0.3)).fetchall()]
    jp = {ph: FX._integrate(rows, a, b, FX.IDLE_W) for ph, (a, b) in marks.items()}
    row = dict(run_id=run_id, qid=pool["qid"], question=pool["question"], gold=pool["answer"], arm=arm, mode="live",
               started_at=t_start, ended_at=t_end, config_hash="fxlc00000000", llm_backend="fake", model_id="fake-llm",
               tokens_in=tin, tokens_out=4, context_tokens=ctx, n_selected=len(kept), selected_json=json.dumps(kept),
               ttft_s=marks["prefill"][1] - t_start, select_s=(marks.get("select", (0, 0))[1] - marks.get("select", (0, 0))[0]),
               prefill_s=marks["prefill"][1] - marks["prefill"][0], generate_s=marks["generate"][1] - marks["generate"][0],
               total_s=t_end - t_start, idle_watts=FX.IDLE_W, joules_gross=FX._integrate(rows, t_start, t_end),
               joules_net=FX._integrate(rows, t_start, t_end, FX.IDLE_W), joules_select=jp.get("select", 0.0),
               joules_prefill=jp.get("prefill"), joules_generate=jp.get("generate"), energy_source="rapl_pdh",
               answer=answer, em=1.0, f1=1.0, sf_recall=None, referee_correct=1, referee_score=0.9,
               referee_reason="Synthetic.", referee_model="synthetic", bench_id=None, escalated=int(esc))
    cols = [r[1] for r in con.execute("PRAGMA table_info(runs)").fetchall()]
    row_db = {k: v for k, v in row.items() if k in cols}
    con.execute(f"INSERT OR REPLACE INTO runs({','.join(row_db)}) VALUES({','.join('?' * len(row_db))})", list(row_db.values()))
    for ph, (a, b) in marks.items():
        con.execute("INSERT OR REPLACE INTO phase_markers VALUES(?,?,?,?)", (run_id, ph, a, b))
    if arm != "full_lc":
        FX._insert_qaoa(con, run_id, pool["qid"], diag, jp.get("select", 0.0))
    con.close()
    on_event({"type": "result", "row": row})
    return row

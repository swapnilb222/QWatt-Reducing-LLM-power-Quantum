"""SYNTHETIC preview data for dashboard development. NOT MEASUREMENTS.

    .venv/Scripts/python -m app.dev_fixture          # writes results/dev_fixture.sqlite

The dashboard uses this DB only when QWATT_DASH_FIXTURE=1, and then shows a
"SYNTHETIC PREVIEW DATA" banner on every view. It also provides `simulate_lane`, a fake
pipeline used by Live mode in fixture mode (rows tagged llm_backend='fake').
"""
from __future__ import annotations

import itertools
import json
import math
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DB = ROOT / "results" / "dev_fixture.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS power_samples(ts REAL PRIMARY KEY, pkg_watts REAL, pp0_watts REAL, pp1_watts REAL,
    pkg_joules_cum REAL, sys_watts REAL, source TEXT);
CREATE TABLE IF NOT EXISTS sampler_heartbeat(id INTEGER PRIMARY KEY CHECK(id=1), ts REAL, pid INTEGER, source TEXT);
CREATE TABLE IF NOT EXISTS phase_markers(run_id TEXT, phase TEXT, t0 REAL, t1 REAL, PRIMARY KEY(run_id, phase));
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, qid TEXT, question TEXT, gold TEXT, arm TEXT, mode TEXT,
    started_at REAL, ended_at REAL, config_hash TEXT, llm_backend TEXT, model_id TEXT,
    tokens_in INTEGER, tokens_out INTEGER, context_tokens INTEGER, n_selected INTEGER, selected_json TEXT,
    ttft_s REAL, select_s REAL, prefill_s REAL, generate_s REAL, total_s REAL,
    idle_watts REAL, joules_gross REAL, joules_net REAL,
    joules_select REAL, joules_prefill REAL, joules_generate REAL, energy_source TEXT,
    answer TEXT, em REAL, f1 REAL, sf_recall REAL,
    referee_correct INTEGER, referee_score REAL, referee_reason TEXT, referee_model TEXT, bench_id TEXT);
CREATE TABLE IF NOT EXISTS qaoa_runs(run_id TEXT PRIMARY KEY, qid TEXT, backend TEXT, n_qubits INTEGER,
    n_edges INTEGER, depth INTEGER, gate_count INTEGER, shots INTEGER, gamma_json TEXT, beta_json TEXT,
    wall_s REAL, joules REAL, best_cost REAL, exact_cost REAL, approx_ratio REAL, amplification REAL,
    opt_in_shots INTEGER, dispatch TEXT, feasible_fallback INTEGER, diag_json TEXT);
CREATE TABLE IF NOT EXISTS fixture_json(name TEXT PRIMARY KEY, json TEXT);
CREATE TABLE IF NOT EXISTS fixture_pools(qid TEXT PRIMARY KEY, json TEXT);
"""

IDLE_W = 3.3
EPS_IN, EPS_OUT, E0 = 0.0285, 0.21, 4.1
BUDGET = 120

# (title, text, relevance, is_gold)
TEMPLATES = [
    {
        "question": "In which city was the director of the film Inception born?",
        "answer": "London", "qtype": "bridge",
        "sents": [
            ("Inception", "Inception is a 2010 science fiction action film written and directed by Christopher Nolan.", 0.97, True),
            ("Christopher Nolan", "Nolan was born on 30 July 1970 in Westminster, London, England.", 0.86, True),
            ("Inception", "The film stars Leonardo DiCaprio as a thief who steals information by entering dreams.", 0.55, False),
            ("Christopher Nolan", "Christopher Nolan is the director of Inception and of The Dark Knight trilogy.", 0.90, False),
            ("Leonardo DiCaprio", "DiCaprio was born in Los Angeles, California, in 1974.", 0.62, False),
            ("Hans Zimmer", "Hans Zimmer, who composed the score for Inception, was born in Frankfurt.", 0.58, False),
            ("Inception", "Inception premiered in London on 8 July 2010 and grossed over $836 million.", 0.66, False),
            ("Christopher Nolan", "He studied English literature at University College London.", 0.41, False),
            ("Interstellar", "Interstellar is a 2014 epic science fiction film directed by Christopher Nolan.", 0.47, False),
            ("Memento", "Memento is a 2000 neo-noir thriller written and directed by Christopher Nolan.", 0.44, False),
            ("Westminster", "Westminster is an area of central London on the north bank of the Thames.", 0.38, False),
            ("Inception", "Inception won four Academy Awards, including Best Cinematography.", 0.29, False),
            ("Elliot Page", "Elliot Page played the architect Ariadne in the film.", 0.22, False),
            ("The Dark Knight", "The Dark Knight was released in 2008 to wide critical acclaim.", 0.18, False),
            ("Warner Bros.", "Warner Bros. distributed the film worldwide.", 0.12, False),
            ("Leonardo DiCaprio", "DiCaprio won the Academy Award for Best Actor for The Revenant.", 0.05, False),
        ],
        "bridges": [(0, 1), (3, 1), (1, 10)], "reds": [(0, 3, 0.82), (0, 8, 0.64), (8, 9, 0.7), (6, 11, 0.61)],
        "full_answer": "Christopher Nolan was born in London.", "q_answer": "London (Westminster).",
        "topk_answer": "Christopher Nolan directed Inception.",
    },
    {
        "question": "Were the Eiffel Tower and the internal frame of the Statue of Liberty engineered by the same person?",
        "answer": "yes", "qtype": "comparison",
        "sents": [
            ("Eiffel Tower", "The Eiffel Tower was engineered by Gustave Eiffel's company and completed in 1889.", 0.95, True),
            ("Statue of Liberty", "Gustave Eiffel designed the statue's internal iron framework.", 0.92, True),
            ("Statue of Liberty", "The statue was designed by French sculptor Frédéric Auguste Bartholdi.", 0.71, False),
            ("Eiffel Tower", "The tower is named after the engineer Gustave Eiffel, whose company built it.", 0.88, False),
            ("Gustave Eiffel", "Gustave Eiffel was a French civil engineer born in Dijon in 1832.", 0.63, False),
            ("Statue of Liberty", "The Statue of Liberty was dedicated on October 28, 1886.", 0.46, False),
            ("Eiffel Tower", "It was the tallest man-made structure in the world until 1930.", 0.33, False),
            ("Maurice Koechlin", "Maurice Koechlin drew the first sketch of the tower in 1884.", 0.57, False),
            ("Statue of Liberty", "The copper statue was a gift from the people of France.", 0.39, False),
            ("Liberty Island", "Liberty Island lies in New York Harbor.", 0.21, False),
            ("Gustave Eiffel", "Eiffel also built the Garabit viaduct in France.", 0.35, False),
            ("Eiffel Tower", "About seven million people visit the tower every year.", 0.15, False),
            ("Bartholdi", "Bartholdi was born in Colmar, Alsace, in 1834.", 0.27, False),
            ("Eiffel Tower", "The tower is 330 metres tall including antennas.", 0.18, False),
            ("Statue of Liberty", "The torch was replaced in 1986.", 0.09, False),
            ("Paris", "Paris is the capital of France.", 0.04, False),
        ],
        "bridges": [(0, 1), (3, 1), (1, 4)], "reds": [(0, 3, 0.86), (2, 12, 0.6), (0, 6, 0.62)],
        "full_answer": "Yes, Gustave Eiffel engineered both.", "q_answer": "Yes, Gustave Eiffel.",
        "topk_answer": "Yes.",
    },
    {
        "question": "What is the capital of the country where the Rhine Falls are located?",
        "answer": "Bern", "qtype": "bridge",
        "sents": [
            ("Rhine Falls", "The Rhine Falls is the largest plain waterfall in Europe, located in Switzerland.", 0.96, True),
            ("Switzerland", "Bern is the federal city and de facto capital of Switzerland.", 0.84, True),
            ("Switzerland", "Zurich is the largest city in Switzerland.", 0.66, False),
            ("Rhine Falls", "The falls lie on the High Rhine near the town of Schaffhausen.", 0.74, False),
            ("Rhine", "The Rhine flows through Switzerland, Germany and the Netherlands.", 0.52, False),
            ("Rhine Falls", "The Rhine Falls is a waterfall in northern Switzerland near Schaffhausen.", 0.91, False),
            ("Germany", "Berlin is the capital of Germany.", 0.48, False),
            ("Schaffhausen", "Schaffhausen is a town in northern Switzerland.", 0.43, False),
            ("Bern", "The old town of Bern is a UNESCO World Heritage Site.", 0.37, False),
            ("Rhine Falls", "The falls are 23 metres high and 150 metres wide.", 0.31, False),
            ("Lake Constance", "Lake Constance borders Germany, Austria and Switzerland.", 0.25, False),
            ("Switzerland", "Switzerland has four national languages.", 0.2, False),
            ("Rhine", "The Rhine ends in the North Sea.", 0.16, False),
            ("Geneva", "Geneva hosts the European headquarters of the United Nations.", 0.12, False),
            ("Rhine Falls", "Boats take visitors to a rock in the middle of the falls.", 0.08, False),
            ("Alps", "The Alps cover about 60 percent of Switzerland.", 0.03, False),
        ],
        "bridges": [(0, 1), (5, 1), (1, 8)], "reds": [(0, 5, 0.88), (3, 5, 0.66), (3, 7, 0.6)],
        "full_answer": "Bern, Switzerland.", "q_answer": "Bern.", "topk_answer": "Switzerland.",
    },
]


def _tok(text: str) -> int:
    return max(4, int(round(len(text) / 4.2)))


def make_pool(qid: str, tmpl: dict) -> dict:
    sents = [{"idx": i, "text": t, "title": ti, "tokens": _tok(t), "relevance": r, "is_gold": g}
             for i, (ti, t, r, g) in enumerate(tmpl["sents"])]
    return {"qid": qid, "question": tmpl["question"], "answer": tmpl["answer"], "qtype": tmpl["qtype"],
            "sentences": sents, "full_tokens": 1380 + 37 * (hash(qid) % 7), "pool_recall": 1.0,
            "bridges": tmpl["bridges"], "reds": tmpl["reds"]}


def qubo_terms(pool: dict, lam=1.0, nu=0.5, kappa=1.0):
    n = len(pool["sentences"])
    r = np.array([s["relevance"] for s in pool["sentences"]])
    t = np.array([s["tokens"] for s in pool["sentences"]], dtype=float)
    h = -r + kappa * EPS_IN * t * 0.9
    J = {}
    for i, j, w in pool["reds"]:
        J[(i, j)] = J.get((i, j), 0) + lam * w
    for i, j in pool["bridges"]:
        J[(min(i, j), max(i, j))] = J.get((min(i, j), max(i, j)), 0) - nu * 0.8
    return h, J, t, n


def solve(pool: dict, budget: int, shots: int, rng: np.random.Generator) -> dict:
    h, J, t, n = qubo_terms(pool)
    N = 1 << n
    X = ((np.arange(N)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)  # col i == qubit i
    cost = X @ h
    for (i, j), w in J.items():
        cost += w * X[:, i] * X[:, j]
    tok = X @ t
    feas = tok <= budget
    c_opt_idx = int(np.argmin(np.where(feas, cost, np.inf)))
    c_opt = cost[c_opt_idx]
    c_worst = float(np.max(np.where(feas, cost, -np.inf)))
    # QAOA-like distribution: Boltzmann over cost, penalised when over budget, plus uniform noise
    eff = cost + np.where(feas, 0.0, 0.6 + 0.004 * (tok - budget))
    p = np.exp(-(eff - eff.min()) / 0.11)
    p = 0.85 * p / p.sum() + 0.15 / N
    draws = rng.choice(N, size=shots, p=p)
    uniq, counts = np.unique(draws, return_counts=True)
    bits_of = lambda k: "".join(str(int(v)) for v in X[k].astype(int))  # noqa: E731
    samples = [{"bits": bits_of(k), "count": int(c), "cost": round(float(cost[k]), 4), "tokens": int(tok[k]),
                "feasible": bool(feas[k])} for k, c in zip(uniq, counts)]
    samples.sort(key=lambda s: -s["count"])
    fs = [s for s in samples if s["feasible"]]
    best = min(fs, key=lambda s: s["cost"]) if fs else min(samples, key=lambda s: s["cost"])
    top = np.argsort(cost)[: max(1, N // 100)]
    p_top = np.isin(draws, top).mean()
    amp = float(p_top / (len(top) / N))
    ar = (c_worst - best["cost"]) / (c_worst - c_opt) if c_worst != c_opt else 1.0
    edges = [[i, j, round(w, 4)] for (i, j), w in J.items()]
    return {
        "arm": "qwatt", "dispatch": "circuit", "backend": "QrBackendV2 · Quantum Rings (synthetic)",
        "n_qubits": n, "n_edges": len(edges), "depth": 2 * 1 + len(edges) // 3 + 3, "gate_count": n * 3 + len(edges) * 3,
        "shots": shots, "gamma": [0.412], "beta": [0.297], "circuit_wall_s": round(float(rng.uniform(0.55, 0.9)), 3),
        "samples": samples, "best_bits": best["bits"], "best_cost": best["cost"],
        "exact_bits": bits_of(c_opt_idx), "exact_cost": round(float(c_opt), 4), "approx_ratio": round(float(ar), 4),
        "opt_in_shots": any(s["bits"] == bits_of(c_opt_idx) for s in samples), "amplification": round(amp, 2),
        "feasible_fallback": False, "h": [round(float(v), 4) for v in h], "J_edges": edges,
        "t": [int(v) for v in t], "relevance": [float(s["relevance"]) for s in pool["sentences"]],
        "red_edges": [[i, j, w] for i, j, w in pool["reds"]], "bridge_edges": [[i, j] for i, j in pool["bridges"]],
        "cost": best["cost"], "tokens": best["tokens"], "solver_wall_s": 0.0,
    }


# ------------------------------------------------------------------ power traces
def watts_for(phase: str, rng) -> float:
    base = {"idle": IDLE_W, "select": 8.6, "prefill": 15.2, "generate": 11.8}[phase]
    return max(1.0, base + rng.normal(0, 0.35 if phase == "idle" else 0.8))


class _Cum:
    def __init__(self):
        self.j = 1000.0
        self.last = None


def _sample_row(ts, w, cum: _Cum, rng):
    if cum.last is not None:
        cum.j += w * (ts - cum.last)
    cum.last = ts
    return (ts, round(w, 3), round(w * 0.72, 3), round(w * 0.08, 3), round(cum.j, 3), round(w + 4.2 + rng.normal(0, .3), 3), "rapl_pdh")


def _integrate(rows, t0, t1, idle=None):
    pts = [(r[0], r[1]) for r in rows if t0 <= r[0] <= t1]
    if len(pts) < 2:
        return 0.0
    ts, ws = zip(*pts)
    ws = np.array(ws) - (idle or 0)
    return float(np.trapezoid(np.clip(ws, 0, None), ts))


def _lane_plan(arm: str, pool: dict, sel_tokens: int, rng):
    if arm == "full":
        tin = pool["full_tokens"] + 62
        ctx = pool["full_tokens"]
        sel_s = 0.0
    else:
        tin = sel_tokens + 62
        ctx = sel_tokens
        sel_s = {"qwatt": rng.uniform(0.6, 0.95), "topk": 0.01, "greedy": 0.04}[arm]
    tout = int(rng.integers(7, 14))
    return {"tin": tin, "ctx": ctx, "tout": tout, "select_s": sel_s,
            "prefill_s": tin * 0.0105 * rng.uniform(0.95, 1.08), "generate_s": tout * 0.085 * rng.uniform(.9, 1.1)}


def selection_for(arm: str, pool: dict, rng) -> tuple[list[int], dict]:
    n = len(pool["sentences"])
    toks = [s["tokens"] for s in pool["sentences"]]
    if arm == "full":
        return list(range(n)), {"arm": "full"}
    if arm == "qwatt":
        d = solve(pool, BUDGET, 1024, rng)
        idx = [i for i, c in enumerate(d["best_bits"]) if c == "1"]
        return idx, d
    order = sorted(range(n), key=lambda i: -pool["sentences"][i]["relevance"])
    if arm == "greedy":
        order = order[:2] + sorted(order[2:], key=lambda i: -(pool["sentences"][i]["relevance"] - 0.02 * toks[i]))
    out, tot = [], 0
    for i in order:
        if tot + toks[i] <= BUDGET:
            out.append(i)
            tot += toks[i]
    return sorted(out), {"arm": arm, "tokens": tot}


# ------------------------------------------------------------------ build
def build(path: Path = FIXTURE_DB, n_questions: int = 20, seed: int = 11) -> Path:
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(str(path), timeout=10)
    for (tbl,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        con.execute(f"DROP TABLE IF EXISTS {tbl}")  # drop instead of unlink: a running dashboard may hold it open
    con.commit()
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    pools = []
    for k in range(n_questions):
        qid = f"fx_{k + 1:02d}"
        pools.append(make_pool(qid, TEMPLATES[k % len(TEMPLATES)]))
    for p in pools:
        con.execute("INSERT INTO fixture_pools VALUES(?,?)", (p["qid"], json.dumps(p)))

    t = time.time() - 5 * 3600
    cum = _Cum()
    samples = []
    bench_id = "fixture-bench-1"
    # correctness plan
    ok_rate = {"full": 0.85, "topk": 0.6, "greedy": 0.7, "qwatt": 0.8}
    for qi, pool in enumerate(pools):
        tmpl = TEMPLATES[qi % len(TEMPLATES)]
        arms = ["full", "topk", "greedy", "qwatt"]
        rng.shuffle(arms)
        for arm in arms:
            # idle gap
            gap_end = t + rng.uniform(5, 9)
            while t < gap_end:
                samples.append(_sample_row(t, watts_for("idle", rng), cum, rng))
                t += 0.25
            idx, diag = selection_for(arm, pool, rng)
            sel_tokens = sum(pool["sentences"][i]["tokens"] for i in idx)
            plan = _lane_plan(arm, pool, sel_tokens, rng)
            run_id = f"{bench_id}:{pool['qid']}:{arm}"
            t_start = t
            marks = {}
            for ph in ("select", "prefill", "generate"):
                dur = plan[f"{ph}_s"]
                if dur <= 0:
                    continue
                marks[ph] = (t, t + dur)
                end = t + dur
                while t < end:
                    samples.append(_sample_row(t, watts_for(ph, rng), cum, rng))
                    t += 0.25
            t_end = t
            jp = {ph: _integrate(samples[-400:], a, b, IDLE_W) for ph, (a, b) in marks.items()}
            jg = _integrate(samples[-400:], t_start, t_end)
            jn = _integrate(samples[-400:], t_start, t_end, IDLE_W)
            good = rng.random() < ok_rate[arm]
            ans = {"full": tmpl["full_answer"], "qwatt": tmpl["q_answer"]}.get(arm, tmpl["topk_answer"])
            if not good:
                ans = "I cannot tell from the context." if arm != "full" else tmpl["topk_answer"]
            gold_idx = [s["idx"] for s in pool["sentences"] if s["is_gold"]]
            sf = len(set(idx) & set(gold_idx)) / len(gold_idx)
            row = dict(run_id=run_id, qid=pool["qid"], question=pool["question"], gold=pool["answer"], arm=arm,
                       mode="bench", started_at=t_start, ended_at=t_end, config_hash="fx0000000000",
                       llm_backend="lmstudio", model_id="gemma-4-e4b (synthetic)", tokens_in=plan["tin"],
                       tokens_out=plan["tout"], context_tokens=plan["ctx"], n_selected=len(idx),
                       selected_json=json.dumps(idx), ttft_s=plan["select_s"] + plan["prefill_s"] + 0.08,
                       select_s=plan["select_s"], prefill_s=plan["prefill_s"], generate_s=plan["generate_s"],
                       total_s=t_end - t_start, idle_watts=IDLE_W, joules_gross=jg, joules_net=jn,
                       joules_select=jp.get("select", 0.0), joules_prefill=jp.get("prefill"),
                       joules_generate=jp.get("generate"), energy_source="rapl_pdh", answer=ans,
                       em=1.0 if good and arm == "qwatt" else 0.0, f1=0.8 if good else 0.15, sf_recall=sf,
                       referee_correct=int(good), referee_score=0.9 if good else 0.1,
                       referee_reason="Matches the gold answer." if good else "Does not state the gold answer.",
                       referee_model="gemini-flash (synthetic)", bench_id=bench_id)
            con.execute(f"INSERT INTO runs({','.join(row)}) VALUES({','.join('?' * len(row))})", list(row.values()))
            for ph, (a, b) in marks.items():
                con.execute("INSERT INTO phase_markers VALUES(?,?,?,?)", (run_id, ph, a, b))
            if arm == "qwatt":
                _insert_qaoa(con, run_id, pool["qid"], diag, jp.get("select", 0.0))
    con.executemany("INSERT INTO power_samples VALUES(?,?,?,?,?,?,?)", samples)
    from app import lc_fixture as LCF  # long-context (pivot) synthetic data
    LCF.build_lc(con, t + 10.0, cum)
    con.execute("INSERT INTO sampler_heartbeat VALUES(1,?,?,?)", (time.time(), 0, "rapl_pdh"))

    # results JSON (synthetic)
    pts = []
    for k in range(24):
        tin = int(rng.integers(80, 1800))
        tout = int(rng.integers(6, 48))
        jn = E0 + EPS_IN * tin + EPS_OUT * tout + rng.normal(0, 1.2)
        pts.append({"tokens_in": tin, "tokens_out": tout, "joules_net": jn, "joules_gross": jn + 3.3 * tin * 0.0105,
                    "seconds": tin * 0.0105 + tout * 0.085})
    fx = {
        "calibration": {"e0": E0, "eps_in": EPS_IN, "eps_out": EPS_OUT, "idle_watts": IDLE_W, "r2": 0.986,
                        "chars_per_token": 4.2, "points": pts, "model_id": "gemma-4-e4b (synthetic)",
                        "llm_backend": "lmstudio", "energy_source": "rapl_pdh", "measured_at": time.time() - 5 * 3600},
        "angles": {"p": 1, "gamma": [0.412], "beta": [0.297], "n_train": 30, "per_instance": [], "normalization": "rms"},
        "hyperparams": {"lam": 1.0, "nu": 0.5, "kappa": 1.0, "red_threshold": 0.6, "budget_tokens": BUDGET, "n_pool": 16},
        "demo_questions": [p["qid"] for p in pools[:3]],
        "selection_eval": {
            "full": {"sf_recall_mean": 1.0, "sf_recall_ci": [1.0, 1.0], "tokens_mean": 1420, "both_gold_rate": 1.0, "n": 100},
            "topk": {"sf_recall_mean": 0.71, "sf_recall_ci": [0.65, 0.77], "tokens_mean": 114, "both_gold_rate": 0.48, "n": 100},
            "greedy": {"sf_recall_mean": 0.76, "sf_recall_ci": [0.70, 0.82], "tokens_mean": 113, "both_gold_rate": 0.56, "n": 100},
            "exact": {"sf_recall_mean": 0.83, "sf_recall_ci": [0.78, 0.88], "tokens_mean": 112, "both_gold_rate": 0.69, "n": 100},
            "anneal": {"sf_recall_mean": 0.82, "sf_recall_ci": [0.77, 0.87], "tokens_mean": 112, "both_gold_rate": 0.67, "n": 100},
            "random_k": {"sf_recall_mean": 0.64, "sf_recall_ci": [0.57, 0.70], "tokens_mean": 109, "both_gold_rate": 0.40, "n": 100},
            "qwatt": {"sf_recall_mean": 0.81, "sf_recall_ci": [0.76, 0.86], "tokens_mean": 111, "both_gold_rate": 0.66, "n": 100},
            "pool_recall_mean": 0.93,
        },
    }
    per = []
    for k in range(40):
        per.append({"qid": f"fx_t{k}", "qaoa": {"approx_ratio": float(min(1, rng.beta(14, 1.2))), "opt_hit": bool(rng.random() < .8),
                                                "amplification": float(rng.uniform(6, 22))},
                    "exact": {"approx_ratio": 1.0, "opt_hit": True},
                    "anneal": {"approx_ratio": float(min(1, rng.beta(25, 1))), "opt_hit": bool(rng.random() < .9)},
                    "greedy": {"approx_ratio": float(rng.beta(8, 1.6)), "opt_hit": bool(rng.random() < .45)},
                    "random_k": {"approx_ratio": float(rng.beta(5, 2.2)), "opt_hit": bool(rng.random() < .15),
                                 "amplification": 1.0}})
    fx["solver_quality"] = {
        "summary": {
            "qaoa": {"approx_ratio_mean": 0.958, "opt_hit_rate": 0.8, "amplification_mean": 13.4, "wall_s_mean": 0.71},
            "exact": {"approx_ratio_mean": 1.0, "opt_hit_rate": 1.0, "wall_s_mean": 0.21},
            "anneal": {"approx_ratio_mean": 0.975, "opt_hit_rate": 0.9, "wall_s_mean": 0.09},
            "greedy": {"approx_ratio_mean": 0.83, "opt_hit_rate": 0.45, "wall_s_mean": 0.002},
            "random_k": {"approx_ratio_mean": 0.69, "opt_hit_rate": 0.15, "amplification_mean": 1.0, "wall_s_mean": 0.001},
        },
        "per_instance": per, "qr_vs_numpy_tvd": 0.031,
    }
    for k, v in fx.items():
        con.execute("INSERT INTO fixture_json VALUES(?,?)", (k, json.dumps(v)))
    con.commit()
    con.close()
    return path


def _insert_qaoa(con, run_id, qid, d, joules):
    con.execute("INSERT OR REPLACE INTO qaoa_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        run_id, qid, d["backend"], d["n_qubits"], d["n_edges"], d["depth"], d["gate_count"], d["shots"],
        json.dumps(d["gamma"]), json.dumps(d["beta"]), d["circuit_wall_s"], joules, d["best_cost"], d["exact_cost"],
        d["approx_ratio"], d["amplification"], (None if d["opt_in_shots"] is None else int(d["opt_in_shots"])), d["dispatch"], int(d["feasible_fallback"]),
        json.dumps(d)))


# ------------------------------------------------------------------ live simulation (fixture mode)
_load = {"phase": "idle"}
_sampler_lock = threading.Lock()
_sampler_started = False


def ensure_fixture_sampler(path: Path = FIXTURE_DB):
    """A fake 4 Hz sampler thread writing into the fixture DB (stands in for scripts/run_sampler.py)."""
    global _sampler_started
    with _sampler_lock:
        if _sampler_started:
            return
        _sampler_started = True

    def loop():
        rng = np.random.default_rng()
        con = sqlite3.connect(str(path), timeout=10, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL")
        cum = _Cum()
        r = con.execute("SELECT pkg_joules_cum FROM power_samples ORDER BY ts DESC LIMIT 1").fetchone()
        cum.j = r[0] if r else 1000.0
        while True:
            ts = time.time()
            row = _sample_row(ts, watts_for(_load["phase"], rng), cum, rng)
            try:
                con.execute("INSERT OR REPLACE INTO power_samples VALUES(?,?,?,?,?,?,?)", row)
                con.execute("INSERT OR REPLACE INTO sampler_heartbeat VALUES(1,?,?,?)", (ts, 0, "rapl_pdh"))
            except sqlite3.Error:
                pass
            time.sleep(0.25)

    threading.Thread(target=loop, daemon=True, name="fixture-sampler").start()


def simulate_lane(pool: dict, arm: str, on_event, path: Path = FIXTURE_DB, speed: float = 1.0) -> dict:
    """Fake pipeline.run_question for fixture mode: same event protocol as CONTRACTS.md."""
    ensure_fixture_sampler(path)
    rng = np.random.default_rng()
    tmpl = next((t for t in TEMPLATES if t["question"] == pool["question"]), TEMPLATES[0])
    run_id = f"live:{uuid.uuid4().hex[:8]}:{arm}"
    marks = {}
    t_start = time.time()
    if arm != "full":
        _load["phase"] = "select"
        t0 = time.time()
        on_event({"type": "phase", "phase": "select", "t": t0})
        idx, diag = selection_for(arm, pool, rng)
        time.sleep(max(0.0, 0.8 / speed - (time.time() - t0)))
        marks["select"] = (t0, time.time())
        on_event({"type": "selection", "indices": idx, "diag": diag})
    else:
        idx, diag = selection_for(arm, pool, rng)
        on_event({"type": "selection", "indices": idx, "diag": diag})
    sel_tokens = sum(pool["sentences"][i]["tokens"] for i in idx)
    plan = _lane_plan(arm, pool, sel_tokens, rng)
    _load["phase"] = "prefill"
    t0 = time.time()
    on_event({"type": "phase", "phase": "prefill", "t": t0})
    time.sleep(plan["prefill_s"] / speed)
    marks["prefill"] = (t0, time.time())
    _load["phase"] = "generate"
    t0 = time.time()
    on_event({"type": "phase", "phase": "generate", "t": t0})
    ans = {"full": tmpl["full_answer"], "qwatt": tmpl["q_answer"]}.get(arm, tmpl["topk_answer"])
    words = ans.split(" ")
    for k, w in enumerate(words):
        time.sleep(plan["generate_s"] / max(1, len(words)) / speed)
        on_event({"type": "token", "text": (" " if k else "") + w})
    marks["generate"] = (t0, time.time())
    _load["phase"] = "idle"
    t_end = time.time()
    on_event({"type": "phase", "phase": "done", "t": t_end})
    time.sleep(0.3)
    con = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    rows = con.execute("SELECT ts, pkg_watts FROM power_samples WHERE ts BETWEEN ? AND ? ORDER BY ts",
                       (t_start - 0.3, t_end + 0.3)).fetchall()
    rows = [(r[0], r[1]) for r in rows]
    jp = {ph: _integrate(rows, a, b, IDLE_W) for ph, (a, b) in marks.items()}
    row = dict(run_id=run_id, qid=pool["qid"], question=pool["question"], gold=pool["answer"], arm=arm, mode="live",
               started_at=t_start, ended_at=t_end, config_hash="fx0000000000", llm_backend="fake",
               model_id="fake-llm", tokens_in=plan["tin"], tokens_out=len(words), context_tokens=plan["ctx"],
               n_selected=len(idx), selected_json=json.dumps(idx),
               ttft_s=(marks["generate"][0] - t_start), select_s=marks.get("select", (0, 0))[1] - marks.get("select", (0, 0))[0],
               prefill_s=marks["prefill"][1] - marks["prefill"][0], generate_s=marks["generate"][1] - marks["generate"][0],
               total_s=t_end - t_start, idle_watts=IDLE_W, joules_gross=_integrate(rows, t_start, t_end),
               joules_net=_integrate(rows, t_start, t_end, IDLE_W), joules_select=jp.get("select", 0.0),
               joules_prefill=jp.get("prefill"), joules_generate=jp.get("generate"), energy_source="rapl_pdh",
               answer=ans, em=1.0, f1=1.0, sf_recall=None, referee_correct=1, referee_score=0.9,
               referee_reason="Synthetic.", referee_model="synthetic", bench_id=None)
    con.execute(f"INSERT OR REPLACE INTO runs({','.join(row)}) VALUES({','.join('?' * len(row))})", list(row.values()))
    for ph, (a, b) in marks.items():
        con.execute("INSERT OR REPLACE INTO phase_markers VALUES(?,?,?,?)", (run_id, ph, a, b))
    if arm == "qwatt":
        _insert_qaoa(con, run_id, pool["qid"], diag, jp.get("select", 0.0))
    con.close()
    on_event({"type": "result", "row": row})
    return row


if __name__ == "__main__":
    t0 = time.time()
    p = build()
    print(f"SYNTHETIC fixture written to {p} in {time.time() - t0:.1f}s")

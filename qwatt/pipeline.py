"""One question through one arm: select -> prefill -> generate, with phase markers and energy.

The same code path for every arm; only the selector differs. Energy is computed AFTER the run by
joining phase markers to the sampler's samples (qwatt.store.phase_energy). The selection phase's
joules are charged to the arm (so QAOA's own cost is part of A3's total).

Row conventions (see CONTRACTS.md change log):
  joules_select / joules_prefill / joules_generate are NET of idle (idle_watts * dt removed);
  joules_gross / joules_net are the sums over the three phases.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Callable

from qwatt import config, llm, power, prompt, store
from qwatt.types import Pool

ARMS = ("full", "topk", "greedy", "qwatt")


# --------------------------------------------------------------------------- scoring
def _em_f1(pred: str, gold: str) -> tuple[float, float]:
    try:
        from qwatt import referee
        return referee.exact_match(pred, gold), referee.f1(pred, gold)
    except Exception:  # noqa: BLE001  - fallback: HotpotQA-standard normalisation, local copy
        import re
        import string
        from collections import Counter

        def norm(s):
            s = (s or "").lower()
            s = "".join(c for c in s if c not in set(string.punctuation))
            s = re.sub(r"\b(a|an|the)\b", " ", s)
            return " ".join(s.split())
        p, g = norm(pred), norm(gold)
        em = float(p == g)
        if (p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}) and p != g:
            return em, 0.0
        common = Counter(p.split()) & Counter(g.split())
        n = sum(common.values())
        if n == 0:
            return em, 0.0
        pr, rc = n / len(p.split()), n / len(g.split())
        return em, 2 * pr * rc / (pr + rc)


def sf_recall(pool: Pool, indices, arm: str) -> float:
    """Fraction of ALL gold supporting sentences that reach the prompt. Arm 'full' sends every
    paragraph, so every gold sentence is present."""
    if arm in ("full", "full_lc"):
        return 1.0
    tot = pool.n_gold_total or sum(1 for s in pool.sentences if s.is_gold)
    if tot <= 0:
        return 0.0
    got = sum(1 for i in indices if pool.sentences[i].is_gold)
    return got / tot


# --------------------------------------------------------------------------- idle estimate
def idle_estimate(con, window_s: float = 3.0, lookback_s: float = 20.0) -> float | None:
    """Lowest mean package power over any `window_s` window in the last `lookback_s` seconds
    (an estimate of idle that is robust to being called right after a run)."""
    now = time.time()
    s = store.read_samples(con, now - lookback_s, now)
    if len(s) < 4:
        return None
    best = None
    i = 0
    for j in range(len(s)):
        while s[j]["ts"] - s[i]["ts"] > window_s:
            i += 1
        if s[j]["ts"] - s[i]["ts"] >= window_s * 0.8:
            w = store.mean_watts(s[i:j + 1])
            if w is not None and (best is None or w < best):
                best = w
    return best


# --------------------------------------------------------------------------- main
FULL_ARMS = ("full", "full_lc")
SAFE_ARMS = ("qwatt_safe",)
SELECTOR_FOR_ARM = {"full_lc": "full", "qwatt_lc": "qwatt_lc", "qwatt_safe": "qwatt_lc"}
QAOA_ARMS = ("qwatt", "qwatt_lc", "qwatt_safe")
LC_ARMS = ("full_lc", "qwatt_lc", "qwatt_safe")


def _select(pool: Pool, arm: str, settings):
    if arm in FULL_ARMS:   # trivial arm: the whole document, no selector call needed
        return list(range(pool.n)), {"arm": arm, "solver_wall_s": 0.0, "cost": 0.0,
                                     "tokens": int(pool.full_tokens), "pool_tokens": int(pool.tokens.sum())}
    from qwatt.selectors import get_selector
    sel = get_selector(SELECTOR_FOR_ARM.get(arm, arm))
    budget, price_diag = settings.budget_tokens, {}
    if arm in SAFE_ARMS + ("qwatt_lc",):
        budget, price_diag = energy_budget_tokens(settings)
    indices, diag = sel(pool, budget, settings)
    if price_diag and isinstance(diag, dict):
        diag.update(price_diag)
    return indices, diag


def energy_budget_tokens(settings) -> tuple[int, dict]:
    """Tiger-in-the-loop budget. OFF by default (benchmarks stay reproducible). When
    settings.extra['energy_budget_j'] or env QWATT_ENERGY_BUDGET_J is set: budget_tokens =
    max(120, (E_budget - e0) / eps_in) at the LIVE price from Tiger (fallback: results/calibration.json)."""
    import os
    e_budget = (getattr(settings, "extra", None) or {}).get("energy_budget_j") or os.getenv("QWATT_ENERGY_BUDGET_J")
    if not e_budget:
        return settings.budget_tokens, {}
    e_budget = float(e_budget)
    price = None
    try:
        from qwatt import tiger
        price = tiger.live_price()
    except Exception:  # noqa: BLE001
        price = None
    if price:
        eps, e0, src = price["eps_in"], price["e0"], "tiger"
    else:
        cal = config.load_calibration() or {}
        eps, e0, src = float(cal.get("eps_in") or config.eps_in_default()), float(cal.get("e0") or 0.0), "calibration"
    tokens = max(120, int((e_budget - e0) / eps)) if eps > 0 else settings.budget_tokens
    return tokens, {"price_source": src, "eps_in_live": eps, "e0_live": e0,
                    "energy_budget_j": e_budget, "budget_tokens_derived": tokens}


def run_question(pool: Pool, arm: str, settings, mode: str = "bench",
                 on_event: Callable[[dict], None] | None = None, *,
                 bench_id: str | None = None, idle_watts: float | None = None,
                 nonce: str | None = None, db_path=None, flush_wait_s: float = 0.5) -> dict:
    """Run one question through one arm; persist and return the `runs` row.

    Arm `qwatt_safe` = QAOA selection + SAFETY NET: if the pruned-context answer abstains
    (NOT FOUND / empty / "does not mention" ...), the question is re-asked with the full document under a
    NEW nonce (phases escalate_prefill / escalate_generate). All energy of both attempts is charged to the arm.
    """
    def emit(ev: dict):
        if on_event:
            try:
                on_event(ev)
            except Exception:  # noqa: BLE001 - UI callbacks must never break a run
                pass

    con = store.connect(db_path)
    power.ensure_sampler()
    run_id = f"{arm}-{pool.qid}-{uuid.uuid4().hex[:8]}"
    nonce = nonce or prompt.new_nonce()
    if idle_watts is None:
        idle_watts = idle_estimate(con)
    started = time.time()
    fake_gold = pool.answer if llm.is_fake() else None

    # ---- select
    emit({"type": "phase", "phase": "select", "t": time.time()})
    t_s0 = time.time()
    indices, diag = _select(pool, arm, settings)
    t_s1 = time.time()
    indices = sorted(int(i) for i in indices)
    store.insert_marker(con, run_id, "select", t_s0, t_s1)
    emit({"type": "selection", "indices": indices, "diag": diag})

    # ---- prefill + generate (one streamed call)
    def attempt(messages, phase_pre: str, phase_gen: str):
        first = {"seen": False}

        def on_token(piece: str):
            if not first["seen"]:
                first["seen"] = True
                emit({"type": "phase", "phase": phase_gen, "t": time.time()})
            emit({"type": "token", "text": piece})

        emit({"type": "phase", "phase": phase_pre, "t": time.time()})
        r = llm.generate(messages, settings, on_token=on_token, fake_gold=fake_gold)
        store.insert_marker(con, run_id, phase_pre, r.t_start, r.t_first)
        store.insert_marker(con, run_id, phase_gen, r.t_first, r.t_end)
        return r

    res = attempt(prompt.build_messages(pool, indices, arm, nonce), "prefill", "generate")
    ended = res.t_end
    res2 = None
    escalated = 0
    abstain_reason = ""
    answer_pruned = None
    if arm in SAFE_ARMS:
        abst, abstain_reason = prompt.is_abstention(res.text)
        if abst:
            escalated = 1
            answer_pruned = res.text
            emit({"type": "escalate", "reason": abstain_reason, "pruned_answer": res.text, "t": time.time()})
            res2 = attempt(prompt.build_messages(pool, None, "full_lc", prompt.new_nonce()),
                           "escalate_prefill", "escalate_generate")
            ended = res2.t_end
    final = res2 or res
    emit({"type": "phase", "phase": "done", "t": ended})

    # ---- energy: let the sampler flush past the end of the run, then join markers to samples
    store.wait_for_samples(con, ended + 0.05, timeout=max(flush_wait_s, 1.0))
    time.sleep(0.1)
    pe = store.phase_energy(run_id, con=con, idle_watts=idle_watts or 0.0)

    def g(ph, key):
        return (pe.get(ph) or {}).get(key)

    def total(phases, key):
        vals = [g(p, key) for p in phases]
        vals = [v for v in vals if v is not None]
        return sum(vals) if vals else None

    j_gross = total(store.ALL_PHASES, "gross_j")
    j_net = total(store.ALL_PHASES, "net_j")
    src = next((pe[p]["source"] for p in store.ALL_PHASES if p in pe and pe[p].get("source")), None)
    if src is None:
        src = power.sampler_source(con)

    golds = [x for x in (pool.meta.get("answers") if isinstance(pool.meta, dict) else None) or [pool.answer] if x]
    scored = [_em_f1(final.text, gd) for gd in golds] or [(0.0, 0.0)]
    em, f1 = max(x[0] for x in scored), max(x[1] for x in scored)   # SQuAD: best over the gold answers
    if arm in FULL_ARMS or escalated:
        context_tokens = int(pool.full_tokens)
    else:
        context_tokens = int(sum(pool.sentences[i].tokens for i in indices))
    both = [r for r in (res, res2) if r is not None]
    row = {
        "run_id": run_id, "qid": pool.qid, "question": pool.question, "gold": pool.answer,
        "arm": arm, "mode": mode, "started_at": started, "ended_at": ended,
        "config_hash": settings.hash(), "llm_backend": res.backend, "model_id": res.model_id,
        "tokens_in": sum(r.tokens_in for r in both), "tokens_out": sum(r.tokens_out for r in both),
        "context_tokens": context_tokens,
        "n_selected": pool.n if arm in FULL_ARMS else len(indices),
        "selected_json": json.dumps(indices),
        "ttft_s": res.ttft_s, "select_s": t_s1 - t_s0, "prefill_s": sum(r.prefill_s for r in both),
        "generate_s": sum(r.generate_s for r in both), "total_s": ended - started,
        "idle_watts": idle_watts, "joules_gross": j_gross, "joules_net": j_net,
        "joules_select": g("select", "net_j"), "joules_prefill": g("prefill", "net_j"),
        "joules_generate": g("generate", "net_j"), "energy_source": src,
        "answer": final.text, "em": em, "f1": f1, "sf_recall": sf_recall(pool, indices, arm),
        "referee_correct": None, "referee_score": None, "referee_reason": None, "referee_model": None,
        "bench_id": bench_id,
        "escalated": escalated, "joules_escalate": total(store.ESCALATE_PHASES, "net_j") if escalated else 0.0,
        "answer_pruned": answer_pruned,
    }
    store.insert_run(con, row)

    if arm in ("qwatt_lc", "qwatt_safe") and isinstance(diag, dict):
        # uncharged reference diagnostics (anneal/random), computed AFTER all measured phases and energy
        try:
            from qwatt import selectors
            ref = getattr(selectors, "lc_reference", None)
            if ref is not None:
                extra = ref(pool, diag, settings)
                if isinstance(extra, dict):
                    diag.update(extra)
                    emit({"type": "selection", "indices": indices, "diag": diag})
        except Exception:  # noqa: BLE001 - never fail a run because of diagnostics
            pass
    if arm in QAOA_ARMS and isinstance(diag, dict):
        store.insert_qaoa_run(con, {
            "run_id": run_id, "qid": pool.qid, "backend": diag.get("backend"),
            "n_qubits": diag.get("n_qubits"), "n_edges": diag.get("n_edges"),
            "depth": diag.get("depth"), "gate_count": diag.get("gate_count"), "shots": diag.get("shots"),
            "gamma_json": json.dumps(diag.get("gamma")), "beta_json": json.dumps(diag.get("beta")),
            "wall_s": diag.get("solver_wall_s"), "joules": g("select", "net_j"),
            "best_cost": diag.get("best_cost"), "exact_cost": diag.get("exact_cost"),
            "approx_ratio": diag.get("approx_ratio"), "amplification": diag.get("amplification"),
            "opt_in_shots": int(bool(diag.get("opt_in_shots"))), "dispatch": diag.get("dispatch"),
            "feasible_fallback": int(bool(diag.get("feasible_fallback"))),
            "diag_json": json.dumps(diag, default=_default),
        })
    con.close()
    row["energy_detail"] = {k: {kk: vv for kk, vv in v.items() if kk != "source"} for k, v in pe.items()}
    row["abstain_reason"] = abstain_reason
    emit({"type": "result", "row": row})
    return row


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # noqa: BLE001
        pass
    return str(o)

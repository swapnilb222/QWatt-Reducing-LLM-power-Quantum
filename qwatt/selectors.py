"""Selector registry (CONTRACTS.md). Every selector: (pool, budget, settings) -> (sorted indices, diag).

diag always has: arm, solver_wall_s, cost (QUBO cost of the selection), tokens (sum t_i of the selection;
for "full" it is pool.full_tokens because the pipeline sends pool.full_context).

"qwatt" (A3) = dispatch gate -> QAOA circuit on Quantum Rings (stored angles, one execution) -> budget decode.
Its diag also carries the exact optimum (brute force) for verification. Verification time is reported
separately as verify_wall_s and is NOT part of solver_wall_s (the selection does not depend on it).
"""
from __future__ import annotations

import hashlib
import os
import json
import time
from functools import lru_cache

import numpy as np

from qwatt import config, dispatch, qubo, solvers_classical as sc
from qwatt.types import Pool

SELECTOR_NAMES = ["full", "topk", "greedy", "exact", "anneal", "random_k", "qwatt"]
DEFAULT_ANGLES = {"p": 1, "gamma": [0.44], "beta": [2.55], "source": "default (angles.json missing)"}


@lru_cache(maxsize=1)
def _angles_cached(mtime: float) -> dict:
    if config.ANGLES_JSON.exists():
        d = json.loads(config.ANGLES_JSON.read_text())
        return {"p": d["p"], "gamma": d["gamma"], "beta": d["beta"], "source": "results/angles.json"}
    return DEFAULT_ANGLES


def load_angles() -> dict:
    m = config.ANGLES_JSON.stat().st_mtime if config.ANGLES_JSON.exists() else 0.0
    return _angles_cached(m)


def _qubo(pool: Pool, settings):
    eps_in = config.eps_in_default()
    h, J = qubo.build_qubo(pool, settings, eps_in)
    return h, J, pool.tokens, eps_in


def _finish(arm: str, x, h, J, t, wall: float, extra: dict | None = None) -> tuple[list[int], dict]:
    x = np.asarray(x).astype(np.uint8)
    idx = [int(i) for i in np.flatnonzero(x)]
    d = {"arm": arm, "solver_wall_s": float(wall), "cost": qubo.qubo_cost(x, h, J), "tokens": int(round(float(x @ t))),
         "bits": qubo.bits_to_str(x)}
    d.update(extra or {})
    return idx, d


# ---------------------------------------------------------------- exact verification cache
_EXACT: dict[str, tuple] = {}


def exact_solution(h, J, t, budget) -> tuple[np.ndarray, dict, np.ndarray]:
    """Brute force over 2^n, cached per (h, J, t, budget) so it is computed once per instance."""
    key = hashlib.sha1(np.concatenate([h, J.ravel(), t, [budget]]).tobytes()).hexdigest()
    if key not in _EXACT:
        costs = qubo.all_costs(h, J)
        x, info = sc.brute_force(h, J, t, budget, costs=costs)
        if len(_EXACT) > 64:
            _EXACT.clear()
        _EXACT[key] = (x, info, costs)
    return _EXACT[key]


# ---------------------------------------------------------------- classical arms
def sel_full(pool: Pool, budget: int, settings):
    h, J, t, _ = _qubo(pool, settings)
    idx, d = _finish("full", np.ones(pool.n), h, J, t, 0.0)
    d["tokens"] = int(pool.full_tokens)
    d["pool_tokens"] = int(t.sum())
    return list(range(pool.n)), d


def sel_topk(pool: Pool, budget: int, settings):
    h, J, t, _ = _qubo(pool, settings)
    t0 = time.perf_counter()
    x = sc.topk(pool.relevance, t, budget)
    return _finish("topk", x, h, J, t, time.perf_counter() - t0)


def sel_greedy(pool: Pool, budget: int, settings):
    h, J, t, _ = _qubo(pool, settings)
    t0 = time.perf_counter()
    x = sc.greedy(h, J, t, budget)
    return _finish("greedy", x, h, J, t, time.perf_counter() - t0)


def sel_exact(pool: Pool, budget: int, settings):
    h, J, t, _ = _qubo(pool, settings)
    t0 = time.perf_counter()
    x, info = sc.brute_force(h, J, t, budget)
    return _finish("exact", x, h, J, t, time.perf_counter() - t0, {"n_feasible": info["n_feasible"]})


def sel_anneal(pool: Pool, budget: int, settings):
    h, J, t, _ = _qubo(pool, settings)
    t0 = time.perf_counter()
    x = sc.anneal(h, J, t, budget, seed=settings.seed)
    return _finish("anneal", x, h, J, t, time.perf_counter() - t0)


def _qid_seed(qid: str, base: int) -> int:
    return base + int(hashlib.sha1(qid.encode()).hexdigest()[:6], 16)


def sel_random_k(pool: Pool, budget: int, settings):
    """A6 control: settings.shots uniform bitstrings, decoded exactly like the QAOA shots."""
    h, J, t, _ = _qubo(pool, settings)
    t0 = time.perf_counter()
    counts = sc.random_k(pool.n, settings.shots, seed=_qid_seed(pool.qid, settings.seed))
    samples = qubo.make_samples(counts, h, J, t, budget)
    bits, fb = qubo.decode_budget(samples, budget, h, t)
    return _finish("random_k", qubo.str_to_bits(bits), h, J, t, time.perf_counter() - t0, {"feasible_fallback": fb})


# ---------------------------------------------------------------- Q-Watt (A3)
def solver_metrics(samples: list[dict], best_bits: str, h, J, t, budget, shots: int) -> dict:
    """Plan 6.9 metrics against the exact optimum (brute force, cached)."""
    x_opt, info, costs = exact_solution(h, J, t, budget)
    best_cost = qubo.qubo_cost(qubo.str_to_bits(best_bits), h, J)
    span = info["c_worst_feasible"] - info["c_opt"]
    from qwatt.qaoa_sim import top_mask

    mask = top_mask(costs, 0.01)
    hits_top = sum(s["count"] for s in samples if mask[qubo.bits_to_int(qubo.str_to_bits(s["bits"]))])
    exact_bits = qubo.bits_to_str(x_opt)
    return {
        "exact_bits": exact_bits, "exact_cost": info["c_opt"],
        "approx_ratio": float((info["c_worst_feasible"] - best_cost) / span) if span > 1e-12 else 1.0,
        "opt_in_shots": bool(any(s["bits"] == exact_bits for s in samples)),
        "amplification": float((hits_top / max(1, shots)) / (mask.sum() / costs.size)),
        "c_worst_feasible": info["c_worst_feasible"], "n_feasible": info["n_feasible"],
        "unconstrained_opt_bits": qubo.bits_to_str(qubo.int_to_bits(info["k_opt_unconstrained"], len(h))),
        "unconstrained_opt_cost": info["c_opt_unconstrained"],
        "unconstrained_opt_feasible": info["unconstrained_opt_feasible"],
    }


def sel_qwatt(pool: Pool, budget: int, settings, on_progress=None):
    t_start = time.perf_counter()
    h, J, t, eps_in = _qubo(pool, settings)
    route, x_cf = dispatch.decide(h, J, t, budget)
    angles = load_angles()
    shots = int(settings.shots)
    if route == "circuit":
        from qwatt import qaoa_qr

        a, b, c0, s = qubo.normalized_ising(h, J)
        counts, info = qaoa_qr.run_qaoa(a, b, angles["gamma"], angles["beta"], shots=shots)
        samples = qubo.make_samples(counts, h, J, t, budget)
        best_bits, fallback = qubo.decode_budget(samples, budget, h, t)
    else:
        a, b, c0, s = qubo.normalized_ising(h, J)
        bits = qubo.bits_to_str(x_cf)
        samples = qubo.make_samples({bits: shots}, h, J, t, budget)
        best_bits, fallback = bits, False
        info = {"backend": "none (closed form)", "n_qubits": pool.n, "n_edges": len(qubo.edges(J)), "depth": 0,
                "gate_count": 0, "shots": 0, "gamma": [], "beta": [], "circuit_wall_s": 0.0, "backend_init_s": 0.0}
    solver_wall = time.perf_counter() - t_start

    t_v = time.perf_counter()
    metrics = solver_metrics(samples, best_bits, h, J, t, budget, shots if route == "circuit" else 1)
    verify_wall = time.perf_counter() - t_v

    best_cost = qubo.qubo_cost(qubo.str_to_bits(best_bits), h, J)
    iu = np.triu_indices(pool.n, 1)
    diag = {
        "dispatch": route, **{k: info[k] for k in ("backend", "n_qubits", "n_edges", "depth", "gate_count", "shots",
                                                   "gamma", "beta", "circuit_wall_s")},
        "backend_init_s": info.get("backend_init_s", 0.0), "ops": info.get("ops", {}),
        "transpiled_depth": info.get("transpiled_depth", 0), "transpiled_gate_count": info.get("transpiled_gate_count", 0),
        "angles_source": angles["source"], "p": len(info["gamma"]) or angles["p"],
        "samples": samples, "n_distinct": len(samples),
        "feasible_shot_frac": float(sum(x["count"] for x in samples if x["feasible"]) / max(1, sum(x["count"] for x in samples))),
        "best_bits": best_bits, "best_cost": best_cost, **metrics, "feasible_fallback": bool(fallback),
        "h": [float(v) for v in h], "t": [int(v) for v in t], "relevance": [float(v) for v in pool.relevance],
        "J_edges": [[i, j, w] for i, j, w in qubo.edges(J)],
        "ising_a": [float(v) for v in a], "ising_scale": float(s), "ising_c0": float(c0),
        "red_edges": [[int(i), int(j), float(pool.red[i, j])] for i, j in zip(*iu) if pool.red[i, j] > 0],
        "bridge_edges": [[int(i), int(j)] for i, j in zip(*iu) if pool.br[i, j] > 0],
        "eps_in": eps_in, "kappa": qubo.effective_kappa(settings, eps_in), "budget": int(budget),
        "verify_wall_s": verify_wall, "select_total_s": solver_wall + verify_wall,
    }
    return _finish("qwatt", qubo.str_to_bits(best_bits), h, J, t, solver_wall, diag)


# ================================================================ long-context selectors (pivot 04:35)
LC_SELECTOR_NAMES = ["qwatt_lc", "anneal_lc", "random_lc"]
SAMPLE_CAP = 300          # samples stored in the diag: best 300 distinct by cost (n_distinct reports the total)
EXACT_MAX_N = 22          # brute-force verification only up to this many qubits


def _lc_settings(settings):
    """LC selectors always use long-context settings; a 16-qubit Settings passed by mistake is replaced."""
    from qwatt import longdoc

    return settings if (settings is not None and (settings.extra or {}).get("mode") == "lc") else longdoc.lc_settings()


def load_lc_angles() -> dict:
    from qwatt import longdoc

    if longdoc.LC_ANGLES_JSON.exists():
        d = json.loads(longdoc.LC_ANGLES_JSON.read_text())
        return {"p": d["p"], "gamma": d["gamma"], "beta": d["beta"], "source": "results/lc_angles.json"}
    a = dict(load_angles())
    a["source"] = a["source"] + " (16-qubit angles; lc_angles.json missing)"
    return a


def _lc_qubo(pool: Pool, budget, settings):
    eps_in = config.eps_in_default()
    h, J, info = qubo.build_qubo_lc(pool, settings, eps_in, budget)
    return h, J, pool.tokens, info


def lc_anneal(h, J, t, budget, seed=0) -> np.ndarray:
    """Strong classical reference for n > 22 (no brute force): best of simulated annealing (budget-feasible moves)
    and greedy, each finished with a feasible 1-flip polish."""
    cands = [qubo.local_polish(sc.anneal(h, J, t, budget, sweeps=300, restarts=3, seed=seed), h, J, t, budget),
             qubo.local_polish(sc.greedy(h, J, t, budget), h, J, t, budget)]
    return min(cands, key=lambda x: qubo.qubo_cost(x, h, J))


def _gold_kept(pool: Pool, x) -> bool:
    return bool(any(pool.sentences[i].is_gold for i in np.flatnonzero(x)))


def sel_anneal_lc(pool: Pool, budget: int, settings):
    settings = _lc_settings(settings)
    h, J, t, info = _lc_qubo(pool, budget, settings)
    t0 = time.perf_counter()
    x = lc_anneal(h, J, t, budget, seed=settings.seed)
    return _finish("anneal_lc", x, h, J, t, time.perf_counter() - t0,
                   {"kept_bits": qubo.bits_to_str(x), "gold_kept": _gold_kept(pool, x), **info})


def _random_samples(pool, settings, shots, h, J, t, budget):
    counts = sc.random_k(pool.n, shots, seed=_qid_seed(pool.qid, settings.seed))
    return qubo.make_samples(counts, h, J, t, budget)


def sel_random_lc(pool: Pool, budget: int, settings):
    """K = shots uniform random bitstrings, decoded exactly like the QAOA shots (incl. repair)."""
    settings = _lc_settings(settings)
    h, J, t, info = _lc_qubo(pool, budget, settings)
    t0 = time.perf_counter()
    samples = _random_samples(pool, settings, int(settings.shots), h, J, t, budget)
    bits, dinfo = qubo.decode_budget_lc(samples, budget, h, J, t)
    x = qubo.str_to_bits(bits)
    return _finish("random_lc", x, h, J, t, time.perf_counter() - t0,
                   {"kept_bits": bits, "decode": dinfo, "feasible_fallback": dinfo["source"] != "raw_feasible",
                    "gold_kept": _gold_kept(pool, x), **info})


LC_REFERENCE_DEFAULT = os.getenv("QWATT_LC_REFERENCE", "0") == "1"
LC_CIRCUIT_RENT_DEFAULT = os.getenv("QWATT_LC_CIRCUIT_RENT", "1") == "1"   # budget-calibrated circuit rent (on)


def sel_qwatt_lc(pool: Pool, budget: int, settings, reference: bool | None = None):
    """QAOA p=1 (transferred angles) on Quantum Rings, one qubit per document sentence, budget decode with repair.

    The charged select path is ONLY: QUBO build -> circuit -> shot costs -> budget decode (+repair).
    Diagnostics (annealing reference, random-K control, cost histograms, 1-flip polish) live in lc_reference(),
    which the pipeline calls AFTER the select phase marker closes. reference=True (or env QWATT_LC_REFERENCE=1,
    or settings.extra["lc_reference"]) runs it inline for offline scripts.
    """
    from qwatt import qaoa_qr

    settings = _lc_settings(settings)
    if reference is None:
        reference = bool((settings.extra or {}).get("lc_reference", LC_REFERENCE_DEFAULT))
    shots = int(settings.shots)
    t_start = time.perf_counter()
    h, J, t, info = _lc_qubo(pool, budget, settings)
    route, x_cf = dispatch.decide(h, J, t, budget)
    angles = load_lc_angles()
    # Budget-calibrated circuit rent: raise the energy price in the CIRCUIT Hamiltonian until its typical
    # measurement fits the budget (exact p=1 marginals, cheap at any n). Shots are still scored with the original h.
    cal = None
    a, b, c0, s = qubo.normalized_ising(h, J)
    if (settings.extra or {}).get("circuit_rent", LC_CIRCUIT_RENT_DEFAULT) and route == "circuit":
        r_pow = pool.relevance ** float(settings.extra.get("r_power", 1.0))
        cal = qubo.calibrate_circuit_rent(h, J, t, r_pow, budget, angles["gamma"][0], angles["beta"][0], info["rho"],
                                          target=settings.extra.get("circuit_target"))
        a, b = cal.pop("a"), cal.pop("b")
    t_built = time.perf_counter()
    if route == "circuit":
        counts, cinfo = qaoa_qr.run_qaoa(a, b, angles["gamma"], angles["beta"], shots=shots)
    else:
        counts = {qubo.bits_to_str(x_cf): shots}
        cinfo = {"backend": "none (closed form)", "n_qubits": pool.n, "n_edges": len(qubo.edges(J)), "depth": 0,
                 "gate_count": 0, "shots": 0, "gamma": [], "beta": [], "circuit_wall_s": 0.0}
    t_circ = time.perf_counter()
    samples = qubo.make_samples(counts, h, J, t, budget)
    best_bits, dinfo = qubo.decode_budget_lc(samples, budget, h, J, t)
    t_end = time.perf_counter()
    solver_wall = t_end - t_start

    x = qubo.str_to_bits(best_bits)
    diag = {
        "dispatch": route, **{k: cinfo[k] for k in ("backend", "n_qubits", "n_edges", "depth", "gate_count", "shots",
                                                    "gamma", "beta", "circuit_wall_s")},
        "transpiled_depth": cinfo.get("transpiled_depth", 0), "transpiled_gate_count": cinfo.get("transpiled_gate_count", 0),
        "qubo_build_s": t_built - t_start, "circuit_call_s": t_circ - t_built, "decode_s": t_end - t_circ,
        "backend_init_s": cinfo.get("backend_init_s", 0.0), "qr_threads": cinfo.get("max_threads"),
        "angles_source": angles["source"], "p": len(cinfo["gamma"]) or angles["p"], "W": pool.meta.get("W"),
        "samples": samples[:SAMPLE_CAP], "n_distinct": len(samples),
        "shot_cost_counts": [[z["cost"], z["count"]] for z in samples],
        "feasible_shot_frac": float(sum(z["count"] for z in samples if z["feasible"]) / max(1, shots)),
        "mean_shot_tokens": float(sum(z["tokens"] * z["count"] for z in samples) / max(1, shots)),
        "decode": dinfo, "feasible_fallback": dinfo["source"] != "raw_feasible",
        "best_bits": best_bits, "best_cost": qubo.qubo_cost(x, h, J), "kept_bits": best_bits,
        "gold_kept": _gold_kept(pool, x), "paragraph_of": pool.meta.get("paragraph_of"),
        "h": [float(v) for v in h], "t": [int(v) for v in t], "relevance": [float(v) for v in pool.relevance],
        "J_edges": [[i, j, w] for i, j, w in qubo.edges(J)],
        "red_edges": [[int(i), int(j), float(pool.red[i, j])] for i, j in zip(*np.nonzero(np.triu(pool.red, 1)))],
        "bridge_edges": [[int(i), int(j)] for i, j in zip(*np.nonzero(np.triu(pool.br, 1)))],
        "ising_scale": float(s), "budget": int(budget), "reference_done": False, **info,
        # circuit Hamiltonian = original QUBO + mu_circuit per kept sentence (relevance units; in joules that is the
        # price of an average sentence, eps_in * tbar); kappa_circuit = mu_circuit / (eps_in * tbar)
        "rent_calibrated": cal is not None,
        "mu_circuit": cal["mu_circuit"] if cal else 0.0,
        "kappa_circuit": (cal["mu_circuit"] / (info["eps_in"] * float(t.mean()))) if cal else 0.0,
        "circuit_scale": cal["scale"] if cal else 1.0,
        "gamma_effective": cal["gamma_effective"] if cal else angles["gamma"][0],
        "predicted_feasible_frac": cal["predicted_feasible_frac"] if cal else None,
        "predicted_shot_tokens": cal["predicted_shot_tokens"] if cal else None,
        "from_circuit_shot": dinfo["source"] == "raw_feasible",
    }
    if reference:
        diag.update(lc_reference(pool, diag, settings))
    return _finish("qwatt_lc", x, h, J, t, solver_wall, diag)


def lc_reference(pool: Pool, diag: dict, settings=None) -> dict:
    """Dashboard / metrics diagnostics for a qwatt_lc diag. Call AFTER the select phase (not charged to it);
    merge the result into diag. Keys: reference{anneal_cost, random_best_cost, ...}, ratio_vs_anneal,
    cost_hist_edges / cost_hist_qaoa / cost_hist_random (same bins), mean_shot_cost_*, polished_* (1-flip polish,
    reported separately), exact metrics when n <= 22, reference_wall_s."""
    settings = _lc_settings(settings)
    t0 = time.perf_counter()
    budget = diag["budget"]
    h, J, t, _ = _lc_qubo(pool, budget, settings)
    shots = max(1, int(diag.get("shots") or settings.shots))
    x = qubo.str_to_bits(diag["best_bits"])
    best_cost = qubo.qubo_cost(x, h, J)
    xp = qubo.local_polish(x, h, J, t, budget)
    xa = lc_anneal(h, J, t, budget, seed=settings.seed)
    rs = _random_samples(pool, settings, shots, h, J, t, budget)
    rbits, rinfo = qubo.decode_budget_lc(rs, budget, h, J, t)
    ca, cr = qubo.qubo_cost(xa, h, J), qubo.qubo_cost(qubo.str_to_bits(rbits), h, J)
    sc_ = np.array(diag["shot_cost_counts"], float).reshape(-1, 2)
    rc = np.array([[z["cost"], z["count"]] for z in rs], float)
    hist = qubo.cost_histograms(sc_[:, 0], sc_[:, 1], rc[:, 0], rc[:, 1])
    out = {
        "reference": {"anneal_cost": ca, "random_best_cost": cr, "anneal_bits": qubo.bits_to_str(xa),
                      "random_bits": rbits, "random_source": rinfo["source"],
                      "random_feasible_shot_frac": float(sum(z["count"] for z in rs if z["feasible"]) / shots),
                      "random_ratio_vs_anneal": float(cr / ca) if ca < -1e-12 else None,
                      "anneal_gold_kept": _gold_kept(pool, xa), "random_gold_kept": _gold_kept(pool, qubo.str_to_bits(rbits))},
        # good selections have negative cost: ratio 1.0 = as good as the classical reference, < 1 worse, > 1 better
        "ratio_vs_anneal": float(best_cost / ca) if ca < -1e-12 else None,
        # same bins for both: {"edges": [nbins+1], "counts": [nbins]} (counts = fraction of shots per bin)
        "cost_hist_edges": hist["edges"], "cost_hist_qaoa": {"edges": hist["edges"], "counts": hist["qaoa"]},
        "cost_hist_random": {"edges": hist["edges"], "counts": hist["random"]},
        "mean_shot_cost_qaoa": float((sc_[:, 0] * sc_[:, 1]).sum() / shots),
        "mean_shot_cost_random": float((rc[:, 0] * rc[:, 1]).sum() / shots),
        "polished_bits": qubo.bits_to_str(xp), "polished_cost": qubo.qubo_cost(xp, h, J),
        "polished_gold_kept": _gold_kept(pool, xp), "reference_done": True,
    }
    if pool.n <= EXACT_MAX_N:
        out.update(solver_metrics(diag["samples"], diag["best_bits"], h, J, t, budget, shots))
    out["reference_wall_s"] = time.perf_counter() - t0
    return out


_REGISTRY = {"full": sel_full, "topk": sel_topk, "greedy": sel_greedy, "exact": sel_exact,
             "anneal": sel_anneal, "random_k": sel_random_k, "qwatt": sel_qwatt,
             "qwatt_lc": sel_qwatt_lc, "anneal_lc": sel_anneal_lc, "random_lc": sel_random_lc}


def get_selector(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown selector {name!r}; choose from {SELECTOR_NAMES}")
    return _REGISTRY[name]

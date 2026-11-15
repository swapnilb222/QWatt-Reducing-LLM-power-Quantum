"""Pure computations behind the dashboard numbers (plan sections 6.7, 6.9, 6.10)."""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

# Arms hidden from every dashboard view (presentation only; DB rows untouched). Empty the set to show them again.
HIDDEN_ARMS: set[str] = {"topk", "topk_lc"}


def visible(name) -> bool:
    return str(name) not in HIDDEN_ARMS


PHONE_CHARGE_WH = 15.0   # stated on screen: one phone charge = 15 Wh
QUESTIONS_PER_DAY = 1000


def fnum(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def joules(row) -> float | None:
    """Net-of-idle joules (plan 6.10), gross as fallback."""
    v = fnum(row.get("joules_net")) if hasattr(row, "get") else None
    if v is None:
        v = fnum(row.get("joules_gross"))
    return v


def correct(row) -> bool | None:
    """Referee verdict when present, else local exact match / F1 >= 0.6. None if unknown."""
    rc = fnum(row.get("referee_correct"))
    if rc is not None:
        return rc >= 0.5
    em = fnum(row.get("em"))
    f1 = fnum(row.get("f1"))
    if em is None and f1 is None:
        return None
    return bool((em or 0) >= 1 or (f1 or 0) >= 0.6)


def correctness_source(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    if "referee_correct" in df and df["referee_correct"].notna().all():
        return "Gemini referee"
    if "referee_correct" in df and df["referee_correct"].notna().any():
        return "Gemini referee (EM/F1 where not yet refereed)"
    return "EM / F1 ≥ 0.6 (referee pending)"


def bootstrap_median_ci(x, n_boot: int = 2000, seed: int = 7, alpha: float = 0.05):
    x = np.asarray([v for v in x if v is not None and not math.isnan(v)], dtype=float)
    if x.size == 0:
        return None, None, None
    med = float(np.median(x))
    if x.size < 2:
        return med, None, None
    rng = np.random.default_rng(seed)
    bs = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return med, float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def bootstrap_mean_ci(x, n_boot: int = 2000, seed: int = 7, alpha: float = 0.05):
    x = np.asarray([v for v in x if v is not None and not math.isnan(v)], dtype=float)
    if x.size == 0:
        return None, None, None
    m = float(np.mean(x))
    if x.size < 2:
        return m, None, None
    rng = np.random.default_rng(seed)
    bs = np.mean(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return m, float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def bench_runs(runs: pd.DataFrame, bench_id: str | None = None) -> pd.DataFrame:
    if runs.empty or "mode" not in runs:
        return pd.DataFrame()
    df = runs[runs["mode"] == "bench"].copy()
    if "arm" in df:
        df = df[~df["arm"].isin(HIDDEN_ARMS)]
    if bench_id is not None and "bench_id" in df:
        df = df[df["bench_id"].fillna("(unlabeled)") == bench_id]
    if df.empty:
        return df
    # latest row per (qid, arm)
    df = df.sort_values("started_at").groupby(["qid", "arm"], as_index=False).tail(1)
    df["J"] = df.apply(joules, axis=1)
    # A non-positive net-of-idle energy means the idle baseline was measured above the run's own power
    # (a baseline glitch, not a free answer): exclude it from energy statistics and say so on screen.
    df["J_invalid"] = df["J"].notna() & (df["J"] <= 0)
    df.loc[df["J_invalid"], "J"] = np.nan
    df["ok"] = df.apply(correct, axis=1)
    return df


LC_ARMS = ("full_lc", "qwatt_safe", "qwatt_lc")


def is_lc_batch(bid) -> bool:
    return str(bid).startswith("lc-")


def bench_ids(runs: pd.DataFrame, kind: str = "16") -> list[str]:
    """Benchmark batches of one kind ('16' = original 16-qubit arms, 'lc' = long-context batches 'lc-*'),
    the default first: most questions, then most recent."""
    if runs.empty or "mode" not in runs:
        return []
    b = runs[runs["mode"] == "bench"].copy()
    if b.empty:
        return []
    b["bid"] = b["bench_id"].fillna("(unlabeled)") if "bench_id" in b else "(unlabeled)"
    lcmask = b["bid"].map(is_lc_batch) | b["arm"].isin(LC_ARMS)
    b = b[lcmask] if kind == "lc" else b[~lcmask]
    if b.empty:
        return []
    g = b.groupby("bid").agg(nq=("qid", "nunique"), last=("started_at", "max"))
    return list(g.sort_values(["nq", "last"], ascending=[False, False]).index)


def paired(df: pd.DataFrame, a: str = "full", b: str = "qwatt") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pa = df[df["arm"] == a].set_index("qid")
    pb = df[df["arm"] == b].set_index("qid")
    q = pa.index.intersection(pb.index)
    if len(q) == 0:
        return pd.DataFrame()
    out = pd.DataFrame({
        "J_a": pa.loc[q, "J"], "J_b": pb.loc[q, "J"],
        "tok_a": pa.loc[q, "tokens_in"], "tok_b": pb.loc[q, "tokens_in"],
        "ctx_a": pa.loc[q, "context_tokens"], "ctx_b": pb.loc[q, "context_tokens"],
    })
    out["G_a"] = pa.loc[q, "joules_gross"]
    out["G_b"] = pb.loc[q, "joules_gross"]
    out["excluded"] = out["J_a"].isna() | out["J_b"].isna()
    out.attrs["n_excluded"] = int(out["excluded"].sum())
    out.attrs["gross_saving"] = (1 - out["G_b"] / out["G_a"]).dropna().tolist()
    n_ex = out.attrs["n_excluded"]
    gs = out.attrs["gross_saving"]
    out = out[~out["excluded"]].copy()
    out.attrs["n_excluded"], out.attrs["gross_saving"] = n_ex, gs
    out["saving"] = 1 - out["J_b"] / out["J_a"]
    out["delta_J"] = out["J_a"] - out["J_b"]
    return out


def jpc(df: pd.DataFrame, arm: str) -> tuple[float | None, int, int]:
    d = df[df["arm"] == arm] if not df.empty else df
    if d.empty:
        return None, 0, 0
    d = d[d["J"].notna()]  # same rows in numerator and denominator
    tot = d["J"].dropna().sum()
    n_ok = int(d["ok"].fillna(False).astype(bool).sum())
    n = int(d["ok"].notna().sum())
    return (float(tot) / n_ok if n_ok else None), n_ok, n


def correct_counts(df: pd.DataFrame, arm: str) -> tuple[int, int]:
    """(correct, graded) over every row of the arm; correctness does not depend on energy validity."""
    d = df[df["arm"] == arm] if not df.empty else df
    if d.empty:
        return 0, 0
    return int(d["ok"].fillna(False).astype(bool).sum()), int(d["ok"].notna().sum())


def hero(df: pd.DataFrame) -> dict:
    p = paired(df)
    med, lo, hi = bootstrap_median_ci(list(p["saving"])) if not p.empty else (None, None, None)
    jf, _, _ = jpc(df, "full")
    jq, _, _ = jpc(df, "qwatt")
    okf, nf = correct_counts(df, "full")
    okq, nq = correct_counts(df, "qwatt")
    tok_removed = None
    tok_frac = None
    if not p.empty:
        d = (p["tok_a"] - p["tok_b"]).dropna()
        if not d.empty:
            tok_removed = float(np.median(d))
            tok_frac = float(np.median((d / p.loc[d.index, "tok_a"]).dropna()))
    dj = float(np.median(p["delta_J"])) if not p.empty else None
    gmed, glo, ghi = bootstrap_median_ci(p.attrs.get("gross_saving", [])) if not p.empty else (None, None, None)
    return {"saving": med, "saving_lo": lo, "saving_hi": hi, "n_pairs": len(p),
            "n_excluded": p.attrs.get("n_excluded", 0) if not p.empty else 0,
            "gross_saving": gmed, "gross_lo": glo, "gross_hi": ghi,
            "n_invalid": int(df["J_invalid"].sum()) if "J_invalid" in df else 0,
            "jpc_full": jf, "jpc_qwatt": jq, "ok_full": okf, "n_full": nf, "ok_qwatt": okq, "n_qwatt": nq,
            "tok_removed": tok_removed, "tok_frac": tok_frac, "delta_J_median": dj}


def scoreboard(df: pd.DataFrame, arms=("full", "topk", "greedy", "qwatt")) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for arm in [a for a in arms if a in set(df["arm"]) and visible(a)]:
        d = df[df["arm"] == arm]
        m, lo, hi = bootstrap_mean_ci(list(d["J"].dropna()))
        j, _, _ = jpc(df, arm)
        ok, n = correct_counts(df, arm)
        rs = d["referee_score"].dropna() if "referee_score" in d else pd.Series(dtype=float)
        rows.append({
            "arm": arm, "n": len(d), "J_mean": m, "J_lo": lo, "J_hi": hi, "JPC": j,
            "correct": ok, "graded": n,
            "tokens_in": d["tokens_in"].mean(), "tokens_out": d["tokens_out"].mean(),
            "ttft_s": d["ttft_s"].mean(), "total_s": d["total_s"].mean(),
            "EM": d["em"].mean(), "F1": d["f1"].mean(),
            "referee": rs.mean() if len(rs) else None,
            "sf_recall": d["sf_recall"].mean() if "sf_recall" in d else None,
            "J_select": max(0.0, d["joules_select"].mean()) if d["joules_select"].notna().any() else None,
            "J_prefill": d["joules_prefill"].mean(), "J_generate": d["joules_generate"].mean(),
            "n_invalid": int(d["J_invalid"].sum()) if "J_invalid" in d else 0,
            "escalated": float(d["escalated"].fillna(0).mean()) if "escalated" in d else None,
            "J_escalate": float(_escalation_j(d).mean()) if "escalated" in d else 0.0,
        })
    return pd.DataFrame(rows)


def _escalation_j(d: pd.DataFrame) -> pd.Series:
    """Energy of the safety-net second attempt: total minus first-attempt phases (escalated runs only)."""
    first = d[["joules_select", "joules_prefill", "joules_generate"]].fillna(0).sum(axis=1)
    extra = (d["J"].fillna(0) - first).clip(lower=0)
    return extra.where(d["escalated"].fillna(0) > 0, 0.0)


def extrapolate(delta_J_per_q: float | None) -> dict | None:
    if delta_J_per_q is None:
        return None
    wh = delta_J_per_q * QUESTIONS_PER_DAY / 3600.0
    return {"wh_per_day": wh, "phone_charges": wh / PHONE_CHARGE_WH,
            "kwh_per_year": wh * 365 / 1000.0}


# ---------------------------------------------------------------- energy model / dial
def predicted_joules(cal: dict | None, tokens_in: float, tokens_out: float) -> float | None:
    if not cal:
        return None
    e0, ei, eo = fnum(cal.get("e0")), fnum(cal.get("eps_in")), fnum(cal.get("eps_out"))
    if ei is None:
        return None
    return (e0 or 0.0) + ei * tokens_in + (eo or 0.0) * tokens_out


def decode_local(samples: list[dict], budget: float) -> dict | None:
    """Fallback for qwatt.qubo.decode_budget: min cost among samples with tokens <= budget."""
    feas = [s for s in samples if fnum(s.get("tokens")) is not None and s["tokens"] <= budget]
    if not feas:
        return None
    return min(feas, key=lambda s: s.get("cost", float("inf")))


def decode(samples: list[dict], budget: float, diag: dict | None = None) -> tuple[dict | None, str]:
    """(sample dict with bits/cost/tokens[/fallback], which decoder). Prefers qwatt.qubo.decode_budget."""
    diag = diag or {}
    h = diag.get("h")
    t = diag.get("t")
    try:
        from qwatt.qubo import decode_budget
        hh = np.asarray(h, dtype=float) if h else None
        tt = np.asarray(t, dtype=float) if t else None
        res = decode_budget(samples, budget, hh, tt)
        fb = bool(res[1]) if isinstance(res, tuple) and len(res) > 1 and isinstance(res[1], (bool, np.bool_)) else False
        s = _coerce_sample(res, samples)
        if s is not None:
            s = dict(s)
            if s.get("tokens") is None and t:
                s["tokens"] = int(sum(int(t[i]) for i, c in enumerate(s["bits"]) if c == "1"))
            if s.get("cost") is None and h:
                s["cost"] = bits_cost(s["bits"], h, diag.get("J_edges") or [])
            s["fallback"] = fb
            return s, "qwatt.qubo.decode_budget"
    except Exception:
        pass
    return decode_local(samples, budget), "local filter"


def bits_cost(bits: str, h: list, j_edges: list) -> float:
    x = [1 if c == "1" else 0 for c in bits]
    c = sum(float(h[i]) * x[i] for i in range(min(len(h), len(x))))
    for e in j_edges:
        i, j, w = int(e[0]), int(e[1]), float(e[2])
        if i < len(x) and j < len(x):
            c += w * x[i] * x[j]
    return c


def _coerce_sample(res: Any, samples: list[dict]) -> dict | None:
    if res is None:
        return None
    if isinstance(res, dict):
        if "bits" in res:
            base = next((s for s in samples if s.get("bits") == res["bits"]), {})
            return {**base, **res}
        if "best_bits" in res:
            return _coerce_sample(res["best_bits"], samples)
        return None
    if isinstance(res, str):
        return next((s for s in samples if s.get("bits") == res), {"bits": res})
    if isinstance(res, (tuple, list)):
        for el in res:
            if isinstance(el, str) and set(el) <= {"0", "1"}:
                return _coerce_sample(el, samples)
            if isinstance(el, dict):
                c = _coerce_sample(el, samples)
                if c:
                    return c
        # a list of indices?
        if all(isinstance(v, (int, np.integer)) for v in res) and samples:
            n = len(samples[0].get("bits", ""))
            bits = "".join("1" if i in set(int(v) for v in res) else "0" for i in range(n))
            return _coerce_sample(bits, samples)
    return None


def frontier(samples: list[dict]) -> list[tuple[int, float]]:
    """Best cost achievable at each token budget, from the stored shots (step function)."""
    pts = sorted(((int(s["tokens"]), float(s["cost"])) for s in samples
                  if fnum(s.get("tokens")) is not None and fnum(s.get("cost")) is not None))
    out, best = [], float("inf")
    for t, c in pts:
        if c < best:
            best = c
            out.append((t, c))
    return out


def bits_to_indices(bits: str | None) -> list[int]:
    if not bits:
        return []
    return [i for i, ch in enumerate(bits) if ch == "1"]


def parse_selected(v) -> list[int]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return []
    try:
        obj = json.loads(v) if isinstance(v, str) else v
    except Exception:
        return []
    if isinstance(obj, dict):
        obj = obj.get("indices") or obj.get("selected") or []
    try:
        return [int(i) for i in obj]
    except Exception:
        return []


def integrate_joules(samples: pd.DataFrame, t0: float, t1: float, idle: float | None) -> float | None:
    """Net-of-idle joules over [t0, t1] from the power trace (live counter)."""
    if samples is None or samples.empty:
        return None
    s = samples[(samples["ts"] >= t0) & (samples["ts"] <= t1)]
    if len(s) < 2:
        return None
    w = s["pkg_watts"].astype(float).to_numpy()
    if idle is not None:
        w = np.clip(w - idle, 0, None)
    t = s["ts"].astype(float).to_numpy()
    return float(np.trapezoid(w, t)) if hasattr(np, "trapezoid") else float(np.trapz(w, t))


def all_costs(h: list, j_edges: list) -> np.ndarray:
    """QUBO cost of every one of the 2^n bitstrings (uniform-random baseline), from diag h / J_edges."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    idx = np.arange(1 << n, dtype=np.int64)
    X = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float32)
    c = X @ h.astype(np.float32)
    for e in j_edges or []:
        i, j, w = int(e[0]), int(e[1]), float(e[2])
        c += np.float32(w) * X[:, i] * X[:, j]
    return c.astype(np.float64)


def cost_distribution(diag: dict, n_bins: int = 34) -> dict | None:
    """Shot-weighted cost histogram of the QAOA samples vs the exact uniform distribution over 2^n."""
    h, samples = diag.get("h"), diag.get("samples") or []
    if not h or not samples or len(h) > 22:
        return None
    allc = all_costs(h, diag.get("J_edges") or [])
    sc = np.array([fnum(s.get("cost")) if fnum(s.get("cost")) is not None else np.nan for s in samples])
    cnt = np.array([int(s.get("count", 0)) for s in samples], dtype=float)
    ok = ~np.isnan(sc)
    sc, cnt = sc[ok], cnt[ok]
    shots = cnt.sum()
    if shots <= 0:
        return None
    lo, hi = float(min(allc.min(), sc.min())), float(max(allc.max(), sc.max()))
    edges = np.linspace(lo, hi, n_bins + 1)
    u, _ = np.histogram(allc, bins=edges)
    q, _ = np.histogram(sc, bins=edges, weights=cnt)
    thr = float(np.quantile(allc, 0.01))
    p_top_q = float(cnt[sc <= thr].sum() / shots)
    p_top_u = float((allc <= thr).mean())
    return {"edges": edges, "uniform": 100 * u / allc.size, "qaoa": 100 * q / shots, "thr": thr,
            "p_top_q": p_top_q, "p_top_u": p_top_u, "amp": p_top_q / p_top_u if p_top_u else None,
            "mean_q": float((sc * cnt).sum() / shots), "mean_u": float(allc.mean()),
            "exact_cost": fnum(diag.get("exact_cost")), "shots": int(shots), "n": len(h)}

"""SQLite (WAL) store: schema, inserts, sample reads, phase energy integration.

Source of truth for power samples, phase markers, runs and qaoa_runs.
Energy per phase is derived by joining phase markers to samples (never by the pipeline itself).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from qwatt import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS power_samples(
    ts REAL PRIMARY KEY, pkg_watts REAL, pp0_watts REAL, pp1_watts REAL,
    pkg_joules_cum REAL, sys_watts REAL, source TEXT);
CREATE TABLE IF NOT EXISTS sampler_heartbeat(
    id INTEGER PRIMARY KEY CHECK(id=1), ts REAL, pid INTEGER, source TEXT);
CREATE TABLE IF NOT EXISTS phase_markers(
    run_id TEXT, phase TEXT, t0 REAL, t1 REAL, PRIMARY KEY(run_id, phase));
CREATE TABLE IF NOT EXISTS runs(
    run_id TEXT PRIMARY KEY, qid TEXT, question TEXT, gold TEXT, arm TEXT, mode TEXT,
    started_at REAL, ended_at REAL, config_hash TEXT, llm_backend TEXT, model_id TEXT,
    tokens_in INTEGER, tokens_out INTEGER, context_tokens INTEGER, n_selected INTEGER,
    selected_json TEXT,
    ttft_s REAL, select_s REAL, prefill_s REAL, generate_s REAL, total_s REAL,
    idle_watts REAL, joules_gross REAL, joules_net REAL,
    joules_select REAL, joules_prefill REAL, joules_generate REAL, energy_source TEXT,
    answer TEXT, em REAL, f1 REAL, sf_recall REAL,
    referee_correct INTEGER, referee_score REAL, referee_reason TEXT, referee_model TEXT,
    bench_id TEXT, escalated INTEGER, joules_escalate REAL, answer_pruned TEXT);
CREATE TABLE IF NOT EXISTS qaoa_runs(
    run_id TEXT PRIMARY KEY, qid TEXT, backend TEXT, n_qubits INTEGER, n_edges INTEGER,
    depth INTEGER, gate_count INTEGER, shots INTEGER, gamma_json TEXT, beta_json TEXT,
    wall_s REAL, joules REAL, best_cost REAL, exact_cost REAL, approx_ratio REAL,
    amplification REAL, opt_in_shots INTEGER, dispatch TEXT, feasible_fallback INTEGER,
    diag_json TEXT);
"""

RUN_COLS = [
    "run_id", "qid", "question", "gold", "arm", "mode", "started_at", "ended_at", "config_hash",
    "llm_backend", "model_id", "tokens_in", "tokens_out", "context_tokens", "n_selected",
    "selected_json", "ttft_s", "select_s", "prefill_s", "generate_s", "total_s", "idle_watts",
    "joules_gross", "joules_net", "joules_select", "joules_prefill", "joules_generate",
    "energy_source", "answer", "em", "f1", "sf_recall", "referee_correct", "referee_score",
    "referee_reason", "referee_model", "bench_id", "escalated", "joules_escalate", "answer_pruned",
]
# additive columns (added to pre-existing databases by connect())
EXTRA_RUN_COLS = {"escalated": "INTEGER", "joules_escalate": "REAL", "answer_pruned": "TEXT"}
QAOA_COLS = [
    "run_id", "qid", "backend", "n_qubits", "n_edges", "depth", "gate_count", "shots",
    "gamma_json", "beta_json", "wall_s", "joules", "best_cost", "exact_cost", "approx_ratio",
    "amplification", "opt_in_shots", "dispatch", "feasible_fallback", "diag_json",
]
SAMPLE_COLS = ["ts", "pkg_watts", "pp0_watts", "pp1_watts", "pkg_joules_cum", "sys_watts", "source"]

PHASES = ("select", "prefill", "generate")
ESCALATE_PHASES = ("escalate_prefill", "escalate_generate")
ALL_PHASES = PHASES + ESCALATE_PHASES


def connect(path: str | Path | None = None, timeout: float = 30.0) -> sqlite3.Connection:
    p = Path(path) if path else config.DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=timeout, isolation_level=None)  # autocommit
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    for col, typ in EXTRA_RUN_COLS.items():
        if col not in have:
            try:
                con.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass  # another process added it concurrently
    try:
        con.execute("CREATE INDEX IF NOT EXISTS idx_runs_bench ON runs(bench_id, qid, arm)")
    except sqlite3.OperationalError:
        pass
    return con


def _jsonable(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v)
    try:  # numpy scalars
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass
    return v


def _insert(con, table: str, cols: Sequence[str], row: dict, replace: bool = True) -> None:
    vals = [_jsonable(row.get(c)) for c in cols]
    q = f"INSERT OR {'REPLACE' if replace else 'IGNORE'} INTO {table}({','.join(cols)}) " \
        f"VALUES({','.join('?' * len(cols))})"
    con.execute(q, vals)


def insert_run(con, row: dict) -> None:
    _insert(con, "runs", RUN_COLS, row)


def update_run(con, run_id: str, **fields) -> None:
    ks = [k for k in fields if k in RUN_COLS and k != "run_id"]
    if not ks:
        return
    con.execute(f"UPDATE runs SET {','.join(k + '=?' for k in ks)} WHERE run_id=?",
                [_jsonable(fields[k]) for k in ks] + [run_id])


def insert_qaoa_run(con, row: dict) -> None:
    _insert(con, "qaoa_runs", QAOA_COLS, row)


def insert_marker(con, run_id: str, phase: str, t0: float, t1: float | None) -> None:
    con.execute("INSERT OR REPLACE INTO phase_markers(run_id,phase,t0,t1) VALUES(?,?,?,?)",
                (run_id, phase, t0, t1))


def insert_samples(con, rows: Iterable[dict]) -> None:
    con.execute("BEGIN")
    try:
        for r in rows:
            _insert(con, "power_samples", SAMPLE_COLS, r)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def set_heartbeat(con, ts: float, pid: int, source: str) -> None:
    con.execute("INSERT OR REPLACE INTO sampler_heartbeat(id,ts,pid,source) VALUES(1,?,?,?)",
                (ts, pid, source))


def get_heartbeat(con=None) -> dict | None:
    own = con is None
    con = con or connect()
    try:
        r = con.execute("SELECT ts,pid,source FROM sampler_heartbeat WHERE id=1").fetchone()
        return dict(r) if r else None
    finally:
        if own:
            con.close()


def read_samples(con, t0: float, t1: float, pad: float = 0.0) -> list[dict]:
    rows = con.execute("SELECT * FROM power_samples WHERE ts BETWEEN ? AND ? ORDER BY ts",
                       (t0 - pad, t1 + pad)).fetchall()
    return [dict(r) for r in rows]


def latest_sample(con) -> dict | None:
    r = con.execute("SELECT * FROM power_samples ORDER BY ts DESC LIMIT 1").fetchone()
    return dict(r) if r else None


# --------------------------------------------------------------------------- energy math
def _cum_at(samples: Sequence[dict], t: float) -> float | None:
    """Linear interpolation of the cumulative joule counter at time t (None if not bracketed)."""
    pts = [(s["ts"], s["pkg_joules_cum"]) for s in samples if s.get("pkg_joules_cum") is not None]
    if len(pts) < 2:
        return None
    if t < pts[0][0] or t > pts[-1][0]:
        return None
    lo, hi = 0, len(pts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pts[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (ta, ca), (tb, cb) = pts[lo], pts[hi]
    if tb == ta:
        return ca
    return ca + (cb - ca) * (t - ta) / (tb - ta)


def energy_cum(samples: Sequence[dict], t0: float, t1: float) -> float | None:
    """Gross joules in [t0,t1] by differencing the interpolated cumulative energy counter."""
    a, b = _cum_at(samples, t0), _cum_at(samples, t1)
    if a is None or b is None:
        return None
    return max(0.0, b - a)


def energy_trapz(samples: Sequence[dict], t0: float, t1: float, key: str = "pkg_watts") -> float | None:
    """Gross joules in [t0,t1] by trapezoid on the watts column (cross-check)."""
    pts = [(s["ts"], s[key]) for s in samples if s.get(key) is not None]
    if len(pts) < 2:
        return None
    # clip with linear interpolation at the edges
    def w_at(t):
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        for i in range(1, len(pts)):
            if pts[i][0] >= t:
                (ta, wa), (tb, wb) = pts[i - 1], pts[i]
                return wa + (wb - wa) * (t - ta) / (tb - ta) if tb > ta else wa
        return pts[-1][1]
    inner = [(t, w) for t, w in pts if t0 < t < t1]
    seq = [(t0, w_at(t0))] + inner + [(t1, w_at(t1))]
    return sum((seq[i + 1][0] - seq[i][0]) * (seq[i + 1][1] + seq[i][1]) / 2 for i in range(len(seq) - 1))


def mean_watts(samples: Sequence[dict], t0: float | None = None, t1: float | None = None) -> float | None:
    """Mean package watts over a window, from the cumulative counter when possible."""
    if not samples:
        return None
    t0 = samples[0]["ts"] if t0 is None else t0
    t1 = samples[-1]["ts"] if t1 is None else t1
    if t1 <= t0:
        return None
    e = energy_cum(samples, t0, t1)
    if e is None:
        e = energy_trapz(samples, t0, t1)
    return None if e is None else e / (t1 - t0)


def phase_energy_from(samples: Sequence[dict], markers: dict[str, tuple[float, float]],
                      idle_watts: float | None) -> dict[str, dict]:
    """Pure function: per phase gross/net joules from samples + marker windows."""
    out: dict[str, dict] = {}
    src = next((s.get("source") for s in samples if s.get("source")), None)
    for ph, (t0, t1) in markers.items():
        if t1 is None:
            continue
        dt = max(0.0, t1 - t0)
        g = energy_cum(samples, t0, t1)
        method = "cum"
        if g is None:
            g = energy_trapz(samples, t0, t1)
            method = "trapz"
        tz = energy_trapz(samples, t0, t1)
        net = None if g is None else g - (idle_watts or 0.0) * dt
        out[ph] = {"t0": t0, "t1": t1, "seconds": dt, "gross_j": g, "net_j": net,
                   "trapz_j": tz, "method": method, "source": src}
    return out


def get_markers(con, run_id: str) -> dict[str, tuple[float, float]]:
    rows = con.execute("SELECT phase,t0,t1 FROM phase_markers WHERE run_id=?", (run_id,)).fetchall()
    return {r["phase"]: (r["t0"], r["t1"]) for r in rows}


def phase_energy(run_id: str, con=None, idle_watts: float | None = None) -> dict[str, dict]:
    """Per phase energy for a run (gross, net of idle_watts*dt). Includes a 'total' pseudo-phase
    spanning the first marker start to the last marker end."""
    own = con is None
    con = con or connect()
    try:
        markers = get_markers(con, run_id)
        if not markers:
            return {}
        if idle_watts is None:
            r = con.execute("SELECT idle_watts FROM runs WHERE run_id=?", (run_id,)).fetchone()
            idle_watts = r["idle_watts"] if r and r["idle_watts"] is not None else 0.0
        t0 = min(m[0] for m in markers.values())
        t1 = max(m[1] for m in markers.values() if m[1] is not None)
        samples = read_samples(con, t0, t1, pad=1.5)
        out = phase_energy_from(samples, markers, idle_watts)
        out.update(phase_energy_from(samples, {"total": (t0, t1)}, idle_watts))
        return out
    finally:
        if own:
            con.close()


def wait_for_samples(con, t: float, timeout: float = 1.5) -> bool:
    """Block until a sample at/after t exists (sampler flushed past t)."""
    end = time.time() + timeout
    while time.time() < end:
        r = con.execute("SELECT MAX(ts) AS m FROM power_samples").fetchone()
        if r and r["m"] is not None and r["m"] >= t:
            return True
        time.sleep(0.05)
    return False


def idle_watts_recent(con, window_s: float = 3.0, end: float | None = None) -> float | None:
    end = end or time.time()
    s = read_samples(con, end - window_s, end)
    return mean_watts(s)


def measure_idle(seconds: float = 3.0, con=None) -> float | None:
    """Idle baseline: sleep `seconds` and return mean package watts over that window
    (differencing the cumulative counter). Requires the sampler to be running."""
    own = con is None
    con = con or connect()
    try:
        t0 = time.time()
        time.sleep(seconds)
        wait_for_samples(con, time.time() - 0.3, 1.0)
        s = read_samples(con, t0 - 1.0, time.time() + 1.0)
        return mean_watts(s, t0, t0 + seconds)
    finally:
        if own:
            con.close()


def run_done(con, bench_id: str, qid: str, arm: str) -> bool:
    r = con.execute("SELECT 1 FROM runs WHERE bench_id=? AND qid=? AND arm=? AND ended_at IS NOT NULL "
                    "AND joules_gross IS NOT NULL", (bench_id, qid, arm)).fetchone()
    return r is not None


def get_run(con, run_id: str) -> dict | None:
    r = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(r) if r else None

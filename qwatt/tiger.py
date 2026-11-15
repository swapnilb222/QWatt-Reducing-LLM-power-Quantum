"""Tiger Data (TimescaleDB on Tiger Cloud) analytics layer. SQLite stays the source of truth.

Everything no-ops gracefully (returns None / 0) when TIGER_DATABASE_URL is unset or unreachable.
Never prints the connection string.
"""
from __future__ import annotations

import bisect
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from qwatt import config

SCHEMA_PATH = config.ROOT / "sql" / "tiger_schema.sql"
OVERLAP_S = 120.0           # re-sync window so late phase markers can still tag samples
BATCH = 2000
_TS_COLS = {"started_at", "ended_at", "t0", "t1"}
_schema_ok = False          # ensure_schema() once per process
_row_hash: dict = {}        # (table, key) -> hash of last upserted row (skip unchanged rows)
_tcols: dict = {}
_fail_until = 0.0           # circuit breaker so the dashboard is not slowed by a dead link


def configured() -> bool:
    return bool(config.TIGER_DATABASE_URL)


def connect(autocommit: bool = True, timeout: int = 8):
    """Return a psycopg connection, or None if not configured/reachable."""
    global _fail_until
    if not configured() or time.time() < _fail_until:
        return None
    try:
        import psycopg
        return psycopg.connect(config.TIGER_DATABASE_URL, connect_timeout=timeout, autocommit=autocommit)
    except Exception as e:  # noqa: BLE001
        _fail_until = time.time() + 30
        print(f"[tiger] unreachable: {type(e).__name__}")
        return None


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
def _blocks(sql_text: str):
    """Yield (name, flags, sql) for each '-- @@ name [flags]' block."""
    parts = re.split(r"^-- @@ (.*)$", sql_text, flags=re.M)
    for i in range(1, len(parts), 2):
        header = parts[i].split()
        yield header[0], header[1:], parts[i + 1].strip()


def ensure_schema(conn=None) -> dict[str, str]:
    """Create tables/hypertable/cagg/views. Returns {block: 'ok'|'failed: ...'|'skipped'}."""
    own = conn is None
    conn = conn or connect()
    if conn is None:
        return {}
    status: dict[str, str] = {}
    try:
        for name, flags, sql in _blocks(SCHEMA_PATH.read_text(encoding="utf-8")):
            fb = next((f.split("=", 1)[1] for f in flags if f.startswith("fallback-for=")), None)
            if fb and not status.get(fb, "").startswith("failed"):
                status[name] = "skipped"
                continue
            try:
                conn.execute(sql)
                status[name] = "ok"
            except Exception as e:  # noqa: BLE001
                status[name] = f"failed: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
                if "optional" not in flags and not fb:
                    raise RuntimeError(f"required schema block {name} failed: {status[name]}") from e
        return status
    finally:
        if own:
            conn.close()


# --------------------------------------------------------------------------
# Sync (SQLite -> Tiger)
# --------------------------------------------------------------------------
def _dt(x):
    return None if x is None else datetime.fromtimestamp(float(x), tz=timezone.utc)


def _upsert(cur, table: str, cols: list[str], rows: list[tuple], conflict: str, update_cols: list[str]):
    if not rows:
        return
    ph = ",".join(["%s"] * len(cols))
    upd = ",".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
    sql = (f"INSERT INTO {table} ({','.join(cols)}) VALUES ({ph}) "
           f"ON CONFLICT ({conflict}) DO " + (f"UPDATE SET {upd}" if upd else "NOTHING"))
    for i in range(0, len(rows), BATCH):
        cur.executemany(sql, rows[i:i + BATCH])


def _sqlite_cols(lite, table):
    return [r[1] for r in lite.execute(f"PRAGMA table_info({table})")]


def _tiger_cols(cur, table):
    if table in _tcols:
        return _tcols[table]
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name=%s", (table,))
    _tcols[table] = {r[0] for r in cur.fetchall()}
    return _tcols[table]


def sync_once(db_path=None, conn=None) -> dict | None:
    """One idempotent incremental sync. Returns counts, or None when Tiger is unavailable."""
    db_path = db_path or config.DB_PATH
    own = conn is None
    conn = conn or connect()
    if conn is None:
        return None
    lite = sqlite3.connect(str(db_path), timeout=30)
    lite.row_factory = sqlite3.Row
    counts = {"samples": 0, "runs": 0, "qaoa_runs": 0, "phase_markers": 0}
    try:
        global _schema_ok
        if not _schema_ok:
            ensure_schema(conn)
            _schema_ok = True
        cur = conn.cursor()

        # ---- small tables: full upsert each time ----
        for table, key, tscols in (("runs", "run_id", None), ("qaoa_runs", "run_id", None),
                                   ("phase_markers", "run_id, phase", None)):
            try:
                scols = _sqlite_cols(lite, table)
            except sqlite3.Error:
                continue
            cols = [c for c in scols if c in _tiger_cols(cur, table)]
            if not cols:
                continue
            rows = []
            for r in lite.execute(f"SELECT {','.join(cols)} FROM {table}"):
                rows.append(tuple(_dt(v) if c in _TS_COLS else v for c, v in zip(cols, r)))
            keys = {k.strip() for k in key.split(",")}
            kidx = [cols.index(k) for k in keys]
            changed, pend = [], {}
            for row in rows:
                hk = (table, tuple(row[i] for i in kidx))
                h = hash(repr(row))
                if _row_hash.get(hk) != h:
                    changed.append(row)
                    pend[hk] = h
            _upsert(cur, table, cols, changed, key, [c for c in cols if c not in keys])
            _row_hash.update(pend)
            counts[table] = len(changed)

        # ---- samples: incremental by watermark, tagged by phase markers ----
        cur.execute("SELECT extract(epoch FROM max(ts)) FROM power_samples")
        wm = cur.fetchone()[0]
        since = (float(wm) - OVERLAP_S) if wm is not None else 0.0

        markers = sorted((r["t0"], r["t1"], r["run_id"], r["phase"]) for r in lite.execute(
            "SELECT run_id, phase, t0, t1 FROM phase_markers WHERE t0 IS NOT NULL AND t1 IS NOT NULL"))
        t0s = [m[0] for m in markers]
        arm_of = {r["run_id"]: r["arm"] for r in lite.execute("SELECT run_id, arm FROM runs")}

        def tag(ts):
            i = bisect.bisect_right(t0s, ts) - 1
            if i >= 0 and ts <= markers[i][1]:
                _, _, rid, ph = markers[i]
                return rid, arm_of.get(rid), ph
            return None, None, None

        cols = ["ts", "run_id", "arm", "phase", "pkg_watts", "pp0_watts", "pp1_watts",
                "pkg_joules_cum", "sys_watts", "source"]
        upd = cols[1:]
        sql = (f"INSERT INTO power_samples ({','.join(cols)}) VALUES ({','.join(['%s'] * len(cols))}) "
               "ON CONFLICT (ts) DO UPDATE SET "
               "run_id=COALESCE(EXCLUDED.run_id, power_samples.run_id), "
               "arm=COALESCE(EXCLUDED.arm, power_samples.arm), "
               "phase=COALESCE(EXCLUDED.phase, power_samples.phase), "
               + ",".join(f"{c}=EXCLUDED.{c}" for c in upd[3:]))
        batch = []
        q = lite.execute("SELECT ts, pkg_watts, pp0_watts, pp1_watts, pkg_joules_cum, sys_watts, source "
                         "FROM power_samples WHERE ts > ? ORDER BY ts", (since,))
        for r in q:
            rid, arm, ph = tag(r["ts"])
            batch.append((_dt(r["ts"]), rid, arm, ph, r["pkg_watts"], r["pp0_watts"], r["pp1_watts"],
                          r["pkg_joules_cum"], r["sys_watts"], r["source"]))
            if len(batch) >= BATCH:
                cur.executemany(sql, batch)
                counts["samples"] += len(batch)
                batch = []
        if batch:
            cur.executemany(sql, batch)
            counts["samples"] += len(batch)

        try:
            refresh_derived(conn)
        except Exception as e:  # noqa: BLE001
            print(f"[tiger] refresh_derived failed: {type(e).__name__}: {str(e)[:100]}")
        try:  # refresh the continuous aggregate (no-op / error if it is a plain view)
            conn.execute("CALL refresh_continuous_aggregate('power_1s', NULL, NULL)")
        except Exception:  # noqa: BLE001
            pass
        return counts
    finally:
        lite.close()
        if own:
            conn.close()


_REFRESH_SQL = [
    # recent phases may have had samples still arriving: recompute them
    "DELETE FROM phase_energy_tbl WHERE t1 > now() - INTERVAL '3 minutes'",
    """INSERT INTO phase_energy_tbl (run_id, phase, t0, t1, dur_s, n_samples, gross_j_tw, gross_j_counter)
       SELECT run_id, phase, t0, t1, dur_s, n_samples, gross_j_tw, gross_j_counter FROM (
         SELECT m.run_id, m.phase, m.t0, m.t1, extract(epoch FROM (m.t1 - m.t0)) AS dur_s,
                extract(epoch FROM (max(s.ts) - min(s.ts))) AS span_s,
                integral(time_weight('Linear', s.ts, s.pkg_watts), 'second') AS tw_j,
                max(s.pkg_joules_cum) - min(s.pkg_joules_cum) AS ctr_j, count(*) AS n_samples
         FROM phase_markers m JOIN power_samples s ON s.ts >= m.t0 AND s.ts <= m.t1
         WHERE m.t1 IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM phase_energy_tbl e WHERE e.run_id = m.run_id AND e.phase = m.phase)
         GROUP BY m.run_id, m.phase, m.t0, m.t1) q,
       LATERAL (SELECT tw_j * CASE WHEN span_s > 0 THEN dur_s / span_s ELSE 1 END AS gross_j_tw,
                       ctr_j * CASE WHEN span_s > 0 THEN dur_s / span_s ELSE 1 END AS gross_j_counter) k
       ON CONFLICT (run_id, phase) DO NOTHING""",
    "DELETE FROM run_gaps_tbl WHERE run_id IN (SELECT run_id FROM runs WHERE ended_at > now() - INTERVAL '3 minutes')",
    """INSERT INTO run_gaps_tbl (run_id, max_gap_s)
       SELECT run_id, max(extract(epoch FROM (ts - prev_ts))) FROM (
         SELECT b.run_id, s.ts, lag(s.ts) OVER (PARTITION BY b.run_id ORDER BY s.ts) AS prev_ts
         FROM runs b JOIN power_samples s ON s.ts >= b.started_at AND s.ts <= b.ended_at
         WHERE b.started_at IS NOT NULL AND b.ended_at IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM run_gaps_tbl g WHERE g.run_id = b.run_id)) x
       GROUP BY run_id ON CONFLICT (run_id) DO NOTHING""",
]


def refresh_derived(conn) -> None:
    """Fill phase_energy_tbl / run_gaps_tbl for new (or just-finished) runs."""
    for sql in _REFRESH_SQL:
        conn.execute(sql)


# --------------------------------------------------------------------------
# Joules in SQL, and comparison with the Python joules in SQLite
# --------------------------------------------------------------------------
def compare_joules(db_path=None, conn=None) -> list[dict] | None:
    """Per run: SQL trapezoid / counter / toolkit joules vs Python joules (runs.joules_gross in SQLite)."""
    db_path = db_path or config.DB_PATH
    own = conn is None
    conn = conn or connect()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT run_id, arm, joules_trapezoid, joules_counter, n_samples FROM run_joules_sql")
        sql_rows = {r[0]: r for r in cur.fetchall()}
        toolkit = {}
        try:
            cur.execute("SELECT run_id, joules_toolkit FROM run_joules_toolkit")
            toolkit = dict(cur.fetchall())
        except Exception:  # noqa: BLE001
            pass
        lite = sqlite3.connect(str(db_path))
        try:
            py = {r[0]: r[1:] for r in lite.execute("SELECT run_id, joules_gross, joules_net FROM runs")}
        finally:
            lite.close()
        out = []
        for rid, (_, arm, trap, ctr, n) in sql_rows.items():
            pj = py.get(rid, (None, None))[0]
            out.append({
                "run_id": rid, "arm": arm, "n_samples": n,
                "sql_trapezoid_j": trap, "sql_counter_j": ctr, "sql_toolkit_j": toolkit.get(rid),
                "python_j": pj,
                "delta_pct": (100.0 * (trap - pj) / pj) if (pj and trap is not None) else None,
            })
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[tiger] compare_joules failed: {type(e).__name__}")
        return None
    finally:
        if own:
            conn.close()


# --------------------------------------------------------------------------
# Dashboard helpers: None means "not configured / unreachable; fall back to SQLite"
# --------------------------------------------------------------------------
def _rows_as_dicts(cur) -> list[dict]:
    names = [d.name for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def scoreboard() -> list[dict] | None:
    conn = connect(timeout=4)
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM scoreboard ORDER BY arm")
        return _rows_as_dicts(cur)
    except Exception as e:  # noqa: BLE001
        print(f"[tiger] scoreboard failed: {type(e).__name__}")
        return None
    finally:
        conn.close()


def power_1s(run_ids) -> list[dict] | None:
    conn = connect(timeout=4)
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT extract(epoch FROM bucket) AS t, run_id, arm, avg_watts, max_watts, n "
                    "FROM power_1s WHERE run_id = ANY(%s) ORDER BY bucket", (list(run_ids),))
        return _rows_as_dicts(cur)
    except Exception as e:  # noqa: BLE001
        print(f"[tiger] power_1s failed: {type(e).__name__}")
        return None
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Tiger in the decision loop: per-phase ledger, live energy price, integrity, perf, async sync
# --------------------------------------------------------------------------
import threading  # noqa: E402

_sync_lock = threading.Lock()
_sync_state = {"running": False, "last": None, "last_t": 0.0}


def sync_async(db_path=None) -> bool:
    """Fire one incremental sync in a background thread. Non-blocking; errors swallowed.
    Returns False if Tiger is not configured or a sync is already running."""
    if not configured() or not _sync_lock.acquire(blocking=False):
        return False

    def _run():
        try:
            _sync_state["running"] = True
            _sync_state["last"] = sync_once(db_path)
            _sync_state["last_t"] = time.time()
        except Exception:  # noqa: BLE001
            _sync_state["last"] = None
        finally:
            _sync_state["running"] = False
            _sync_lock.release()

    threading.Thread(target=_run, daemon=True, name="tiger-sync").start()
    return True


def phase_ledger(run_ids=None, limit: int = 200) -> list[dict] | None:
    """Per (run, phase): SQL joules (gross, net of idle, counter) vs Python net joules."""
    conn = connect(timeout=4)
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        sql = ("SELECT run_id, arm, bench_id, qid, extract(epoch FROM started_at) AS started, phase, dur_s, "
               "sql_gross_j, sql_counter_j, sql_net_j, python_net_j, "
               "100.0 * (sql_net_j - python_net_j) / NULLIF(python_net_j, 0) AS delta_pct FROM phase_ledger ")
        if run_ids:
            cur.execute(sql + "WHERE run_id = ANY(%s) ORDER BY started, phase", (list(run_ids),))
        else:
            cur.execute(sql + "ORDER BY started DESC, phase LIMIT %s", (limit,))
        return _rows_as_dicts(cur)
    except Exception as e:  # noqa: BLE001
        print(f"[tiger] phase_ledger failed: {type(e).__name__}")
        return None
    finally:
        conn.close()


_price_cache = {"t": 0.0, "val": None}


def live_price(max_age_s: float = 60.0) -> dict | None:
    """{eps_in, e0, r2, n, source:'tiger'} from the energy_price_live view (cached; 3 s timeout).
    None on any failure: callers fall back to results/calibration.json."""
    now = time.time()
    if _price_cache["val"] is not None and now - _price_cache["t"] < max_age_s:
        return _price_cache["val"]
    conn = connect(timeout=3)
    if conn is None:
        return None
    try:
        conn.execute("SET statement_timeout = 3000")
        cur = conn.cursor()
        cur.execute("SELECT eps_in, e0, r2, n FROM energy_price_live")
        eps, e0, r2, n = cur.fetchone()
        if eps is None or eps <= 0 or not n or n < 5:
            return None
        val = {"eps_in": float(eps), "e0": float(e0), "r2": float(r2) if r2 is not None else None,
               "n": int(n), "source": "tiger"}
        _price_cache.update(t=now, val=val)
        return val
    except Exception as e:  # noqa: BLE001
        print(f"[tiger] live_price failed: {type(e).__name__}")
        return None
    finally:
        conn.close()


def price_drift() -> list[dict] | None:
    conn = connect(timeout=4)
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT extract(epoch FROM bucket) AS t, eps_in, e0, r2, j_per_token_avg, n "
                    "FROM energy_price_drift WHERE n >= 5 ORDER BY bucket")
        return _rows_as_dicts(cur)
    except Exception as e:  # noqa: BLE001
        print(f"[tiger] price_drift failed: {type(e).__name__}")
        return None
    finally:
        conn.close()


def integrity_summary() -> dict | None:
    conn = connect(timeout=4)
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) AS n_runs, count(*) FILTER (WHERE flagged) AS n_flagged, "
                    "count(*) FILTER (WHERE flag_idle) AS n_idle, count(*) FILTER (WHERE flag_gap) AS n_gap, "
                    "count(*) FILTER (WHERE flag_energy) AS n_energy, "
                    "percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(sql_vs_python_pct)) AS median_abs_sql_vs_python_pct "
                    "FROM run_integrity")
        return _rows_as_dicts(cur)[0]
    except Exception as e:  # noqa: BLE001
        print(f"[tiger] integrity_summary failed: {type(e).__name__}")
        return None
    finally:
        conn.close()


def perf_stats() -> dict | None:
    """Compression ratio, row counts and client-side latencies (ms) of the dashboard queries."""
    conn = connect(timeout=4)
    if conn is None:
        return None
    out: dict = {}
    try:
        cur = conn.cursor()

        def timed(name, sql):
            t = time.perf_counter()
            cur.execute(sql)
            cur.fetchall()
            out[name] = round((time.perf_counter() - t) * 1000, 1)

        timed("scoreboard_ms", "SELECT * FROM scoreboard")
        timed("live_price_ms", "SELECT * FROM energy_price_live")
        timed("power_1s_ms", "SELECT * FROM power_1s ORDER BY bucket DESC LIMIT 300")
        cur.execute("SELECT before_compression_total_bytes, after_compression_total_bytes, "
                    "number_compressed_chunks, total_chunks FROM hypertable_compression_stats('power_samples')")
        b, a, nc, nt = cur.fetchone()
        out.update(compressed_chunks=nc, total_chunks=nt, bytes_before=b, bytes_after=a,
                   compression_ratio=(b / a) if a else None)
        cur.execute("SELECT count(*) FROM power_samples")
        out["power_samples_rows"] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM runs")
        out["runs_rows"] = cur.fetchone()[0]
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[tiger] perf_stats failed: {type(e).__name__}")
        return out or None
    finally:
        conn.close()


def compress_old_chunks(older_than: str = "1 hour") -> int:
    """Compress every chunk whose data is older than the interval (idempotent)."""
    conn = connect()
    if conn is None:
        return 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT compress_chunk(c, if_not_compressed => true) FROM show_chunks('power_samples', "
                    "older_than => now() - %s::interval) c", (older_than,))
        return len(cur.fetchall())
    finally:
        conn.close()

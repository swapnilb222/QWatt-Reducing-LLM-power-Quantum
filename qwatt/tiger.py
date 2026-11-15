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
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name=%s", (table,))
    return {r[0] for r in cur.fetchall()}


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
        ensure_schema(conn)
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
            _upsert(cur, table, cols, rows, key, [c for c in cols if c not in keys])
            counts[table] = len(rows)

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

        try:  # refresh the continuous aggregate (no-op / error if it is a plain view)
            conn.execute("CALL refresh_continuous_aggregate('power_1s', NULL, NULL)")
        except Exception:  # noqa: BLE001
            pass
        return counts
    finally:
        lite.close()
        if own:
            conn.close()


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

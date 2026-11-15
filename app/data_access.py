"""Read-only data access for the dashboard. Every function degrades to an empty result.

Sources (see CONTRACTS.md):
- SQLite `results/qwatt.sqlite` (or `results/dev_fixture.sqlite` when QWATT_DASH_FIXTURE=1)
- results/*.json files (in fixture mode these come from the fixture DB table `fixture_json`)
- qwatt.data.get_pool(qid) for sentence text (fixture mode: table `fixture_pools`)
- qwatt.tiger.scoreboard() when available (cloud, guarded by a timeout)
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE_DB = ROOT / "results" / "dev_fixture.sqlite"
RESULT_FILES = {
    "calibration": "calibration.json",
    "angles": "angles.json",
    "hyperparams": "hyperparams.json",
    "selection_eval": "selection_eval.json",
    "solver_quality": "solver_quality.json",
    "demo_questions": "demo_questions.json",
    "scaling": "scaling.json",
    "calibration_lc": "calibration_lc.json",
    "lc_demo_questions": "lc_demo_questions.json",
    "lc_selection": ("lc_selection.json", "lc_eval.json", "lc_selection_eval.json", "lc_solver_quality.json"),
}


def is_fixture() -> bool:
    return os.getenv("QWATT_DASH_FIXTURE", "0") == "1"


def db_path() -> Path:
    if is_fixture():
        return FIXTURE_DB
    try:
        from qwatt import config
        return Path(config.DB_PATH)
    except Exception:
        return ROOT / "results" / "qwatt.sqlite"


def results_dir() -> Path:
    try:
        from qwatt import config
        return Path(config.RESULTS_DIR)
    except Exception:
        return ROOT / "results"


# ---------------------------------------------------------------- SQLite
def _connect() -> sqlite3.Connection | None:
    p = db_path()
    if not p.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True, timeout=5)
        con.execute("PRAGMA busy_timeout=5000")
        return con
    except sqlite3.Error:
        return None


def query(sql: str, params: tuple | list = ()) -> pd.DataFrame:
    con = _connect()
    if con is None:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(sql, con, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def db_status() -> dict:
    p = db_path()
    out = {"path": str(p), "name": p.name, "exists": p.exists(), "tables": {}}
    if not p.exists():
        return out
    for t in ("runs", "qaoa_runs", "power_samples", "phase_markers"):
        df = query(f"SELECT COUNT(*) AS n FROM {t}")
        out["tables"][t] = int(df["n"].iloc[0]) if not df.empty else None
    return out


def runs_df() -> pd.DataFrame:
    df = query("SELECT * FROM runs ORDER BY started_at")
    if df.empty:
        return df
    for c in ("joules_net", "joules_gross", "joules_select", "joules_prefill", "joules_generate",
              "tokens_in", "tokens_out", "context_tokens", "ttft_s", "total_s", "em", "f1",
              "sf_recall", "referee_correct", "referee_score", "idle_watts", "started_at", "ended_at",
              "escalated"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def qaoa_df() -> pd.DataFrame:
    df = query("SELECT * FROM qaoa_runs")
    return df


def qaoa_diag(run_id: str) -> dict | None:
    df = query("SELECT diag_json FROM qaoa_runs WHERE run_id=?", (run_id,))
    if df.empty or not isinstance(df["diag_json"].iloc[0], str):
        return None
    try:
        return json.loads(df["diag_json"].iloc[0])
    except Exception:
        return None


def phase_markers(run_id: str) -> dict[str, tuple[float, float | None]]:
    df = query("SELECT phase, t0, t1 FROM phase_markers WHERE run_id=?", (run_id,))
    out = {}
    for _, r in df.iterrows():
        t1 = r["t1"] if r["t1"] is not None and not (isinstance(r["t1"], float) and math.isnan(r["t1"])) else None
        out[str(r["phase"])] = (float(r["t0"]), None if t1 is None else float(t1))
    return out


def power_window(t0: float, t1: float, pad: float = 0.0) -> pd.DataFrame:
    return query("SELECT ts, pkg_watts, pkg_joules_cum, sys_watts, source FROM power_samples "
                 "WHERE ts BETWEEN ? AND ? ORDER BY ts", (t0 - pad, t1 + pad))


def power_recent(seconds: float = 120.0) -> pd.DataFrame:
    df = query("SELECT MAX(ts) AS m FROM power_samples")
    if df.empty or df["m"].iloc[0] is None:
        return pd.DataFrame()
    m = float(df["m"].iloc[0])
    return power_window(m - seconds, m)


def heartbeat() -> dict | None:
    df = query("SELECT ts, pid, source FROM sampler_heartbeat WHERE id=1")
    if df.empty:
        return None
    return df.iloc[0].to_dict()


# ---------------------------------------------------------------- results JSON
def load_result(name: str) -> Any | None:
    """name is a key of RESULT_FILES. Fixture mode reads the fixture DB instead of results/."""
    if is_fixture():
        df = query("SELECT json FROM fixture_json WHERE name=?", (name,))
        if df.empty:
            return None
        try:
            return json.loads(df["json"].iloc[0])
        except Exception:
            return None
    fns = RESULT_FILES.get(name, name)
    for fn in (fns if isinstance(fns, tuple) else (fns,)):
        p = results_dir() / fn
        if not p.exists():
            continue
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


# ---------------------------------------------------------------- pools (sentence text)
def _pool_to_dict(pool) -> dict:
    sents = []
    for s in pool.sentences:
        sents.append({"idx": int(s.idx), "text": str(s.text), "title": str(s.title),
                      "tokens": int(s.tokens), "relevance": float(s.relevance), "is_gold": bool(s.is_gold)})
    return {"qid": pool.qid, "question": pool.question, "answer": pool.answer, "qtype": pool.qtype,
            "sentences": sents, "full_tokens": int(getattr(pool, "full_tokens", 0) or 0),
            "pool_recall": float(getattr(pool, "pool_recall", float("nan")))}


def get_pool_dict(qid: str) -> dict | None:
    if not qid:
        return None
    if is_fixture():
        df = query("SELECT json FROM fixture_pools WHERE qid=?", (qid,))
        if df.empty:
            return None
        return json.loads(df["json"].iloc[0])
    try:
        from qwatt.data import get_pool
        return _pool_to_dict(get_pool(qid))
    except Exception:
        return None


def get_pool_object(qid: str):
    """The real Pool for live runs (None in fixture mode or if qwatt.data is missing)."""
    if is_fixture():
        return None
    from qwatt.data import get_pool  # raises if missing: caller shows the error
    return get_pool(qid)


def get_lc_pool_object(qid: str):
    if is_fixture():
        return None
    from qwatt.longdoc import get_lc_pool
    return get_lc_pool(qid)


def get_lc_pool_dict(qid: str) -> dict | None:
    if not qid:
        return None
    if is_fixture():
        return get_pool_dict(qid)
    try:
        pool = get_lc_pool_object(qid)
        d = _pool_to_dict(pool)
        meta = getattr(pool, "meta", {}) or {}
        d["paragraph_of"] = meta.get("paragraph_of")
        d["doc_id"] = meta.get("doc_id")
        return d
    except Exception:
        return None


def demo_question_list() -> list[dict]:
    """[{qid, question}] for the live-mode picker."""
    out: list[dict] = []
    qids = load_result("demo_questions") or []
    if isinstance(qids, dict):
        qids = qids.get("qids") or qids.get("questions") or []
    for q in qids:
        qid = q if isinstance(q, str) else (q.get("qid") if isinstance(q, dict) else None)
        if not qid:
            continue
        text = q.get("question") if isinstance(q, dict) else None
        if not text and not is_fixture():
            try:  # cheap: raw dataset item, no embedding / pool build
                from qwatt.data import get_item
                text = (get_item(qid) or {}).get("question")
            except Exception:
                text = None
        if not text:
            pdict = get_pool_dict(qid)
            text = pdict["question"] if pdict else None
        if not text:
            r = query("SELECT question FROM runs WHERE qid=? LIMIT 1", (qid,))
            text = r["question"].iloc[0] if not r.empty else qid
        out.append({"qid": qid, "question": text})
    if not out:  # fall back to questions seen in the benchmark
        r = query("SELECT DISTINCT qid, question FROM runs WHERE mode='bench' LIMIT 30")
        out = [{"qid": a, "question": b} for a, b in zip(r.get("qid", []), r.get("question", []))]
    return out


# ---------------------------------------------------------------- Tiger Data
def tiger_scoreboard(timeout_s: float = 4.0) -> tuple[Any | None, str]:
    """(scoreboard or None, status message). Never blocks longer than timeout_s."""
    if is_fixture():
        return None, "disabled in fixture mode"
    try:
        from qwatt import config
        if not getattr(config, "TIGER_DATABASE_URL", ""):
            return None, "TIGER_DATABASE_URL not set"
        from qwatt import tiger  # noqa: F401
    except Exception as e:  # module missing
        return None, f"qwatt.tiger unavailable ({type(e).__name__})"
    if not hasattr(tiger, "scoreboard"):
        return None, "qwatt.tiger.scoreboard missing"
    ex = cf.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(tiger.scoreboard)
    try:
        res = fut.result(timeout=timeout_s)
        return (res, "ok") if res else (None, "no rows (unreachable or not synced yet)")
    except cf.TimeoutError:
        return None, f"timed out after {timeout_s:.0f}s (offline?)"
    except Exception as e:
        return None, f"error: {type(e).__name__}"
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def explain_selection(question: str, kept: list[str], dropped: list[str], timeout_s: float = 12.0) -> str | None:
    try:
        from qwatt.referee import explain_run
    except Exception:
        return None
    ex = cf.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(explain_run, question, kept, dropped)
    try:
        return fut.result(timeout=timeout_s)
    except Exception:
        return None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

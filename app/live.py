"""Live mode: run one question through Q-Watt, then Full context (sequential, never two LLM calls at once).

A background thread calls qwatt.pipeline.run_question(..., on_event=cb). Events go into a queue that the
dashboard drains from an st.fragment(run_every=0.5). The LiveRun object is a module-level singleton so it
survives reruns and page refreshes, and so a second browser tab cannot start a second LLM run.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

from app import data_access as DA


@dataclass
class Lane:
    arm: str
    status: str = "waiting"          # waiting | running | done | error
    phase: str | None = None
    t0: float | None = None
    t1: float | None = None
    markers: dict = field(default_factory=dict)   # phase -> [t0, t1|None]
    text: str = ""
    indices: list | None = None
    diag: dict | None = None
    row: dict | None = None
    error: str | None = None
    ref_status: str | None = None     # None | pending | done | failed (live Gemini referee)
    escalated: bool = False


@dataclass
class LiveRun:
    qid: str
    question: str
    arms: list
    lanes: dict
    q: queue.Queue = field(default_factory=queue.Queue)
    status: str = "starting"          # starting | running | done | error
    message: str = "Preparing…"
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    pool: dict | None = None
    thread: threading.Thread | None = None
    kind: str = "16"                   # '16' (original) | 'lc' (long-context)

    @property
    def active(self) -> bool:
        return self.status in ("starting", "running")


_LOCK = threading.Lock()
_CURRENT: LiveRun | None = None
_WARM = {"started": False, "done": False, "error": None}


def warm_up(n_qubits: int = 16) -> None:
    """Untimed one-off cost: heavy imports + Quantum Rings provider/backend construction.

    Without it the first live Q-Watt lane in a fresh server process pays seconds of cold start inside
    its measured 'select' phase, which bench runs (warm process) never paid. No circuit is run here.
    """
    if _WARM["done"]:
        return
    try:
        from qwatt import pipeline, selectors  # noqa: F401  (imports qiskit etc.)
        from qwatt import qaoa_qr
        qaoa_qr.get_backend(int(n_qubits))
        _WARM["done"] = True
    except Exception as e:  # noqa: BLE001
        _WARM["error"] = f"{type(e).__name__}: {e}"[:200]


_WARM_SIZES: set = set()
LIVE_ENERGY_BUDGET_J = 35.0   # prefill energy budget per question; Tiger's live J/token turns it into a token budget


def warm_size(pool, settings, kind: str) -> None:
    """One unmeasured selection at this document's qubit count: builds the per-size backend and pays the
    first-run cost before the measured lane, exactly as the benchmark's warm process does."""
    try:
        from qwatt import qaoa_qr
        qaoa_qr.get_backend(int(pool.n))
        if kind == "lc" and pool.n not in _WARM_SIZES:
            from qwatt.selectors import get_selector
            get_selector("qwatt_lc")(pool, settings.budget_tokens, settings)
            _WARM_SIZES.add(pool.n)
    except Exception:  # noqa: BLE001
        pass


def prewarm_async(n_qubits: int = 16) -> None:
    """Start warm_up once per server process in the background (called when Live mode is opened)."""
    with _LOCK:
        if _WARM["started"] or DA.is_fixture():
            return
        _WARM["started"] = True
    threading.Thread(target=warm_up, args=(n_qubits,), daemon=True, name="qwatt-warm").start()


def _referee_async(run: "LiveRun", arm: str, row: dict) -> None:
    """Grade a live answer with the Gemini referee in the background and persist the verdict."""
    def job():
        try:
            from qwatt import referee
            rid = row.get("run_id") or f"live-{arm}"
            res = referee.grade_items([{"id": rid, "question": row.get("question") or run.question,
                                        "gold": row.get("gold") or "", "answer": row.get("answer") or ""}])
            v = res.get(rid)
            if not v:
                run.q.put((arm, {"type": "referee", "ok": False}))
                return
            run.q.put((arm, {"type": "referee", "ok": True, "verdict": v}))
            try:
                from qwatt import store
                con = store.connect()
                store.update_run(con, rid, referee_correct=int(bool(v.get("correct"))),
                                 referee_score=v.get("score"), referee_reason=v.get("reason"),
                                 referee_model=v.get("model"))
                con.close()
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            run.q.put((arm, {"type": "referee", "ok": False}))

    threading.Thread(target=job, daemon=True, name=f"qwatt-referee-{arm}").start()


def current() -> LiveRun | None:
    return _CURRENT


def clear() -> None:
    global _CURRENT
    with _LOCK:
        if _CURRENT is None or not _CURRENT.active:
            _CURRENT = None


def start(qid: str, question: str, include_topk: bool = False, arms: list | None = None,
          kind: str = "16") -> LiveRun | None:
    """Start a live run unless one is already active. Returns the run."""
    global _CURRENT
    with _LOCK:
        if _CURRENT is not None and _CURRENT.active:
            return _CURRENT
        arms = list(arms) if arms else ["qwatt", "full"] + (["topk"] if include_topk else [])
        run = LiveRun(qid=qid, question=question, arms=arms, lanes={a: Lane(a) for a in arms}, kind=kind)
        run.thread = threading.Thread(target=_worker, args=(run,), daemon=True, name="qwatt-live")
        _CURRENT = run
        run.thread.start()
        return run


def _worker(run: LiveRun) -> None:
    put = run.q.put
    try:
        put(("_run", {"type": "status", "status": "running", "message": "Loading the question…"}))
        if DA.is_fixture():
            from app import dev_fixture as FX
            pool_dict = DA.get_pool_dict(run.qid)
            if pool_dict is None:
                raise RuntimeError(f"fixture pool {run.qid} not found")
            put(("_run", {"type": "pool", "pool": pool_dict}))
            FX.ensure_fixture_sampler()
            if run.kind == "lc":
                from app import lc_fixture as LCF
                runner = lambda arm, cb: LCF.simulate_lc_lane(pool_dict, arm, cb)  # noqa: E731
            else:
                runner = lambda arm, cb: FX.simulate_lane(pool_dict, arm, cb)  # noqa: E731
        else:
            from qwatt.config import load_settings
            from qwatt import pipeline
            pool = DA.get_lc_pool_object(run.qid) if run.kind == "lc" else DA.get_pool_object(run.qid)
            put(("_run", {"type": "pool", "pool": DA._pool_to_dict(pool)}))
            put(("_run", {"type": "status", "status": "running", "message": "Starting the power sampler…"}))
            try:
                from qwatt.power import ensure_sampler
                ok = ensure_sampler()
                if not ok:
                    put(("_run", {"type": "warn", "message": "Power sampler did not report a heartbeat."}))
            except Exception as e:  # noqa: BLE001
                put(("_run", {"type": "warn", "message": f"Power sampler unavailable: {type(e).__name__}"}))
            if run.kind == "lc":
                from qwatt import longdoc
                import copy
                settings = copy.deepcopy(longdoc.lc_settings())   # same shots/weights as the long-document benchmark
                # Closed loop: the excerpt's size comes from an ENERGY budget in joules, priced per token live from
                # Tiger Data (regression over recent runs); falls back to the calibration file if Tiger is unreachable.
                settings.extra = {**(settings.extra or {}), "energy_budget_j": LIVE_ENERGY_BUDGET_J}
            else:
                settings = load_settings()
            # The Quantum Rings backend is cached per qubit count and documents differ in size, so always make sure
            # the backend for THIS document exists before the measured run (instant when already built).
            put(("_run", {"type": "status", "status": "running",
                          "message": f"Preparing the {pool.n}-qubit Quantum Rings backend (not measured)…"}))
            warm_up(pool.n)
            warm_size(pool, settings, run.kind)
            # One idle baseline per race: the first lane measures it on a quiet machine; later lanes reuse it, so the
            # previous lane's after-run work (diagnostics, referee, browser redraw) can't inflate their baseline.
            _idle = {"w": None}

            def runner(arm, cb):
                row = pipeline.run_question(pool, arm, settings, mode="live", on_event=cb, idle_watts=_idle["w"])
                if _idle["w"] is None and isinstance(row, dict) and row.get("idle_watts"):
                    _idle["w"] = float(row["idle_watts"])
                return row
        for k, arm in enumerate(run.arms):
            if k:
                time.sleep(1.0)
            put((arm, {"type": "lane_start", "t": time.time()}))
            try:
                row = runner(arm, lambda ev, a=arm: put((a, ev)))
                put((arm, {"type": "lane_end", "t": time.time(), "row": row}))
                try:  # near-real-time: push this lane's samples/run to Tiger in the background
                    from qwatt import tiger
                    tiger.sync_async()
                except Exception:  # noqa: BLE001
                    pass
                if not DA.is_fixture() and isinstance(row, dict) and row.get("referee_correct") is None:
                    put((arm, {"type": "referee_start"}))
                    _referee_async(run, arm, row)
            except Exception as e:  # noqa: BLE001
                put((arm, {"type": "lane_error", "t": time.time(), "error": f"{type(e).__name__}: {e}"[:300],
                           "trace": traceback.format_exc()[-1500:]}))
        put(("_run", {"type": "status", "status": "done", "message": "Done"}))
    except Exception as e:  # noqa: BLE001
        put(("_run", {"type": "status", "status": "error", "message": f"{type(e).__name__}: {e}"[:400]}))


def drain(run: LiveRun) -> bool:
    """Apply queued events to the run state (main script thread). Returns True if anything changed."""
    changed = False
    while True:
        try:
            who, ev = run.q.get_nowait()
        except queue.Empty:
            break
        changed = True
        typ = ev.get("type")
        if who == "_run":
            if typ == "status":
                run.status = ev["status"]
                run.message = ev.get("message", "")
                if run.status in ("done", "error"):
                    run.finished = time.time()
                    if run.status == "error":
                        run.error = run.message
            elif typ == "pool":
                run.pool = ev["pool"]
            elif typ == "warn":
                run.message = ev["message"]
            continue
        lane = run.lanes.get(who)
        if lane is None:
            continue
        t = float(ev.get("t") or time.time())
        if typ == "lane_start":
            lane.status, lane.t0 = "running", t
            names = {"qwatt": "Q-Watt", "full": "Full context", "topk": "Top-k", "qwatt_safe": "Q-Watt",
                     "full_lc": "Full document", "qwatt_lc": "Q-Watt (no safety net)"}
            run.message = f"Running the {names.get(who, who)} lane…"
        elif typ == "phase":
            ph = ev.get("phase")
            for p, m in lane.markers.items():
                if m[1] is None:
                    m[1] = t
            if str(ph).startswith("escalate"):
                lane.escalated = True
            if ph == "done":
                lane.phase = "done"
            else:
                lane.phase = ph
                lane.markers[ph] = [t, None]
        elif typ == "selection":
            lane.indices = list(ev.get("indices") or [])
            lane.diag = ev.get("diag") or {}
        elif typ == "token":
            lane.text += str(ev.get("text", ""))
        elif typ == "result":
            lane.row = ev.get("row")
        elif typ == "lane_end":
            lane.status, lane.t1 = "done", t
            run.message = "Lane finished."
            lane.phase = "done"
            if ev.get("row") is not None:
                lane.row = ev["row"]
            for m in lane.markers.values():
                if m[1] is None:
                    m[1] = t
        elif typ == "escalate":
            lane.escalated = True
            run.message = "The excerpt was not enough: safety net escalates to the full document…"
        elif typ == "referee_start":
            lane.ref_status = "pending"
        elif typ == "referee":
            if ev.get("ok") and lane.row is not None:
                v = ev.get("verdict") or {}
                lane.row = {**lane.row, "referee_correct": int(bool(v.get("correct"))),
                            "referee_score": v.get("score"), "referee_reason": v.get("reason"),
                            "referee_model": v.get("model")}
                lane.ref_status = "done"
            else:
                lane.ref_status = "failed"
        elif typ == "lane_error":
            lane.status, lane.t1, lane.error = "error", t, ev.get("error")
    return changed

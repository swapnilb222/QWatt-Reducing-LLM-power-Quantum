"""Randomized A/B benchmark runner with the controls from plan.md section 7.

Controls implemented here: randomized arm order per question (seeded per qid), per-run nonce,
cooldown until package power is back inside the idle band (capped at settings.cooldown_max_s),
idle baseline measured before each question, config hash + bench_id on every row, model warm-up,
resumability (skip completed (qid, arm, bench_id)), progress log under logs/.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from qwatt import config, llm, pipeline, power, store

DEFAULT_ARMS = ("full", "topk", "qwatt")


class BenchLog:
    def __init__(self, bench_id: str, echo: bool = True):
        d = config.ROOT / "logs"
        d.mkdir(exist_ok=True)
        self.path = d / f"bench_{bench_id}.log"
        self.echo = echo

    def __call__(self, msg: str):
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)


def make_bench_id(settings, arms, fresh: bool = False, prefix: str = "bench") -> str:
    base = f"{settings.hash()}-{hashlib.sha1(','.join(sorted(arms)).encode()).hexdigest()[:4]}"
    if llm.is_fake():
        base += "-fake"
    if fresh:
        base += "-" + datetime.now().strftime("%m%d%H%M%S")
    return f"{prefix}-" + base


def cooldown(con, idle_ref: float | None, settings, log: Callable[[str], None] | None = None) -> dict:
    """Wait until recent (1.5 s) mean package power is within idle_band_w of idle_ref, capped."""
    t0 = time.time()
    cap = settings.cooldown_max_s
    cur = None
    if idle_ref is None:
        time.sleep(min(2.0, cap))
        return {"waited_s": time.time() - t0, "reached": None, "watts": None}
    while True:
        now = time.time()
        s = store.read_samples(con, now - 1.5, now)
        cur = store.mean_watts(s) if len(s) >= 3 else None
        if cur is not None and cur <= idle_ref + settings.idle_band_w:
            return {"waited_s": now - t0, "reached": True, "watts": cur}
        if now - t0 >= cap:
            if log:
                log(f"  cooldown cap hit ({cap:.0f}s): {cur if cur is None else round(cur, 2)} W vs idle {idle_ref:.2f} W")
            return {"waited_s": now - t0, "reached": False, "watts": cur}
        time.sleep(0.25)


def measure_idle_baseline(con, seconds: float = 3.0) -> float | None:
    return store.measure_idle(seconds, con=con)


def warmup(settings, log, quantum_warm: Callable[[], None] | None = None) -> None:
    """One throwaway generation so model load / first-call costs never land in a measured run."""
    try:
        r = llm.generate([{"role": "user", "content": f"[ref warmup-{random.random():.6f}] Say OK."}], settings)
        log(f"warmup done: ttft={r.ttft_s:.2f}s backend={r.backend} model={r.model_id}")
    except Exception as e:  # noqa: BLE001
        log(f"warmup failed: {e}")
        raise
    # Same for the quantum backend: provider/license handshake and first-run setup happen once per
    # process (as in a long-running gateway), so they are not charged to the first measured question.
    try:
        if quantum_warm is not None:
            quantum_warm()
        else:
            from qwatt import data
            from qwatt.selectors import get_selector
            dev = data.load_split("dev")[0]
            get_selector("qwatt")(data.get_pool(dev["_id"], settings), settings.budget_tokens, settings)
        log("quantum warmup done (one circuit on Quantum Rings)")
    except Exception as e:  # noqa: BLE001
        log(f"quantum warmup skipped: {e}")


def run_bench(arms=DEFAULT_ARMS, n_questions: int | None = None, settings=None, bench_id: str | None = None,
              fresh: bool = False, seed: int | None = None, echo: bool = True, db_path=None,
              questions: list[dict] | None = None, get_pool=None, do_warmup: bool = True,
              on_progress: Callable[[dict], None] | None = None, prefix: str = "bench",
              quantum_warm: Callable[[], None] | None = None) -> dict:
    """Run the benchmark. Returns {'bench_id', 'n_done', 'n_skipped', 'n_failed'}."""
    from qwatt import data
    settings = settings or config.load_settings()
    arms = tuple(arms)
    bench_id = bench_id or make_bench_id(settings, arms, fresh, prefix)
    log = BenchLog(bench_id, echo)
    seed = settings.seed if seed is None else seed
    get_pool = get_pool or (lambda qid: data.get_pool(qid, settings))
    items = questions if questions is not None else data.load_split("bench")
    if n_questions:
        items = items[:n_questions]

    con = store.connect(db_path)
    ok = power.ensure_sampler()
    hb_src = power.sampler_source(con)
    log(f"bench_id={bench_id} arms={arms} n_questions={len(items)} config_hash={settings.hash()} "
        f"llm={'FAKE' if llm.is_fake() else 'lmstudio'} sampler_ok={ok} energy_source={hb_src}")
    if hb_src == "modeled":
        log("WARNING: energy source is 'modeled' (PDH unavailable); results are estimates")
    if do_warmup:
        warmup(settings, log, quantum_warm)

    idle_ref = measure_idle_baseline(con, 4.0)
    log(f"initial idle baseline: {idle_ref if idle_ref is None else round(idle_ref, 2)} W")
    stats = {"bench_id": bench_id, "n_done": 0, "n_skipped": 0, "n_failed": 0, "total": len(items) * len(arms)}

    def progress(extra: dict | None = None):
        p = dict(stats, ts=time.time(), **(extra or {}))
        try:
            (config.RESULTS_DIR / "bench_progress.json").write_text(json.dumps(p))
        except Exception:  # noqa: BLE001
            pass
        if on_progress:
            on_progress(p)

    for qi, item in enumerate(items):
        qid = item.get("_id") or item.get("qid") or item.get("id")
        pending = [a for a in arms if not store.run_done(con, bench_id, qid, a)]
        stats["n_skipped"] += len(arms) - len(pending)
        if not pending:
            log(f"[{qi + 1}/{len(items)}] {qid}: all arms already done, skipping")
            continue
        order = list(pending)
        random.Random(f"{seed}:{qid}").shuffle(order)
        try:
            pool = get_pool(qid)
        except Exception as e:  # noqa: BLE001
            log(f"[{qi + 1}/{len(items)}] {qid}: pool failed: {e}")
            stats["n_failed"] += len(order)
            continue
        cd = cooldown(con, idle_ref, settings, log)
        idle_q = measure_idle_baseline(con, 2.0)
        if idle_q is not None:
            idle_ref = min(idle_ref, idle_q) if idle_ref is not None else idle_q
        log(f"[{qi + 1}/{len(items)}] {qid}: order={order} idle_baseline={idle_q if idle_q is None else round(idle_q, 2)} W "
            f"(cooldown {cd['waited_s']:.1f}s)")
        for arm in order:
            cd = cooldown(con, idle_ref, settings, log)
            try:
                row = pipeline.run_question(pool, arm, settings, mode="bench", bench_id=bench_id,
                                            idle_watts=idle_q, db_path=db_path)
                stats["n_done"] += 1
                log(f"  {arm:6s} tok_in={row['tokens_in']} ttft={row['ttft_s']:.2f}s total={row['total_s']:.1f}s "
                    f"J_gross={_f(row['joules_gross'])} J_net={_f(row['joules_net'])} em={row['em']:.0f} "
                    f"f1={row['f1']:.2f} sf={row['sf_recall']:.2f} esc={row.get('escalated') or 0} ans={row['answer'][:50]!r} cool={cd['waited_s']:.1f}s")
            except Exception as e:  # noqa: BLE001
                stats["n_failed"] += 1
                log(f"  {arm}: FAILED {type(e).__name__}: {str(e)[:300]}")
                if "context length" in str(e).lower():
                    log("ABORT: LM Studio context length is too small for the long documents. "
                        "Reload the model with context 8192 and resume with --bench-id " + bench_id)
                    progress({"qid": qid, "arm": arm})
                    con.close()
                    raise
            progress({"qid": qid, "arm": arm})
    progress()
    log(f"bench finished: {stats}")
    con.close()
    return stats


def _f(x):
    return "None" if x is None else f"{x:.1f}"

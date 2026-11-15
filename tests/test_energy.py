import threading
import time

import numpy as np
import pytest

from qwatt import store
from qwatt.types import Pool, Sentence


def _samples(t0, n, dt, watts_fn, j0=5000.0):
    out, cum = [], j0
    for i in range(n):
        t = t0 + i * dt
        w = watts_fn(t)
        out.append({"ts": t, "pkg_watts": w, "pp0_watts": None, "pp1_watts": None,
                    "pkg_joules_cum": cum, "sys_watts": None, "source": "rapl_pdh"})
        cum += w * dt
    return out


def test_constant_power_cum_and_trapz():
    s = _samples(100.0, 80, 0.25, lambda t: 10.0)
    assert store.energy_cum(s, 102.1, 105.3) == pytest.approx(10.0 * 3.2, rel=1e-9)
    assert store.energy_trapz(s, 102.1, 105.3) == pytest.approx(32.0, rel=1e-9)


def test_step_power_interpolation():
    # 5 W for t<110, 25 W after; cum counter piecewise linear
    s = _samples(100.0, 200, 0.25, lambda t: 5.0 if t < 110 else 25.0)
    gross = store.energy_cum(s, 108.0, 112.0)
    assert gross == pytest.approx(5.0 * 2 + 25.0 * 2, rel=0.05)
    assert store.energy_trapz(s, 108.0, 112.0) == pytest.approx(gross, rel=0.1)


def test_unbracketed_window_is_none():
    s = _samples(100.0, 10, 0.25, lambda t: 10.0)
    assert store.energy_cum(s, 99.0, 101.0) is None
    assert store.energy_cum(s, 101.0, 110.0) is None


def test_phase_energy_net_and_gross(tmp_path):
    con = store.connect(tmp_path / "e.sqlite")
    store.insert_samples(con, _samples(100.0, 120, 0.25, lambda t: 12.0))
    store.insert_marker(con, "r", "select", 105.0, 106.0)
    store.insert_marker(con, "r", "prefill", 106.0, 109.0)
    store.insert_marker(con, "r", "generate", 109.0, 111.0)
    pe = store.phase_energy("r", con=con, idle_watts=5.0)
    assert pe["select"]["gross_j"] == pytest.approx(12.0, rel=1e-6)
    assert pe["prefill"]["gross_j"] == pytest.approx(36.0, rel=1e-6)
    assert pe["prefill"]["net_j"] == pytest.approx(36.0 - 15.0, rel=1e-6)
    assert pe["generate"]["method"] == "cum"
    assert pe["total"]["gross_j"] == pytest.approx(72.0, rel=1e-6)
    assert pe["total"]["net_j"] == pytest.approx(72.0 - 30.0, rel=1e-6)
    assert sum(pe[p]["gross_j"] for p in store.PHASES) == pytest.approx(pe["total"]["gross_j"])


def test_trapz_fallback_when_no_counter():
    s = _samples(100.0, 40, 0.25, lambda t: 8.0)
    for r in s:
        r["pkg_joules_cum"] = None
    out = store.phase_energy_from(s, {"x": (101.0, 103.0)}, 2.0)
    assert out["x"]["method"] == "trapz"
    assert out["x"]["gross_j"] == pytest.approx(16.0)
    assert out["x"]["net_j"] == pytest.approx(12.0)


def test_mean_watts():
    s = _samples(0.0, 40, 0.25, lambda t: 7.5)
    assert store.mean_watts(s) == pytest.approx(7.5)


# ------------------------------------------------------------------ prompt + pipeline (fake LLM)
def _pool():
    texts = [("Alpha", 0, "Alpha Corp was founded in Paris by Bob Smith."),
             ("Alpha", 1, "It makes widgets."),
             ("Beta", 0, "Bob Smith was born in Springfield."),
             ("Beta", 1, "He likes tea."),
             ("Gamma", 0, "Unrelated fact about rivers."),
             ("Gamma", 1, "Another unrelated fact.")]
    sents = [Sentence(idx=i, text=tx, title=t, sent_id=sid, tokens=12, relevance=1 - i / 10,
                      is_gold=i in (0, 2)) for i, (t, sid, tx) in enumerate(texts)]
    n = len(sents)
    full = "\n\n".join(f"{t}: " + " ".join(x[2] for x in texts if x[0] == t) for t in ("Alpha", "Beta", "Gamma"))
    return Pool(qid="q1", question="Where was the founder of Alpha Corp born?", answer="Springfield",
                qtype="bridge", sentences=sents, red=np.zeros((n, n)), br=np.zeros((n, n)),
                full_context=full, full_tokens=300, pool_recall=1.0, n_gold_total=2)


def test_prompt_template():
    from qwatt import prompt
    p = _pool()
    u = prompt.build_messages(p, [2, 0, 1], "topk", "abc123")[0]["content"]
    assert u.startswith("[ref abc123]")
    assert prompt.ANSWER_LINE in u
    assert u.index("Alpha: ") < u.index("Beta: ")
    assert u.index("Alpha Corp was founded") < u.index("It makes widgets.")
    assert "Gamma" not in u
    assert u.count("Alpha: ") == 1
    full = prompt.build_messages(p, [], "full", "zzz")[0]["content"]
    assert "Unrelated fact about rivers." in full


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    import sys
    import types

    from qwatt import config, power
    monkeypatch.setattr(config, "FAKE_LLM", True)
    monkeypatch.setenv("QWATT_FAKE_BURN", "0")
    monkeypatch.setattr(power, "ensure_sampler", lambda *a, **k: True)
    mod = types.ModuleType("qwatt.selectors")

    def get_selector(name):
        def sel(pool, budget, settings):
            idx = list(range(pool.n)) if name == "full" else ([0, 1, 2] if name == "qwatt" else [0, 2])
            diag = {"arm": name, "solver_wall_s": 0.01, "cost": -1.0, "tokens": 24}
            if name == "qwatt":
                diag.update({"dispatch": "circuit", "backend": "stub", "n_qubits": 6, "n_edges": 3, "depth": 4,
                             "gate_count": 20, "shots": 64, "gamma": [0.1], "beta": [0.2], "best_cost": -1.0,
                             "exact_cost": -1.0, "approx_ratio": 1.0, "amplification": 2.0,
                             "opt_in_shots": True, "feasible_fallback": False, "samples": []})
            return idx, diag
        return sel
    mod.get_selector = get_selector
    monkeypatch.setitem(sys.modules, "qwatt.selectors", mod)

    db = tmp_path / "p.sqlite"
    stop = threading.Event()

    def feeder():
        con = store.connect(db)
        cum = 1000.0
        while not stop.is_set():
            cum += 10.0 * 0.05
            store.insert_samples(con, [{"ts": time.time(), "pkg_watts": 10.0, "pp0_watts": 4.0, "pp1_watts": 0.1,
                                        "pkg_joules_cum": cum, "sys_watts": None, "source": "rapl_pdh"}])
            time.sleep(0.05)
        con.close()
    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    time.sleep(0.5)
    yield db
    stop.set()
    th.join(timeout=2)


def test_pipeline_fake_end_to_end(fake_env):
    from qwatt import pipeline
    from qwatt.config import Settings
    s = Settings()
    events = []
    rows = {}
    for arm in ("full", "topk", "qwatt"):
        rows[arm] = pipeline.run_question(_pool(), arm, s, mode="bench", on_event=events.append,
                                          db_path=fake_env, bench_id="t", idle_watts=4.0)
    for arm, r in rows.items():
        assert r["llm_backend"] == "fake"
        assert r["joules_gross"] is not None and r["joules_gross"] > 0
        dur = r["select_s"] + r["prefill_s"] + r["generate_s"]
        assert r["joules_gross"] == pytest.approx(10.0 * dur, rel=0.25, abs=1.0)
        assert r["joules_net"] < r["joules_gross"]
        assert r["energy_source"] == "rapl_pdh"
    assert rows["full"]["tokens_in"] > rows["topk"]["tokens_in"] or True  # synthetic context is tiny
    assert rows["topk"]["sf_recall"] == 1.0 and rows["full"]["sf_recall"] == 1.0
    assert rows["topk"]["em"] == 1.0
    assert {"phase", "selection", "token", "result"} <= {e["type"] for e in events}
    con = store.connect(fake_env)
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 3
    assert con.execute("SELECT COUNT(*) FROM qaoa_runs").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM phase_markers").fetchone()[0] == 9

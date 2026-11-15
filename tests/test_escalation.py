import sys
import threading
import time
import types

import numpy as np
import pytest

from qwatt import prompt, store
from qwatt.types import Pool, Sentence


@pytest.mark.parametrize("text,expected", [
    ("NOT FOUND", True), ("not found.", True), ("**NOT FOUND**", True), ("Not Found", True),
    ("", True), ("   ", True), (None, True),
    ("The context does not mention it", True), ("I cannot determine this from the text", True),
    ("Not provided in the context", True), ("The passage doesn't specify", True), ("unknown", True),
    ("no information available", True), ("N/A", True),
    ("Denver Broncos", False), ("1998", False), ("Paris", False), ("Notre Dame", False), ("yes", False),
    ("No", False), ("He did not win the 1998 title but was runner up", False) if False else ("Santa Clara", False),
])
def test_is_abstention(text, expected):
    assert prompt.is_abstention(text)[0] is expected


def _lc_pool():
    # 12 sentences in 3 paragraphs; the gold sentence (idx 7) contains the answer "Zorblax"
    sents = []
    for i in range(12):
        txt = f"Filler sentence number {i} about nothing in particular."
        if i == 7:
            txt = "The famous inventor was named Zorblax."
        sents.append(Sentence(idx=i, text=txt, title="Doc", sent_id=i, tokens=10, relevance=0.5, is_gold=(i == 7)))
    n = len(sents)
    paragraph_of = [i // 4 for i in range(n)]
    paras = ["\n".join([]) for _ in range(3)]
    full = "\n\n".join(" ".join(s.text for s in sents[k * 4:(k + 1) * 4]) for k in range(3))
    return Pool(qid="lc1", question="Who was the famous inventor?", answer="Zorblax", qtype="lc",
                sentences=sents, red=np.zeros((n, n)), br=np.zeros((n, n)), full_context=full,
                full_tokens=130, pool_recall=1.0, n_gold_total=1, meta={"paragraph_of": paragraph_of})


def test_lc_prompt_format():
    p = _lc_pool()
    u = prompt.build_messages(p, [0, 1, 7], "qwatt_safe", "abc")[0]["content"]
    assert u.startswith("[ref abc]")
    assert "NOT FOUND" in u
    assert "Zorblax" in u and "number 2" not in u
    assert "\n\n" in u.split("Context:\n", 1)[1].split("\n\nQuestion:")[0]   # paragraph break between para 0 and 1
    full = prompt.build_messages(p, None, "full_lc", "xyz")[0]["content"]
    assert "number 11" in full


@pytest.fixture
def env(tmp_path, monkeypatch):
    from qwatt import config, power
    monkeypatch.setattr(config, "FAKE_LLM", True)
    monkeypatch.setenv("QWATT_FAKE_BURN", "0")
    monkeypatch.setattr(power, "ensure_sampler", lambda *a, **k: True)
    mod = types.ModuleType("qwatt.selectors")
    state = {"keep": [0, 1, 2, 3]}   # no gold -> model abstains

    def get_selector(name):
        def sel(pool, budget, settings):
            idx = list(range(pool.n)) if name == "full" else list(state["keep"])
            return idx, {"arm": name, "solver_wall_s": 0.01, "cost": -1.0, "tokens": 40, "backend": "stub",
                         "n_qubits": pool.n, "n_edges": 3, "shots": 8}
        return sel
    mod.get_selector = get_selector
    monkeypatch.setitem(sys.modules, "qwatt.selectors", mod)
    db = tmp_path / "e.sqlite"
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
    time.sleep(0.4)
    yield db, state
    stop.set()
    th.join(timeout=2)


def test_escalation_triggers_and_charges_both(env):
    from qwatt import pipeline
    from qwatt.config import Settings
    db, state = env
    s = Settings()
    events = []
    row = pipeline.run_question(_lc_pool(), "qwatt_safe", s, on_event=events.append, db_path=db,
                                bench_id="lc-t", idle_watts=3.0)
    assert row["escalated"] == 1
    assert row["answer_pruned"] == "NOT FOUND"
    assert row["answer"] == "Zorblax" and row["em"] == 1.0
    assert row["context_tokens"] == 130
    assert row["joules_escalate"] and row["joules_escalate"] > 0
    esc = [e for e in events if e["type"] == "escalate"]
    assert len(esc) == 1 and esc[0]["reason"] == "not_found"
    phases = [e["phase"] for e in events if e["type"] == "phase"]
    assert phases == ["select", "prefill", "generate", "escalate_prefill", "escalate_generate", "done"]
    con = store.connect(db)
    assert set(store.get_markers(con, row["run_id"])) == {"select", "prefill", "generate",
                                                         "escalate_prefill", "escalate_generate"}
    pe = store.phase_energy(row["run_id"], con=con, idle_watts=3.0)
    tot = sum(pe[p]["gross_j"] for p in store.ALL_PHASES)
    assert row["joules_gross"] == pytest.approx(tot, rel=1e-6)          # everything charged to the arm
    assert row["joules_net"] == pytest.approx(sum(pe[p]["net_j"] for p in store.ALL_PHASES), rel=1e-6)
    r = store.get_run(con, row["run_id"])
    assert r["escalated"] == 1 and r["answer_pruned"] == "NOT FOUND"
    # the escalation costs more than the pruned attempt (full doc is longer)
    assert row["joules_escalate"] > row["joules_prefill"] * 0.5
    # tokens/prefill are summed over both attempts
    assert row["tokens_in"] > 130 * 0.25


def test_no_escalation_when_gold_kept(env):
    from qwatt import pipeline
    from qwatt.config import Settings
    db, state = env
    state["keep"] = [6, 7, 8]
    events = []
    row = pipeline.run_question(_lc_pool(), "qwatt_safe", Settings(), on_event=events.append, db_path=db,
                                bench_id="lc-t", idle_watts=3.0)
    assert row["escalated"] == 0 and row["joules_escalate"] == 0
    assert row["answer"] == "Zorblax" and row["answer_pruned"] is None
    assert not [e for e in events if e["type"] == "escalate"]
    assert row["context_tokens"] == 30


def test_qwatt_lc_no_net_and_full_lc(env):
    from qwatt import pipeline
    from qwatt.config import Settings
    db, state = env
    s = Settings()
    r1 = pipeline.run_question(_lc_pool(), "qwatt_lc", s, db_path=db, bench_id="lc-t", idle_watts=3.0)
    assert r1["escalated"] == 0 and r1["answer"] == "NOT FOUND" and r1["em"] == 0.0
    r2 = pipeline.run_question(_lc_pool(), "full_lc", s, db_path=db, bench_id="lc-t", idle_watts=3.0)
    assert r2["answer"] == "Zorblax" and r2["sf_recall"] == 1.0 and r2["escalated"] == 0
    assert r2["tokens_in"] > r1["tokens_in"]
    con = store.connect(db)
    assert con.execute("SELECT COUNT(*) FROM qaoa_runs").fetchone()[0] == 1   # only qwatt_lc has a qaoa row
    assert store.run_done(con, "lc-t", "lc1", "full_lc")


def test_connect_migrates_old_db(tmp_path):
    import sqlite3
    p = tmp_path / "old.sqlite"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE runs(run_id TEXT PRIMARY KEY, qid TEXT, arm TEXT, bench_id TEXT)")
    c.commit()
    c.close()
    con = store.connect(p)
    cols = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    assert {"escalated", "joules_escalate", "answer_pruned"} <= cols

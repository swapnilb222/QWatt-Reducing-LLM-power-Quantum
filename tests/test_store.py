import pytest

from qwatt import store


def _samples(t0, n, dt, watts, j0=1000.0):
    out, cum = [], j0
    for i in range(n):
        out.append({"ts": t0 + i * dt, "pkg_watts": watts, "pp0_watts": 1.0, "pp1_watts": 0.0,
                    "pkg_joules_cum": cum, "sys_watts": None, "source": "rapl_pdh"})
        cum += watts * dt
    return out


def test_schema_and_wal(tmp_path):
    con = store.connect(tmp_path / "t.sqlite")
    assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"power_samples", "sampler_heartbeat", "phase_markers", "runs", "qaoa_runs"} <= names
    cols = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    assert {"joules_gross", "joules_net", "bench_id", "referee_model", "energy_source"} <= cols


def test_insert_and_read(tmp_path):
    con = store.connect(tmp_path / "t.sqlite")
    store.insert_samples(con, _samples(100.0, 10, 0.25, 8.0))
    got = store.read_samples(con, 100.5, 101.5)
    assert [round(s["ts"], 2) for s in got] == [100.5, 100.75, 101.0, 101.25, 101.5]
    store.set_heartbeat(con, 5.0, 123, "rapl_pdh")
    store.set_heartbeat(con, 6.0, 123, "rapl_pdh")
    assert store.get_heartbeat(con)["ts"] == 6.0
    store.insert_run(con, {"run_id": "r1", "qid": "q", "arm": "full", "bench_id": "b"})
    assert store.run_done(con, "b", "q", "full") is False   # not finished (no ended_at/joules)
    store.update_run(con, "r1", ended_at=1.0, joules_gross=3.0)
    assert store.run_done(con, "b", "q", "full") is True
    store.insert_qaoa_run(con, {"run_id": "r1", "qid": "q", "gamma_json": [0.1], "diag_json": {"a": 1}})
    assert con.execute("SELECT gamma_json FROM qaoa_runs").fetchone()[0] == "[0.1]"


def test_markers_roundtrip(tmp_path):
    con = store.connect(tmp_path / "t.sqlite")
    store.insert_marker(con, "r", "select", 1.0, 2.0)
    store.insert_marker(con, "r", "prefill", 2.0, 3.0)
    assert store.get_markers(con, "r") == {"select": (1.0, 2.0), "prefill": (2.0, 3.0)}

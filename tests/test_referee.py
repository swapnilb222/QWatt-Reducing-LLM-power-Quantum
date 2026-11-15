import json
import sqlite3

from qwatt import referee


def test_em_f1_basic():
    assert referee.exact_match("The Eiffel Tower!", "eiffel tower") == 1.0
    assert referee.exact_match("Paris", "London") == 0.0
    assert referee.f1("new york city", "new york") > 0.79
    assert abs(referee.f1("new york city", "new york") - 0.8) < 1e-9
    assert referee.f1("", "x") == 0.0


def test_yes_no_handling():
    assert referee.f1("yes", "no") == 0.0
    assert referee.f1("yes it is", "yes") == 0.0  # normalized pred not exactly yes -> token overlap path
    assert referee.f1("Yes", "yes") == 1.0
    assert referee.f1("noanswer", "yes") == 0.0


def test_cache_key_stable():
    a = referee.cache_key("q", "g", "a")
    assert a == referee.cache_key("q", "g", "a")
    assert a != referee.cache_key("q", "g", "b")
    assert len(a) == 64


class FakeModels:
    def __init__(self, parent):
        self.parent = parent

    def list(self):
        from types import SimpleNamespace as N
        g = ["generateContent"]
        return [N(name="models/gemini-2.5-flash", supported_actions=g),
                N(name="models/gemini-3.8-flash", supported_actions=g),
                N(name="models/gemini-3.9-flash-image", supported_actions=g),
                N(name="models/gemini-4-argon", supported_actions=g),
                N(name="models/gemini-3.5-flash-lite", supported_actions=g),
                N(name="models/gemini-embedding-2", supported_actions=["embedContent"])]

    def generate_content(self, model, contents, config):
        self.parent.calls += 1
        items = json.loads(contents.split("ITEMS:\n", 1)[1])
        out = [{"id": it["id"], "correct": it["gold"] == it["model_answer"],
                "score": 1 if it["gold"] == it["model_answer"] else 0, "reason": "ok"} for it in items]

        class R:
            text = json.dumps(out)
        return R()


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.models = FakeModels(self)


def test_batching_and_cache(tmp_path):
    items = [{"id": str(i), "question": "q", "gold": "g", "answer": "g" if i % 2 else "x%d" % i}
             for i in range(25)]
    c = FakeClient()
    cp = tmp_path / "cache.json"
    res = referee.grade_items(items, client=c, model="fake", cache_path=cp)
    assert c.calls == 3  # 10 + 10 + 5
    assert len(res) == 25 and res["1"]["correct"] and not res["0"]["correct"]
    c2 = FakeClient()
    res2 = referee.grade_items(items, client=c2, model="fake", cache_path=cp)
    assert c2.calls == 0 and res2 == res


def test_grade_pending(tmp_path):
    db = tmp_path / "t.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, question TEXT, gold TEXT, answer TEXT, "
                "referee_correct INTEGER, referee_score REAL, referee_reason TEXT, referee_model TEXT)")
    con.execute("INSERT INTO runs (run_id, question, gold, answer) VALUES ('r1','q1','g','g')")
    con.execute("INSERT INTO runs (run_id, question, gold, answer) VALUES ('r2','q2','g','zzz')")
    con.commit()
    con.close()
    referee.CACHE_PATH = tmp_path / "rc.json"
    n = referee.grade_pending(db, client=FakeClient())
    assert n == 2
    con = sqlite3.connect(db)
    got = dict((r[0], r[1]) for r in con.execute("SELECT run_id, referee_correct FROM runs"))
    assert got == {"r1": 1, "r2": 0}


def test_pick_model_newest_flash(monkeypatch):
    monkeypatch.setattr(referee.config, "GEMINI_MODEL", "")
    assert referee.pick_model(FakeClient()) == "gemini-3.8-flash"
    assert referee.pick_models(FakeClient())[-1] == "gemini-3.5-flash-lite"

"""Referee: HotpotQA EM/F1 (official normalization) + offline Gemini grading.

Never in the hot path. Gemini grades each stored answer against the GOLD answer,
batched (~10 per request), cached by sha256(question, gold, answer), and capped
at MAX_REQUESTS requests per process/session.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import string
import time
from collections import Counter
from pathlib import Path

from qwatt import config

# --------------------------------------------------------------------------
# Official HotpotQA evaluation (hotpot_evaluate_v1.py)
# --------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    s = "" if s is None else str(s)

    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_pr(pred: str, gold: str) -> tuple[float, float, float]:
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    zero = (0.0, 0.0, 0.0)
    special = {"yes", "no", "noanswer"}
    if np_ in special and np_ != ng:
        return zero
    if ng in special and np_ != ng:
        return zero
    pt, gt = np_.split(), ng.split()
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return zero
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return (2 * precision * recall) / (precision + recall), precision, recall


def f1(pred: str, gold: str) -> float:
    return f1_pr(pred, gold)[0]


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


# --------------------------------------------------------------------------
# Cache / budget
# --------------------------------------------------------------------------
CACHE_PATH = config.CACHE_DIR / "referee_cache.json"
MAX_REQUESTS = 50          # per session (process)
BATCH_SIZE = 10
_requests_used = 0


def requests_used() -> int:
    return _requests_used


def cache_key(question: str, gold: str, answer: str) -> str:
    blob = json.dumps([question, gold, answer], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_cache(path: Path = None) -> dict:
    path = path or CACHE_PATH
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict, path: Path = None) -> None:
    path = path or CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------
# Gemini client / model choice
# --------------------------------------------------------------------------
_FLASH_RE = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash$")
_FLASH_LITE_RE = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash-lite$")


def _client():
    from google import genai
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def pick_models(client=None) -> list[str]:
    """config.GEMINI_MODEL if set, else Flash-tier text models supporting generateContent, newest first
    (the rest are fallbacks when the newest is overloaded or unavailable)."""
    if config.GEMINI_MODEL:
        return [config.GEMINI_MODEL]
    client = client or _client()
    flash, lite = [], []
    for m in client.models.list():
        name = m.name.split("/", 1)[-1]
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" not in actions:
            continue
        mm = _FLASH_RE.match(name)
        if mm:
            flash.append((float(mm.group(1)), name))
        mm = _FLASH_LITE_RE.match(name)
        if mm:
            lite.append((float(mm.group(1)), name))
    ranked = [n for _, n in sorted(flash, reverse=True)] + [n for _, n in sorted(lite, reverse=True)]
    if not ranked:
        raise RuntimeError("no Flash-tier generateContent model found")
    return ranked


def pick_model(client=None) -> str:
    return pick_models(client)[0]


_GRADE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "STRING"},
            "correct": {"type": "BOOLEAN"},
            "score": {"type": "NUMBER"},
            "reason": {"type": "STRING"},
        },
        "required": ["id", "correct", "score", "reason"],
    },
}

_GRADE_PROMPT = (
    "You are a strict but fair grader for question answering. For each item compare the MODEL ANSWER "
    "to the GOLD answer for the QUESTION. Grade against the gold answer only; do not use outside "
    "knowledge to overrule it. score = 1 if the model answer is semantically equivalent to gold "
    "(extra harmless words, different formatting or an alias are fine); 0.5 if partially correct "
    "(e.g. one of two required parts, or correct but contradicted by extra claims); 0 otherwise. "
    "correct = true only when score is 1. reason = one short sentence. Return a JSON array with one "
    "object per item, echoing each item's id.\n\nITEMS:\n"
)


def _call_with_retry(client, models, prompt: str, schema, tries_per_model: int = 2, max_models: int = 4):
    """One logical request. Tries each model (newest first) with backoff on transient errors.
    Every HTTP attempt counts against MAX_REQUESTS. Returns (parsed_json, model_used)."""
    global _requests_used
    from google.genai import types
    if isinstance(models, str):
        models = [models]
    last = None
    for model in models[:max_models]:
        delay = 2.0
        for attempt in range(tries_per_model):
            if _requests_used >= MAX_REQUESTS:
                raise RuntimeError(f"Gemini request budget exhausted ({MAX_REQUESTS})")
            _requests_used += 1
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.0,
                        http_options=types.HttpOptions(timeout=25000),
                    ),
                )
                return json.loads(resp.text), model
            except Exception as e:  # noqa: BLE001
                last = e
                msg = str(e)
                if any(t in msg for t in ("404", "NOT_FOUND", "504", "DEADLINE")):
                    break  # model unusable / hanging: go to next model
                transient = any(t in msg for t in ("429", "500", "503", "504", "RESOURCE_EXHAUSTED",
                                                   "UNAVAILABLE", "DEADLINE")) or isinstance(e, json.JSONDecodeError)
                if not transient:
                    raise RuntimeError(f"Gemini call failed: {type(e).__name__}: {msg[:200]}") from e
                if attempt < tries_per_model - 1:
                    time.sleep(delay)
                    delay *= 2
    global LAST_STATUS
    LAST_STATUS = ("quota exhausted (429)" if "RESOURCE_EXHAUSTED" in str(last) or "429" in str(last)
                   else f"unavailable ({type(last).__name__})")
    raise RuntimeError(f"Gemini call failed: {type(last).__name__}: {str(last)[:200]}")


LAST_STATUS: str | None = None   # None = no failure seen; else a short human-readable reason


def probe() -> str:
    """One tiny grading call: 'ok' or a short reason such as 'quota exhausted (429)'. Cached by the caller."""
    global LAST_STATUS
    if not config.GEMINI_API_KEY:
        return "no API key"
    LAST_STATUS = None
    try:
        res = grade_items([{"id": "probe", "question": "Capital of France?", "gold": "Paris", "answer": "Paris"}])
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"
    return "ok" if res else (LAST_STATUS or "unavailable")


def _norm_score(v) -> float:
    try:
        v = float(v)
    except Exception:
        return 0.0
    return 1.0 if v >= 0.75 else 0.5 if v >= 0.25 else 0.0


def grade_items(items: list[dict], client=None, model: str | None = None,
                cache_path: Path | None = None, batch_size: int = BATCH_SIZE) -> dict[str, dict]:
    """items: [{id, question, gold, answer}] -> {id: {correct, score, reason, model}}.

    Uses the cache; sends only uncached items, batch_size per request. Failed batches
    are skipped (their ids absent from the result) so they can be retried later.
    """
    cache = _load_cache(cache_path)
    out: dict[str, dict] = {}
    todo = []
    for it in items:
        k = cache_key(it["question"], it["gold"], it["answer"])
        if k in cache:
            out[it["id"]] = cache[k]
        else:
            todo.append((k, it))
    if not todo:
        return out
    client = client or _client()
    models = [model] if model else pick_models(client)
    print(f"[referee] models(in order)={models[:4]} uncached={len(todo)} cached={len(out)}")
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        payload = [{"id": it["id"], "question": it["question"], "gold": it["gold"],
                    "model_answer": it["answer"]} for _, it in chunk]
        prompt = _GRADE_PROMPT + json.dumps(payload, ensure_ascii=False)
        try:
            res, model = _call_with_retry(client, models, prompt, _GRADE_SCHEMA)
            models = [model] + [m for m in models if m != model]  # stick with what worked
        except RuntimeError as e:
            print(f"[referee] batch skipped: {e}")
            if "budget exhausted" in str(e):
                break
            continue
        by_id = {str(r.get("id")): r for r in res if isinstance(r, dict)}
        for k, it in chunk:
            r = by_id.get(str(it["id"]))
            if r is None:
                continue
            sc = _norm_score(r.get("score"))
            verdict = {"correct": sc == 1.0,
                       "score": sc, "reason": str(r.get("reason", ""))[:300], "model": model}
            cache[k] = verdict
            out[it["id"]] = verdict
        _save_cache(cache, cache_path)
    return out


def grade_pending(db_path=None, client=None, limit: int | None = None) -> int:
    """Grade runs lacking a verdict and write it back. Returns number of rows updated."""
    db_path = db_path or config.DB_PATH
    con = sqlite3.connect(str(db_path), timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")
        q = ("SELECT run_id, question, gold, answer FROM runs WHERE referee_correct IS NULL "
             "AND answer IS NOT NULL AND gold IS NOT NULL")
        if limit:
            q += f" LIMIT {int(limit)}"
        rows = con.execute(q).fetchall()
        items = [{"id": r["run_id"], "question": r["question"] or "", "gold": r["gold"] or "",
                  "answer": r["answer"] or ""} for r in rows]
        if not items:
            return 0
        verdicts = grade_items(items, client=client)
        n = 0
        for rid, v in verdicts.items():
            con.execute("UPDATE runs SET referee_correct=?, referee_score=?, referee_reason=?, "
                        "referee_model=? WHERE run_id=?",
                        (int(v["correct"]), v["score"], v["reason"], v["model"], rid))
            n += 1
        con.commit()
        return n
    finally:
        con.close()


# --------------------------------------------------------------------------
# Optional: explain_run
# --------------------------------------------------------------------------
_EXPLAIN_SCHEMA = {"type": "OBJECT", "properties": {"explanation": {"type": "STRING"}},
                   "required": ["explanation"]}


def explain_run(question: str, kept_sentences: list[str], dropped_sentences: list[str],
                client=None, model: str | None = None) -> str | None:
    """Two plain sentences on what the selector kept and why. Cached; None on failure."""
    cache = _load_cache()
    k = "explain:" + hashlib.sha256(json.dumps([question, kept_sentences, dropped_sentences],
                                               ensure_ascii=False).encode()).hexdigest()
    if k in cache:
        return cache[k]["text"]
    try:
        client = client or _client()
        models = [model] if model else pick_models(client)
        prompt = (
            "A selector chose which sentences to feed a small language model to save energy. "
            "In exactly two plain-English sentences for a non-expert, say what was kept and why "
            "it was enough to answer, and what kind of text was dropped.\n"
            f"QUESTION: {question}\nKEPT: {json.dumps(kept_sentences[:12], ensure_ascii=False)}\n"
            f"DROPPED (sample): {json.dumps(dropped_sentences[:8], ensure_ascii=False)}")
        res, _ = _call_with_retry(client, models, prompt, _EXPLAIN_SCHEMA)
        text = str(res.get("explanation", "")).strip()
    except Exception as e:  # noqa: BLE001
        print(f"[referee] explain_run failed: {str(e)[:150]}")
        return None
    cache[k] = {"text": text}
    _save_cache(cache)
    return text

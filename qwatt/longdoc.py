"""Q-Watt Long-Context: one qubit per sentence of a WHOLE document (SQuAD v1.1 dev articles), no shortlist.

Document  = contiguous paragraphs of one SQuAD article, <= MAX_SENTS sentences and <= MAX_TOKENS estimated tokens.
Question  = a SQuAD question whose paragraph lies in the document. Gold sentence = the sentence containing
            answer_start (first annotator).
Pool      = qwatt.types.Pool with sentences in DOCUMENT ORDER (qubit i = sentence i), n up to 200.

QUBO couplings are LOCAL ONLY (|i - j| <= W in document order), because the Quantum Rings simulator is fast for
banded circuits and very slow for random long-range ones (CONTRACTS.md, pivot):
  red_ij = max(0, cos_ij - theta)/(1 - theta)        for 0 < |i-j| <= W           (penalty, weight lam)
  br_ij  = coherence for adjacent sentences of the same paragraph (j = i+1):     (reward, weight nu)
           1.0 if sentence j opens with a pronoun / connective ("He", "This", "However", ...) else COH_BASE.
Relevance r_i = minmax(0.75 cos(q, sentence) + 0.25 cos(q, its paragraph)) with per-document TF-IDF.

Settings for this mode come from `lc_settings()` (results/lc_hyperparams.json); results/hyperparams.json (the
16-qubit HotpotQA mode) is never touched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import re

import numpy as np

from qwatt import config
from qwatt.config import Settings
from qwatt.data import count_tokens
from qwatt.embed import EMBED_BACKEND, get_embedder
from qwatt.types import Pool, Sentence

RAW_PARQUET = config.RAW_DIR / "squad_v1_dev.parquet"
SUBSET_JSON = config.DATA_DIR / "squad_lc_subset.json"
LC_SPLITS_JSON = config.RESULTS_DIR / "lc_splits.json"
LC_HYPERPARAMS_JSON = config.RESULTS_DIR / "lc_hyperparams.json"
LC_ANGLES_JSON = config.RESULTS_DIR / "lc_angles.json"
POOL_VERSION = "lc1"
MAX_SENTS = 200
MAX_TOKENS = 5500
SCALING_SENTS = 200
MIN_SENTS = 60          # shorter tail chunks are not used as documents
W_DEFAULT = 2
COH_BASE = 0.3
SENT_W = 0.75           # relevance = SENT_W * cos(q, sentence) + (1 - SENT_W) * cos(q, paragraph)

LC_DEFAULTS = {"lam": 1.0, "nu": 0.3, "red_threshold": 0.3, "budget_tokens": 300, "shots": 1024, "p": 1,
               "extra": {"rho": 0.0025, "r_power": 2, "W": W_DEFAULT, "rent_mode": "budget_matched", "fill": 1.0}}


def lc_settings() -> Settings:
    """Settings for the long-context mode: LC_DEFAULTS overridden by results/lc_hyperparams.json."""
    base = config.load_settings()          # keeps chars_per_token from calibration
    s = Settings(chars_per_token=base.chars_per_token, seed=base.seed)
    data = copy.deepcopy(LC_DEFAULTS)
    if LC_HYPERPARAMS_JSON.exists():
        hp = json.loads(LC_HYPERPARAMS_JSON.read_text())
        extra = {**data["extra"], **hp.get("extra", {})}
        data.update(hp)
        data["extra"] = extra
    for k, v in data.items():
        if hasattr(s, k):
            setattr(s, k, v)
    s.n_pool = MAX_SENTS
    s.extra = {**(s.extra or {}), "mode": "lc"}
    return s


# ---------------------------------------------------------------- sentences
_ABBR = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "mt", "vs", "etc", "no", "inc", "ltd", "co", "gen", "col",
         "lt", "sgt", "capt", "prof", "rev", "fig", "approx", "ca", "u.s", "e.g", "i.e"}
_BOUND = re.compile(r'[.!?]["\')\]]*\s+(?=["\'(\[]?[A-Z0-9])')


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Character spans (start, end) of sentences. Avoids splitting after common abbreviations / initials."""
    spans, start = [], 0
    for m in _BOUND.finditer(text):
        word = re.findall(r"[\w.]+$", text[start:m.start() + 1].rstrip(".!?\"')]"))
        w = word[0].lower() if word else ""
        if w in _ABBR or (len(w) == 1 and w.isalpha()) or re.fullmatch(r"(\w\.)+\w?", w or "-"):
            continue
        end = m.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


_COH_START = re.compile(r"^(he|she|it|they|this|these|those|its|his|her|their|there|such|however|also|"
                        r"in addition|additionally|furthermore|moreover|thus|therefore|as a result|later|"
                        r"meanwhile|both|each|the latter|the former)\b", re.I)


# ---------------------------------------------------------------- documents and splits
def _articles() -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(RAW_PARQUET)
    arts: dict[str, dict] = {}
    for r in df.itertuples(index=False):
        a = arts.setdefault(r.title, {"title": r.title, "paragraphs": [], "pidx": {}, "questions": []})
        if r.context not in a["pidx"]:
            a["pidx"][r.context] = len(a["paragraphs"])
            a["paragraphs"].append(r.context)
        a["questions"].append({"id": r.id, "question": r.question, "para": a["pidx"][r.context],
                               "answers": [str(x) for x in r.answers["text"]],
                               "answer_start": int(r.answers["answer_start"][0])})
    return list(arts.values())


def make_documents(settings) -> tuple[list[dict], list[dict]]:
    """Chunk every article into contiguous-paragraph documents under the caps."""
    docs, qs = [], []
    for a in _articles():
        chunk, n_s, n_t = [], 0, 0
        chunks = []
        for p, text in enumerate(a["paragraphs"]):
            ns, nt = len(split_sentences(text)), count_tokens(text, settings)
            if chunk and (n_s + ns > MAX_SENTS or n_t + nt > MAX_TOKENS):
                chunks.append(chunk)
                chunk, n_s, n_t = [], 0, 0
            chunk.append(p)
            n_s += ns
            n_t += nt
        if chunk:
            chunks.append(chunk)
        for c, paras in enumerate(chunks):
            n_s = sum(len(split_sentences(a["paragraphs"][p])) for p in paras)
            if n_s < MIN_SENTS:
                continue
            doc_id = f"{a['title']}#{c}"
            docs.append({"doc_id": doc_id, "title": a["title"].replace("_", " "),
                         "paragraphs": [a["paragraphs"][p] for p in paras], "n_sentences": n_s})
            local = {p: k for k, p in enumerate(paras)}
            for q in a["questions"]:
                if q["para"] in local:
                    qs.append({**q, "doc_id": doc_id, "para": local[q["para"]]})
    return docs, qs


def make_lc_splits(seed: int = 7) -> dict:
    """lc_dev: 30 q from 3 docs; lc_test: 100 q from 10 other docs; lc_bench: 20 q (4 from each of 5 test docs).
    Documents with 100-200 sentences are preferred. Dev and test documents are disjoint (no tuning leakage)."""
    settings = config.load_settings()
    docs, qs = make_documents(settings)
    rng = np.random.default_rng(seed)
    by_doc: dict[str, list] = {}
    for q in qs:
        by_doc.setdefault(q["doc_id"], []).append(q)
    good = [d for d in docs if 100 <= d["n_sentences"] <= MAX_SENTS and len(by_doc.get(d["doc_id"], [])) >= 10]
    order = [good[i] for i in rng.permutation(len(good))]
    dev_docs, test_docs = order[:3], order[3:13]
    pick = lambda d, k: [by_doc[d["doc_id"]][i]["id"] for i in sorted(rng.choice(len(by_doc[d["doc_id"]]), k, replace=False))]
    dev = [q for d in dev_docs for q in pick(d, 10)]
    test = [q for d in test_docs for q in pick(d, 10)]
    bench = [q for d in range(5) for q in test[d * 10: d * 10 + 4]]
    # lc_tune = lc_dev + 10 q from each of 7 further documents (never test): 30 dev questions overfit the weights
    tune_docs = order[13:20]
    tune = dev + [q for d in tune_docs for q in pick(d, 10)]
    used = {d["doc_id"] for d in dev_docs + test_docs + tune_docs}
    splits = {"source": "SQuAD v1.1 dev (HF rajpurkar/squad plain_text/validation)", "seed": seed,
              "n_docs_total": len(docs), "caps": {"max_sentences": MAX_SENTS, "max_tokens": MAX_TOKENS},
              "lc_dev": dev, "lc_test": test, "lc_bench": bench, "lc_tune": tune,
              "tune_docs": [d["doc_id"] for d in tune_docs],
              "dev_docs": [d["doc_id"] for d in dev_docs], "test_docs": [d["doc_id"] for d in test_docs]}
    # Scaling document (results/scaling.json only): the article of the first test document, paragraphs from
    # its start until >= SCALING_SENTS sentences, token cap ignored, so circuits up to 200 qubits use real text.
    art = next(a for a in _articles() if a["title"].replace("_", " ") == test_docs[0]["title"])
    paras, n_s = [], 0
    for text in art["paragraphs"]:
        paras.append(text)
        n_s += len(split_sentences(text))
        if n_s >= SCALING_SENTS:
            break
    sdoc = {"doc_id": f"SCALING:{art['title']}", "title": art["title"].replace("_", " "), "paragraphs": paras,
            "n_sentences": n_s}
    sq = next(q for q in art["questions"] if q["para"] == 0)
    sq = {**sq, "id": "scaling-" + sq["id"], "doc_id": sdoc["doc_id"]}
    splits["scaling_qid"] = sq["id"]
    LC_SPLITS_JSON.write_text(json.dumps(splits, indent=1))
    keep = set(dev) | set(test) | set(tune)
    SUBSET_JSON.write_text(json.dumps({"docs": [d for d in docs if d["doc_id"] in used] + [sdoc],
                                       "questions": [q for q in qs if q["id"] in keep] + [sq]}, ensure_ascii=False),
                           encoding="utf-8")
    return splits


_SUB: dict | None = None


def _subset() -> dict:
    global _SUB
    if _SUB is None:
        raw = json.loads(SUBSET_JSON.read_text(encoding="utf-8"))
        _SUB = {"docs": {d["doc_id"]: d for d in raw["docs"]}, "q": {q["id"]: q for q in raw["questions"]}}
    return _SUB


def load_lc_split(name: str) -> list[dict]:
    """Question dicts for 'lc_dev' | 'lc_test' | 'lc_bench'."""
    sp = json.loads(LC_SPLITS_JSON.read_text())
    return [_subset()["q"][q] for q in sp[name]]


def get_doc(doc_id: str) -> dict:
    return _subset()["docs"][doc_id]


# ---------------------------------------------------------------- pool
def doc_sentences(doc: dict) -> list[tuple[int, int, int, str]]:
    """[(paragraph, start, end, text)] in document order (offsets within the paragraph)."""
    out = []
    for p, text in enumerate(doc["paragraphs"]):
        for a, b in split_sentences(text):
            out.append((p, a, b, text[a:b].strip()))
    return out


def local_mask(n: int, W: int) -> np.ndarray:
    i, j = np.indices((n, n))
    return (np.abs(i - j) <= W) & (i != j)


def build_lc_pool(q: dict, settings, embedder=None, max_n: int | None = None) -> Pool:
    doc = get_doc(q["doc_id"])
    rows = doc_sentences(doc)
    if max_n:
        rows = rows[:max_n]
    n = len(rows)
    W = int((settings.extra or {}).get("W", W_DEFAULT))
    texts = [r[3] for r in rows]
    emb = embedder or get_embedder()
    emb.fit(texts + doc["paragraphs"] + [q["question"]])
    E_q = emb.encode([q["question"]])[0]
    E_s = emb.encode(texts)
    E_p = emb.encode(doc["paragraphs"])
    raw = SENT_W * (E_s @ E_q) + (1 - SENT_W) * (E_p @ E_q)[[r[0] for r in rows]]
    r = (raw - raw.min()) / (raw.max() - raw.min()) if raw.max() > raw.min() else np.ones(n)
    sim = E_s @ E_s.T
    np.fill_diagonal(sim, 0.0)
    band = local_mask(n, W)
    theta = settings.red_threshold
    red = np.where(band, np.clip((sim - theta) / (1 - theta), 0, 1), 0.0)
    br = np.zeros((n, n))
    coh = []
    for i in range(n - 1):
        if rows[i][0] == rows[i + 1][0]:
            w = 1.0 if _COH_START.match(texts[i + 1]) else COH_BASE
            br[i, i + 1] = br[i + 1, i] = w
            coh.append([i, i + 1, w])
    gold_i = [i for i, (p, a, b, _) in enumerate(rows) if p == q["para"] and a <= q["answer_start"] < b]
    sents = [Sentence(idx=i, text=texts[i], title=doc["title"], sent_id=i, tokens=max(1, count_tokens(texts[i], settings)),
                      relevance=float(r[i]), is_gold=i in gold_i) for i in range(n)]
    full = f"{doc['title']}:\n" + "\n\n".join(doc["paragraphs"])
    return Pool(qid=q["id"], question=q["question"], answer=q["answers"][0], qtype="squad_lc", sentences=sents,
                red=red, br=br, full_context=full, full_tokens=count_tokens(full, settings),
                pool_recall=1.0 if gold_i else 0.0, n_gold_total=len(gold_i),
                meta={"doc_id": doc["doc_id"], "paragraph_of": [r[0] for r in rows], "coherence_edges": coh, "W": W,
                      "answers": q["answers"], "embed_backend": getattr(emb, "name", "?"), "pool_version": POOL_VERSION,
                      "sim_band": [[i, j, round(float(sim[i, j]), 5)] for i in range(n) for j in range(i + 1, min(n, i + W + 1))]})


def get_lc_pool(qid: str, settings=None, max_n: int | None = None) -> Pool:
    """Long-context pool (cached under data/cache/; key includes the settings that shape the pool)."""
    settings = settings or lc_settings()
    key = json.dumps([POOL_VERSION, EMBED_BACKEND, settings.red_threshold, settings.chars_per_token,
                      (settings.extra or {}).get("W", W_DEFAULT), COH_BASE, SENT_W, max_n]).encode()
    path = config.CACHE_DIR / f"lcpool_{qid}_{hashlib.sha1(key).hexdigest()[:10]}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    pool = build_lc_pool(_subset()["q"][qid], settings, max_n=max_n)
    with open(path, "wb") as f:
        pickle.dump(pool, f)
    return pool


def truncate_pool(pool: Pool, n: int) -> Pool:
    """First n sentences of a pool (scaling runs, angle-training windows). Relevance is NOT renormalized."""
    p = copy.copy(pool)
    p.sentences = pool.sentences[:n]
    p.red, p.br = pool.red[:n, :n], pool.br[:n, :n]
    p.meta = {**pool.meta, "paragraph_of": pool.meta["paragraph_of"][:n]}
    return p


def window_pool(pool: Pool, start: int, n: int) -> Pool:
    p = copy.copy(pool)
    p.sentences = pool.sentences[start:start + n]
    p.red, p.br = pool.red[start:start + n, start:start + n], pool.br[start:start + n, start:start + n]
    p.meta = {**pool.meta, "paragraph_of": pool.meta["paragraph_of"][start:start + n]}
    return p


def gold_recall(pool: Pool, indices) -> float:
    return sum(pool.sentences[i].is_gold for i in indices) / pool.n_gold_total if pool.n_gold_total else 0.0

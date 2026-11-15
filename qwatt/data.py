"""HotpotQA (distractor, dev) loading, splits, and candidate pools.

One question -> one Pool of n=16 candidate sentences (qubit i == pool.sentences[i]).

Pool construction (plan 6.1 / 7):
  1. Every sentence of the 10 paragraphs gets a relevance score
        rel_raw = 0.5 * cos(q, "Title: sentence") + 0.5 * cos(q, whole paragraph)
     (the paragraph term lets pronoun-only sentences like "He was born in ..." inherit context).
  2. Seeds = top `N_SEEDS` sentences by rel_raw.
  3. Bridge neighbors: sentences linked to a seed by a bridge (different paragraph, one mentions the
     other's title) are added next, best first.
  4. Remaining slots filled by relevance. Pool is ordered by relevance (qubit 0 = most relevant).
  Within the pool: r = minmax( minmax(rel_raw) + LEAD_BONUS * [sent_id == 0] ), a lead-sentence prior
  (a paragraph's first sentence usually defines its subject). It is applied after pool selection (applying it
  before pulled every distractor's first sentence into the pool and cut pool recall) and it feeds ALL arms.
Pool recall (gold sentences in the pool / all gold sentences) is reported because it caps everything.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import re

import numpy as np

from qwatt import config
from qwatt.embed import EMBED_BACKEND, get_embedder
from qwatt.types import Pool, Sentence

RAW_PARQUET = config.RAW_DIR / "hotpot_dev_distractor.parquet"
RAW_JSON = config.RAW_DIR / "hotpot_dev_distractor_v1.json"
SUBSET_JSON = config.DATA_DIR / "hotpot_subset.json"
POOL_VERSION = "v3"
N_SEEDS = 10
LEAD_BONUS = 0.3   # lead-sentence prior: a paragraph's first sentence usually names its subject (tuned on dev)
N_DEV, N_TEST, N_BENCH = 30, 100, 20
N_TUNE_EXTRA = 170   # "tune" = dev + 170 more non-test items: 30 questions overfit the QUBO weights


# ---------------------------------------------------------------- tokens
def count_tokens(text: str, settings) -> int:
    """Token estimate used everywhere (QUBO, budget, ledger). Refined by calibration's chars_per_token."""
    return int(math.ceil(len(text) / float(settings.chars_per_token))) if text else 0


# ---------------------------------------------------------------- raw data and splits
def _load_raw() -> list[dict]:
    """All 7405 dev-distractor items in the original HotpotQA JSON layout.

    The CMU host did not resolve from this network, so the HuggingFace parquet mirror
    (hotpotqa/hotpot_qa, distractor/validation) is used; same 7405 items.
    """
    if RAW_JSON.exists():
        return json.loads(RAW_JSON.read_text(encoding="utf-8"))
    import pandas as pd

    df = pd.read_parquet(RAW_PARQUET)
    items = []
    for r in df.itertuples(index=False):
        sf = r.supporting_facts
        ctx = r.context
        items.append({
            "_id": r.id, "question": r.question, "answer": r.answer, "type": r.type, "level": r.level,
            "supporting_facts": [[str(t), int(s)] for t, s in zip(sf["title"], sf["sent_id"])],
            "context": [[str(t), [str(x) for x in sents]] for t, sents in zip(ctx["title"], ctx["sentences"])],
        })
    return items


def make_splits(seed: int = 7) -> dict:
    """Fixed-seed split: 30 dev, 100 test, bench = first 20 of test, tune = dev + 170 further non-test items.
    One permutation, so dev/test/bench do not depend on the tune size. Writes splits.json + subset json."""
    items = _load_raw()
    perm = np.random.default_rng(seed).permutation(len(items))
    dev_idx = perm[:N_DEV].tolist()
    test_idx = perm[N_DEV:N_DEV + N_TEST].tolist()
    bench_idx = test_idx[:N_BENCH]
    tune_idx = dev_idx + perm[N_DEV + N_TEST:N_DEV + N_TEST + N_TUNE_EXTRA].tolist()
    splits = {
        "source": "HotpotQA dev distractor v1 (HF mirror hotpotqa/hotpot_qa distractor/validation)",
        "n_total": len(items), "seed": seed,
        "dev": [items[i]["_id"] for i in dev_idx],
        "test": [items[i]["_id"] for i in test_idx],
        "bench": [items[i]["_id"] for i in bench_idx],
        "tune": [items[i]["_id"] for i in tune_idx],
        "dev_row_index": dev_idx, "test_row_index": test_idx, "bench_row_index": bench_idx,
        "tune_row_index": tune_idx,
    }
    config.SPLITS_JSON.write_text(json.dumps(splits, indent=1))
    keep = sorted(set(dev_idx) | set(test_idx) | set(tune_idx))
    SUBSET_JSON.write_text(json.dumps([items[i] for i in keep], ensure_ascii=False), encoding="utf-8")
    return splits


_SUBSET: dict[str, dict] | None = None


def _subset() -> dict[str, dict]:
    global _SUBSET
    if _SUBSET is None:
        _SUBSET = {it["_id"]: it for it in json.loads(SUBSET_JSON.read_text(encoding="utf-8"))}
    return _SUBSET


def load_split(name: str) -> list[dict]:
    """Raw HotpotQA items for 'dev' | 'test' | 'bench' | 'tune', in split order."""
    splits = json.loads(config.SPLITS_JSON.read_text())
    sub = _subset()
    return [sub[q] for q in splits[name]]


def get_item(qid: str) -> dict:
    return _subset()[qid]


# ---------------------------------------------------------------- bridges
_PAREN = re.compile(r"\s*\(.*?\)\s*$")


def title_key(title: str) -> str:
    """'Ed Wood (film)' -> 'ed wood'."""
    return _PAREN.sub("", title).strip().lower()


def mentions(text: str, title: str) -> bool:
    key = title_key(title)
    if len(key) < 3:
        return False
    return re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", text.lower()) is not None


def bridge_matrix(texts: list[str], titles: list[str]) -> np.ndarray:
    """br_ij = 1 if i, j come from different paragraphs and one mentions the other's title."""
    n = len(texts)
    br = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if titles[i] != titles[j] and (mentions(texts[i], titles[j]) or mentions(texts[j], titles[i])):
                br[i, j] = br[j, i] = 1.0
    return br


def red_from_sim(sim: np.ndarray, theta: float) -> np.ndarray:
    """red_ij = max(0, cos_ij - theta) / (1 - theta), zero diagonal."""
    red = np.clip((sim - theta) / (1.0 - theta), 0.0, 1.0)
    np.fill_diagonal(red, 0.0)
    return red


# ---------------------------------------------------------------- pool
def format_paragraph(title: str, sents: list[str]) -> str:
    return f"{title}: " + " ".join(s.strip() for s in sents)


def build_pool(item: dict, settings, embedder=None) -> Pool:
    n_pool = settings.n_pool
    gold = {(t, int(s)) for t, s in item["supporting_facts"]}
    rows = []  # (title, sent_id, text, para_idx)
    for p, (title, sents) in enumerate(item["context"]):
        for k, s in enumerate(sents):
            if s.strip():
                rows.append((title, k, s.strip(), p))
    n_gold_total = sum(1 for (t, k, _, _) in rows if (t, k) in gold)

    paras = [format_paragraph(t, s) for t, s in item["context"]]
    sent_texts = [r[2] for r in rows]
    titled = [f"{r[0]}: {r[2]}" for r in rows]
    emb = embedder or get_embedder()
    emb.fit(titled + paras + [item["question"]])
    E_q = emb.encode([item["question"]])[0]
    E_titled = emb.encode(titled)
    E_para = emb.encode(paras)
    E_sent = emb.encode(sent_texts)
    rel_raw = 0.5 * (E_titled @ E_q) + 0.5 * (E_para @ E_q)[[r[3] for r in rows]]

    titles = [r[0] for r in rows]
    br_all = bridge_matrix(sent_texts, titles)
    order = list(np.argsort(-rel_raw, kind="stable"))
    chosen = order[:min(N_SEEDS, n_pool)]
    neigh = [j for j in order if j not in chosen and br_all[j, chosen].any()]
    chosen += neigh[: n_pool - len(chosen)]
    chosen += [j for j in order if j not in chosen][: n_pool - len(chosen)]

    def minmax(v):
        return (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.ones_like(v)

    r = minmax(minmax(rel_raw[chosen]) + LEAD_BONUS * np.array([rows[j][1] == 0 for j in chosen], dtype=float))
    order_pool = np.argsort(-r, kind="stable")          # qubit 0 = most relevant
    chosen, r = [chosen[k] for k in order_pool], r[order_pool]
    sim = E_sent[chosen] @ E_sent[chosen].T
    np.fill_diagonal(sim, 0.0)
    sentences = [
        Sentence(idx=i, text=rows[j][2], title=rows[j][0], sent_id=rows[j][1],
                 tokens=max(1, count_tokens(rows[j][2], settings)), relevance=float(r[i]),
                 is_gold=(rows[j][0], rows[j][1]) in gold)
        for i, j in enumerate(chosen)
    ]
    full_context = "\n\n".join(paras)
    n_gold_pool = sum(s.is_gold for s in sentences)
    return Pool(
        qid=item["_id"], question=item["question"], answer=item["answer"], qtype=item["type"],
        sentences=sentences, red=red_from_sim(sim, settings.red_threshold), br=br_all[np.ix_(chosen, chosen)],
        full_context=full_context, full_tokens=count_tokens(full_context, settings),
        pool_recall=n_gold_pool / n_gold_total if n_gold_total else 0.0, n_gold_total=n_gold_total,
        meta={"sim": np.round(sim, 6).tolist(), "embed_backend": getattr(emb, "name", "?"),
              "gold_titles": sorted({t for t, _ in gold}), "n_sentences_total": len(rows),
              "pool_version": POOL_VERSION},
    )


def _cache_key(qid: str, settings) -> str:
    blob = json.dumps([POOL_VERSION, EMBED_BACKEND, settings.n_pool, settings.red_threshold,
                       settings.chars_per_token, N_SEEDS, LEAD_BONUS]).encode()
    return f"pool_{qid}_{hashlib.sha1(blob).hexdigest()[:10]}.pkl"


def get_pool(qid: str, settings=None) -> Pool:
    """Pool for a question, cached on disk under data/cache/ (key includes the pool settings)."""
    settings = settings or config.load_settings()
    path = config.CACHE_DIR / _cache_key(qid, settings)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    pool = build_pool(get_item(qid), settings)
    with open(path, "wb") as f:
        pickle.dump(pool, f)
    return pool


def with_theta(pool: Pool, theta: float) -> Pool:
    """Same pool, redundancy recomputed at a different threshold (used by tuning)."""
    import copy

    p = copy.copy(pool)
    p.red = red_from_sim(np.array(pool.meta["sim"]), theta)
    return p


# ---------------------------------------------------------------- selection metrics
def sf_recall(pool: Pool, indices) -> float:
    if not pool.n_gold_total:
        return 0.0
    return sum(pool.sentences[i].is_gold for i in indices) / pool.n_gold_total


def both_gold(pool: Pool, indices) -> bool:
    """Selection contains at least one supporting sentence from every gold paragraph."""
    got = {pool.sentences[i].title for i in indices if pool.sentences[i].is_gold}
    return set(pool.meta["gold_titles"]) <= got

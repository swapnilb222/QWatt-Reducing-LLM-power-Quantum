"""Sentence embeddings for relevance and redundancy.

Default backend: TF-IDF (scikit-learn), fitted per question on that question's own 10 paragraphs.
Offline, deterministic, ~1 ms per question, no corpus or network needed.

Optional backend: LM Studio /v1/embeddings (set QWATT_EMBED=lmstudio). Tuning and runtime must use
the same backend; the backend name is recorded in results/hyperparams.json (extra.embed_backend)
and in every pool's meta.
"""
from __future__ import annotations

import os

import numpy as np

EMBED_BACKEND = os.getenv("QWATT_EMBED", "tfidf")


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class TfidfEmbedder:
    """Word (1-2 gram) + character (3-5 gram) TF-IDF, concatenated, L2-normalized.

    Character n-grams catch morphology ("America" vs "American"), word n-grams catch names.
    `fit(corpus)` is called once per question with all of its sentences.
    """

    name = "tfidf"

    def __init__(self) -> None:
        self._word = None
        self._char = None

    def fit(self, corpus: list[str]) -> "TfidfEmbedder":
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._word = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2),
                                     lowercase=True).fit(corpus)
        self._char = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5),
                                     lowercase=True).fit(corpus)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        w = _l2(self._word.transform(texts).toarray())
        c = _l2(self._char.transform(texts).toarray())
        return _l2(np.hstack([w, c]))


class LMStudioEmbedder:
    """OpenAI-compatible /v1/embeddings on LM Studio. Model id read from /v1/models (never hardcoded)."""

    name = "lmstudio"

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        from qwatt import config

        self.base_url = (base_url or config.LMSTUDIO_BASE_URL).rstrip("/")
        self.model = model or os.getenv("QWATT_EMBED_MODEL", "")

    def fit(self, corpus: list[str]) -> "LMStudioEmbedder":
        return self

    def _pick_model(self) -> str:
        import requests

        ids = [m["id"] for m in requests.get(f"{self.base_url}/models", timeout=5).json()["data"]]
        emb = [i for i in ids if "embed" in i.lower()]
        if not emb:
            raise RuntimeError("no embedding model loaded in LM Studio")
        return emb[0]

    def encode(self, texts: list[str]) -> np.ndarray:
        import requests

        if not self.model:
            self.model = self._pick_model()
        r = requests.post(f"{self.base_url}/embeddings", json={"model": self.model, "input": texts}, timeout=60)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return _l2(np.array([d["embedding"] for d in data], dtype=float))


def get_embedder(name: str | None = None):
    name = name or EMBED_BACKEND
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "lmstudio":
        return LMStudioEmbedder()
    raise ValueError(f"unknown embed backend {name!r}")

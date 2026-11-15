"""Shared data types. Owned by the lead; change only by agreement (see CONTRACTS.md)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


@dataclass
class Sentence:
    idx: int            # position in the pool == qubit index
    text: str
    title: str          # paragraph title it came from
    sent_id: int        # sentence index within its paragraph (HotpotQA numbering)
    tokens: int         # token count of the sentence text
    relevance: float    # r_i in [0,1], min-max normalized over the pool
    is_gold: bool       # HotpotQA supporting fact


@dataclass
class Pool:
    qid: str
    question: str
    answer: str                      # gold answer
    qtype: str                       # 'bridge' | 'comparison'
    sentences: list[Sentence]
    red: np.ndarray                  # (n,n) symmetric, zero diagonal, in [0,1]
    br: np.ndarray                   # (n,n) symmetric, zero diagonal, {0,1}
    full_context: str                # A0 context: all 10 paragraphs, formatted
    full_tokens: int
    pool_recall: float               # fraction of gold sentences present in the pool
    n_gold_total: int
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.sentences)

    @property
    def tokens(self) -> np.ndarray:
        return np.array([s.tokens for s in self.sentences], dtype=float)

    @property
    def relevance(self) -> np.ndarray:
        return np.array([s.relevance for s in self.sentences], dtype=float)


class Selector(Protocol):
    """Every selector: candidates in, (selected_indices, diagnostics) out."""

    def __call__(self, pool: Pool, budget: int, settings: Any) -> tuple[list[int], dict]: ...


EventCallback = Callable[[dict], None]

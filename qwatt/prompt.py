"""One prompt template for every arm (same template, different context)."""
from __future__ import annotations

import re
import secrets
from typing import Sequence

from qwatt.types import Pool

ANSWER_LINE = ("Reply with only the short answer (a name, entity, date, number, or yes/no), at most 6 words. "
               "No full sentence, no explanation. "
               "If the context does not contain the answer, reply exactly: NOT FOUND")
NOT_FOUND = "NOT FOUND"
INSTRUCTION = "Answer the question using only the context below."
FULL_ARMS = ("full", "full_lc")


def new_nonce() -> str:
    return secrets.token_hex(6)


def format_selected(pool: Pool, indices: Sequence[int]) -> str:
    """Selected sentences, in the SAME paragraph format as pool.full_context, paragraphs separated by
    a blank line, original order.
    - HotpotQA pools: grouped under the paragraph title ("Title: sentence sentence"), title paid once.
    - Long-document pools (pool.meta['paragraph_of'] present): document order, paragraph breaks only."""
    idx = sorted(set(int(i) for i in indices))
    sents = [pool.sentences[i] for i in idx]
    pof = pool.meta.get("paragraph_of") if isinstance(pool.meta, dict) else None
    if pof:
        paras: list[list[str]] = []
        last = object()
        for i in idx:
            if pof[i] != last:
                paras.append([])
                last = pof[i]
            paras[-1].append(pool.sentences[i].text.strip())
        return "\n\n".join(" ".join(p) for p in paras)
    groups: dict[str, list] = {}
    for s in sents:
        groups.setdefault(s.title, []).append(s)

    def para_pos(title: str) -> int:
        k = pool.full_context.find(f"{title}: ")
        return k if k >= 0 else 10**9

    out = []
    for title in sorted(groups, key=lambda t: (para_pos(t), t)):
        body = " ".join(s.text.strip() for s in sorted(groups[title], key=lambda s: s.sent_id))
        out.append(f"{title}: {body}")
    return "\n\n".join(out)


def build_messages(pool: Pool, indices: Sequence[int] | None, arm: str, nonce: str) -> list[dict]:
    """Chat messages for `arm`. Arms 'full'/'full_lc' use pool.full_context; others use the selected
    sentences. The nonce line is the very first line of the prompt (defeats llama.cpp prefix caching)."""
    context = pool.full_context if arm in FULL_ARMS else format_selected(pool, indices or [])
    user = (
        f"[ref {nonce}]\n"
        f"{INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {pool.question}\n"
        f"{ANSWER_LINE}"
    )
    return [{"role": "user", "content": user}]


def context_chars(pool: Pool, indices: Sequence[int] | None, arm: str) -> int:
    return len(pool.full_context if arm in FULL_ARMS else format_selected(pool, indices or []))


# ------------------------------------------------------------------ abstention detection
_ABSTAIN = re.compile(
    r"not[\s_-]*found|no[\s_-]*answer|cannot[\s\w]{0,15}determine|can'?t[\s\w]{0,15}determine|"
    r"unable to|"
    r"not (?:provided|mentioned|stated|specified|available|given|present|included|"
    r"in the (?:context|text|passage|document))|"
    r"(?:does not|doesn'?t|do not|don'?t|did not|didn'?t)\s+(?:\w+\s+){0,2}"
    r"(?:mention|contain|say|specify|state|provide|include|give|list|have)|"
    r"insufficient|no (?:relevant |such )?(?:information|mention|data|details?)|not enough|"
    r"context (?:is )?(?:silent|unclear)|"
    r"^\W*(?:n/?a|none|unknown|unsure|unclear)\W*$|i (?:do not|don'?t|cannot|can'?t) (?:know|find)",
    re.I,
)


def is_abstention(text: str | None) -> tuple[bool, str]:
    """(abstained, reason). Empty output counts as abstention. The regex only looks at short replies
    (the prompt asks for <= 6 words) so a long real answer that happens to say 'not' is not misread."""
    t = (text or "").strip()
    if not t:
        return True, "empty"
    if t.upper().strip(" .!\"'`*") == NOT_FOUND:
        return True, "not_found"
    if len(t.split()) <= 25 and _ABSTAIN.search(t):
        return True, "abstain_phrase"
    return False, ""

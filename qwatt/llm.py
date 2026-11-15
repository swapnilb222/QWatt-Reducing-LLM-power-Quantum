"""LM Studio (OpenAI-compatible) streaming client with client-side TTFT, plus a FAKE mode.

Timing definitions (client-side):
  prefill_s  = request start -> first content token      (== ttft_s)
  generate_s = first content token -> stream end
Reasoning ("thinking") content is never counted as an answer token for timing; it is stripped from
the text and reported in `reasoning_tokens` so a leak is visible, not hidden.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from qwatt import config

BACKEND_REAL = "lmstudio"
BACKEND_FAKE = "fake"

# Fake-mode speed model (tokens/s). Tagged llm_backend='fake' everywhere.
FAKE_PREFILL_TOK_S = 450.0
FAKE_DECODE_TOK_S = 14.0
FAKE_BASE_LATENCY_S = 0.08


@dataclass
class LLMResult:
    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_tokens: int = 0
    ttft_s: float = 0.0
    prefill_s: float = 0.0
    generate_s: float = 0.0
    total_s: float = 0.0
    t_start: float = 0.0
    t_first: float = 0.0
    t_end: float = 0.0
    model_id: str = ""
    backend: str = BACKEND_REAL
    usage_estimated: bool = False
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)


def is_fake() -> bool:
    return bool(config.FAKE_LLM)


def _base() -> str:
    return config.LMSTUDIO_BASE_URL.rstrip("/")


def list_models(timeout: float = 5.0) -> list[str]:
    import requests
    r = requests.get(_base() + "/models", timeout=timeout)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def pick_model_id(ids: list[str] | None = None) -> str:
    """Prefer a gemma chat model, skip embedding models. Env LMSTUDIO_MODEL overrides."""
    forced = os.getenv("LMSTUDIO_MODEL")
    if forced:
        return forced
    if is_fake():
        return "fake-gemma-4-e4b"
    ids = ids if ids is not None else list_models()
    chat = [i for i in ids if "embed" not in i.lower() and "nomic" not in i.lower()]
    for i in chat:
        if "gemma" in i.lower():
            return i
    if chat:
        return chat[0]
    raise RuntimeError(f"No chat model loaded in LM Studio (models: {ids})")


def health() -> dict:
    """{'ok': bool, 'backend': ..., 'model_id': ..., 'error': ...} - never raises."""
    if is_fake():
        return {"ok": True, "backend": BACKEND_FAKE, "model_id": pick_model_id()}
    try:
        return {"ok": True, "backend": BACKEND_REAL, "model_id": pick_model_id()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "backend": BACKEND_REAL, "model_id": None, "error": str(e)[:200]}


def estimate_tokens(text: str, cpt: float | None = None) -> int:
    cpt = cpt or config.Settings().chars_per_token
    return max(1, int(round(len(text) / cpt)))


def messages_text(messages: list[dict]) -> str:
    return "\n".join(m.get("content", "") for m in messages)


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_reasoning(text: str) -> str:
    text = _THINK_RE.sub("", text)
    if "</think>" in text:           # stream began inside a think block
        text = text.split("</think>", 1)[1]
    return text.strip()


# --------------------------------------------------------------------------- real
def _payload(model: str, messages: list[dict], settings, thinking_off: bool) -> dict:
    p = {
        "model": model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "seed": settings.seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if thinking_off:
        # Best effort: LM Studio / llama.cpp templates read enable_thinking; `reasoning_effort` is
        # honoured by models that support it. Unknown fields are ignored or rejected (we retry without).
        p["chat_template_kwargs"] = {"enable_thinking": False}
        p["reasoning_effort"] = "none"
    return p


def _generate_real(messages, settings, on_token, timeout) -> LLMResult:
    import requests
    model = pick_model_id()
    res = LLMResult(model_id=model, backend=BACKEND_REAL)
    resp = None
    for thinking_off in (True, False):
        t_start = time.time()
        resp = requests.post(_base() + "/chat/completions", json=_payload(model, messages, settings, thinking_off),
                             stream=True, timeout=(10, timeout))
        if resp.status_code == 400 and thinking_off:
            resp.close()
            continue
        break
    if resp.status_code >= 400:
        body = ""
        try:
            body = resp.text[:400]
        except Exception:  # noqa: BLE001
            pass
        low = body.lower()
        if any(k in low for k in ("context length", "context size", "n_ctx", "too many tokens", "exceed", "context window")):
            raise RuntimeError("LM Studio rejected the prompt: it exceeds the model's context length. Reload the "
                               "model with a larger context (8192 for long-context runs). Server said: " + body)
        raise RuntimeError(f"LM Studio HTTP {resp.status_code}: {body}")
    res.t_start = t_start
    parts: list[str] = []
    n_chunks = 0
    t_first = None
    usage = None
    reasoning_chars = 0
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices", []):
            d = ch.get("delta", {})
            if d.get("reasoning_content") or d.get("reasoning"):
                reasoning_chars += len(d.get("reasoning_content") or d.get("reasoning") or "")
            piece = d.get("content")
            if piece:
                if t_first is None:
                    t_first = time.time()
                parts.append(piece)
                n_chunks += 1
                if on_token:
                    on_token(piece)
            if ch.get("finish_reason"):
                res.finish_reason = ch["finish_reason"]
    t_end = time.time()
    t_first = t_first or t_end
    res.t_first, res.t_end = t_first, t_end
    res.ttft_s = res.prefill_s = t_first - res.t_start
    res.generate_s = t_end - t_first
    res.total_s = t_end - res.t_start
    res.text = strip_reasoning("".join(parts))
    if usage:
        res.tokens_in = int(usage.get("prompt_tokens", 0))
        res.tokens_out = int(usage.get("completion_tokens", 0))
        det = usage.get("completion_tokens_details") or {}
        res.reasoning_tokens = int(det.get("reasoning_tokens", 0) or 0)
        res.raw["usage"] = usage
    else:
        res.usage_estimated = True
        res.tokens_in = estimate_tokens(messages_text(messages))
        res.tokens_out = max(n_chunks, estimate_tokens(res.text))
    if reasoning_chars and not res.reasoning_tokens:
        res.reasoning_tokens = max(1, reasoning_chars // 4)
    return res


# --------------------------------------------------------------------------- fake
def _spin(seconds: float) -> None:
    """One-core busy loop so fake runs still produce a power signal (fake mode only)."""
    end = time.perf_counter() + seconds
    x = 1
    while time.perf_counter() < end:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF


def _fake_answer(messages: list[dict], fake_gold: str | None) -> str:
    full = messages_text(messages)
    h = int(hashlib.sha256(full.encode()).hexdigest(), 16)
    # judge the CONTEXT only (the question may mention the answer)
    ctx = full.split("Context:", 1)[-1].split("\n\nQuestion:", 1)[0]
    if fake_gold:
        if fake_gold.lower() in ctx.lower():
            return fake_gold   # simulated reader: answers correctly when the evidence is present
        return "NOT FOUND"     # ... and abstains when it is not (exercises the safety net)
    caps = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b", ctx)
    return caps[h % len(caps)] if caps else "NOT FOUND"


def _generate_fake(messages, settings, on_token, fake_gold) -> LLMResult:
    burn = os.getenv("QWATT_FAKE_BURN", "1") == "1"
    res = LLMResult(model_id=pick_model_id(), backend=BACKEND_FAKE)
    text_in = messages_text(messages)
    res.tokens_in = estimate_tokens(text_in, 4.0) + 8       # chat template overhead
    answer = _fake_answer(messages, fake_gold)
    words = answer.split(" ")
    res.t_start = time.time()
    prefill = FAKE_BASE_LATENCY_S + res.tokens_in / FAKE_PREFILL_TOK_S
    (_spin if burn else time.sleep)(prefill)
    res.t_first = time.time()
    parts = []
    n_tok = 0
    for i, w in enumerate(words):
        piece = (" " if i else "") + w
        time.sleep(1.4 / FAKE_DECODE_TOK_S)
        parts.append(piece)
        n_tok += max(1, int(round(len(piece) / 3.5)))
        if on_token:
            on_token(piece)
        if n_tok >= settings.max_tokens:
            break
    res.t_end = time.time()
    res.text = "".join(parts)
    res.tokens_out = min(n_tok, settings.max_tokens)
    res.ttft_s = res.prefill_s = res.t_first - res.t_start
    res.generate_s = res.t_end - res.t_first
    res.total_s = res.t_end - res.t_start
    res.finish_reason = "stop"
    return res


def generate(messages: list[dict], settings, on_token: Callable[[str], None] | None = None,
             fake_gold: str | None = None, timeout: float = 900.0) -> LLMResult:
    """Stream one chat completion. `fake_gold` is only used in FAKE mode (simulated reader)."""
    if is_fake():
        return _generate_fake(messages, settings, on_token, fake_gold)
    return _generate_real(messages, settings, on_token, timeout)


def loaded_context_length() -> int | None:
    """Best effort: context length the model is loaded with (LM Studio REST /api/v0/models). None if unknown."""
    if is_fake():
        return 8192
    try:
        import requests
        root = _base().rsplit("/v1", 1)[0]
        r = requests.get(root + "/api/v0/models", timeout=5)
        r.raise_for_status()
        mid = pick_model_id()
        for m in r.json().get("data", []):
            if m.get("id") == mid and m.get("loaded_context_length"):
                return int(m["loaded_context_length"])
    except Exception:  # noqa: BLE001
        pass
    return None

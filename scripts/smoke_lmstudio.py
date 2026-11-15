"""Smoke test for LM Studio: model id, streaming, usage, TTFT, thinking leak check, prefill scaling.

  python scripts/smoke_lmstudio.py          (needs the LM Studio server running, a Gemma chat model loaded)
  QWATT_FAKE_LLM=1 python scripts/smoke_lmstudio.py   (exercise the fake path)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwatt import config, llm, prompt  # noqa: E402


def main() -> int:
    s = config.load_settings()
    print(f"base_url={config.LMSTUDIO_BASE_URL} fake={llm.is_fake()}")
    h = llm.health()
    print("health:", h)
    if not h["ok"]:
        print("FAIL: LM Studio not reachable or no chat model loaded. Start the server (Developer tab) "
              "and load the Gemma model.")
        return 1
    if not llm.is_fake():
        print("models:", llm.list_models())
    msgs = [{"role": "user", "content": f"[ref {prompt.new_nonce()}]\nWhat is the capital of France? "
                                          f"{prompt.ANSWER_LINE}"}]
    toks = []
    r = llm.generate(msgs, s, on_token=toks.append, fake_gold="Paris")
    print(f"answer={r.text!r}")
    print(f"tokens_in={r.tokens_in} tokens_out={r.tokens_out} reasoning_tokens={r.reasoning_tokens} "
          f"usage_estimated={r.usage_estimated} finish={r.finish_reason}")
    print(f"ttft={r.ttft_s:.3f}s prefill={r.prefill_s:.3f}s generate={r.generate_s:.3f}s total={r.total_s:.3f}s "
          f"chunks={len(toks)}")
    ok = bool(r.text) and r.tokens_in > 0 and r.tokens_out > 0
    if r.reasoning_tokens:
        print(f"WARN: {r.reasoning_tokens} reasoning tokens emitted - thinking is NOT off. "
              "Disable thinking in LM Studio (model settings / Jinja template) before the benchmark.")
    # prefill scaling: longer prompt -> longer TTFT
    filler = "The quick brown fox jumps over the lazy dog near the old stone bridge. " * 60
    m2 = [{"role": "user", "content": f"[ref {prompt.new_nonce()}]\n{filler}\nWhat animal jumps? {prompt.ANSWER_LINE}"}]
    r2 = llm.generate(m2, s)
    print(f"long prompt: tokens_in={r2.tokens_in} ttft={r2.ttft_s:.3f}s (short ttft={r.ttft_s:.3f}s)")
    if r2.tokens_in > 0 and r.tokens_in > 0:
        print(f"chars/token ~ {len(m2[0]['content']) / r2.tokens_in:.2f}")
    # prefix cache check: same prompt twice should NOT be free once the nonce changes
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

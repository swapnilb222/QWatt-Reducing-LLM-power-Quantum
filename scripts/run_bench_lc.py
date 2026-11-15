"""Long-context energy benchmark: arms full_lc, qwatt_safe, qwatt_lc (resumable, randomized order).

  python scripts/run_bench_lc.py                    # all three arms on the lc_bench questions
  python scripts/run_bench_lc.py --n 2              # first 2 questions (G4-style check)
  python scripts/run_bench_lc.py --bench-id lc-XXXX # resume
  python scripts/run_bench_lc.py --arms full_lc,qwatt_safe
QWATT_FAKE_LLM=1 runs without LM Studio (rows tagged llm_backend='fake').
full_lc requests are ~4-5k tokens (45-60 s each on this CPU); LM Studio context must be >= 8192.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwatt import bench, config, llm  # noqa: E402

LC_ARMS = ("full_lc", "qwatt_safe", "qwatt_lc")


def lc_items(split: str = "lc_bench") -> list[dict]:
    from qwatt import longdoc
    for name in ("load_lc_split", "load_split", "load_lc"):
        fn = getattr(longdoc, name, None)
        if fn:
            return fn(split)
    raise RuntimeError("qwatt.longdoc has no split loader (expected load_lc_split(name))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(LC_ARMS))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default="lc_bench")
    ap.add_argument("--bench-id", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()
    from qwatt import longdoc
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    settings = longdoc.lc_settings() if hasattr(longdoc, "lc_settings") else config.load_settings()
    items = lc_items(a.split)
    if a.n:
        items = items[:a.n]
    ctx = llm.loaded_context_length()
    if ctx is not None and ctx < 6500:
        print(f"ERROR: LM Studio model is loaded with context length {ctx}; long documents need >= 8192. "
              "Reload the model with a larger context and rerun.")
        sys.exit(2)
    print(f"LM Studio loaded context length: {ctx if ctx else 'unknown (could not query)'}")

    def qwarm():
        from qwatt.selectors import get_selector
        p = longdoc.get_lc_pool(items[0].get("_id") or items[0].get("qid") or items[0].get("id"))
        get_selector("qwatt_lc")(p, settings.budget_tokens, settings)

    st = bench.run_bench(arms=arms, settings=settings, bench_id=a.bench_id, fresh=a.fresh, prefix="lc",
                         questions=items, get_pool=lambda qid: longdoc.get_lc_pool(qid),
                         do_warmup=not a.no_warmup, quantum_warm=qwarm)
    print(st)


if __name__ == "__main__":
    main()

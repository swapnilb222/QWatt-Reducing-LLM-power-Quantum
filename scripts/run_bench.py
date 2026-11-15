"""Run the energy benchmark (resumable).

  python scripts/run_bench.py                      # arms full,topk,qwatt on the 20 bench questions
  python scripts/run_bench.py --arms full,topk,greedy,qwatt --n 5
  python scripts/run_bench.py --bench-id bench-XXXX   # resume an earlier run (skips completed)
  python scripts/run_bench.py --fresh                 # new bench_id even with same config
Set QWATT_FAKE_LLM=1 to run without LM Studio (rows tagged llm_backend='fake').
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwatt import bench, config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(bench.DEFAULT_ARMS))
    ap.add_argument("--n", type=int, default=None, help="only the first N bench questions")
    ap.add_argument("--bench-id", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    st = bench.run_bench(arms=arms, n_questions=a.n, bench_id=a.bench_id, fresh=a.fresh,
                         settings=config.load_settings(), do_warmup=not a.no_warmup)
    print(st)


if __name__ == "__main__":
    main()

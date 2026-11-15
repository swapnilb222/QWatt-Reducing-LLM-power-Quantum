"""Build long-context documents, fixed splits, the committed subset, and cached pools.

Run:  .venv/Scripts/python scripts/prepare_lc.py        (needs data/raw/squad_v1_dev.parquet the first time)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from qwatt import longdoc  # noqa: E402


def main() -> None:
    if not longdoc.SUBSET_JSON.exists() or not longdoc.LC_SPLITS_JSON.exists():
        sp = longdoc.make_lc_splits()
        print(f"docs={sp['n_docs_total']} dev={len(sp['lc_dev'])} test={len(sp['lc_test'])} bench={len(sp['lc_bench'])}")
    s = longdoc.lc_settings()
    for split in ("lc_dev", "lc_test", "lc_bench"):
        qs = longdoc.load_lc_split(split)
        pools = [longdoc.get_lc_pool(q["id"], s) for q in qs]
        n = [p.n for p in pools]
        tok = [p.full_tokens for p in pools]
        gold = [p.n_gold_total for p in pools]
        rank = []
        for p in pools:
            r = p.relevance
            g = [i for i, x in enumerate(p.sentences) if x.is_gold]
            if g:
                rank.append(int((r > r[g[0]]).sum()))
        edges = [int(np.count_nonzero(np.triu(p.red + p.br, 1))) for p in pools]
        print(f"{split}: q={len(qs)} docs={len({p.meta['doc_id'] for p in pools})} n_sent={np.mean(n):.0f} "
              f"[{min(n)}-{max(n)}] full_tokens={np.mean(tok):.0f} [{min(tok)}-{max(tok)}] gold_found={np.mean(np.array(gold) > 0):.2f} "
              f"gold_rank_median={np.median(rank):.0f} top10={np.mean(np.array(rank) < 10):.2f} edges={np.mean(edges):.0f}")


if __name__ == "__main__":
    main()

"""Make fixed splits, the compact committed subset, and cached pools; report pool recall.

Run:  .venv/Scripts/python scripts/prepare_data.py
Needs data/raw/hotpot_dev_distractor.parquet (HF mirror) or data/raw/hotpot_dev_distractor_v1.json
only the first time; afterwards everything reads data/hotpot_subset.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from qwatt import config, data  # noqa: E402


def main() -> None:
    settings = config.load_settings()
    if not data.SUBSET_JSON.exists() or not config.SPLITS_JSON.exists():
        s = data.make_splits(settings.seed)
        print(f"splits: dev={len(s['dev'])} test={len(s['test'])} bench={len(s['bench'])} of {s['n_total']}")
    for split in ("dev", "test"):
        items = data.load_split(split)
        rec, edges_br, edges_red, toks, full = [], [], [], [], []
        for it in items:
            p = data.get_pool(it["_id"], settings)
            rec.append(p.pool_recall)
            iu = np.triu_indices(p.n, 1)
            edges_br.append(int((p.br[iu] > 0).sum()))
            edges_red.append(int((p.red[iu] > 0).sum()))
            toks.append(p.tokens.sum())
            full.append(p.full_tokens)
        print(f"{split}: n={len(items)} pool_recall={np.mean(rec):.3f} full_recall_rate={np.mean(np.array(rec) == 1):.2f} "
              f"bridge_edges={np.mean(edges_br):.1f} red_edges(theta={settings.red_threshold})={np.mean(edges_red):.1f} "
              f"pool_tokens={np.mean(toks):.0f} full_tokens={np.mean(full):.0f}")


if __name__ == "__main__":
    main()

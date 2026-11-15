"""Quantum cost vs document length: QAOA p=1 on Quantum Rings with REAL long-context QUBOs.

One document (the 205-sentence scaling document in data/squad_lc_subset.json), truncated to its first n sentences,
n = 16, 24, 50, 100, 150, 200. For each config a fresh subprocess (timeout) runs the full charged select path
(QUBO build -> circuit -> decode, i.e. sel_qwatt_lc with reference=False) and measures:
  wall time per stage, net joules = RAPL package energy counter difference minus idle_watts * dt,
  edges, depth, gate count, shots, threads, feasible-shot fraction, gold kept.
Next to it: prefill joules saved by sending the selection instead of the first-n-sentence document,
eps_in * (doc_tokens - selected_tokens), eps_in from results/calibration.json.

Run:  .venv/Scripts/python scripts/scaling_qr.py            (all configs, ~5 min)
      .venv/Scripts/python scripts/scaling_qr.py --one N SHOTS THREADS IDLE_W   (internal)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIZES = [16, 24, 50, 100, 150, 200]
BASE = (256, 2)                                    # shots, threads for the size sweep (the operating point)
EXTRA = [(n, sh, th) for n in (150, 200) for sh, th in ((512, 1), (512, 2), (1024, 1), (1024, 2), (2048, 2))]
TIMEOUT_S = 240
REPS = 2


def one(n: int, shots: int, threads: int, idle_w: float) -> dict:
    os.environ["QWATT_QR_THREADS"] = str(threads)
    import numpy as np  # noqa: F401

    from qwatt import config, longdoc, qaoa_qr
    from qwatt.power import PdhRapl
    from qwatt.selectors import sel_qwatt_lc

    s = longdoc.lc_settings()
    s.shots = shots
    sp = json.loads(longdoc.LC_SPLITS_JSON.read_text())
    pool = longdoc.truncate_pool(longdoc.get_lc_pool(sp["scaling_qid"], s), n)
    doc_tokens = int(pool.tokens.sum())
    B = min(s.budget_tokens, doc_tokens)
    qaoa_qr.get_backend(n)                          # provider/backend init is per process, not per query
    rapl = PdhRapl()
    reps = []
    for _ in range(REPS):
        time.sleep(1.5)                             # settle
        e0 = rapl.read()["pkg_joules_cum"]
        t0 = time.perf_counter()
        idx, d = sel_qwatt_lc(pool, B, s, reference=False)
        dt = time.perf_counter() - t0
        time.sleep(0.6)                             # let the energy counter catch up
        e1 = rapl.read()["pkg_joules_cum"]
        net = (e1 - e0) - idle_w * (dt + 0.6)
        reps.append({"wall_s": dt, "net_joules": net, "gross_joules": e1 - e0, "circuit_wall_s": d["circuit_wall_s"],
                     "qubo_build_s": d["qubo_build_s"], "decode_s": d["decode_s"], "tokens": d["tokens"],
                     "feasible_shot_frac": d["feasible_shot_frac"], "decode_source": d["decode"]["source"],
                     "gold_kept": d["gold_kept"], "mean_shot_tokens": d["mean_shot_tokens"]})
    best = min(reps, key=lambda r: r["net_joules"])
    eps_in = config.eps_in_default()
    return {"n": n, "shots": shots, "threads": threads, "edges": d["n_edges"], "depth": d["depth"],
            "gate_count": d["gate_count"], "backend": d["backend"], "budget": B, "doc_tokens": doc_tokens,
            "selected_tokens": best["tokens"], "prefill_joules_saved": eps_in * (doc_tokens - best["tokens"]),
            "eps_in": eps_in, "reps": reps,
            "wall_s_median": sorted(r["wall_s"] for r in reps)[len(reps) // 2],
            "net_joules_median": sorted(r["net_joules"] for r in reps)[len(reps) // 2]}


def measure_idle(seconds: float = 5.0) -> float:
    from qwatt.power import PdhRapl

    r = PdhRapl()
    time.sleep(1.0)
    a = r.read()["pkg_joules_cum"]
    time.sleep(seconds)
    return (r.read()["pkg_joules_cum"] - a) / seconds


def main() -> None:
    from qwatt import config

    idle = measure_idle()
    print(f"idle {idle:.2f} W", flush=True)
    configs = [(n, *BASE) for n in SIZES] + EXTRA
    rows = []
    for n, sh, th in configs:
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, __file__, "--one", str(n), str(sh), str(th), str(idle)], cwd=ROOT,
                               capture_output=True, text=True, timeout=TIMEOUT_S)
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            row = json.loads(line[-1]) if line else {"n": n, "shots": sh, "threads": th, "error": p.stderr[-800:]}
        except subprocess.TimeoutExpired:
            row = {"n": n, "shots": sh, "threads": th, "error": f"timeout after {TIMEOUT_S}s"}
        row["process_wall_s"] = time.perf_counter() - t0
        rows.append(row)
        if "error" in row:
            print(f"n={n} shots={sh} threads={th}: ERROR {row['error'][:200]}", flush=True)
        else:
            print(f"n={n:3d} shots={sh:4d} thr={th} edges={row['edges']:3d} depth={row['depth']:2d} "
                  f"wall={row['wall_s_median']:.2f}s net={row['net_joules_median']:.1f}J saved={row['prefill_joules_saved']:.0f}J "
                  f"gold={[r['gold_kept'] for r in row['reps']]}", flush=True)
    import numpy as np

    from qwatt import longdoc

    sp = json.loads(longdoc.LC_SPLITS_JSON.read_text())
    s = longdoc.lc_settings()
    bench_n = [longdoc.get_lc_pool(q, s).n for q in sp["lc_bench"]]
    scal = longdoc.get_lc_pool(sp["scaling_qid"], s)
    tps = float(scal.tokens.mean())
    ok = [r for r in rows if "error" not in r]
    base = [r for r in ok if (r["shots"], r["threads"]) == BASE]
    keep = [r["selected_tokens"] / r["doc_tokens"] for r in base if r["n"] >= 100]
    out = {"eps_in": config.eps_in_default(), "tokens_per_sentence": tps,
           "keep_frac": float(np.mean(keep)) if keep else None,
           "points": [{"n_qubits": r["n"], "n_edges": r["edges"], "wall_s": r["wall_s_median"],
                       "joules": r["net_joules_median"], "shots": r["shots"], "threads": r["threads"],
                       "depth": r["depth"], "gate_count": r["gate_count"], "prefill_joules_saved": r["prefill_joules_saved"]}
                      for r in base],
           "docs": {"n_min": int(min(bench_n)), "n_max": int(max(bench_n)), "n_median": float(np.median(bench_n))},
           "energy_grid": [{"n_qubits": r["n"], "shots": r["shots"], "threads": r["threads"], "wall_s": r["wall_s_median"],
                            "joules": r["net_joules_median"]} for r in ok if (r["shots"], r["threads"]) != BASE],
           "idle_watts": idle, "base": {"shots": BASE[0], "threads": BASE[1]}, "reps": REPS,
           "document": "scaling doc (first n sentences of one SQuAD article)", "rows": rows,
           "measured_at": time.time(), "note": "net J = RAPL package counter difference - idle_watts * dt, "
                                               "charged select path only (QUBO build + circuit + decode)"}
    (config.RESULTS_DIR / "scaling.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--one"]:
        n, sh, th, idle = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
        print(json.dumps(one(n, sh, th, idle)))
    else:
        main()

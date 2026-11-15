"""Calibrate the LLM energy model  E = e0 + eps_in*Tin + eps_out*Tout  on this device.

Prompts of ~100/300/600/1200 tokens (HotpotQA filler), 2 repeats each (repeat 2 asks for a longer
answer so Tout varies and the 3-parameter fit is identifiable), fixed output cap = settings.max_tokens.
Writes results/calibration.json (real LM Studio) or results/calibration_fake.json (QWATT_FAKE_LLM=1),
never mixing the two.

  python scripts/calibrate.py [--sizes 100,300,600,1200] [--repeats 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from qwatt import bench, config, llm, power, prompt, store  # noqa: E402

FALLBACK_TEXT = (
    "The river rises in the northern hills and flows south through three provinces before reaching the sea. "
    "Its basin supports agriculture, fishing and a number of small towns, several of which were founded in "
    "the twelfth century. The largest city on its banks hosts a university, a cathedral and a regional museum. "
    "In the nineteenth century the construction of canals linked it to neighbouring waterways, which boosted trade "
    "in timber, grain and salt. Historians disagree on the exact date of the first stone bridge, although a charter "
    "from 1204 mentions a crossing. The surrounding wetlands are home to migratory birds and are protected. "
) * 40


def filler_corpus() -> str:
    try:
        from qwatt import data
        parts = []
        for it in data.load_split("dev"):
            for title, sents in it["context"]:
                parts.append(f"{title}: " + " ".join(s.strip() for s in sents))
        return "\n\n".join(parts)
    except Exception:  # noqa: BLE001
        return FALLBACK_TEXT


def make_prompt(corpus: str, chars: int, offset: int, nonce: str, long_answer: bool) -> list[dict]:
    chunk = corpus[offset:offset + chars]
    if len(chunk) < chars:
        chunk = (corpus + corpus)[offset:offset + chars]
    ans = "Answer in about 40 words." if long_answer else prompt.ANSWER_LINE
    user = (f"[ref {nonce}]\n{prompt.INSTRUCTION}\n\nContext:\n{chunk}\n\n"
            f"Question: What is the main subject of the context?\n{ans}")
    return [{"role": "user", "content": user}]


def fit(points: list[dict]) -> dict:
    tin = np.array([p["tokens_in"] for p in points], float)
    tout = np.array([p["tokens_out"] for p in points], float)
    y = np.array([p["joules_net"] for p in points], float)
    full_rank = len(points) >= 4 and np.ptp(tout) > 2 and np.ptp(tin) > 10
    note = ""
    if full_rank:
        A = np.column_stack([np.ones_like(tin), tin, tout])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        e0, eps_in, eps_out = (float(c) for c in coef)
        pred = A @ coef
    else:
        A = np.column_stack([np.ones_like(tin), tin])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        e0, eps_in = float(coef[0]), float(coef[1])
        eps_out = float(np.mean([p.get("generate_net", 0.0) / max(1, p["tokens_out"]) for p in points]))
        pred = A @ coef + eps_out * tout
        note = "Tout not identifiable; eps_out from generate-phase energy per token"
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"e0": e0, "eps_in": eps_in, "eps_out": eps_out, "r2": r2, "fit_note": note,
            "residuals": [float(v) for v in (y - pred)]}


def nonlinearity(points: list[dict]) -> dict:
    """Does energy bend with prompt length? Compare linear vs quadratic-in-Tin fits (net joules) and
    report prefill-phase joules per input token at each size (flat => linear)."""
    tin = np.array([p["tokens_in"] for p in points], float)
    tout = np.array([p["tokens_out"] for p in points], float)
    y = np.array([p["joules_net"] for p in points], float)

    def r2_of(A):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef
        tot = float(np.sum((y - y.mean()) ** 2))
        return (1 - float(np.sum(res ** 2)) / tot if tot > 0 else 0.0), coef

    one = np.ones_like(tin)
    r_lin, c_lin = r2_of(np.column_stack([one, tin, tout]))
    r_quad, c_quad = r2_of(np.column_stack([one, tin, tout, (tin / 1000.0) ** 2]))
    by_size: dict[int, list[float]] = {}
    for p in points:
        by_size.setdefault(int(p.get("size") or round(p["tokens_in"], -2)), []).append(p["prefill_net"] / max(1, p["tokens_in"]))
    jpt = {k: round(float(np.mean(v)), 4) for k, v in sorted(by_size.items())}
    quad = float(c_quad[3])
    nonlinear = bool((r_quad - r_lin) > 0.03 and quad > 0)
    return {"r2_linear": float(r_lin), "r2_quadratic": float(r_quad), "quad_coef": quad,
            "quad_coef_note": "J per (1000 tokens)^2 added to the linear model",
            "nonlinear": nonlinear, "j_per_token_by_size": jpt,
            "quad_params": [float(c) for c in c_quad]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100,300,600,1200")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--long", action="store_true",
                    help="long-context sizes 300,1000,2500,4500 -> results/calibration_lc.json (needs 8k context)")
    a = ap.parse_args()
    if a.long and a.sizes == "100,300,600,1200":
        a.sizes = "300,1000,2500,4500"
    sizes = [int(x) for x in a.sizes.split(",")]
    settings = config.load_settings()
    fake = llm.is_fake()
    stem = "calibration_lc" if a.long else "calibration"
    out_path = config.RESULTS_DIR / (f"{stem}_fake.json" if fake else f"{stem}.json")
    log = bench.BenchLog("calibrate", echo=True)
    con = store.connect()
    power.ensure_sampler()
    src = power.sampler_source(con)
    log(f"calibrate: fake={fake} energy_source={src} sizes={sizes} repeats={a.repeats} -> {out_path.name}")
    bench.warmup(settings, log)
    idle0 = bench.measure_idle_baseline(con, 5.0)
    log(f"idle baseline: {idle0:.2f} W")
    corpus = filler_corpus()
    cpt0 = settings.chars_per_token
    pts, chars_list = [], []
    plan = [(s, r) for r in range(a.repeats) for s in sizes]
    for k, (size, rep) in enumerate(plan):
        bench.cooldown(con, idle0, settings, log)
        idle_pre = bench.measure_idle_baseline(con, 1.5) or idle0
        msgs = make_prompt(corpus, int(size * cpt0), offset=(k * 977) % max(1, len(corpus) // 2),
                           nonce=prompt.new_nonce(), long_answer=(rep % 2 == 1))
        run_id = f"cal-{size}-{rep}-{uuid.uuid4().hex[:6]}"
        res = llm.generate(msgs, settings)
        store.insert_marker(con, run_id, "prefill", res.t_start, res.t_first)
        store.insert_marker(con, run_id, "generate", res.t_first, res.t_end)
        store.wait_for_samples(con, res.t_end + 0.05, 1.5)
        time.sleep(0.1)
        pe = store.phase_energy(run_id, con=con, idle_watts=idle_pre)
        tot = pe.get("total")
        if not tot or tot["gross_j"] is None:
            log(f"  size={size} rep={rep}: no energy data, skipped")
            continue
        pts.append({"tokens_in": res.tokens_in, "tokens_out": res.tokens_out,
                    "joules_net": tot["net_j"], "joules_gross": tot["gross_j"], "seconds": tot["seconds"],
                    "prefill_net": pe["prefill"]["net_j"], "generate_net": pe["generate"]["net_j"],
                    "ttft_s": res.ttft_s, "size": size})
        chars_list.append(len(msgs[0]["content"]))
        log(f"  size~{size} rep={rep}: Tin={res.tokens_in} Tout={res.tokens_out} ttft={res.ttft_s:.2f}s "
            f"net={tot['net_j']:.1f}J gross={tot['gross_j']:.1f}J in {tot['seconds']:.1f}s")
    if len(pts) < 3:
        log("not enough points to fit; aborting")
        sys.exit(1)
    f = fit(pts)
    nl = nonlinearity(pts)
    tin = np.array([p["tokens_in"] for p in pts], float)
    ch = np.array(chars_list, float)
    cpt = float(np.polyfit(tin, ch, 1)[0]) if np.ptp(tin) > 10 else float(ch.sum() / tin.sum())
    cpt_ratio = float(ch.sum() / tin.sum())
    idle1 = bench.measure_idle_baseline(con, 3.0)
    idle = float(np.mean([v for v in (idle0, idle1) if v is not None]))
    model_id = llm.pick_model_id()
    cal = {
        "e0": f["e0"], "eps_in": f["eps_in"], "eps_out": f["eps_out"], "idle_watts": idle, "r2": f["r2"],
        "chars_per_token": cpt, "chars_per_token_ratio": cpt_ratio,
        "points": [{k: p[k] for k in ("tokens_in", "tokens_out", "joules_net", "joules_gross", "seconds")} for p in pts],
        "model_id": model_id, "llm_backend": llm.BACKEND_FAKE if fake else llm.BACKEND_REAL,
        "energy_source": src, "measured_at": datetime.now().isoformat(timespec="seconds"),
        "eps_out_direct": float(np.mean([p["generate_net"] / max(1, p["tokens_out"]) for p in pts])),
        "fit_note": f["fit_note"], "residuals": f["residuals"],
        "max_tokens": settings.max_tokens, "output_unit": "joules (net of idle)",
        "long_context": bool(a.long), "nonlinearity": nl,
    }
    out_path.write_text(json.dumps(cal, indent=1))
    log(f"linearity: r2_linear={nl['r2_linear']:.3f} r2_quadratic={nl['r2_quadratic']:.3f} "
        f"quad_coef={nl['quad_coef']:.3e} nonlinear={nl['nonlinear']}  J/token by size: {nl['j_per_token_by_size']}")
    log(f"fit: e0={cal['e0']:.2f} J  eps_in={cal['eps_in']:.4f} J/tok  eps_out={cal['eps_out']:.4f} J/tok  "
        f"r2={cal['r2']:.3f}  chars/token={cpt:.2f}  idle={idle:.2f} W -> {out_path}")
    if fake:
        log(f"NOTE: FAKE LLM calibration written to {out_path.name} (not used by the pipeline)")


if __name__ == "__main__":
    main()

"""Q-Watt dashboard (plan section 10).

    .venv/Scripts/streamlit run app/dashboard.py                       # real data: results/qwatt.sqlite
    QWATT_DASH_FIXTURE=1 .venv/Scripts/streamlit run app/dashboard.py  # SYNTHETIC preview data

Modes: Benchmark (browse stored runs, replay a race) and Live (run a question now: Q-Watt lane first,
then Full context, sequentially).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import charts as CH  # noqa: E402
from app import components as C  # noqa: E402
from app import data_access as DA  # noqa: E402
from app import live as LV  # noqa: E402
from app import stats as S  # noqa: E402
from app import theme as T  # noqa: E402
from app import lc_view as LCV  # noqa: E402

st.set_page_config(page_title="Q-Watt · quantum token selection", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")
st.html(T.CSS)

FIX = DA.is_fixture()


def html(s: str) -> None:
    if s:
        st.html(s)


def plot(fig, key: str) -> None:
    st.plotly_chart(fig, theme=None, config=T.PLOTLY_CONFIG, key=key)


# ---------------------------------------------------------------- cached loaders
@st.cache_data(ttl=3, show_spinner=False)
def load_runs(_db: str) -> pd.DataFrame:
    return DA.runs_df()


@st.cache_data(ttl=10, show_spinner=False)
def load_qaoa(_db: str) -> pd.DataFrame:
    return DA.qaoa_df()


@st.cache_data(ttl=30, show_spinner=False)
def load_json(name: str, _db: str):
    return DA.load_result(name)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def _cost_dist_cached(key: str, h: tuple, j_edges: tuple, samples_key: tuple):
    return S.cost_distribution({"h": list(h), "J_edges": [list(e) for e in j_edges],
                                "samples": [{"cost": c, "count": n} for c, n in samples_key]})


def cost_dist(diag: dict):
    try:
        d = _cost_dist_cached(str(diag.get("best_bits")), tuple(diag.get("h") or ()),
                              tuple(tuple(e) for e in (diag.get("J_edges") or ())),
                              tuple((s.get("cost"), int(s.get("count", 0))) for s in (diag.get("samples") or [])))
        if d is not None:
            d = dict(d)
            d["exact_cost"] = S.fnum(diag.get("exact_cost"))
        return d
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def load_pool(qid: str, _db: str):
    return DA.get_pool_dict(qid)


@st.cache_data(ttl=60, show_spinner=False)
def load_demo_questions(_db: str):
    return DA.demo_question_list()


@st.cache_data(ttl=60, show_spinner=False)
def load_tiger():
    res, msg = DA.tiger_scoreboard()
    if res is not None and not isinstance(res, pd.DataFrame):
        try:
            res = pd.DataFrame(res)
        except Exception:
            res, msg = None, f"unrecognized scoreboard type {type(res).__name__}"
    return res, msg


def modules_ok(*mods: str) -> list[str]:
    import importlib.util
    missing = []
    for m in mods:
        try:
            if importlib.util.find_spec(m) is None:
                missing.append(m)
        except Exception:
            missing.append(m)
    return missing


DB = str(DA.db_path())
runs = load_runs(DB)
cal = load_json("calibration", DB)
hyper = load_json("hyperparams", DB) or {}
default_budget = int(hyper.get("budget_tokens") or 120)
if not FIX:
    try:
        from qwatt.config import load_settings
        default_budget = int(load_settings().budget_tokens)
    except Exception:
        pass

# ---------------------------------------------------------------- mode & top bar
qp_mode = st.query_params.get("mode", "bench")
if "mode" not in st.session_state:
    st.session_state["mode"] = "Live" if qp_mode == "live" else "Benchmark"

top_l, top_r = st.columns([5.2, 1.3], vertical_alignment="center")
with top_r:
    mode_label = st.segmented_control("Mode", ["Live", "Benchmark"], key="mode", label_visibility="collapsed", required=True,
                                      width="stretch")
mode = "live" if mode_label == "Live" else "bench"
if st.query_params.get("mode") != mode:
    st.query_params["mode"] = mode

pills: list[tuple[str, str]] = []
status = DA.db_status()
if FIX:
    pills.append(("warn", "data: SYNTHETIC fixture"))
elif status["exists"]:
    nr = status["tables"].get("runs")
    pills.append(("ok", f"SQLite · {status['name']} · {nr if nr is not None else 0} runs"))
else:
    pills.append(("warn", "no results/qwatt.sqlite yet"))
hb = DA.heartbeat()
if hb and hb.get("ts") and time.time() - float(hb["ts"]) < 3 and not FIX:
    pills.append(("ok", f"power sampler live · {hb.get('source')}"))
elif hb and not FIX:
    pills.append(("", f"sampler idle · {hb.get('source')}"))
if not FIX:
    try:
        from qwatt import config as _cfg
        if _cfg.FAKE_LLM:
            pills.append(("warn", "LLM: FAKE (QWATT_FAKE_LLM=1)"))
        pills.append(("ok" if _cfg.TIGER_DATABASE_URL else "", "Tiger Data " + ("configured" if _cfg.TIGER_DATABASE_URL else "off")))
    except Exception:
        pass
with top_l:
    html(C.topbar(mode, pills))

if FIX:
    html(C.synthetic_banner())

# ---------------------------------------------------------------- view switch (long-context hero vs 16-qubit check)
VIEW_LC, VIEW_16 = "Long document · up to 200 qubits", "Verified exactly at 16 qubits"
if "view" not in st.session_state:
    st.session_state["view"] = VIEW_16 if st.query_params.get("view") == "16" else VIEW_LC
st.segmented_control("View", [VIEW_LC, VIEW_16], key="view", label_visibility="collapsed", required=True)
VIEW = "16" if st.session_state.get("view") == VIEW_16 else "lc"
if st.query_params.get("view") != VIEW:
    st.query_params["view"] = VIEW

flag_list = []
if not runs.empty:
    if "llm_backend" in runs and (runs["llm_backend"] == "fake").any():
        n = int((runs["llm_backend"] == "fake").sum())
        flag_list.append(f"{n} run(s) used the FAKE LLM backend (llm_backend='fake') — not a real model")
    if "energy_source" in runs and (runs["energy_source"] == "modeled").any():
        n = int((runs["energy_source"] == "modeled").sum())
        flag_list.append(f"{n} run(s) have MODELED energy (energy_source='modeled') — not sensor readings")
if cal is None:
    flag_list.append("calibration.json missing — J/token uses a placeholder")
elif cal.get("energy_source") == "modeled" or cal.get("llm_backend") == "fake":
    flag_list.append(f"calibration from {cal.get('energy_source')} energy / {cal.get('llm_backend')} LLM")
html(C.flags(flag_list))

# ---------------------------------------------------------------- 01 hero
bids = S.bench_ids(runs, "16")
bench_sel = None
if len(bids) > 1:
    bench_sel = st.session_state.get("bench_id_sel", bids[0])
    if bench_sel not in bids:
        bench_sel = bids[0]
        st.session_state["bench_id_sel"] = bench_sel
bench = S.bench_runs(runs, bench_sel if bench_sel else (bids[0] if bids else None))
h = S.hero(bench)
n_q = bench["qid"].nunique() if not bench.empty else 0
if VIEW == "16":
    html(C.section("01", "Headline", f"benchmark · {n_q} questions" + (f" · batch {bids[0] if not bench_sel else bench_sel}" if bids else "")
               + " · energy = CPU package joules, net of idle"))
if VIEW == "16" and bench.empty:
    html(C.empty("No benchmark runs yet",
                 "Run <code>scripts/run_bench.py</code>; the headline fills in from <code>runs</code> where "
                 "<code>mode='bench'</code>. Set <code>QWATT_DASH_FIXTURE=1</code> to preview with synthetic data."))
elif VIEW == "16":
    html(C.hero(h, cal, S.correctness_source(bench), n_q))
if VIEW == "16" and len(bids) > 1:
    st.selectbox("Benchmark batch", bids, key="bench_id_sel")


# ---------------------------------------------------------------- lane views
def correct_label(row: dict) -> str | None:
    if row is None:
        return None
    if S.fnum(row.get("referee_correct")) is not None:
        return "Gemini referee"
    if S.fnum(row.get("em")) is not None or S.fnum(row.get("f1")) is not None:
        return "EM / F1"
    return None


def view_from_row(row: dict, clip_elapsed: float | None = None) -> dict:
    t0 = float(row["started_at"])
    t1 = float(row["ended_at"]) if S.fnum(row.get("ended_at")) is not None else t0 + (S.fnum(row.get("total_s")) or 0)
    markers = DA.phase_markers(row["run_id"])
    samples = DA.power_window(t0, t1, pad=0.5)
    v = {"arm": row["arm"], "row": row, "t0": t0, "t_now": t1, "markers": markers, "samples": samples,
         "idle": S.fnum(row.get("idle_watts")), "status": "done", "phase": "done",
         "joules": S.joules(row), "joules_note": "net of idle · measured" + (
             f" · gross {row['joules_gross']:,.0f} J" if S.fnum(row.get("joules_gross")) is not None else ""),
         "answer": row.get("answer"),
         "streaming": False, "duration": t1 - t0,
         "escalated": bool(S.fnum(row.get("escalated")) or any(str(k).startswith("escalate") for k in markers))}
    if v["joules"] is not None and v["joules"] <= 0 and S.fnum(row.get("joules_gross")) is not None:
        v["joules"] = float(row["joules_gross"])
        v["joules_note"] = "gross · net-of-idle invalid for this run (idle baseline glitch)"
    if clip_elapsed is not None and clip_elapsed < (t1 - t0):
        t_now = t0 + max(0.0, clip_elapsed)
        v["t_now"] = t_now
        v["status"] = "running"
        ph = None
        for p in ("select", "prefill", "generate", "escalate_prefill", "escalate_generate"):
            if p in markers and markers[p][0] <= t_now:
                ph = p
        v["escalated"] = bool(ph and ph.startswith("escalate"))
        v["phase"] = ph or ("prefill" if row["arm"] in ("full", "full_lc") else "select")
        v["markers"] = {p: (a, b if b is not None and b <= t_now else None) for p, (a, b) in markers.items() if a <= t_now}
        v["joules"] = S.integrate_joules(samples, t0, t_now, v["idle"])
        v["joules_note"] = "replay · counting…"
        ans = row.get("answer") or ""
        gk = "escalate_generate" if v["phase"] == "escalate_generate" else "generate"
        if v["phase"] in ("generate", "escalate_generate") and gk in markers:
            g0, g1 = markers[gk]
            frac = (t_now - g0) / max(1e-6, (g1 or t1) - g0)
            words = ans.split(" ")
            v["answer"] = " ".join(words[: int(len(words) * min(1, frac))])
        else:
            v["answer"] = ""
        v["streaming"] = True
        v["row"] = {**row, "tokens_out": None, "ttft_s": row.get("ttft_s") if v["phase"] == "generate" else None,
                    "total_s": t_now - t0}
    return v


def view_from_lane(lane: LV.Lane, idle_default: float | None) -> dict:
    now = time.time()
    t0 = lane.t0 or now
    t_now = lane.t1 or now
    row = lane.row or {}
    samples = DA.power_window(t0, t_now, pad=0.5) if lane.t0 else pd.DataFrame()
    idle = S.fnum(row.get("idle_watts")) if row else None
    idle = idle if idle is not None else idle_default
    markers = {p: (m[0], m[1]) for p, m in lane.markers.items()}
    if lane.status == "done" and row:
        J, note = S.joules(row), "net of idle · measured"
        if S.fnum(row.get("joules_gross")) is not None:
            note += f" · gross {row['joules_gross']:,.0f} J · idle {S.fnum(row.get('idle_watts')) or 0:.1f} W"
    elif lane.t0:
        J, note = S.integrate_joules(samples, t0, t_now, idle), "live · net of idle"
    else:
        J, note = None, "waiting for its turn"
    if lane.status == "error":
        note = "error"
    shown = dict(row) if row else {}
    if not row and lane.t0:
        shown = {"total_s": t_now - t0}
    return {"arm": lane.arm, "row": shown, "t0": t0, "t_now": t_now, "markers": markers, "samples": samples,
            "idle": idle, "status": lane.status, "phase": lane.phase, "joules": J, "joules_note": note,
            "answer": row.get("answer") if (lane.status == "done" and row.get("answer")) else lane.text,
            "streaming": lane.status == "running" and str(lane.phase).endswith("generate"),
            "duration": (t_now - t0) if lane.t0 else 0.0, "error": lane.error, "ref_status": lane.ref_status,
            "escalated": bool(lane.escalated or S.fnum(row.get("escalated")))}


def render_lane(v: dict, x_max: float, y_max: float, gold: str | None, key: str) -> None:
    with st.container(border=True, key=f"lane_{key}"):
        tag = {"full": "all retrieved paragraphs", "qwatt": "QAOA on Quantum Rings", "topk": "highest relevance, same budget",
               "full_lc": "the whole document", "qwatt_safe": "QAOA on Quantum Rings + safety net",
               "qwatt_lc": "QAOA selection, no safety net"}
        jn = v["joules_note"] + (" · incl. safety-net read" if v.get("escalated") and v["status"] == "done" else "")
        html(C.lane_header(v["arm"], v["joules"], jn, tag.get(v["arm"], "")))
        html(C.phase_chips(v["arm"], v["phase"], v["status"], escalated=bool(v.get("escalated"))))
        if v["status"] == "waiting":
            html(C.empty("Waiting", "This lane starts after the previous one finishes (one LLM call at a time on a 15 W laptop)."))
        else:
            plot(CH.watt_trace(v["samples"], v["t0"], v["t_now"], T.ARM_COLOR.get(v["arm"], T.INK_2), v["idle"],
                               v["markers"], x_max, y_max, live=v["status"] == "running",
                               quantum_band=v["arm"] in ("qwatt", "qwatt_safe", "qwatt_lc")), key=f"trace_{key}")
        html(C.lane_stats(v["row"] or {}))
        ok = S.correct(v["row"]) if v["status"] == "done" and v["row"] else None
        if v["status"] == "done" and v.get("ref_status") == "pending":
            badge = '<span class="qw-badge pend">⏳ asking the Gemini referee…</span>'
        else:
            badge = (C.referee_badge(ok, (v["row"] or {}).get("referee_reason"), correct_label(v["row"]))
                     if v["status"] == "done" else "")
        if v.get("error"):
            html(C.empty("Lane failed", C.esc(v["error"])))
        html(C.answer_box(v["answer"], v["streaming"], badge, gold if v["status"] == "done" else None))


def shared_axes(views: list[dict]) -> tuple[float, float]:
    durs = [v["duration"] for v in views if v["status"] != "waiting"]
    x_max = max(durs + [4.0]) * 1.04
    ys = [float(v["samples"]["pkg_watts"].max()) for v in views if v["samples"] is not None and not v["samples"].empty]
    y_max = max(ys + [10.0]) * 1.18
    return x_max, y_max


# ---------------------------------------------------------------- 02/03/04 race + lens + dial
def lens_and_dial(qid: str | None, pool: dict | None, diag: dict | None, lens_state: str, kept_default: list[int],
                  qrow: dict | None, frow: dict | None, key: str, charts: bool = True) -> None:
    html(C.section("03", "Quantum lens", "one qubit per candidate sentence"))
    html(C.caption())
    if pool is None and diag is None:
        html(C.empty("No circuit to show yet", "Pick a question with a Q-Watt run, or start a live run."))
        return
    samples = (diag or {}).get("samples") or []
    # ---- budget dial (re-decodes the stored shots; no new circuit)
    budget = default_budget
    chosen = None
    decoder = ""
    kept = kept_default
    dial_on = bool(samples) and lens_state == "done"
    if dial_on:
        toks_all = [int(s.get("tokens", 0)) for s in samples]
        pool_tok = sum(int(s.get("tokens") or 0) for s in (pool or {}).get("sentences", [])) or max(toks_all)
        lo = max(5, int(min(toks_all)))
        hi = max(lo + 10, int(pool_tok))
        dkey = f"dial_{key}_{qid}"
        if dkey not in st.session_state:
            st.session_state[dkey] = int(min(max(default_budget, lo), hi))
        budget = int(st.session_state[dkey])
        chosen, decoder = S.decode(samples, budget, diag)
        if chosen is not None and budget != default_budget:
            kept = S.bits_to_indices(chosen.get("bits"))
        elif chosen is not None and not kept:
            kept = S.bits_to_indices(chosen.get("bits"))
    html(C.qubit_tiles(pool, diag, kept, state=lens_state))
    html(C.lens_legend())
    if diag and diag.get("dispatch") and diag.get("dispatch") != "circuit":
        html(C.empty("Dispatch gate skipped the circuit",
                     f"<code>{C.esc(diag.get('dispatch'))}</code>: the problem had a closed-form answer, so no joules "
                     "were spent on a circuit for this question."))
    if lens_state != "done":
        return
    if diag:
        html(C.facts(diag, (qrow or {}).get("joules_select"), S.fnum((cal or {}).get("eps_in"))))
    if samples and not charts:
        html('<div class="qw-mini">The shot-distribution chart and the budget dial appear when the race finishes: '
             'redrawing thousands of points every half second would itself load this 15 W CPU while we measure.</div>')
    if samples and charts:
        c1, c2 = st.columns([1, 1.15], gap="medium")
        with c1:
            with st.container(border=True, key=f"histcard_{key}"):
                dist = cost_dist(diag)
                tab1, tab2 = st.tabs(["Where the shots land", "Most measured bitstrings"])
                with tab1:
                    if dist:
                        amp = S.fnum(diag.get("amplification")) or dist["amp"]
                        html(f'<div class="qw-shift"><b>{100 * dist["p_top_q"]:.0f}%</b> of the circuit&#39;s shots land in '
                             f'the best 1% of all 2^{dist["n"]} = {2 ** dist["n"]:,} possible selections. Random guessing '
                             f'lands there {100 * dist["p_top_u"]:.0f}% of the time: <b>{amp:.0f}× amplification</b>.</div>')
                        plot(CH.cost_shift(dist, amp), key=f"shift_{key}")
                        html('<div class="qw-mini">Interference shifts probability toward low-cost (good) selections. '
                             'Gray = every possible subset counted once (exact, brute force). Violet = what the circuit '
                             f'actually measured in {dist["shots"]:,} shots.</div>')
                    else:
                        html(C.empty("Cost distribution unavailable", "Needs <code>h</code> and <code>J_edges</code> in the diag."))
                with tab2:
                    plot(CH.bitstring_hist(diag, (chosen or {}).get("bits") or diag.get("best_bits")), key=f"hist_{key}")
                    html(f'<div class="qw-mini">{len(samples):,} distinct bitstrings in {S.fnum(diag.get("shots")) and int(diag["shots"]) or sum(int(s.get("count", 0)) for s in samples):,} shots. '
                         f'With 2^{diag.get("n_qubits", "n")} possible outcomes most appear only a few times, which is why the '
                         'first tab bins them by cost. Violet = fits the budget, gray = over it.</div>')
        with c2:
            with st.container(border=True, key=f"dialcard_{key}"):
                html(C.section("04", "Energy budget dial", "re-picks from the same shots, instantly"))
                toks_all = [int(s.get("tokens", 0)) for s in samples]
                pool_tok = sum(int(s.get("tokens") or 0) for s in (pool or {}).get("sentences", [])) or max(toks_all)
                lo = max(5, int(min(toks_all)))
                hi = max(lo + 10, int(pool_tok))
                st.slider("Context budget B (tokens)", min_value=lo, max_value=hi, step=1, key=f"dial_{key}_{qid}")
                dial_stats(chosen, budget, qrow, frow, decoder, len(samples))
                plot(CH.frontier_scatter(diag, budget, chosen, S.frontier(samples)), key=f"front_{key}")
    if not FIX and pool and kept:
        explain_box(qid, pool, kept, key)


def dial_stats(chosen: dict | None, budget: int, qrow: dict | None, frow: dict | None, decoder: str, n_s: int) -> None:
    overhead = 62.0
    tout = 12.0
    if qrow and S.fnum(qrow.get("tokens_in")) is not None and S.fnum(qrow.get("context_tokens")) is not None:
        overhead = float(qrow["tokens_in"]) - float(qrow["context_tokens"])
        tout = S.fnum(qrow.get("tokens_out")) or tout
    if chosen is None:
        html(C.empty("Nothing fits", f"No measured bitstring fits within {budget} tokens; raise the budget."))
        return
    toks = int(chosen.get("tokens") or 0)
    if chosen.get("fallback"):
        decoder += "; nothing fit, so the best shot was trimmed to the budget"
    nsent = str(chosen.get("bits", "")).count("1")
    pj = S.predicted_joules(cal, toks + overhead, tout)
    pf = S.predicted_joules(cal, float(frow["tokens_in"]), tout) if frow and S.fnum(frow.get("tokens_in")) else None
    vs = f"−{100 * (1 - pj / pf):.0f}%" if pj and pf else "—"
    cells = [("sentences", f"{nsent}"), ("tokens", f"{toks:,}"),
             ("predicted", f"{pj:,.1f} J" if pj is not None else "—"), ("vs full context", vs)]
    html('<div class="qw-kv">' + "".join(f'<div><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cells)
         + "</div>")
    html(f'<div class="qw-mini" style="margin-top:6px">No new circuit: a filter over the {n_s:,} stored bitstrings '
         f'({C.esc(decoder)}). Predicted joules from the calibrated model e0 + ε_in·tokens_in + ε_out·tokens_out.</div>')


def explain_box(qid: str, pool: dict, kept: list[int], key: str) -> None:
    sk = f"explain_{qid}_{','.join(map(str, kept))}"
    cols = st.columns([1, 4], vertical_alignment="center")
    with cols[0]:
        if st.button("Ask Gemini why", key=f"btn_{key}_{sk}", help="Two plain sentences from the Gemini referee"):
            sents = pool.get("sentences", [])
            with st.spinner("Asking Gemini…"):
                st.session_state[sk] = DA.explain_selection(
                    pool.get("question", ""), [s["text"] for s in sents if s["idx"] in set(kept)],
                    [s["text"] for s in sents if s["idx"] not in set(kept)]) or "(Gemini unavailable right now.)"
    with cols[1]:
        if st.session_state.get(sk):
            html(f'<div class="qw-explain">{C.esc(st.session_state[sk])}</div>')


def race_bench() -> None:
    html(C.section("02", "The race", "stored benchmark runs · energy measured on this laptop"))
    if bench.empty:
        html(C.empty("Nothing to race yet", "Benchmark runs appear here once <code>runs</code> has Full and Q-Watt rows."))
        lens_and_dial(None, None, None, "pending", [], None, None, "bench")
        return
    have = bench.groupby("qid")["arm"].apply(set)
    qids = [q for q, arms in have.items() if {"full", "qwatt"} <= arms] or list(have.index)
    qtext = bench.drop_duplicates("qid").set_index("qid")["question"].to_dict()
    demo = load_json("demo_questions", DB) or []
    demo = [d if isinstance(d, str) else d.get("qid") for d in demo] if isinstance(demo, list) else []
    demo_in = [q for q in demo if q in qids]
    qids = demo_in + [q for q in qids if q not in set(demo_in)]
    default = qids[0]
    if "bench_qid" not in st.session_state or st.session_state["bench_qid"] not in qids:
        st.session_state["bench_qid"] = default
    c1, c2 = st.columns([5, 1.2], vertical_alignment="bottom")
    with c1:
        st.selectbox("Question", qids, key="bench_qid",
                     format_func=lambda q: ("★ demo · " if q in set(demo_in) else "") + f"{str(qtext.get(q, ''))[:110]}  ·  {q}")
    qid = st.session_state["bench_qid"]
    with c2:
        if st.button("▶ Replay race", key="replay_btn", width="stretch"):
            st.session_state["replay"] = {"qid": qid, "start": time.time()}
            st.rerun()
    rep = st.session_state.get("replay")
    replaying = bool(rep and rep.get("qid") == qid)

    @st.fragment(run_every=0.4 if replaying else None)
    def frag():
        rep = st.session_state.get("replay")
        elapsed = (time.time() - rep["start"]) if rep and rep.get("qid") == qid else None
        rows = {a: bench[(bench["qid"] == qid) & (bench["arm"] == a)] for a in ("qwatt", "full", "topk") if S.visible(a)}
        rows = {a: r.iloc[0].to_dict() for a, r in rows.items() if not r.empty}
        views = {a: view_from_row(r, elapsed) for a, r in rows.items()}
        gold = next((r.get("gold") for r in rows.values() if r.get("gold")), None)
        st.markdown(f"<div style='font-size:22px;font-weight:700;color:{T.INK};letter-spacing:-.01em'>“{C.esc(qtext.get(qid, ''))}”</div>",
                    unsafe_allow_html=True)
        main = [views[a] for a in ("qwatt", "full") if a in views]
        x_max, y_max = shared_axes(list(views.values()))
        cols = st.columns(2, gap="medium")
        for col, a in zip(cols, ("qwatt", "full")):
            with col:
                if a in views:
                    render_lane(views[a], x_max, y_max, gold, f"b_{a}")
                else:
                    html(C.empty(f"No {T.ARM_LABEL[a]} run", "for this question in the selected batch."))
        if "topk" in views and S.visible("topk"):
            with st.expander("Top-k lane · the classical shortcut at the same token budget", expanded=False):
                render_lane(views["topk"], x_max, y_max, gold, "b_topk")
        if all(v["status"] == "done" for v in main) and len(main) == 2:
            html(C.verdict(rows["qwatt"], rows["full"], views["qwatt"]["joules"], views["full"]["joules"],
                           rows["qwatt"].get("joules_select"), S.fnum((cal or {}).get("eps_in")),
                           (S.fnum(rows["full"].get("tokens_in")) or 0) - (S.fnum(rows["qwatt"].get("tokens_in")) or 0)))
        # lens
        qrow, frow = rows.get("qwatt"), rows.get("full")
        diag = DA.qaoa_diag(qrow["run_id"]) if qrow else None
        pool = load_pool(qid, DB)
        kept = S.parse_selected(qrow.get("selected_json")) if qrow else []
        if not kept and diag:
            kept = S.bits_to_indices(diag.get("best_bits"))
        state = "done"
        if elapsed is not None and "qwatt" in views:
            vq = views["qwatt"]
            if vq["status"] == "running" and vq["phase"] == "select":
                state = "running"
        lens_and_dial(qid, pool, diag, state, kept, qrow, frow, "bench")
        if elapsed is not None and elapsed > max([v["duration"] for v in views.values()] + [0]) + 1.0:
            st.session_state.pop("replay", None)
            st.rerun(scope="app")

    frag()


def race_live() -> None:
    html(C.section("02", "The race", "live on this laptop · Q-Watt first, then Full context"))
    missing = [] if FIX else modules_ok("qwatt.pipeline", "qwatt.data", "qwatt.selectors")
    qs = load_demo_questions(DB)
    run = LV.current()
    c1, c2, c3 = st.columns([4.2, 1.1, 1.2], vertical_alignment="bottom")
    with c1:
        if qs:
            st.selectbox("Question", list(range(len(qs))), key="live_q",
                         format_func=lambda i: f"{str(qs[i]['question'])[:110]}  ·  {qs[i]['qid']}")
        else:
            html(C.empty("No demo questions", "Add <code>results/demo_questions.json</code> (a list of qids)."))
    with c2:
        topk = (st.toggle("+ Top-k lane", key="live_topk", help="Also run the classical Top-k arm after Full context")
                if S.visible("topk") else False)
    with c3:
        busy = run is not None and run.active
        disabled = busy or not qs or bool(missing)
        if st.button("⚡ Run live" if not busy else "Running…", key="run_live", disabled=disabled, width="stretch"):
            q = qs[int(st.session_state.get("live_q") or 0)]
            LV.start(q["qid"], q["question"], include_topk=topk)
            st.rerun()
    if missing:
        html(C.empty("Live mode is not wired yet",
                     "Waiting for module(s): " + ", ".join(f"<code>{m}</code>" for m in missing)
                     + ". Benchmark mode still works."))
    if not missing:
        LV.prewarm_async()
    run = LV.current()
    if run is not None and run.kind != "16":
        if run.active:
            html(C.empty("A long-document live run is in progress", "Switch to the long-document view to watch it."))
        run = None
    if run is None:
        html(C.empty("Ready", "Pick a question and press <b>Run live</b>. The Q-Watt lane runs first: tiles appear, "
                     "the circuit runs on Quantum Rings, tiles light up, then the full-context lane grinds."))
        lens_and_dial(None, None, None, "pending", [], None, None, "live")
        return
    idle_default = S.fnum((cal or {}).get("idle_watts"))

    ref_pending = any(l.ref_status == "pending" for l in run.lanes.values())

    @st.fragment(run_every=0.75 if run.active else (1.0 if ref_pending else None))
    def frag():
        LV.drain(run)
        st.markdown(f"<div style='font-size:22px;font-weight:700;color:{T.INK};letter-spacing:-.01em'>“{C.esc(run.question)}”</div>",
                    unsafe_allow_html=True)
        if run.status == "error":
            html(C.empty("Live run failed", C.esc(run.error or "")))
        elif run.active:
            html(f'<div class="qw-mini">{C.esc(run.message)}</div>')
        views = {a: view_from_lane(run.lanes[a], idle_default) for a in run.arms}
        x_max, y_max = shared_axes(list(views.values()))
        gold = (run.pool or {}).get("answer")
        cols = st.columns(2, gap="medium")
        for col, a in zip(cols, ("qwatt", "full")):
            with col:
                render_lane(views[a], x_max, y_max, gold, f"l_{a}")
        if "topk" in views and S.visible("topk"):
            with st.expander("Top-k lane · the classical shortcut at the same token budget", expanded=True):
                render_lane(views["topk"], x_max, y_max, gold, "l_topk")
        lq, lf = run.lanes["qwatt"], run.lanes["full"]
        if lq.status == "done" and lf.status == "done" and lq.row and lf.row:
            html(C.verdict(lq.row, lf.row, views["qwatt"]["joules"], views["full"]["joules"], lq.row.get("joules_select"),
                           S.fnum((cal or {}).get("eps_in")),
                           (S.fnum(lf.row.get("tokens_in")) or 0) - (S.fnum(lq.row.get("tokens_in")) or 0)))
        # lens
        if lq.diag is not None:
            state = "done"
        elif lq.status == "running" and lq.phase in (None, "select"):
            state = "running"
        else:
            state = "pending"
        kept = list(lq.indices or [])
        lens_and_dial(run.qid, run.pool, lq.diag, state, kept, lq.row, lf.row, "live", charts=not run.active)
        still_ref = any(l.ref_status == "pending" for l in run.lanes.values())
        done_key = (run.started, still_ref)
        if not run.active and st.session_state.get("_live_last_done") != done_key:
            st.session_state["_live_last_done"] = done_key
            st.rerun(scope="app")

    frag()


if VIEW == "lc":
    LCV.render(globals())
    st.stop()

html(f'<div class="qw-mini" style="margin-top:4px">This view is the exact check: with 16 candidate sentences a classical '
     f'computer can try all 65,536 selections, so every circuit result here is verified against the true optimum. '
     f'The long-document view runs the same Hamiltonian on up to 200 qubits.</div>')
if mode == "live":
    race_live()
else:
    race_bench()

# ---------------------------------------------------------------- 05 scoreboard
html(C.section("05", "Scoreboard", "per arm, benchmark batch · intervals are bootstrap 95%"))
sb = S.scoreboard(bench)
tiger_df, tiger_msg = (None, "disabled in fixture mode") if FIX else load_tiger()
src_pill = (f'<span class="qw-pill ok"><span class="dot"></span>Tiger Data · SQL aggregates in the cloud</span>'
            if tiger_df is not None else
            f'<span class="qw-pill"><span class="dot"></span>source: local SQLite (Python) · Tiger: {C.esc(tiger_msg)}</span>')
html(f'<div style="display:flex;gap:8px;margin-bottom:2px">{src_pill}</div>')
if sb.empty:
    html(C.empty("No scoreboard yet", "Needs benchmark rows in <code>runs</code>."))
else:
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True, key="sb1"):
            st.markdown("**Joules per query** · mean with 95% CI")
            plot(CH.arm_bars(sb, "J_mean", "J_lo", "J_hi", "J"), key="chart_jq")
    with c2:
        with st.container(border=True, key="sb2"):
            st.markdown("**Joules per correct answer** · the headline metric")
            plot(CH.arm_bars(sb, "JPC", None, None, "J"), key="chart_jpc")
    with c3:
        with st.container(border=True, key="sb3"):
            st.markdown("**Energy ledger** · where each arm's joules went")
            plot(CH.ledger(sb), key="chart_ledger")
    tbl = pd.DataFrame({
        "Arm": [T.ARM_LABEL.get(a, a) for a in sb["arm"]],
        "Runs": sb["n"],
        "J / query": sb["J_mean"],
        "95% CI": [f"{lo:,.1f}–{hi:,.1f}" if S.fnum(lo) is not None and S.fnum(hi) is not None else "—"
                   for lo, hi in zip(sb["J_lo"], sb["J_hi"])],
        "J / correct": sb["JPC"],
        "Correct": [f"{a} / {b}" for a, b in zip(sb["correct"], sb["graded"])],
        "Tokens in": sb["tokens_in"], "Tokens out": sb["tokens_out"], "TTFT (s)": sb["ttft_s"],
        "EM": sb["EM"], "F1": sb["F1"], "Referee": sb["referee"], "SF recall": sb["sf_recall"],
        "J select": sb["J_select"],
    })
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "J / query": st.column_config.NumberColumn(format="%.1f"),
        "J / correct": st.column_config.NumberColumn(format="%.1f"),
        "Tokens in": st.column_config.NumberColumn(format="%.0f"),
        "Tokens out": st.column_config.NumberColumn(format="%.1f"),
        "TTFT (s)": st.column_config.NumberColumn(format="%.2f"),
        "EM": st.column_config.NumberColumn(format="%.2f"), "F1": st.column_config.NumberColumn(format="%.2f"),
        "Referee": st.column_config.NumberColumn(format="%.2f"), "SF recall": st.column_config.NumberColumn(format="%.2f"),
        "J select": st.column_config.NumberColumn(format="%.2f"),
    })
    n_inv = int(bench["J_invalid"].sum()) if "J_invalid" in bench else 0
    inv_txt = (f" {n_inv} run(s) with net energy ≤ 0 (idle baseline read above the run's power) are left out of the "
               "joule statistics but still count for correctness." if n_inv else "")
    html(f'<div class="qw-mini">Correct = {C.esc(S.correctness_source(bench))}. Joules = CPU package energy net of the '
         f'idle baseline measured before each run; selector energy is charged to the selection arms.{C.esc(inv_txt)}</div>')
if tiger_df is not None:
    with st.expander("Tiger Data scoreboard (computed in SQL, cross-check of the numbers above)", expanded=False):
        st.dataframe(tiger_df, hide_index=True, width="stretch")


# ---------------------------------------------------------------- 06 show your work
def approx_dist(sq: dict) -> dict[str, list[float]]:
    dist: dict[str, list[float]] = {}

    def add(name, v):
        f = S.fnum(v)
        if f is not None:
            dist.setdefault(name, []).append(f)

    per = sq.get("per_instance") or sq.get("instances") or []
    if isinstance(per, list):
        for row in per:
            if not isinstance(row, dict):
                continue
            for k, v in row.items():
                if isinstance(v, dict):
                    add(k, v.get("approx_ratio", v.get("rho")))
                elif k.endswith("approx_ratio") and k != "approx_ratio":
                    add(k[: -len("_approx_ratio")], v)
            if "solver" in row and "approx_ratio" in row:
                add(str(row["solver"]), row["approx_ratio"])
    elif isinstance(per, dict):
        for k, rows in per.items():
            if isinstance(rows, list):
                for r in rows:
                    add(k, r.get("approx_ratio") if isinstance(r, dict) else r)
    for k, v in sq.items():
        if isinstance(v, dict):
            for kk in ("approx_ratios", "approx_ratio_list"):
                if isinstance(v.get(kk), list):
                    for x in v[kk]:
                        add(k, x)
            if isinstance(v.get("per_instance"), list):
                for r in v["per_instance"]:
                    add(k, r.get("approx_ratio") if isinstance(r, dict) else r)
    return dist


def summary_table(sq: dict) -> pd.DataFrame:
    summ = sq.get("summary")
    if not isinstance(summ, dict):
        summ = {k: v for k, v in sq.items() if isinstance(v, dict) and any(S.fnum(x) is not None for x in v.values())}
    rows = []
    for k, v in summ.items():
        if isinstance(v, dict):
            r = {"solver": T.ARM_LABEL.get(k, k)}
            r.update({kk: vv for kk, vv in v.items() if S.fnum(vv) is not None or isinstance(vv, (str, bool))})
            rows.append(r)
    return pd.DataFrame(rows)


def find_key(obj, needles: tuple[str, ...]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(n in str(k).lower() for n in needles) and S.fnum(v) is not None:
                return k, float(v)
            r = find_key(v, needles)
            if r:
                return r
    return None


with st.expander("06 · Show your work — solver quality, break-even, calibration, sensor cross-check", expanded=False):
    sq = load_json("solver_quality", DB)
    qa = load_qaoa(DB)
    sel = load_json("selection_eval", DB)
    a1, a2 = st.columns(2, gap="large")
    with a1:
        html('<div class="qw-subh">QAOA vs exact vs annealing vs greedy vs random-K</div>')
        dist = approx_dist(sq) if isinstance(sq, dict) else {}
        dist = {k: v for k, v in dist.items() if S.visible(k)}
        if not dist and not qa.empty and "approx_ratio" in qa:
            v = pd.to_numeric(qa["approx_ratio"], errors="coerce").dropna().tolist()
            if v:
                dist = {"qaoa": v}
        if dist:
            plot(CH.approx_ratio_box(dist), key="sw_ar")
            st.caption("Approximation ratio ρ = (C_worst − C(x*)) / (C_worst − C_opt) over feasible subsets, per instance."
                       + ("" if isinstance(sq, dict) else " Source: qaoa_runs (solver_quality.json not found)."))
        else:
            html(C.empty("No solver-quality results yet", "Produced by <code>scripts/eval_solvers.py</code> → "
                         "<code>results/solver_quality.json</code>."))
        if isinstance(sq, dict):
            stbl = summary_table(sq)
            if not stbl.empty and "solver" in stbl:
                hidden_labels = {T.ARM_LABEL.get(a, a) for a in S.HIDDEN_ARMS} | set(S.HIDDEN_ARMS)
                stbl = stbl[~stbl["solver"].isin(hidden_labels)]
            if not stbl.empty:
                st.dataframe(stbl.round(3), hide_index=True, width="stretch")
            tvl = sq.get("tvd") if isinstance(sq.get("tvd"), list) else None
            tv = None if tvl else find_key(sq, ("tvd", "total_variation"))
            if tvl:
                def _med(k1, k2):
                    v = [S.fnum((r.get("n16") or r).get(k1)) for r in tvl if isinstance(r, dict)]
                    f = [S.fnum((r.get("n16") or r).get(k2)) for r in tvl if isinstance(r, dict)]
                    v, f = [x for x in v if x is not None], [x for x in f if x is not None]
                    return (float(np.median(v)) if v else None), (float(np.median(f)) if f else None)
                a, af = _med("tvd_qr_vs_exact", "tvd_noise_floor")
                b, bf = _med("tvd_hamming_qr", "tvd_hamming_floor")
                if a is not None:
                    st.markdown(f"**Quantum Rings counts vs exact statevector probabilities** (median over instances): "
                                f"total variation distance **{a:.3f}** vs **{af:.3f}** expected from shot noise alone"
                                + (f"; by Hamming weight **{b:.3f}** vs **{bf:.3f}**" if b is not None and bf is not None else "")
                                + ". Matching the noise floor means the Quantum Rings samples agree with the exact "
                                  "circuit up to finite-shot noise.")
            if tv:
                st.markdown(f"**Quantum Rings counts vs exact statevector probabilities:** total variation distance "
                            f"**{tv[1]:.3f}** (`{tv[0]}`; 0 = identical distributions).")
    with a2:
        html('<div class="qw-subh">Solver overhead and break-even</div>')
        qrows = bench[bench["arm"] == "qwatt"] if not bench.empty else pd.DataFrame()
        eps = S.fnum((cal or {}).get("eps_in"))
        if not qrows.empty:
            js = qrows["joules_select"].dropna()
            p = S.paired(bench)
            removed = float(np.median(p["tok_a"] - p["tok_b"])) if not p.empty else None
            be = float(js.median()) / eps if eps and len(js) else None
            cells = [("quantum layer energy (median)", f"{js.median():.2f} J" if len(js) else "—"),
                     ("break-even tokens", f"{be:,.0f}" if be else "—"),
                     ("tokens actually removed", f"{removed:,.0f}" if removed is not None else "—"),
                     ("select time (median)", C.fmt_s(qrows["select_s"].median() if "select_s" in qrows else None))]
            html('<div class="qw-kv">' + "".join(f'<div><div class="k">{k}</div><div class="v">{v}</div></div>'
                                                 for k, v in cells) + "</div>")
            st.caption("Break-even = E_select / ε_in: the quantum layer pays for itself once it removes more tokens "
                       "than this.")
        else:
            html(C.empty("No Q-Watt benchmark runs", "Break-even needs <code>runs.joules_select</code>."))
        if not qa.empty and "dispatch" in qa:
            dc = qa["dispatch"].fillna("unknown").value_counts()
            tot = int(dc.sum())
            st.markdown("**Dispatch gate:** " + " · ".join(f"{k.replace('_', ' ')} {v} of {tot}" for k, v in dc.items()))
        if not qa.empty and "amplification" in qa:
            am = pd.to_numeric(qa["amplification"], errors="coerce").dropna()
            oh = pd.to_numeric(qa.get("opt_in_shots"), errors="coerce").dropna() if "opt_in_shots" in qa else pd.Series(dtype=float)
            if len(am):
                st.markdown(f"**Amplification** median **{am.median():.1f}×** over random sampling "
                            f"(1× = coin flips) · exact optimum in the shots for **{int(oh.sum())} of {len(oh)}** questions.")
    b1, b2 = st.columns(2, gap="large")
    with b1:
        html('<div class="qw-subh">Calibration: joules vs input tokens</div>')
        if cal and cal.get("points"):
            plot(CH.calibration_fit(cal), key="sw_cal")
            st.caption(f"Fit e0 + ε_in·tokens_in + ε_out·tokens_out on this laptop · energy source "
                       f"{cal.get('energy_source', '?')} · model {cal.get('model_id', '?')}.")
        else:
            html(C.empty("No calibration yet", "<code>scripts/calibrate.py</code> → <code>results/calibration.json</code>."))
    with b2:
        html('<div class="qw-subh">Package sensor vs battery cross-check</div>')
        win = pd.DataFrame()
        frows = bench[bench["arm"] == "full"] if not bench.empty else pd.DataFrame()
        if not frows.empty:
            r0 = frows.sort_values("started_at").iloc[-1]
            win = DA.power_window(float(r0["started_at"]) - 5, float(r0["ended_at"]) + 5)
        if win.empty:
            win = DA.power_recent(90)
        if not win.empty:
            plot(CH.sensor_crosscheck(win), key="sw_sensor")
            if not win["sys_watts"].notna().any():
                st.caption("Battery discharge is empty while plugged in; unplug for a cross-check run.")
            else:
                st.caption("Battery discharge covers the whole system (screen, RAM, SSD), so it sits above the package line; "
                           "the shapes should move together.")
        else:
            html(C.empty("No power samples", "Start <code>scripts/run_sampler.py</code>."))
    html('<div class="qw-subh">Selection quality without any LLM call (HotpotQA supporting facts)</div>')
    if isinstance(sel, dict):
        sel = {k: v for k, v in sel.items() if S.visible(k)}
    if isinstance(sel, dict) and any(isinstance(v, dict) for v in sel.values()):
        plot(CH.selection_recall(sel), key="sw_sel")
        pr = S.fnum(sel.get("pool_recall_mean"))
        if pr is not None:
            st.caption(f"Pool recall {pr:.2f}: the share of gold sentences that made it into the 16-sentence pool, "
                       "which caps every selector.")
    else:
        html(C.empty("No selection eval yet", "<code>scripts/eval_selection.py</code> → <code>results/selection_eval.json</code>."))
    if not bench.empty:
        html('<div class="qw-subh">Raw runs (benchmark batch)</div>')
        cols = [c for c in ("qid", "arm", "tokens_in", "tokens_out", "joules_net", "joules_gross", "joules_select",
                            "ttft_s", "total_s", "answer", "gold", "em", "f1", "referee_correct", "llm_backend",
                            "energy_source", "config_hash") if c in bench]
        st.dataframe(bench[cols].sort_values(["qid", "arm"]), hide_index=True, width="stretch", height=280)

# ---------------------------------------------------------------- 07 footer
ex = S.extrapolate(h.get("delta_J_median"))
html(C.footer(ex, h.get("delta_J_median"), FIX))
html(f'<div class="qw-mini" style="text-align:center;margin-top:10px">Q-Watt · Hack the Nest 2026 · Quantum Rings SDK '
     f'(Qiskit bridge) · Gemma via LM Studio · Gemini referee · Tiger Data · data: {C.esc(Path(DB).name)}</div>')

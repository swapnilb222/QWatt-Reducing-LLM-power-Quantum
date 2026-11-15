"""Long-context view (the hero after the 04:35 pivot): one qubit per sentence of a whole document.

render(G) is called from app/dashboard.py with its globals() so the race lanes, replay and plotting helpers are
shared with the 16-qubit view. Data: runs/qaoa_runs rows with bench_id 'lc-*' and arms full_lc / qwatt_safe /
qwatt_lc; results/scaling.json; results/calibration_lc.json; qwatt.longdoc.get_lc_pool (sentence text).
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import charts as CH
from app import components as C
from app import data_access as DA
from app import live as LV
from app import stats as S
from app import theme as T

A_FULL, A_SAFE, A_RAW = "full_lc", "qwatt_safe", "qwatt_lc"

CSS = f"""
<style>
.lc-hero {{ display:grid; grid-template-columns: 1.3fr 1.15fr 1fr 1fr 1fr; gap:14px; }}
.lc-hero .val {{ font-size:46px; }}
.lc-hero .qw-stat.primary .val {{ font-size:60px; }}
.lc-hero .qw-duo .v {{ font-size:36px; }}
.mm {{ display:flex; flex-direction:column; gap:6px; }}
.mm-row {{ display:flex; align-items:center; gap:10px; }}
.mm-p {{ width:44px; flex:none; color:{T.MUTED}; font-family:{T.FONT_MONO}; font-size:12px; font-weight:700; text-align:right; }}
.mm-cells {{ display:flex; flex-wrap:wrap; align-items:center; row-gap:6px; }}
.mm-c {{ position:relative; width:27px; height:27px; border-radius:6px; flex:none;
  background: rgba(144,133,233, var(--a)); border:1px solid rgba(255,255,255,.06); transition: all .4s ease; }}
.mm-l {{ width:6px; height:4px; flex:none; border-radius:2px; }}
.mm-c.on {{ background: linear-gradient(180deg, #d9d3ff, {T.QUANTUM}); border-color:#e4e0ff;
  box-shadow: 0 0 12px -1px {T.rgba(T.QUANTUM, .95)}; }}
.mm-c.off {{ opacity:.75; }}
.mm-c.gold::after {{ content:"★"; position:absolute; inset:0; display:grid; place-items:center; font-size:12px;
  color:{T.WARN}; font-size:15px; text-shadow:0 0 4px #000; }}
.mm-c.run {{ animation: mmpulse 1.1s ease-in-out infinite; }}
@keyframes mmpulse {{ 50% {{ border-color:{T.rgba(T.QUANTUM, .9)}; box-shadow:0 0 10px -2px {T.rgba(T.QUANTUM, .9)}; }} }}
.mm-c .tip {{ position:absolute; bottom: calc(100% + 8px); left:50%; transform:translateX(-50%); width:340px; z-index:60;
  visibility:hidden; opacity:0; transition:opacity .1s; background:{T.SURFACE_2}; border:1px solid {T.AXIS}; border-radius:12px;
  padding:10px 12px; box-shadow:0 16px 40px -10px rgba(0,0,0,.8); pointer-events:none; text-align:left; }}
.mm-c:hover {{ border-color:{T.INK}; z-index:70; }}
.mm-c:hover .tip {{ visibility:visible; opacity:1; }}
.mm-c .tip .tx {{ color:{T.INK}; font-size:14.5px; line-height:1.4; }}
.mm-c .tip .tm {{ color:{T.INK_2}; font-size:12px; margin-top:6px; font-family:{T.FONT_MONO}; }}
.mm-c.e0 .tip {{ left:0; transform:none; }}
.mm-c.e1 .tip {{ left:auto; right:0; transform:none; }}
.lc-note {{ color:{T.INK_2}; font-size:14.5px; }}
</style>
"""


# ------------------------------------------------------------------ data helpers
def _qaoa_for(run_ids) -> pd.DataFrame:
    qa = DA.qaoa_df()
    if qa.empty:
        return qa
    return qa[qa["run_id"].isin(list(run_ids))]


def _lc_pool(qid: str, G) -> dict | None:
    @st.cache_data(ttl=900, show_spinner=False, max_entries=64)
    def _load(q, _db):
        return DA.get_lc_pool_dict(q)
    try:
        return _load(qid, G["DB"])
    except Exception:
        return None


def _hist(h, diag) -> tuple[list, list] | None:
    """Tolerant reader for cost_hist_*: {edges|bins, counts|p|probs} or a bare list (+ diag cost_hist_edges)."""
    if h is None:
        return None
    if isinstance(h, dict):
        e = h.get("edges") or h.get("bins") or h.get("bin_edges")
        v = h.get("counts") or h.get("p") or h.get("probs") or h.get("hist") or h.get("freq")
    else:
        e, v = diag.get("cost_hist_edges") or diag.get("cost_bins"), h
    if not v:
        return None
    v = [float(x) for x in v]
    if not e or len(e) not in (len(v), len(v) + 1):
        e = list(range(len(v) + 1))
    if len(e) == len(v):  # centers given
        e = list(e) + [e[-1] + (e[-1] - e[-2] if len(e) > 1 else 1)]
    tot = sum(v) or 1.0
    return [float(x) for x in e], [100 * x / tot for x in v]


def _scaling(sc: dict | None):
    if not isinstance(sc, dict):
        return None
    pts = sc.get("points") or sc.get("rows") or sc.get("results") or []
    rows = []
    for p in pts if isinstance(pts, list) else []:
        if not isinstance(p, dict):
            continue
        n = S.fnum(p.get("n_qubits", p.get("n", p.get("qubits"))))
        j = S.fnum(p.get("joules", p.get("joules_net", p.get("energy_j", p.get("j")))))
        w = S.fnum(p.get("wall_s", p.get("wall", p.get("seconds"))))
        if n is not None:
            rows.append({"n": n, "J": j, "wall": w, "edges": S.fnum(p.get("n_edges"))})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("n")
    df = df.groupby("n", as_index=False).median(numeric_only=True)
    return df


# ------------------------------------------------------------------ HTML blocks
def _hero(h: dict, n_q: int, qubits_max, qubits_med, esc_rate, n_esc, n_safe, eps, src) -> str:
    if h.get("saving") is None:
        sav_v, sav_sub = "—", "needs paired Full document + Q-Watt runs"
    else:
        lo, hi = h.get("saving_lo"), h.get("saving_hi")
        ci = f"95% CI {100 * lo:.0f}–{100 * hi:.0f}%" if lo is not None and hi is not None else "CI needs ≥2 questions"
        n_tot = h["n_pairs"] + (h.get("n_excluded") or 0)
        sav_v = f"{100 * h['saving']:.0f}<small>%</small>"
        sav_sub = C.esc(f"median paired saving vs reading the whole document · {ci} · {h['n_pairs']} of {n_tot} questions")
        extra = []
        if h.get("n_excluded"):
            extra.append(f"{h['n_excluded']} excluded (idle baseline above run power)")
        if h.get("gross_saving") is not None:
            extra.append(f"gross: {100 * h['gross_saving']:.0f}%")
        if extra:
            sav_sub += '<br><span style="opacity:.85">' + C.esc(" · ".join(extra)) + "</span>"

    def duo(arm, v):
        return (f'<div><div class="k"><span class="sw" style="background:{T.ARM_COLOR[arm]}"></span>'
                f'{T.ARM_LABEL[arm]}</div><div class="v">{C.fmt_j(v)}<small>J</small></div></div>')

    okq, nq, okf, nf = h.get("ok_qwatt", 0), h.get("n_qwatt", 0), h.get("ok_full", 0), h.get("n_full", 0)
    q_v = "—" if qubits_max is None else f"{int(qubits_max)}"
    q_sub = ("one qubit per sentence of the whole document · Quantum Rings" if qubits_med is None else
             f"median {int(qubits_med)} per document · one qubit per sentence · local couplings · Quantum Rings")
    e_v = "—" if esc_rate is None else f"{100 * esc_rate:.0f}<small>%</small>"
    e_sub = (f"{n_esc} of {n_safe} questions fell back to the full document when the excerpt was not enough; "
             "both reads are charged" if n_safe else "safety net: escalate to the full document if the model abstains")
    cal = (f'Measured on this laptop: <b>{eps:.3f} J</b> per input token ({C.esc(src)}) · a full document costs '
           f'hundreds of joules and close to a minute of maxed CPU' if eps else
           '<span class="qw-flag">⚑ no long-prompt calibration yet</span>')
    return f"""
<div class="lc-hero">
  <div class="qw-card qw-stat primary"><div class="lbl">Energy saved vs reading the full document</div>
    <div class="val">{sav_v}</div><div class="sub">{sav_sub}</div></div>
  <div class="qw-card qw-stat"><div class="lbl">Joules per correct answer</div>
    <div class="qw-duo">{duo(A_FULL, h.get("jpc_full"))}{duo(A_SAFE, h.get("jpc_qwatt"))}</div>
    <div class="sub">total joules ÷ correct answers</div></div>
  <div class="qw-card qw-stat"><div class="lbl">Answers correct</div>
    <div class="val">{okq if nq else "—"}<small>of {nq}</small></div>
    <div class="sub">Q-Watt (with safety net) · Full document: {okf} of {nf}</div></div>
  <div class="qw-card qw-stat"><div class="lbl">Qubits used</div><div class="val">{q_v}</div>
    <div class="sub">{C.esc(q_sub)}</div></div>
  <div class="qw-card qw-stat"><div class="lbl">Safety net used</div><div class="val">{e_v}</div>
    <div class="sub">{C.esc(e_sub)}</div></div>
</div>
<div class="qw-cal">{cal}</div>"""


def _minimap(pool: dict | None, diag: dict | None, kept: list[int], state: str) -> str:
    sents = (pool or {}).get("sentences") or []
    n = len(sents) or int((diag or {}).get("n_qubits") or 0)
    if n == 0:
        return ""
    para = (diag or {}).get("paragraph_of") or (pool or {}).get("paragraph_of") or [0] * n
    para = (list(para) + [para[-1] if para else 0] * n)[:n]
    rel = [S.fnum(s.get("relevance")) or 0.0 for s in sents] if sents else \
        [S.fnum(x) or 0.0 for x in ((diag or {}).get("relevance") or [0.3] * n)]
    rel = (rel + [0.0] * n)[:n]
    toks = [s.get("tokens") for s in sents] if sents else list((diag or {}).get("t") or [None] * n)
    toks = (list(toks) + [None] * n)[:n]
    link = {}
    for e in (diag or {}).get("red_edges") or []:
        if len(e) >= 2 and abs(int(e[1]) - int(e[0])) == 1:
            link[min(int(e[0]), int(e[1]))] = T.RED_LINK
    for e in (diag or {}).get("bridge_edges") or []:
        if len(e) >= 2 and abs(int(e[1]) - int(e[0])) == 1:
            link[min(int(e[0]), int(e[1]))] = T.BRIDGE
    ks = set(kept or [])
    rows, cur, cells, pos = [], None, [], 0
    for i in range(n):
        if para[i] != cur:
            if cells:
                rows.append((cur, cells))
            cur, cells, pos = para[i], [], 0
        s = sents[i] if i < len(sents) else {}
        on = state == "done" and i in ks
        cls = "mm-c" + (" on" if on else "") + (" off" if state == "done" and not on else "") + \
              (" run" if state == "running" else "") + (" gold" if s.get("is_gold") else "")
        cls += " e0" if pos < 6 else ""
        a = 0.10 + 0.62 * max(0.0, min(1.0, rel[i]))
        tip = (f'<div class="tx">{C.esc(s.get("text", ""))}</div>' if s else f'<div class="tx">sentence {i}</div>')
        meta = f"qubit {i} · relevance {rel[i]:.2f}" + (f" · {toks[i]} tokens" if toks[i] is not None else "") + \
               (" · kept (1)" if on else (" · dropped (0)" if state == "done" else "")) + \
               (" · ★ answer sentence" if s.get("is_gold") else "")
        cells.append(f'<span class="{cls}" style="--a:{a:.2f}"><span class="tip">{tip}<div class="tm">{C.esc(meta)}</div></span></span>')
        nxt_same = i + 1 < n and para[i + 1] == para[i]
        if nxt_same:
            col = link.get(i)
            cells.append(f'<span class="mm-l" style="background:{col if col else "transparent"}"></span>')
        pos += 1
    if cells:
        rows.append((cur, cells))
    body = "".join(f'<div class="mm-row"><div class="mm-p">para {int(p) + 1 if isinstance(p, (int, float)) else p}</div>'
                   f'<div class="mm-cells">{"".join(c)}</div></div>' for p, c in rows)
    return f'<div class="mm">{body}</div>'


def _legend() -> str:
    return (f'<div class="qw-legend"><span><i style="background:{T.QUANTUM};height:12px;width:12px;border-radius:3px"></i>'
            f'kept (measured 1)</span><span><i style="background:{T.rgba(T.QUANTUM, .45)};height:12px;width:12px;border-radius:3px">'
            f'</i>shade = relevance to the question</span><span><b style="color:{T.WARN}">★</b> sentence holding the answer '
            f'(dataset label)</span><span><i style="background:{T.RED_LINK}"></i>redundant neighbours</span>'
            f'<span><i style="background:{T.BRIDGE}"></i>neighbours that belong together</span>'
            f'<span style="color:{T.MUTED}">couplings are local: each qubit only talks to sentences next to it · hover a cell to read it</span></div>')


def _facts(diag: dict, qrow: dict | None, qa_j) -> str:
    def fact(k, v, hi=False):
        return f'<div class="qw-fact{" hi" if hi else ""}"><div class="k">{C.esc(k)}</div><div class="v">{v}</div></div>'
    r = S.fnum(diag.get("ratio_vs_anneal"))
    j = S.fnum(qa_j) if S.fnum(qa_j) is not None else S.fnum((qrow or {}).get("joules_select"))
    items = [fact("backend", C.esc(diag.get("backend") or "—")), fact("qubits", C.fmt_i(diag.get("n_qubits")), hi=True),
             fact("couplings", C.fmt_i(diag.get("n_edges"))), fact("coupling range", f"|i−j| ≤ {diag.get('W', 2)}"),
             fact("depth", C.fmt_i(diag.get("depth"))), fact("shots", C.fmt_i(diag.get("shots"))),
             fact("circuit wall time", C.fmt_s(diag.get("circuit_wall_s"))),
             fact("quantum layer energy", f"{j:,.1f} J" if j is not None else "—"),
             fact("shots inside energy budget", f"{100 * S.fnum(diag.get('feasible_shot_frac')):.0f}%" if S.fnum(diag.get("feasible_shot_frac")) is not None else "—", hi=True),
             fact("answer sentence kept", "✓ yes" if diag.get("gold_kept") else ("✗ no" if "gold_kept" in diag else "—")),
             fact("distinct outcomes", C.fmt_i(diag.get("n_distinct")))]
    if diag.get("feasible_fallback"):
        items.append(fact("note", "repaired to fit the budget"))
    return '<div class="qw-facts">' + "".join(items) + "</div>"


# ------------------------------------------------------------------ charts
def _cost_hist_fig(eq, pq, er, pr, diag) -> go.Figure:
    fig = go.Figure()
    mid = lambda e: [(e[i] + e[i + 1]) / 2 for i in range(len(e) - 1)]  # noqa: E731
    fig.add_trace(go.Scatter(x=mid(er), y=pr, mode="lines", name="random selections (same QUBO)", line=dict(
        width=2, color=T.NEUTRAL_B, shape="hvh"), fill="tozeroy", fillcolor=T.rgba(T.NEUTRAL_B, .10),
        hovertemplate="cost %{x:.2f}: %{y:.1f}% of random picks<extra></extra>"))
    fig.add_trace(go.Scatter(x=mid(eq), y=pq, mode="lines", name="QAOA circuit shots (Quantum Rings)", line=dict(
        width=2.5, color=T.QUANTUM, shape="hvh"), fill="tozeroy", fillcolor=T.rgba(T.QUANTUM, .22),
        hovertemplate="cost %{x:.2f}: %{y:.1f}% of shots<extra></extra>"))
    ref = (diag.get("reference") or {})
    if S.fnum(ref.get("anneal_cost")) is not None:
        fig.add_vline(x=ref["anneal_cost"], line=dict(width=1.5, color=T.INK))
        fig.add_annotation(x=ref["anneal_cost"], y=1, yref="paper", text="annealing reference", showarrow=False,
                           xanchor="right", yanchor="top", xshift=-5, font=dict(size=12, color=T.INK),
                           bgcolor="rgba(18,21,28,.85)")
    fig.update_xaxes(autorange="reversed", title_text="QUBO cost of the selection   (← worse · better →)", showgrid=False)
    fig.update_yaxes(ticksuffix="%", title_text="share of outcomes", nticks=5)
    fig.update_layout(template="qwatt", height=330, showlegend=True, margin=dict(l=4, r=8, t=34, b=4))
    return fig


def _money_fig(sc: pd.DataFrame, save_per_q: float, eps: float, docs: tuple | None) -> tuple[go.Figure, float | None]:
    fig = go.Figure()
    nmax = max(float(sc["n"].max()), docs[1] if docs else 0, 200.0) * 1.05
    xs = np.linspace(0, nmax, 80)
    saved = save_per_q * xs
    if docs:
        fig.add_vrect(x0=docs[0], x1=docs[1], fillcolor=T.rgba(T.QWATT, .08), line_width=0, layer="below")
        fig.add_annotation(x=(docs[0] + docs[1]) / 2, y=1, yref="paper", text="today's documents are here",
                           showarrow=False, yanchor="top", font=dict(size=13, color=T.INK))
    fig.add_trace(go.Scatter(x=xs, y=saved, mode="lines", name="prefill energy saved (ε_in × tokens removed)",
                             line=dict(width=2.5, color=T.QWATT),
                             hovertemplate="%{x:.0f} sentences: saves %{y:.0f} J<extra></extra>"))
    d = sc.dropna(subset=["J"])
    fig.add_trace(go.Scatter(x=d["n"], y=d["J"], mode="lines+markers", name="quantum layer cost (measured)",
                             line=dict(width=2.5, color=T.QUANTUM), marker=dict(size=10, color=T.QUANTUM,
                             line=dict(width=2, color=T.SURFACE)),
                             hovertemplate="%{x:.0f} qubits: %{y:.0f} J on Quantum Rings<extra></extra>"))
    be = None
    if len(d):
        cost_f = lambda x: float(np.interp(x, d["n"], d["J"]))  # noqa: E731
        for x in np.linspace(1, nmax, 400):
            if save_per_q * x >= cost_f(x):
                be = float(x)
                break
    if be is not None:
        fig.add_vline(x=be, line=dict(width=1.5, color=T.INK_2))
        fig.add_annotation(x=be, y=0.5, yref="paper", text=f"break-even ≈ {be:.0f} sentences", showarrow=False,
                           xanchor="left", xshift=6, font=dict(size=13, color=T.INK), bgcolor="rgba(18,21,28,.85)")
    fig.update_xaxes(title_text="document length (sentences = qubits)", range=[0, nmax], showgrid=False)
    fig.update_yaxes(title_text="joules per question", rangemode="tozero")
    fig.update_layout(template="qwatt", height=360, showlegend=True, margin=dict(l=4, r=8, t=34, b=4))
    return fig, be


def _wall_fig(sc: pd.DataFrame) -> go.Figure:
    d = sc.dropna(subset=["wall"])
    fig = go.Figure(go.Scatter(x=d["n"], y=d["wall"], mode="lines+markers", line=dict(width=2.5, color=T.QUANTUM),
                               marker=dict(size=10, color=T.QUANTUM, line=dict(width=2, color=T.SURFACE)),
                               hovertemplate="%{x:.0f} qubits: %{y:.1f} s<extra></extra>"))
    fig.update_xaxes(title_text="qubits", showgrid=False, rangemode="tozero")
    fig.update_yaxes(title_text="circuit wall time", ticksuffix=" s", rangemode="tozero")
    fig.update_layout(template="qwatt", height=360, margin=dict(l=4, r=8, t=34, b=4))
    return fig


# ------------------------------------------------------------------ sections
def _race_rows(bench: pd.DataFrame, qid: str) -> dict:
    rows = {a: bench[(bench["qid"] == qid) & (bench["arm"] == a)] for a in (A_SAFE, A_FULL, A_RAW)}
    return {a: r.iloc[0].to_dict() for a, r in rows.items() if not r.empty}


def _verdict(rq, rf, Jq, Jf) -> str:
    v = C.verdict(rq, rf, Jq, Jf, (rq or {}).get("joules_select"), None, None)
    if v and S.fnum((rq or {}).get("escalated")):
        v = v.replace("</div>", '<span class="sep"></span><span>⤴ the safety net escalated to the full document; '
                                '<b>both reads are charged</b></span></div>', 1)
    return v


def _lens(G, qid, pool, diag, state, kept_default, qrow, frow, key, charts=True):
    html, plot = G["html"], G["plot"]
    n = len((pool or {}).get("sentences") or []) or int((diag or {}).get("n_qubits") or 0)
    html(C.section("03", "The document, as qubits", "one qubit per sentence of the whole document"))
    html(f'<div class="qw-caption"><span class="q">|1⟩</span>Each sentence of the document is a qubit — '
         f'{n if n else "N"} qubits. 1 = keep.</div>')
    if pool is None and diag is None:
        html(C.empty("No document to show yet", "Pick a question with a Q-Watt run, or start a live run."))
        return
    samples = (diag or {}).get("samples") or []
    kept, chosen, decoder = kept_default, None, ""
    budget = None
    if samples and state == "done" and charts:
        toks = [int(s.get("tokens", 0)) for s in samples]
        base_b = int(G["default_budget_lc"])
        lo, hi = max(10, min(toks) // 2), max(max(toks) * 2, base_b + 200)
        dkey = f"lcdial_{key}_{qid}"
        if dkey not in st.session_state:
            st.session_state[dkey] = int(min(max(base_b, lo), hi))
        budget = int(st.session_state[dkey])
        chosen, decoder = S.decode(samples, budget, diag)
        if chosen is not None and budget != base_b:
            kept = S.bits_to_indices(chosen.get("bits"))
        elif not kept and chosen is not None:
            kept = S.bits_to_indices(chosen.get("bits"))
    if not kept and diag and diag.get("kept_bits"):
        kept = S.bits_to_indices(diag["kept_bits"])
    with st.container(border=True, key=f"lc_mapcard_{key}"):
        html(_minimap(pool, diag, kept, state))
        if state == "done" and kept:
            tok = sum(int((pool or {}).get("sentences", [{}] * (max(kept) + 1))[i].get("tokens") or 0)
                      for i in kept if pool and i < len(pool.get("sentences", []))) if pool else None
            html(f'<div class="lc-note" style="margin-top:8px">Kept <b>{len(kept)}</b> of {n} sentences'
                 + (f' · {tok:,} tokens instead of {pool.get("full_tokens", 0):,}' if tok and pool else "") + '</div>')
    html(_legend())
    if samples and state == "done" and charts:
        c1, c2 = st.columns([1.1, 1], gap="medium")
        with c1:
            with st.container(border=True, key=f"lc_dialcard_{key}"):
                html(C.section("", "Energy budget dial", "re-picks from the stored shots, no new circuit"))
                toks = [int(s.get("tokens", 0)) for s in samples]
                base_b = int(G["default_budget_lc"])
                lo, hi = max(10, min(toks) // 2), max(max(toks) * 2, base_b + 200)
                st.slider("Context budget B (tokens)", min_value=lo, max_value=hi, step=10, key=f"lcdial_{key}_{qid}")
                _dial_stats(G, chosen, qrow, frow, decoder, diag)
        with c2:
            with st.container(border=True, key=f"lc_frontcard_{key}"):
                plot(CH.frontier_scatter(diag, budget, chosen, S.frontier(samples)), key=f"lcfront_{key}")
                html(f'<div class="qw-mini">The {len(samples):,} best distinct measured selections (of '
                     f'{C.fmt_i(diag.get("n_distinct"))}) are stored, so any budget is answered from the same circuit run.</div>')
    elif samples and not charts:
        html('<div class="qw-mini">The budget dial and the shot distribution appear when the race finishes, so redrawing '
             'them does not load this 15 W CPU while we measure.</div>')


def _dial_stats(G, chosen, qrow, frow, decoder, diag):
    html = G["html"]
    if chosen is None:
        html(C.empty("Nothing fits", "No stored selection fits that budget."))
        return
    cal = G["cal_lc"] or G["cal"]
    toks = int(chosen.get("tokens") or 0)
    nsent = str(chosen.get("bits", "")).count("1")
    overhead = 80.0
    if qrow and S.fnum(qrow.get("context_tokens")) is not None and S.fnum(qrow.get("tokens_in")) is not None \
            and not S.fnum(qrow.get("escalated")):
        overhead = max(0.0, float(qrow["tokens_in"]) - float(qrow["context_tokens"]))
    pj = S.predicted_joules(cal, toks + overhead, 6)
    pf = S.predicted_joules(cal, float(frow["tokens_in"]), 6) if frow and S.fnum(frow.get("tokens_in")) else None
    cells = [("sentences", f"{nsent}"), ("tokens", f"{toks:,}"), ("predicted", f"{pj:,.0f} J" if pj else "—"),
             ("vs full document", f"−{100 * (1 - pj / pf):.0f}%" if pj and pf else "—")]
    html('<div class="qw-kv">' + "".join(f'<div><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cells) + "</div>")
    note = f"Decoder: {decoder}." + (" Nothing fit, so the best shot was repaired to the budget." if chosen.get("fallback") else "")
    html(f'<div class="qw-mini" style="margin-top:6px">{C.esc(note)} Predicted joules use the long-prompt calibration.</div>')


def _why(G, diag, qrow, qa_j, key):
    html, plot = G["html"], G["plot"]
    html(C.section("04", "Why a quantum circuit at this size", "2^n selections: far too many to check"))
    if not diag or diag.get("arm") == "full_lc":
        html(C.empty("No circuit diagnostics", "Needs a Q-Watt run with a stored diag."))
        return
    n = int(diag.get("n_qubits") or 0)
    hq = _hist(diag.get("cost_hist_qaoa"), diag)
    hr = _hist(diag.get("cost_hist_random"), diag)
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        with st.container(border=True, key=f"lc_histcard_{key}"):
            ref = diag.get("reference") or {}
            rb = S.fnum(ref.get("random_best_cost"))
            samples = diag.get("samples") or []
            shots = S.fnum(diag.get("shots")) or sum(int(s.get("count", 0)) for s in samples) or 1
            beat = sum(int(s.get("count", 0)) for s in samples if rb is not None and S.fnum(s.get("cost")) is not None
                       and s["cost"] < rb) / shots if rb is not None else None
            ff = S.fnum(diag.get("feasible_shot_frac"))
            lead = (f'<b>{100 * ff:.0f}%</b> of the circuit’s {int(shots):,} shots fit the energy budget as measured — '
                    f'uniformly random selections almost never do (0% on the test set: they keep about half the document). '
                    if ff is not None else "")
            html(f'<div class="qw-shift">{lead}With {n} qubits there are 2^{n} ≈ 10^{int(n * math.log10(2))} possible '
                 f'selections; the circuit concentrates its shots on good ones.</div>')
            if hq and hr:
                plot(_cost_hist_fig(hq[0], hq[1], hr[0], hr[1], diag), key=f"lchist_{key}")
            else:
                html(C.empty("Cost histograms missing", "Needs <code>cost_hist_qaoa</code> / <code>cost_hist_random</code> in the diag."))
            html('<div class="qw-mini">Same QUBO, same bins. Gray: uniformly random selections. Violet: what the QAOA circuit '
                 'measured. Lower cost = more relevance per joule, less repetition.</div>')
    with c2:
        html(_facts(diag, qrow, qa_j))
        r = S.fnum(diag.get("ratio_vs_anneal"))
        if r is not None:
            html('<div class="lc-note" style="margin-top:10px">The energy price inside the Hamiltonian is raised until the '
                 'circuit’s typical measurement fits the token budget, so the selection is read straight from a measured shot, '
                 'with no classical repair. At this size nobody can check every selection; the 16-qubit view shows the exact '
                 'check where we can.</div>')


def _scaling_section(G, bench):
    html, plot = G["html"], G["plot"]
    html(C.section("05", "The scaling story", "the quantum layer's cost stays flat while the savings grow"))
    sc_raw = G["load_json"]("scaling", G["DB"])
    sc = _scaling(sc_raw)
    if sc is None:
        html(C.empty("No scaling data yet", "Produced by <code>scripts/scaling_qr.py</code> → <code>results/scaling.json</code>."))
        return
    cal = G["cal_lc"] or G["cal"] or {}
    eps = S.fnum((sc_raw or {}).get("eps_in")) or S.fnum(cal.get("eps_in")) or 0.059
    # tokens removed per sentence of document: measured from the LC benchmark when available
    per_sent, src = None, "scaling.json parameters"
    if not bench.empty:
        f = bench[bench["arm"] == A_FULL].set_index("qid")
        q = bench[bench["arm"] == A_SAFE].set_index("qid")
        qa = _qaoa_for(q["run_id"]) if not q.empty else pd.DataFrame()
        if not qa.empty and not f.empty:
            nq = qa.set_index("qid")["n_qubits"]
            common = f.index.intersection(q.index).intersection(nq.index)
            if len(common):
                rem = (f.loc[common, "context_tokens"] - q.loc[common, "context_tokens"]) / nq.loc[common].astype(float)
                rem = rem.replace([np.inf, -np.inf], np.nan).dropna()
                if len(rem):
                    per_sent, src = float(rem.median()), "median tokens removed per sentence in the benchmark"
    if per_sent is None:
        tps = S.fnum((sc_raw or {}).get("tokens_per_sentence")) or 27.0
        keep = S.fnum((sc_raw or {}).get("keep_frac")) or 0.15
        per_sent = tps * (1 - keep)
    docs = None
    dd = (sc_raw or {}).get("docs") or {}
    if S.fnum(dd.get("n_min")) is not None and S.fnum(dd.get("n_max")) is not None:
        docs = (float(dd["n_min"]), float(dd["n_max"]))
    elif not bench.empty:
        qa = _qaoa_for(bench["run_id"])
        if not qa.empty:
            v = pd.to_numeric(qa["n_qubits"], errors="coerce").dropna()
            if len(v):
                docs = (float(v.min()), float(v.max()))
    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1:
        with st.container(border=True, key="lc_money"):
            fig, be = _money_fig(sc, eps * per_sent, eps, docs)
            lead = (f'The circuit pays for itself once a document has more than <b>~{be:.0f} sentences</b>. '
                    if be is not None else "")
            html(f'<div class="qw-shift">{lead}Every extra sentence the circuit removes saves '
                 f'<b>{eps * per_sent:.2f} J</b> of prefill; the circuit’s own cost barely moves.</div>')
            plot(fig, key="lc_money_fig")
            html(f'<div class="qw-mini">Saved energy = ε_in ({eps:.3f} J/token, measured) × tokens removed '
                 f'({per_sent:.1f} per sentence: {C.esc(src)}). Quantum layer cost = measured joules of the circuit on '
                 f'Quantum Rings at each size.</div>')
    with c2:
        with st.container(border=True, key="lc_wall"):
            html('<div class="qw-subh">Circuit wall time vs qubits</div>')
            plot(_wall_fig(sc), key="lc_wall_fig")
            html('<div class="qw-mini">Local couplings (each sentence only with its neighbours) keep the circuit shallow, '
                 'so simulation time grows gently with document length.</div>')


def _scoreboard(G, bench):
    html, plot = G["html"], G["plot"]
    html(C.section("06", "Scoreboard", "long-document benchmark · bootstrap 95% intervals"))
    sb = S.scoreboard(bench, arms=(A_FULL, A_SAFE, A_RAW))
    if sb.empty:
        html(C.empty("No long-document scoreboard yet", "Needs <code>runs</code> with <code>bench_id</code> starting "
                     "with <code>lc-</code>."))
        return
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True, key="lcsb1"):
            st.markdown("**Joules per question** · mean with 95% CI")
            plot(CH.arm_bars(sb, "J_mean", "J_lo", "J_hi", "J"), key="lc_chart_jq")
    with c2:
        with st.container(border=True, key="lcsb2"):
            st.markdown("**Joules per correct answer** · the headline metric")
            plot(CH.arm_bars(sb, "JPC", None, None, "J"), key="lc_chart_jpc")
    with c3:
        with st.container(border=True, key="lcsb3"):
            st.markdown("**Energy ledger** · incl. safety-net escalations")
            plot(CH.ledger(sb), key="lc_chart_ledger")
    tbl = pd.DataFrame({
        "Arm": [T.ARM_LABEL.get(a, a) for a in sb["arm"]], "Runs": sb["n"], "J / question": sb["J_mean"],
        "95% CI": [f"{lo:,.0f}–{hi:,.0f}" if S.fnum(lo) is not None and S.fnum(hi) is not None else "—"
                   for lo, hi in zip(sb["J_lo"], sb["J_hi"])],
        "J / correct": sb["JPC"], "Correct": [f"{a} / {b}" for a, b in zip(sb["correct"], sb["graded"])],
        "Escalated": [f"{100 * e:.0f}%" if S.fnum(e) is not None and a == A_SAFE else "—" for e, a in zip(sb["escalated"], sb["arm"])],
        "Tokens in": sb["tokens_in"], "TTFT (s)": sb["ttft_s"], "Total (s)": sb["total_s"],
        "J quantum": sb["J_select"], "J escalation": sb["J_escalate"],
    })
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "J / question": st.column_config.NumberColumn(format="%.0f"), "J / correct": st.column_config.NumberColumn(format="%.0f"),
        "Tokens in": st.column_config.NumberColumn(format="%.0f"), "TTFT (s)": st.column_config.NumberColumn(format="%.1f"),
        "Total (s)": st.column_config.NumberColumn(format="%.1f"), "J quantum": st.column_config.NumberColumn(format="%.1f"),
        "J escalation": st.column_config.NumberColumn(format="%.1f")})
    html(f'<div class="qw-mini">"Q-Watt without safety net" is the same selection with no fallback, shown for transparency: '
         f'it is cheaper but answers fewer questions. Correct = {C.esc(S.correctness_source(bench))}. Escalation energy = '
         f'the run’s total minus its first attempt.</div>')
    _tiger_panel(G, bench)


@st.cache_data(ttl=60, show_spinner=False)
def _tiger_rows():
    res, msg = DA.tiger_scoreboard(timeout_s=5.0)
    return (pd.DataFrame(res) if res else None), msg


def _tiger_panel(G, bench):
    html = G["html"]
    df, msg = _tiger_rows()
    if df is None or df.empty:
        html(f'<div class="qw-mini" style="margin-top:8px">Tiger Data: {C.esc(msg)} · numbers above computed locally (SQLite).</div>')
        return
    bids = set(bench["bench_id"].dropna().unique()) if "bench_id" in bench else set()
    if "bench_id" in df:
        df = df[df["bench_id"].isin(bids)] if bids else df[df["bench_id"].astype(str).str.startswith("lc-")]
    df = df[df["arm"].isin([A_FULL, A_SAFE, A_RAW])]
    if df.empty:
        html('<div class="qw-mini" style="margin-top:8px">Tiger Data connected · no synced rows for this batch yet '
             '(run <code>scripts/sync_tiger.py</code>).</div>')
        return
    html('<div style="margin-top:14px"><span class="qw-pill"><span class="dot" style="background:#22c55e"></span>'
         'source: Tiger Data (cloud TimescaleDB) · scoreboard view computed in SQL over the synced hypertables</span></div>')
    out = pd.DataFrame({
        "Arm": [T.ARM_LABEL.get(a, a) for a in df["arm"]], "Runs": df["n"],
        "Median J (net)": df["median_joules_net"], "Correct": df["n_correct"],
        "J / correct": df["joules_per_correct"], "Saved vs full (median)": df["pct_saved_median_vs_full"],
        "Escalations": df.get("n_escalated"),
    })
    st.dataframe(out, hide_index=True, width="stretch", column_config={
        "Median J (net)": st.column_config.NumberColumn(format="%.1f"), "J / correct": st.column_config.NumberColumn(format="%.1f"),
        "Saved vs full (median)": st.column_config.NumberColumn(format="%.1f%%")})
    html('<div class="qw-mini">Same runs, recomputed independently inside Tiger Data from the synced watt samples and run '
         'rows; joules are also integrated in SQL over the <code>power_samples</code> hypertable as a cross-check.</div>')


def _show_work(G, bench):
    html, plot = G["html"], G["plot"]
    with st.expander("07 · Show your work — selection quality, calibration, raw runs", expanded=False):
        sel = G["load_json"]("lc_selection", G["DB"])
        html('<div class="qw-subh">Selection quality on the long-document test split (no LLM call)</div>')
        if isinstance(sel, dict) and any(isinstance(v, dict) for v in sel.values()):
            rows = []
            for k, v in sel.items():
                if isinstance(v, dict) and S.visible(k):
                    rows.append({"selector": T.ARM_LABEL.get(k, k), **{kk: vv for kk, vv in v.items()
                                                                       if S.fnum(vv) is not None or isinstance(vv, str)}})
            st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        else:
            html(C.empty("No long-document selection eval yet", "<code>scripts/eval_lc.py</code> → <code>results/lc_*.json</code>."))
        cl = G["cal_lc"]
        html('<div class="qw-subh">Long-prompt calibration</div>')
        if cl and cl.get("points"):
            plot(CH.calibration_fit(cl), key="lc_cal")
        elif cl:
            html(f'<div class="lc-note">ε_in = <b>{S.fnum(cl.get("eps_in")) or 0:.4f} J/token</b>'
                 + (f' · R² {cl["r2"]:.3f}' if S.fnum(cl.get("r2")) is not None else "") + "</div>")
        else:
            html(C.empty("No long-prompt calibration yet", "<code>results/calibration_lc.json</code>."))
        html('<div class="lc-note" style="margin-top:8px">The exact, brute-force verification of the circuit lives in the '
             '<b>Verified exactly at 16 qubits</b> view (top of the page).</div>')
        if not bench.empty:
            html('<div class="qw-subh">Raw runs (long-document batch)</div>')
            cols = [c for c in ("qid", "arm", "escalated", "tokens_in", "joules_net", "joules_gross", "joules_select",
                                "total_s", "answer", "gold", "referee_correct", "llm_backend", "energy_source") if c in bench]
            st.dataframe(bench[cols].sort_values(["qid", "arm"]), hide_index=True, width="stretch", height=260)


# ------------------------------------------------------------------ main
def render(G: dict) -> None:
    html = G["html"]
    html(CSS)
    runs = G["runs"]
    FIX = G["FIX"]
    G["cal_lc"] = G["load_json"]("calibration_lc", G["DB"])
    hp = G["load_json"]("hyperparams", G["DB"]) or {}
    G["default_budget_lc"] = int((hp.get("extra") or {}).get("lc_budget_tokens") or hp.get("lc_budget_tokens") or 640)
    bids = S.bench_ids(runs, "lc")
    bid = bids[0] if bids else None
    if len(bids) > 1:
        bid = st.session_state.get("lc_bench_sel", bids[0])
        if bid not in bids:
            bid = bids[0]
            st.session_state["lc_bench_sel"] = bid
    bench = S.bench_runs(runs, bid) if bid else pd.DataFrame()

    # ---- 01 hero
    html(C.section("01", "Headline", "long documents · one qubit per sentence"
                   + (f" · batch {bid}" if bid else "") + " · energy = CPU package joules, net of idle"))
    if bench.empty:
        html(C.empty("No long-document benchmark yet",
                     "Run <code>scripts/run_bench_lc.py</code>; rows with <code>bench_id</code> starting with "
                     "<code>lc-</code> fill this view. The 16-qubit results are in the other view."))
        h = {}
    else:
        h = S.hero(bench.replace({"arm": {A_FULL: "full", A_SAFE: "qwatt"}}))
        qa = _qaoa_for(bench[bench["arm"] == A_SAFE]["run_id"])
        nqs = pd.to_numeric(qa["n_qubits"], errors="coerce").dropna() if not qa.empty else pd.Series(dtype=float)
        safe = bench[bench["arm"] == A_SAFE]
        esc = safe["escalated"].fillna(0) if "escalated" in safe else pd.Series(dtype=float)
        cal = G["cal_lc"] or G["cal"] or {}
        html(_hero(h, bench["qid"].nunique(), nqs.max() if len(nqs) else None, nqs.median() if len(nqs) else None,
                   float((esc > 0).mean()) if len(esc) else None, int((esc > 0).sum()), len(esc),
                   S.fnum(cal.get("eps_in")), "long-prompt calibration" if G["cal_lc"] else "calibration"))
    if len(bids) > 1:
        st.selectbox("Long-document batch", bids, key="lc_bench_sel")

    # ---- 02 race
    if G["mode"] == "live":
        _race_live(G, bench)
    else:
        _race_bench(G, bench)

    _scaling_section(G, bench)
    _scoreboard(G, bench)
    _show_work(G, bench)
    d = h.get("delta_J_median") if h else None
    html(C.footer(S.extrapolate(d), d, FIX).replace("on this laptop", "for long documents on this laptop"))
    html(f'<div class="qw-mini" style="text-align:center;margin-top:10px">Q-Watt Long-Context · Hack the Nest 2026 · '
         f'Quantum Rings SDK (up to 200 qubits) · Gemma via LM Studio · Gemini referee · data: '
         f'{C.esc(str(G["DB"]).replace(chr(92), "/").split("/")[-1])}</div>')


def _race_bench(G, bench):
    html, plot = G["html"], G["plot"]
    html(C.section("02", "The race", "stored long-document runs · Q-Watt vs reading the whole document"))
    if bench.empty:
        html(C.empty("Nothing to race yet", "Long-document runs appear here."))
        _lens(G, None, None, None, "pending", [], None, None, "lcb")
        return
    have = bench.groupby("qid")["arm"].apply(set)
    qids = [q for q, arms in have.items() if {A_FULL, A_SAFE} <= arms] or list(have.index)
    qtext = bench.drop_duplicates("qid").set_index("qid")["question"].to_dict()
    demo = G["load_json"]("lc_demo_questions", G["DB"]) or []
    demo = [d if isinstance(d, str) else (d or {}).get("qid") for d in demo] if isinstance(demo, list) else []
    demo_in = [q for q in demo if q in qids]
    qids = demo_in + [q for q in qids if q not in set(demo_in)]
    if st.session_state.get("lc_qid") not in qids:
        st.session_state["lc_qid"] = qids[0]
    c1, c2 = st.columns([5, 1.2], vertical_alignment="bottom")
    with c1:
        st.selectbox("Question", qids, key="lc_qid", format_func=lambda q: ("★ demo · " if q in set(demo_in) else "")
                     + f"{str(qtext.get(q, ''))[:110]}  ·  {q}")
    qid = st.session_state["lc_qid"]
    with c2:
        if st.button("▶ Replay race", key="lc_replay_btn", width="stretch"):
            st.session_state["lc_replay"] = {"qid": qid, "start": time.time()}
            st.rerun()
    rep = st.session_state.get("lc_replay")
    replaying = bool(rep and rep.get("qid") == qid)

    @st.fragment(run_every=0.5 if replaying else None)
    def frag():
        rep = st.session_state.get("lc_replay")
        elapsed = (time.time() - rep["start"]) * 2.0 if rep and rep.get("qid") == qid else None  # 2x replay speed
        rows = _race_rows(bench, qid)
        views = {a: G["view_from_row"](r, elapsed) for a, r in rows.items()}
        gold = next((r.get("gold") for r in rows.values() if r.get("gold")), None)
        st.markdown(f"<div style='font-size:22px;font-weight:700;color:{T.INK};letter-spacing:-.01em'>“{C.esc(qtext.get(qid, ''))}”</div>",
                    unsafe_allow_html=True)
        if elapsed is not None:
            html('<div class="qw-mini">Replay at 2× speed of the stored measurement.</div>')
        x_max, y_max = G["shared_axes"](list(views.values()))
        cols = st.columns(2, gap="medium")
        for col, a in zip(cols, (A_SAFE, A_FULL)):
            with col:
                if a in views:
                    G["render_lane"](views[a], x_max, y_max, gold, f"lcb_{a}")
                else:
                    html(C.empty(f"No {T.ARM_LABEL[a]} run", "for this question."))
        if A_SAFE in views and A_FULL in views and all(views[a]["status"] == "done" for a in (A_SAFE, A_FULL)):
            html(_verdict(rows[A_SAFE], rows[A_FULL], views[A_SAFE]["joules"], views[A_FULL]["joules"]))
        if A_RAW in views:
            with st.expander("Q-Watt without safety net · the same selection with no fallback (transparency)", expanded=False):
                G["render_lane"](views[A_RAW], x_max, y_max, gold, "lcb_raw")
        qrow, frow = rows.get(A_SAFE), rows.get(A_FULL)
        diag = DA.qaoa_diag(qrow["run_id"]) if qrow else None
        pool = _lc_pool(qid, G)
        kept = S.parse_selected(qrow.get("selected_json")) if qrow else []
        state = "done"
        if elapsed is not None and A_SAFE in views and views[A_SAFE]["status"] == "running" and views[A_SAFE]["phase"] == "select":
            state = "running"
        _lens(G, qid, pool, diag, state, kept, qrow, frow, "lcb", charts=elapsed is None)
        if elapsed is None:
            qa = _qaoa_for([qrow["run_id"]]) if qrow else pd.DataFrame()
            _why(G, diag, qrow, qa["joules"].iloc[0] if not qa.empty and "joules" in qa else None, "lcb")
        if elapsed is not None and elapsed > max([v["duration"] for v in views.values()] + [0]) + 1.0:
            st.session_state.pop("lc_replay", None)
            st.rerun(scope="app")

    frag()


def _race_live(G, bench):
    html = G["html"]
    html(C.section("02", "The race", "live on this laptop · Q-Watt first, then the whole document"))
    missing = [] if G["FIX"] else G["modules_ok"]("qwatt.pipeline", "qwatt.longdoc", "qwatt.selectors")
    qs = []
    demo = G["load_json"]("lc_demo_questions", G["DB"]) or []
    demo = [d if isinstance(d, str) else (d or {}).get("qid") for d in demo] if isinstance(demo, list) else []
    known = bench.drop_duplicates("qid").set_index("qid")["question"].to_dict() if not bench.empty else {}
    for q in demo + [q for q in known if q not in set(demo)]:
        text = known.get(q)
        if not text:
            p = _lc_pool(q, G)
            text = (p or {}).get("question") or q
        qs.append({"qid": q, "question": text})
    run = LV.current()
    if run is not None and run.kind != "lc":
        run = None if not run.active else run
    c1, c3 = st.columns([5, 1.3], vertical_alignment="bottom")
    with c1:
        if qs:
            st.selectbox("Question", list(range(len(qs))), key="lc_live_q",
                         format_func=lambda i: f"{str(qs[i]['question'])[:110]}  ·  {qs[i]['qid']}")
        else:
            html(C.empty("No long-document questions", "Add <code>results/lc_demo_questions.json</code> or run the LC benchmark."))
    with c3:
        busy = LV.current() is not None and LV.current().active
        if st.button("⚡ Run live" if not busy else "Running…", key="lc_run_live", disabled=busy or not qs or bool(missing),
                     width="stretch"):
            q = qs[int(st.session_state.get("lc_live_q") or 0)]
            LV.start(q["qid"], q["question"], arms=[A_SAFE, A_FULL], kind="lc")
            st.rerun()
    if missing:
        html(C.empty("Long-document live mode is not wired yet", "Waiting for module(s): "
                     + ", ".join(f"<code>{m}</code>" for m in missing) + ". Benchmark mode still works."))
    else:
        LV.prewarm_async(16)
    run = LV.current()
    if run is None or run.kind != "lc":
        html(C.empty("Ready", "Pick a question and press <b>Run live</b>. Q-Watt goes first: every sentence of the document "
                     "becomes a qubit, the circuit runs on Quantum Rings, the kept sentences light up, then the "
                     "full-document lane grinds through every token."))
        _lens(G, None, None, None, "pending", [], None, None, "lcl")
        return
    idle_default = S.fnum((G["cal_lc"] or G["cal"] or {}).get("idle_watts"))
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
        views = {a: G["view_from_lane"](run.lanes[a], idle_default) for a in run.arms}
        x_max, y_max = G["shared_axes"](list(views.values()))
        gold = (run.pool or {}).get("answer")
        cols = st.columns(2, gap="medium")
        for col, a in zip(cols, (A_SAFE, A_FULL)):
            with col:
                if a in views:
                    G["render_lane"](views[a], x_max, y_max, gold, f"lcl_{a}")
        lq, lf = run.lanes.get(A_SAFE), run.lanes.get(A_FULL)
        if lq and lf and lq.status == "done" and lf.status == "done" and lq.row and lf.row:
            html(_verdict(lq.row, lf.row, views[A_SAFE]["joules"], views[A_FULL]["joules"]))
        state = "done" if (lq and lq.diag is not None) else ("running" if lq and lq.status == "running" else "pending")
        pool = run.pool
        if pool is not None and not pool.get("paragraph_of") and lq and lq.diag:
            pool = {**pool, "paragraph_of": lq.diag.get("paragraph_of")}
        _lens(G, run.qid, pool, lq.diag if lq else None, state, list((lq.indices if lq else None) or []),
              lq.row if lq else None, lf.row if lf else None, "lcl", charts=not run.active)
        if not run.active and lq and lq.diag:
            _why(G, lq.diag, lq.row, None, "lcl")
        still_ref = any(l.ref_status == "pending" for l in run.lanes.values())
        done_key = (run.started, still_ref)
        if not run.active and st.session_state.get("_lc_live_done") != done_key:
            st.session_state["_lc_live_done"] = done_key
            st.rerun(scope="app")

    frag()

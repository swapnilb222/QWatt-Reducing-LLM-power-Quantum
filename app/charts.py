"""Plotly figures. All use the 'qwatt' template from app.theme (transparent, hairline grid, one axis)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app import theme as T
from app.stats import fnum

PHASE_LABEL = {"select": "selecting", "prefill": "reading prompt", "generate": "generating",
               "escalate_prefill": "escalated: full document", "escalate_generate": ""}


def _base(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(template="qwatt", height=height, showlegend=kw.pop("showlegend", False), **kw)
    return fig


# ---------------------------------------------------------------- watt trace
def watt_trace(samples: pd.DataFrame, t0: float, t_now: float, color: str, idle: float | None,
               markers: dict | None, x_max: float, y_max: float, live: bool = False,
               quantum_band: bool = True) -> go.Figure:
    fig = go.Figure()
    if samples is not None and not samples.empty:
        s = samples[(samples["ts"] >= t0 - 0.3) & (samples["ts"] <= t_now + 0.05)]
        x = (s["ts"] - t0).clip(lower=0).to_numpy()
        y = s["pkg_watts"].astype(float).to_numpy()
    else:
        x, y = np.array([]), np.array([])
    base = idle if idle is not None else 0.0
    # phase bands (very subtle) with labels on top
    for ph, (a, b) in (markers or {}).items():
        if ph not in PHASE_LABEL:
            continue
        b = t_now if b is None else b
        xa, xb = max(0.0, a - t0), max(0.0, b - t0)
        if xb - xa <= 0.02:
            continue
        if ph.startswith("escalate"):
            fig.add_vrect(x0=xa, x1=xb, fillcolor=T.rgba(T.FULL, 0.16), line_width=0, layer="below")
            if ph == "escalate_prefill":
                fig.add_annotation(x=xa, y=0.97, yref="paper", text="⤴ safety net: full document", showarrow=False,
                                   xanchor="left", yanchor="top", xshift=3, font=dict(size=12, color="#9a3412"))
            continue
        is_q = ph == "select" and quantum_band
        if ph == "select" and not quantum_band:
            continue
        fig.add_vrect(x0=xa, x1=xb, fillcolor=T.rgba(T.QUANTUM, 0.22) if is_q else "rgba(15,23,42,0.035)",
                      line_width=0, layer="below")
        if is_q or (xb - xa) / max(x_max, 1e-6) > 0.07:
            if is_q:
                fig.add_annotation(x=xa, y=0.97, yref="paper", text="⚛ circuit", showarrow=False, xanchor="left",
                                   yanchor="top", xshift=2, font=dict(size=12, color=T.QUANTUM))
            else:
                fig.add_annotation(x=(xa + xb) / 2, y=1.0, yref="paper", text=PHASE_LABEL[ph], showarrow=False,
                                   font=dict(size=12, color=T.MUTED), yanchor="bottom")
    if len(x):
        fig.add_trace(go.Scatter(x=x, y=np.full_like(y, base), mode="lines", line=dict(width=1, color=T.AXIS),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=np.maximum(y, base), mode="lines", line=dict(width=2.2, color=color, shape="spline",
                                 smoothing=0.6), fill="tonexty", fillcolor=T.rgba(color, 0.22),
                                 hovertemplate="%{x:.1f} s · %{y:.1f} W<extra></extra>", showlegend=False))
        if live:
            fig.add_trace(go.Scatter(x=[x[-1]], y=[max(y[-1], base)], mode="markers",
                                     marker=dict(size=10, color=color, line=dict(width=2, color=T.SURFACE)),
                                     hoverinfo="skip", showlegend=False))
    if idle is not None:
        fig.add_annotation(x=x_max, y=base, text=f"idle {idle:.1f} W", showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(size=11, color=T.MUTED))
    fig.update_xaxes(range=[0, x_max], ticksuffix=" s", showgrid=False)
    fig.update_yaxes(range=[0, y_max], ticksuffix=" W", nticks=4)
    return _base(fig, 190, margin=dict(l=4, r=6, t=20, b=4))


# ---------------------------------------------------------------- bitstring histogram
def bitstring_hist(diag: dict, chosen_bits: str | None, top: int = 10) -> go.Figure:
    samples = sorted(diag.get("samples") or [], key=lambda s: -int(s.get("count", 0)))[:top]
    shots = sum(int(s.get("count", 0)) for s in (diag.get("samples") or [])) or diag.get("shots") or 1
    exact = diag.get("exact_bits")
    samples = samples[::-1]  # most frequent on top
    ys = [s["bits"] for s in samples]
    xs = [int(s.get("count", 0)) for s in samples]
    colors = [T.QUANTUM if s.get("feasible", True) else T.NEUTRAL_A for s in samples]
    custom = [[fnum(s.get("cost")) or 0.0, s.get("tokens", 0), "within budget" if s.get("feasible", True) else "over budget",
               100 * int(s.get("count", 0)) / shots] for s in samples]
    fig = go.Figure(go.Bar(
        x=xs, y=ys, orientation="h", marker=dict(color=colors, cornerradius=4), width=0.62,
        customdata=custom,
        hovertemplate="<b>%{y}</b><br>%{x} shots (%{customdata[3]:.1f}%)<br>cost %{customdata[0]:.3f} · "
                      "%{customdata[1]} tokens · %{customdata[2]}<extra></extra>"))
    xmax = max(xs) if xs else 1
    for k, (s, x) in enumerate(zip(samples, xs)):
        tags = []
        if exact and s["bits"] == exact:
            tags.append("★ optimum")
        if chosen_bits and s["bits"] == chosen_bits and s["bits"] != exact:
            tags.append("◆ picked")
        label = f"  {x}  <span style='color:{T.MUTED}'>cost {fnum(s.get('cost')) or 0:+.2f}</span>"
        if tags:
            label += "   <b>" + " · ".join(tags) + "</b>"
        # numeric category index: a digit-only bitstring would be parsed as a number and wreck the range
        fig.add_annotation(x=x, y=k, text=label, showarrow=False, xanchor="left",
                           font=dict(size=13, color=T.INK))
    fig.update_yaxes(type="category", range=[-0.6, len(samples) - 0.4], showgrid=False, tickfont=dict(family=T.FONT_MONO, size=13, color=T.INK_2))
    fig.update_xaxes(range=[0, xmax * 2.35], showgrid=True, gridcolor=T.GRID, title_text="times measured (shots)")
    return _base(fig, 34 * max(4, len(samples)) + 50, margin=dict(l=4, r=8, t=4, b=4), bargap=0.25)


# ---------------------------------------------------------------- budget dial scatter
def frontier_scatter(diag: dict, budget: int, chosen: dict | None, frontier_pts: list) -> go.Figure:
    samples = diag.get("samples") or []
    fig = go.Figure()
    if samples:
        tok = np.array([s.get("tokens", 0) for s in samples], dtype=float)
        cost = np.array([fnum(s.get("cost")) or 0.0 for s in samples], dtype=float)
        cnt = np.array([int(s.get("count", 1)) for s in samples], dtype=float)
        inb = tok <= budget
        size = 6 + 10 * np.sqrt(cnt / cnt.max())
        for mask, col, name, op in ((~inb, T.NEUTRAL_A, "over the budget", 0.45), (inb, T.QUANTUM, "within the budget", 0.85)):
            fig.add_trace(go.Scatter(
                x=tok[mask], y=cost[mask], mode="markers", name=name,
                marker=dict(size=size[mask], color=T.rgba(col, op) if col.startswith("#") else col,
                            line=dict(width=1, color=T.SURFACE)),
                customdata=np.stack([cnt[mask]], axis=-1) if mask.any() else None,
                hovertemplate="%{x} tokens · cost %{y:.3f}<br>%{customdata[0]} shots<extra></extra>"))
    if frontier_pts:
        fx = [p[0] for p in frontier_pts]
        fy = [p[1] for p in frontier_pts]
        xmax = max(s.get("tokens", 0) for s in samples) if samples else fx[-1]
        fig.add_trace(go.Scatter(x=fx + [xmax], y=fy + [fy[-1]], mode="lines", name="best at each budget",
                                 line=dict(width=2, color=T.INK_2, shape="hv"), hoverinfo="skip"))
    fig.add_vline(x=budget, line=dict(width=1.5, color=T.QWATT))
    fig.add_annotation(x=budget, y=1, yref="paper", text=f"budget {budget} tokens", showarrow=False, xanchor="left",
                       yanchor="top", xshift=6, font=dict(size=12, color=T.INK))
    if chosen:
        fig.add_trace(go.Scatter(x=[chosen.get("tokens")], y=[chosen.get("cost")], mode="markers", name="picked",
                                 marker=dict(size=18, color=T.QWATT, line=dict(width=3, color=T.SURFACE), symbol="diamond"),
                                 hovertemplate="picked: %{x} tokens · cost %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="tokens kept", showgrid=False)
    fig.update_yaxes(title_text="QUBO cost (lower is better)")
    return _base(fig, 330, showlegend=True, margin=dict(l=4, r=8, t=30, b=4))


# ---------------------------------------------------------------- scoreboard
def arm_bars(sb: pd.DataFrame, col: str, lo: str | None, hi: str | None, title_unit: str) -> go.Figure:
    d = sb.dropna(subset=[col]) if not sb.empty else sb
    fig = go.Figure()
    if d.empty:
        return _base(fig, 220)
    d = d.iloc[::-1]
    labels = [T.ARM_LABEL.get(a, a) for a in d["arm"]]
    err = None
    if lo and hi and d[lo].notna().all() and d[hi].notna().all():
        err = dict(type="data", symmetric=False, array=(d[hi] - d[col]).to_numpy(),
                   arrayminus=(d[col] - d[lo]).to_numpy(), color=T.INK_2, thickness=1.5, width=5)
    fig.add_trace(go.Bar(x=d[col], y=labels, orientation="h", width=0.55,
                         marker=dict(color=[T.ARM_COLOR.get(a, T.INK_2) for a in d["arm"]], cornerradius=4),
                         error_x=err, cliponaxis=False,
                         hovertemplate="%{y}: %{x:.2f} " + title_unit + "<extra></extra>"))
    ends = (d[hi] if err else d[col]).to_numpy()
    for lab, v, e in zip(labels, d[col], ends):
        fig.add_annotation(x=e, y=lab, text=f"  {v:,.1f} {title_unit}", showarrow=False, xanchor="left",
                           font=dict(color=T.INK, size=14))
    fig.update_xaxes(showgrid=True, gridcolor=T.GRID, range=[0, float((d[hi] if err else d[col]).max()) * 1.35])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=14, color=T.INK_2))
    return _base(fig, 60 + 46 * len(d), margin=dict(l=4, r=12, t=4, b=4))


def ledger(sb: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if sb.empty:
        return _base(fig, 220)
    d = sb.iloc[::-1]
    labels = [T.ARM_LABEL.get(a, a) for a in d["arm"]]
    parts = [("J_select", "quantum / selector", T.QUANTUM), ("J_prefill", "reading prompt (prefill)", T.NEUTRAL_A),
             ("J_generate", "generating", T.NEUTRAL_B)]
    if "J_escalate" in d and d["J_escalate"].fillna(0).sum() > 0:
        parts.append(("J_escalate", "safety-net escalation (full document)", T.FULL))
    for col, name, color in parts:
        vals = d[col].fillna(0).clip(lower=0).to_numpy()
        fig.add_trace(go.Bar(y=labels, x=vals, name=name, orientation="h", width=0.55,
                             marker=dict(color=color, line=dict(width=2, color=T.BG)),
                             hovertemplate="%{y} · " + name + ": %{x:.2f} J<extra></extra>"))
    tot = d[[p[0] for p in parts]].fillna(0).clip(lower=0).sum(axis=1)
    for lab, v in zip(labels, tot):
        fig.add_annotation(x=v, y=lab, text=f"  {v:,.1f} J", showarrow=False, xanchor="left", font=dict(color=T.INK, size=14))
    fig.update_layout(barmode="stack", legend=dict(orientation="h", traceorder="normal", y=1.02, x=0,
                                                   yanchor="bottom", font=dict(size=12)))
    fig.update_xaxes(showgrid=True, gridcolor=T.GRID, ticksuffix=" J", range=[0, float(tot.max()) * 1.25 if len(tot) else 1])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=14, color=T.INK_2))
    return _base(fig, 90 + 46 * len(d), showlegend=True, margin=dict(l=4, r=12, t=30, b=4))


# ---------------------------------------------------------------- show your work
SOLVER_COLOR = {"qaoa": T.QUANTUM, "qwatt": T.QUANTUM, "qaoa_qr": T.QUANTUM, "qaoa_exact_sim": "#a79ff0", "exact": T.INK_2, "anneal": T.NEUTRAL_B,
                "greedy": T.GREEDY, "random_k": T.NEUTRAL_A}


def approx_ratio_box(dist: dict[str, list[float]]) -> go.Figure:
    fig = go.Figure()
    known = ("random_k", "greedy", "anneal", "exact", "qaoa_exact_sim", "qaoa", "qwatt", "qaoa_qr")
    order = [k for k in known if k in dist] + [k for k in dist if k not in known]
    for k in order:
        v = [x for x in dist[k] if x is not None and not math.isnan(x)]
        if not v:
            continue
        c = SOLVER_COLOR.get(k, T.INK_2)
        fig.add_trace(go.Box(x=v, name=T.ARM_LABEL.get(k, k), orientation="h", boxpoints="all", jitter=0.4, pointpos=0,
                             marker=dict(size=6, color=T.rgba(c, 0.75)), line=dict(color=c, width=1.5),
                             fillcolor=T.rgba(c, 0.12), hoverinfo="x+name"))
    fig.update_xaxes(title_text="approximation ratio (1 = exact optimum)", showgrid=True, gridcolor=T.GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=13, color=T.INK_2))
    return _base(fig, 70 + 52 * max(1, len(fig.data)), margin=dict(l=4, r=8, t=4, b=4))


def calibration_fit(cal: dict) -> go.Figure:
    pts = cal.get("points") or []
    fig = go.Figure()
    if not pts:
        return _base(fig, 280)
    x = np.array([fnum(p.get("tokens_in")) or 0 for p in pts])
    y = np.array([fnum(p.get("joules_net")) or 0 for p in pts])
    tout = np.array([fnum(p.get("tokens_out")) or 0 for p in pts])
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="measured run",
                             marker=dict(size=9, color=T.TOPK, line=dict(width=2, color=T.SURFACE)),
                             customdata=tout, hovertemplate="%{x} tokens in · %{customdata} out<br>%{y:.1f} J<extra></extra>"))
    e0, ei, eo = fnum(cal.get("e0")) or 0, fnum(cal.get("eps_in")) or 0, fnum(cal.get("eps_out")) or 0
    xs = np.linspace(0, x.max() * 1.03, 50)
    fig.add_trace(go.Scatter(x=xs, y=e0 + ei * xs + eo * float(np.mean(tout)), mode="lines", name="fit",
                             line=dict(width=2, color=T.INK_2), hoverinfo="skip"))
    fig.add_annotation(x=xs[-1], y=e0 + ei * xs[-1] + eo * float(np.mean(tout)), xanchor="right", yanchor="bottom",
                       text=f"{ei:.4f} J / input token", showarrow=False, font=dict(color=T.INK, size=13))
    fig.update_xaxes(title_text="input tokens", showgrid=False)
    fig.update_yaxes(title_text="joules (net of idle)")
    return _base(fig, 300, margin=dict(l=4, r=8, t=4, b=4))


def sensor_crosscheck(samples: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if samples.empty:
        return _base(fig, 260)
    t0 = samples["ts"].min()
    x = samples["ts"] - t0
    fig.add_trace(go.Scatter(x=x, y=samples["pkg_watts"], mode="lines", name="CPU package (RAPL)",
                             line=dict(width=2, color=T.TOPK)))
    if samples["sys_watts"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=samples["sys_watts"], mode="lines", name="battery discharge (whole system)",
                                 line=dict(width=2, color=T.GREEDY)))
    fig.update_xaxes(ticksuffix=" s", showgrid=False)
    fig.update_yaxes(ticksuffix=" W")
    return _base(fig, 260, showlegend=True, margin=dict(l=4, r=8, t=30, b=4))


def selection_recall(sel: dict) -> go.Figure:
    rows = [(k, v) for k, v in sel.items() if isinstance(v, dict) and fnum(v.get("sf_recall_mean")) is not None]
    order = {"random_k": 0, "topk": 1, "greedy": 2, "anneal": 3, "exact": 4, "qwatt": 5, "full": 6}
    rows.sort(key=lambda kv: order.get(kv[0], 9))
    fig = go.Figure()
    if not rows:
        return _base(fig, 220)
    labels = [T.ARM_LABEL.get(k, k) for k, _ in rows]
    vals = [v["sf_recall_mean"] for _, v in rows]
    ci = [v.get("sf_recall_ci") or [None, None] for _, v in rows]
    has_ci = all(c and c[0] is not None for c in ci)
    colors = [T.ARM_COLOR.get(k, T.NEUTRAL_B) if k in ("topk", "greedy", "qwatt", "full") else T.NEUTRAL_B for k, _ in rows]
    fig.add_trace(go.Bar(x=vals, y=labels, orientation="h", width=0.55, marker=dict(color=colors, cornerradius=4),
                         error_x=dict(type="data", symmetric=False, array=[c[1] - v for c, v in zip(ci, vals)],
                                      arrayminus=[v - c[0] for c, v in zip(ci, vals)], color=T.INK_2, thickness=1.5,
                                      width=5) if has_ci else None,
                         text=[f"{v:.2f}" for v in vals], textposition="outside", cliponaxis=False,
                         textfont=dict(color=T.INK, size=13), hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.15], title_text="supporting-fact recall at the same token budget", showgrid=True,
                     gridcolor=T.GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=13, color=T.INK_2))
    return _base(fig, 60 + 42 * len(rows), margin=dict(l=4, r=12, t=4, b=4))


def cost_shift(dist: dict, amp_diag: float | None = None) -> go.Figure:
    """Where the circuit's shots land vs random guessing, by QUBO cost (better = right)."""
    e = dist["edges"]
    mids = (e[:-1] + e[1:]) / 2
    fig = go.Figure()
    fig.add_vrect(x0=float(e[0]) - 1e-9, x1=dist["thr"], fillcolor=T.rgba(T.QWATT, 0.10), line_width=0, layer="below")
    fig.add_annotation(x=dist["thr"], y=0.02, yref="paper", text="best 1% →", showarrow=False,
                       xanchor="right", yanchor="bottom", xshift=-4, font=dict(size=12, color=T.QWATT))
    fig.add_trace(go.Scatter(x=mids, y=dist["uniform"], mode="lines", name="random guessing (all 2^n subsets)",
                             line=dict(width=2, color=T.NEUTRAL_B, shape="hvh"), fill="tozeroy",
                             fillcolor=T.rgba(T.NEUTRAL_B, 0.10),
                             hovertemplate="cost %{x:.2f}: %{y:.1f}% of random subsets<extra></extra>"))
    fig.add_trace(go.Scatter(x=mids, y=dist["qaoa"], mode="lines", name="QAOA circuit shots (Quantum Rings)",
                             line=dict(width=2.5, color=T.QUANTUM, shape="hvh"), fill="tozeroy",
                             fillcolor=T.rgba(T.QUANTUM, 0.22),
                             hovertemplate="cost %{x:.2f}: %{y:.1f}% of shots<extra></extra>"))
    if dist.get("exact_cost") is not None:
        fig.add_vline(x=dist["exact_cost"], line=dict(width=1.5, color=T.INK))
        fig.add_annotation(x=dist["exact_cost"], y=1.0, yref="paper", text="★ best selection within budget",
                           showarrow=False, xanchor="right", yanchor="top", xshift=-6, align="right",
                           font=dict(size=12, color=T.INK), bgcolor="rgba(255,255,255,0.9)")
    fig.update_xaxes(autorange="reversed", title_text="QUBO cost of the measured selection   (← worse · better →)",
                     showgrid=False, zeroline=False)
    fig.update_yaxes(ticksuffix="%", title_text="share of outcomes", nticks=5)
    return _base(fig, 330, showlegend=True, margin=dict(l=4, r=8, t=34, b=4))

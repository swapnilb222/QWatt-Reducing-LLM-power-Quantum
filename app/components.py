"""HTML building blocks (rendered with st.html; no JavaScript, no CDN)."""
from __future__ import annotations

import html
import math

from app import theme as T
from app.stats import fnum

CAPTION = ("Each sentence is a qubit. 1 means keep. The circuit is tuned so good combinations are the "
           "most likely to be measured.")


def esc(s) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def fmt_j(v, digits: int | None = None) -> str:
    v = fnum(v)
    if v is None:
        return "—"
    if digits is None:
        digits = 0 if abs(v) >= 100 else 1
    return f"{v:,.{digits}f}"


def fmt_s(v) -> str:
    v = fnum(v)
    return "—" if v is None else (f"{v:.2f} s" if v < 10 else f"{v:.1f} s")


def fmt_i(v) -> str:
    v = fnum(v)
    return "—" if v is None else f"{int(round(v)):,}"


# ---------------------------------------------------------------- top bar & banners
def topbar(mode: str, pills: list[tuple[str, str]]) -> str:
    mode_pill = ('<span class="qw-pill live"><span class="dot"></span>LIVE · running on this laptop</span>'
                 if mode == "live" else
                 '<span class="qw-pill bench"><span class="dot"></span>BENCHMARK · stored runs</span>')
    extra = "".join(f'<span class="qw-pill {cls}"><span class="dot"></span>{esc(txt)}</span>' for cls, txt in pills)
    return f"""
<div class="qw-top">
  <div class="qw-brand">
    <div class="qw-logo"><span>Q</span></div>
    <div><div class="qw-title">Q-<b>Watt</b></div>
      <div class="qw-sub">A quantum circuit decides which tokens are worth the electricity.</div></div>
  </div>
  <div class="qw-pills">{mode_pill}{extra}</div>
</div>"""


def synthetic_banner() -> str:
    return ('<div class="qw-synth">⚠ SYNTHETIC PREVIEW DATA — NOT MEASUREMENTS'
            '<small>QWATT_DASH_FIXTURE=1 · results/dev_fixture.sqlite · every number on this page is generated</small></div>')


def flags(items: list[str]) -> str:
    if not items:
        return ""
    return '<div style="display:flex;gap:8px;flex-wrap:wrap">' + "".join(
        f'<span class="qw-flag">⚑ {esc(t)}</span>' for t in items) + "</div>"


def section(num: str, title: str, sub: str = "") -> str:
    return (f'<div class="qw-h"><span class="n">{esc(num)}</span><h2>{esc(title)}</h2>'
            f'<span class="s">{esc(sub)}</span></div>')


def empty(title: str, body: str) -> str:
    return f'<div class="qw-empty"><b>{esc(title)}</b>{body}</div>'


# ---------------------------------------------------------------- hero
def hero(h: dict, cal: dict | None, ok_src: str, n_q: int) -> str:
    sav = h.get("saving")
    if sav is None:
        sav_v = "—"
        sav_sub = "needs paired Full + Q-Watt benchmark runs"
    else:
        sav_v = f"{100 * sav:.0f}<small>%</small>"
        lo, hi = h.get("saving_lo"), h.get("saving_hi")
        ci = f"95% CI {100 * lo:.0f}–{100 * hi:.0f}%" if lo is not None and hi is not None else "CI needs ≥2 questions"
        n_tot = h['n_pairs'] + (h.get('n_excluded') or 0)
        sav_sub = esc(f"median paired saving, net of idle · {ci} · {h['n_pairs']} of {n_tot} questions")
        extra = []
        if h.get("n_excluded"):
            extra.append(f"{h['n_excluded']} excluded: its idle baseline read above the run's own power (net ≤ 0)")
        if h.get("gross_saving") is not None:
            extra.append(f"gross, no idle subtraction: {100 * h['gross_saving']:.0f}% over all {n_tot}")
        if extra:
            sav_sub += '<br><span style="opacity:.85">' + esc(" · ".join(extra)) + "</span>"

    def duo(arm, v):
        return (f'<div><div class="k"><span class="sw" style="background:{T.ARM_COLOR[arm]}"></span>'
                f'{T.ARM_LABEL[arm]}</div><div class="v">{fmt_j(v)}<small>J</small></div></div>')

    tr = h.get("tok_removed")
    tok_v = "—" if tr is None else f"{tr:,.0f}"
    tok_sub = ("median input tokens removed per question" if tr is None or h.get("tok_frac") is None else
               f"median per question · {100 * h['tok_frac']:.0f}% of the full prompt")
    okq, nq, okf, nf = h.get("ok_qwatt", 0), h.get("n_qwatt", 0), h.get("ok_full", 0), h.get("n_full", 0)
    ok_v = "—" if not nq else f"{okq}<small>of {nq}</small>"
    ok_sub = f"Q-Watt answers correct · Full context: {okf} of {nf}" if nf else "Q-Watt answers correct"
    if ok_src:
        ok_sub += f" · judged by {ok_src}"

    if cal and fnum(cal.get("eps_in")) is not None:
        r2 = fnum(cal.get("r2"))
        npts = len(cal.get("points") or [])
        cal_line = (f'Measured on this laptop: <b>{cal["eps_in"]:.4f} J</b> per input token'
                    + (f' · <b>{cal["eps_out"]:.3f} J</b> per output token' if fnum(cal.get("eps_out")) else "")
                    + (f' · fit R² {r2:.3f} over {npts} runs' if r2 is not None else "")
                    + f' · idle {fmt_j(cal.get("idle_watts"))} W · Intel Core Ultra 7 265U (15 W class)')
    else:
        cal_line = ('<span class="qw-flag">⚑ Not calibrated yet</span> run <code>scripts/calibrate.py</code> to '
                    'measure joules per token on this laptop')
    return f"""
<div class="qw-hero">
  <div class="qw-card qw-stat primary"><div class="lbl">Energy saved by Q-Watt vs Full context</div>
    <div class="val">{sav_v}</div><div class="sub">{sav_sub}</div></div>
  <div class="qw-card qw-stat"><div class="lbl">Joules per correct answer</div>
    <div class="qw-duo">{duo("full", h.get("jpc_full"))}{duo("qwatt", h.get("jpc_qwatt"))}</div>
    <div class="sub">total joules ÷ correct answers, per lane</div></div>
  <div class="qw-card qw-stat"><div class="lbl">Tokens removed</div><div class="val">{tok_v}</div>
    <div class="sub">{esc(tok_sub)}</div></div>
  <div class="qw-card qw-stat"><div class="lbl">Answers still correct</div><div class="val">{ok_v}</div>
    <div class="sub">{esc(ok_sub)}</div></div>
</div>
<div class="qw-cal">{cal_line}</div>"""


# ---------------------------------------------------------------- lanes
PHASES = [("select", "selecting"), ("prefill", "reading prompt"), ("generate", "generating"), ("done", "done")]


def lane_header(arm: str, joules, joules_note: str, tag: str = "") -> str:
    c = T.ARM_COLOR.get(arm, T.INK_2)
    return f"""
<div class="qw-lane-h">
  <div class="qw-lane-name"><span class="bar" style="background:{c};box-shadow:0 0 16px {T.rgba(c, .7)}"></span>
    {esc(T.ARM_LABEL.get(arm, arm))}<span class="tag">{esc(tag)}</span></div>
  <div class="qw-joules"><div class="v">{fmt_j(joules)}<small>J</small></div><div class="s">{esc(joules_note)}</div></div>
</div>"""


LC_PHASES = [("select", "selecting"), ("prefill", "reading excerpt"), ("generate", "generating"),
             ("escalate", "safety net: reading full document"), ("done", "done")]


def phase_chips(arm: str, phase: str | None, status: str, escalated: bool = False) -> str:
    c = T.ARM_COLOR.get(arm, T.INK_2)
    if arm in ("qwatt_safe", "qwatt_lc", "full_lc"):
        return _lc_chips(arm, phase, status, escalated, c)
    keys = [p for p, _ in PHASES]
    cur = keys.index(phase) if phase in keys else (-1 if status != "done" else len(keys) - 1)
    out = []
    for i, (k, lab) in enumerate(PHASES):
        cls = "qw-chip"
        if arm == "full" and k == "select":
            cls += " skip"
            lab = "no selection"
        elif status == "done" and k == "done":
            cls += " on"
        elif i < cur or status == "done":
            cls += " past"
        elif i == cur and status == "running":
            cls += " on"
        out.append(f'<span class="{cls}" style="--c:{c}">{esc(lab)}</span>')
    return '<div class="qw-chips">' + "".join(out) + "</div>"


def _lc_chips(arm, phase, status, escalated, c) -> str:
    keys = [p for p, _ in LC_PHASES]
    ph = "escalate" if (phase or "").startswith("escalate") else phase
    cur = keys.index(ph) if ph in keys else (-1 if status != "done" else len(keys) - 1)
    out = []
    for i, (k, lab) in enumerate(LC_PHASES):
        cls = "qw-chip"
        if arm == "full_lc" and k == "select":
            cls, lab = cls + " skip", "no selection"
        elif arm == "full_lc" and k == "prefill":
            lab = "reading full document"
        if k == "escalate":
            if arm != "qwatt_safe":
                continue
            if not escalated and (status == "done" or (cur > i)):
                out.append(f'<span class="qw-chip skip" style="--c:{c}">safety net not needed</span>')
                continue
            if not escalated:
                out.append(f'<span class="qw-chip" style="--c:{T.FULL};opacity:.55">safety net (if the excerpt is not enough)</span>')
                continue
            ce = T.FULL
            if status == "running" and ph == "escalate":
                out.append(f'<span class="qw-chip on" style="--c:{ce}">⤴ escalated: reading full document</span>')
            else:
                out.append(f'<span class="qw-chip past" style="--c:{ce};color:#ffcfb3">escalated: read full document</span>')
            continue
        if status == "done" and k == "done":
            cls += " on"
        elif i < cur or status == "done":
            cls += " past"
        elif i == cur and status == "running":
            cls += " on"
        out.append(f'<span class="{cls}" style="--c:{c}">{esc(lab)}</span>')
    return '<div class="qw-chips">' + "".join(out) + "</div>"


def lane_stats(row: dict) -> str:
    tin, tout = fmt_i(row.get("tokens_in")), fmt_i(row.get("tokens_out"))
    return f"""
<div class="qw-kv">
  <div><div class="k">tokens in</div><div class="v">{tin}</div></div>
  <div><div class="k">tokens out</div><div class="v">{tout}</div></div>
  <div><div class="k">first token</div><div class="v">{fmt_s(row.get("ttft_s"))}</div></div>
  <div><div class="k">total time</div><div class="v">{fmt_s(row.get("total_s"))}</div></div>
</div>"""


def referee_badge(ok: bool | None, reason: str | None, graded_by: str | None) -> str:
    if ok is None:
        return '<span class="qw-badge pend">… awaiting referee</span>'
    tip = esc(reason or "")
    who = esc(graded_by or "")
    if ok:
        return f'<span class="qw-badge ok" title="{tip}">✓ correct <span style="opacity:.7;font-weight:600">· {who}</span></span>'
    return f'<span class="qw-badge bad" title="{tip}">✗ not correct <span style="opacity:.7;font-weight:600">· {who}</span></span>'


def answer_box(text: str | None, streaming: bool, badge: str, gold: str | None) -> str:
    if text:
        body = f'<div class="a">{esc(text)}{"<span class=cursor></span>" if streaming else ""}</div>'
    else:
        body = f'<div class="a dim">{"waiting for the first token…" if streaming else "no answer yet"}</div>'
    gold_html = f'<span class="qw-mini">gold: <b style="color:{T.INK}">{esc(gold)}</b></span>' if gold else ""
    return (f'<div class="qw-answer"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px">'
            f'<span class="q">answer</span><span style="display:flex;gap:10px;align-items:center">{gold_html}{badge}</span></div>'
            f'{body}</div>')


def verdict(row_q: dict, row_f: dict, Jq, Jf, select_j, eps_in, removed) -> str:
    Jq, Jf = fnum(Jq), fnum(Jf)
    if Jq is None or Jf is None or Jf <= 0:
        return ""
    save = 1 - Jq / Jf
    gq, gf = fnum((row_q or {}).get("joules_gross")), fnum((row_f or {}).get("joules_gross"))
    gs = (1 - gq / gf) if gq is not None and gf and gf > 0 else None
    gtxt = ("" if gs is None else (f' · gross <b>{100 * gs:.0f}%</b> less' if gs >= 0 else f' · gross <b>{-100 * gs:.0f}%</b> more'))
    head = (f'<span>Q-Watt used <b>{100 * save:.0f}% less energy</b> on this question'
            f'<span style="color:var(--ink2);font-size:15px"> (net of idle{gtxt})</span></span>' if save >= 0 else
            f'<span>On this run Q-Watt used <b>{-100 * save:.0f}% more energy</b> net of idle{gtxt.replace("<b>", "").replace("</b>", "")}'
            f' (single live runs are noisy; see the benchmark headline)</span>')
    parts = [head,
             '<span class="sep"></span>',
             f'<span><b>{fmt_j(Jq)} J</b> vs <b>{fmt_j(Jf)} J</b></span>']
    if fnum(select_j) is not None and fnum(eps_in):
        be = select_j / eps_in
        parts += ['<span class="sep"></span>',
                  f'<span>quantum layer cost <b>{fmt_j(select_j, 1)} J</b> · paid for itself after '
                  f'<b>{be:,.0f}</b> tokens · removed <b>{fmt_i(removed)}</b></span>']
    return '<div class="qw-verdict">' + "".join(parts) + "</div>"


# ---------------------------------------------------------------- quantum lens
def _svg_img(svg: str, height: int) -> str:
    """st.html sanitizes inline <svg>; an <img> with a data URI survives."""
    import base64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (f'<img alt="" src="data:image/svg+xml;base64,{b64}" '
            f'style="width:100%;height:{height}px;display:block;pointer-events:none">')


def qubit_tiles(pool: dict | None, diag: dict | None, kept: list[int] | None, state: str = "done",
                n_fallback: int = 0) -> str:
    """Row of n qubit tiles with redundancy (red, above) and bridge (green, below) links.

    state: 'pending' (no circuit yet), 'running' (circuit sampling), 'done' (kept tiles light up).
    """
    sents = (pool or {}).get("sentences") or []
    n = len(sents) or int((diag or {}).get("n_qubits") or n_fallback or 0)
    if n == 0:
        return ""
    h = (diag or {}).get("h") or []
    if sents:
        rel = [fnum(s.get("relevance")) or 0.0 for s in sents]
        toks = [s.get("tokens") for s in sents]
    elif (diag or {}).get("relevance"):
        rel = [fnum(x) or 0.0 for x in diag["relevance"]]
        toks = list((diag or {}).get("t") or [None] * n)
    else:  # derive a relevance proxy from the linear QUBO term if the pool is unavailable
        hv = [fnum(x) or 0.0 for x in h] or [0.0] * n
        lo, hi = min(hv), max(hv)
        rel = [(hi - v) / (hi - lo) if hi > lo else 0.5 for v in hv]
        toks = list((diag or {}).get("t") or [None] * n)
    rel = (list(rel) + [0.0] * n)[:n]
    toks = (list(toks) + [None] * n)[:n]
    kept_set = set(kept or [])
    reds = (diag or {}).get("red_edges") or []
    bridges = (diag or {}).get("bridge_edges") or []

    W, H_TOP, H_BOT = n * 100, 64, 52

    def arc(i, j, up: bool, color: str, w: float, op: float):
        xi, xj = (i + 0.5) * 100, (j + 0.5) * 100
        span = abs(j - i)
        hgt = min(1.0, 0.25 + 0.12 * span)
        if up:
            y0, yc = H_TOP, H_TOP - 2 * hgt * (H_TOP - 6)
        else:
            y0, yc = 0, 2 * hgt * (H_BOT - 6)
        lit = (i in kept_set and j in kept_set) and state == "done"
        return (f'<path d="M{xi:.0f},{y0} Q{(xi + xj) / 2:.0f},{yc:.0f} {xj:.0f},{y0}" fill="none" stroke="{color}" '
                f'stroke-width="{w:.1f}" stroke-opacity="{1.0 if lit else op}" vector-effect="non-scaling-stroke" '
                f'stroke-linecap="round"/>')

    top_paths = "".join(arc(int(e[0]), int(e[1]), True, T.RED_LINK, 1.5 + 2.5 * (fnum(e[2]) or 0.5) if len(e) > 2 else 2.5, 0.55)
                        for e in reds if len(e) >= 2 and int(e[0]) < n and int(e[1]) < n)
    bot_paths = "".join(arc(int(e[0]), int(e[1]), False, T.BRIDGE, 3.0, 0.6)
                        for e in bridges if len(e) >= 2 and int(e[0]) < n and int(e[1]) < n)
    svg_top = _svg_img(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H_TOP}" preserveAspectRatio="none">'
                       f'{top_paths}</svg>', H_TOP)
    svg_bot = _svg_img(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H_BOT}" preserveAspectRatio="none">'
                       f'{bot_paths}</svg>', H_BOT)

    tiles = []
    for i in range(n):
        s = sents[i] if i < len(sents) else {}
        on = i in kept_set and state == "done"
        off = state == "done" and not on
        r = max(0.04, min(1.0, rel[i]))
        cls = "tile" + (" on" if on else "") + (" off" if off else "") + (" run" if state == "running" else "")
        align = "left:0" if i < 3 else ("right:0" if i > n - 4 else "left:50%;transform:translateX(-50%)")
        gold = '<span class="gold" title="HotpotQA supporting fact">★</span>' if s.get("is_gold") else ""
        tok = f"{toks[i]}" if toks[i] is not None else "–"
        tip_txt = (f'<div class="tt">{esc(s.get("title", ""))}</div><div class="tx">{esc(s.get("text", ""))}</div>'
                   if s else f'<div class="tx">qubit {i} (sentence text unavailable)</div>')
        tip_meta = f'relevance {rel[i]:.2f} · {tok} tokens'
        if i < len(h) and fnum(h[i]) is not None:
            tip_meta += f' · h = {fnum(h[i]):+.3f}'
        bit = "1" if on else ("0" if off else "·")
        tiles.append(f"""
<div class="{cls}" style="--d:{i * 35}ms">
  <div class="qi">q{i}{gold}</div>
  <div class="barbox"><div class="bar" style="height:{r * 100:.0f}%"></div></div>
  <div class="tk">{tok}<span> tok</span></div>
  <div class="bit">{bit}</div>
  <div class="tip" style="{align}">{tip_txt}<div class="tm">{esc(tip_meta)}{' · ★ gold supporting fact' if s.get('is_gold') else ''}</div></div>
</div>""")
    bits = "".join("1" if (i in kept_set) else "0" for i in range(n)) if state == "done" else ""
    css = f"""
<style>
.qlens {{ position:relative; }}
.qlens .row {{ display:grid; grid-template-columns: repeat({n}, minmax(0,1fr)); gap:6px; }}
.qlens .tile {{ position:relative; height:170px; border-radius:12px; background:{T.SURFACE_2};
  border:1px solid rgba(255,255,255,.07); display:flex; flex-direction:column; align-items:center; padding:7px 4px 6px;
  transition: all .45s ease; transition-delay: var(--d); cursor: default; }}
.qlens .tile .qi {{ font-family:{T.FONT_MONO}; font-size:12px; color:{T.MUTED}; font-weight:700; position:relative; }}
.qlens .tile .gold {{ color:{T.WARN}; font-size:11px; margin-left:2px; }}
.qlens .tile .barbox {{ flex:1; width:56%; margin:6px 0 4px; display:flex; align-items:flex-end;
  background: rgba(255,255,255,.035); border-radius:6px 6px 2px 2px; overflow:hidden; }}
.qlens .tile .bar {{ width:100%; border-radius:4px 4px 0 0; background:{T.NEUTRAL_A}; transition: all .45s ease; transition-delay: var(--d); }}
.qlens .tile .tk {{ font-size:13px; color:{T.INK_2}; font-weight:650; }}
.qlens .tile .tk span {{ font-size:10.5px; color:{T.MUTED}; font-weight:600; }}
.qlens .tile .bit {{ margin-top:4px; font-family:{T.FONT_MONO}; font-weight:800; font-size:18px; color:{T.MUTED};
  width:30px; height:26px; border-radius:7px; display:grid; place-items:center; background: rgba(255,255,255,.04); }}
.qlens .tile.on {{ border-color:{T.rgba(T.QUANTUM, .9)}; background: linear-gradient(180deg, {T.rgba(T.QUANTUM, .28)}, {T.rgba(T.QUANTUM, .08)});
  box-shadow: 0 0 0 1px {T.rgba(T.QUANTUM, .35)}, 0 0 26px -4px {T.rgba(T.QUANTUM, .85)}; }}
.qlens .tile.on .bar {{ background: linear-gradient(180deg, #c9c2ff, {T.QUANTUM}); }}
.qlens .tile.on .bit {{ background:{T.QUANTUM}; color:#0b0d12; }}
.qlens .tile.on .qi, .qlens .tile.on .tk {{ color:{T.INK}; }}
.qlens .tile.off {{ opacity:.42; }}
.qlens .tile.run {{ animation: qshimmer 1.1s ease-in-out infinite; animation-delay: var(--d); }}
@keyframes qshimmer {{ 0%,100% {{ border-color: rgba(255,255,255,.07); box-shadow:none; }}
  50% {{ border-color:{T.rgba(T.QUANTUM, .7)}; box-shadow: 0 0 18px -6px {T.rgba(T.QUANTUM, .9)}; }} }}
.qlens .tile .tip {{ position:absolute; bottom: calc(100% + 10px); width: 340px; z-index: 50; visibility:hidden; opacity:0;
  transition: opacity .12s; background:{T.SURFACE_2}; border:1px solid {T.AXIS}; border-radius:12px; padding:11px 13px;
  box-shadow: 0 16px 40px -10px rgba(0,0,0,.8); pointer-events:none; text-align:left; }}
.qlens .tile:hover {{ border-color: {T.INK_2}; opacity:1; }}
.qlens .tile:hover .tip {{ visibility:visible; opacity:1; }}
.qlens .tip .tt {{ color:{T.MUTED}; font-size:12px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; }}
.qlens .tip .tx {{ color:{T.INK}; font-size:15px; line-height:1.4; margin-top:3px; }}
.qlens .tip .tm {{ color:{T.INK_2}; font-size:12.5px; margin-top:7px; font-family:{T.FONT_MONO}; }}
.qlens .bits {{ text-align:center; color:{T.INK_2}; font-size:14px; margin-top:2px; }}
.qlens .bits b {{ font-family:{T.FONT_MONO}; letter-spacing:.32em; font-size:16px; }}
</style>"""
    bits_html = f'<div class="bits">measured bitstring → <b style="color:{T.INK}">{bits}</b></div>' if bits else ""
    return f'{css}<div class="qlens">{svg_top}<div class="row">{"".join(tiles)}</div>{svg_bot}{bits_html}</div>'


def lens_legend() -> str:
    return (f'<div class="qw-legend"><span><i style="background:{T.RED_LINK}"></i>redundant pair (arcs above) · pushes '
            f'one of them to 0</span><span><i style="background:{T.BRIDGE}"></i>bridge pair (arcs below) · pulled in '
            f'together</span><span><i style="background:{T.QUANTUM};height:10px;width:10px;border-radius:3px"></i>'
            f'kept (measured 1)</span><span>bar height = relevance · number = token cost</span>'
            f'<span><b style="color:{T.WARN}">★</b> gold supporting fact (dataset label)</span>'
            f'<span style="color:{T.MUTED}">hover a tile to read its sentence</span></div>')


def caption() -> str:
    return f'<div class="qw-caption"><span class="q">|1⟩</span>{esc(CAPTION)}</div>'


def facts(diag: dict, joules_select, eps_in) -> str:
    def fact(k, v, hi=False):
        return f'<div class="qw-fact{" hi" if hi else ""}"><div class="k">{esc(k)}</div><div class="v">{v}</div></div>'

    amp = fnum(diag.get("amplification"))
    ar = fnum(diag.get("approx_ratio"))
    disp = diag.get("dispatch") or "—"
    items = [
        fact("backend", esc(diag.get("backend") or "—")),
        fact("qubits", fmt_i(diag.get("n_qubits"))),
        fact("couplings", fmt_i(diag.get("n_edges"))),
        fact("depth", fmt_i(diag.get("depth"))),
        fact("gates", fmt_i(diag.get("gate_count"))),
        fact("shots", fmt_i(diag.get("shots"))),
        fact("wall time", fmt_s(diag.get("circuit_wall_s"))),
        fact("energy spent", f"{fmt_j(joules_select, 2)} J" if fnum(joules_select) is not None else "—"),
        fact("amplification", f"{amp:.0f}× better than random" if amp is not None else "—", hi=True),
        fact("approx. ratio", f"{ar:.3f}" if ar is not None else "—"),
        fact("exact optimum in shots", "✓ yes" if diag.get("opt_in_shots") else ("✗ no" if "opt_in_shots" in diag else "—")),
        fact("dispatch", esc(disp.replace("_", " "))),
    ]
    if fnum(joules_select) is not None and fnum(eps_in):
        items.append(fact("break-even", f"{joules_select / eps_in:,.0f} tokens"))
    if diag.get("feasible_fallback"):
        items.append(fact("note", "fallback: trimmed to fit budget"))
    return '<div class="qw-facts">' + "".join(items) + "</div>"


def footer(ex: dict | None, delta_j: float | None, synthetic: bool) -> str:
    if not ex:
        body = '<div class="big">Extrapolation appears once paired benchmark runs exist.</div>'
    else:
        body = (f'<div class="big">At <b>1,000</b> questions a day this saves <b>{ex["wh_per_day"]:,.1f} Wh</b>, '
                f'about <b>{ex["phone_charges"]:,.1f}</b> phone charges.</div>')
    note = (f"Defined unit: one phone charge = 15 Wh. Based on the median paired saving of "
            f"{fmt_j(delta_j)} J per question on this laptop (net of idle).") if ex else "One phone charge = 15 Wh."
    if synthetic:
        note += " SYNTHETIC preview numbers."
    return f'<div class="qw-foot">{body}<div class="def">{esc(note)}</div></div>'

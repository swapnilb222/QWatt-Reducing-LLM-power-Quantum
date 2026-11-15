"""Tiger Data in the loop: a self-contained, light-theme Streamlit section.

    from app import tiger_panel
    tiger_panel.render(st)            # theme: optional dict {"accent": "#...", "muted": "#..."}

Degrades to a one-line note when Tiger is not configured / unreachable.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

_cache: dict = {}


def _cached(key: str, fn, ttl: float = 20.0):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def _fmt(x, nd=3, unit=""):
    return "n/a" if x is None else f"{float(x):.{nd}f}{unit}"


def render(st, theme: dict | None = None, energy_budget_j: float | None = None) -> None:
    from qwatt import config, tiger

    theme = theme or {}
    accent = theme.get("accent", "#2F6FDE")
    muted = theme.get("muted", "#6B7280")

    st.subheader("Tiger Data in the loop")
    if not tiger.configured():
        st.caption("Tiger Data is not configured (TIGER_DATABASE_URL empty); showing local SQLite results only.")
        return
    perf = _cached("perf", tiger.perf_stats)
    if perf is None:
        st.caption("Tiger Data is unreachable right now; the dashboard is using local SQLite results.")
        return

    price = _cached("price", lambda: tiger.live_price(max_age_s=20))
    cal = config.load_calibration() or {}
    drift = _cached("drift", tiger.price_drift)
    integ = _cached("integ", tiger.integrity_summary)

    # ---- live price
    c1, c2, c3, c4 = st.columns(4)
    if price:
        c1.metric("Live energy price", f"{price['eps_in'] * 1000:.1f} mJ/token",
                  help="OLS slope of prefill joules vs input tokens over the 60 most recent runs, "
                       "computed in SQL by Tiger Data.")
        c2.metric("Fit quality", f"r2 {_fmt(price['r2'], 2)}", f"n = {price['n']} runs", delta_color="off")
    else:
        c1.metric("Live energy price", "n/a")
        c2.metric("Fit quality", "n/a")
    if cal.get("eps_in"):
        c3.metric("Calibration price", f"{cal['eps_in'] * 1000:.1f} mJ/token",
                  (f"{100 * (price['eps_in'] / cal['eps_in'] - 1):+.0f}% live vs calibration" if price else None),
                  delta_color="off")
    ratio = perf.get("compression_ratio")
    c4.metric("Compression", f"{ratio:.1f}x" if ratio else "n/a",
              f"{perf.get('power_samples_rows', 0):,} samples", delta_color="off")

    # ---- drift chart
    if drift and len(drift) >= 2:
        try:
            import plotly.graph_objects as go
            xs = [datetime.fromtimestamp(float(d["t"]), tz=timezone.utc) for d in drift]
            ys = [float(d["eps_in"]) * 1000 for d in drift]
            fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=accent, width=2),
                                       marker=dict(size=6), name="mJ/token"))
            if cal.get("eps_in"):
                fig.add_hline(y=cal["eps_in"] * 1000, line_dash="dot", line_color=muted,
                              annotation_text="calibration", annotation_font_color=muted)
            fig.update_layout(template="plotly_white", height=190, margin=dict(l=10, r=10, t=24, b=10),
                              title=dict(text="Energy price drift (15-minute buckets)", font=dict(size=13)),
                              yaxis_title="mJ per input token", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:  # noqa: BLE001
            pass

    # ---- budget line
    eb = energy_budget_j or os.getenv("QWATT_ENERGY_BUDGET_J")
    if eb and price:
        eb = float(eb)
        toks = max(120, int((eb - price["e0"]) / price["eps_in"]))
        st.markdown(f"**Energy budget {eb:.0f} J = {toks:,} tokens** at the current price "
                    f"({price['eps_in'] * 1000:.1f} mJ/token, fixed cost {price['e0']:.1f} J).")
    elif price:
        ex = 60.0
        toks = max(120, int((ex - price["e0"]) / price["eps_in"]))
        st.caption(f"Example: an energy budget of {ex:.0f} J buys about {toks:,} context tokens at the current "
                   "price. The selector's token budget is derived from this in live mode.")

    # ---- ledger
    led = _cached("ledger", lambda: tiger.phase_ledger(limit=60))
    if led:
        import pandas as pd
        df = pd.DataFrame(led)
        for c in ("sql_net_j", "python_net_j", "delta_pct", "dur_s"):
            df[c] = df[c].astype(float)
        lc = df[df["arm"].isin(["full_lc", "qwatt_lc", "qwatt_safe"])]
        df = (lc if len(lc) else df).copy()
        df = df[df["python_net_j"].fillna(0) > 0].head(12)
        if len(df):
            out = pd.DataFrame({
                "arm": df["arm"], "phase": df["phase"],
                "SQL net J": df["sql_net_j"].round(1), "Python net J": df["python_net_j"].round(1),
                "diff %": df["delta_pct"].round(1),
            })
            st.markdown("**Per-phase joules: SQL integral (Tiger) vs Python** (net of idle, latest runs)")
            st.dataframe(out, hide_index=True, use_container_width=True)

    # ---- integrity + latency
    d1, d2 = st.columns(2)
    if integ:
        med = integ.get("median_abs_sql_vs_python_pct")
        d1.markdown(
            f"**Integrity:** {integ['n_flagged']} of {integ['n_runs']} runs flagged "
            f"(idle drift {integ['n_idle']}, sample gaps {integ['n_gap']}, SQL vs Python > 15%: {integ['n_energy']}). "
            f"Median |SQL - Python| = {_fmt(med, 1)}%. Flagged runs are excluded from the scoreboard.")
    d2.markdown(
        f"**Query latency (client side):** scoreboard {_fmt(perf.get('scoreboard_ms'), 0)} ms, "
        f"live price {_fmt(perf.get('live_price_ms'), 0)} ms, 1 s power buckets {_fmt(perf.get('power_1s_ms'), 0)} ms.")

"""Visual system for the Q-Watt dashboard: light theme, colors, fonts, CSS, Plotly template.

Restrained palette on white: one accent for Q-Watt (teal), one muted warm color for Full document /
Full context (terracotta), a soft violet only for quantum elements. Marks clear 3:1 on white; text uses the
slate ink scale (body #0f172a / secondary #475569 / muted #5b6678, all >= 4.5:1 on white).
Red (redundancy) vs green (bridge) links are also separated by position/label, never hue alone.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ---- surfaces & ink ----
BG = "#f6f7f9"
SURFACE = "#ffffff"
SURFACE_2 = "#f8fafc"
BORDER = "#e3e7ee"
INK = "#0f172a"
INK_2 = "#475569"
MUTED = "#5b6678"
GRID = "#edf0f4"
AXIS = "#cfd6e0"

# ---- identity colors ----
FULL = "#c8693b"      # Full context / Full document (muted terracotta)
QWATT = "#0f8f7a"     # Q-Watt (teal)
QUANTUM = "#6f62d4"   # qubits / the quantum layer (soft violet)
TOPK = "#4a7fd6"      # Top-k (hidden) / Q-Watt without safety net
GREEDY = "#b07d1a"    # Greedy (A2)
RED_LINK = "#d25454"  # redundancy pairs
BRIDGE = "#2f9e5b"    # bridge pairs
NEUTRAL_A = "#7b8799"  # ledger: prefill
NEUTRAL_B = "#b7c0cd"  # ledger: generate

# ---- status (reserved; always with icon + label) ----
GOOD = "#15803d"
WARN = "#b45309"
CRIT = "#b91c1c"

ARM_COLOR = {"full": FULL, "topk": TOPK, "greedy": GREEDY, "qwatt": QWATT,
             "full_lc": FULL, "qwatt_safe": QWATT, "qwatt_lc": TOPK, "anneal_lc": NEUTRAL_B, "random_lc": NEUTRAL_A,
             "exact": INK_2, "anneal": NEUTRAL_B, "random_k": NEUTRAL_A}
ARM_LABEL = {"full": "Full context", "topk": "Top-k", "greedy": "Greedy", "qwatt": "Q-Watt",
             "qaoa_qr": "QAOA (Quantum Rings)", "qaoa_exact_sim": "QAOA (exact statevector)",
             "full_lc": "Full document", "qwatt_safe": "Q-Watt", "qwatt_lc": "Q-Watt without safety net",
             "anneal_lc": "Annealing (classical reference)", "random_lc": "Random selection",
             "exact": "Exact (brute force)", "anneal": "Annealing", "random_k": "Random-K",
             "qaoa": "QAOA (Quantum Rings)"}
ARM_ORDER = ["full", "topk", "greedy", "qwatt"]

FONT_SANS = ('system-ui, -apple-system, "Segoe UI Variable Text", "Segoe UI", "Inter", Roboto, '
             '"Helvetica Neue", Arial, sans-serif')
FONT_MONO = '"Cascadia Mono", "JetBrains Mono", Consolas, "SFMono-Regular", Menlo, monospace'


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# ---------------------------------------------------------------- Plotly template
def _register_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_SANS, size=13, color=INK_2),
        colorway=[FULL, QWATT, QUANTUM, TOPK, GREEDY],
        margin=dict(l=8, r=8, t=8, b=8),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(family=FONT_SANS, size=13, color=INK)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=AXIS, linewidth=1, ticks="",
                   tickfont=dict(color=MUTED, size=12), title=dict(font=dict(color=MUTED, size=12)),
                   automargin=True),
        yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
                   tickfont=dict(color=MUTED, size=12), title=dict(font=dict(color=MUTED, size=12)),
                   automargin=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2, size=12),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        bargap=0.35,
        hovermode="closest",
    )
    pio.templates["qwatt"] = t
    pio.templates.default = "qwatt"


_register_template()

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

SHADOW = "0 1px 2px rgba(15,23,42,.04), 0 1px 3px rgba(15,23,42,.06)"

# ---------------------------------------------------------------- page CSS
CSS = f"""
<style>
:root {{
  --bg:{BG}; --surface:{SURFACE}; --surface2:{SURFACE_2}; --border:{BORDER};
  --ink:{INK}; --ink2:{INK_2}; --muted:{MUTED};
  --full:{FULL}; --qwatt:{QWATT}; --quantum:{QUANTUM}; --topk:{TOPK}; --greedy:{GREEDY};
  --red:{RED_LINK}; --bridge:{BRIDGE}; --good:{GOOD}; --warn:{WARN}; --crit:{CRIT};
  --sans:{FONT_SANS}; --mono:{FONT_MONO}; --shadow:{SHADOW};
}}
/* ---------- hide Streamlit chrome ---------- */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
[data-testid="stAppDeployButton"] {{ display:none !important; visibility:hidden !important; }}
[data-stale="true"], .stale-element, .element-container[data-stale="true"] {{
  opacity:1 !important; transition:none !important; filter:none !important; }}
html, body, [data-testid="stAppViewContainer"], .stApp {{ background: var(--bg) !important; color: var(--ink); }}
.stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{ font-family: var(--sans); }}
.block-container {{ padding-top: 1.4rem !important; padding-bottom: 2.5rem !important;
  max-width: 1440px !important; padding-left: 2.4rem !important; padding-right: 2.4rem !important; }}
[data-testid="stVerticalBlock"] {{ gap: 0.85rem; }}
p, li, label, .stMarkdown {{ font-size: 0.98rem; color: var(--ink2); }}
code, pre, .mono {{ font-family: var(--mono) !important; }}
code {{ color: var(--ink) !important; background: #eef1f5 !important; }}

/* ---------- widgets ---------- */
[data-testid="stSlider"] label p, .stSelectbox label p, .stRadio label p {{
  color: var(--muted) !important; font-size: .85rem !important; font-weight:600; }}
div[data-baseweb="select"] > div {{ background: var(--surface) !important; border-color: var(--border) !important;
  border-radius: 10px !important; }}
.stButton > button {{ border-radius: 10px; border: 1px solid {QWATT}; background: {QWATT}; color: #fff;
  font-weight: 600; font-size: .95rem; padding: .5rem 1.1rem; box-shadow: var(--shadow); }}
.stButton > button p {{ color: #fff !important; }}
.stButton > button:hover {{ background: #0c7a68; border-color:#0c7a68; color:#fff; }}
.stButton > button:disabled {{ background: #e8ecf1; border-color: var(--border); color: var(--muted); box-shadow:none; }}
.stButton > button:disabled p {{ color: var(--muted) !important; }}
[data-testid="stExpander"] details {{ background: var(--surface); border: 1px solid var(--border) !important;
  border-radius: 12px !important; box-shadow: var(--shadow); }}
[data-testid="stExpander"] summary p {{ font-size: .95rem !important; font-weight: 600; color: var(--ink) !important; }}
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; border:1px solid var(--border); }}
.st-key-view button p, .st-key-mode button p {{ font-size: .88rem !important; font-weight: 600 !important; }}

/* ---------- header ---------- */
.qw-top {{ display:flex; align-items:center; justify-content:space-between; gap:20px; }}
.qw-brand {{ display:flex; align-items:center; gap:12px; }}
.qw-logo {{ width:36px; height:36px; border-radius:10px; display:grid; place-items:center; background: {QWATT}; }}
.qw-logo span {{ color:#fff; font-weight:750; font-size:17px; letter-spacing:-.03em; }}
.qw-title {{ font-size:26px; font-weight:720; letter-spacing:-.02em; color:var(--ink); line-height:1.1; }}
.qw-title b {{ color: {QWATT}; font-weight:720; }}
.qw-sub {{ color:var(--ink2); font-size:14.5px; margin-top:2px; }}
.qw-status {{ color: var(--muted); font-size: 12.5px; margin-top: 4px; display:flex; gap:14px; flex-wrap:wrap; }}
.qw-status span {{ display:inline-flex; align-items:center; gap:6px; }}
.qw-status i {{ width:7px; height:7px; border-radius:50%; background:#b7c0cd; display:inline-block; }}
.qw-status i.ok {{ background:{GOOD}; }}
.qw-status i.warn {{ background:{WARN}; }}
.qw-status i.live {{ background:#dc2626; animation: qwpulse 1.4s infinite; }}
@keyframes qwpulse {{ 0% {{ box-shadow:0 0 0 0 rgba(220,38,38,.5);}} 70% {{ box-shadow:0 0 0 6px rgba(220,38,38,0);}}
  100% {{ box-shadow:0 0 0 0 rgba(220,38,38,0);}} }}
.qw-pills {{ display:none; }}
.qw-pill {{ display:inline-flex; align-items:center; gap:7px; padding:4px 10px; border-radius:999px;
  border:1px solid var(--border); background:var(--surface); color:var(--ink2); font-size:12.5px; font-weight:600; }}
.qw-pill .dot {{ width:7px; height:7px; border-radius:50%; background:#b7c0cd; }}
.qw-pill.ok .dot {{ background:{GOOD}; }}

/* ---------- banners ---------- */
.qw-synth {{ margin: 2px 0 2px; padding: 8px 14px; border-radius: 10px; font-weight: 650; font-size: 13px;
  color: #7c2d12; background: #fff7ed; border: 1px solid #fed7aa; display:flex; gap:10px; align-items:center; }}
.qw-synth small {{ font-weight:500; color:#9a3412; }}
.qw-flag {{ display:inline-flex; align-items:center; gap:6px; padding:3px 9px; border-radius:6px; font-size:12px;
  font-weight:600; background: #fff7ed; color:#9a3412; border:1px solid #fed7aa; }}

/* ---------- section headers ---------- */
.qw-h {{ display:flex; align-items:baseline; gap:12px; margin: 22px 0 2px; padding-top: 6px; }}
.qw-h .n {{ display:none; }}
.qw-h h2 {{ font-size: 20px !important; font-weight: 680 !important; letter-spacing:-.01em; color:var(--ink) !important;
  margin:0 !important; padding:0 !important; }}
.qw-h .s {{ color: var(--muted); font-size: 13.5px; }}

/* ---------- cards ---------- */
.qw-card {{ background: var(--surface); border:1px solid var(--border); border-radius:14px; padding:18px 20px;
  position:relative; box-shadow: var(--shadow); }}
.qw-hero {{ display:grid; grid-template-columns: 1.3fr 1.2fr 1fr 1fr; gap:14px; }}
.qw-stat .lbl {{ color:var(--muted); font-size:13px; font-weight:600; }}
.qw-stat .val {{ color:var(--ink); font-size:40px; font-weight:700; letter-spacing:-.03em; line-height:1.1; margin-top:6px;
  font-variant-numeric: tabular-nums; }}
.qw-stat .val small {{ font-size:18px; font-weight:600; color:var(--muted); letter-spacing:0; margin-left:3px; }}
.qw-stat .sub {{ color:var(--muted); font-size:12.5px; margin-top:6px; line-height:1.45; }}
.qw-stat.primary {{ border-top: 3px solid {QWATT}; }}
.qw-stat.primary .val {{ color: {QWATT}; font-size:46px; }}
.qw-duo {{ display:flex; gap:24px; align-items:flex-end; margin-top:6px; }}
.qw-duo .k {{ display:flex; align-items:center; gap:6px; color:var(--muted); font-size:12.5px; font-weight:600; }}
.qw-duo .v {{ color:var(--ink); font-size:32px; font-weight:700; letter-spacing:-.02em; line-height:1.1;
  font-variant-numeric: tabular-nums; }}
.qw-duo .v small {{ font-size:15px; color:var(--muted); margin-left:2px; }}
.sw {{ display:inline-block; width:9px; height:9px; border-radius:2px; }}
.qw-cal {{ color:var(--muted); font-size:12.5px; margin-top:6px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
.qw-cal b {{ color:var(--ink); font-variant-numeric: tabular-nums; }}

/* ---------- bordered containers = cards ---------- */
[class*="st-key-lane_"], [class*="st-key-lc_lane_"], [class*="st-key-sb"], [class*="st-key-lcsb"],
[class*="st-key-histcard"], [class*="st-key-dialcard"], [class*="st-key-lc_mapcard"], [class*="st-key-lc_dialcard"],
[class*="st-key-lc_frontcard"], [class*="st-key-lc_histcard"], [class*="st-key-lc_money"] {{
  background: var(--surface) !important; border-radius: 14px !important; border: 1px solid var(--border) !important;
  padding: 16px 18px !important; box-shadow: var(--shadow); }}
[class*="st-key-lane_"][class*="qwatt"] {{ border-top: 3px solid {QWATT} !important; }}
[class*="st-key-lane_"][class*="full"] {{ border-top: 3px solid {FULL} !important; }}
[class*="st-key-lane_"][class*="raw"], [class*="st-key-lane_"][class*="topk"] {{ border-top: 3px solid {TOPK} !important; }}

/* ---------- lanes ---------- */
.qw-lane-h {{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }}
.qw-lane-name {{ flex:none; white-space:nowrap; display:flex; align-items:center; gap:8px; font-size:18px; font-weight:680;
  color:var(--ink); letter-spacing:-.01em; }}
.qw-lane-name .bar {{ display:none; }}
.qw-lane-name .tag {{ font-size:12px; font-weight:500; color:var(--muted); }}
.qw-joules {{ text-align:right; }}
.qw-joules .v {{ font-size:36px; font-weight:700; color:var(--ink); letter-spacing:-.03em; line-height:1;
  font-variant-numeric: tabular-nums; }}
.qw-joules .v small {{ font-size:16px; color:var(--muted); font-weight:600; margin-left:2px; }}
.qw-joules .s {{ color:var(--muted); font-size:11.5px; margin-top:4px; max-width:300px; margin-left:auto; }}
.qw-chips {{ display:flex; gap:6px; margin-top:10px; flex-wrap:wrap; }}
.qw-chip {{ padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; color:var(--muted);
  background: var(--surface2); border:1px solid var(--border); }}
.qw-chip.on {{ color: var(--ink); border-color: var(--c); background: color-mix(in srgb, var(--c) 12%, white); }}
.qw-chip.on::before {{ content:""; display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--c);
  margin-right:6px; vertical-align:1px; animation: qwblink 1s infinite; }}
.qw-chip.past {{ color: var(--ink2); }}
.qw-chip.past::before {{ content:"✓ "; color: var(--ink2); }}
.qw-chip.skip {{ text-decoration: line-through; opacity:.6; }}
@keyframes qwblink {{ 50% {{ opacity:.35; }} }}
.qw-kv {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:0; margin-top:2px; border-top:1px solid var(--border);
  padding-top:8px; }}
.qw-kv > div {{ padding:2px 4px; }}
.qw-kv .k {{ color:var(--muted); font-size:11.5px; font-weight:600; }}
.qw-kv .v {{ color:var(--ink); font-size:17px; font-weight:650; font-variant-numeric: tabular-nums; }}
.qw-answer {{ margin-top:10px; padding:10px 12px; border-radius:10px; background: var(--surface2); border:1px solid var(--border); }}
.qw-answer .q {{ color:var(--muted); font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }}
.qw-answer .a {{ color:var(--ink); font-size:16.5px; font-weight:600; margin-top:3px; line-height:1.35; }}
.qw-answer .a.dim {{ color: var(--muted); font-weight:500; font-style: italic; }}
.qw-answer .cursor {{ display:inline-block; width:8px; height:17px; background:var(--ink2); vertical-align:-3px;
  margin-left:2px; animation: qwblink .8s infinite; }}
.qw-badge {{ display:inline-flex; align-items:center; gap:5px; padding:2px 9px; border-radius:999px; font-size:12px; font-weight:650; }}
.qw-badge.ok {{ color:#166534; background:#f0fdf4; border:1px solid #bbf7d0; }}
.qw-badge.bad {{ color:#991b1b; background:#fef2f2; border:1px solid #fecaca; }}
.qw-badge.pend {{ color: var(--ink2); background: var(--surface2); border:1px solid var(--border); }}
.qw-verdict {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; padding:12px 16px; border-radius:12px;
  background: var(--surface); border:1px solid var(--border); border-left: 3px solid {QWATT}; font-size:15px;
  color:var(--ink2); box-shadow: var(--shadow); }}
.qw-verdict b {{ font-size:19px; color: var(--ink); font-variant-numeric: tabular-nums; }}
.qw-verdict .sep {{ width:1px; height:20px; background: var(--border); }}
.qw-empty {{ border:1px dashed #cfd6e0; border-radius:12px; padding:16px; color:var(--muted);
  background: var(--surface); text-align:center; font-size:14px; }}
.qw-empty > b:first-child {{ color:var(--ink); display:block; font-size:15px; margin-bottom:3px; }}
.qw-empty b {{ color:var(--ink); }}
.qw-subh {{ font-size:15px; font-weight:650; color:var(--ink); margin:6px 0 2px; }}

/* ---------- quantum lens ---------- */
.qw-caption {{ display:flex; align-items:center; gap:10px; padding:0; color:var(--ink2); font-size:15px; font-weight:500; }}
.qw-caption .q {{ width:24px; height:24px; border-radius:6px; display:grid; place-items:center; font-weight:700;
  background: {rgba(QUANTUM, .12)}; color: {QUANTUM}; font-family: var(--mono); font-size: 12px; flex: none; }}
.qw-facts {{ display:flex; flex-wrap:wrap; gap:8px; }}
.qw-fact {{ padding:7px 11px; border-radius:10px; background: var(--surface); border:1px solid var(--border); }}
.qw-fact .k {{ color:var(--muted); font-size:10.5px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; }}
.qw-fact .v {{ color:var(--ink); font-size:15px; font-weight:650; font-variant-numeric: tabular-nums; }}
.qw-fact.hi {{ border-color: {rgba(QUANTUM, .45)}; background: {rgba(QUANTUM, .05)}; }}
.qw-legend {{ display:flex; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:12.5px; align-items:center; }}
.qw-legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.qw-legend i {{ display:inline-block; width:18px; height:3px; border-radius:2px; }}
.qw-mini {{ color: var(--muted); font-size: 12.5px; }}
.qw-foot {{ margin-top: 16px; padding: 16px 20px; border-radius: 12px; border:1px solid var(--border); background: var(--surface);
  display:flex; gap:20px; align-items:center; justify-content:space-between; flex-wrap:wrap; box-shadow: var(--shadow); }}
.qw-foot .big {{ font-size: 18px; font-weight: 600; color: var(--ink); }}
.qw-foot .big b {{ font-size: 22px; color: {QWATT}; font-variant-numeric: tabular-nums; }}
.qw-foot .def {{ color: var(--muted); font-size: 12.5px; }}
.qw-shift {{ font-size:14.5px; color:var(--ink2); line-height:1.45; margin: 2px 0 2px; }}
.qw-shift b {{ font-size:16px; color: var(--ink); }}
[data-testid="stTabs"] button p {{ font-size: 14px !important; font-weight: 600; }}
.qw-explain {{ padding:12px 14px; border-radius:10px; background: var(--surface); border:1px solid var(--border);
  color: var(--ink); font-size: 15px; line-height:1.45; }}
</style>
"""

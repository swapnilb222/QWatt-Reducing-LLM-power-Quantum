"""Visual system for the Q-Watt dashboard: colors, fonts, CSS, Plotly template.

Palette checked with the dataviz validator against the dark chart surface #12151c:
- lane triad FULL / QWATT / QUANTUM passes every check under --pairs all;
- arm order full, topk, greedy, qwatt passes every adjacent check.
Red (redundancy) vs green (bridge) links fail CVD on hue alone, so they are also separated by
position (red arcs above the tiles, green arcs below) and labeled in the legend.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ---- surfaces & ink ----
BG = "#0b0d12"
SURFACE = "#12151c"
SURFACE_2 = "#171b25"
BORDER = "rgba(255,255,255,0.08)"
INK = "#eef0f5"
INK_2 = "#b6bbc8"
MUTED = "#7f8699"
GRID = "#222734"
AXIS = "#343a4a"

# ---- identity colors ----
FULL = "#e0682f"      # Full context lane (A0)
QWATT = "#14ab7e"     # Q-Watt lane (A3)
QUANTUM = "#9085e9"   # qubits / the quantum layer
TOPK = "#3987e5"      # Top-k (A1)
GREEDY = "#c98500"    # Greedy (A2)
RED_LINK = "#e66767"  # redundancy pairs
BRIDGE = "#22a85c"    # bridge pairs
NEUTRAL_A = "#5d6478"  # ledger: prefill
NEUTRAL_B = "#8d94a8"  # ledger: generate

# ---- status (reserved; always with icon + label) ----
GOOD = "#0ca30c"
WARN = "#fab219"
CRIT = "#d03b3b"

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

FONT_SANS = ('"Inter", "Segoe UI Variable Text", "Segoe UI Variable Display", "Segoe UI", '
             'system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif')
FONT_MONO = '"JetBrains Mono", "Cascadia Code", "Cascadia Mono", Consolas, "Courier New", monospace'


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
        font=dict(family=FONT_SANS, size=14, color=INK_2),
        colorway=[FULL, TOPK, GREEDY, QWATT, QUANTUM],
        margin=dict(l=8, r=8, t=8, b=8),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=AXIS,
                        font=dict(family=FONT_SANS, size=14, color=INK)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=AXIS, linewidth=1, ticks="",
                   tickfont=dict(color=MUTED, size=13), title=dict(font=dict(color=MUTED, size=13)),
                   automargin=True),
        yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
                   tickfont=dict(color=MUTED, size=13), title=dict(font=dict(color=MUTED, size=13)),
                   automargin=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2, size=13),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        bargap=0.35,
        hovermode="closest",
    )
    pio.templates["qwatt"] = t
    pio.templates.default = "qwatt"


_register_template()

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


# ---------------------------------------------------------------- page CSS
CSS = f"""
<style>
:root {{
  --bg:{BG}; --surface:{SURFACE}; --surface2:{SURFACE_2}; --border:{BORDER};
  --ink:{INK}; --ink2:{INK_2}; --muted:{MUTED};
  --full:{FULL}; --qwatt:{QWATT}; --quantum:{QUANTUM}; --topk:{TOPK}; --greedy:{GREEDY};
  --red:{RED_LINK}; --bridge:{BRIDGE}; --good:{GOOD}; --warn:{WARN}; --crit:{CRIT};
  --sans:{FONT_SANS}; --mono:{FONT_MONO};
}}
/* ---------- hide Streamlit chrome ---------- */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
[data-testid="stAppDeployButton"] {{ display:none !important; visibility:hidden !important; }}
/* no fade while fragments re-run: keeps live updates smooth */
[data-stale="true"], .stale-element, .element-container[data-stale="true"] {{
  opacity:1 !important; transition:none !important; filter:none !important; }}
html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: radial-gradient(1200px 600px at 12% -10%, rgba(144,133,233,0.10), transparent 60%),
              radial-gradient(900px 500px at 95% 0%, rgba(20,171,126,0.08), transparent 55%),
              var(--bg) !important;
  color: var(--ink); font-family: var(--sans); }}
.stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{ font-family: var(--sans); }}
.block-container {{ padding-top: 1.1rem !important; padding-bottom: 2rem !important;
  max-width: 1560px !important; padding-left: 2.2rem !important; padding-right: 2.2rem !important; }}
[data-testid="stVerticalBlock"] {{ gap: 0.9rem; }}
p, li, label, .stMarkdown {{ font-size: 1.02rem; }}
code, pre, .mono {{ font-family: var(--mono) !important; }}

/* ---------- widgets ---------- */
[data-testid="stSlider"] label p, .stSelectbox label p, .stRadio label p {{
  color: var(--ink2) !important; font-size: .95rem !important; letter-spacing:.02em; }}
div[data-baseweb="select"] > div {{ background: var(--surface2) !important; border-color: var(--border) !important;
  border-radius: 12px !important; }}
.stButton > button, .stDownloadButton > button {{
  border-radius: 12px; border: 1px solid rgba(20,171,126,.55);
  background: linear-gradient(180deg, rgba(20,171,126,.25), rgba(20,171,126,.10));
  color: var(--ink); font-weight: 650; font-size: 1.05rem; padding: .6rem 1.2rem;
  box-shadow: 0 0 0 1px rgba(20,171,126,.15), 0 8px 30px -10px rgba(20,171,126,.6); }}
.stButton > button:hover {{ border-color: var(--qwatt); color: #fff;
  box-shadow: 0 0 0 1px rgba(20,171,126,.35), 0 10px 34px -8px rgba(20,171,126,.8); }}
.stButton > button:disabled {{ opacity:.45; box-shadow:none; }}
button[kind="secondary"] {{ }}
[data-testid="stExpander"] details {{ background: var(--surface); border: 1px solid var(--border) !important;
  border-radius: 16px !important; }}
[data-testid="stExpander"] summary p {{ font-size: 1.08rem !important; font-weight: 650; color: var(--ink) !important; }}
[data-testid="stSegmentedControl"] button, [data-testid="stBaseButton-segmented_control"],
[data-testid="stBaseButton-segmented_controlActive"] {{ font-size: 1rem !important; }}
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}

/* ---------- bordered containers = cards ---------- */
[class*="st-key-lane_"], [class*="st-key-sb"], [class*="st-key-histcard"], [class*="st-key-dialcard"] {{
  background: linear-gradient(180deg, rgba(255,255,255,.025), rgba(255,255,255,0)), var(--surface) !important;
  border-radius: 18px !important; border: 1px solid var(--border) !important; padding: 16px 18px !important; }}
[class*="st-key-lane_"][class*="qwatt"], [class*="st-key-lc_lane_"][class*="qwatt"] {{ border-color: rgba(20,171,126,.45) !important;
  box-shadow: 0 0 0 1px rgba(20,171,126,.10), 0 24px 60px -36px rgba(20,171,126,.9); }}
[class*="st-key-lane_"][class*="full"], [class*="st-key-lc_lane_"][class*="full"] {{ border-color: rgba(224,104,47,.40) !important;
  box-shadow: 0 0 0 1px rgba(224,104,47,.08), 0 24px 60px -36px rgba(224,104,47,.8); }}
[class*="st-key-lane_"][class*="topk"] {{ border-color: rgba(57,135,229,.40) !important; }}

/* ---------- top bar ---------- */
.qw-top {{ display:flex; align-items:center; justify-content:space-between; gap:20px; margin-bottom:4px; }}
.qw-brand {{ display:flex; align-items:center; gap:14px; }}
.qw-logo {{ width:46px; height:46px; border-radius:14px; display:grid; place-items:center;
  background: conic-gradient(from 210deg, var(--quantum), var(--qwatt), var(--quantum));
  box-shadow: 0 0 28px -4px rgba(144,133,233,.7); }}
.qw-logo span {{ width:38px; height:38px; border-radius:11px; background:var(--bg); display:grid; place-items:center;
  font-weight:800; font-size:20px; color:var(--ink); letter-spacing:-.04em; }}
.qw-title {{ font-size:30px; font-weight:780; letter-spacing:-.03em; color:var(--ink); line-height:1; }}
.qw-title b {{ background: linear-gradient(90deg, var(--quantum), var(--qwatt)); -webkit-background-clip:text;
  background-clip:text; color:transparent; }}
.qw-sub {{ color:var(--ink2); font-size:15px; margin-top:5px; }}
.qw-pills {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
.qw-pill {{ display:inline-flex; align-items:center; gap:8px; padding:7px 13px; border-radius:999px;
  border:1px solid var(--border); background:var(--surface); color:var(--ink2); font-size:13.5px; font-weight:600;
  white-space:nowrap; }}
.qw-pill .dot {{ width:9px; height:9px; border-radius:50%; background:var(--muted); }}
.qw-pill.live {{ border-color: rgba(230,103,103,.5); color:#fff; background: rgba(208,59,59,.16); }}
.qw-pill.live .dot {{ background:#ff5a5a; box-shadow:0 0 0 0 rgba(255,90,90,.7); animation: qwpulse 1.4s infinite; }}
.qw-pill.bench {{ border-color: rgba(144,133,233,.45); color:#fff; background: rgba(144,133,233,.12); }}
.qw-pill.bench .dot {{ background: var(--quantum); }}
.qw-pill.ok .dot {{ background: var(--good); }}
.qw-pill.warn {{ border-color: rgba(250,178,25,.45); color: #ffe2a3; }}
.qw-pill.warn .dot {{ background: var(--warn); }}
@keyframes qwpulse {{ 0% {{ box-shadow:0 0 0 0 rgba(255,90,90,.7);}} 70% {{ box-shadow:0 0 0 9px rgba(255,90,90,0);}}
  100% {{ box-shadow:0 0 0 0 rgba(255,90,90,0);}} }}

/* ---------- banners ---------- */
.qw-synth {{ margin: 6px 0 4px; padding: 12px 18px; border-radius: 14px; font-weight: 800; letter-spacing: .08em;
  color: #1a1200; font-size: 16px; text-align:center;
  background: repeating-linear-gradient(135deg, #fab219 0 18px, #f3a200 18px 36px);
  box-shadow: 0 0 0 2px rgba(250,178,25,.35), 0 10px 40px -12px rgba(250,178,25,.7); }}
.qw-synth small {{ display:block; font-weight:600; letter-spacing:.01em; font-size:13px; margin-top:2px; }}
.qw-flag {{ display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:8px; font-size:12.5px;
  font-weight:700; letter-spacing:.03em; background: rgba(250,178,25,.12); color:#ffd27a;
  border:1px solid rgba(250,178,25,.4); }}

/* ---------- section headers ---------- */
.qw-h {{ display:flex; align-items:baseline; gap:14px; margin: 18px 0 2px; }}
.qw-h .n {{ font-family:var(--mono); color:var(--muted); font-size:13px; font-weight:700; letter-spacing:.1em; }}
.qw-h h2 {{ font-size: 24px !important; font-weight: 760 !important; letter-spacing:-.02em; color:var(--ink) !important;
  margin:0 !important; padding:0 !important; }}
.qw-h .s {{ color: var(--muted); font-size: 14.5px; }}

/* ---------- cards ---------- */
.qw-card {{ background: linear-gradient(180deg, rgba(255,255,255,.025), rgba(255,255,255,0)) , var(--surface);
  border:1px solid var(--border); border-radius:18px; padding:18px 20px; position:relative; overflow:hidden; }}
.qw-hero {{ display:grid; grid-template-columns: 1.35fr 1.2fr 1fr 1fr; gap:14px; }}
.qw-stat .lbl {{ color:var(--ink2); font-size:14.5px; font-weight:600; letter-spacing:.01em; }}
.qw-stat .val {{ color:var(--ink); font-size:50px; font-weight:780; letter-spacing:-.035em; line-height:1.05; margin-top:6px; }}
.qw-stat .val small {{ font-size:22px; font-weight:650; color:var(--ink2); letter-spacing:-.01em; margin-left:4px; }}
.qw-stat .sub {{ color:var(--muted); font-size:13.5px; margin-top:6px; }}
.qw-stat.primary {{ border-color: rgba(20,171,126,.45);
  box-shadow: 0 0 0 1px rgba(20,171,126,.12), 0 20px 60px -30px rgba(20,171,126,.9); }}
.qw-stat.primary::before {{ content:""; position:absolute; inset:0; pointer-events:none;
  background: radial-gradient(400px 160px at 0% 0%, rgba(20,171,126,.20), transparent 70%); }}
.qw-stat.primary .val {{ font-size:64px; }}
.qw-duo {{ display:flex; gap:26px; align-items:flex-end; margin-top:6px; }}
.qw-duo .k {{ display:flex; align-items:center; gap:7px; color:var(--ink2); font-size:13.5px; font-weight:600; }}
.qw-duo .v {{ color:var(--ink); font-size:40px; font-weight:760; letter-spacing:-.03em; line-height:1.05; }}
.qw-duo .v small {{ font-size:18px; color:var(--ink2); margin-left:3px; }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:3px; }}
.qw-cal {{ color:var(--ink2); font-size:14.5px; margin-top:8px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
.qw-cal b {{ color:var(--ink); font-family:var(--mono); }}

/* ---------- lanes ---------- */
.qw-lane-h {{ display:flex; align-items:center; justify-content:space-between; gap:12px; }}
.qw-lane-name {{ flex:none; white-space:nowrap; display:flex; align-items:center; gap:10px; font-size:21px; font-weight:760; color:var(--ink); letter-spacing:-.02em; }}
.qw-lane-name .bar {{ width:6px; height:26px; border-radius:3px; }}
.qw-lane-name .tag {{ font-size:12.5px; font-weight:650; color:var(--muted); letter-spacing:.04em; }}
.qw-joules {{ text-align:right; }}
.qw-joules .v {{ font-size:46px; font-weight:780; color:var(--ink); letter-spacing:-.035em; line-height:1; }}
.qw-joules .v small {{ font-size:19px; color:var(--ink2); font-weight:650; margin-left:3px; }}
.qw-joules .s {{ color:var(--muted); font-size:12.5px; margin-top:3px; max-width:300px; margin-left:auto; }}
.qw-chips {{ display:flex; gap:6px; margin-top:12px; flex-wrap:wrap; }}
.qw-chip {{ padding:5px 11px; border-radius:999px; font-size:13px; font-weight:650; color:var(--muted);
  background: rgba(255,255,255,.03); border:1px solid var(--border); }}
.qw-chip.on {{ color:#fff; border-color: var(--c); background: color-mix(in srgb, var(--c) 22%, transparent);
  box-shadow: 0 0 18px -6px var(--c); }}
.qw-chip.on::before {{ content:""; display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--c);
  margin-right:7px; vertical-align:1px; animation: qwblink 1s infinite; }}
.qw-chip.past {{ color: var(--ink2); }}
.qw-chip.past::before {{ content:"✓ "; color: var(--ink2); }}
.qw-chip.skip {{ text-decoration: line-through; opacity:.5; }}
@keyframes qwblink {{ 50% {{ opacity:.35; }} }}
.qw-kv {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-top:4px; }}
.qw-kv > div {{ background: rgba(255,255,255,.025); border:1px solid var(--border); border-radius:12px; padding:9px 12px; }}
.qw-kv .k {{ color:var(--muted); font-size:12.5px; font-weight:600; }}
.qw-kv .v {{ color:var(--ink); font-size:21px; font-weight:720; letter-spacing:-.02em; }}
.qw-answer {{ margin-top:12px; padding:13px 15px; border-radius:14px; background: rgba(255,255,255,.03);
  border:1px solid var(--border); min-height:64px; }}
.qw-answer .q {{ color:var(--muted); font-size:12.5px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }}
.qw-answer .a {{ color:var(--ink); font-size:20px; font-weight:600; margin-top:4px; line-height:1.35; }}
.qw-answer .a.dim {{ color: var(--muted); font-weight:500; font-style: italic; }}
.qw-answer .cursor {{ display:inline-block; width:9px; height:20px; background:var(--ink2); vertical-align:-3px;
  margin-left:2px; animation: qwblink .8s infinite; }}
.qw-badge {{ display:inline-flex; align-items:center; gap:6px; padding:4px 11px; border-radius:999px; font-size:13px;
  font-weight:750; }}
.qw-badge.ok {{ color:#c9f7c9; background: rgba(12,163,12,.16); border:1px solid rgba(12,163,12,.55); }}
.qw-badge.bad {{ color:#ffc9c9; background: rgba(208,59,59,.16); border:1px solid rgba(208,59,59,.6); }}
.qw-badge.pend {{ color: var(--ink2); background: rgba(255,255,255,.04); border:1px solid var(--border); }}
.qw-verdict {{ display:flex; align-items:center; gap:18px; flex-wrap:wrap; padding:14px 20px; border-radius:16px;
  background: linear-gradient(90deg, rgba(20,171,126,.14), rgba(144,133,233,.08)); border:1px solid rgba(20,171,126,.35);
  font-size:17px; color:var(--ink); }}
.qw-verdict b {{ font-size:24px; letter-spacing:-.02em; }}
.qw-verdict .sep {{ width:1px; height:26px; background: var(--border); }}
.qw-empty {{ border:1px dashed rgba(255,255,255,.14); border-radius:16px; padding:22px; color:var(--ink2);
  background: rgba(255,255,255,.015); text-align:center; font-size:15.5px; }}
.qw-empty > b:first-child {{ color:var(--ink); display:block; font-size:17px; margin-bottom:4px; }}
.qw-empty b {{ color:var(--ink); }}
.qw-subh {{ font-size:18px; font-weight:720; color:var(--ink); letter-spacing:-.01em; margin:6px 0 2px; }}
.qw-empty code {{ color: var(--ink); background: rgba(255,255,255,.06); padding:1px 6px; border-radius:6px; }}

/* ---------- quantum lens ---------- */
.qw-caption {{ display:flex; align-items:center; gap:12px; padding:12px 16px; border-radius:14px;
  background: rgba(144,133,233,.10); border:1px solid rgba(144,133,233,.35); color:var(--ink); font-size:17px; font-weight:560; }}
.qw-caption .q {{ width:28px; height:28px; border-radius:8px; display:grid; place-items:center; font-weight:800;
  background: var(--quantum); color: #0b0d12; font-family: var(--mono); font-size: 14px; flex: none; }}
.qw-facts {{ display:flex; flex-wrap:wrap; gap:8px; }}
.qw-fact {{ padding:8px 12px; border-radius:12px; background: var(--surface); border:1px solid var(--border); }}
.qw-fact .k {{ color:var(--muted); font-size:11.5px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }}
.qw-fact .v {{ color:var(--ink); font-size:18px; font-weight:720; letter-spacing:-.01em; font-family: var(--sans); }}
.qw-fact.hi {{ border-color: rgba(144,133,233,.55); background: rgba(144,133,233,.10);
  box-shadow: 0 0 30px -12px rgba(144,133,233,.8); }}
.qw-legend {{ display:flex; gap:18px; flex-wrap:wrap; color:var(--ink2); font-size:13.5px; align-items:center; }}
.qw-legend span {{ display:inline-flex; align-items:center; gap:7px; }}
.qw-legend i {{ display:inline-block; width:22px; height:3px; border-radius:2px; }}
.qw-mini {{ color: var(--muted); font-size: 13.5px; }}
.qw-foot {{ margin-top: 18px; padding: 22px 26px; border-radius: 18px; border:1px solid var(--border);
  background: linear-gradient(90deg, rgba(20,171,126,.12), rgba(144,133,233,.10)); display:flex; gap:24px;
  align-items:center; justify-content:space-between; flex-wrap:wrap; }}
.qw-foot .big {{ font-size: 26px; font-weight: 760; color: var(--ink); letter-spacing:-.02em; }}
.qw-foot .big b {{ font-size: 34px; }}
.qw-foot .def {{ color: var(--ink2); font-size: 14px; }}
.qw-shift {{ font-size:16px; color:var(--ink); line-height:1.45; margin: 2px 0 2px; }}
.qw-shift b {{ font-size:20px; letter-spacing:-.01em; }}
[data-testid="stTabs"] button p {{ font-size: 15px !important; font-weight: 650; }}
.st-key-view button p {{ font-size: 15px !important; font-weight: 650 !important; }}
.st-key-view button {{ padding: 6px 16px !important; }}
.qw-explain {{ padding:14px 16px; border-radius:14px; background: rgba(255,255,255,.03); border:1px solid var(--border);
  color: var(--ink); font-size: 16px; line-height:1.45; }}
</style>
"""

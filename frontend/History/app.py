import json
import requests
import pandas as pd
import streamlit as st
from textwrap import dedent


# ---------------------------
# Column icons for gun table
# ---------------------------
GUNS_ICON_CFG = {
    # key      label shown in header   tooltip text
    "Powder": st.column_config.NumberColumn(
        "💨", help="Powder", step=0.1, format="%.2f"
    ),
    "Air": st.column_config.NumberColumn(
        "◁", help="Air", step=0.1, format="%.2f"
    ),
    "kV": st.column_config.NumberColumn(
        "kV", help="High voltage (kV)", step=1.0, format="%.0f"
    ),
    "µA": st.column_config.NumberColumn(
        "µA", help="Current (µA)", step=0.1, format="%.1f"
    ),
    "EClean": st.column_config.NumberColumn(
        "⬅️", help="Electrode cleaning", step=0.1, format="%.2f"
    ),
    "OnOff": st.column_config.NumberColumn(
        "|↔|", help="On/Off distance (before/after obj)", step=0.1, format="%.2f"
    ),
}


# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="OptiParam AI | Pilot", layout="wide", initial_sidebar_state="collapsed")
APP_TITLE = "OptiParam AI – Pilot"

# ---------------------------
# Session state defaults
# ---------------------------
ss = st.session_state
ss.setdefault("active_tab", "Prediction")
ss.setdefault("last_recommendation", None)
ss.setdefault("last_current_features", {})
ss.setdefault("last_prediction", None)
ss.setdefault("last_pred_meta", {})
ss.setdefault("last_rec_meta", {})
ss.setdefault("last_type_code", None)

# NEW: store the actual DataFrames we want to keep updated
ss.setdefault("guns_df", None)
ss.setdefault("axes_df", None)

# ---------------------------
# Query param helpers (for deep links)
# ---------------------------
def _get_query_params():
    try:
        return dict(st.query_params)  # Streamlit >= 1.30
    except Exception:
        qp = st.experimental_get_query_params()
        return {k: (v[0] if isinstance(v, list) else v) for k, v in qp.items()}

def _set_query_params(**kwargs):
    try:
        st.query_params.update(kwargs)  # Streamlit >= 1.30
    except Exception:
        st.experimental_set_query_params(**kwargs)

# Sync active_tab with URL (?view=Prediction|Recommendation)
_qp_view = _get_query_params().get("view")
if _qp_view in ("Prediction", "Recommendation") and _qp_view != ss.active_tab:
    ss.active_tab = _qp_view
elif _qp_view is None:
    _set_query_params(view=ss.active_tab)

def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def switch_view(view: str):
    ss["active_tab"] = view
    _set_query_params(view=view)
    do_rerun()

def _on_nav_change():
    choice = st.session_state.get("__nav_radio__", "Prediction")
    if choice in ("Prediction", "Recommendation") and choice != ss.active_tab:
        ss.active_tab = choice
        _set_query_params(view=choice)

# ---------------------------
# Sidebar: API base + health
# ---------------------------
API = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")

@st.cache_data(ttl=60)
def load_health(base):
    return requests.get(f"{base}/health", timeout=5).json()

@st.cache_data(ttl=60)
def load_types(base):
    try:
        return requests.get(f"{base}/types", timeout=5).json().get("types", [])
    except Exception:
        return []

@st.cache_data(ttl=60)
def load_specs(base, type_code):
    try:
        return requests.get(f"{base}/specs/{type_code}", timeout=5).json().get("specs", {})
    except Exception:
        return {}

@st.cache_data(ttl=60)
def load_bounds(base, type_code):
    try:
        return requests.get(f"{base}/feature-bounds/{type_code}", timeout=5).json().get("bounds", {})
    except Exception:
        return {}

# Health chip
try:
    h = load_health(API)
    st.sidebar.success(f"API OK • Models loaded: {h.get('models_loaded')} • v{h.get('version')}")
    ver = h.get("version", "—")
    models_loaded = h.get("models_loaded", "—")
    status_html = f"Model Version: {ver} • Models Loaded: {models_loaded}"
except Exception as e:
    st.sidebar.error(f"Health check failed: {e}")
    status_html = "API unavailable"

# ---------------------------
# Topbar + dim-white theme + centered sticky BIG pill radio + button/tooltip colors
# ---------------------------
st.markdown(dedent("""
<style>
:root {
  --topbar-h: 56px;
  --accent-h: 8px;
  --nav-h: 68px;
  --bg-dim: #f5f7fb;
  --text-main: #111111;
  --muted: #eef2f7;
  --pill-on: #2563eb;
  --pill-on-border: #1e40af;
}

/* --- Global background + black text (outside tables) --- */
html, body, .stApp, [data-testid="stAppViewContainer"], .block-container {
  background: var(--bg-dim) !important;
  color: var(--text-main) !important;
  color-scheme: light !important;
}
h1, h2, h3, h4, h5, h6,
p, span, label, small, strong, em,
[data-testid="stMarkdownContainer"],
[data-testid="stText"], [data-testid="stCaptionContainer"],
[data-testid="stForm"], [data-testid="stNumberInputContainer"],
[data-testid="stSliderContainer"], [data-testid="stSelectboxContainer"] {
  color: var(--text-main) !important;
}
/* Keep dataframe colors */
[data-testid="stDataFrame"] * { color: inherit !important; }

/* --- Buttons (NOT the tab selector): white text, hover blue, even when disabled --- */
div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button,
div[data-testid="stFormSubmitButton"] > button {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
  color: #2563eb !important;
  -webkit-text-fill-color: #2563eb !important;
}
div[data-testid="stButton"] > button *,
div[data-testid="stDownloadButton"] > button *,
div[data-testid="stFormSubmitButton"] > button * {
  color: inherit !important;
  -webkit-text-fill-color: inherit !important;
}
div[data-testid="stButton"] > button:disabled,
div[data-testid="stDownloadButton"] > button:disabled,
div[data-testid="stFormSubmitButton"] > button:disabled {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  opacity: 1 !important;
  filter: none !important;
}

/* --- Hide Streamlit default header/toolbar --- */
header[data-testid="stHeader"] { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }

/* --- Space for topbar + accent + tabs row --- */
.block-container,
[data-testid="stAppViewContainer"] .main .block-container {
  padding-top: var(--topbar-h) !important;
}

/* --- Topbar --- */
.opta-top-wrap { position: fixed; top: 0; left: 0; right: 0; z-index: 20000; }
.opta-accent {
  height: var(--accent-h);
  background: linear-gradient(90deg,
    #7f1d1d 0%, #a11f22 6%, #c81e1e 12%, #e0312b 18%, #ef4444 24%,
    #7e22ce 34%, #5b2bbf 40%, #3d3fb0 46%, #273f9a 52%, #1e3a8a 58%,
    #1d4ed8 64%, #3b82f6 70%, #22d3ee 78%, #06b6d4 82%, #10b981 88%,
    #22c55e 94%, #2e7d32 100%
  );
  z-index: 20001;
}
.opta-topbar {
  position: relative; isolation: isolate;
  height: var(--topbar-h);
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center; gap: 12px; padding: 0 16px;
  background: #ffffff !important;
  border-bottom: 1px solid rgba(0,0,0,0.08);
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  z-index: 20000;
}
.opta-topbar::before { content: ""; position: absolute; inset: 0; background: transparent !important; }
.opta-title { color: #111111 !important; font-weight: 700; font-size: 1.06rem; letter-spacing: 0.2px; }
.opta-right { color: #2b2b2b !important; opacity: 0.9; font-size: 0.9rem; white-space: nowrap; text-align: right; }

/* --- Sticky row that holds the radio (centered with columns below) --- */
#opta-tab-row {
  position: sticky; top: calc(var(--topbar-h) + var(--accent-h) + 6px);
  z-index: 19999; background: var(--bg-dim);
  padding: 6px 0 10px 0; border-bottom: 1px solid rgba(0,0,0,.05);
}

/* --- BIG pill radio styles (target by aria-label="Choose view") --- */
div[data-testid="stRadio"][aria-label="Choose view"] {
  display: inline-flex; align-items: center; gap: 10px;
  padding: 6px; background: var(--muted);
  border-radius: 999px; border: 1px solid rgba(0,0,0,0.10);
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
/* Hide the dots (but keep text) */
div[data-testid="stRadio"][aria-label="Choose view"] input[type="radio"] {
  position: absolute !important; opacity: 0 !important; width: 0 !important; height: 0 !important;
}
div[data-testid="stRadio"][aria-label="Choose view"] [role="radio"] > div:first-child { display: none !important; }
div[data-testid="stRadio"][aria-label="Choose view"] [data-baseweb="radio"] > label > div:first-child { display: none !important; }
div[data-testid="stRadio"][aria-label="Choose view"] svg { display: none !important; }
/* BIG labels */
div[data-testid="stRadio"][aria-label="Choose view"] label {
  display: inline-flex; align-items: center; justify-content: center;
  padding: 14px 28px; border-radius: 999px;
  border: 2px solid transparent; background: transparent;
  color: #1f2937; font-weight: 900; font-size: 1.15rem; line-height: 1.2;
  min-width: 220px; cursor: pointer; user-select: none;
}
div[data-testid="stRadio"][aria-label="Choose view"] [role="radio"][aria-checked="true"] label,
div[data-testid="stRadio"][aria-label="Choose view"] [data-baseweb="radio"][aria-checked="true"] > label {
  background: var(--pill-on); color: #ffffff !important; border-color: var(--pill-on-border);
  box-shadow: 0 4px 10px rgba(0,0,0,0.12), inset 0 -1px 0 rgba(0,0,0,0.04);
}
div[data-testid="stRadio"][aria-label="Choose view"] ~ [data-testid="stWidgetLabel"] { display: none !important; }
@media (max-width: 600px) {
  div[data-testid="stRadio"][aria-label="Choose view"] label {
    font-size: 1.0rem; padding: 12px 20px; min-width: 160px;
  }
}

/* --- Expander header --- */
details[data-testid="stExpander"] { background: transparent !important; }
details[data-testid="stExpander"] > summary {
  background-color: #24262c !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  opacity: 1 !important;
  outline: none !important;
}
details[data-testid="stExpander"][open] > summary {
  background-color: #24262c !important;
  color: #ffffff !important;
}
details[data-testid="stExpander"] > summary:hover,
details[data-testid="stExpander"] > summary:focus,
details[data-testid="stExpander"] > summary:active {
  background-color: #2b2e36 !important;
  color: #ffffff !important;
}
details[data-testid="stExpander"] > summary svg,
details[data-testid="stExpander"] > summary svg * {
  fill: #ffffff !important; stroke: #ffffff !important; color: #ffffff !important;
}
details[data-testid="stExpander"] > div { background: transparent !important; }
details[data-testid="stExpander"] > summary::marker { color: transparent !important; content: "" !important; }

/* --- Tooltips --- */
div[role="tooltip"], div[role="tooltip"] *,
[data-baseweb="tooltip"], [data-baseweb="tooltip"] * {
  color: #ffffff !important;
}
div[role="tooltip"], [data-baseweb="tooltip"] {
  background: rgba(0,0,0,0.92) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}

/* --- DataFrame hover toolbar --- */
[data-testid="stElementToolbar"] {
  background: rgba(0,0,0,0.85) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  color: #ffffff !important;
  box-shadow: 0 4px 10px rgba(0,0,0,0.25) !important;
  border-radius: 10px !important;
}
[data-testid="stElementToolbar"] * {
  background: transparent !important;
  box-shadow: none !important;
  filter: none !important;
}
[data-testid="stElementToolbar"] [data-testid="stElementToolbarButton"] [data-testid="stTooltipHoverTarget"] button,
[data-testid="stElementToolbar"] [data-testid="stElementToolbarButton"] button[kind="elementToolbar"] {
  background: transparent !important;
  border: none !important;
  color: #ffffff !important;
  outline: none !important;
  opacity: 1 !important;
}
[data-testid="stElementToolbar"] button svg path[fill="none"] { fill: none !important; }
[data-testid="stElementToolbar"] button svg { color: inherit !important; background: transparent !important; }
[data-testid="stElementToolbar"] [data-testid="stElementToolbarButtonIcon"],
[data-testid="stElementToolbar"] [data-testid="stElementToolbarButtonIcon"] * {
  background: transparent !important;
}
[data-testid="stElementToolbar"] [data-testid="stElementToolbarButton"] button:hover,
[data-testid="stElementToolbar"] [data-testid="stElementToolbarButton"] button:focus {
  color: #ffffff !important;
}

/* === Center headers & cells in ALL Streamlit tables === */

/* Make the cell / header containers flex and center their children */
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataEditor"] [role="columnheader"],
[data-testid="stDataEditor"] [role="gridcell"] {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

/* Make the inner span / text block fill and center its text */
[data-testid="stDataFrame"] [role="columnheader"] span,
[data-testid="stDataFrame"] [role="gridcell"] span,
[data-testid="stDataEditor"] [role="columnheader"] span,
[data-testid="stDataEditor"] [role="gridcell"] span {
  width: 100% !important;
  text-align: center !important;
}

/* Classic HTML tables (pandas Styler etc.) */
[data-testid="stDataFrame"] table th,
[data-testid="stDataFrame"] table td {
  text-align: center !important;
}

                
</style>
"""), unsafe_allow_html=True)

# ---------------------------
# Topbar HTML
# ---------------------------
st.markdown(dedent(f"""
<div class="opta-top-wrap">
  <div class="opta-accent"></div>
  <div class="opta-topbar">
    <div class="opta-title">{APP_TITLE}</div>
    <div></div>
    <div class="opta-right">{status_html}</div>
  </div>
</div>
"""), unsafe_allow_html=True)

# Centered sticky row with a 3-column layout; radio goes in the middle
st.markdown('<div id="opta-tab-row">', unsafe_allow_html=True)
c_left, c_mid, c_right = st.columns([1, 2, 1])
with c_mid:
    st.radio(
        "Choose view",
        options=["Prediction", "Recommendation"],
        index=(0 if ss.active_tab == "Prediction" else 1),
        key="__nav_radio__",
        horizontal=True,
        label_visibility="collapsed",
        on_change=_on_nav_change,
    )
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# API helpers
# ---------------------------
def api_get(path: str, params=None):
    url = f"{API}{path}"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def api_post(path: str, payload: dict):
    url = f"{API}{path}"
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def api_post_raw(path: str, payload: dict):
    url = f"{API}{path}"
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r

# ---------------------------
# Helpers for bounds/specs and feature tables
# ---------------------------
def mid(bounds: dict, name: str, default: float = 0.0) -> float:
    b = bounds.get(name)
    if b is None:
        return float(default)
    return float(b.get("mid", default))

def specs_to_df(specs: dict) -> pd.DataFrame:
    rows = []
    for p, s in (specs or {}).items():
        rows.append({
            "Point": p,
            "Lower (μm)": s.get("lo"),
            "Target (μm)": s.get("target"),
            "Upper (μm)": s.get("hi"),
            "Weight": s.get("w", 1.0),
        })
    df = pd.DataFrame(rows, columns=["Point", "Lower (μm)", "Target (μm)", "Upper (μm)", "Weight"])
    return df.sort_values("Point") if not df.empty else df

# ---- Dataset field builders ----
def gun_field(rec: str, gun: str, col_key: str) -> str:
    suffix = {
        "Powder": "Powder",
        "Air": "Total_Air",
        "kV": "Voltage",
        "µA": "Current",
        "EClean": "Electrode_Cleaning",
        "OnOff": "onoff_before_after_obj",
    }[col_key]
    return f"Rec_{rec}_Gun{gun}_{suffix}"

def axis_field(rec: str, axis: str, col_key: str) -> str:
    if col_key == "OnOff":
        if rec == "D" and axis == "Z":
            return "Rec_D_Axis-Z_Spray_onoff_before_after"
        return f"Rec_{rec}_Axis-{axis}_onoff_before_after"
    suffix = {
        "Upper": "Upper_Point",
        "Lower": "Lower_Point",
        "Speed": "Speed",
        "SprayDist": "Spray_Dist",
    }[col_key]
    return f"Rec_{rec}_Axis-{axis}_{suffix}"

# ---- Default table builders from bounds ----
def default_guns_df(bounds: dict) -> pd.DataFrame:
    rows = []
    layout = [("A","1"), ("B","1"), ("C","1"), ("C","2"),
              ("D","1"), ("D","2"), ("E","1"), ("F","1")]
    for rec, gun in layout:
        rows.append({
            "Rec": rec,
            "Gun": gun,
            "Powder":     mid(bounds, gun_field(rec, gun, "Powder"), 0.0),
            "Air":        mid(bounds, gun_field(rec, gun, "Air"), 0.0),
            "kV":         mid(bounds, gun_field(rec, gun, "kV"), 0.0),
            "µA":         mid(bounds, gun_field(rec, gun, "µA"), 0.0),
            "EClean":     mid(bounds, gun_field(rec, gun, "EClean"), 0.0),
            "OnOff":      mid(bounds, gun_field(rec, gun, "OnOff"), 0.0),
        })
    return pd.DataFrame(rows)

def default_axes_df(bounds: dict) -> pd.DataFrame:
    rows = []
    for rec in list("ABCDEF"):
        for axis in ("Z","X"):
            rows.append({
                "Rec": rec,
                "Axis": axis,
                "Upper":     mid(bounds, axis_field(rec, axis, "Upper"), 0.0),
                "Lower":     mid(bounds, axis_field(rec, axis, "Lower"), 0.0),
                "Speed":     mid(bounds, axis_field(rec, axis, "Speed"), 0.0),
                "SprayDist": mid(bounds, axis_field(rec, axis, "SprayDist"), 0.0),
                "OnOff":     mid(bounds, axis_field(rec, axis, "OnOff"), 0.0),
            })
    return pd.DataFrame(rows)

def guns_df_to_features(df: pd.DataFrame) -> dict:
    features = {}
    for _, r in df.iterrows():
        rec, gun = str(r["Rec"]), str(r["Gun"])
        for k in ["Powder","Air","kV","µA","EClean","OnOff"]:
            features[gun_field(rec, gun, k)] = float(r[k]) if pd.notna(r[k]) else 0.0
    return features

def axes_df_to_features(df: pd.DataFrame) -> dict:
    features = {}
    for _, r in df.iterrows():
        rec, axis = str(r["Rec"]), str(r["Axis"])
        for k in ["Upper","Lower","Speed","SprayDist","OnOff"]:
            fname = axis_field(rec, axis, k)
            features[fname] = float(r[k]) if pd.notna(r[k]) else 0.0
    return features

def apply_to_guns_df(base_guns_df: pd.DataFrame, features: dict) -> pd.DataFrame:
    df = base_guns_df.copy()
    for idx, r in df.iterrows():
        rec, gun = str(r["Rec"]), str(r["Gun"])
        for k in ["Powder","Air","kV","µA","EClean","OnOff"]:
            fname = gun_field(rec, gun, k)
            if fname in features and pd.notna(features[fname]):
                df.at[idx, k] = float(features[fname])
    return df

def apply_to_axes_df(base_axes_df: pd.DataFrame, features: dict) -> pd.DataFrame:
    df = base_axes_df.copy()
    for idx, r in df.iterrows():
        rec, axis = str(r["Rec"]), str(r["Axis"])
        for k in ["Upper","Lower","Speed","SprayDist","OnOff"]:
            fname = axis_field(rec, axis, k)
            if fname in features and pd.notna(features[fname]):
                df.at[idx, k] = float(features[fname])
    return df

def build_deltas_table(current: dict, proposed: dict) -> pd.DataFrame:
    rows = []
    for k, v in proposed.items():
        cur = current.get(k)
        if cur is None:
            continue
        try:
            rows.append({"Feature": k, "Current": float(cur), "Proposed": float(v), "Delta": float(v) - float(cur)})
        except Exception:
            pass
    df = pd.DataFrame(rows)
    return df.sort_values("Feature") if not df.empty else df

# ---- Styling helpers for changed cells ----
def _highlight_diff(df_rec: pd.DataFrame, df_cur: pd.DataFrame, numeric_cols: list, tol: float = 1e-9):
    styles = pd.DataFrame('', index=df_rec.index, columns=df_rec.columns)
    for c in numeric_cols:
        if c in df_rec.columns and c in df_cur.columns:
            try:
                delta = df_rec[c].astype(float) - df_cur[c].astype(float)
            except Exception:
                continue
            up = delta > tol
            down = delta < -tol
            styles.loc[up, c] = 'background-color: #c7b295; color: #000;'
            styles.loc[down, c] = 'background-color: #c7b295; color: #000;'
    return styles

def style_guns(rec_df: pd.DataFrame, cur_df: pd.DataFrame):
    numeric_cols = ["Powder","Air","kV","µA","EClean","OnOff"]
    return rec_df.style.apply(lambda _: _highlight_diff(rec_df, cur_df, numeric_cols), axis=None).format(precision=2)

def style_axes(rec_df: pd.DataFrame, cur_df: pd.DataFrame):
    numeric_cols = ["Upper","Lower","Speed","SprayDist","OnOff"]
    return rec_df.style.apply(lambda _: _highlight_diff(rec_df, cur_df, numeric_cols), axis=None).format(precision=2)

# ---------------------------
# Data editor on_change callbacks (StackOverflow-style fix)
# ---------------------------
def _update_guns_df():
    """Apply edits from data_editor 'guns_editor' into ss.guns_df."""
    if ss.guns_df is None:
        return
    data_df = ss.guns_df
    import streamlit.elements.widgets.data_editor as de
    import pyarrow as pa

    arrow_table = pa.Table.from_pandas(data_df)
    dataframe_schema = de.determine_dataframe_schema(data_df, arrow_table.schema)
    de._apply_dataframe_edits(data_df, st.session_state["guns_editor"], dataframe_schema)

    ss.guns_df = data_df

def _update_axes_df():
    """Apply edits from data_editor 'axes_editor' into ss.axes_df."""
    if ss.axes_df is None:
        return
    data_df = ss.axes_df
    import streamlit.elements.widgets.data_editor as de
    import pyarrow as pa

    arrow_table = pa.Table.from_pandas(data_df)
    dataframe_schema = de.determine_dataframe_schema(data_df, arrow_table.schema)
    de._apply_dataframe_edits(data_df, st.session_state["axes_editor"], dataframe_schema)

    ss.axes_df = data_df

# ---------------------------
# Types/specs/bounds
# ---------------------------
types = load_types(API)
if not types:
    st.warning("No product types returned. Ensure DB is seeded or training populated product_type.")

type_code = st.selectbox("Product Type", types or ["Type_1"])
specs = load_specs(API, type_code)
bounds = load_bounds(API, type_code)

with st.expander("Thickness specs (A–G)", expanded=False):
    if specs:
        st.dataframe(specs_to_df(specs), use_container_width=True)
    else:
        st.error("No thickness specs found. Seed the DB (sql/seed_specs.sql) or run training to populate dbo.thickness_spec.")

# Initialize / reset editors and cached results on type change
if ss.last_type_code != type_code:
    ss.guns_df = default_guns_df(bounds)
    ss.axes_df = default_axes_df(bounds)
    ss.last_type_code = type_code
    ss.last_prediction = None
    ss.last_recommendation = None
    ss.last_current_features = {}

# ---------------------------
# View selection (segmented radio drives it)
# ---------------------------
selected_view = ss.get("active_tab", "Prediction")

# ===========================
# Prediction View
# ===========================
if selected_view == "Prediction":
    st.subheader("Prediction")

    # Environment / Context
    st.markdown("Environment / Context")
    c1, c2, c3 = st.columns(3)
    with c1:
        humidity = st.number_input("Humidity (%)", value=float(bounds.get("Humidity", {}).get("mid", 50.0)))
    with c2:
        temperature = st.number_input("Temperature (°C)", value=float(bounds.get("Temperature", {}).get("mid", 23.0)))
    with c3:
        valve_status = st.number_input("Valve_Filter_Status", value=float(bounds.get("Valve_Filter_Status", {}).get("mid", 0.0)))

    # Spray gun table (editable)
    st.markdown("Spray gun parameters")

    guns_cfg = GUNS_ICON_CFG
    

    # guns_cfg = {
    #     "Powder":     st.column_config.NumberColumn("Powder", step=0.1, format="%.2f"),
    #     "Air":        st.column_config.NumberColumn("Air", step=0.1, format="%.2f"),
    #     "kV":         st.column_config.NumberColumn("kV", step=1.0, format="%.0f"),
    #     "µA":         st.column_config.NumberColumn("µA", step=0.1, format="%.1f"),
    #     "EClean":     st.column_config.NumberColumn("Electrode Clean", step=0.1, format="%.2f"),
    #     "OnOff":      st.column_config.NumberColumn("On/Off (before/after obj)", step=0.1, format="%.2f"),
    # }

    if ss.guns_df is None:
        ss.guns_df = default_guns_df(bounds)

    st.data_editor(
        ss.guns_df,
        hide_index=True,
        use_container_width=True,
        column_config=guns_cfg,
        disabled=["Rec","Gun"],
        key="guns_editor",
        on_change=_update_guns_df,
    )

    # Axis table (editable)
    st.markdown("Axis parameters")
    axes_cfg = {
        "Upper":     st.column_config.NumberColumn("Upper", step=1.0, format="%.0f"),
        "Lower":     st.column_config.NumberColumn("Lower", step=1.0, format="%.0f"),
        "Speed":     st.column_config.NumberColumn("Speed", step=0.1, format="%.1f"),
        "SprayDist": st.column_config.NumberColumn("Spray Dist", step=0.1, format="%.1f"),
        "OnOff":     st.column_config.NumberColumn("On/Off", step=1.0, format="%.0f"),
    }

    if ss.axes_df is None:
        ss.axes_df = default_axes_df(bounds)

    st.data_editor(
        ss.axes_df,
        hide_index=True,
        use_container_width=True,
        column_config=axes_cfg,
        disabled=["Rec","Axis"],
        key="axes_editor",
        on_change=_update_axes_df,
    )

    c_left, _, c_right = st.columns([1,1,1])
    with c_left:
        if st.button("Reset to medians"):
            ss.guns_df = default_guns_df(bounds)
            ss.axes_df = default_axes_df(bounds)
            ss.last_current_features = {}
            do_rerun()

    with c_right:
        if st.button("Predict", type="primary"):
            # Use the DataFrames stored in session (already updated by callbacks)
            guns_df = ss.guns_df
            axes_df = ss.axes_df

            feat = {}
            feat.update(guns_df_to_features(guns_df))
            feat.update(axes_df_to_features(axes_df))
            feat["Humidity"] = float(humidity)
            feat["Temperature"] = float(temperature)
            feat["Valve_Filter_Status"] = float(valve_status)
            payload = {"type_code": type_code, "features": feat}

            with st.spinner("Predicting..."):
                r = api_post_raw("/predict", payload)
                resp = r.json()
                req_id = r.headers.get("X-Request-ID")
                run_id = r.headers.get("X-Run-ID")

            ss.last_prediction = resp
            ss.last_pred_meta = {"req_id": req_id, "run_id": run_id}
            switch_view("Prediction")

    # Show last prediction
    pred_cache = ss.get("last_prediction")
    if pred_cache:
        preds = pred_cache.get("predictions", {})
        in_spec = pred_cache.get("in_spec", {})
        margins = pred_cache.get("margins", {})
        meta = ss.get("last_pred_meta", {})
        df = pd.DataFrame({
            "Predicted (μm)": preds,
            "In spec": in_spec,
            "Margin to limit (μm)": margins
        })
        st.success("Prediction results")
        st.dataframe(df, use_container_width=True)
        st.caption(f"Model version: {pred_cache.get('model_version')} • Request: {meta.get('req_id')} • Run: {meta.get('run_id')}")

# ===========================
# Recommendation View
# ===========================
if selected_view == "Recommendation":
    st.subheader("Recommendation")
    st.caption("Set targets (defaults to spec midpoint). Optimizer searches within learned ranges (q05–q95) for tunable parameters. You can also bound moves around current (± step %).")

    if not specs:
        st.info("Targets cannot be set until specs are available.")
    else:
        default_targets = {p: float((s["lo"] + s["hi"]) / 2.0) for p, s in specs.items()}

        with st.form("recommend_form"):
            tgt_cols = st.columns(4)
            targets = {}
            points_sorted = sorted(specs.keys())
            for i, p in enumerate(points_sorted):
                col = tgt_cols[i % 4]
                with col:
                    s = specs[p]
                    targets[p] = st.number_input(
                        f"Target Point {p} (μm) [{s['lo']}–{s['hi']}]",
                        min_value=float(s["lo"]),
                        max_value=float(s["hi"]),
                        value=float(default_targets[p]),
                        step=1.0,
                    )

            st.write("Context (fixed values; optimizer will not change these)")
            c1, c2, c3 = st.columns(3)
            ctx = {}
            with c1:
                ctx["Humidity"] = st.number_input("Ctx Humidity (%)", value=float(bounds.get("Humidity", {}).get("mid", 50.0)), key="ctx_hum")
            with c2:
                ctx["Temperature"] = st.number_input("Ctx Temperature (°C)", value=float(bounds.get("Temperature", {}).get("mid", 23.0)), key="ctx_temp")
            with c3:
                ctx["Valve_Filter_Status"] = st.number_input("Ctx Valve_Filter_Status", value=float(bounds.get("Valve_Filter_Status", {}).get("mid", 0.0)), key="ctx_valve")

            trials = st.slider("Optimization trials", min_value=50, max_value=600, value=150, step=25)
            timeout = st.slider("Timeout (seconds)", min_value=10, max_value=50, value=15, step=5)
            max_step_pct = st.slider("Max step (%) around current (per side, of q05–q95 range)", 0.0, 20.0, 2.0, 0.5)

            submit_rec = st.form_submit_button("Recommend")

        if submit_rec:
            guns_df = ss.guns_df if ss.guns_df is not None else default_guns_df(bounds)
            axes_df = ss.axes_df if ss.axes_df is not None else default_axes_df(bounds)

            current = {}
            current.update(guns_df_to_features(guns_df))
            current.update(axes_df_to_features(axes_df))

            payload = {
                "type_code": type_code,
                "targets": targets,
                "fixed_context": ctx,
                "current": current,
                "step_pct": float(max_step_pct) / 100.0,
                "n_trials": int(trials),
                "timeout_sec": int(timeout),
            }

            with st.spinner("Optimizing..."):
                r = api_post_raw("/recommend", payload)
                resp = r.json()
                req_id = r.headers.get("X-Request-ID")
                run_id = r.headers.get("X-Run-ID")

            ss.last_recommendation = resp
            ss.last_rec_meta = {"req_id": req_id, "run_id": run_id}
            ss.last_current_features = current
            switch_view("Recommendation")

    # Render last recommendation
    rec_resp = ss.get("last_recommendation")
    if rec_resp:
        rec_dict = rec_resp.get("recommended", {}) or {}
        predicted = rec_resp.get("predicted", {}) or {}

        base_guns_df = ss.guns_df if ss.guns_df is not None else default_guns_df(bounds)
        base_axes_df = ss.axes_df if ss.axes_df is not None else default_axes_df(bounds)

        current = ss.get("last_current_features", {})
        if not current:
            current = {}
            current.update(guns_df_to_features(base_guns_df))
            current.update(axes_df_to_features(base_axes_df))

        rec_guns_df = apply_to_guns_df(base_guns_df, rec_dict)
        rec_axes_df = apply_to_axes_df(base_axes_df, rec_dict)

        st.success("Recommended settings (tables)")
        st.markdown("Recommended – Spray gun parameters")
        try:
            st.dataframe(
                style_guns(rec_guns_df, base_guns_df),
                use_container_width=True,
                column_config=GUNS_ICON_CFG,   # same icons
            )
        except Exception:
            st.dataframe(
                rec_guns_df,
                use_container_width=True,
                column_config=GUNS_ICON_CFG,
            )


        st.markdown("Recommended – Axis parameters")
        try:
            st.dataframe(style_axes(rec_axes_df, base_axes_df), use_container_width=True)
        except Exception:
            st.dataframe(rec_axes_df, use_container_width=True)

        st.markdown("Proposed deltas (tunable features)")
        deltas_df = build_deltas_table(current, rec_dict)
        if not deltas_df.empty:
            st.dataframe(deltas_df, use_container_width=True)
        else:
            st.info("No tunable feature deltas were proposed (optimizer may have found current settings close to optimal).")

        preds_df = pd.DataFrame({
            "Predicted (μm)": predicted,
            "In spec": {k: (specs.get(k, {}).get("lo", -1e9) <= v <= specs.get(k, {}).get("hi", 1e9)) for k, v in predicted.items()},
            "Margin to limit (μm)": {
                k: min(v - specs.get(k, {}).get("lo", v), specs.get(k, {}).get("hi", v) - v) for k, v in predicted.items()
            }
        })
        st.subheader("Predicted thickness (μm)")
        st.dataframe(preds_df, use_container_width=True)

        c_apply, c_download, c_clear = st.columns([1,1,1])
        with c_apply:
            if st.button("Apply recommended to editors", type="primary"):
                ss.guns_df = rec_guns_df
                ss.axes_df = rec_axes_df
                do_rerun()
        with c_download:
            st.download_button(
                "Download recommendation JSON",
                data=json.dumps(rec_resp, indent=2),
                file_name="recommendation.json",
                mime="application/json",
            )
        with c_clear:
            if st.button("Clear recommended result"):
                ss.last_recommendation = None
                ss.last_current_features = {}
                do_rerun()

# Footer
st.markdown("---")
st.caption("Use the selector above to switch views. Tables default to learned medians; Reset restores them.")

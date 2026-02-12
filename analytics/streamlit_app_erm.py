# streamlit_app_erm.py
"""
ERM Analytics Dashboard
Three-tab structure: Stratifications | Scenario Analysis | Static Pools
Focused on Equity Release Mortgage portfolio analytics
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import subprocess
import yaml
import base64
import io

# ============================
# Azure Blob Storage integration (optional)
# ============================
try:
    from blob_storage import (
        is_azure_configured,
        list_canonical_csvs,
        download_blob_to_dataframe,
    )
    BLOB_STORAGE_AVAILABLE = is_azure_configured()
except ImportError:
    BLOB_STORAGE_AVAILABLE = False

# ============================
# 1. CONFIGURATION & THEME
# ============================

# Default Fallbacks (used if YAML is missing)
PRIMARY_COLOR = "#232D55"    # Navy
SECONDARY_COLOR = "#919DD1"  # Light Blue
ACCENT_COLOR = "#BFBFBF"     # Grey (Default)
TEXT_DARK = "#2D2D2D"
TEXT_LIGHT = "#6B6B6B"
BACKGROUND_LIGHT = "#F8F9FA"
BORDER_COLOR = "#E0E0E0"     # NEW: Neutral grey for UI borders (decoupled from brand accent)
CHART_COLORS = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR]
MAX_ROWS = 100000
CLIENT_DISPLAY_NAME = "Portfolio Analytics Platform"

# Load YAML to override defaults
def load_client_config():
    config_path = Path("config_ERM_UK.yaml")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return {}
    return {}

CLIENT_CONFIG = load_client_config()

# Apply Overrides
if CLIENT_CONFIG.get("client", {}).get("display_name"):
    CLIENT_DISPLAY_NAME = CLIENT_CONFIG["client"]["display_name"]

theme_conf = CLIENT_CONFIG.get("mi", {}).get("branding", {}).get("theme", {})
if theme_conf:
    PRIMARY_COLOR = theme_conf.get("primary_color", PRIMARY_COLOR)
    SECONDARY_COLOR = theme_conf.get("secondary_color", SECONDARY_COLOR)
    # Note: We keep BORDER_COLOR neutral, but allow Accent to change for highlights
    ACCENT_COLOR = theme_conf.get("accent_color", ACCENT_COLOR)
    
    # Rebuild Chart Palette dynamically
    CHART_COLORS = [
        PRIMARY_COLOR, 
        SECONDARY_COLOR, 
        ACCENT_COLOR, 
        "#7EBAB5", # Teal-ish fallback
        "#A3CCC9"  # Light teal fallback
    ]
      
try:
    from scenario_engine import (
        project_portfolio,
        ScenarioAssumptions,
        PRESET_SCENARIOS,
        compare_scenarios,
    )
    SCENARIO_ENGINE_AVAILABLE = True
except ImportError:
    SCENARIO_ENGINE_AVAILABLE = False
    print("Warning: scenario_engine module not found. Scenario Analysis tab will be disabled.")

from mi_prep import (
    assert_trusted_canonical,
    add_presentation_aliases,
    add_buckets,
    format_currency,
    weighted_average,
)

# Risk monitoring
try:
    from risk_monitor import RiskMonitor, LimitCheck
    from risk_limits_config import ALL_LIMITS, LIMIT_CATEGORIES
    RISK_MONITORING_AVAILABLE = True
except ImportError:
    RISK_MONITORING_AVAILABLE = False
    print("Warning: Risk monitoring modules not found. Risk tab will be disabled.")

# Upload page integration
try:
    from upload_page import show_upload_page, reset_upload_state
    UPLOAD_PAGE_AVAILABLE = True
except ImportError:
    UPLOAD_PAGE_AVAILABLE = False

# ============================
# Refactor split imports (data + charts layers)
# ============================

from static_pools_core import (
    StaticPoolsSpec,
    build_static_pools_panel,
    add_segment_label,
)

from charts_plotly import (
    apply_chart_theme,
    strat_bar_chart_pure,
)

# ============================
# FILE UTILITIES (inline - no data_layer)
# ============================

def validate_file_path_pure(path_str: str):
    """Validate file path and return resolved Path object."""
    from pathlib import Path
    p = Path(path_str).resolve()
    if not p.exists():
        raise ValueError(f"File not found: {path_str}")
    if not p.is_file():
        raise ValueError("Path must be a file.")
    return p

def validate_file_path(path_str: str):
    """UI wrapper: preserves original st.error behaviour."""
    try:
        return validate_file_path_pure(path_str)
    except ValueError as e:
        st.error(str(e)) 
        return None

@st.cache_data
def load_data(path: str):
    """Load canonical CSV and prepare for dashboard presentation."""
    try:
        # -----------------------------
        # 0) Read CSV (guard MAX_ROWS)
        # -----------------------------
        nrows = None
        if isinstance(MAX_ROWS, int) and MAX_ROWS > 0:
            nrows = MAX_ROWS

        df = pd.read_csv(path, low_memory=False)

        # -----------------------------
        # Helpers: robust numeric parsing
        # -----------------------------
        def _to_num(series: pd.Series) -> pd.Series:
            """
            Convert strings like '£100,000', '100,000', '100 000', '(1,234.56)'
            into numerics. Returns float with NaN for non-parsable.
            """
            if series is None:
                return series
            if pd.api.types.is_numeric_dtype(series):
                return series

            s = series.astype("string").str.strip()

            # Handle parentheses as negatives: (123) -> -123
            s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

            # Strip currency symbols/letters; keep digits, sign, dot, comma
            s = s.str.replace(r"[^\d\-\.\,]", "", regex=True)

            # Remove thousand separators (commas)
            s = s.str.replace(",", "", regex=False)

            return pd.to_numeric(s, errors="coerce")

        # -----------------------------
        # 1) Normalization
        # -----------------------------
        # Interest rate normalization
        if "current_interest_rate" in df.columns:
            df["current_interest_rate"] = _to_num(df["current_interest_rate"])
            non_null = df["current_interest_rate"].dropna()
            if not non_null.empty and non_null.median() > 1:
                df["current_interest_rate"] = df["current_interest_rate"] / 100.0

        # LTV normalization (TARGET STATE: 0-100)
        for ltv_col in ["current_loan_to_value", "original_loan_to_value"]:
            if ltv_col in df.columns:
                df[ltv_col] = _to_num(df[ltv_col])
                non_null = df[ltv_col].dropna()
                # If legacy data (0.36), convert to 36.0
                if not non_null.empty and non_null.median() <= 1.0:
                    df[ltv_col] = df[ltv_col] * 100.0

        # Balance columns (robust parse)
        for bal_col in ["current_outstanding_balance", "current_principal_balance", "total_balance", "arrears_balance"]:
            if bal_col in df.columns:
                df[bal_col] = _to_num(df[bal_col]).fillna(0.0)

        # STEP A: Create 'total_balance' (Used by Stratifications/Charts)
        if "total_balance" not in df.columns:
            if "current_outstanding_balance" in df.columns:
                df["total_balance"] = df["current_outstanding_balance"]
            elif "current_principal_balance" in df.columns:
                df["total_balance"] = df["current_principal_balance"]
            else:
                df["total_balance"] = 0.0
        
        # STEP B: Fix Risk Monitor (It demands 'current_principal_balance')
        # If the file has 'current_outstanding_balance' but NOT 'current_principal_balance',
        # we must copy the data over, otherwise Risk Monitor sees 0.
        if "current_principal_balance" not in df.columns or df["current_principal_balance"].sum() == 0:
             df["current_principal_balance"] = df["total_balance"]

        # Age columns
        if "youngest_borrower_age" in df.columns:
            df["youngest_borrower_age"] = _to_num(df["youngest_borrower_age"])

        # Date parsing - handle multiple formats
        date_cols = ["origination_date", "maturity_date", "application_date"]

        for col in date_cols:
            if col in df.columns:
                s = df[col].astype("string").str.strip()

                # Two-pass parse:
                # 1) ISO-like values (YYYY-MM-DD...) parsed explicitly
                iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}")
                parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

                if iso_mask.any():
                    parsed.loc[iso_mask] = pd.to_datetime(s.loc[iso_mask], errors="coerce", format="%Y-%m-%d")

                # 2) Everything else parsed as day-first (UK)
                non_iso = ~iso_mask
                if non_iso.any():
                    parsed.loc[non_iso] = pd.to_datetime(s.loc[non_iso], errors="coerce", dayfirst=True)

                df[col] = parsed

        # -----------------------------
        # 2) Optional diagnostics (useful until stable)
        # -----------------------------
        # st.caption(f"DEBUG read_csv rows: {len(df):,}")
        # if "total_balance" in df.columns:
        #     st.caption(f"DEBUG total_balance non-null: {df['total_balance'].notna().sum():,}")

        # -----------------------------
        # 3) Validate + presentation layer
        # -----------------------------
        check_result = mi_prep.assert_trusted_canonical(df)
        if not check_result.ok:
            st.warning(f"⚠️ Input may not be canonical pipeline output. Missing: {check_result.missing_required}")
            for note in check_result.notes:
                st.info(note)

        df = mi_prep.add_presentation_aliases(df)
        df = mi_prep.add_buckets(df)

        return df

    except Exception as e:
        raise

@st.cache_data
def load_data_from_blob(blob_name: str, container: str | None = None):
    """Load canonical CSV from Azure Blob Storage and prepare for dashboard."""
    try:
        df = download_blob_to_dataframe(blob_name, container)
        return _prepare_dataframe(df)
    except Exception:
        raise


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared normalization + validation logic used by both local and blob loaders.
    Extracted from load_data so both paths produce identical results.
    """
    # --- Helpers: robust numeric parsing ---
    def _to_num(series: pd.Series) -> pd.Series:
        if series is None:
            return series
        if pd.api.types.is_numeric_dtype(series):
            return series
        s = series.astype("string").str.strip()
        s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        s = s.str.replace(r"[^\d\-\.\,]", "", regex=True)
        s = s.str.replace(",", "", regex=False)
        return pd.to_numeric(s, errors="coerce")

    # Interest rate normalization
    if "current_interest_rate" in df.columns:
        df["current_interest_rate"] = _to_num(df["current_interest_rate"])
        non_null = df["current_interest_rate"].dropna()
        if not non_null.empty and non_null.median() > 1:
            df["current_interest_rate"] = df["current_interest_rate"] / 100.0

    # LTV normalization (TARGET STATE: 0-100)
    for ltv_col in ["current_loan_to_value", "original_loan_to_value"]:
        if ltv_col in df.columns:
            df[ltv_col] = _to_num(df[ltv_col])
            non_null = df[ltv_col].dropna()
            if not non_null.empty and non_null.median() <= 1.0:
                df[ltv_col] = df[ltv_col] * 100.0

    # Balance columns (robust parse)
    for bal_col in ["current_outstanding_balance", "current_principal_balance", "total_balance", "arrears_balance"]:
        if bal_col in df.columns:
            df[bal_col] = _to_num(df[bal_col]).fillna(0.0)

    if "total_balance" not in df.columns:
        if "current_outstanding_balance" in df.columns:
            df["total_balance"] = df["current_outstanding_balance"]
        elif "current_principal_balance" in df.columns:
            df["total_balance"] = df["current_principal_balance"]
        else:
            df["total_balance"] = 0.0

    if "current_principal_balance" not in df.columns or df["current_principal_balance"].sum() == 0:
        df["current_principal_balance"] = df["total_balance"]

    if "youngest_borrower_age" in df.columns:
        df["youngest_borrower_age"] = _to_num(df["youngest_borrower_age"])

    # Date parsing
    date_cols = ["origination_date", "maturity_date", "application_date"]
    for col in date_cols:
        if col in df.columns:
            s = df[col].astype("string").str.strip()
            iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}")
            parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
            if iso_mask.any():
                parsed.loc[iso_mask] = pd.to_datetime(s.loc[iso_mask], errors="coerce", format="%Y-%m-%d")
            non_iso = ~iso_mask
            if non_iso.any():
                parsed.loc[non_iso] = pd.to_datetime(s.loc[non_iso], errors="coerce", dayfirst=True)
            df[col] = parsed

    check_result = mi_prep.assert_trusted_canonical(df)
    if not check_result.ok:
        st.warning(f"⚠️ Input may not be canonical pipeline output. Missing: {check_result.missing_required}")
        for note in check_result.notes:
            st.info(note)

    df = mi_prep.add_presentation_aliases(df)
    df = mi_prep.add_buckets(df)
    return df


def fmt_float(x, decimals=1, suffix="", na="N/A"):
    """Safely format floats that may be None/NaN."""
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return na
        return f"{float(x):.{decimals}f}{suffix}"
    except Exception:
        return na

# Chart Wrapper to FORCE COLORS from YAML
def strat_bar_chart(df, group_col, value_col="total_balance", agg="sum", title=""):
    """Wrapper that enforces the correct PRIMARY_COLOR on bars."""
    # Call the pure generator
    fig, msg, level = strat_bar_chart_pure(df, group_col, value_col, agg, title)
    
    if fig:
        # CRITICAL FIX: Overwrite the color to match the loaded config (Navy)
        # because charts_plotly.py uses its own static import which might be wrong.
        fig.update_traces(marker_color=PRIMARY_COLOR)
        
    if msg:
        st.info(msg) if level == "info" else st.warning(msg)
        return None
    return fig

def coerce_datetime(series: pd.Series, dayfirst: bool = True) -> pd.Series:
    """
    Robust datetime coercion for mixed user-edited CSVs.
    - Handles object/str, datetime, and period-like series safely.
    - Forces datetime64[ns] output (or NaT).
    """
    if series is None:
        return series
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        if pd.api.types.is_period_dtype(series):
            return series.dt.to_timestamp()
        return pd.to_datetime(series.astype(str), errors="coerce", dayfirst=dayfirst)
    except Exception:
        return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)

# ============================
# CLI + LAST RUN TYPED FILE SUPPORT
# ============================

CLI_FILE_PATH = None

if "--file" in sys.argv:
    try:
        CLI_FILE_PATH = sys.argv[sys.argv.index("--file") + 1]
    except (IndexError, ValueError):
        CLI_FILE_PATH = None

LATEST_TYPED_PATH_FILE = Path("latest_typed_path.txt")
LAST_RUN_TYPED_PATH = None

if LATEST_TYPED_PATH_FILE.exists():
    try:
        LAST_RUN_TYPED_PATH = LATEST_TYPED_PATH_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        LAST_RUN_TYPED_PATH = None

# ============================
# PAGE SETTINGS
# ============================

st.set_page_config(
    page_title="ERM Analytics Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# MATERIAL ICONS - START
# ===========================================================================

st.markdown("""
<style>
/* Ultra-simple fix: Hide ALL Material Icons text, don't try to render them */

/* Hide dropdown arrow text - Streamlit has default arrows anyway */
[data-baseweb="select"] svg,
[data-baseweb="select"] [class*="IconContainer"],
.stSelectbox svg,
.stMultiSelect svg {
    font-size: 0 !important;
    color: transparent !important;
}

/* Hide expander arrow text - Streamlit has default arrows */
[data-testid="stExpanderToggleIcon"] {
    font-size: 0 !important;
    color: transparent !important;
}

/* Nuclear option: Hide ANY text containing "keyboard" */
body * {
    text-rendering: optimizeLegibility;
}

/* Hide any element that has keyboard in its content */
*:not(script):not(style) {
    font-variant-ligatures: none !important;
}
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# MATERIAL ICONS - END
# ===========================================================================

# Upload page routing
if UPLOAD_PAGE_AVAILABLE:
    show_upload_ui = not CLI_FILE_PATH and not LAST_RUN_TYPED_PATH
    if st.session_state.get("force_upload_page", False):
        show_upload_ui = True
    
    if show_upload_ui:
        should_continue = show_upload_page()
        if not should_continue:
            st.stop()

# ============================
# CSS / THEME
# ============================

st.markdown(f"""
<style>
/* Global Font & Reset */
body, p, div, span, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stMetric {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
    color: {TEXT_DARK};
}}

/* Hide Streamlit Elements */
div[data-testid="stDecoration"], header {{
    display: none !important;
}}

.block-container {{
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: none;
}}

/* ===== HEADER ===== */
.header-container {{
    background: linear-gradient(90deg, #1E2540 0%, {PRIMARY_COLOR} 100%);
    padding: 1.5rem 2.5rem;
    margin: -1rem -1rem 2rem -1rem;
    border-radius: 0 0 12px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    width: calc(100% + 2rem);
    color: white;
}}

.header-left h1 {{ font-size: 26px; font-weight: 700; margin: 0; color: white !important; }}
.header-left p {{ font-size: 15px; margin: 4px 0 0 0; color: rgba(255, 255, 255, 0.9) !important; font-weight: 400; }}

.header-right {{ display: flex; align-items: center; gap: 20px; }}
.header-date {{ text-align: right; }}
.date-label {{ font-size: 11px; text-transform: uppercase; color: rgba(255, 255, 255, 0.6) !important; font-weight: 600; display: block; }}
.date-value {{ font-size: 15px; font-weight: 600; color: white !important; }}

.header-logo-box {{ 
    background: rgba(255, 255, 255, 0.95); 
    padding: 8px 16px; 
    border-radius: 8px; 
    height: 50px; 
    min-width: 120px; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}}
.header-logo-box img {{ height: 32px !important; width: auto !important; object-fit: contain; }}

/* ===== KPI TILES (Neutral Grey Borders) ===== */
.kpi-box {{
    background: white;
    border: 1px solid {BORDER_COLOR}; 
    border-radius: 12px;
    padding: 1.5rem 1rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    transition: all 0.3s ease;
    height: 100%;
}}
.kpi-box:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border-color: {PRIMARY_COLOR};
}}
.kpi-label {{ font-size: 12px; color: {TEXT_LIGHT}; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem; }}
.kpi-value {{ font-size: 32px; font-weight: 700; color: {PRIMARY_COLOR}; margin: 0.2rem 0; line-height: 1.2; }}
.kpi-subtitle {{ font-size: 11px; color: {TEXT_LIGHT}; margin-top: 0.5rem; }}

/* ===== TABS (Readable Text) ===== */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background-color: {BACKGROUND_LIGHT};
    padding: 0.5rem;
    border-radius: 8px;
}}

.stTabs [data-baseweb="tab"] {{
    flex: 1;
    background-color: white;
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    color: {TEXT_DARK};
    text-align: center;
}}

/* Active Tab Styling */
.stTabs [aria-selected="true"] {{
    background-color: {PRIMARY_COLOR} !important;
    border-color: {PRIMARY_COLOR} !important;
}}

/* Force Text White in Active Tab */
.stTabs [data-baseweb="tab"][aria-selected="true"] p, 
.stTabs [data-baseweb="tab"][aria-selected="true"] span {{
    color: #FFFFFF !important;
}}

/* ===== BUTTONS ===== */
.stButton > button {{
    background-color: {PRIMARY_COLOR};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 2rem;
    font-weight: 600;
}}
.stButton > button:hover {{
    background-color: {SECONDARY_COLOR};
}}

/* ===== RISK CARDS ===== */
.risk-status-card {{
    background: white;
    border: 1px solid {BORDER_COLOR};
    border-left-width: 4px;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}}
.risk-status-card.green {{ border-left-color: #28A745; }}
.risk-status-card.amber {{ border-left-color: #FFC107; }}
.risk-status-card.red {{ border-left-color: #DC3545; }}

/* ===== BREACH ITEMS ===== */
.breach-item {{
    border: 1px solid {ACCENT_COLOR};
    border-radius: 10px;
    padding: 10px 12px;
    margin: 8px 0;
    background: #fff;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}}

/* ===== TABS (FIXED TEXT COLOR) ===== */
.stTabs [data-baseweb="tab-list"] {{
    width: 100%;
    display: flex;
    gap: 8px;
    background-color: {BACKGROUND_LIGHT};
    padding: 0.5rem;
    border-radius: 8px;
}}

.stTabs [data-baseweb="tab"] {{
    flex: 1;
    background-color: white;
    border: 1px solid {ACCENT_COLOR};
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    color: {TEXT_DARK};
    text-align: center;
    white-space: nowrap;
}}

/* Selected Tab Container */
.stTabs [aria-selected="true"] {{
    background-color: {PRIMARY_COLOR} !important;
    color: white !important;
    border-color: {PRIMARY_COLOR} !important;
}}

/* CRITICAL FIX: Force inner text (p tags) to be white when selected */
.stTabs [data-baseweb="tab"][aria-selected="true"] p, 
.stTabs [data-baseweb="tab"][aria-selected="true"] span {{
    color: white !important;
}}

/* ===== BUTTONS ===== */
.stButton > button {{
    background-color: {PRIMARY_COLOR};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 2rem;
    font-weight: 600;
}}
.stButton > button:hover {{
    background-color: {SECONDARY_COLOR};
}}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {{ background-color: #f8f9fa; }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {PRIMARY_COLOR};
    font-size: 16px;
    font-weight: 600;
    border-bottom: 2px solid {SECONDARY_COLOR};
}}
</style>
""", unsafe_allow_html=True)

# ============================
# HEADER WITH LOGO
# ============================

# Locate Logo
search_dirs = [
    Path(__file__).resolve().parent,
    Path.cwd(),
]

logo_path = None
for d in search_dirs:
    p = d / "ere_logo.png"
    if p.exists():
        logo_path = str(p)
        break

def get_logo_html(path):
    """Generates the HTML image tag or a fallback text if missing."""
    if path and Path(path).exists():
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f'<img src="data:image/png;base64,{b64}">'
        except Exception:
            return ""
    return "<span style='color:#232D55; font-weight:bold;'>ERE</span>" # Text fallback

# Render Header
st.markdown(f"""
<div class="header-container">
    <div class="header-left">
        <h1>Portfolio Analytics Dashboard</h1>
        <p>{CLIENT_DISPLAY_NAME}</p>
    </div>
    <div class="header-right">
        <div class="header-date">
            <span class="date-label">Data as of</span>
            <span class="date-value">{datetime.now().strftime('%d %B %Y')}</span>
        </div>
        <div class="header-logo-box">
            {get_logo_html(logo_path)}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================
# 7. MAIN LOGIC (Sidebar & Loading)
# ============================

# --- DETERMINE DEFAULT PATH ---
# Priority: Session State > CLI > Last Run
default_path = ""

if "canonical_file_path" in st.session_state:
    default_path = st.session_state["canonical_file_path"]
elif CLI_FILE_PATH:
    default_path = CLI_FILE_PATH
    # CRITICAL FIX: Pre-seed session state so widget picks it up
    st.session_state["canonical_file_path"] = CLI_FILE_PATH
elif LAST_RUN_TYPED_PATH:
    default_path = LAST_RUN_TYPED_PATH

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Data source selector: Local file vs Azure Blob Storage
    source_options = ["Local file"]
    if BLOB_STORAGE_AVAILABLE:
        source_options.append("Azure Blob Storage")

    data_source = st.radio(
        "Data source",
        source_options,
        key="data_source_radio",
        horizontal=True,
    )

    input_path = ""
    selected_blob = ""

    if data_source == "Local file":
        # Input widget with value=default_path
        input_path = st.text_input(
            "Path to canonical CSV",
            value=default_path,
            key="canonical_file_path_widget",
        )
        # Sync widget back to variable
        if input_path:
            st.session_state["canonical_file_path"] = input_path
    else:
        # Azure Blob Storage browser
        st.markdown("##### Browse output files")
        blob_prefix = st.text_input(
            "Filter by prefix (e.g. tape/out/)",
            value="",
            key="blob_prefix_filter",
        )
        try:
            csv_blobs = list_canonical_csvs(prefix=blob_prefix)
            if csv_blobs:
                selected_blob = st.selectbox(
                    "Select a CSV file",
                    options=csv_blobs,
                    key="blob_file_selector",
                )
            else:
                st.info("No CSV files found in the outbound container.")
        except Exception as e:
            st.error(f"Could not list blobs: {e}")

# --- RESOLVE PATH & LOAD DATA ---
final_path_str = input_path if input_path else default_path
use_blob = data_source == "Azure Blob Storage" and selected_blob

if not final_path_str and not use_blob:
    if UPLOAD_PAGE_AVAILABLE:
        pass
    st.warning("👈 Please enter a file path or select a blob in the sidebar to proceed.")
    st.stop()

# --- LOAD DATA ---
try:
    if use_blob:
        with st.spinner(f"Loading from Azure: {selected_blob}..."):
            df = load_data_from_blob(selected_blob)
    else:
        validated_path = validate_file_path_pure(final_path_str)
        with st.spinner(f"Loading data from {validated_path.name}..."):
            df = load_data(str(validated_path))

    if df is None or df.empty:
        st.error("❌ Data load returned empty result.")
        st.stop()

    # Success
    source_label = selected_blob if use_blob else str(validated_path)
    st.success(f"✓ Loaded {len(df):,} loans from {Path(source_label).name}")

    # Save successful path for next time (local files only)
    if not use_blob:
        try:
            LATEST_TYPED_PATH_FILE.write_text(str(validated_path), encoding="utf-8")
        except OSError:
            pass

    # Create View Copy
    df_view = df.copy()

except ValueError as e:
    st.error(f"❌ {e}")
    st.stop()
except Exception as e:
    st.error("❌ Critical Error during load:")
    st.exception(e)
    st.stop()

# ============================
# 2. SIDEBAR FILTERS (Moved to Main Area)
# ============================

    # --- A. VINTAGE FILTER ---
    st.markdown("### 📅 Vintage year")
    if "origination_year" in df.columns:
        # Use df (original) to get the full list of options, not df_view
        vintages = sorted(df["origination_year"].dropna().unique())
        if vintages:
            sel_vintages = st.multiselect(
                "Select vintages (leave empty = all)",
                options=vintages,
                default=[],
                key="filter_vintages",
            )

            if sel_vintages:
                df_view = df_view[df_view["origination_year"].isin(sel_vintages)]

    # --- B. PRODUCT FILTER ---
    st.markdown("### 🏠 Product type")
    if "erm_product_type" in df.columns:
        # Use df (original) for options
        products = sorted(df["erm_product_type"].dropna().unique())
        if products:
            sel_products = st.multiselect(
                "Select products",
                options=products,
                default=[],
                key="filter_products",
            )
            if sel_products:
                # FIX: Filter df_view, do NOT overwrite df
                df_view = df_view[df_view["erm_product_type"].isin(sel_products)]

    # --- C. GEOGRAPHIC FILTER ---
    st.markdown("### 🗺️ Geography")
    if "geographic_region" in df.columns:
        # Use df (original) for options
        regions = sorted(df["geographic_region"].dropna().unique())
        if regions:
            sel_regions = st.multiselect(
                "Select regions (leave empty = all)",
                options=regions,
                default=[], 
                key="filter_regions",
            )
            if sel_regions:
                df_view = df_view[df_view["geographic_region"].isin(sel_regions)]

    # --- D. FILTER IMPACT & RESET ---
    # FIX: Compare df_view (Filtered) vs df (Original)
    if len(df_view) < len(df):
        reduction = (1 - len(df_view) / len(df)) * 100
        st.info(f"📊 **{len(df_view):,} of {len(df):,} loans** ({reduction:.1f}% filtered)")
        
        if st.button("🔄 Reset all filters"):
            for k in ["filter_vintages", "filter_products", "filter_regions"]:
                st.session_state.pop(k, None)
            st.rerun()
    else:
        st.info(f"📊 **{len(df):,} loans** (no filters)")

    df = df_view

    st.markdown("---")
    st.subheader("📥 Export")

    # CSV Export (filtered view)
    csv_bytes = df_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📄 Download CSV",
        data=csv_bytes,
        file_name=f"erm_portfolio_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Excel Export (filtered view)
    try:
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_view.to_excel(writer, index=False, sheet_name="Portfolio")
        excel_bytes = buffer.getvalue()

        st.download_button(
            "📊 Download Excel",
            data=excel_bytes,
            file_name=f"erm_portfolio_{datetime.now():%Y%m%d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except ImportError:
        pass

    st.markdown("---")
    st.subheader("📊 Generate Report")

    # PPT Generation Button
    if st.button("🎯 Generate PowerPoint", type="primary", use_container_width=True):
        with st.spinner("Generating presentation..."):
            try:
                # Save current filtered dataset to temp file
                temp_dir = Path("temp_pptx_data")
                temp_dir.mkdir(exist_ok=True)
                
                temp_csv = temp_dir / f"temp_data_{datetime.now():%Y%m%d_%H%M%S}.csv"
                df.to_csv(temp_csv, index=False)
                
                # Output filename
                output_pptx = Path("reports") / f"erm_report_{datetime.now():%Y%m%d_%H%M%S}.pptx"
                output_pptx.parent.mkdir(exist_ok=True)
                
                # Run PPTX generator
                result = subprocess.run(
                    [
                        sys.executable,
                        "generate_pptx_erm.py",
                        "--input", str(temp_csv),
                        "--output", str(output_pptx)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    # Success - provide download link
                    st.success("✅ Presentation generated successfully!")
                    
                    # Read the generated file
                    with open(output_pptx, "rb") as f:
                        pptx_bytes = f.read()
                    
                    # Download button
                    st.download_button(
                        label="📥 Download PowerPoint",
                        data=pptx_bytes,
                        file_name=output_pptx.name,
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True
                    )
                    
                    # Show stats
                    st.info(f"""
                    **Report Details:**
                    - 📊 {len(df):,} loans included
                    - 📄 File: {output_pptx.name}
                    - 📁 Saved to: reports/
                    """)
                    
                    # Clean up temp file
                    temp_csv.unlink(missing_ok=True)
                    
                else:
                    st.error(f"❌ Generation failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                st.error("❌ Generation timed out (took >5 minutes)")
            except Exception as e:
                st.error(f"❌ Error generating presentation: {e}")

    st.caption("💡 Tip: Apply filters before generating to customize your report")


# ============================
# MAIN TABS
# ============================

tab_names = [
    "📊 Stratifications",
    "🎯 Scenario Analysis",
    "📈 Static Pools"
]

if RISK_MONITORING_AVAILABLE:
    tab_names.append("🚦 Risk Monitoring")
    tab1, tab2, tab3, tab4 = st.tabs(tab_names)
else:
    tab1, tab2, tab3 = st.tabs(tab_names)


# ============================
# TAB 1: STRATIFICATIONS
# ============================

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ===== KPI STRIP =====
    st.markdown("### 📋 Portfolio Overview")
    st.markdown("#### Portfolio Metrics")  # NEW: Subheading for metrics
    
    total_loans = len(df)
    total_balance = df["total_balance"].sum()
    
    # Average and maximum loan size
    avg_loan_size = total_balance / total_loans if total_loans > 0 else 0.0
    max_loan_size = df["total_balance"].max() if "total_balance" in df.columns else 0.0
    
    # WA calculations (properly handling NaN values)
    if total_balance > 0:
        wa_current_ltv = mi_prep.weighted_average(df["current_loan_to_value"], df["total_balance"])
        if pd.isna(wa_current_ltv):
            wa_current_ltv = 0
            
        wa_rate = mi_prep.weighted_average(df["current_interest_rate"], df["total_balance"])
        if pd.isna(wa_rate):
            wa_rate = 0
        
        wa_age = mi_prep.weighted_average(df["youngest_borrower_age"], df["total_balance"])
        if pd.isna(wa_age):
            wa_age = 0
        
    else:
        wa_current_ltv = 0
        wa_rate = 0
        wa_age = 0
    
    # Original LTV calculation
    if "original_loan_to_value" in df.columns and total_balance > 0:
        wa_original_ltv = mi_prep.weighted_average(df["original_loan_to_value"], df["total_balance"])
    else:
        wa_original_ltv = None
    
# Display KPIs - First row (5 plates)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Total Loans</div>
            <div class="kpi-value">{total_loans:,}</div>
            <div class="kpi-subtitle">Portfolio count</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Portfolio Balance</div>
            <div class="kpi-value">{mi_prep.format_currency(total_balance)}</div>
            <div class="kpi-subtitle">Outstanding + accrued</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">WA Current LTV</div>
            <div class="kpi-value">{wa_current_ltv:.1f}%</div>
            <div class="kpi-subtitle">Indexed to current value</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">WA Interest Rate</div>
            <div class="kpi-value">{wa_rate:.2%}</div>
            <div class="kpi-subtitle">Balance-weighted</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">WA Borrower Age</div>
            <div class="kpi-value">{wa_age:.0f}</div>
            <div class="kpi-subtitle">Youngest borrower</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row (5 plates)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Calculate largest geographic exposure
    if "geographic_region" in df.columns and total_balance > 0:
        geo_balance = df.groupby("geographic_region")["total_balance"].sum()
        largest_geo = geo_balance.idxmax() if not geo_balance.empty else "N/A"
        largest_geo_pct = (geo_balance.max() / total_balance * 100) if not geo_balance.empty else 0
    else:
        largest_geo = "N/A"
        largest_geo_pct = 0
    
    # Calculate largest broker
    if "broker_channel" in df.columns and total_balance > 0:
        broker_balance = df.groupby("broker_channel")["total_balance"].sum()
        largest_broker = broker_balance.idxmax() if not broker_balance.empty else "N/A"
        largest_broker_pct = (broker_balance.max() / total_balance * 100) if not broker_balance.empty else 0
    else:
        largest_broker = "N/A"
        largest_broker_pct = 0
    
    col6, col7, col8, col9, col10 = st.columns(5)
    
    with col6:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">WA Original LTV</div>
            <div class="kpi-value">{fmt_float(wa_original_ltv, 1, "%")}</div>
            <div class="kpi-subtitle">At origination</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col7:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Avg. Loan Size</div>
            <div class="kpi-value">{mi_prep.format_currency(avg_loan_size)}</div>
            <div class="kpi-subtitle">Balance-weighted average</div>
        </div>
        """, unsafe_allow_html=True)

    with col8:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Largest Loan</div>
            <div class="kpi-value">{mi_prep.format_currency(max_loan_size)}</div>
            <div class="kpi-subtitle">Maximum single exposure</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col9:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Largest Geographic</div>
            <div class="kpi-value">{fmt_float(largest_geo_pct, 1, "%")}</div>
            <div class="kpi-subtitle">{largest_geo}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col10:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Largest Broker</div>
            <div class="kpi-value">{fmt_float(largest_broker_pct, 1, "%")}</div>
            <div class="kpi-subtitle">{largest_broker}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== STRATIFICATION CHARTS =====
    st.markdown("### 📊 Portfolio Stratifications")
    
    # LTV Distribution
    st.markdown("#### Current LTV Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = strat_bar_chart(df, "ltv_bucket", "total_balance", "sum", "Balance by LTV Bucket")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = strat_bar_chart(df, "ltv_bucket", "loan_id", "count", "Loan Count by LTV Bucket")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    
    # Product Type
    st.markdown("#### Product Type Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = strat_bar_chart(df, "erm_product_type", "total_balance", "sum", "Balance by Product Type")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = strat_bar_chart(df, "erm_product_type", "loan_id", "count", "Loan Count by Product Type")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    
    # Geographic Region
    st.markdown("#### Geographic Distribution")
    col1, col2 = st.columns(2)
    
    # Top 10 regions by balance
    geo_group = df.groupby("geographic_region", observed=True)["total_balance"].sum().reset_index()
    geo_group = geo_group.sort_values("total_balance", ascending=False)
    top_geo = geo_group.head(10)
     
    with col1:
        if not top_geo.empty:
            fig_geo_bal = px.treemap(
                top_geo,
                path=[px.Constant("All Regions"), "geographic_region"],
                values="total_balance",
                color="total_balance",
                # Gradient: Light Grey (#F0F2F6) -> Navy (PRIMARY_COLOR)
                color_continuous_scale=[(0, "#F0F2F6"), (1, PRIMARY_COLOR)],
            )
            fig_geo_bal.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            fig_geo_bal.update_layout(title_text="Balance by Region (Top 10)", title_x=0)
            
            fig_geo_bal.update_traces(
                textinfo="label+value+percent entry",
                texttemplate="<b>%{label}</b><br>£%{value:,.0f}<br>(%{percentEntry:.1%})",
                # FIX: Remove 'textfont' entirely to let Plotly auto-contrast (Black on Light, White on Dark)
            )
            st.plotly_chart(fig_geo_bal, use_container_width=True)
        else:
            st.info("No geographic data available")

    with col2:
        geo_count = df.groupby("geographic_region", observed=True).size().reset_index(name="loan_count")
        geo_count = geo_count.sort_values("loan_count", ascending=False)
        top_geo_count = geo_count.head(10)

        if not top_geo_count.empty:
            fig_geo_cnt = px.treemap(
                top_geo_count,
                path=[px.Constant("All Regions"), "geographic_region"],
                values="loan_count",
                color="loan_count",
                color_continuous_scale=[(0, "#F0F2F6"), (1, PRIMARY_COLOR)],
            )
            fig_geo_cnt.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            fig_geo_cnt.update_layout(title_text="Loan Count by Region (Top 10)", title_x=0)
            
            fig_geo_cnt.update_traces(
                textinfo="label+value",
                texttemplate="<b>%{label}</b><br>%{value:,} loans<br>(%{percentEntry:.1%})",
                # FIX: Remove 'textfont' entirely
            )
            st.plotly_chart(fig_geo_cnt, use_container_width=True)
        else:
            st.info("No geographic data available")

    # Borrower Age
    st.markdown("#### Borrower Age Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = strat_bar_chart(df, "age_bucket", "total_balance", "sum", "Balance by Age Bucket")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = strat_bar_chart(df, "age_bucket", "loan_id", "count", "Loan Count by Age Bucket")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
      
    # Origination Month (derive from origination_date, do not trust origination_month)
    # --- DYNAMIC VINTAGE CHART ---
    st.markdown("#### Vintage Distribution")
    
    # 1. Granularity Controls
    # We use columns to keep the radio button compact on the left
    v_ctl, _ = st.columns([1, 3])
    with v_ctl:
        granularity = st.radio(
            "Group By:", 
            ["Month", "Quarter", "Year"], 
            horizontal=True, 
            index=1, # Default to Quarter for seasoned books
            key="vintage_granularity"
        )

    # 2. Data Processing & Aggregation
    if "origination_date" not in df_view.columns:
        st.info("Origination date not available for vintage charts")
    else:
        # Filter for valid dates
        tmp = df_view.copy().dropna(subset=["origination_date"])
        
        if tmp.empty:
            st.info("No valid origination dates found.")
        else:
            # Dynamic Resampling based on selection
            if granularity == "Month":
                tmp["cohort"] = tmp["origination_date"].dt.to_period("M")
                fmt = "%b-%y"   # e.g., Jan-25
            elif granularity == "Quarter":
                tmp["cohort"] = tmp["origination_date"].dt.to_period("Q")
                fmt = "Q%q %Y"  # e.g., Q1 2025
            else:
                tmp["cohort"] = tmp["origination_date"].dt.to_period("Y")
                fmt = "%Y"      # e.g., 2025

            # Aggregate Balance
            v_bal = (
                tmp.groupby("cohort", observed=True)["total_balance"]
                .sum()
                .reset_index()
                .sort_values("cohort")
            )
            
            # Aggregate Count
            v_cnt = (
                tmp.groupby("cohort", observed=True)
                .size()
                .reset_index(name="loan_count")
                .sort_values("cohort")
            )

            # Create Display Labels (Convert Period -> Timestamp -> String)
            # This ensures Plotly treats the axis as Categories, preventing timeline squashing
            v_bal["label"] = v_bal["cohort"].dt.to_timestamp().dt.strftime(fmt)
            v_cnt["label"] = v_cnt["cohort"].dt.to_timestamp().dt.strftime(fmt)

            # 3. Plotting
            col1, col2 = st.columns(2)

            # LEFT: Balance Chart
            with col1:
                fig_v_bal = go.Figure()
                fig_v_bal.add_trace(go.Bar(
                    x=v_bal["label"],
                    y=v_bal["total_balance"],
                    marker_color=PRIMARY_COLOR,
                    text=v_bal["total_balance"].apply(format_currency), # Preserves your currency formatting
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Balance: %{text}<extra></extra>"
                ))
                fig_v_bal = apply_chart_theme(fig_v_bal, f"Balance by Vintage ({granularity})")
                fig_v_bal.update_xaxes(title_text="Cohort", type="category")
                fig_v_bal.update_yaxes(title_text="Outstanding Balance (£)")
                fig_v_bal.update_layout(showlegend=False)
                st.plotly_chart(fig_v_bal, use_container_width=True)

            # RIGHT: Count Chart
            with col2:
                fig_v_cnt = go.Figure()
                fig_v_cnt.add_trace(go.Bar(
                    x=v_cnt["label"],
                    y=v_cnt["loan_count"],
                    marker_color=PRIMARY_COLOR,
                    text=v_cnt["loan_count"],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Loans: %{text}<extra></extra>"
                ))
                fig_v_cnt = apply_chart_theme(fig_v_cnt, f"Loan Count by Vintage ({granularity})")
                fig_v_cnt.update_xaxes(title_text="Cohort", type="category")
                fig_v_cnt.update_yaxes(title_text="Loan Count")
                fig_v_cnt.update_layout(showlegend=False)
                st.plotly_chart(fig_v_cnt, use_container_width=True)
        
        # Ticket Size Distribution
        st.markdown("#### Balance by Ticket Size")
        col1, col2 = st.columns(2)

        df["total_balance"] = pd.to_numeric(df["total_balance"], errors="coerce")

        bucket_order = ['<£75K', '£75K-£100K', '£100K-£175K', '£175K-£250K', '>£250K', 'Unknown']

        df["ticket_bucket"] = pd.cut(
            df["total_balance"],
            bins=[0, 75000, 100000, 175000, 250000, float("inf")],
            labels=bucket_order[:-1],  # all except Unknown
            include_lowest=True
        ).astype("string").fillna("Unknown")

        ticket_bal = (
            df.groupby("ticket_bucket", observed=True)["total_balance"]
            .sum()
            .reset_index()
        )
        ticket_bal = ticket_bal[ticket_bal["total_balance"] > 0]

        ticket_cnt = (
            df.groupby("ticket_bucket", observed=True)
            .size()
            .reset_index(name="loan_count")
        )
        ticket_cnt = ticket_cnt[ticket_cnt["loan_count"] > 0]

        # Define bucket order
        bucket_order = ['<£75K', '£75K-£100K', '£100K-£175K', '£175K-£250K', '>£250K']

        with col1:
            if not ticket_bal.empty:
                # Ensure buckets are in correct order
                ticket_bal["ticket_bucket"] = pd.Categorical(
                    ticket_bal["ticket_bucket"],
                    categories=bucket_order,
                    ordered=True
                )
                ticket_bal = ticket_bal.sort_values("ticket_bucket")

                fig_ticket_bal = go.Figure()
                fig_ticket_bal.add_trace(go.Bar(
                    x=ticket_bal["ticket_bucket"].astype(str),
                    y=ticket_bal["total_balance"],
                    marker_color=PRIMARY_COLOR,
                    text=ticket_bal["total_balance"].apply(format_currency),
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Balance: %{text}<extra></extra>"
                ))

                fig_ticket_bal = apply_chart_theme(fig_ticket_bal, "Balance by Ticket Size")
                fig_ticket_bal.update_xaxes(
                    title_text="Ticket Size",
                    type="category",
                    categoryorder="array",
                    categoryarray=bucket_order
                )
                fig_ticket_bal.update_yaxes(title_text="Outstanding Balance (£)")
                fig_ticket_bal.update_layout(showlegend=False)
                st.plotly_chart(fig_ticket_bal, use_container_width=True)
            else:
                st.info("No ticket size data available")

        with col2:
            if not ticket_cnt.empty:
                # Ensure buckets are in correct order
                ticket_cnt["ticket_bucket"] = pd.Categorical(
                    ticket_cnt["ticket_bucket"],
                    categories=bucket_order,
                    ordered=True
                )
                ticket_cnt = ticket_cnt.sort_values("ticket_bucket")

                fig_ticket_cnt = go.Figure()
                fig_ticket_cnt.add_trace(go.Bar(
                    x=ticket_cnt["ticket_bucket"].astype(str),
                    y=ticket_cnt["loan_count"],
                    marker_color=PRIMARY_COLOR,
                    text=ticket_cnt["loan_count"],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Loans: %{text}<extra></extra>"
                ))

                fig_ticket_cnt = apply_chart_theme(fig_ticket_cnt, "Loan Count by Ticket Size")
                fig_ticket_cnt.update_xaxes(
                    title_text="Ticket Size",
                    type="category",
                    categoryorder="array",
                    categoryarray=bucket_order
                )
                fig_ticket_cnt.update_yaxes(title_text="Loan Count")
                fig_ticket_cnt.update_layout(showlegend=False)
                st.plotly_chart(fig_ticket_cnt, use_container_width=True)
            else:
                st.info("No ticket size data available")
        
# Broker Channel
    st.markdown("#### Broker Channel Distribution")
    col1, col2 = st.columns(2)
    
    broker_bal = df.groupby("broker_channel", observed=True)["total_balance"].sum().reset_index()
    broker_bal = broker_bal.sort_values("total_balance", ascending=False).head(15)

    with col1:
        if not broker_bal.empty:
            fig_broker_bal = px.treemap(
                broker_bal,
                path=[px.Constant("All Brokers"), "broker_channel"],
                values="total_balance",
                color="total_balance",
                color_continuous_scale=[(0, "#F0F2F6"), (1, PRIMARY_COLOR)],
            )
            fig_broker_bal.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            fig_broker_bal.update_layout(title_text="Balance by Broker (Top 15)", title_x=0)
            
            fig_broker_bal.update_traces(
                textinfo="label+value+percent entry",
                texttemplate="<b>%{label}</b><br>£%{value:,.0f}<br>(%{percentEntry:.1%})",
                # FIX: Remove 'textfont' entirely
            )
            st.plotly_chart(fig_broker_bal, use_container_width=True)
        else:
            st.info("No broker data available")

    with col2:
        broker_cnt = df.groupby("broker_channel", observed=True).size().reset_index(name="loan_count")
        broker_cnt = broker_cnt.sort_values("loan_count", ascending=False).head(15)

        if not broker_cnt.empty:
            fig_broker_cnt = px.treemap(
                broker_cnt,
                path=[px.Constant("All Brokers"), "broker_channel"],
                values="loan_count",
                color="loan_count",
                color_continuous_scale=[(0, "#F0F2F6"), (1, PRIMARY_COLOR)],
            )
            fig_broker_cnt.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            fig_broker_cnt.update_layout(title_text="Loan Count by Broker (Top 15)", title_x=0)
            
            fig_broker_cnt.update_traces(
                textinfo="label+value+percent entry",
                texttemplate="<b>%{label}</b><br>%{value:,} loans<br>(%{percentEntry:.1%})",
                # FIX: Remove 'textfont' entirely
            )
            st.plotly_chart(fig_broker_cnt, use_container_width=True)
        else:
            st.info("No broker data available")
        
    # ===== BUBBLE CHARTS =====
    st.markdown("### 🫧 Relationship Analysis")
    
    # Bubble 1: Outstanding Balance vs Property Value
    st.markdown("#### Balance vs Current Property Value")
    
    # 1. Drop NA 
    bubble_df = df.dropna(subset=["current_valuation_amount", "current_outstanding_balance", "total_balance"]).copy()
    
    # 2. Filter out negative/zero balances
    bubble_df = bubble_df[bubble_df["total_balance"] > 0]
    
    if not bubble_df.empty:
        # Sample for performance if large
        if len(bubble_df) > 1000:
            bubble_df = bubble_df.sample(1000, random_state=42)
            st.caption("📊 Showing 1,000 random loans for performance")
        
        if "loan_identifier" in bubble_df.columns and "loan_id" not in bubble_df.columns:
            bubble_df["loan_id"] = bubble_df["loan_identifier"]

        fig = px.scatter(
            bubble_df,
            x="current_valuation_amount",
            y="current_outstanding_balance",
            size="total_balance",
            color="geographic_region" if "geographic_region" in bubble_df.columns else None,
            hover_data={
                "loan_id": True,
                "current_valuation_amount": ":,.0f",
                "current_outstanding_balance": ":,.0f",
                "total_balance": ":,.0f",
                "current_loan_to_value": ":.1%",
            },
            # FIXED: Use your Config-driven Chart Colors (Navy, Light Blue, etc.)
            color_discrete_sequence=CHART_COLORS,
        )

        fig = apply_chart_theme(fig, "Outstanding Balance vs Current Property Value")
        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color="white")))
        fig.update_xaxes(title_text="Current Property Value (£)")
        fig.update_yaxes(title_text="Outstanding Balance (£)")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid valuation or balance data available for this chart.")
    
    # Bubble 2: LTV vs Borrower Age
    st.markdown("#### LTV vs Youngest Borrower Age")
    
    bubble_df2 = df.dropna(subset=["youngest_borrower_age", "current_loan_to_value", "total_balance"]).copy()
    
    # LTV is already normalized to 0-1, convert to percentage for display
    if not bubble_df2.empty:
        bubble_df2["current_loan_to_value_pct"] = bubble_df2["current_loan_to_value"]
    
    if not bubble_df2.empty:
        if len(bubble_df2) > 2000:
            bubble_df2 = bubble_df2.sample(2000, random_state=42)
            st.caption("📊 Showing 2,000 random loans for performance")
        
        fig = px.scatter(
            bubble_df2,
            x="youngest_borrower_age",
            y="current_loan_to_value_pct",
            size="total_balance",
            color="erm_product_type" if "erm_product_type" in bubble_df2.columns else None,
            hover_data={
                "loan_id": True,
                "youngest_borrower_age": ":.0f",
                "current_loan_to_value_pct": ":.1f",
                "total_balance": ":,.0f",
            },
            # FIXED: Use your Config-driven Chart Colors
            color_discrete_sequence=CHART_COLORS,
        )
        
        fig = apply_chart_theme(fig, "LTV vs Youngest Borrower Age")
        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color="white")))
        fig.update_xaxes(title_text="Youngest Borrower Age (years)")
        fig.update_yaxes(title_text="Current LTV (%)")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient data for bubble chart 2")

# ... (End of Bubble Charts section) ...

    st.markdown("---")
    st.markdown("### 🔎 Portfolio Concentration Matrix")
    st.caption("Deep dive into the intersection of any two risk dimensions.")
    
    # 1. Configuration Controls
    col_x, col_y, col_m = st.columns(3)
    
    # Define friendly names for columns
    dim_map = {
        "Region": "geographic_region",
        "Product Type": "erm_product_type",
        "LTV Bucket": "ltv_bucket",
        "Age Bucket": "age_bucket",
        "Ticket Size": "ticket_bucket",
        "Vintage": "origination_year",
        "Broker Channel": "broker_channel"
    }
    
    # Filter to only available columns in current view
    avail = [k for k, v in dim_map.items() if v in df_view.columns]

    with col_x:
        # Default to Region (index 0) if available
        r_idx = 0 if avail else None
        row_label = st.selectbox("Rows (Y-Axis)", avail, index=r_idx)
        
    with col_y:
        # Default to Product (index 1) or LTV (index 2) if available
        c_idx = 2 if len(avail) > 2 else 0
        col_label = st.selectbox("Columns (X-Axis)", avail, index=c_idx)
        
    with col_m:
        metric_choice = st.radio("Metric", options=["Balance", "Count"], horizontal=True)

    # 2. Logic & Plotting
    if row_label and col_label:
        row_col = dim_map[row_label]
        col_col = dim_map[col_label]
        
        if metric_choice == "Balance":
            # Sum Balance
            mat = df_view.groupby([row_col, col_col], observed=True)["total_balance"].sum().unstack(fill_value=0)
            # Text matrix for display (£X.X M)
            txt = mat.applymap(lambda x: f"£{x/1_000_000:.1f}M" if x > 0 else "")
            hovertemplate = "%{y}, %{x}<br><b>£%{z:,.0f}</b><extra></extra>"
        else:
            # Count Loans
            mat = df_view.groupby([row_col, col_col], observed=True).size().unstack(fill_value=0)
            # Text matrix for display (Integer)
            txt = mat.applymap(lambda x: f"{int(x):,}" if x > 0 else "")
            hovertemplate = "%{y}, %{x}<br><b>%{z:,} Loans</b><extra></extra>"
        
        # 3. Create Heatmap
        fig_mx = go.Figure(data=go.Heatmap(
            z=mat.values,
            x=[str(c).replace("_", " ").title() for c in mat.columns],
            y=[str(i).replace("_", " ").title() for i in mat.index],
            text=txt.values,
            texttemplate="%{text}",
            textfont={"size": 11},
            # Consistent Gradient: Light Grey -> Primary Navy
            colorscale=[[0, "#F0F2F6"], [1, PRIMARY_COLOR]],
            showscale=True,
            hovertemplate=hovertemplate,
            xgap=2,
            ygap=2,
        ))
        
        # 4. Strict Formatting
        fig_mx.update_layout(
            title=dict(
                text=f"<b>{metric_choice}</b>: {row_label} vs {col_label}",
                font=dict(size=16, color=PRIMARY_COLOR)
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=500,
            xaxis=dict(
                title=col_label,
                side="bottom",
                showline=True,
                linewidth=2,
                linecolor="#E0E0E0",
                mirror=True,
                tickfont=dict(size=11),
                title_font=dict(size=13, color="#555")
            ),
            yaxis=dict(
                title=row_label,
                showline=True,
                linewidth=2,
                linecolor="#E0E0E0",
                mirror=True,
                tickfont=dict(size=11),
                title_font=dict(size=13, color="#555"),
                autorange="reversed" # Read top-to-bottom
            ),
        )
        
        # Hide color bar to reduce clutter
        fig_mx.update_traces(colorbar=dict(
            title="", 
            thickness=15, 
            len=0.5,
            tickfont=dict(size=10)
        ))

        st.plotly_chart(fig_mx, use_container_width=True)

# ============================
# TAB 2: SCENARIO ANALYSIS
# ============================

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Scenario Analysis")

    if not SCENARIO_ENGINE_AVAILABLE:
        st.warning("Scenario Analysis is unavailable — the `scenario_engine` module was not found.")
        st.stop()

    # Check if we have required columns
    required_cols = [
        "current_principal_balance",
        "current_valuation_amount",
        "current_interest_rate",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        st.error(
            "❌ Missing required columns for scenario analysis: "
            + ", ".join(missing_cols)
        )
    else:
        # ==================== SCENARIO CONFIGURATION ====================
        st.markdown("#### ⚙️ Scenario Configuration")
        
        col_config1, col_config2 = st.columns([2, 1])
        
        with col_config1:
            preset_names = ["Custom"] + list(PRESET_SCENARIOS.keys())
            selected_preset = st.selectbox(
                "Preset scenario",
                options=preset_names,
                help="Choose a preset or select 'Custom' to define your own assumptions.",
            )
            
            if selected_preset != "Custom":
                preset_assumptions = PRESET_SCENARIOS[selected_preset]
                st.caption(f"Using preset: **{selected_preset}**")
            else:
                preset_assumptions = None
        
        with col_config2:
            projection_years = st.slider(
                "Projection horizon (years)",
                min_value=5,
                max_value=50,
                value=25,
                step=5,
                help="Number of years to project forward.",
            )
        
        st.markdown("---")
        
        # ==================== ASSUMPTION INPUTS ====================
        st.markdown("#### 📊 Scenario Assumptions")
        
        # Defaults from preset or sensible base case
        if preset_assumptions:
            default_hpi = preset_assumptions.hpi_rate
            default_spread = preset_assumptions.interest_rate_spread
            default_prepay = preset_assumptions.voluntary_prepay_rate
            default_mortality = preset_assumptions.mortality_rate
            default_care = preset_assumptions.move_to_care_rate
            default_sale_cost = preset_assumptions.sale_cost_pct
        else:
            default_hpi = 0.02
            default_spread = 0.0
            default_prepay = 0.02
            default_mortality = 0.03
            default_care = 0.01
            default_sale_cost = 0.05
        
        col_a1, col_a2, col_a3 = st.columns(3)
        
        with col_a1:
            st.markdown("**🏠 Economic assumptions**")
            
            hpi_rate = (
                st.number_input(
                    "House price growth (annual, %)",
                    min_value=-50.0,
                    max_value=50.0,
                    value=default_hpi * 100,
                    step=1.0,
                    format="%.1f",
                )
                / 100.0
            )
            
            interest_spread = (
                st.number_input(
                    "Interest rate shock (pp)",
                    min_value=-10.0,
                    max_value=10.0,
                    value=default_spread * 100,
                    step=0.5,
                    format="%.1f",
                    help="Parallel shift (in percentage points) applied to all loan rates.",
                )
                / 100.0
            )
            
            sale_cost_pct = (
                st.number_input(
                    "Property sale costs (%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=default_sale_cost * 100,
                    step=0.5,
                    format="%.1f",
                )
                / 100.0
            )
        
        with col_a2:
            st.markdown("**👤 Borrower behaviour**")
            
            voluntary_prepay = (
                st.number_input(
                    "Voluntary prepayment (annual, %)",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_prepay * 100,
                    step=1.0,
                    format="%.1f",
                )
                / 100.0
            )
            
            mortality_rate = (
                st.number_input(
                    "Mortality rate (annual, %)",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_mortality * 100,
                    step=1.0,
                    format="%.1f",
                )
                / 100.0
            )
            
            move_to_care_rate = (
                st.number_input(
                    "Move to care (annual, %)",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_care * 100,
                    step=0.5,
                    format="%.1f",
                )
                / 100.0
            )
        
        with col_a3:
            st.markdown("**📈 Combined metrics**")
            
            # Combined annual exit rate
            p_exit = 1.0 - (
                (1.0 - voluntary_prepay)
                * (1.0 - mortality_rate)
                * (1.0 - move_to_care_rate)
            )
            p_exit = max(0.0, min(1.0, p_exit))
            
            st.metric(
                "Total exit rate (annual)",
                f"{p_exit:.2%}",
                help="Combined probability of any exit event (prepay, death, move to care).",
            )
            
            expected_life = 1.0 / p_exit if p_exit > 0 else float("inf")
            st.metric(
                "Expected remaining life",
                f"{expected_life:.1f} years" if expected_life < 100 else "100+ years",
                help="Average time until loan exits portfolio under these assumptions.",
            )
            
            # Effective WA rate after spread
            total_balance_for_rate = df["current_principal_balance"].sum()
            if total_balance_for_rate > 0:
                current_wa_rate = (
                    (df["current_interest_rate"] * df["current_principal_balance"]).sum()
                    / total_balance_for_rate
                )
            else:
                current_wa_rate = 0.0
            
            effective_rate = current_wa_rate + interest_spread
            st.metric(
                "Effective WA rate",
                f"{effective_rate:.2%}",
                delta=f"{interest_spread:+.2%}" if abs(interest_spread) > 0.0005 else None,
                help="Weighted average coupon after interest rate shock.",
            )
        
        st.markdown("---")
        
        # ==================== RUN SCENARIO ====================
        col_run1, col_run2, col_run3 = st.columns([2, 1, 1])
        
        with col_run1:
            run_button = st.button(
                "🚀 Run scenario",
                type="primary",
                use_container_width=True,
            )
        
        with col_run2:
            multi_scenario = st.checkbox(
                "Compare with house price stress",
                value=False,
                help="Run a simple comparison vs a -5% HPI stress.",
            )
        
        with col_run3:
            show_loan_detail = st.checkbox(
                "Include loan-level drill-down",
                value=False,
                help="Add summary loan-level projections for export.",
            )
        
        if run_button:
            with st.spinner("Projecting ERM portfolio..."):
                try:
                    assumptions = ScenarioAssumptions(
                        hpi_rate=hpi_rate,
                        interest_rate_spread=interest_spread,
                        voluntary_prepay_rate=voluntary_prepay,
                        mortality_rate=mortality_rate,
                        move_to_care_rate=move_to_care_rate,
                        sale_cost_pct=sale_cost_pct,
                        n_years=projection_years,
                    )
                    
                    if show_loan_detail:
                        projection, loan_detail = project_portfolio(
                            df,
                            assumptions,
                            return_loan_level=True,
                        )
                    else:
                        projection = project_portfolio(
                            df,
                            assumptions,
                            return_loan_level=False,
                        )
                        loan_detail = None
                    
                    st.session_state["scenario_projection"] = projection
                    st.session_state["scenario_assumptions"] = assumptions
                    st.session_state["scenario_loan_detail"] = loan_detail
                    st.session_state["scenario_comparison"] = None
                    
                    if multi_scenario:
                        stress_assumptions = ScenarioAssumptions(
                            hpi_rate=-0.05,  # 5% annual decline
                            interest_rate_spread=interest_spread,
                            voluntary_prepay_rate=voluntary_prepay * 1.5,
                            mortality_rate=mortality_rate,
                            move_to_care_rate=move_to_care_rate,
                            sale_cost_pct=sale_cost_pct,
                            n_years=projection_years,
                        )
                        scenarios = {
                            "Base" if selected_preset == "Custom" else selected_preset: assumptions,
                            "House Price Stress": stress_assumptions,
                        }
                        comparison = compare_scenarios(df, scenarios)
                        st.session_state["scenario_comparison"] = comparison
                    
                    st.success("✅ Scenario projection complete.")
                
                except Exception as e:
                    st.error(f"❌ Scenario projection failed: {e}")
        
        # ==================== DISPLAY RESULTS ====================
        if "scenario_projection" in st.session_state:
            projection = st.session_state["scenario_projection"]
            assumptions = st.session_state["scenario_assumptions"]
            loan_detail = st.session_state.get("scenario_loan_detail")
            comparison = st.session_state.get("scenario_comparison")

            st.markdown("---")
            st.markdown("#### 🎯 Key Projections")

            # Determine key years
            max_year = int(projection["year"].max())
            comparison_year = min(10, max_year)
            year_25 = min(25, max_year)

            # Current balance (Year 0)
            current_balance = projection.loc[
                projection["year"] == 0, "portfolio_balance"
            ].iloc[0]

            # Balance in comparison year (e.g. Year 10)
            future_balance = projection.loc[
                projection["year"] == comparison_year, "portfolio_balance"
            ].iloc[0]
            balance_change_pct = (
                (future_balance / current_balance - 1) * 100
                if current_balance > 0
                else 0.0
            )

            # Remaining balance ratio in comparison year
            remaining_ratio = projection.loc[
                projection["year"] == comparison_year, "remaining_balance_ratio"
            ].iloc[0]

            # 25Y cumulative NNEG (or final year if <25)
            nneg_25 = projection.loc[
                projection["year"] == year_25, "cumulative_expected_nneg"
            ].iloc[0]

            # Portfolio LTV drift to comparison year
            future_ltv = projection.loc[
                projection["year"] == comparison_year, "portfolio_ltv"
            ].iloc[0]
            current_ltv = projection.loc[
                projection["year"] == 0, "portfolio_ltv"
            ].iloc[0]
            ltv_change = future_ltv - current_ltv

            # Final balance (last projection year)
            final_row = projection.loc[projection["year"] == max_year].iloc[0]
            final_balance = final_row["portfolio_balance"]
            final_ratio = (
                final_balance / current_balance if current_balance > 0 else 0.0
            )

            col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)

            with col_k1:
                st.markdown(
                    f"""
                    <div class="kpi-box">
                        <div class="kpi-label">Balance in Year {comparison_year}</div>
                        <div class="kpi-value">{mi_prep.format_currency(future_balance)}</div>
                        <div class="kpi-subtitle" style="color: {'#FF6B35' if balance_change_pct > 0 else '#449C95'};">
                            {balance_change_pct:+.1f}% vs today
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_k2:
                st.markdown(
                    f"""
                    <div class="kpi-box">
                        <div class="kpi-label">Remaining balance</div>
                        <div class="kpi-value">{remaining_ratio:.1%}</div>
                        <div class="kpi-subtitle">
                            Of original principal (Year {comparison_year})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_k3:
                st.markdown(
                    f"""
                    <div class="kpi-box">
                        <div class="kpi-label">25Y cumulative NNEG</div>
                        <div class="kpi-value">{mi_prep.format_currency(nneg_25)}</div>
                        <div class="kpi-subtitle">
                            Expected NNEG to Year {year_25}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_k4:
                st.markdown(
                    f"""
                    <div class="kpi-box">
                        <div class="kpi-label">Portfolio LTV Year {comparison_year}</div>
                        <div class="kpi-value">{future_ltv:.1f}%</div>
                        <div class="kpi-subtitle" style="color: {'#FF6B35' if ltv_change > 0 else '#449C95'};">
                            {ltv_change:+.1f}pp drift
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_k5:
                st.markdown(
                    f"""
                    <div class="kpi-box">
                        <div class="kpi-label">Final balance (Year {max_year})</div>
                        <div class="kpi-value">{mi_prep.format_currency(final_balance)}</div>
                        <div class="kpi-subtitle">
                            {final_ratio:.1%} of current
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ==================== CHARTS ====================
            st.markdown("#### 📈 Scenario charts")
            ctab1, ctab2, ctab3 = st.tabs(
                ["Balance runoff", "NNEG losses", "LTV over time"]
            )

            with ctab1:
                fig_bal = go.Figure()

                if comparison is not None:
                    for scen in comparison["scenario_name"].unique():
                        data = comparison[comparison["scenario_name"] == scen]
                        fig_bal.add_trace(
                            go.Scatter(
                                x=data["year"],
                                y=data["portfolio_balance"],
                                mode="lines",
                                name=scen,
                                line=dict(width=3),
                            )
                        )
                else:
                    fig_bal.add_trace(
                        go.Scatter(
                            x=projection["year"],
                            y=projection["portfolio_balance"],
                            mode="lines",
                            name="Portfolio balance",
                            line=dict(color=PRIMARY_COLOR, width=3),
                        )
                    )

                fig_bal = apply_chart_theme(fig_bal, "Portfolio balance projection")
                fig_bal.update_xaxes(title_text="Year")
                fig_bal.update_yaxes(title_text="Outstanding balance")
                st.plotly_chart(fig_bal, use_container_width=True)

            with ctab2:
                fig_nneg = go.Figure()

                fig_nneg.add_trace(
                    go.Bar(
                        x=projection["year"],
                        y=projection["expected_nneg_loss"],
                        name="Annual NNEG loss",
                        opacity=0.7,
                    )
                )

                fig_nneg.add_trace(
                    go.Scatter(
                        x=projection["year"],
                        y=projection["cumulative_expected_nneg"],
                        name="Cumulative NNEG",
                        mode="lines+markers",
                        yaxis="y2",
                        line=dict(color=ACCENT_COLOR, width=3),
                    )
                )

                fig_nneg.update_layout(
                    yaxis=dict(title="Annual NNEG loss"),
                    yaxis2=dict(
                        title="Cumulative NNEG",
                        overlaying="y",
                        side="right",
                    ),
                    hovermode="x unified",
                )

                fig_nneg = apply_chart_theme(
                    fig_nneg, "Expected NNEG losses over time"
                )
                fig_nneg.update_xaxes(title_text="Year")
                st.plotly_chart(fig_nneg, use_container_width=True)

            with ctab3:
                fig_ltv = go.Figure()

                if comparison is not None:
                    for scen in comparison["scenario_name"].unique():
                        data = comparison[comparison["scenario_name"] == scen]
                        fig_ltv.add_trace(
                            go.Scatter(
                                x=data["year"],
                                y=data["portfolio_ltv"],
                                mode="lines+markers",
                                name=scen,
                                line=dict(width=3),
                            )
                        )
                else:
                    fig_ltv.add_trace(
                        go.Scatter(
                            x=projection["year"],
                            y=projection["portfolio_ltv"],
                            mode="lines+markers",
                            name="Portfolio LTV",
                            line=dict(color=PRIMARY_COLOR, width=3),
                        )
                    )

                fig_ltv = apply_chart_theme(fig_ltv, "Portfolio LTV projection")
                fig_ltv.update_xaxes(title_text="Year")
                fig_ltv.update_yaxes(title_text="Portfolio LTV (%)")
                st.plotly_chart(fig_ltv, use_container_width=True)

            # ==================== LOAN-LEVEL DETAIL (OPTIONAL) ====================
            if loan_detail is not None:
                st.markdown("---")
                st.markdown("#### 🔍 Loan-level drill-down")
                st.markdown("### View loan-level summary projections")
                display_detail = loan_detail.copy()
                for col in display_detail.columns:
                    if "balance" in col or "property" in col:
                        display_detail[col] = display_detail[col].apply(
                            format_currency
                        )
                    elif "ltv" in col:
                        display_detail[col] = display_detail[col].apply(
                            lambda x: f"{x:.1f}%"
                        )

                    st.dataframe(
                        display_detail,
                        use_container_width=True,
                        height=350,
                    )

                    csv_ld = loan_detail.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download loan-level CSV",
                        data=csv_ld,
                        file_name=f"erm_scenario_loan_detail_{datetime.now():%Y%m%d}.csv",
                        mime="text/csv",
                    )

            # ==================== EXPORT ====================
            st.markdown("---")
            st.markdown("#### 📥 Export scenario results")

            col_e1, col_e2, col_e3 = st.columns(3)

            with col_e1:
                csv_proj = projection.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📊 Download projection CSV",
                    data=csv_proj,
                    file_name=f"erm_scenario_projection_{datetime.now():%Y%m%d}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with col_e2:
                import json as _json

                assumptions_json = _json.dumps(assumptions.to_dict(), indent=2)
                st.download_button(
                    "⚙️ Download assumptions JSON",
                    data=assumptions_json,
                    file_name=f"erm_scenario_assumptions_{datetime.now():%Y%m%d}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with col_e3:
                if comparison is not None:
                    csv_cmp = comparison.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📈 Download comparison CSV",
                        data=csv_cmp,
                        file_name=f"erm_scenario_comparison_{datetime.now():%Y%m%d}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "Comparison not available",
                        disabled=True,
                        use_container_width=True,
                        help="Enable 'Compare with house price stress' and re-run.",
                    )
        else:
            st.info("Configure assumptions and click **Run scenario** to see projections.")

# ============================
# TAB 3: STATIC POOLS (Fully Corrected)
# ============================

with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Static Pool Analysis – Equity Release")

    # ── Prepare data ──────────────────────────────────────────────────
    df_sp = df.copy()

    # 1. Account ID Setup
    possible_id_cols = ["loan_id", "account_id", "unique_identifier", "id"]
    account_id_col = next((col for col in possible_id_cols if col in df_sp.columns), None)

    if account_id_col is not None:
        df_sp[account_id_col] = df_sp[account_id_col].astype(str)
        null_mask = df_sp[account_id_col].isna()
        if null_mask.any():
            st.warning(f"Filling {null_mask.sum()} nulls in '{account_id_col}' with fallback IDs.")
            fallback_ids = [f"Fallback_{i:06d}" for i in range(null_mask.sum())]
            df_sp.loc[null_mask, account_id_col] = fallback_ids
    else:
        st.info("No ID column found → generating synthetic account IDs.")
        df_sp["account_id"] = [f"ACC_{i:06d}" for i in range(len(df_sp))]
        account_id_col = "account_id"

    # 2. As-of Date Setup
    possible_asof_cols = ["data_cut_off_date", "as_of_date", "reporting_date", "cut_off_date", "cutoff_date"]
    as_of_col = next((col for col in possible_asof_cols if col in df_sp.columns), None)

    if as_of_col is not None:
        df_sp[as_of_col] = coerce_datetime(df_sp[as_of_col], dayfirst=True)
        null_mask = df_sp[as_of_col].isna()
        if null_mask.any():
            st.warning(f"Filling {null_mask.sum()} null as-of dates with today's date.")
            df_sp.loc[null_mask, as_of_col] = pd.Timestamp.today().normalize()
    else:
        st.info("No as-of date column found → using today's date for all rows.")
        df_sp["as_of_date"] = pd.Timestamp.today().normalize()
        as_of_col = "as_of_date"

    # 3. Status Setup
    possible_status_cols = ["account_status", "loan_status", "performance_status", "status"]
    status_col = next((col for col in possible_status_cols if col in df_sp.columns), None)

    if status_col is not None:
        df_sp[status_col] = df_sp[status_col].astype(str)
        df_sp.loc[df_sp[status_col].isna(), status_col] = "Unknown"
    else:
        df_sp["account_status"] = "Unknown"
        status_col = "account_status"

    # 4. Origination Date
    if "origination_date" not in df_sp.columns:
        st.error("❌ Missing required field: origination_date")
        st.stop()
    df_sp["origination_date"] = coerce_datetime(df_sp["origination_date"], dayfirst=True)

    # ── CRITICAL DATA PREP (Restored) ──────────────────────────────────
    
    # A. Create Risk Buckets (Original LTV)
    if "original_loan_to_value" in df_sp.columns:
        ltv = pd.to_numeric(df_sp["original_loan_to_value"], errors="coerce")
        if ltv.median() > 1:
            ltv = ltv / 100
        df_sp["risk_bucket"] = pd.cut(
            ltv, bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")],
            labels=["<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%", ">100%"],
            include_lowest=True
        ).astype(str).fillna("Unknown")
    else:
        df_sp["risk_bucket"] = "ALL"

    # B. Normalize Current LTV (needed for Config charts)
    if "current_ltv" not in df_sp.columns:
        possible_ltv_cols = [
            "current_loan_to_value", "curr_ltv", "LTV", "Cur LTV %",
            "currentLoanToValue", "ltv_latest"
        ]
        ltv_source = next((c for c in possible_ltv_cols if c in df_sp.columns), None)
        
        if ltv_source:
            df_sp["current_ltv"] = pd.to_numeric(df_sp[ltv_source], errors='coerce')
            if df_sp["current_ltv"].median(skipna=True) > 1:
                df_sp["current_ltv"] = df_sp["current_ltv"] / 100
        else:
            # Fallback if strictly needed to prevent chart crash
            df_sp["current_ltv"] = 0.0

    # C. Normalize Interest Rate (needed for Config charts)
    if "interest_rate" not in df_sp.columns:
        possible_rate_cols = [
            "current_interest_rate", "interest_rate", "Loan Interest Rate",
            "interestRate", "current_coupon"
        ]
        rate_source = next((c for c in possible_rate_cols if c in df_sp.columns), None)
        
        if rate_source:
            df_sp["interest_rate"] = pd.to_numeric(df_sp[rate_source], errors='coerce')
            if df_sp["interest_rate"].median(skipna=True) > 1:
                df_sp["interest_rate"] = df_sp["interest_rate"] / 100
        else:
             df_sp["interest_rate"] = 0.0

    # D. Ensure Optional Columns
    if "interest_accrued" not in df_sp.columns:
        df_sp["interest_accrued"] = 0.0
    if "prepayment_amount" not in df_sp.columns:
        df_sp["prepayment_amount"] = 0.0

    # ── BUILD SPEC (With Total Balance Fix) ────────────────────────────
    sp_spec = StaticPoolsSpec(
        account_id=account_id_col,
        as_of_date=as_of_col,
        origination_date="origination_date",
        geo_region="geographic_region_classification", # Ensure this matches your data
        product_type="erm_product_type",
        risk_bucket="risk_bucket",
        account_status=status_col,
        
        # FIX: Point to total_balance to ensure data flows through
        principal_outstanding="total_balance",
        
        interest_accrued="interest_accrued",
        prepayment_amount="prepayment_amount"
    )

    # ── BUILD PANEL ────────────────────────────────────────────────────
    with st.spinner("Computing static pools panel..."):
        panel = build_static_pools_panel(df_sp, spec=sp_spec)

    if panel.empty:
        st.info("No valid data for static pool analysis.")
        st.stop()

    # ── CONFIG LOADER ──────────────────────────────────────────────────
    config_path = "static_pools_config_erm.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        charts = config.get("static_pools", {}).get("charts", [])
    except Exception:
        # Fallback charts if config missing
        charts = [
            {"title": "Weighted Avg LTV", "metric": "current_ltv", "agg": "mean", "format": ".1%"},
            {"title": "Weighted Avg Interest Rate", "metric": "interest_rate", "agg": "mean", "format": ".2%"}
        ]

    # ── SNAPSHOT SELECTOR ──────────────────────────────────────────────
    unique_dates = sorted(panel[sp_spec.as_of_date].dropna().unique())
    selected_date = unique_dates[-1]
    
    if len(unique_dates) > 1:
        selected_date = st.selectbox(
            "Snapshot date",
            options=unique_dates,
            format_func=lambda d: pd.Timestamp(d).strftime("%d-%b-%Y"),
            index=len(unique_dates)-1
        )

    panel_current = panel[panel[sp_spec.as_of_date] == selected_date].copy()

    # ── CHARTS LOOP (Improved UI) ──────────────────────────────────────
    for chart_def in charts:
        title = chart_def["title"]
        metric = chart_def["metric"]
        agg = chart_def["agg"]
        fmt = chart_def.get("format", ".1%" if any(x in metric.lower() for x in ["rate", "ltv", "cpr", "pct"]) else None)

        if metric not in panel_current.columns:
            continue

        st.subheader(title)

        segment_choice = st.radio(
            "Segment by",
            options=["Total Portfolio", "Geography", "Product Type"],
            index=0,
            horizontal=True,
            key=f"seg_{title}_{metric}"
        )

        segment_dim = None
        if segment_choice == "Geography":
            segment_dim = sp_spec.geo_region
        elif segment_choice == "Product Type":
            segment_dim = sp_spec.product_type

        dims = [sp_spec.origination_month]
        if segment_dim:
            dims.append(segment_dim)

        if agg == "mean":
            def weighted_mean(group):
                w = group[sp_spec.principal_outstanding].sum()
                if w == 0: return np.nan
                return np.average(group[metric], weights=group[sp_spec.principal_outstanding])
            agg_df = panel_current.groupby(dims, dropna=False).apply(weighted_mean).reset_index(name="value")
        else:
            agg_df = panel_current.groupby(dims, dropna=False)[metric].sum().reset_index(name="value")

        if segment_dim:
            agg_df["segment"] = agg_df[segment_dim].astype(str)
            color_col = "segment"
        else:
            color_col = None
            agg_df["segment"] = "Total"

        # Prepare x-axis for Plotly
        x_col = sp_spec.origination_month
        if x_col in agg_df.columns:
            # 1. Ensure it is a datetime for sorting
            if pd.api.types.is_period_dtype(agg_df[x_col]):
                agg_df["_sort_date"] = agg_df[x_col].dt.to_timestamp()
            else:
                agg_df["_sort_date"] = pd.to_datetime(agg_df[x_col])
            
            # 2. Create the DISPLAY label (Force String)
            agg_df["vintage_label"] = agg_df["_sort_date"].dt.strftime("%b-%Y")
            
            # 3. Sort correctly
            agg_df = agg_df.sort_values("_sort_date")

        # 4. Plot using the LABEL, not the date
        if agg == "mean":
            fig = px.line(
                agg_df, 
                x="vintage_label",  # <--- CHANGE THIS
                y="value", 
                color=color_col,
                markers=True
            )
        else:
            fig = px.bar(
                agg_df, 
                x="vintage_label", # <--- CHANGE THIS
                y="value", 
                color=color_col
            )
            
        # 5. Force category type (Strict safety net)
        fig.update_xaxes(type="category", title_text="Origination Vintage")

        fig = apply_chart_theme(fig, title)
        fig.update_xaxes(title_text="Origination Vintage", type="category")
        st.plotly_chart(fig, use_container_width=True)

# ============================
# TAB 4: RISK MONITORING
# ============================

if RISK_MONITORING_AVAILABLE:
    with tab4:
        st.markdown("### 🚦 Risk Limit Monitoring")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Initialize risk monitor
        monitor = RiskMonitor(df)
        results = monitor.check_all_limits()
        summary = monitor.get_summary_stats(results)
        
        # Risk Monitor Summary Tiles
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">Total Limits</div>
                <div class="kpi-value">{summary['total_limits']}</div>
                <div class="kpi-subtitle">Monitored limits</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-box" style="border-left: 4px solid #DC3545;">
                <div class="kpi-label">🔴 Breaches</div>
                <div class="kpi-value" style="color: #DC3545;">{summary['breaches']}</div>
                <div class="kpi-subtitle">Limits exceeded</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="kpi-box" style="border-left: 4px solid #FFC107;">
                <div class="kpi-label">🟡 Warnings</div>
                <div class="kpi-value" style="color: #FFC107;">{summary['warnings']}</div>
                <div class="kpi-subtitle">Approaching limits</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="kpi-box" style="border-left: 4px solid #28A745;">
                <div class="kpi-label">🟢 Compliant</div>
                <div class="kpi-value" style="color: #28A745;">{summary['compliant']}</div>
                <div class="kpi-subtitle">Within limits</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
              
        # ===== DETAILED LIMIT TABLE =====
        st.markdown("#### Detailed Limit Status")
        
        # Filter controls
        col1, col2 = st.columns([2, 2])
               
        with col1:
            status_filter = st.multiselect(
                "Filter by status",
                options=["red", "amber", "green", "unknown"],
                default=["red", "amber", "green"]
            )
        
        with col2:
            severity_filter = st.multiselect(
                "Filter by severity",
                options=["critical", "high"],
                default=["critical", "high"]
            )
        
            # Apply filters
            filtered_results = [
                r
                for r in results
                if r.status in status_filter
                and r.severity in severity_filter
            ]
        
        # Sort by status (red first, then amber, then green)
        status_order = {"red": 0, "amber": 1, "green": 2, "unknown": 3}
        filtered_results.sort(key=lambda r: (status_order[r.status], r.severity == "high"))
        
        # Build table data
        table_data = []
        for r in filtered_results:
            # Status indicator
            status_icon = {
                "red": "🔴",
                "amber": "🟡",
                "green": "🟢",
                "unknown": "⚪"
            }[r.status]
            
            # Format values based on type
            if "pct" in r.limit_id or "concentration" in r.limit_id or "ltv" in r.limit_id or "dpd" in r.limit_id:
                current_str = f"{r.current_value:.1f}%" if not np.isnan(r.current_value) else "N/A"
                limit_str = f"{r.limit_value:.1f}%"
            elif "absolute" in r.limit_id:
                current_str = f"£{r.current_value:,.0f}" if not np.isnan(r.current_value) else "N/A"
                limit_str = f"£{r.limit_value:,.0f}"
            elif "age" in r.limit_id or "rate" in r.limit_id:
                current_str = f"{r.current_value:.1f}" if not np.isnan(r.current_value) else "N/A"
                limit_str = f"{r.limit_value:.1f}"
            elif "size" in r.limit_id:
                current_str = f"{int(r.current_value)}" if not np.isnan(r.current_value) else "N/A"
                limit_str = f"{int(r.limit_value)}"
            else:
                current_str = f"{r.current_value:.2f}" if not np.isnan(r.current_value) else "N/A"
                limit_str = f"{r.limit_value:.2f}"
            
            # Utilization bar
            util_pct = min(r.utilization_pct, 150)  # Cap at 150% for display
            util_color = "#DC3545" if r.status == "red" else "#FFC107" if r.status == "amber" else "#28A745"
            
            table_data.append({
                "Status": status_icon,
                "Category": r.category,
                "Description": r.description,
                "Current": current_str,
                "Limit": limit_str,
                "Utilization": r.utilization_pct,
                "Severity": r.severity.upper()
            })
        
        if table_data:
            df_table = pd.DataFrame(table_data)
            
            st.dataframe(
                df_table,
                use_container_width=True,
                height=600,
                hide_index=True
            )
            
            # Export button
            csv = df_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Limit Report",
                data=csv,
                file_name=f"risk_limits_{datetime.now():%Y%m%d}.csv",
                mime="text/csv",
                use_container_width=False
            )
        else:
            st.info("No limits match the current filters.")
        
        # ===== BREACH DRILL-DOWN =====
        if summary["breaches"] > 0:
            st.markdown("---")
            st.markdown("#### 🚨 Breach Details")
            
            breached = [r for r in results if r.status == "red"]
            breached.sort(key=lambda r: (r.severity == "high", -r.utilization_pct))
            
            for r in breached:
                severity_color = "#DC3545" if r.severity == "critical" else "#FF6B35"
                
                if st.checkbox(f"{'🚨' if r.severity == 'critical' else '⚠️'} {r.description}", 
                            value=(r.severity == "critical"), 
                            key=f"breach_{r.limit_id}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Category:** {r.category}")
                        st.markdown(f"**Severity:** {r.severity.upper()}")
                        st.markdown(f"**Current Value:** {r.current_value:.2f}")
                        st.markdown(f"**Limit:** {r.limit_value:.2f}")
                        
                        if r.breach_amount is not None:
                            st.markdown(f"**Breach Amount:** {r.breach_amount:.2f}")
                            st.markdown(f"**Utilization:** {r.utilization_pct:.1f}%")
                    
                    with col2:
                    # Gauge chart for utilization
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=r.utilization_pct,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [None, 150]},
                                'bar': {'color': severity_color},
                                'steps': [
                                    {'range': [0, 80], 'color': "#E8F5E9"},
                                    {'range': [80, 100], 'color': "#FFF3E0"},
                                    {'range': [100, 150], 'color': "#FFEBEE"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 100
                               }
                            },
                            number={'suffix': "%"}
                        ))
                        
                        fig_gauge.update_layout(
                            height=200,
                            margin=dict(l=20, r=20, t=20, b=20)
                        )
                        
                        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # ===== CONFIGURATION INFO =====
        st.markdown("### ℹ️ About Risk Limits")
        st.markdown("""
            **Risk Limit Framework**
            
            This dashboard monitors portfolio compliance against pre-defined risk limits across six categories:
            
            1. **Single Exposure**: Maximum ticket size and concentration in individual loans
            2. **Concentration**: Geographic, broker, and top-N exposure limits
            3. **Credit Quality**: LTV thresholds and delinquency limits
            4. **Borrower Characteristics**: Age constraints and portfolio averages
            5. **Portfolio Metrics**: Weighted average LTV, rate, and portfolio size
            6. **ERM-Specific**: NNEG exposure and product mix requirements
            
            **Status Indicators:**
            - 🟢 **Green**: Within limits (compliant)
            - 🟡 **Amber**: Approaching limit (warning zone, typically >80% utilization)
            - 🔴 **Red**: Limit breached (immediate action required)
            
            **Updating Limits:**
            Limits are configured in `risk_limits_config.py`. Contact your portfolio manager 
            to adjust thresholds based on mandate requirements.
            """)


# ============================
# FOOTER
# ============================

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: {TEXT_LIGHT}; font-size: 11px; padding: 0.5rem 0 1rem 0;'>
        <b>Confidential</b> • Powered by Digifin •
        Data as of {datetime.now():%B %d, %Y}
    </div>
    """,
    unsafe_allow_html=True,
)

"""
Shared configuration constants for the ERM analytics suite.

Both streamlit_app_erm.py and generate_pptx_client.py import from here
so theme colours and limits stay in one place.
"""

# ── Brand colours ──────────────────────────────────────────────────────────
PRIMARY_COLOR = "#232D55"       # Navy
SECONDARY_COLOR = "#919DD1"     # Light blue
ACCENT_COLOR = "#BFBFBF"        # Grey
TEXT_DARK = "#2D2D2D"
TEXT_LIGHT = "#6B6B6B"
BACKGROUND_LIGHT = "#F8F9FA"
BORDER_COLOR = "#E0E0E0"        # Neutral grey for UI borders
CHART_COLORS = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR]

# ── Limits ─────────────────────────────────────────────────────────────────
MAX_ROWS = 100_000              # Max rows loaded by the Streamlit dashboard
MAX_FILE_SIZE_MB = 500          # Max CSV file size accepted by PPTX generator
MAX_PPTX_ROWS = 100_000         # Max rows read when generating a PowerPoint

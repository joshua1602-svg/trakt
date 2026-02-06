# config.py
"""
Shared configuration for the SME Loan Analytics toolkit.
Used by:
- streamlit_app.py
- pptx_generator.py
- loan_amortisation.py
"""

from pathlib import Path

# ---------- DATA / PATHS ----------

DEFAULT_CANONICAL_PATH = Path(
    "out_typed/ERE_Portfolio_102025_ESMA_Annex2_canonical_ESMA_Annex2_typed.csv"
)

REQUIRED_COLUMNS = [
    "unique_identifier",
    "original_principal_balance",
    "current_principal_balance",
    "current_interest_rate",
    "origination_date",
    "maturity_date",
]

OPTIONAL_COLUMNS = [
    "number_of_days_in_arrears",
    "geographic_region_classification",
    "originator_establishment_country",
    "account_status",
    "arrears_balance",
    "current_loan_to_value",
    "purpose",
    "scheduled_interest_payment_frequency",
    
]

# Limits
MAX_FILE_SIZE_MB = 100
MAX_ROWS = 500_000
MAX_PPTX_ROWS = 1_000_000

# ---------- THEME ----------

PRIMARY_COLOR = "#232D55"
SECONDARY_COLOR = "#919DD1"
ACCENT_COLOR = "#BFBFBF"
TEXT_DARK = "#2D2D2D"
TEXT_LIGHT = "#6B6B6B"
BACKGROUND_LIGHT = "#F8F9FA"
DANGER_COLOR = "#DC3545"
WARNING_COLOR = "#FFC107"
SUCCESS_COLOR = "#28A745"

CHART_COLORS = [
    PRIMARY_COLOR,
    "#7EBAB5",
    "#5AA9A3",
    ACCENT_COLOR,
    "#A3CCC9",
]
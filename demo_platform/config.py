"""demo_platform.config — the ONE place the demonstration narrative is defined.

Fictional client identity, portfolio identifiers, reporting periods, source
schemas, target metrics and tolerances all live here so the whole demo (data
generation, orchestration, assertions, Remotion captions) can be re-pointed by
editing a single module.

Nothing here is derived from, or resembles, any real client. The names below are
invented for the demonstration and are asserted against a prohibited-string list
by :mod:`demo_platform.safety`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Stamped into every generated source file, manifest and fixture.
SYNTHETIC_BANNER = "SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER"

#: Short form used where a full banner will not fit (CSV filename suffix, footers).
SYNTHETIC_SHORT = "SYNTHETIC DEMONSTRATION DATA"


# --------------------------------------------------------------------------- #
# Fictional client identity
# --------------------------------------------------------------------------- #
#: Managed-service client id. Lowercase slug — it becomes a storage path segment.
CLIENT_ID = "alderbridge"

#: Display name shown in the MI Agent header, investor deck and film.
CLIENT_DISPLAY_NAME = "Alderbridge Lending Platform"

#: Short display form for tight UI chrome.
CLIENT_SHORT_NAME = "Alderbridge"

#: Fictional seller for the acquired book (never a real market participant).
ACQUIRED_SELLER_NAME = "Cranmere Life Assurance (synthetic)"


@dataclass(frozen=True)
class PortfolioSpec:
    """One onboarded source portfolio under the platform.

    ``source_portfolio_id`` must satisfy ``engine.provenance._ID_RE`` (a
    lowercase slug), so the neutral commercial identifier used on screen
    (``ALP_ORIGINATION``) is carried separately in ``display_id``.
    """

    key: str                     # "A" | "B" — storyboard label
    source_portfolio_id: str     # canonical provenance slug
    source_portfolio_type: str   # direct | acquired
    display_id: str              # neutral identifier shown on screen
    label: str                   # human label (source_portfolio_label)
    short_label: str             # tight UI form
    description: str             # one-line commercial description
    schema_name: str             # which source schema this portfolio arrives in
    acquisition_date: Optional[str] = None
    seller_name: Optional[str] = None
    #: Where the portfolio sits relative to the sponsor's balance sheet:
    #: "warehoused" (held, destined for a future deal) or "sold" (derecognised).
    #:
    #: Distinct from ``source_portfolio_type``, which the production pipeline defines as
    #: origination PROVENANCE — direct or acquired — and which says nothing about who
    #: owns the book now. A securitisation the sponsor originated and then sold is
    #: `direct` provenance and `sold` status, and both facts are true at once.
    balance_sheet_status: str = "warehoused"
    #: One-line plain-English status for the film's consolidation lanes.
    status_detail: str = ""
    #: Target loan count in the CURRENT reporting period.
    target_current_loans: int = 0
    #: Loans that have redeemed/closed historically and remain on the tape.
    target_closed_loans: int = 0


PORTFOLIO_A = PortfolioSpec(
    key="A",
    source_portfolio_id="alp_origination",
    source_portfolio_type="direct",
    display_id="ALP_ORIGINATION",
    label="ALP Origination Book",
    short_label="Origination",
    description="Organically originated lifetime mortgage portfolio",
    schema_name="origination_extract",
    balance_sheet_status="warehoused",
    status_detail="Warehoused · destined for SPV2",
    target_current_loans=7_100,
    target_closed_loans=520,
)

PORTFOLIO_B = PortfolioSpec(
    key="B",
    source_portfolio_id="alp_acquired",
    source_portfolio_type="acquired",
    display_id="ALP_ACQUIRED",
    label="ALP Acquired Back Book",
    short_label="Acquired",
    description="Acquired UK lifetime mortgage back book",
    schema_name="servicer_account_extract",
    acquisition_date="2024-09-30",
    seller_name=ACQUIRED_SELLER_NAME,
    balance_sheet_status="warehoused",
    status_detail="Warehoused · destined for SPV2",
    target_current_loans=3_900,
    target_closed_loans=310,
)

#: Portfolio S — the securitisation this sponsor originated and SOLD.
#:
#: It is derecognised: off the sponsor's balance sheet, owned by the noteholders. The
#: sponsor nonetheless retains servicing, risk retention and investor-reporting
#: obligations, so it still has to be governed — and it is the portfolio their own
#: ledger can no longer tell them about.
#:
#: Deliberately NOT in :data:`PORTFOLIOS`. That tuple is the WAREHOUSED platform, whose
#: assembled canonical is the £1.96bn / 11,035-loan figure every downstream artefact and
#: MI answer is built on. SPV1 is governed alongside it and aggregates with it only in
#: the sponsor view; putting it in the platform set would silently restate every figure
#: in the demonstration.
PORTFOLIO_S = PortfolioSpec(
    key="S",
    # `engine.provenance` requires a lowercase slug with at least one underscore, so the
    # stable id carries the status the display id has no room for.
    source_portfolio_id="spv1_sponsored",
    # The sponsor ORIGINATED these loans and later securitised them, so their
    # provenance is direct. The production pipeline's portfolio type is about
    # provenance only; that the book has since been sold is `balance_sheet_status`.
    source_portfolio_type="direct",
    display_id="SPV1",
    label="SPV1 Sponsored Securitisation",
    short_label="SPV1",
    description="Sponsored securitisation, sold — servicing and reporting retained",
    schema_name="trustee_deal_extract",
    balance_sheet_status="sold",
    status_detail="Servicing · risk retention · investor reporting",
    target_current_loans=4_180,
    target_closed_loans=260,
)

#: The WAREHOUSED platform: what the assembler consolidates, and the scope of every
#: figure in the demonstration except the sponsor consolidation.
PORTFOLIOS: Tuple[PortfolioSpec, ...] = (PORTFOLIO_A, PORTFOLIO_B)

#: Everything the sponsor is on the hook for: the platform plus the deal they sold.
#: Used by generation, onboarding and orchestration — every portfolio here is governed
#: through the same gates — and by the sponsor-scope aggregate in the metrics export.
ALL_PORTFOLIOS: Tuple[PortfolioSpec, ...] = PORTFOLIOS + (PORTFOLIO_S,)


def portfolio(key_or_id: str) -> PortfolioSpec:
    """Look a portfolio up by storyboard key, slug or display id."""
    needle = str(key_or_id).strip().lower()
    for p in ALL_PORTFOLIOS:
        if needle in (p.key.lower(), p.source_portfolio_id, p.display_id.lower()):
            return p
    raise KeyError(f"unknown demo portfolio: {key_or_id!r}")


# --------------------------------------------------------------------------- #
# Reporting periods
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PeriodSpec:
    """One month-end reporting period."""

    run_id: str            # mi_2026_04 — the managed-service run identifier
    reporting_date: str    # 2026-04-30 — month-end cut-off
    label: str             # "April 2026"
    short_label: str       # "Apr 26"
    role: str              # opening | prior | current

    @property
    def period(self) -> str:
        """``YYYY-MM`` — the period key the MI services use."""
        return self.reporting_date[:7]


#: Three month-ends. All historical relative to any plausible execution date.
#: The film compares CURRENT vs PRIOR; the third (opening) period exists so
#: evolution charts and the bridge have a genuine three-point series.
PERIODS: Tuple[PeriodSpec, ...] = (
    PeriodSpec("mi_2026_04", "2026-04-30", "April 2026", "Apr 26", "opening"),
    PeriodSpec("mi_2026_05", "2026-05-31", "May 2026", "May 26", "prior"),
    PeriodSpec("mi_2026_06", "2026-06-30", "June 2026", "Jun 26", "current"),
)


def period(role_or_run: str) -> PeriodSpec:
    """Look a period up by role (``current``/``prior``/``opening``) or run id."""
    needle = str(role_or_run).strip().lower()
    for p in PERIODS:
        if needle in (p.role, p.run_id, p.reporting_date, p.period):
            return p
    raise KeyError(f"unknown demo period: {role_or_run!r}")


CURRENT_PERIOD = period("current")
PRIOR_PERIOD = period("prior")
OPENING_PERIOD = period("opening")


# --------------------------------------------------------------------------- #
# Engineered narrative targets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricTarget:
    """A target the generated data must genuinely produce, with a tolerance.

    The demo run FAILS when a computed metric falls outside ``tolerance``. Values
    are never patched to match the script — the source data is engineered so the
    deterministic output lands here.
    """

    name: str
    target: float
    tolerance: float
    unit: str
    description: str

    def ok(self, value: Optional[float]) -> bool:
        if value is None:
            return False
        return abs(float(value) - self.target) <= self.tolerance

    def delta(self, value: Optional[float]) -> Optional[float]:
        return None if value is None else round(float(value) - self.target, 6)


#: Consolidated month-on-month funded balance movement (current vs prior).
TARGET_MOVEMENT_TOTAL = MetricTarget(
    "consolidated_funded_balance_movement", 18_200_000.0, 250_000.0, "GBP",
    "Consolidated funded balance increase, current vs prior reporting month.")

#: Per-portfolio contribution to that movement.
TARGET_MOVEMENT_A = MetricTarget(
    "portfolio_a_funded_balance_movement", 11_600_000.0, 200_000.0, "GBP",
    "ALP Origination Book contribution to the consolidated movement.")
TARGET_MOVEMENT_B = MetricTarget(
    "portfolio_b_funded_balance_movement", 6_600_000.0, 200_000.0, "GBP",
    "ALP Acquired Back Book contribution to the consolidated movement.")

#: Weighted-average current LTV, in percentage points, current period.
TARGET_WA_LTV_CURRENT = MetricTarget(
    "consolidated_wa_current_ltv", 43.1, 0.25, "percent_points",
    "Balance-weighted average current LTV at the current reporting date.")

#: Average youngest-borrower age (the product's existing metric definition:
#: ``evolution.assemble_funded_evolution`` uses a simple mean of
#: ``youngest_borrower_age``).
TARGET_BORROWER_AGE_CURRENT = MetricTarget(
    "consolidated_avg_borrower_age", 71.4, 0.15, "years",
    "Average youngest-borrower age at the current reporting date.")

#: Consolidated current loan count must sit in the commercially credible band.
LOAN_COUNT_BAND: Tuple[int, int] = (9_000, 12_000)

#: The region that must be the single largest positive contributor to the
#: consolidated movement, per ``evolution.funded_bridge``.
PRIMARY_MOVEMENT_REGION = "South East"

#: The primary region must carry at least this share of the total increase, AND
#: be at least ``PRIMARY_REGION_MIN_LEAD`` times the second-largest contributor,
#: for the word "primarily" to be defensible. The lead test is the stronger of
#: the two: a 30% share with the runner-up on 14% genuinely is "primarily".
PRIMARY_REGION_MIN_SHARE = 0.25
PRIMARY_REGION_MIN_LEAD = 1.8

#: Borrower age must rise (directionally) current vs prior — "increased slightly".
BORROWER_AGE_MUST_INCREASE = True

#: WA LTV must be "broadly stable": absolute change under this many points.
WA_LTV_STABILITY_MAX_CHANGE_POINTS = 0.6

#: SPV1's own funded balance at the current reporting date. Calibrated independently of
#: the platform's movement targets — a sold deal has its own economics and must not
#: perturb a single figure in the warehoused platform.
TARGET_BALANCE_S = MetricTarget(
    "spv1_funded_balance", 842_600_000.0, 50_000.0, "GBP",
    "Sponsored securitisation, sold — funded balance at the reporting date.")

METRIC_TARGETS: Tuple[MetricTarget, ...] = (
    TARGET_MOVEMENT_TOTAL,
    TARGET_MOVEMENT_A,
    TARGET_MOVEMENT_B,
    TARGET_WA_LTV_CURRENT,
    TARGET_BORROWER_AGE_CURRENT,
    TARGET_BALANCE_S,
)


# --------------------------------------------------------------------------- #
# UK distribution shape (synthetic, plausible, internally consistent)
# --------------------------------------------------------------------------- #
#: ITL1 region → share of the CURRENT consolidated book by loan count.
#: Weighted toward the South East / London / South West, which is the shape of
#: the UK lifetime-mortgage market (owner-occupied equity concentration).
REGION_WEIGHTS: Dict[str, float] = {
    "South East": 0.215,
    "London": 0.125,
    "South West": 0.120,
    "East of England": 0.105,
    "North West": 0.098,
    "West Midlands": 0.082,
    "Yorkshire and The Humber": 0.075,
    "East Midlands": 0.070,
    "Scotland": 0.052,
    "North East": 0.030,
    "Wales": 0.028,
}

#: Share of the monthly NEW-BUSINESS balance by region (completions). Deliberately
#: more South-East weighted than the stock — the origination book's broker network
#: is concentrated there — so the South East is the primary contributor to the
#: monthly movement. This is the mechanism behind the film's narrative; the
#: narrative is then *verified* against the computed bridge, not assumed.
COMPLETION_REGION_WEIGHTS: Dict[str, float] = {
    "South East": 0.400,
    "London": 0.148,
    "South West": 0.112,
    "East of England": 0.092,
    "North West": 0.070,
    "West Midlands": 0.055,
    "Yorkshire and The Humber": 0.042,
    "East Midlands": 0.039,
    "Scotland": 0.022,
    "North East": 0.011,
    "Wales": 0.009,
}

#: ITL1 names as they appear in ``uk_itl_master_lookup_v2.csv``, mapped to the
#: clean display labels above, so generated postcodes are genuinely inside the
#: region each loan claims.
ITL1_LOOKUP_NAMES: Dict[str, str] = {
    "South East": "South East (England)",
    "London": "London",
    "South West": "South West (England)",
    "East of England": "East (England)",
    "North West": "North West (England)",
    "West Midlands": "West Midlands (England)",
    "Yorkshire and The Humber": "Yorkshire and The Humber",
    "East Midlands": "East Midlands (England)",
    "Scotland": "Scotland",
    "North East": "North East (England)",
    "Wales": "Wales",
}

#: Regional indexed-valuation level relative to the national average. Drives the
#: property-value distribution so LTVs reconcile regionally.
REGION_VALUE_INDEX: Dict[str, float] = {
    "South East": 1.34,
    "London": 1.92,
    "South West": 1.12,
    "East of England": 1.18,
    "North West": 0.78,
    "West Midlands": 0.84,
    "Yorkshire and The Humber": 0.74,
    "East Midlands": 0.82,
    "Scotland": 0.72,
    "North East": 0.62,
    "Wales": 0.72,
}


@dataclass(frozen=True)
class PortfolioShape:
    """The synthetic distribution shape for one portfolio.

    All of these are *starting points*. ``demo_platform.generator`` runs a
    deterministic calibration that solves a small number of scalar adjustments
    (initial-LTV shift, origination-age shift, new-business volume, drawdown
    volume) so the produced tape genuinely lands on the narrative targets. The
    solved values are recorded in the demo manifest.
    """

    #: Mean / sd of the youngest borrower's age at origination.
    origination_age_mean: float
    origination_age_sd: float
    #: Mean / sd of the initial advance as a fraction of the original valuation.
    initial_ltv_mean: float
    initial_ltv_sd: float
    #: Fixed lifetime roll-up rate distribution (annual, percentage points).
    rate_mean: float
    rate_sd: float
    #: Origination vintage window (inclusive years).
    vintage_range: Tuple[int, int]
    #: National average original property valuation (GBP), before the regional index.
    base_valuation: float
    base_valuation_sd: float
    #: Monthly house-price index growth applied to the indexed valuation.
    monthly_hpi: float
    #: New completions per reporting month, oldest → newest period.
    monthly_completions: Tuple[int, int, int]
    #: Redemptions per reporting month, oldest → newest period.
    monthly_redemptions: Tuple[int, int, int]
    #: Reserve-facility drawdowns (further advances) per month: number of loans.
    monthly_drawdown_loans: Tuple[int, int, int]
    #: Average drawdown tranche (GBP) when a loan draws.
    drawdown_mean: float
    drawdown_sd: float
    #: Origination channel reported for every loan in the book. Drawn from the
    #: values the delivery rules already recognise, so the ESMA Annex 2 code
    #: (RREL26) resolves without touching production configuration.
    origination_channel: str = "Direct"


#: Portfolio A — organically originated, recent vintages, currently writing new
#: business. Growth comes from completions plus interest roll-up.
SHAPE_A = PortfolioShape(
    origination_age_mean=68.2, origination_age_sd=5.1,
    initial_ltv_mean=0.357, initial_ltv_sd=0.078,
    rate_mean=6.42, rate_sd=0.72,
    vintage_range=(2019, 2026),
    base_valuation=402_000.0, base_valuation_sd=118_000.0,
    monthly_hpi=0.00090,
    monthly_completions=(52, 54, 57),
    monthly_redemptions=(23, 24, 26),
    monthly_drawdown_loans=(0, 0, 0),
    drawdown_mean=0.0, drawdown_sd=0.0,
    origination_channel="Direct",
)

#: Portfolio B — the acquired back book: closed to new origination, seasoned,
#: higher legacy roll-up rates. Growth comes from interest roll-up plus
#: reserve-facility drawdowns (further advances) on existing loans.
SHAPE_B = PortfolioShape(
    origination_age_mean=63.5, origination_age_sd=5.6,
    initial_ltv_mean=0.268, initial_ltv_sd=0.070,
    rate_mean=6.85, rate_sd=0.66,
    vintage_range=(2014, 2022),
    base_valuation=318_000.0, base_valuation_sd=96_000.0,
    monthly_hpi=0.00090,
    monthly_completions=(0, 0, 0),
    monthly_redemptions=(14, 15, 9),
    monthly_drawdown_loans=(96, 101, 107),
    drawdown_mean=41_300.0, drawdown_sd=11_400.0,
    # The acquired book was written through intermediaries.
    origination_channel="Broker",
)

#: Portfolio S — the sold securitisation. Seasoned, closed to new origination, no
#: reserve facility: a static pool being run off, which is what a sold deal is. Its
#: month-on-month movement is pure interest roll-up less redemptions, and it takes no
#: part in the platform's movement calibration.
SHAPE_S = PortfolioShape(
    origination_age_mean=66.4, origination_age_sd=5.3,
    initial_ltv_mean=0.292, initial_ltv_sd=0.074,
    rate_mean=6.71, rate_sd=0.61,
    vintage_range=(2016, 2023),
    base_valuation=352_000.0, base_valuation_sd=104_000.0,
    monthly_hpi=0.00090,
    monthly_completions=(0, 0, 0),
    monthly_redemptions=(11, 12, 10),
    monthly_drawdown_loans=(0, 0, 0),
    drawdown_mean=0.0, drawdown_sd=0.0,
    origination_channel="Broker",
)

SHAPES: Dict[str, PortfolioShape] = {
    PORTFOLIO_A.source_portfolio_id: SHAPE_A,
    PORTFOLIO_B.source_portfolio_id: SHAPE_B,
    PORTFOLIO_S.source_portfolio_id: SHAPE_S,
}

#: Deterministic seed. Every run with this seed reproduces byte-identical data.
RANDOM_SEED = 20260630

#: Deliberate data-quality exceptions injected per portfolio per period, so the
#: validation gate has something genuine to report. Small, bounded, and confined
#: to NON-core canonical fields: blanking a core field would hard-fail Gate 2,
#: and the demonstration should show a run that completes with a realistic
#: exception profile, not one that fails.
INJECTED_EXCEPTIONS_PER_PERIOD = 5


# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
def demo_root() -> Path:
    """Root of all generated demo state. Overridable via ``TRAKT_DEMO_ROOT``."""
    override = os.environ.get("TRAKT_DEMO_ROOT")
    return Path(override) if override else REPO_ROOT / "demo_platform" / "workspace"


def source_dir() -> Path:
    """Where the generated raw source extracts land (the 'inbound' location)."""
    return demo_root() / "source"


def onboarding_dir() -> Path:
    """Onboarding contracts + diagnostics, one directory per portfolio."""
    return demo_root() / "onboarding"


def runs_dir() -> Path:
    """Per-portfolio-per-period pipeline output (canonical, validation, lineage)."""
    return demo_root() / "runs"


def local_blob_root() -> Path:
    """``TRAKT_LOCAL_BLOB_ROOT`` — where ``blob://`` URIs resolve on disk.

    The demo publishes the assembled platform canonicals into the SAME dated
    layout production uses, so the MI Agent resolves them through the normal
    storage abstraction with no code change.
    """
    return demo_root() / "store"


def platform_prefix() -> str:
    """``blob://`` prefix for this client's dated platform canonicals."""
    return f"blob://processed/platform/{CLIENT_ID}"


def deck_root() -> Path:
    """``MI_AGENT_DECK_ROOT`` layout: ``{client}/latest|{period}/investor_pack.pptx``."""
    return demo_root() / "decks"


def artefact_dir() -> Path:
    """Generated governed outputs shown in the closing scenes."""
    return demo_root() / "artefacts"


def export_dir() -> Path:
    """Machine-readable exports the Remotion project consumes."""
    return demo_root() / "export"


def video_fixture_dir() -> Path:
    """Where the exported fixtures are copied for the Remotion bundle."""
    return REPO_ROOT / "demo-video" / "public" / "fixtures"


def demo_aliases_dir() -> Path:
    """Client-specific alias overlay applied at Gate 1 for this demo only."""
    return Path(__file__).resolve().parent / "aliases"


def client_docs_dir() -> Path:
    """The fictional client's governing documents (a synthetic Schedule 8)."""
    return Path(__file__).resolve().parent / "docs"


def portfolio_registry_path() -> Path:
    """The demo client's OWN governed portfolio registry.

    Deliberately inside the demo workspace rather than ``config/client/``: a
    demonstration must not write shared production configuration. A real
    deployment points ``TRAKT_PORTFOLIO_REGISTRY`` at its own file in exactly
    this way, so the mechanism exercised here is the production one.
    """
    return onboarding_dir() / "portfolio_registry.yaml"


def demo_client_config() -> Path:
    """The demo client master config (isolated from production client configs)."""
    return Path(__file__).resolve().parent / "config" / "config_client_ALDERBRIDGE_DEMO.yaml"


def run_output_dir(portfolio_id: str, run_id: str) -> Path:
    return runs_dir() / portfolio_id / run_id / "out"


def run_validation_dir(portfolio_id: str, run_id: str) -> Path:
    return runs_dir() / portfolio_id / run_id / "out_validation"


def source_file(portfolio: PortfolioSpec, prd: PeriodSpec) -> Path:
    """The generated raw extract for one portfolio / period.

    The filename mirrors how a lender or servicer actually drops files: portfolio
    identifier + reporting period, with an explicit SYNTHETIC marker.
    """
    return source_dir() / prd.run_id / (
        f"{portfolio.display_id}_{prd.reporting_date.replace('-', '')}"
        f"_SYNTHETIC.csv"
    )


# --------------------------------------------------------------------------- #
# Environment for the MI Agent API (deterministic, local, no Azure/auth)
# --------------------------------------------------------------------------- #
def mi_env(*, period_role: str = "current") -> Dict[str, str]:
    """The environment that points the MI Agent API at the demo platform.

    Mirrors a production deployment exactly, but resolved through the filesystem
    storage backend: ``blob://`` URIs map under ``TRAKT_LOCAL_BLOB_ROOT``.
    """
    prd = period(period_role)
    return {
        # Filesystem storage backend — blob:// resolves under the demo store.
        "TRAKT_STORAGE_BACKEND": "file",
        "TRAKT_LOCAL_BLOB_ROOT": str(local_blob_root()),
        # The active (latest) consolidated dataset the dashboard + chat answer from.
        "MI_AGENT_PLATFORM_URI": f"{platform_prefix()}/latest",
        # The dated platform root, which gives multi-period evolution / bridge.
        "MI_AGENT_ONBOARDING_OUTPUT_ROOT": platform_prefix(),
        "MI_AGENT_CLIENT_ID": CLIENT_ID,
        # The governed per-portfolio metadata onboarding published — including
        # each book's ASSET CLASS, which is what lets MI compose the common
        # semantic core plus this asset's own metrics and dimensions.
        "TRAKT_PORTFOLIO_REGISTRY": str(portfolio_registry_path()),
        "MI_AGENT_RUN_ID": prd.reporting_date,
        "MI_AGENT_REPORTING_DATE": prd.reporting_date,
        # Investor decks, resolved locally (no signed blob URLs at render time).
        "MI_AGENT_DECK_ROOT": str(deck_root()),
        # The client's governing documents. The Schedule 8 concentration limits
        # here are an invented document; the production Schedule 8 extractor
        # parses it live, so the risk-monitoring output is genuinely computed.
        "MI_AGENT_CLIENT_DOCS_ROOT": str(client_docs_dir()),
        # Deterministic: never call an LLM parser during the demo run.
        "MI_AGENT_LLM_ENABLED": "0",
        "MI_AGENT_SCRATCH": str(demo_root() / "scratch"),
        # Local API surface: the Easy-Auth header guard is explicitly off for
        # fixture capture (there is no Entra ID tenant in the demo environment).
        # This is a local capture-time setting only; it is never deployed.
        "MI_AGENT_AUTH_ENABLED": "false",
    }


# --------------------------------------------------------------------------- #
# Film narrative (kept beside the data so captions cannot drift from the run)
# --------------------------------------------------------------------------- #
#: The two questions BOTH surfaces must answer, in order.
MI_QUESTIONS: Tuple[str, ...] = (
    "Summarise the portfolio.",
    "What has changed versus the prior month?",
)

#: The optional third Copilot action (exists in production: getLatestInvestorDeck).
COPILOT_ARTEFACT_QUESTION = "Give me the latest investor deck."

#: Prohibited strings. The render FAILS if any appears in a fixture or artefact.
PROHIBITED_STRINGS: Tuple[str, ...] = (
    "ERE",
    "Equity Release Europe",
    "ere_funding",
    "ere funding",
    "SYNTHETIC_ERE",
    "ERE_Portfolio",
)

#: Region labels the film is allowed to show (guards against a stray real label).
ALLOWED_REGIONS: Tuple[str, ...] = tuple(REGION_WEIGHTS)


def as_dict() -> Dict[str, object]:
    """The whole narrative configuration, serialised for the demo manifest."""
    return {
        "synthetic_banner": SYNTHETIC_BANNER,
        "client": {
            "client_id": CLIENT_ID,
            "display_name": CLIENT_DISPLAY_NAME,
            "short_name": CLIENT_SHORT_NAME,
        },
        "portfolios": [
            {
                "key": p.key,
                "source_portfolio_id": p.source_portfolio_id,
                "source_portfolio_type": p.source_portfolio_type,
                "display_id": p.display_id,
                "label": p.label,
                "short_label": p.short_label,
                "description": p.description,
                "schema_name": p.schema_name,
                "acquisition_date": p.acquisition_date,
                "seller_name": p.seller_name,
                "balance_sheet_status": p.balance_sheet_status,
                "status_detail": p.status_detail,
            }
            # ALL portfolios: the film's consolidation beat needs the sold deal too.
            for p in ALL_PORTFOLIOS
        ],
        "periods": [
            {
                "run_id": pr.run_id,
                "reporting_date": pr.reporting_date,
                "period": pr.period,
                "label": pr.label,
                "short_label": pr.short_label,
                "role": pr.role,
            }
            for pr in PERIODS
        ],
        "targets": [
            {
                "name": t.name, "target": t.target, "tolerance": t.tolerance,
                "unit": t.unit, "description": t.description,
            }
            for t in METRIC_TARGETS
        ],
        "primary_movement_region": PRIMARY_MOVEMENT_REGION,
        "loan_count_band": list(LOAN_COUNT_BAND),
        "random_seed": RANDOM_SEED,
    }

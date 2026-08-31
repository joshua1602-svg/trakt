"""Deterministic funded-portfolio snapshot + reporting-run discovery.

This module backs the React MI landing page. It is deliberately **deterministic**
and never touches the natural-language parser:

  * :func:`discover_snapshots` walks the local onboarding output root and reports
    the available portfolios and reporting runs (``mi_2025_10`` / ``mi_2025_11``),
    each with its funded loan count and current outstanding balance, so the UI's
    portfolio / reporting-date dropdowns are data-driven (only real runs appear).

  * :func:`compute_funded_snapshot` derives the landing-page KPI tiles (current
    funded balance, loans funded, weighted-average LTV / rate / age / months on
    book, average loan balance) and the month-on-month change versus the prior
    available run (loan-count / balance change, new / exited loans) straight from
    the prepared MI dataset and its dataset contract — not via ``run_mi_agent_query``.

The funded tape is period-scoped, so the snapshot inherently reflects the funded
universe (33 / 73 loans), never pipeline rows.
"""

from __future__ import annotations

import calendar
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from . import presentation as _presentation

from . import currency as currency_mod
from .funded_prep import prepare_funded_mi_dataset
from .mi_dataset_contract import build_dataset_contract
from trakt_core import perf as _perf

_CENTRAL_TAPE_NAME = "18_central_lender_tape.csv"

# A reporting-run directory name carries a YEAR_MONTH (mi_2025_10, 2025-11, …).
_RUN_RE = re.compile(r"(\d{4})[_\-.](\d{2})")
# Path components that are pipeline scaffolding, never a client identifier.
_NON_CLIENT_PARTS = {"output", "outputs", "runs", "onboarding", "central", "mi", ""}


# --------------------------------------------------------------------------- #
# Numeric helpers (deterministic, contract-aware)
# --------------------------------------------------------------------------- #
def _num(series: pd.Series) -> pd.Series:
    return coerce_numeric(series)


def _weighted_average(values: pd.Series, weights: pd.Series) -> Optional[float]:
    """Weight-by-balance average over rows where both value and weight are valid.

    Falls back to a simple mean when the weights sum to zero. Returns ``None``
    when there is no usable value at all.
    """
    v = _num(values)
    w = _num(weights)
    mask = v.notna() & w.notna()
    if not mask.any():
        return None
    vv, ww = v[mask], w[mask]
    total = float(ww.sum())
    if total <= 0:
        return float(vv.mean()) if not vv.empty else None
    return float((vv * ww).sum() / total)


def _simple_mean(values: pd.Series) -> Optional[float]:
    v = _num(values).dropna()
    return float(v.mean()) if not v.empty else None


def _balance_sum(df: pd.DataFrame, col: str = "current_outstanding_balance") -> float:
    if col in df.columns:
        return float(_num(df[col]).sum())
    return 0.0


# --------------------------------------------------------------------------- #
# Reporting date + run/client inference
# --------------------------------------------------------------------------- #
def _last_day_of_month(year: int, month: int) -> str:
    last = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-{last:02d}"


def infer_reporting_date(run_id: str, df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """The reporting (cut-off) date for a run: prefer the dataset's own column,
    otherwise parse ``mi_YYYY_MM`` style run ids to the month-end."""
    if df is not None:
        for col in ("reporting_date", "data_cut_off_date", "cut_off_date"):
            if col in df.columns:
                rd = pd.to_datetime(df[col], errors="coerce").dropna()
                if not rd.empty:
                    return rd.max().date().isoformat()
    m = _RUN_RE.search(run_id or "")
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return _last_day_of_month(year, month)
    return None


def _infer_client_and_run(tape: Path, root: Path) -> Tuple[Optional[str], Optional[str]]:
    """Infer ``(client_id, run_id)`` from a central-tape path under ``root``.

    Layout is ``.../<client_id>/<run_id>/[output/]central/18_central_lender_tape.csv``
    (with tolerated variants). The run dir is the nearest ancestor whose name
    carries a YEAR_MONTH; the client id is the nearest meaningful ancestor above it.
    """
    try:
        rel_parts = tape.relative_to(root).parts
    except ValueError:
        rel_parts = tape.parts
    parts = list(rel_parts[:-1])  # drop the filename
    run_idx = None
    for i in range(len(parts) - 1, -1, -1):
        if _RUN_RE.search(parts[i]):
            run_idx = i
            break
    if run_idx is None:
        return None, None
    run_id = parts[run_idx]
    client_id = None
    for j in range(run_idx - 1, -1, -1):
        if parts[j].lower() not in _NON_CLIENT_PARTS:
            client_id = parts[j]
            break
    return client_id, run_id


def resolve_tape_path(output_root: str | os.PathLike, client_id: str, run_id: str) -> Optional[Path]:
    """Find the promoted central lender tape for a specific client / run."""
    root = Path(output_root)
    candidates = [
        root / client_id / run_id / "output" / "central" / _CENTRAL_TAPE_NAME,
        root / client_id / run_id / "central" / _CENTRAL_TAPE_NAME,
        root / "runs" / client_id / "onboarding" / run_id / "central" / _CENTRAL_TAPE_NAME,
        root / run_id / "output" / "central" / _CENTRAL_TAPE_NAME,
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = sorted(root.glob(f"**/{run_id}/**/{_CENTRAL_TAPE_NAME}"))
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _portfolio_label(client_id: str) -> str:
    return str(client_id).upper()


def discover_snapshots(output_root: str | os.PathLike) -> Dict[str, Any]:
    """Discover available portfolios and reporting runs under ``output_root``.

    Returns ``{"portfolios": [{client_id, label, runs: [{run_id, reporting_date,
    loan_count, current_outstanding_balance}]}]}``. Runs are ordered oldest →
    newest by reporting date so the UI can default to the latest and resolve the
    prior run for month-on-month change. Unreadable tapes are skipped, never fatal.
    """
    root = Path(output_root)
    portfolios: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return {"portfolios": []}

    for tape in sorted(root.glob(f"**/{_CENTRAL_TAPE_NAME}")):
        client_id, run_id = _infer_client_and_run(tape, root)
        if not client_id or not run_id:
            continue
        try:
            df = pd.read_csv(tape, low_memory=False)
        except Exception:  # noqa: BLE001 - a bad tape must not break discovery
            continue
        run = {
            "run_id": run_id,
            "reporting_date": infer_reporting_date(run_id, df),
            "loan_count": int(len(df)),
            "current_outstanding_balance": round(_balance_sum(df), 2),
        }
        pf = portfolios.setdefault(
            client_id, {"client_id": client_id, "label": _portfolio_label(client_id), "runs": {}}
        )
        pf["runs"][run_id] = run

    out: List[Dict[str, Any]] = []
    for pf in portfolios.values():
        runs = sorted(
            pf["runs"].values(),
            key=lambda r: (r["reporting_date"] or "", r["run_id"]),
        )
        out.append({"client_id": pf["client_id"], "label": pf["label"], "runs": runs})
    out.sort(key=lambda p: p["client_id"])
    return {"portfolios": out}


def find_prior_run(snapshots: Dict[str, Any], client_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    """The previous available run for ``client_id`` before ``run_id`` (by date)."""
    for pf in snapshots.get("portfolios", []):
        if pf.get("client_id") != client_id:
            continue
        runs = pf.get("runs", [])
        idx = next((i for i, r in enumerate(runs) if r["run_id"] == run_id), None)
        if idx is None or idx == 0:
            return None
        return runs[idx - 1]
    return None


# --------------------------------------------------------------------------- #
# Snapshot KPI computation
# --------------------------------------------------------------------------- #
def _hint_scale(contract: Dict[str, Any], field: str) -> Optional[str]:
    return (contract.get("display_hints", {}) or {}).get(field, {}).get("scale")


def _to_points(value: Optional[float], scale: Optional[str]) -> Optional[float]:
    from mi_agent.mi_dataset_profile import to_display_points
    return to_display_points(value, scale)


def _fmt_gbp(value: Optional[float], *, signed: bool = False) -> str:
    # Name kept for call-site stability; the symbol is the request's resolved
    # currency (tape -> config -> GBP). KPI tiles use BN/MM/K suffixes.
    return currency_mod.format_money(value, signed=signed, suffixes=("BN", "MM", "K"))


def _fmt_pct_points(points: Optional[float], *, signed: bool = False) -> str:
    if points is None:
        return "—"
    sign = "+" if (signed and points >= 0) else ""
    return f"{sign}{points:.1f}%"


def _fmt_int(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "—"
    iv = int(round(value))
    return f"{iv:+d}" if signed else f"{iv:,d}"


def _fmt_decimal(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1f}"


# --------------------------------------------------------------------------- #
# MEASURE BASIS — stated, because these measures legitimately do not tie.
#
# A reader who divides "average loan balance" by "weighted average property
# value" expects to land on "weighted average current LTV". They do not, and on
# a real book the gap is large — 15.2 percentage points on the QA fixture. There
# are two independent reasons, and BOTH are correct behaviour:
#
#   1. WEIGHTING. Average loan balance is unweighted, one vote per loan.
#      Property value is balance-weighted, one vote per pound. Averages over
#      different populations do not divide into one another.
#
#   2. AVERAGE OF RATIOS vs RATIO OF AVERAGES. The LTV tile is the mean of each
#      loan's own LTV — the typical POUND's gearing. Dividing the two money
#      tiles gives the ratio of the aggregates — the BOOK's gearing. They are
#      different economic statements and differ by Jensen's inequality on any
#      book with dispersion.
#
# The fix is therefore NOT to redefine weighted average LTV so the tiles tie.
# It is to say what each measure is, on the measure, so no reasonable reader
# infers an algebraic relationship that was never claimed.
# --------------------------------------------------------------------------- #

#: Weighting bases, in the words a funder reads.
BASIS_UNWEIGHTED = "per loan, unweighted"
BASIS_BALANCE_WEIGHTED = "balance-weighted"
BASIS_RATIO_OF_AGGREGATES = "ratio of aggregates"
BASIS_COUNT_SHARE = "share of loans, unweighted"


def _kpi(kpi_id: str, label: str, value: str, *, fmt: str, raw: Optional[float],
         available: bool = True, delta: Optional[str] = None,
         delta_intent: Optional[str] = None, hint: Optional[str] = None,
         basis: Optional[str] = None, numerator: Optional[str] = None,
         denominator: Optional[str] = None) -> Dict[str, Any]:
    """One governed KPI tile.

    ``basis`` / ``numerator`` / ``denominator`` state HOW the measure was formed.
    They are part of the measure, not decoration: two tiles on one page with
    different weighting bases are only honest if each says which it used.
    """
    return {
        "id": kpi_id,
        "label": label,
        "value": value,
        "format": fmt,
        "raw": raw,
        "available": available,
        "delta": delta,
        "deltaIntent": delta_intent,
        "hint": hint,
        "basis": basis,
        "numerator": numerator,
        "denominator": denominator,
    }


def _has_values(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and _num(df[col]).notna().any()


def portfolio_risk_type(df: pd.DataFrame) -> str:
    """``"erm"`` (equity release / lifetime mortgage) vs ``"standard"`` amortising.

    ERM is detected from product/plan text; this deployment defaults to ERM."""
    for col in ("erm_product_type", "product", "product_type", "loan_plan"):
        if col in df.columns:
            v = df[col].astype(str).str.lower()
            if v.str.contains(
                    r"lifetime|equity[ -]?release|\berm\b|drawdown|lump sum|roll[ -]?up",
                    regex=True, na=False).any():
                return "erm"
    return "erm"


def _risk_tile(df: pd.DataFrame) -> Dict[str, Any]:
    """Portfolio-type-aware risk tile (replaces the duplicate loan-movement tile).

    ERM -> a current-NNEG exposure proxy (balance above current valuation); a
    standard book -> arrears balance. When the inputs are missing, a controlled
    "unavailable" tile lists exactly which fields are absent (never fabricated)."""
    if portfolio_risk_type(df) == "erm":
        bal_col, val_col = "current_outstanding_balance", "current_valuation_amount"
        missing = [c for c in (bal_col, val_col) if not _has_values(df, c)]
        if missing:
            return _kpi("nneg_risk", "NNEG exposure (current)", "Unavailable",
                        fmt="text", raw=None, available=False, delta_intent="neutral",
                        hint="missing inputs: " + ", ".join(missing))
        bal, val = _num(df[bal_col]), _num(df[val_col])
        mask = (bal > val) & bal.notna() & val.notna()
        nneg = float((bal[mask] - val[mask]).sum())
        cnt = int(mask.sum())
        return _kpi("nneg_risk", "NNEG exposure (current)", _fmt_gbp(nneg), fmt="gbp",
                    raw=round(nneg, 2),
                    delta_intent="negative" if nneg > 0 else "positive",
                    hint=f"{cnt} loan(s) with balance above current valuation")
    # standard amortising book -> arrears
    if not _has_values(df, "arrears_balance"):
        return _kpi("arrears_risk", "Arrears balance", "Unavailable", fmt="text",
                    raw=None, available=False, delta_intent="neutral",
                    hint="missing inputs: arrears_balance / days_in_arrears")
    tot = float(_num(df["arrears_balance"]).sum())
    return _kpi("arrears_risk", "Arrears balance", _fmt_gbp(tot), fmt="gbp",
                raw=round(tot, 2), delta_intent="negative" if tot > 0 else "neutral")


def _loan_ids(df: pd.DataFrame) -> set:
    if "loan_identifier" not in df.columns:
        return set()
    return set(df["loan_identifier"].astype(str).str.strip())


# --------------------------------------------------------------------------- #
# Point-in-time funded stratifications (balance / share by a dimension). Fills
# the Funded tab's gap: Pipeline and Forecast carry breakdowns, Funded did not.
# --------------------------------------------------------------------------- #
# The funded stratification catalogue. Deliberately CONCISE: each dimension is a
# distinct asset-relevant cut a lender/investor actually asks for, drawn from a
# canonical field with an existing governed derivation. Charts are not added to
# fill space, and a dimension a portfolio does not report is reported as
# unavailable (with the reason) rather than rendered blank — see
# ``_funded_stratifications``. The set is the same for every portfolio type:
# an acquired back book gets the same funded depth as a direct one.
_STRAT_DIMS = [
    ("ltv", "By LTV band"),
    ("age", "By borrower age"),
    ("region", "By region"),
    ("rate", "By rate band"),
    ("product", "By product"),
    ("vintage", "By origination vintage"),
    ("status", "By account status"),
    ("equity", "By protected equity"),
    # These three used to be computed inside the investor PPTX
    # (``mi_agent_pptx.mi_api._extra_stratifications``), which picked its own
    # source columns and — for ticket size — carried its own bin edges that
    # contradicted config/mi/buckets.yaml. They are governed dimensions in
    # config/mi/stratification_catalogue.yaml (broker_channel,
    # borrower_structure, balance_band); they were simply never declared here.
    # Declaring them gives the deck and the dashboard the same six bands from
    # the same engine, and removes the only second economic definition that was
    # living in a renderer.
    ("broker", "By broker / channel"),
    ("borrower_type", "By borrower type"),
    ("ticket", "By ticket size"),
]
_EQUITY_BINS = [0, 5, 10, 20, 30, 50, 101]
_EQUITY_LABELS = ["<5%", "5–10%", "10–20%", "20–30%", "30–50%", "50%+"]
#: The canonical rate-band column materialised by ``funded_prep`` from
#: ``config/mi/buckets.yaml``. This is the sole economic definition.
_INTEREST_RATE_BUCKET = "interest_rate_bucket"
#: The canonical ticket-size band column materialised by ``funded_prep``
#: from ``config/mi/buckets.yaml`` ``balance_band``. Sole definition.
_TICKET_BUCKET = "ticket_bucket"

#: Backwards-compatible fallback bands, used ONLY when a frame carries no
#: canonical bucket column (i.e. it never went through funded_prep).
_RATE_BINS = [0, 3, 4, 5, 6, 7, 8, 100]
_RATE_LABELS = ["<3%", "3–4%", "4–5%", "5–6%", "6–7%", "7–8%", "8%+"]


def _strat_series(df: pd.DataFrame, key: str, scope=None):
    """The per-row band/category label for a funded stratification dimension.
    LTV and age reuse the SAME bands as the cohort composition lens (one banding,
    no drift between the two views); region/product read categorical columns;
    rate is banded here (scale-aware fraction→points)."""
    from analytics_lib.numeric import coerce_numeric
    if key in ("ltv", "age"):
        from . import cohorts as _cohorts  # identical banding as the cohort lens
        # P0-1: the FUNDED book is stratified on CURRENT LTV — the same basis the
        # MI Query Agent and Copilot answer "balance by LTV band" from. The
        # cohort lens keeps its origination basis (a static pool is defined at
        # origination); only the column selection differs, never the banding.
        series, _header = _cohorts._dimension_series(
            df, key, "Y", ltv_basis=_cohorts.LTV_BASIS_CURRENT)
        return series
    if key == "region":
        return region_series(df, scope)
    if key == "product":
        for col in ("product_type", "product", "loan_product"):
            if col in df.columns and df[col].notna().any():
                return df[col].astype("string")
        return None
    if key == "vintage":
        from . import cohorts as _cohorts  # same vintage derivation as cohorts
        return _cohorts._vintage_series(df, "Y")
    if key == "status":
        for col in ("account_status", "loan_status", "performance_status"):
            if col in df.columns and df[col].notna().any():
                return df[col].astype("string")
        return None
    if key == "equity":
        # Protected equity: the banded percentage where reported, else the flag.
        if "protected_equity_percentage" in df.columns:
            pct = coerce_numeric(df["protected_equity_percentage"])
            if pct.notna().sum():
                # Scale-aware: a fraction (0.15) is read in points (15).
                points = pct.where(pct.abs() > 1.0, pct * 100.0)
                return pd.cut(points, _EQUITY_BINS, labels=_EQUITY_LABELS,
                              right=False).astype("string")
        if "protected_equity_flag" in df.columns and df["protected_equity_flag"].notna().any():
            return df["protected_equity_flag"].astype("string")
        return None
    if key == "broker":
        # ``funded_prep`` aliases the channel family onto ``origination_channel``
        # (its ``group`` dimension), so that is the canonical column; the rest
        # are accepted for a tape that never went through the prep.
        for col in ("origination_channel", "broker_channel", "broker_name", "broker"):
            if col in df.columns and df[col].notna().any():
                return df[col].astype("string")
        return None
    if key == "borrower_type":
        # Derived by ``funded_prep``. The second-borrower date of birth is the
        # explicit fallback for a tape that carries the fact but not the
        # derivation: a joint life has one, a single life does not.
        if "borrower_type" in df.columns and df["borrower_type"].notna().any():
            return df["borrower_type"].astype("string")
        for col in ("borrower_2_DOB", "borrower_2_dob", "second_borrower_dob",
                    "borrower_2_date_of_birth"):
            if col in df.columns:
                joint = df[col].notna() & (df[col].astype(str).str.strip() != "")
                return joint.map({True: "Joint", False: "Single"}).astype("string")
        return None
    if key == "ticket":
        # The canonical ``ticket_bucket`` (config/mi/buckets.yaml ``balance_band``,
        # materialised by funded_prep) is the SOLE definition of a ticket-size
        # band. The deck used to band this itself on edges that disagreed with
        # the registry — 250k and 400k boundaries the registry does not have.
        # There is no fallback banding here on purpose: a frame that never went
        # through the prep reports the dimension as unavailable rather than
        # inventing a second ladder.
        if _TICKET_BUCKET in df.columns:
            banded = df[_TICKET_BUCKET].astype("string")
            if banded.notna().any():
                return banded
        return None
    if key == "rate":
        # P0-3: the canonical ``interest_rate_bucket`` (config/mi/buckets.yaml,
        # materialised by funded_prep) is the SOLE economic definition of a rate
        # band, and is what the MI Query Agent and Copilot answer from. Consume
        # it whenever the prep produced it.
        if _INTEREST_RATE_BUCKET in df.columns:
            banded = df[_INTEREST_RATE_BUCKET].astype("string")
            if banded.notna().any():
                return banded
        # Explicit backwards-compatible fallback ONLY: a frame that never went
        # through funded_prep (so carries no canonical bucket) still gets bars
        # rather than an empty chart. These bands are not a second definition —
        # they exist so a non-prepared tape degrades visibly, not silently.
        if "current_interest_rate" in df.columns:
            r = coerce_numeric(df["current_interest_rate"])
            if r.notna().sum() == 0:
                return None
            points = r.where(r.abs() > 1.5, r * 100.0)  # fraction (0.05) -> points (5.0)
            return pd.cut(points, _RATE_BINS, labels=_RATE_LABELS,
                          right=False).astype("string")
        return None
    return None


#: Chart availability states. A blank chart area is never acceptable — every
#: dimension reports WHY it has no bars, so the UI can render a distinct,
#: explained state instead of an empty frame.
CHART_AVAILABLE = "available"              # populated and applicable
CHART_ALL_NULL = "present_but_all_null"    # column exists, no values in scope
CHART_NOT_SUPPLIED = "not_supplied"        # not provided for these portfolios
CHART_PARTIAL = "partially_available"      # only some portfolios in scope have it


#: Source columns each stratification dimension can be built from. Used ONLY to
#: tell "column absent" apart from "column present but empty" — the values still
#: come from :func:`_strat_series`.
_STRAT_SOURCE_COLUMNS: Dict[str, tuple] = {
    "ltv": ("current_loan_to_value", "ltv_bucket", "original_loan_to_value",
            "original_ltv_bucket"),
    "age": ("borrower_age", "borrower_age_bucket", "youngest_borrower_age",
            "youngest_borrower_age_bucket"),
    "region": ("canonical_region_detail", "canonical_region_reporting",
               "geographic_region_collateral", "geographic_region_obligor", "region"),
    "product": ("product_type", "product", "loan_product"),
    "rate": ("current_interest_rate",),
    "vintage": ("origination_date", "vintage_year"),
    "status": ("account_status", "loan_status", "performance_status"),
    "equity": ("protected_equity_percentage", "protected_equity_flag"),
    "broker": ("origination_channel", "broker_channel", "broker_name", "broker"),
    "borrower_type": ("borrower_type", "borrower_2_DOB", "borrower_2_dob",
                      "second_borrower_dob", "borrower_2_date_of_birth"),
    "ticket": ("ticket_bucket", "current_outstanding_balance"),
}


def _strat_columns_present(df: pd.DataFrame, key: str) -> bool:
    """True when the tape carries a source column for this dimension at all."""
    cols = set(getattr(df, "columns", []))
    return any(c in cols for c in _STRAT_SOURCE_COLUMNS.get(key, ()))


def region_series(df: pd.DataFrame, scope=None):
    """The governed region label per row, at the granularity the scope calls for.

    A single source portfolio renders its most granular governed value
    (``canonical_region_detail``) — London stays London. A GROUPED scope (Total /
    Direct / Acquired) renders the consolidated reporting taxonomy
    (``canonical_region_reporting``), because that is the only vocabulary in which
    separately-sourced books can be added together honestly.

    Falls back to the raw canonical fields when no taxonomy is configured, so a
    deployment without region harmonisation behaves exactly as before. Nothing
    here cases, renames or merges a region: the values were resolved once, at the
    canonical layer, and are read back as-is.
    """
    from engine import region_taxonomy as _region

    grouped = scope is not None and len(getattr(scope, "portfolio_ids", ()) or ()) > 1
    preferred = ([_region.FIELD_REPORTING, _region.FIELD_DETAIL] if grouped
                 else [_region.FIELD_DETAIL, _region.FIELD_REPORTING])
    for col in preferred + ["geographic_region_collateral",
                            "geographic_region_obligor", "region"]:
        if col in df.columns and df[col].notna().any():
            return df[col].astype("string")
    return None


def _strat_coverage(df: pd.DataFrame, key: str, scope) -> Dict[str, Any]:
    """Per-portfolio contribution for one stratification dimension.

    For a grouped context (Total / Direct / Acquired) this states which
    portfolios could supply the dimension and which could not — the difference
    between "this book does not report borrower age" and "the chart is broken".
    """
    if scope is None or len(getattr(scope, "portfolio_ids", ()) or ()) <= 1:
        return {}
    contributing, missing = [], []
    for pid in scope.portfolio_ids:
        try:
            part = df[df["source_portfolio_id"].astype(str).str.strip().str.casefold()
                      == pid.strip().casefold()] if "source_portfolio_id" in df.columns else df
            series = _strat_series(part, key, scope)
            if series is not None and series.notna().sum() > 0:
                contributing.append(pid)
            else:
                missing.append(pid)
        except Exception:  # noqa: BLE001 - coverage must never break the chart
            missing.append(pid)
    return {"contributingPortfolios": contributing, "missingPortfolios": missing}


def _funded_stratifications(df: pd.DataFrame, scope=None) -> List[Dict[str, Any]]:
    """Balance / loan-count / book-share (and WA LTV) per band for each dimension
    the funded tape can support. Never raises — a bad dimension is skipped.

    Every dimension in the catalogue is REPORTED, with an explicit availability
    state, rather than silently omitted: a chart that cannot be drawn must say
    why (not supplied for these portfolios / present but entirely null / only
    some portfolios in scope carry it) so the workspace never shows an
    unexplained blank."""
    balance_col = "current_outstanding_balance"
    if df is None or balance_col not in getattr(df, "columns", []):
        return []
    from analytics_lib.stratify import stratify as _stratify
    out: List[Dict[str, Any]] = []
    wm = ["current_loan_to_value"] if "current_loan_to_value" in df.columns else None
    for key, label in _STRAT_DIMS:
        entry: Dict[str, Any] = {"key": key, "label": label, "bars": []}
        try:
            series = _strat_series(df, key, scope)
            if series is None:
                # Distinguish "the tape does not carry this at all" from "the
                # column is there but empty for these portfolios" — the two mean
                # very different things to a reader, and neither is a blank box.
                present = _strat_columns_present(df, key)
                entry["availability"] = CHART_ALL_NULL if present else CHART_NOT_SUPPLIED
                dimension = label[3:] if label.startswith("By ") else label
                entry["reason"] = (
                    f"{dimension} is present in the tape but has no values for the "
                    "selected portfolios." if present else
                    f"{dimension} is not supplied for the selected portfolios.")
                out.append(entry)
                continue
            work = df.assign(__dim=series)
            if work["__dim"].notna().sum() == 0:
                entry["availability"] = CHART_ALL_NULL
                entry["reason"] = (f"{label[3:] if label.startswith('By ') else label} "
                                   "is present in the tape but has no values for "
                                   "the selected portfolios.")
                out.append(entry)
                continue
            tbl = _stratify(work, "__dim", balance_col=balance_col, weighted_metrics=wm)
            if tbl.empty:
                entry["availability"] = CHART_ALL_NULL
                entry["reason"] = "No rows contributed to this stratification."
                out.append(entry)
                continue
            bars = []
            for _, r in tbl.iterrows():
                bar = {
                    "label": str(r["__dim"]),
                    "balance": round(float(r["balance_sum"]), 2),
                    "count": int(r["loan_count"]),
                    "sharePct": round(float(r["balance_share"]) * 100.0, 1),
                }
                wl = r.get("current_loan_to_value_weighted_avg")
                if wl is not None and pd.notna(wl):
                    bar["waLtv"] = round(float(wl), 4)
                bars.append(bar)
            # SELECT by materiality, then ORDER for display. ``_stratify``
            # ranks by balance descending, so the top-12 cut keeps the bands
            # that matter; ``presentation.order_bars`` then sequences the
            # survivors the way a reader expects to see them — the governed
            # ladder in config/mi/buckets.yaml for a banded dimension,
            # alphabetical for a nominal one, unknown last.
            #
            # This ordering used to happen in the browser
            # (``lib/stratOrder.sortStratBars``) and not at all in the investor
            # pack, so the same LTV stratification read in band order on screen
            # and in balance order in the deck. It is decided once, here.
            entry["bars"] = _presentation.order_bars(bars[:12], dimension=key)
            entry["displayOrder"] = _presentation.DISPLAY_ORDER_GOVERNED
            entry["ordinal"] = _presentation.is_ordinal(
                key, [b["label"] for b in entry["bars"]])
            coverage = _strat_coverage(df, key, scope)
            entry.update(coverage)
            if coverage.get("missingPortfolios"):
                entry["availability"] = CHART_PARTIAL
                entry["reason"] = (
                    "Covers " + ", ".join(coverage["contributingPortfolios"])
                    + "; not supplied for " + ", ".join(coverage["missingPortfolios"]) + ".")
            else:
                entry["availability"] = CHART_AVAILABLE
            out.append(entry)
        except Exception:  # noqa: BLE001 - a stratification must never break the snapshot
            continue
    return out


# --------------------------------------------------------------------------- #
# Multi-dimensional cross-tabs
#
# Two governed band dimensions crossed on funded balance. This used to live
# inside the investor PPTX (``mi_agent_pptx.mi_api._matrix`` / ``_multidim``),
# which meant the deck owned an analytical definition the React product had no
# way to reach: the dashboard could not draw the same chart even if it wanted
# to, and nothing outside the renderer could check the grouping.
#
# It is a COMPOSITION, not a new calculation. The bands come from
# ``cohorts._dimension_series`` and the governed stratification series — the same
# bands every other funded chart uses — and the only arithmetic is a sum of
# ``current_outstanding_balance`` per cell, which is what ``stratify`` does for
# one dimension. Nothing new is derived, no threshold is introduced, and the
# cells reconcile to the funded total by construction.
# --------------------------------------------------------------------------- #

#: The pairs offered. Each is (key, label, x-dimension, y-dimension), where the
#: dimensions name governed stratification series. Adding a pair is a config
#: decision here, not renderer code.
MULTIDIM_PAIRS: tuple = (
    ("ltv_age", "Balance by LTV x borrower age", "ltv", "age"),
    ("ltv_borrower_type", "Balance by LTV x borrower type", "ltv", "borrower_type"),
    ("ltv_region", "Balance by LTV x region", "ltv", "region"),
)


def cross_tab(df: pd.DataFrame, x_dimension: str, y_dimension: str,
              scope=None) -> Optional[Dict[str, Any]]:
    """Funded balance summed across two governed band dimensions.

    Returns ``{xLabels, yLabels, matrix, points, total}`` or ``None`` when either
    dimension is unavailable. Axis labels are ordered by the SAME governed ladder
    the one-dimensional stratifications use, so an LTV axis reads low-to-high on
    both surfaces and in both chart types.
    """
    balance_col = "current_outstanding_balance"
    if df is None or balance_col not in getattr(df, "columns", []):
        return None
    x_series = _strat_series(df, x_dimension, scope)
    y_series = _strat_series(df, y_dimension, scope)
    if x_series is None or y_series is None:
        return None

    work = pd.DataFrame({
        "x": x_series.astype("string"),
        "y": y_series.astype("string"),
        "balance": coerce_numeric(df[balance_col]),
    }).dropna(subset=["x", "y"])
    if work.empty:
        return None

    x_labels = _presentation.order_categories(work["x"].dropna().unique(),
                                              dimension=x_dimension)
    y_labels = _presentation.order_categories(work["y"].dropna().unique(),
                                              dimension=y_dimension)
    xi = {label: i for i, label in enumerate(x_labels)}
    yi = {label: i for i, label in enumerate(y_labels)}

    matrix = [[0.0] * len(x_labels) for _ in y_labels]
    points: List[Dict[str, Any]] = []
    for (xv, yv), sub in work.groupby(["x", "y"], dropna=True):
        xk = _presentation.clean_label(xv)
        yk = _presentation.clean_label(yv)
        if xk not in xi or yk not in yi:
            continue
        value = round(float(sub["balance"].sum()), 2)
        matrix[yi[yk]][xi[xk]] = value
        points.append({"x": xi[xk], "y": yi[yk], "value": value,
                       "xLabel": xk, "yLabel": yk, "count": int(len(sub))})
    total = round(float(work["balance"].sum()), 2)
    return {"xLabels": x_labels, "yLabels": y_labels, "matrix": matrix,
            "points": points, "total": total,
            "xDimension": x_dimension, "yDimension": y_dimension,
            "measure": "current_outstanding_balance"}


def multidimensional(df: pd.DataFrame, scope=None) -> Dict[str, Any]:
    """Every governed cross-tab this book can support, keyed by pair.

    A pair whose dimensions the tape cannot supply is simply absent, so a caller
    renders what resolved rather than an empty frame.
    """
    out: Dict[str, Any] = {}
    for key, label, x_dim, y_dim in MULTIDIM_PAIRS:
        try:
            table = cross_tab(df, x_dim, y_dim, scope)
        except Exception:  # noqa: BLE001 - one pair must not break the rest
            continue
        if table:
            out[key] = {"label": label, **table}
    return out


@_perf.stage_fn("funded_snapshot_compute")
def compute_funded_snapshot(
    df: pd.DataFrame,
    semantics: dict,
    *,
    client_id: str,
    run_id: str,
    reporting_date: Optional[str] = None,
    prep_report: Optional[Dict[str, Any]] = None,
    prior_df: Optional[pd.DataFrame] = None,
    prior_run_id: Optional[str] = None,
    prior_reporting_date: Optional[str] = None,
    scope=None,
) -> Dict[str, Any]:
    """Deterministic funded-book snapshot for one reporting run.

    Returns KPI tiles, the month-on-month change versus ``prior_df`` (if any),
    and any business-facing warnings (missing data / partial result). All numbers
    are derived from the prepared dataset and the dataset contract — never the parser.

    ``df`` is expected to be ALREADY narrowed to the governed portfolio scope by
    the caller; ``scope`` is passed alongside so the per-dimension chart coverage
    can disclose which portfolios supplied each stratification.
    """
    # Resolve the display currency from this run's tape (falls back to GBP).
    currency_mod.resolve_and_set(df, client_id=client_id)
    contract = build_dataset_contract(df, semantics, prep_report)
    warnings: List[str] = []
    diagnostics: List[str] = []

    balance = _balance_sum(df)
    loan_count = int(len(df))
    bal_series = df.get("current_outstanding_balance", pd.Series(dtype=float))

    kpis: List[Dict[str, Any]] = []
    kpis.append(_kpi("balance", "Current funded balance", _fmt_gbp(balance),
                     fmt="gbp", raw=round(balance, 2)))
    kpis.append(_kpi("loans", "Loans funded", _fmt_int(loan_count), fmt="number",
                     raw=loan_count))

    # Weighted-average current LTV (weight by balance), contract-aware scaling.
    if _has_values(df, "current_loan_to_value"):
        wavg = _weighted_average(df["current_loan_to_value"], bal_series)
        pts = _to_points(wavg, _hint_scale(contract, "current_loan_to_value"))
        kpis.append(_kpi("wa_current_ltv", "Weighted avg current LTV",
                         _fmt_pct_points(pts), fmt="pct", raw=pts,
                         basis=BASIS_BALANCE_WEIGHTED,
                         numerator="Σ (loan LTV × balance)", denominator="Σ balance"))
    else:
        kpis.append(_kpi("wa_current_ltv", "Weighted avg current LTV", "—",
                         fmt="pct", raw=None, available=False,
                         hint="LTV inputs unavailable for this run"))
        warnings.append("Weighted average current LTV unavailable: LTV inputs missing.")

    # Weighted-average original LTV (optional).
    if _has_values(df, "original_loan_to_value"):
        wavg = _weighted_average(df["original_loan_to_value"], bal_series)
        pts = _to_points(wavg, _hint_scale(contract, "original_loan_to_value"))
        kpis.append(_kpi("wa_original_ltv", "Weighted avg original LTV",
                         _fmt_pct_points(pts), fmt="pct", raw=pts,
                         basis=BASIS_BALANCE_WEIGHTED,
                         numerator="Σ (loan original LTV × balance)",
                         denominator="Σ balance"))

    # Average loan balance.
    avg_balance = balance / loan_count if loan_count else None
    kpis.append(_kpi("avg_balance", "Average loan balance", _fmt_gbp(avg_balance),
                     fmt="gbp", raw=round(avg_balance, 2) if avg_balance is not None else None,
                     basis=BASIS_UNWEIGHTED,
                     numerator="Σ balance", denominator="loan count"))

    # Weighted-average current interest rate (optional).
    if _has_values(df, "current_interest_rate"):
        wavg = _weighted_average(df["current_interest_rate"], bal_series)
        pts = _to_points(wavg, _hint_scale(contract, "current_interest_rate"))
        kpis.append(_kpi("wa_rate", "Weighted avg interest rate",
                         _fmt_pct_points(pts), fmt="pct", raw=pts,
                         basis=BASIS_BALANCE_WEIGHTED,
                         numerator="Σ (loan rate × balance)", denominator="Σ balance"))

    # Weighted-average months on book (optional).
    if _has_values(df, "months_on_book"):
        wavg = _weighted_average(df["months_on_book"], bal_series)
        kpis.append(_kpi("wa_months_on_book", "Weighted avg months on book",
                         _fmt_decimal(wavg), fmt="number", raw=wavg,
                         basis=BASIS_BALANCE_WEIGHTED,
                         numerator="Σ (months on book × balance)",
                         denominator="Σ balance"))

    # Weighted-average youngest borrower age (optional).
    if _has_values(df, "youngest_borrower_age"):
        wavg = _weighted_average(df["youngest_borrower_age"], bal_series)
        kpis.append(_kpi("wa_age", "Weighted avg youngest age",
                         _fmt_decimal(wavg), fmt="number", raw=wavg,
                         basis=BASIS_BALANCE_WEIGHTED,
                         numerator="Σ (youngest age × balance)",
                         denominator="Σ balance"))

    # Single-borrower share (optional). ``borrower_type`` is the prepared
    # single/joint dimension (derived from second-applicant presence for ERM,
    # but any asset class that supplies the column gets the tile).
    if "borrower_type" in df.columns:
        btype = df["borrower_type"].astype(str).str.strip().str.lower()
        known = btype.isin(["single", "joint"])
        if known.any():
            single = int((btype == "single").sum())
            pct = single / int(known.sum()) * 100.0
            kpis.append(_kpi("pct_single_borrowers", "Single borrowers",
                             _fmt_pct_points(pct), fmt="pct", raw=round(pct, 1),
                             hint=f"{single:,d} of {int(known.sum()):,d} loans",
                             basis=BASIS_COUNT_SHARE,
                             numerator="loans with a single borrower",
                             denominator="loans whose borrower type is known"))

    # Balance-weighted average property value (optional). Uses the same current
    # valuation input as NNEG/LTV, so it generalises to any collateralised book.
    if _has_values(df, "current_valuation_amount"):
        wavg = _weighted_average(df["current_valuation_amount"], bal_series)
        kpis.append(_kpi("wa_property_value", "Weighted avg property value",
                         _fmt_gbp(wavg), fmt="gbp",
                         raw=round(wavg, 2) if wavg is not None else None,
                         hint="balance-weighted current valuation",
                         basis=BASIS_BALANCE_WEIGHTED,
                         numerator="Σ (valuation × balance)", denominator="Σ balance"))

    # AGGREGATE GEARING — the BOOK's LTV, as distinct from the typical pound's.
    # Σ balance / Σ valuation: the ratio of aggregates a reader gets by dividing
    # the two money tiles. It is surfaced under its OWN name rather than used to
    # redefine weighted average LTV, because the two answer different questions
    # and a funder may legitimately want either. Built from governed aggregates
    # already computed here; no new primitive.
    if _has_values(df, "current_valuation_amount"):
        val_total = float(_num(df["current_valuation_amount"]).sum())
        if val_total > 0:
            gearing = balance / val_total * 100.0
            kpis.append(_kpi("aggregate_gearing", "Aggregate gearing (book LTV)",
                             _fmt_pct_points(gearing), fmt="pct",
                             raw=round(gearing, 4),
                             basis=BASIS_RATIO_OF_AGGREGATES,
                             numerator="Σ balance", denominator="Σ valuation",
                             hint="the book's LTV, not the typical loan's"))

    # ---- month-on-month change vs the prior available run -------------------
    monthly_change: Optional[Dict[str, Any]] = None
    if prior_df is not None:
        prior_balance = _balance_sum(prior_df)
        prior_count = int(len(prior_df))
        loan_delta = loan_count - prior_count
        bal_delta = balance - prior_balance
        bal_delta_pct = (bal_delta / prior_balance * 100.0) if prior_balance else None

        cur_ids, prior_ids = _loan_ids(df), _loan_ids(prior_df)
        ids_identifiable = bool(cur_ids) and bool(prior_ids)
        new_loans = len(cur_ids - prior_ids) if ids_identifiable else None
        exited_loans = len(prior_ids - cur_ids) if ids_identifiable else None

        monthly_change = {
            "prior_run_id": prior_run_id,
            "prior_reporting_date": prior_reporting_date,
            "loan_count_change": loan_delta,
            "balance_change": round(bal_delta, 2),
            "balance_change_pct": round(bal_delta_pct, 2) if bal_delta_pct is not None else None,
            "new_loans": new_loans,
            "exited_loans": exited_loans,
            "loans_identifiable": ids_identifiable,
        }

        # Attach deltas to the headline tiles.
        kpis[0]["delta"] = _fmt_gbp(bal_delta, signed=True)
        kpis[0]["deltaIntent"] = "positive" if bal_delta >= 0 else "negative"
        kpis[0]["hint"] = (f"{_fmt_pct_points(bal_delta_pct, signed=True)} vs prior run"
                           if bal_delta_pct is not None else "vs prior run")
        kpis[1]["delta"] = _fmt_int(loan_delta, signed=True)
        kpis[1]["deltaIntent"] = "positive" if loan_delta >= 0 else "negative"
        kpis[1]["hint"] = f"vs {prior_reporting_date or prior_run_id}"

        kpis.append(_kpi("mom_loans", "Monthly change · loans", _fmt_int(loan_delta, signed=True),
                         fmt="number", raw=loan_delta,
                         delta_intent="positive" if loan_delta >= 0 else "negative",
                         hint=f"vs {prior_reporting_date or prior_run_id}"))
        kpis.append(_kpi("mom_balance", "Monthly change · balance", _fmt_gbp(bal_delta, signed=True),
                         fmt="gbp", raw=round(bal_delta, 2),
                         delta_intent="positive" if bal_delta >= 0 else "negative",
                         hint=_fmt_pct_points(bal_delta_pct, signed=True) if bal_delta_pct is not None else None))
        if ids_identifiable:
            # The net "Monthly change · loans" tile already conveys loan movement;
            # the old "New loans since prior run" duplicated it. Keep exited/redeemed
            # (genuinely distinct) and surface new-loans in the monthly_change block.
            kpis.append(_kpi("exited_loans", "Exited / redeemed loans", _fmt_int(exited_loans),
                             fmt="number", raw=exited_loans,
                             delta_intent="negative" if (exited_loans or 0) > 0 else "neutral"))
        else:
            diagnostics.append("Loan-level new/exited counts not identifiable "
                               "(no loan_identifier on one of the runs).")
    else:
        diagnostics.append("No prior reporting date available for this portfolio.")

    # Portfolio-type-aware risk tile (replaces the duplicate loan-movement tile):
    # ERM -> NNEG exposure; standard -> arrears; controlled "unavailable" otherwise.
    kpis.append(_risk_tile(df))

    # Surface genuinely-missing core dimensions as business warnings (not noise).
    for miss in (prep_report or {}).get("missing_dimensions", []) or []:
        if isinstance(miss, dict) and miss.get("reason") in (
            "no_values_after_preparation", "derivation_inputs_missing"
        ):
            diagnostics.append(f"{miss['dimension']}: {miss.get('detail', miss['reason'])}")

    return {
        "ok": True,
        "portfolio": {
            "client_id": client_id,
            "label": _portfolio_label(client_id),
            "run_id": run_id,
            "reporting_date": reporting_date,
        },
        "prior": (
            {"run_id": prior_run_id, "reporting_date": prior_reporting_date}
            if prior_df is not None else None
        ),
        "loan_count": loan_count,
        "current_outstanding_balance": round(balance, 2),
        # The governed reporting currency for this client, resolved above from
        # the approved client configuration. The browser DISPLAYS this; it never
        # decides it.
        "currencyCode": currency_mod.current_code(),
        "kpis": kpis,
        "stratifications": _funded_stratifications(df, scope),
        "monthly_change": monthly_change,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "datasetContract": contract,
    }


#: Per-run prepared-frame cache, keyed by ``path:mtime_ns:size`` so a re-published
#: tape (new mtime) is a fresh key and reloads, while an unchanged tape is served
#: without re-reading the CSV or re-running the MI prep. Bounded so a long monthly
#: history (evolution walks every run) cannot grow it without limit; insertion
#: order is FIFO-evicted. Mirrors the read-only contract of ``data_source._active``
#: (consumers must not mutate the returned frame in place — they copy first).
_PREPARED_RUN_CACHE: "OrderedDict[str, Tuple[pd.DataFrame, Dict[str, Any]]]" = OrderedDict()
_PREPARED_RUN_CACHE_MAX = 24  # ~2 years of monthly runs kept warm


def _prepared_run_key(path: Path) -> Optional[str]:
    try:
        st = path.stat()
    except OSError:
        return None
    return f"{path}:{st.st_mtime_ns}:{st.st_size}"


@_perf.stage_fn("load_prepared_run")
def load_prepared_run(tape_path: str | os.PathLike) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read a central lender tape and apply the funded MI preparation layer.

    Memoised by ``(path, mtime, size)``: the same tape (hit by ``/mi/snapshot``,
    ``/mi/cohorts``, ``/mi/geo``, forecast and each evolution period) is read and
    prepared once, not on every request. The cached frame is returned directly —
    callers treat it as read-only, exactly as they already do for the active
    ``get_dataframe()``.
    """
    path = Path(tape_path)
    key = _prepared_run_key(path)
    if key is not None:
        hit = _PREPARED_RUN_CACHE.get(key)
        if hit is not None:
            _PREPARED_RUN_CACHE.move_to_end(key)  # mark recently used
            return hit
    raw = pd.read_csv(path, low_memory=False)
    value = prepare_funded_mi_dataset(raw)
    if key is not None:
        _PREPARED_RUN_CACHE[key] = value
        _PREPARED_RUN_CACHE.move_to_end(key)
        while len(_PREPARED_RUN_CACHE) > _PREPARED_RUN_CACHE_MAX:
            _PREPARED_RUN_CACHE.popitem(last=False)  # evict oldest
    return value

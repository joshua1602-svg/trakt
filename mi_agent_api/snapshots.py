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
    """The client's governed name where one is declared, else the identifier.

    See :mod:`mi_agent_api.client_identity` — the name is read only from
    governed per-client sources, never derived from the tape.
    """
    from . import client_identity
    return client_identity.portfolio_label(client_id)


def _governed_client_name(client_id: str) -> Optional[str]:
    """The governed name alone, or ``None`` — so a surface can tell a NAME from
    an identifier without re-deriving one."""
    from . import client_identity
    return client_identity.governed_client_name(client_id)


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
            client_id,
            {"client_id": client_id, "label": _portfolio_label(client_id),
             "client_name": _governed_client_name(client_id), "runs": {}},
        )
        pf["runs"][run_id] = run

    out: List[Dict[str, Any]] = []
    for pf in portfolios.values():
        runs = sorted(
            pf["runs"].values(),
            key=lambda r: (r["reporting_date"] or "", r["run_id"]),
        )
        out.append({"client_id": pf["client_id"], "label": pf["label"],
                    "client_name": pf["client_name"], "runs": runs})
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


def _kpi(kpi_id: str, label: str, value: str, *, fmt: str, raw: Optional[float],
         available: bool = True, delta: Optional[str] = None,
         delta_intent: Optional[str] = None, hint: Optional[str] = None) -> Dict[str, Any]:
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
]
_EQUITY_BINS = [0, 5, 10, 20, 30, 50, 101]
_EQUITY_LABELS = ["<5%", "5–10%", "10–20%", "20–30%", "30–50%", "50%+"]
#: The canonical rate-band column materialised by ``funded_prep`` from
#: ``config/mi/buckets.yaml``. This is the sole economic definition.
_INTEREST_RATE_BUCKET = "interest_rate_bucket"

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
            entry["bars"] = bars[:12]
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
                         _fmt_pct_points(pts), fmt="pct", raw=pts))
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
                         _fmt_pct_points(pts), fmt="pct", raw=pts))

    # Average loan balance.
    avg_balance = balance / loan_count if loan_count else None
    kpis.append(_kpi("avg_balance", "Average loan balance", _fmt_gbp(avg_balance),
                     fmt="gbp", raw=round(avg_balance, 2) if avg_balance is not None else None))

    # Weighted-average current interest rate (optional).
    if _has_values(df, "current_interest_rate"):
        wavg = _weighted_average(df["current_interest_rate"], bal_series)
        pts = _to_points(wavg, _hint_scale(contract, "current_interest_rate"))
        kpis.append(_kpi("wa_rate", "Weighted avg interest rate",
                         _fmt_pct_points(pts), fmt="pct", raw=pts))

    # Weighted-average months on book (optional).
    if _has_values(df, "months_on_book"):
        wavg = _weighted_average(df["months_on_book"], bal_series)
        kpis.append(_kpi("wa_months_on_book", "Weighted avg months on book",
                         _fmt_decimal(wavg), fmt="number", raw=wavg))

    # Weighted-average youngest borrower age (optional).
    if _has_values(df, "youngest_borrower_age"):
        wavg = _weighted_average(df["youngest_borrower_age"], bal_series)
        kpis.append(_kpi("wa_age", "Weighted avg youngest age",
                         _fmt_decimal(wavg), fmt="number", raw=wavg))

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
                             hint=f"{single:,d} of {int(known.sum()):,d} loans"))

    # Balance-weighted average property value (optional). Uses the same current
    # valuation input as NNEG/LTV, so it generalises to any collateralised book.
    if _has_values(df, "current_valuation_amount"):
        wavg = _weighted_average(df["current_valuation_amount"], bal_series)
        kpis.append(_kpi("wa_property_value", "Weighted avg property value",
                         _fmt_gbp(wavg), fmt="gbp",
                         raw=round(wavg, 2) if wavg is not None else None,
                         hint="balance-weighted current valuation"))

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

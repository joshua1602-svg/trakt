"""mi_agent_api/cohorts.py — funded origination-vintage (static-pool) analysis.

Surfaces the per-vintage MI already derivable from the governed funded central
tape — balance, loan count, book share, and balance-weighted LTV / interest rate
/ months-on-book by origination year — using the shared cohort primitives
(:mod:`analytics_lib.cohort`) and the ``vintage_year`` / ``months_on_book`` fields
``funded_prep`` derives. Nothing is fabricated: redemption / completion /
performance curves are NOT computed in the MI path today, so this module does not
emit them. Each returned metric is present only when its source column exists, and
``metricsAvailable`` lists exactly what was computed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analytics_lib.dates import coerce_dates
from analytics_lib.numeric import coerce_numeric
from mi_agent.mi_dataset_profile import PERCENT_POINTS, percent_storage_scale

_BALANCE = "current_outstanding_balance"
_LTV = "current_loan_to_value"
_RATE = "current_interest_rate"
_MOB = "months_on_book"
_VINTAGE = "vintage_year"
_ORIG_DATE = "origination_date"

# Cohort dimensions (asset-class-agnostic). Each groups the static pool by a
# generic origination/risk attribute; metrics are identical across dimensions.
_AGE_BUCKET = "age_bucket"
_YOUNGEST_AGE = "youngest_borrower_age"
_ORIG_LTV_BUCKET = "original_ltv_bucket"
_LTV_BUCKET = "ltv_bucket"
_ORIG_LTV = "original_loan_to_value"
_ORIG_CHANNEL = "origination_channel"
_BROKER = "broker_channel"

_DIMENSION_LABELS = {
    "vintage": "Vintage", "age": "Borrower age",
    "ltv": "LTV band", "channel": "Origination channel",
}
# Fallback bands when the tape has no pre-bucketed column.
_AGE_BINS = [0, 60, 65, 70, 75, 80, 85, 200]
_AGE_LABELS = ["<60", "60–64", "65–69", "70–74", "75–79", "80–84", "85+"]
_LTV_BINS = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 10]
_LTV_LABELS = ["<20%", "20–30%", "30–40%", "40–50%", "50–60%", "60–70%", "70–80%", "80%+"]


def _ltv_as_fraction(series: pd.Series) -> pd.Series:
    """LTV as a fraction (0.55) regardless of tape convention (0.55 or 55)."""
    v = coerce_numeric(series)
    if percent_storage_scale(series) == PERCENT_POINTS:
        v = v / 100.0
    return v


def _has_values(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and coerce_numeric(df[col]).notna().any()


def _has_labels(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = df[col].astype("string").str.strip()
    return s.replace("", pd.NA).notna().any()


def _weighted_avg(values: pd.Series, weights: pd.Series) -> Optional[float]:
    v = coerce_numeric(values)
    w = coerce_numeric(weights)
    mask = v.notna() & w.notna()
    denom = float(w[mask].sum())
    if denom == 0:
        return None
    return round(float((v[mask] * w[mask]).sum() / denom), 4)


def _weighted_avg_pct(values: pd.Series, weights: pd.Series,
                      scale_from: Optional[pd.Series] = None) -> Optional[float]:
    """Balance-weighted average of a PERCENT column, normalised to a FRACTION
    (0.0955 == 9.55%). The funded tape stores LTV as a fraction but the interest
    rate in points (9.55), so a single ×100 formatter turned 9.55% into 955%.
    Detect the column's storage scale and emit a fraction so the UI's percent
    formatter renders every rate/LTV correctly regardless of tape convention.

    ``scale_from`` must be the WHOLE column whenever ``values`` is one cohort's
    slice of it. The storage scale belongs to the column, and deciding it per
    cohort scaled neighbouring rows of the same table differently — a small
    low-LTV band stored in points sits under the 1.5 median threshold, so it
    was read as a fraction and rendered 1.23 as 123%."""
    wavg = _weighted_avg(values, weights)
    if wavg is None:
        return None
    basis = scale_from if scale_from is not None else values
    if percent_storage_scale(basis) == PERCENT_POINTS:
        return round(wavg / 100.0, 6)
    return wavg


def _vintage_series(df: pd.DataFrame, grain: str = "Y") -> Optional[pd.Series]:
    """The origination-cohort label per row at ``grain`` (Y|Q|M). Parsed from
    ``origination_date`` (finer grains need the date); falls back to the derived
    ``vintage_year`` for year grain. None when neither exists.

    A finer grain (quarter / month) is useful for a YOUNG book where every loan
    shares one origination year — a single 'Y' bucket hides the seasoning that
    'Q'/'M' reveals."""
    g = (grain or "Y").upper()
    if _ORIG_DATE in df.columns:
        od = coerce_dates(df[_ORIG_DATE])
        if od.notna().any():
            if g == "Q":
                return (od.dt.year.astype("Int64").astype("string") + "-Q"
                        + od.dt.quarter.astype("Int64").astype("string"))
            if g == "M":
                return od.dt.strftime("%Y-%m").astype("string").where(od.notna())
            return od.dt.year.astype("Int64")
    if g == "Y" and _VINTAGE in df.columns and df[_VINTAGE].notna().any():
        return df[_VINTAGE]
    return None


def _dimension_series(df: pd.DataFrame, dimension: str,
                      grain: str) -> Tuple[Optional[pd.Series], str]:
    """The per-row cohort label for ``dimension``, and its column header.

    Prefers a pre-bucketed column derived by ``funded_prep`` (age_bucket,
    original_ltv_bucket …); falls back to banding the raw value. Returns
    ``(None, header)`` when the tape carries no source for the dimension."""
    header = _DIMENSION_LABELS.get(dimension, "Cohort")
    if dimension == "vintage":
        return _vintage_series(df, grain), header
    if dimension == "age":
        if _has_labels(df, _AGE_BUCKET):
            return df[_AGE_BUCKET].astype("string"), header
        if _has_values(df, _YOUNGEST_AGE):
            banded = pd.cut(coerce_numeric(df[_YOUNGEST_AGE]), _AGE_BINS,
                            labels=_AGE_LABELS, right=False)
            return banded.astype("string"), header
        return None, header
    if dimension == "ltv":
        for col in (_ORIG_LTV_BUCKET, _LTV_BUCKET):
            if _has_labels(df, col):
                return df[col].astype("string"), header
        for col in (_ORIG_LTV, _LTV):
            if _has_values(df, col):
                banded = pd.cut(_ltv_as_fraction(df[col]), _LTV_BINS,
                                labels=_LTV_LABELS, right=False)
                return banded.astype("string"), header
        return None, header
    if dimension == "channel":
        for col in (_ORIG_CHANNEL, _BROKER):
            if _has_labels(df, col):
                return df[col].astype("string"), header
        return None, header
    return None, header


def _available_dimensions(df: pd.DataFrame) -> List[str]:
    """Which cohort dimensions the tape can actually support (drives the UI
    selector so it never offers a lens with no data)."""
    out: List[str] = []
    for dim in ("vintage", "age", "ltv", "channel"):
        series, _ = _dimension_series(df, dim, "Y")
        if series is not None and series.notna().any():
            out.append(dim)
    return out


def cohort_analysis(df: pd.DataFrame, *, client_id: str = "",
                    portfolio_id: str = "",
                    reporting_date: Optional[str] = None,
                    grain: str = "Y", dimension: str = "vintage") -> Dict[str, Any]:
    """Static-pool cohort table for a funded run, grouped by ``dimension``
    (vintage | age | ltv | channel), at ``grain`` (Y|Q|M — vintage only).

    Asset-class-agnostic: the same generic metrics (balance / loan count / book
    share and balance-weighted LTV, interest rate and months-on-book) are shown
    for every dimension. ``available`` is False (with a ``reason``) when the tape
    carries no source for the chosen dimension — the UI then shows an honest
    'no computed cohort data' state rather than a fabricated one.
    """
    dimension = dimension if dimension in _DIMENSION_LABELS else "vintage"
    available_dims = _available_dimensions(df) if df is not None and len(df) else []
    base = {
        "dataset": "cohorts",
        "portfolioId": portfolio_id or client_id,
        "cohortBasis": _ORIG_DATE,
        "period": (grain or "Y").upper(),
        "reportingDate": reporting_date,
        "dimension": dimension,
        "dimensionLabel": _DIMENSION_LABELS[dimension],
        "availableDimensions": available_dims,
    }
    if df is None or len(df) == 0:
        return {**base, "available": False, "reason": "no funded rows for this run",
                "cohorts": [], "metricsAvailable": []}

    series, header = _dimension_series(df, dimension, grain)
    if series is None:
        return {**base, "available": False,
                "reason": f"no {header.lower()} field on the funded tape",
                "cohorts": [], "metricsAvailable": []}

    work = df.copy()
    work["_vintage"] = series
    has_balance = _BALANCE in work.columns
    balance = coerce_numeric(work[_BALANCE]) if has_balance else None
    total_balance = float(balance.sum()) if balance is not None else None

    metrics_available: List[str] = ["loanCount"]
    if has_balance:
        metrics_available.append("balance")
    if _LTV in work.columns:
        metrics_available.append("waLtv")
    if _RATE in work.columns:
        metrics_available.append("waRate")
    if _MOB in work.columns:
        metrics_available.append("waMonthsOnBook")

    cohorts: List[Dict[str, Any]] = []
    # Rows with a missing label go into an explicit "Unknown" bucket so the table
    # reconciles to the book total (never silently dropped).
    for value, sub in work.groupby(work["_vintage"].astype("object"), dropna=False):
        if pd.isna(value) or str(value).strip() in ("", "nan", "None"):
            label = "Unknown"
        elif dimension == "vintage":
            try:
                label = str(int(value))
            except (TypeError, ValueError):
                label = str(value)
        else:
            label = str(value)
        sub_balance = coerce_numeric(sub[_BALANCE]) if has_balance else None
        bal = float(sub_balance.sum()) if sub_balance is not None else None
        # ``cohort`` is the generic label; ``vintage`` kept as an alias for
        # backward compatibility with the vintage-only contract.
        row: Dict[str, Any] = {
            "cohort": label,
            "vintage": label,
            "loanCount": int(len(sub)),
        }
        if bal is not None:
            row["balance"] = round(bal, 2)
            row["sharePct"] = (round(bal / total_balance * 100, 2)
                               if total_balance else None)
        # Scale from the whole column, never this cohort's slice of it.
        if _LTV in sub.columns and sub_balance is not None:
            row["waLtv"] = _weighted_avg_pct(sub[_LTV], sub[_BALANCE], work[_LTV])
        if _RATE in sub.columns and sub_balance is not None:
            row["waRate"] = _weighted_avg_pct(sub[_RATE], sub[_BALANCE], work[_RATE])
        if _MOB in sub.columns and sub_balance is not None:
            row["waMonthsOnBook"] = _weighted_avg(sub[_MOB], sub[_BALANCE])
        cohorts.append(row)

    # Ordering: vintage chronological (lexicographic is chronological across
    # grains); age/LTV by the band's leading number; channel by balance desc.
    # The "Unknown" bucket always sinks to the end.
    def _key(r: Dict[str, Any]):
        label = str(r["cohort"])
        if label == "Unknown":
            return (2, 0.0, "")
        if dimension in ("age", "ltv"):
            m = re.search(r"\d+", label)
            return (0, float(m.group()) if m else 0.0, label)
        if dimension == "channel":
            return (0, -float(r.get("balance") or 0.0), label)
        return (0, 0.0, label)  # vintage — lexicographic

    cohorts.sort(key=_key)

    return {
        **base,
        "available": True,
        "totalBalance": (round(total_balance, 2) if total_balance is not None else None),
        "totalLoanCount": int(len(work)),
        "metricsAvailable": metrics_available,
        "cohorts": cohorts,
        "lineage": {
            "source": f"governed funded central lender tape (by {header.lower()})",
            "metric": "balance / loan count / book share and balance-weighted "
                      "LTV, interest rate and months-on-book",
            "note": "Point-in-time static-pool aggregates only. Redemption / "
                    "completion / performance curves are not computed in the MI "
                    "path and are not shown.",
        },
    }


# --------------------------------------------------------------------------- #
# Vintage formation and static-pool seasoning
#
# The distinction this section exists to enforce:
#
#   portfolio evolution  the whole book at each reporting date — 33 then 73.
#                        Already served by funded_evolution; NOT duplicated here.
#   vintage formation    loans ENTERING in each origination period — 33 then 40.
#   static pool          one vintage followed forward: how those 33 loans behave
#                        as they season, and which of them leave.
#
# funded_cohort_progression already tracks a SELECTED vintage across periods,
# but with no vintage selected it returns the whole book per period, which is
# portfolio evolution wearing a cohort label. Formation is what was missing:
# there was no surface on which November is 40 rather than 73.
#
# Cohort membership comes from `origination_date` — the funded book's policy
# completion / drawdown date (config/system/aliases_onboarding_lending.yaml
# maps "policy completion date" to it). It is a property of the loan, not of
# the reporting period, so a loan belongs to exactly one vintage for life.
# --------------------------------------------------------------------------- #
_LOAN_ID_CANDIDATES = ("loan_id", "loan_identifier", "loan_policy_number",
                       "account_number")


def loan_id_column(df: pd.DataFrame) -> Optional[str]:
    """The column identifying a loan across reporting periods, or None.

    Without one, a static pool cannot be built: survival and exits are defined
    by following the SAME loans forward, not by counting rows.
    """
    for col in _LOAN_ID_CANDIDATES:
        if col in getattr(df, "columns", []) and df[col].notna().any():
            return col
    return None


def _ids(df: pd.DataFrame, id_col: str) -> pd.Series:
    return df[id_col].astype("string").str.strip()


def cohort_entry_map(frames: List[Dict[str, Any]], grain: str = "M"
                     ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Map every loan id to its vintage, plus any late corrections observed.

    A loan's vintage is taken from the FIRST reporting period in which it
    appears, and kept. If a later snapshot restates its origination date into a
    different vintage the first assignment stands and the change is reported —
    silently re-basing a loan would move it between cohorts and make a static
    pool grow, which is the one thing a static pool may never do.
    """
    entry: Dict[str, str] = {}
    corrections: List[Dict[str, Any]] = []
    for fr in frames:
        df = fr.get("df")
        if df is None or not len(df):
            continue
        id_col = loan_id_column(df)
        labels = _vintage_series(df, grain)
        if id_col is None or labels is None:
            continue
        ids = _ids(df, id_col)
        for loan, label in zip(ids, labels.astype("string")):
            if loan is None or pd.isna(loan) or str(loan).strip() == "":
                continue
            if pd.isna(label):
                continue
            label = str(label)
            if loan not in entry:
                entry[loan] = label
            elif entry[loan] != label:
                corrections.append({
                    "loanId": loan, "assigned": entry[loan],
                    "restatedTo": label, "seenIn": fr.get("reporting_date"),
                })
    return entry, corrections


def cohort_formation(frames: List[Dict[str, Any]], *, grain: str = "M",
                     client_id: str = "", portfolio_id: str = ""
                     ) -> Dict[str, Any]:
    """Vintage FORMATION: what entered the book in each origination period.

    Each loan is counted once, in its own vintage, with the metrics observed in
    the first reporting period that carried it — so "original balance" is the
    balance at entry rather than a later amortised one. November is 40 here,
    never 73.
    """
    base = {
        "dataset": "cohort_formation",
        "portfolioId": portfolio_id or client_id,
        "cohortBasis": _ORIG_DATE,
        "grain": (grain or "M").upper(),
    }
    usable = [fr for fr in frames if fr.get("df") is not None and len(fr["df"])]
    if not usable:
        return {**base, "available": False, "reason": "no funded reporting periods",
                "vintages": []}
    first = usable[0]["df"]
    if loan_id_column(first) is None:
        return {**base, "available": False, "vintages": [],
                "reason": "the funded tape carries no loan identifier, so loans "
                          "cannot be followed between reporting periods"}
    if _vintage_series(first, grain) is None:
        return {**base, "available": False, "vintages": [],
                "reason": f"no {_ORIG_DATE} on the funded tape, so loans cannot "
                          "be assigned to an origination vintage"}

    entry, corrections = cohort_entry_map(frames, grain)
    seen: set = set()
    rows: Dict[str, Dict[str, Any]] = {}
    for fr in usable:
        df = fr["df"]
        id_col = loan_id_column(df)
        if id_col is None:
            continue
        ids = _ids(df, id_col)
        fresh = ~ids.isin(seen) & ids.notna()
        if not fresh.any():
            continue
        newly = df[fresh.values]
        new_ids = ids[fresh]
        seen.update(new_ids.dropna().tolist())
        # Group this period's new arrivals by the vintage they were assigned.
        vintages = new_ids.map(entry)
        for label, idx in vintages.groupby(vintages).groups.items():
            sub = newly.loc[[i for i in idx if i in newly.index]]
            if not len(sub):
                continue
            row = rows.setdefault(str(label), {
                "vintage": str(label), "originalLoanCount": 0,
                "originalBalance": 0.0, "firstSeen": fr.get("reporting_date"),
            })
            row["originalLoanCount"] += int(len(sub))
            if _BALANCE in sub.columns:
                row["originalBalance"] += float(coerce_numeric(sub[_BALANCE]).sum())
            row.setdefault("_ltv", []).append(sub)

    out: List[Dict[str, Any]] = []
    for label, row in rows.items():
        parts = row.pop("_ltv", [])
        entry_rows = pd.concat(parts) if parts else None
        if entry_rows is not None and _BALANCE in entry_rows.columns:
            w = entry_rows[_BALANCE]
            if _ORIG_LTV in entry_rows.columns:
                row["waOriginalLtv"] = _weighted_avg_pct(
                    entry_rows[_ORIG_LTV], w, entry_rows[_ORIG_LTV])
            if _LTV in entry_rows.columns:
                row["waEntryLtv"] = _weighted_avg_pct(
                    entry_rows[_LTV], w, entry_rows[_LTV])
            if _RATE in entry_rows.columns:
                row["waRate"] = _weighted_avg_pct(
                    entry_rows[_RATE], w, entry_rows[_RATE])
        row["originalBalance"] = round(row["originalBalance"], 2)
        out.append(row)
    out.sort(key=lambda r: r["vintage"])
    return {
        **base,
        "available": bool(out),
        "reason": None if out else "no loans could be assigned to a vintage",
        "vintages": out,
        "totalLoanCount": sum(r["originalLoanCount"] for r in out),
        "lateCorrections": corrections,
        "lineage": {
            "source": "governed funded reporting periods, by origination vintage",
            "metric": "loans and balance ENTERING the book in each vintage",
            "note": "Formation counts each loan once, in the period it was "
                    "originated — not the book outstanding at a reporting date. "
                    "Portfolio evolution is a separate view.",
        },
    }


def cohort_static_pool(frames: List[Dict[str, Any]], *, vintage: str,
                       grain: str = "M", client_id: str = "",
                       portfolio_id: str = "") -> Dict[str, Any]:
    """One vintage followed forward through reporting periods.

    The pool is fixed at formation: only loans assigned to ``vintage`` are ever
    counted, so the count can fall through redemption but can never rise. Any
    later arrival carrying this vintage is a late correction and is reported
    rather than admitted.
    """
    base = {
        "dataset": "cohort_static_pool",
        "portfolioId": portfolio_id or client_id,
        "cohortBasis": _ORIG_DATE,
        "vintage": vintage,
        "grain": (grain or "M").upper(),
    }
    usable = [fr for fr in frames if fr.get("df") is not None and len(fr["df"])]
    if not usable:
        return {**base, "available": False, "periods": [],
                "reason": "no funded reporting periods"}
    if loan_id_column(usable[0]["df"]) is None:
        return {**base, "available": False, "periods": [],
                "reason": "the funded tape carries no loan identifier, so a "
                          "static pool cannot be followed"}

    entry, _ = cohort_entry_map(frames, grain)
    members = {loan for loan, label in entry.items() if label == str(vintage)}
    if not members:
        return {**base, "available": False, "periods": [],
                "reason": f"no loans were originated in {vintage}"}

    periods: List[Dict[str, Any]] = []
    original_count: Optional[int] = None
    original_balance: Optional[float] = None
    prior_ids: Optional[set] = None
    for fr in usable:
        df = fr["df"]
        id_col = loan_id_column(df)
        if id_col is None:
            continue
        ids = _ids(df, id_col)
        present = ids.isin(members)
        if not present.any() and original_count is None:
            continue  # the vintage has not formed yet
        sub = df[present.values]
        here = set(ids[present].dropna().tolist())
        balance = (float(coerce_numeric(sub[_BALANCE]).sum())
                   if _BALANCE in sub.columns and len(sub) else 0.0)
        if original_count is None:
            original_count, original_balance = len(here), balance
        exits = sorted(prior_ids - here) if prior_ids is not None else []
        row: Dict[str, Any] = {
            "period": (fr.get("reporting_date") or fr.get("run_id") or "")[:7],
            "reportingDate": fr.get("reporting_date"),
            "monthsSinceEntry": _months_between(vintage, fr.get("reporting_date")),
            "survivingLoanCount": len(here),
            "currentBalance": round(balance, 2),
            "loanRetention": (round(len(here) / original_count, 4)
                              if original_count else None),
            "balanceRetention": (round(balance / original_balance, 4)
                                 if original_balance else None),
            "exitsInPeriod": len(exits),
            "cumulativeExits": original_count - len(here) if original_count else 0,
        }
        if len(sub) and _BALANCE in sub.columns:
            w = sub[_BALANCE]
            if _LTV in sub.columns:
                row["waLtv"] = _weighted_avg_pct(sub[_LTV], w, df[_LTV])
            if _RATE in sub.columns:
                row["waRate"] = _weighted_avg_pct(sub[_RATE], w, df[_RATE])
        periods.append(row)
        prior_ids = here

    return {
        **base,
        "available": bool(periods),
        "reason": None if periods else f"no reporting period contains {vintage}",
        "originalLoanCount": original_count,
        "originalBalance": (round(original_balance, 2)
                            if original_balance is not None else None),
        "periods": periods,
        "singlePeriod": len(periods) <= 1,
        "lineage": {
            "source": "governed funded reporting periods (fixed static pool)",
            "metric": "surviving loans, balance, retention and exits for one vintage",
            "note": "The pool is fixed at formation. A falling count is "
                    "redemption or exit; the count can never rise.",
        },
    }


def _months_between(vintage: str, reporting_date: Optional[str]) -> Optional[int]:
    """Whole months from the vintage period to a reporting date. None unless the
    vintage is month- or quarter-grained (a year label has no single start)."""
    if not reporting_date:
        return None
    try:
        rd = pd.Timestamp(reporting_date)
    except (ValueError, TypeError):
        return None
    label = str(vintage)
    try:
        if re.fullmatch(r"\d{4}-\d{2}", label):
            start = pd.Timestamp(label + "-01")
        elif re.fullmatch(r"\d{4}-Q[1-4]", label):
            start = pd.Timestamp(f"{label[:4]}-{(int(label[-1]) - 1) * 3 + 1:02d}-01")
        else:
            return None
    except (ValueError, TypeError):
        return None
    return (rd.year - start.year) * 12 + (rd.month - start.month)

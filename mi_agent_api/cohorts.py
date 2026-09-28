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

import calendar
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
                # to_period is vectorised; dt.strftime formats element by
                # element and was over half the cost of building a cohort.
                return (od.dt.to_period("M").astype("string")
                        .where(od.notna()))
            return od.dt.year.astype("Int64")
    if g == "Y" and _VINTAGE in df.columns and df[_VINTAGE].notna().any():
        return df[_VINTAGE]
    return None


#: Which LTV basis a caller wants banded. A static-pool cohort is defined AT
#: ORIGINATION, so the cohort lens keeps ``origination``. A funded-book
#: stratification describes the book AS IT STANDS, so it asks for ``current``
#: and reconciles with the MI Query Agent, which answers "balance by LTV band"
#: from the canonical ``ltv_bucket``. The bands themselves are unchanged in
#: either case — this selects the column, never the banding.
LTV_BASIS_ORIGINATION = "origination"
LTV_BASIS_CURRENT = "current"

#: (pre-bucketed column, raw column) precedence per basis.
_LTV_BASIS_COLUMNS = {
    LTV_BASIS_ORIGINATION: ((_ORIG_LTV_BUCKET, _LTV_BUCKET), (_ORIG_LTV, _LTV)),
    LTV_BASIS_CURRENT: ((_LTV_BUCKET, _ORIG_LTV_BUCKET), (_LTV, _ORIG_LTV)),
}


def _dimension_series(df: pd.DataFrame, dimension: str,
                      grain: str, *, ltv_basis: str = LTV_BASIS_ORIGINATION
                      ) -> Tuple[Optional[pd.Series], str]:
    """The per-row cohort label for ``dimension``, and its column header.

    Prefers a pre-bucketed column derived by ``funded_prep`` (age_bucket,
    original_ltv_bucket …); falls back to banding the raw value. Returns
    ``(None, header)`` when the tape carries no source for the dimension.

    ``ltv_basis`` selects WHICH LTV the ``ltv`` dimension bands. It defaults to
    ``origination`` so every existing cohort caller is unchanged; the funded
    stratification passes ``current``.
    """
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
        bucket_cols, raw_cols = _LTV_BASIS_COLUMNS.get(
            ltv_basis, _LTV_BASIS_COLUMNS[LTV_BASIS_ORIGINATION])
        for col in bucket_cols:
            if _has_labels(df, col):
                return df[col].astype("string"), header
        for col in raw_cols:
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
# ``loan_identifier`` first, matching the platform assembler's loan key and
# ``evolution._LOAN_ID_COLS``: the canonical key is the one every cut carries.
_LOAN_ID_CANDIDATES = ("loan_identifier", "unique_identifier", "loan_id",
                       "loan_policy_number", "account_number")


def loan_id_column(df: pd.DataFrame) -> Optional[str]:
    """The column identifying a loan across reporting periods, or None.

    Without one, a static pool cannot be built: survival and exits are defined
    by following the SAME loans forward, not by counting rows.
    """
    for col in _LOAN_ID_CANDIDATES:
        if col in getattr(df, "columns", []) and df[col].notna().any():
            return col
    return None


def series_id_column(frames: List[Dict[str, Any]]) -> Optional[str]:
    """The ONE loan-id column used for a whole run of reporting periods.

    Chosen from the first period that has one, then held. Choosing per period
    let the key switch column between cuts (a ``loan_id`` populated only from
    some month on), and a loan keyed differently on either side of the switch
    was counted as a new loan — every vintage formed before it doubled.
    """
    for fr in frames:
        df = fr.get("df")
        if df is not None and len(df):
            col = loan_id_column(df)
            if col is not None:
                return col
    return None


#: Every column that may carry a loan's identity in some cut. A regulatory-regime
#: cut (ESMA Annex 2) can key a loan on its exposure identifier while the lender
#: tape keys it on the loan reference, so the linking column is chosen per cut.
_LINK_CANDIDATES = _LOAN_ID_CANDIDATES + (
    "original_underlying_exposure_identifier", "underlying_exposure_identifier",
    "new_underlying_exposure_identifier")


def frame_id_columns(frames: List[Dict[str, Any]]
                     ) -> Tuple[List[Optional[str]], List[Dict[str, Any]]]:
    """The id column for EACH cut, chosen by what actually links, plus linkage.

    The first cut uses the series key (:func:`series_id_column`). Every later
    cut uses whichever candidate column shares the most ids with the loans
    already seen — so a cut that keys the same loans in a different column is
    followed, not read as a whole new book. When no column links, the series
    key is kept and the linkage block says so.

    Linkage, per cut: the column used and the share of the previous cut's
    loans found in this one. A share near zero between consecutive monthly
    cuts is not redemption; it means the cuts cannot be joined.
    """
    base = series_id_column(frames)
    cols: List[Optional[str]] = []
    linkage: List[Dict[str, Any]] = []
    seen: set = set()
    prior: Optional[set] = None
    for fr in frames:
        df = fr.get("df")
        if df is None or not len(df):
            cols.append(None)
            continue
        present = [c for c in _LINK_CANDIDATES
                   if c in df.columns and df[c].notna().any()]
        chosen = base if base in present else (present[0] if present else None)
        if seen and present:
            best = max(present, key=lambda c: len(set(_ids(df, c).dropna()) & seen))
            if len(set(_ids(df, best).dropna()) & seen) > len(
                    set(_ids(df, chosen).dropna()) & seen if chosen else set()):
                chosen = best
        cols.append(chosen)
        here = set(_ids(df, chosen).dropna()) if chosen else set()
        linkage.append({
            "reportingDate": fr.get("reporting_date"),
            "idColumn": chosen,
            "loans": len(here),
            "linkedFromPriorPct": (round(len(here & prior) / len(prior) * 100, 1)
                                   if prior else None),
        })
        seen |= here
        prior = here
    return cols, linkage


def _ids(df: pd.DataFrame, id_col: Optional[str]) -> pd.Series:
    """Loan ids as comparable strings, or all-missing when the column is absent.

    The same loan must key identically in every cut. A cut whose id column
    was read as float (one blank cell is enough) renders ``1000`` as
    ``"1000.0"``, so a trailing ``.0`` is dropped; blanks and null spellings
    are missing, never an id.
    """
    if id_col is None or id_col not in getattr(df, "columns", ()):
        return pd.Series(pd.NA, index=df.index, dtype="string")
    ids = (df[id_col].astype("string").str.strip()
           .str.replace(r"\.0+$", "", regex=True))
    return ids.mask(ids.isin(["", "nan", "NaN", "None", "<NA>"]))


def cohort_entry_map(frames: List[Dict[str, Any]], grain: str = "M",
                     id_cols: Optional[List[Optional[str]]] = None
                     ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Map every loan id to its vintage, plus any late corrections observed.

    A loan's vintage is taken from the FIRST reporting period in which it
    appears, and kept. If a later snapshot restates its origination date into a
    different vintage the first assignment stands and the change is reported —
    silently re-basing a loan would move it between cohorts and make a static
    pool grow, which is the one thing a static pool may never do.
    """
    # Vectorised: a row-wise loop over every loan in every period dominated the
    # cost of the whole surface. One concat + drop_duplicates does the same
    # work, and keeps the "first assignment wins" rule explicit.
    seen: List[pd.DataFrame] = []
    if id_cols is None:
        id_cols, _ = frame_id_columns(frames)
    for fr, id_col in zip(frames, id_cols):
        df = fr.get("df")
        if df is None or not len(df):
            continue
        labels = _vintage_series(df, grain)
        if id_col is None or id_col not in df.columns or labels is None:
            continue
        part = pd.DataFrame({
            "loan": _ids(df, id_col).to_numpy(),
            "label": labels.astype("string").to_numpy(),
        })
        part["seenIn"] = fr.get("reporting_date")
        seen.append(part[part["loan"].notna() & (part["loan"] != "")
                        & part["label"].notna()])
    if not seen:
        return {}, []

    observed = pd.concat(seen, ignore_index=True)
    first = observed.drop_duplicates(subset="loan", keep="first")
    entry: Dict[str, str] = dict(zip(first["loan"].astype(str),
                                     first["label"].astype(str)))
    assigned = observed["loan"].map(entry)
    restated = observed[observed["label"].astype(str) != assigned.astype(str)]
    corrections = [{
        "loanId": str(r.loan), "assigned": entry[str(r.loan)],
        "restatedTo": str(r.label), "seenIn": r.seenIn,
    } for r in restated.itertuples(index=False)]
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

    id_cols, linkage = frame_id_columns(usable)
    _entry, corrections = cohort_entry_map(usable, grain, id_cols)

    # Each vintage is measured in ONE reporting cut: the first that falls on or
    # after the vintage's formation end, i.e. once it has stopped admitting
    # loans. Membership is that cut's own origination dates, so no loan is
    # matched across cuts — a cut that re-keys its loan identifiers (the live
    # 2025-12 cut did) can no longer count a vintage twice. Accumulating "new"
    # identifiers across cuts is what doubled every 2025 vintage.
    labelled = []
    for fr, id_col in zip(usable, id_cols):
        labels = _vintage_series(fr["df"], grain)
        if labels is not None:
            labelled.append((fr, id_col, labels.astype("string")))
    all_vintages = sorted({str(v) for _fr, _c, lab in labelled
                           for v in lab.dropna().unique()})

    out: List[Dict[str, Any]] = []
    for label in all_vintages:
        end = _formation_end(label, grain)
        present = [(fr, c, lab) for fr, c, lab in labelled if (lab == label).any()]
        if not present:
            continue
        anchored = [t for t in present
                    if end and str(t[0].get("reporting_date") or "")[:10] >= end]
        fr, id_col, lab = anchored[0] if anchored else present[-1]
        df = fr["df"]
        sub = df[(lab == label).fillna(False).to_numpy()]
        if id_col is not None:
            ids = _ids(sub, id_col)
            sub = sub[~(ids.notna() & ids.duplicated()).to_numpy()]
        row: Dict[str, Any] = {
            "vintage": label,
            "originalLoanCount": int(len(sub)),
            "originalBalance": (round(float(coerce_numeric(sub[_BALANCE]).sum()), 2)
                                if _BALANCE in sub.columns else 0.0),
            "firstSeen": present[0][0].get("reporting_date"),
            "measuredAt": fr.get("reporting_date"),
            # Still admitting loans at the latest cut: the count can yet grow.
            "forming": not anchored,
        }
        if _BALANCE in sub.columns and len(sub):
            w = sub[_BALANCE]
            if _ORIG_LTV in sub.columns:
                row["waOriginalLtv"] = _weighted_avg_pct(sub[_ORIG_LTV], w, sub[_ORIG_LTV])
            if _LTV in sub.columns:
                row["waEntryLtv"] = _weighted_avg_pct(sub[_LTV], w, sub[_LTV])
            if _RATE in sub.columns:
                row["waRate"] = _weighted_avg_pct(sub[_RATE], w, sub[_RATE])
        out.append(row)
    out.sort(key=lambda r: r["vintage"])
    return {
        **base,
        "available": bool(out),
        "reason": None if out else "no loans could be assigned to a vintage",
        "vintages": out,
        "totalLoanCount": sum(r["originalLoanCount"] for r in out),
        "lateCorrections": corrections,
        # Which id column joined each cut, and how much of the prior cut it
        # found — the evidence that each loan was counted once.
        "idLinkage": linkage,
        "lineage": {
            "source": "governed funded reporting periods, by origination vintage",
            "metric": "loans and balance ENTERING the book in each vintage",
            "note": "Each vintage is counted in the first reporting cut after it "
                    "stops originating (measuredAt), from that cut's origination "
                    "dates — so each loan is counted once without matching "
                    "identifiers across cuts. Not the book outstanding at a "
                    "reporting date; portfolio evolution is a separate view.",
        },
    }


def _formation_end(vintage: str, grain: str) -> Optional[str]:
    """The ISO date on which a vintage stops taking new loans.

    A static pool is only a pool once its vintage is complete. Anchoring it to
    the first reporting period the vintage is SEEN in — while loans originated
    later in the same year or quarter have yet to arrive — lets the pool grow,
    which is the one thing a static pool may never do, and makes balance
    retention exceed 100% for a reason that has nothing to do with roll-up.
    """
    label = str(vintage or "").strip()
    g = (grain or "M").upper()
    try:
        if g == "Y":
            return f"{int(label[:4]):04d}-12-31"
        if g == "Q":  # "2025-Q3" / "2025Q3"
            year = int(label[:4])
            quarter = int(label.rstrip()[-1])
            month = quarter * 3
            return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
        year, month = int(label[:4]), int(label[5:7])
        return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
    except (ValueError, IndexError):
        return None


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

    id_cols, linkage = frame_id_columns(usable)
    # Membership comes from each cut's OWN origination dates, not from loan
    # ids carried across cuts: a cut that re-keys its identifiers still holds
    # the same loans, so the pool neither doubles nor "exits" wholesale.
    members_by_cut = []
    for fr, id_col in zip(usable, id_cols):
        labels = _vintage_series(fr["df"], grain)
        mask = ((labels.astype("string") == str(vintage)).fillna(False).to_numpy()
                if labels is not None else None)
        members_by_cut.append(mask)
    if not any(m is not None and m.any() for m in members_by_cut):
        return {**base, "available": False, "periods": [],
                "reason": f"no loans were originated in {vintage}"}

    # The pool is anchored once the vintage has STOPPED FORMING. Reporting
    # periods before that are shown, so the reader sees the vintage building,
    # but they are marked `forming` and carry no retention — retention against
    # a pool that is still admitting loans is not a survival rate.
    formation_end = _formation_end(vintage, grain)

    periods: List[Dict[str, Any]] = []
    original_count: Optional[int] = None
    original_balance: Optional[float] = None
    prior_ids: Optional[set] = None
    prior_count: Optional[int] = None
    for fr, id_col, mask in zip(usable, id_cols, members_by_cut):
        df = fr["df"]
        if mask is None:
            continue
        if not mask.any() and original_count is None:
            continue  # the vintage has not formed yet
        reporting_date = str(fr.get("reporting_date") or "")
        forming = bool(formation_end and reporting_date
                       and reporting_date[:10] < formation_end)
        sub = df[mask]
        ids = _ids(sub, id_col)
        sub = sub[~(ids.notna() & ids.duplicated()).to_numpy()]
        here = set(_ids(sub, id_col).dropna().tolist())
        count = int(len(sub))
        balance = (float(coerce_numeric(sub[_BALANCE]).sum())
                   if _BALANCE in sub.columns and len(sub) else 0.0)
        if original_count is None and not forming:
            original_count, original_balance = count, balance
        # Exits only mean something once the pool is fixed. They are named by
        # identifier where this cut links to the previous one; where it does
        # not (fewer than half the previous cut's ids found, e.g. a re-keyed
        # cut) the ids cannot say who left, so the fall in count is used and
        # the period says so.
        rekeyed = False
        if prior_count is None or forming:
            exits = 0
        elif prior_ids and len(prior_ids & here) >= 0.5 * len(prior_ids):
            exits = len(prior_ids - here)
        else:
            rekeyed = bool(prior_ids)
            exits = max(prior_count - count, 0)
        row: Dict[str, Any] = {
            "period": (fr.get("reporting_date") or fr.get("run_id") or "")[:7],
            "reportingDate": fr.get("reporting_date"),
            "monthsSinceEntry": _months_between(vintage, fr.get("reporting_date")),
            "survivingLoanCount": count,
            "currentBalance": round(balance, 2),
            "forming": forming,
            "loanRetention": (round(count / original_count, 4)
                              if original_count and not forming else None),
            "balanceRetention": (round(balance / original_balance, 4)
                                 if original_balance and not forming else None),
            "exitsInPeriod": exits,
            "cumulativeExits": (max(original_count - count, 0)
                                if original_count and not forming else 0),
            "idsRekeyed": rekeyed,
        }
        if len(sub) and _BALANCE in sub.columns:
            w = sub[_BALANCE]
            if _LTV in sub.columns:
                row["waLtv"] = _weighted_avg_pct(sub[_LTV], w, df[_LTV])
            if _RATE in sub.columns:
                row["waRate"] = _weighted_avg_pct(sub[_RATE], w, df[_RATE])
        periods.append(row)
        prior_ids, prior_count = here, count

    return {
        **base,
        "available": bool(periods),
        "reason": None if periods else f"no reporting period contains {vintage}",
        "formationEnd": formation_end,
        "poolAnchored": original_count is not None,
        "formingPeriods": sum(1 for r in periods if r.get("forming")),
        "idLinkage": linkage,
        "originalLoanCount": original_count,
        "originalBalance": (round(original_balance, 2)
                            if original_balance is not None else None),
        "periods": periods,
        "singlePeriod": len(periods) <= 1,
        "lineage": {
            "source": "governed funded reporting periods (fixed static pool)",
            "metric": "surviving loans, balance, retention and exits for one vintage",
            "note": "The pool is fixed once the vintage stops forming. From that "
                    "point a falling count is redemption or exit and the count "
                    "can never rise. Periods before formation completes are "
                    "marked `forming` and carry no retention — the vintage was "
                    "still admitting loans, so there is no survival rate to give.",
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

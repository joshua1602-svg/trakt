#!/usr/bin/env python3
"""Phase 3A — the measures the brief needs that nothing already computes.

Everything else in the brief reuses an existing governed output: pipeline and
completion movement come from ``movement_detail``, conversion from the funnel,
concentration from the concentration tests. Only three families are new, and
they are all read off the SAME prepared weekly frames, using the SAME columns
the existing charts use — so there is no second source of economic truth.

  ticket_size   mean AND median of ``current_outstanding_balance`` per case
  weighted_ltv  balance-weighted ``current_loan_to_value``, with coverage
  band_mix      share of balance by ``ticket_bucket`` / ``ltv_bucket``

Two deliberate refusals:

* the mean is never reported without the median. They answer different
  questions, and a mean that moves while the median does not is a tail effect —
  which is worth surfacing precisely BECAUSE the two disagree;

* the weighted LTV is never silently replaced by a straight average, and is
  reported with the share of balance it actually covers. A weighted average
  over 60% of the book is not the book's LTV, and the caller is given the
  coverage to decide.

Pure: no I/O, no configuration, no thresholds. Frames in, numbers out.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from analytics_lib.numeric import coerce_numeric

#: The governed columns. Identical to the ones the pipeline charts read.
CASE_KEY = "pipeline_case_identifier"
BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
TICKET_BAND = "ticket_bucket"
LTV_BAND = "ltv_bucket"

#: Bumped when any definition in this module changes.
METHODOLOGY_VERSION = "1"

_NULL_IDS = ("", "nan", "none", "nat", "null")


def _cases(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """One row per case, so a duplicated identifier cannot double-count a ticket.

    The balance is summed (matching the movement engine, so the two reconcile);
    the dimensions take the last value, which is the same convention the
    preparation layer applies.
    """
    if df is None or df.empty or CASE_KEY not in df.columns:
        return pd.DataFrame(columns=["_balance", "_ltv", TICKET_BAND, LTV_BAND])
    key = df[CASE_KEY].astype(str).str.strip()
    keep = ~key.str.lower().isin(_NULL_IDS)
    work = pd.DataFrame({
        "_case": key,
        "_balance": (coerce_numeric(df[BALANCE]).fillna(0.0)
                     if BALANCE in df.columns else 0.0),
        "_ltv": (coerce_numeric(df[LTV]) if LTV in df.columns else pd.NA),
        TICKET_BAND: (df[TICKET_BAND].astype(str).str.strip()
                      if TICKET_BAND in df.columns else ""),
        LTV_BAND: (df[LTV_BAND].astype(str).str.strip()
                   if LTV_BAND in df.columns else ""),
    })[keep.values]
    if work.empty:
        return work.drop(columns=["_case"], errors="ignore")
    return work.groupby("_case", sort=True).agg(
        _balance=("_balance", "sum"), _ltv=("_ltv", "last"),
        **{TICKET_BAND: (TICKET_BAND, "last"), LTV_BAND: (LTV_BAND, "last")})


def _pct_change(now: Optional[float], prior: Optional[float]) -> Optional[float]:
    """Percentage change, or None when the base is zero or missing — never an
    infinity dressed up as a number."""
    if now is None or prior in (None, 0) or pd.isna(prior) or pd.isna(now):
        return None
    return round((now - prior) / abs(prior) * 100, 1)


# --------------------------------------------------------------------------- #
# Ticket size
# --------------------------------------------------------------------------- #
def ticket_size(current: Optional[pd.DataFrame],
                prior: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Mean AND median ticket, both periods, both changes.

    Reported together, always. Neither substitutes for the other.
    """
    cur, pri = _cases(current), _cases(prior)

    def stats(f: pd.DataFrame) -> Dict[str, Optional[float]]:
        if f.empty:
            return {"mean": None, "median": None, "cases": 0}
        b = f["_balance"]
        return {"mean": round(float(b.mean()), 2),
                "median": round(float(b.median()), 2),
                "cases": int(len(f))}

    c, p = stats(cur), stats(pri)
    return {
        "current_average": c["mean"], "comparison_average": p["mean"],
        "average_change_pct": _pct_change(c["mean"], p["mean"]),
        "current_median": c["median"], "comparison_median": p["median"],
        "median_change_pct": _pct_change(c["median"], p["median"]),
        "current_case_count": c["cases"], "comparison_case_count": p["cases"],
        "methodology": {
            "basis": "balance per case, duplicate identifiers summed",
            "measure": BALANCE,
            "note": ("Mean and median are reported together: a mean that moves "
                     "while the median does not is a tail effect."),
            "version": METHODOLOGY_VERSION,
        },
    }


# --------------------------------------------------------------------------- #
# Weighted-average LTV
# --------------------------------------------------------------------------- #
def weighted_ltv(current: Optional[pd.DataFrame],
                 prior: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Balance-weighted LTV for both periods, with the coverage behind it.

    Coverage is the share of BALANCE carrying a usable LTV — not the share of
    cases. A handful of large cases with no LTV distorts the weighted average
    far more than the same number of small ones, so cases would be the
    flattering denominator and balance is the honest one.

    A higher LTV is treated as deterioration and a lower LTV as improvement;
    that direction is stated in the returned methodology rather than assumed.
    """
    def one(f: pd.DataFrame) -> Dict[str, Any]:
        if f.empty:
            return {"value": None, "coverage_pct": None,
                    "covered_balance": 0.0, "total_balance": 0.0,
                    "excluded_balance": 0.0, "excluded_cases": 0}
        bal = f["_balance"].astype("float64")
        ltv = pd.to_numeric(f["_ltv"], errors="coerce")
        usable = ltv.notna() & (bal > 0)
        total = float(bal.sum())
        covered = float(bal[usable].sum())
        value = (round(float((ltv[usable] * bal[usable]).sum() / covered), 4)
                 if covered > 0 else None)
        return {
            "value": value,
            "coverage_pct": (round(covered / total * 100, 1) if total else None),
            "covered_balance": round(covered, 2),
            "total_balance": round(total, 2),
            "excluded_balance": round(total - covered, 2),
            "excluded_cases": int((~usable).sum()),
        }

    c, p = one(_cases(current)), one(_cases(prior))
    change_pp = (round((c["value"] - p["value"]) * 100, 2)
                 if c["value"] is not None and p["value"] is not None else None)
    return {
        "current": c["value"], "comparison": p["value"],
        "change_pp": change_pp,
        "current_coverage_pct": c["coverage_pct"],
        "comparison_coverage_pct": p["coverage_pct"],
        "current_excluded_balance": c["excluded_balance"],
        "current_excluded_cases": c["excluded_cases"],
        "current_total_balance": c["total_balance"],
        "methodology": {
            "basis": "balance-weighted mean of current_loan_to_value",
            "weighting": BALANCE,
            "coverage_basis": "share of BALANCE carrying a usable LTV",
            "direction": ("A higher weighted-average LTV is treated as "
                          "deterioration; a lower one as improvement."),
            "version": METHODOLOGY_VERSION,
        },
    }


# --------------------------------------------------------------------------- #
# Band mix
# --------------------------------------------------------------------------- #
def band_mix(current: Optional[pd.DataFrame], prior: Optional[pd.DataFrame],
             band_column: str) -> Dict[str, Any]:
    """Share of BALANCE per band in both periods, and the shift.

    Balance rather than case count: a mix shift matters because exposure moved,
    and two £1m cases replacing twenty £50k ones is the same case count and a
    very different book. The basis is stated in the payload either way.

    Bands come from ``config/mi/buckets.yaml`` via the preparation layer — this
    module never defines a band, so the brief and the stratification charts can
    never disagree about what "500k-1m" means.
    """
    cur, pri = _cases(current), _cases(prior)

    def shares(f: pd.DataFrame) -> Dict[str, float]:
        if f.empty or band_column not in f.columns:
            return {}
        band = f[band_column].replace("", "Unknown").fillna("Unknown")
        total = float(f["_balance"].sum())
        if total <= 0:
            return {}
        grouped = f.groupby(band.values)["_balance"].sum()
        return {str(k): round(float(v) / total * 100, 2) for k, v in grouped.items()}

    c, p = shares(cur), shares(pri)
    bands = sorted(set(c) | set(p))
    rows: List[Dict[str, Any]] = []
    for b in bands:
        now, before = c.get(b, 0.0), p.get(b, 0.0)
        rows.append({"band": b, "current_share_pct": now,
                     "comparison_share_pct": before,
                     "share_change_pp": round(now - before, 2)})
    # Deterministic: largest absolute shift first, ties broken by band name.
    rows.sort(key=lambda r: (-abs(r["share_change_pp"]), r["band"]))
    return {
        "band_column": band_column,
        "basis": "share of pipeline balance",
        "bands": rows,
        "largest_shift": rows[0] if rows else None,
        "methodology": {
            "basis": "share of pipeline balance per band",
            "band_source": "config/mi/buckets.yaml, materialised at preparation",
            "note": ("Describes the observed shift only. No cause is inferred."),
            "version": METHODOLOGY_VERSION,
        },
    }


# --------------------------------------------------------------------------- #
# Data quality, from the same comparison frames
# --------------------------------------------------------------------------- #
def data_quality(current: Optional[pd.DataFrame], prior: Optional[pd.DataFrame],
                 movement_methodology: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Measures that govern whether the rest of the brief can be believed.

    Most come straight from the movement engine's own methodology block (it
    already counts reassignments, duplicates and unkeyed rows while doing the
    attribution) — recomputing them here would be a second implementation of
    the same count. Only the Unknown shares and the LTV coverage are derived
    here, from the same frames.
    """
    m = movement_methodology or {}
    cur = _cases(current)
    total_cases = int(len(cur))
    total_balance = float(cur["_balance"].sum()) if not cur.empty else 0.0

    def unknown_share(col: str) -> Optional[float]:
        if cur.empty or col not in cur.columns or total_balance <= 0:
            return None
        vals = cur[col].replace("", "Unknown").fillna("Unknown")
        bad = vals.astype(str).str.strip().str.lower().isin(("unknown", "nan", "none"))
        return round(float(cur.loc[bad.values, "_balance"].sum()) / total_balance * 100, 2)

    reassign = m.get("dimension_reassignments") or {}
    unmatched = m.get("unmatched_current") or {}
    dupes = (m.get("duplicate_case_identifiers") or {}).get("current", 0)

    def pct_of_cases(n: Any) -> Optional[float]:
        try:
            return round(float(n) / total_cases * 100, 2) if total_cases else None
        except (TypeError, ValueError):
            return None

    ltv_cov = weighted_ltv(current, prior)["current_coverage_pct"]

    return {
        "case_count": total_cases,
        "broker_reassignment_pct": pct_of_cases(reassign.get("broker_channel")),
        "region_reassignment_pct": pct_of_cases(
            reassign.get("geographic_region_obligor")),
        "unusable_case_id_count": unmatched.get("cases", 0),
        "unusable_case_id_balance": unmatched.get("amount", 0.0),
        "unusable_case_id_pct": pct_of_cases(unmatched.get("cases", 0)),
        "duplicate_case_id_count": dupes,
        "duplicate_case_id_pct": pct_of_cases(dupes),
        "unknown_broker_share_pct": unknown_share("broker_channel"),
        "unknown_region_share_pct": unknown_share("geographic_region_obligor"),
        "ltv_coverage_pct": ltv_cov,
        "methodology": {
            "basis": ("Reassignment, duplicate and unusable-identifier counts "
                      "are the movement engine's own, not a second count. "
                      "Unknown shares are by BALANCE."),
            "version": METHODOLOGY_VERSION,
        },
    }

"""mi_agent_api/movement_summary.py

Two governed, composite answers built ENTIRELY on top of the existing
deterministic services — no new metric definitions, no parallel data path:

  * :func:`portfolio_summary` — "summarise the portfolio": the current reporting
    period's headline position (loan count, funded balance, weighted-average
    current LTV, weighted-average interest rate, average youngest-borrower age)
    plus the largest regional exposures and the data cut-off date.

  * :func:`period_movement` — "what has changed versus the prior month": the
    month-on-month movement in those same metrics, the regional attribution of
    the balance movement, how much of the primary region's movement is new
    completions, and each source portfolio's contribution.

Both read from ``mi_agent_api.evolution``:

  * ``funded_evolution`` supplies the per-period metrics (``funded_balance``,
    ``loan_count``, ``wa_ltv``, ``wa_interest_rate``, ``avg_borrower_age``) —
    the same computation the Evolution tab and the temporal-compare route use;
  * ``funded_bridge`` supplies the regional attribution, whose per-category
    deltas sum EXACTLY to the net change, so the narrative reconciles to the book;
  * ``funded_frames`` supplies the prepared per-period frames used for the
    largest-exposure ranking and the completions test.

Why this module exists
----------------------
"What has changed versus the prior month?" is a first-class managed-service
question, and until now no single route answered it: ``_route_compare`` answers
one metric at a time and ``_route_bridge`` answers attribution only. Composing
them here means the React MI Agent and Microsoft 365 Copilot (which delegates to
the same ``/mi/query`` handler) return the identical governed answer, rather than
two surfaces each assembling their own.

Causality discipline
--------------------
The answer only attributes a regional movement to new lending when that is
actually evidenced: :func:`period_movement` measures the balance of loans whose
origination date falls inside the reporting month within the primary region, and
only uses completion wording when those loans account for the majority of that
region's increase. Otherwise it states the movement without asserting a cause.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import evolution as evolution_mod

#: Region columns, in the same preference order the funded bridge uses: the
#: readable analytics label first, then the regulatory code fields.
REGION_COLUMNS: Tuple[str, ...] = (
    "collateral_geography", "geographic_region_collateral", "geographic_region_obligor",
)

_BALANCE = "current_outstanding_balance"
_ORIGINATION = "origination_date"
_PORTFOLIO_ID = "source_portfolio_id"
_PORTFOLIO_LABEL = "source_portfolio_label"

#: A weighted-average LTV move smaller than this (in percentage points) is
#: described as "broadly stable" rather than as a rise or a fall.
LTV_STABILITY_POINTS = 0.6

#: An average-age move smaller than this (years) is described as "unchanged".
AGE_MATERIALITY_YEARS = 0.05

#: The primary region's increase must be at least this multiple of the
#: second-largest for the answer to say "primarily driven by".
PRIMARY_LEAD_MULTIPLE = 1.5

#: New completions must account for at least this share of the primary region's
#: increase before the answer attributes the movement to completions.
COMPLETIONS_ATTRIBUTION_SHARE = 0.5

#: How many regional exposures the summary names.
TOP_REGIONS = 3


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _region_column(df: Optional[pd.DataFrame]) -> Optional[str]:
    """The first region column actually present and populated in ``df``."""
    if df is None:
        return None
    for col in REGION_COLUMNS:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _ltv_to_points(value: Optional[float]) -> Optional[float]:
    """Normalise a weighted-average LTV to percentage points.

    ``funded_prep`` stores LTV as a 0..1 ratio, so the evolution series carries a
    fraction; a tape that already holds percentage points would carry >1. Convert
    on that basis rather than assuming either scale.
    """
    if value is None:
        return None
    v = float(value)
    return v * 100.0 if abs(v) <= 1.5 else v


def _metrics(period: Dict[str, Any]) -> Dict[str, Optional[float]]:
    m = period.get("metrics") or {}
    return {
        "funded_balance": m.get("funded_balance"),
        "loan_count": m.get("loan_count"),
        "wa_ltv_points": _ltv_to_points(m.get("wa_ltv")),
        "wa_interest_rate": m.get("wa_interest_rate"),
        "avg_borrower_age": m.get("avg_borrower_age"),
    }


def _delta(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    if current is None or prior is None:
        return None
    return float(current) - float(prior)


def _regional_exposure(df: pd.DataFrame, region_col: str, *, top_n: int = TOP_REGIONS
                       ) -> List[Dict[str, Any]]:
    """Balance and share by region, largest first."""
    bal = _num(df[_BALANCE]).fillna(0.0) if _BALANCE in df.columns else None
    if bal is None:
        return []
    total = float(bal.sum())
    grouped = (
        pd.DataFrame({"region": df[region_col].astype(str).str.strip(), "balance": bal})
        .replace({"region": {"": "Unattributed", "nan": "Unattributed",
                             "None": "Unattributed"}})
        .groupby("region", as_index=False)["balance"].sum()
        .sort_values("balance", ascending=False)
    )
    out: List[Dict[str, Any]] = []
    for _, row in grouped.head(top_n).iterrows():
        out.append({
            "region": str(row["region"]),
            "balance": round(float(row["balance"]), 2),
            "share": round(float(row["balance"]) / total, 6) if total else None,
        })
    return out


def _cohorts(df: Optional[pd.DataFrame]) -> List[Dict[str, str]]:
    """``[{id, label}]`` for every source portfolio present in the frame."""
    if df is None or _PORTFOLIO_ID not in df.columns:
        return []
    ids = df[_PORTFOLIO_ID].dropna().astype(str).str.strip()
    out: List[Dict[str, str]] = []
    for pid in sorted({p for p in ids.unique() if p and p.lower() != "nan"}):
        label = pid
        if _PORTFOLIO_LABEL in df.columns:
            vals = df.loc[ids == pid, _PORTFOLIO_LABEL].dropna()
            if not vals.empty:
                candidate = str(vals.iloc[0]).strip()
                if candidate and candidate.lower() not in ("nan", "none"):
                    label = candidate
        out.append({"id": pid, "label": label})
    return out


def _completions_balance(df: pd.DataFrame, reporting_date: str,
                         region: Optional[str], region_col: Optional[str]) -> Optional[float]:
    """Balance of loans originated INSIDE the reporting month (optionally in one
    region) — the evidence for attributing a movement to new completions."""
    if _ORIGINATION not in df.columns or _BALANCE not in df.columns:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(reporting_date or "")):
        return None
    month = str(reporting_date)[:7]
    orig = pd.to_datetime(df[_ORIGINATION], errors="coerce")
    in_month = orig.dt.strftime("%Y-%m") == month
    if region and region_col and region_col in df.columns:
        in_month = in_month & (df[region_col].astype(str).str.strip() == region)
    if not bool(in_month.any()):
        return 0.0
    return round(float(_num(df.loc[in_month, _BALANCE]).fillna(0.0).sum()), 2)


# --------------------------------------------------------------------------- #
# Portfolio summary (current period)
# --------------------------------------------------------------------------- #
def portfolio_summary(output_root, client_id: str, *,
                      to_run_id: Optional[str] = None,
                      lens_filters: Optional[Dict[str, str]] = None,
                      lens_label: str = "Total") -> Dict[str, Any]:
    """The current reporting period's headline position for one lens."""
    frames = evolution_mod.funded_frames(output_root, client_id, to_run_id)
    frames = [
        {**f, "df": evolution_mod._scope_frame_lens(f.get("df"), lens_filters)}
        for f in frames
    ]
    frames = [f for f in frames if f["df"] is not None and len(f["df"])]
    if not frames:
        return {"available": False, "lens": lens_label,
                "reason": "no governed reporting period is available for this scope"}

    evo = evolution_mod.assemble_funded_evolution(frames, client_id, to_run_id)
    periods = evo.get("periods") or []
    if not periods:
        return {"available": False, "lens": lens_label,
                "reason": "no governed reporting period is available for this scope"}

    current = periods[-1]
    df = frames[-1]["df"]
    region_col = _region_column(df)
    return {
        "available": True,
        "lens": lens_label,
        "period": current.get("period"),
        "reportingDate": current.get("reporting_date"),
        "metrics": _metrics(current),
        "regionColumn": region_col,
        "topRegions": _regional_exposure(df, region_col) if region_col else [],
        "cohorts": _cohorts(df),
        "cohortBalances": (
            {c["id"]: round(float(_num(
                df.loc[df[_PORTFOLIO_ID].astype(str).str.strip() == c["id"], _BALANCE]
            ).fillna(0.0).sum()), 2) for c in _cohorts(df)}
            if _PORTFOLIO_ID in df.columns and _BALANCE in df.columns else {}
        ),
        "periodCount": len(periods),
        "sourceFiles": [f.get("source") for f in frames],
    }


# --------------------------------------------------------------------------- #
# Period movement (current vs prior)
# --------------------------------------------------------------------------- #
def period_movement(output_root, client_id: str, *,
                    to_run_id: Optional[str] = None,
                    lens_filters: Optional[Dict[str, str]] = None,
                    lens_label: str = "Total") -> Dict[str, Any]:
    """Month-on-month movement across the governed metrics, with attribution."""
    frames = evolution_mod.funded_frames(output_root, client_id, to_run_id)
    scoped = []
    for f in frames:
        d = evolution_mod._scope_frame_lens(f.get("df"), lens_filters)
        if d is not None and len(d):
            scoped.append({**f, "df": d})
    if len(scoped) < 2:
        return {"available": False, "lens": lens_label,
                "reason": "at least two funded reporting periods are needed to "
                          "compare against the prior month"}

    evo = evolution_mod.assemble_funded_evolution(scoped, client_id, to_run_id)
    periods = evo.get("periods") or []
    if len(periods) < 2:
        return {"available": False, "lens": lens_label,
                "reason": "at least two funded reporting periods are needed to "
                          "compare against the prior month"}

    cur_p, pri_p = periods[-1], periods[-2]
    cur_m, pri_m = _metrics(cur_p), _metrics(pri_p)
    cur_df, pri_df = scoped[-1]["df"], scoped[-2]["df"]

    deltas = {
        "funded_balance": _delta(cur_m["funded_balance"], pri_m["funded_balance"]),
        "loan_count": _delta(cur_m["loan_count"], pri_m["loan_count"]),
        "wa_ltv_points": _delta(cur_m["wa_ltv_points"], pri_m["wa_ltv_points"]),
        "wa_interest_rate": _delta(cur_m["wa_interest_rate"], pri_m["wa_interest_rate"]),
        "avg_borrower_age": _delta(cur_m["avg_borrower_age"], pri_m["avg_borrower_age"]),
    }

    # -- Regional attribution: the existing bridge, prior -> current ------- #
    region_col = _region_column(cur_df)
    bridge = evolution_mod.funded_bridge(
        output_root, client_id, list(REGION_COLUMNS),
        start_period=pri_p.get("period"), to_run_id=to_run_id,
        lens_filters=lens_filters, lens_label=lens_label, top_n=24)
    contributions: List[Dict[str, Any]] = []
    if bridge.get("available"):
        contributions = [
            {"region": c["category"], "start": c["start"], "end": c["end"],
             "delta": c["delta"]}
            for c in bridge.get("contributions", [])
            if not c.get("isOther")
        ]
        contributions.sort(key=lambda c: c["delta"], reverse=True)

    primary: Optional[Dict[str, Any]] = None
    runner_up: Optional[Dict[str, Any]] = None
    positive = [c for c in contributions if c["delta"] > 0]
    if positive:
        primary = positive[0]
        runner_up = positive[1] if len(positive) > 1 else None

    total_move = deltas["funded_balance"]
    primary_share = None
    primary_lead = None
    if primary and total_move:
        primary_share = round(primary["delta"] / total_move, 6)
        if runner_up and runner_up["delta"]:
            primary_lead = round(primary["delta"] / runner_up["delta"], 4)

    # -- Completions evidence for the primary region ----------------------- #
    completions_total = _completions_balance(
        cur_df, cur_p.get("reporting_date"), None, region_col)
    completions_primary = (
        _completions_balance(cur_df, cur_p.get("reporting_date"),
                             primary["region"] if primary else None, region_col)
        if primary else None
    )
    completions_share_of_primary = None
    if completions_primary is not None and primary and primary["delta"]:
        completions_share_of_primary = round(completions_primary / primary["delta"], 6)
    completions_evidenced = bool(
        completions_share_of_primary is not None
        and completions_share_of_primary >= COMPLETIONS_ATTRIBUTION_SHARE
    )

    # -- Per-source-portfolio contribution -------------------------------- #
    cohort_moves: List[Dict[str, Any]] = []
    for cohort in _cohorts(cur_df):
        cid = cohort["id"]
        cur_bal = float(_num(cur_df.loc[
            cur_df[_PORTFOLIO_ID].astype(str).str.strip() == cid, _BALANCE]
        ).fillna(0.0).sum()) if _PORTFOLIO_ID in cur_df.columns else 0.0
        pri_bal = float(_num(pri_df.loc[
            pri_df[_PORTFOLIO_ID].astype(str).str.strip() == cid, _BALANCE]
        ).fillna(0.0).sum()) if _PORTFOLIO_ID in pri_df.columns else 0.0
        cohort_moves.append({
            "id": cid, "label": cohort["label"],
            "current": round(cur_bal, 2), "prior": round(pri_bal, 2),
            "delta": round(cur_bal - pri_bal, 2),
        })
    cohort_moves.sort(key=lambda c: c["delta"], reverse=True)

    # Reconciliation: cohort deltas and regional deltas must each sum to the
    # consolidated movement. Reported, not assumed — the UI shows the residual.
    cohort_sum = round(sum(c["delta"] for c in cohort_moves), 2)
    region_sum = round(sum(c["delta"] for c in contributions), 2)

    return {
        "available": True,
        "lens": lens_label,
        "currentPeriod": cur_p.get("period"),
        "priorPeriod": pri_p.get("period"),
        "currentReportingDate": cur_p.get("reporting_date"),
        "priorReportingDate": pri_p.get("reporting_date"),
        "current": cur_m,
        "prior": pri_m,
        "delta": deltas,
        "regionColumn": region_col,
        "regionContributions": contributions,
        "primaryRegion": primary,
        "runnerUpRegion": runner_up,
        "primaryRegionShare": primary_share,
        "primaryRegionLead": primary_lead,
        "primaryRegionIsDominant": bool(
            primary is not None
            and (primary_lead is None or primary_lead >= PRIMARY_LEAD_MULTIPLE)
        ),
        "completionsBalanceInMonth": completions_total,
        "completionsBalanceInPrimaryRegion": completions_primary,
        "completionsShareOfPrimaryRegion": completions_share_of_primary,
        "completionsEvidenced": completions_evidenced,
        "cohortMovements": cohort_moves,
        "reconciliation": {
            "consolidated_movement": None if total_move is None else round(total_move, 2),
            "sum_of_cohort_movements": cohort_sum,
            "sum_of_region_movements": region_sum,
            "cohort_residual": (None if total_move is None
                                else round(cohort_sum - total_move, 2)),
            "region_residual": (None if total_move is None
                                else round(region_sum - total_move, 2)),
        },
        "sourceFiles": [f.get("source") for f in scoped[-2:]],
        "topRegions": _regional_exposure(cur_df, region_col) if region_col else [],
    }


# --------------------------------------------------------------------------- #
# Narrative descriptors (wording rules, kept out of the formatting layer)
# --------------------------------------------------------------------------- #
def ltv_descriptor(delta_points: Optional[float]) -> str:
    """The LTV movement clause, INCLUDING its preposition, so the caller can
    write "Weighted-average LTV {clause} {value}" and get grammatical UK English
    for every case: "remained broadly stable at 43.2%", "rose to 44.1%"."""
    if delta_points is None:
        return "is not available at"
    if abs(delta_points) < LTV_STABILITY_POINTS:
        return "remained broadly stable at"
    return "rose to" if delta_points > 0 else "fell to"


def age_descriptor(delta_years: Optional[float]) -> str:
    """The borrower-age movement clause, including its preposition."""
    if delta_years is None:
        return "is not available at"
    if abs(delta_years) < AGE_MATERIALITY_YEARS:
        return "was unchanged at"
    if abs(delta_years) < 0.5:
        return "increased slightly to" if delta_years > 0 else "decreased slightly to"
    return "increased to" if delta_years > 0 else "decreased to"


def balance_descriptor(delta: Optional[float]) -> str:
    if delta is None:
        return "was not available"
    if delta > 0:
        return "increased"
    if delta < 0:
        return "decreased"
    return "was unchanged"

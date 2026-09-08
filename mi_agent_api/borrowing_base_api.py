"""mi_agent_api/borrowing_base_api.py

The HTTP-facing borrowing-base service. A thin resolution layer over
``mi_agent.borrowing_base`` — it finds the governed frames and the governed
concentration results, and hands them to the pure calculator. No arithmetic
lives here.

Two entry points, one calculation:

* :func:`compute_from_frames` — used by the concentration-test envelope, so the
  Eligibility & Concentrations tab gets the borrowing base and the Schedule 8
  results from ONE evaluation of ONE frame. They cannot disagree.
* :func:`compute_borrowing_base` — the standalone route, for a caller that
  wants the facility position without the concentration table.

Never 500s: a missing facility, an unreconciled population or an unreadable
configuration all come back as controlled states with the reason named.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from mi_agent.borrowing_base.eligibility import eligibility_available
from mi_agent.borrowing_base.service import evaluate, measure_definitions
from trakt_core import perf as _perf

logger = logging.getLogger("mi_agent_api.borrowing_base")


def _portfolio_scope(client_id: str, run_id: Optional[str], scope: Any,
                     facility) -> Dict[str, Any]:
    """What population the figures were measured over, for the receipt."""
    financing = dict(getattr(facility, "financing_portfolio", {}) or {})
    return {
        "client_id": client_id,
        "run_id": run_id,
        "requested_scope": (str(scope) if scope is not None else None),
        "financing_portfolio_selector": financing,
        "financing_portfolio_selector_is_whole_book": not any(
            financing.get(k) for k in
            ("source_portfolio_ids", "source_portfolio_types")),
    }


def compute_from_frames(
    df: Optional[pd.DataFrame],
    *,
    client_id: str,
    facility=None,
    reporting_date: Optional[str] = None,
    run_id: Optional[str] = None,
    scope: Any = None,
    concentration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The borrowing-base envelope for an already-resolved funded frame.

    ``concentration`` is a governed concentration-test envelope (the output of
    ``concentration_tests_api.compute_concentration_tests``). It supplies the
    Schedule 8 results the binding-limit outputs and the receipt cite; the
    borrowing base itself does not depend on it, and nothing is deducted from
    the base because of it.
    """
    from mi_agent.borrowing_base.config import load_facility

    try:
        facility = facility if facility is not None else load_facility(client_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("facility configuration unavailable for %s: %s",
                       client_id, exc)
        facility = None

    if facility is None:
        return {
            "available": False,
            "portfolioId": client_id,
            "reportingDate": reporting_date,
            "toRunId": run_id,
            "facility": None,
            "reason": ("No funding facility is configured for this portfolio. "
                       "Borrowing-base monitoring begins when an approved "
                       "facility configuration is recorded through OCC "
                       "onboarding."),
            "eligibilityDerived": bool(eligibility_available(df)),
            "measures": {},
            "measureDefinitions": measure_definitions(),
        }

    tests: List[Dict[str, Any]] = list((concentration or {}).get("tests") or [])
    snapshot = evaluate(
        df,
        client_id=client_id,
        facility=facility,
        concentration_results=tests,
        concentration_rule_version=str(
            (concentration or {}).get("configurationVersion") or ""),
        concentration_configuration_hash=str(
            (concentration or {}).get("configurationHash") or ""),
        concentration_library_version=str(
            (concentration or {}).get("libraryVersion") or ""),
        as_of_date=reporting_date or "",
        portfolio_scope=_portfolio_scope(client_id, run_id, scope, facility),
        eligibility_derivation=_eligibility_provenance(df),
        run_id=run_id or "",
        # A reconciliation failure must be SHOWN on this surface, not turned
        # into an exception that blanks the tab with no explanation. The
        # envelope carries `reconciles: false` and the failed invariants, and
        # the UI refuses to present the figures as governed.
        strict=False,
    )
    envelope = snapshot.to_dict()
    envelope.update({
        "portfolioId": client_id,
        "reportingDate": reporting_date,
        "toRunId": run_id,
        "eligibilityDerived": bool(eligibility_available(df)),
        "concentrationSource": (concentration or {}).get("source"),
        "concentrationTestCount": len(tests),
        "measureDefinitions": measure_definitions(),
    })
    return envelope


def _eligibility_provenance(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """What the frame itself can say about how eligibility was determined.

    The full derivation receipt is produced in the canonical preparation layer
    and travels with the preparation report; what survives onto the frame is
    the per-loan determination. This reads that back so the borrowing-base
    receipt names the facility and the reasons actually stamped on the loans,
    rather than restating configuration as if it were an observation.
    """
    from mi_agent.borrowing_base.models import (
        FIELD_ELIGIBILITY_REASON,
        FIELD_ELIGIBILITY_STATUS,
        FIELD_FACILITY_ID,
    )
    if df is None or FIELD_ELIGIBILITY_STATUS not in getattr(df, "columns", []):
        return {"applied": False,
                "reason": "The governed eligibility derivation has not run on "
                          "this frame."}
    status = df[FIELD_ELIGIBILITY_STATUS]
    reasons = (df[FIELD_ELIGIBILITY_REASON]
               if FIELD_ELIGIBILITY_REASON in df.columns else None)
    facilities = (df[FIELD_FACILITY_ID]
                  if FIELD_FACILITY_ID in df.columns else None)
    return {
        "applied": True,
        "derivation": "mi_agent.borrowing_base.eligibility.derive_eligibility",
        "status_counts": {str(k): int(v)
                          for k, v in status.value_counts(dropna=True).items()},
        "reason_counts": ({str(k): int(v)
                           for k, v in reasons.value_counts(dropna=True).items()}
                          if reasons is not None else {}),
        "facility_ids": (sorted({str(v) for v in facilities.dropna().unique()})
                         if facilities is not None else []),
        "rows": int(len(df)),
    }


#: Loan-level context shown on an eligibility drill-down, by governed field
#: role. The SAME roles the concentration drill-through discloses, so the two
#: surfaces show a loan the same way, plus the eligibility determination
#: itself. Nothing outside this list is exposed: a drill-down is a reason to
#: show WHY a loan was classified, not an excuse to ship the whole tape.
_DRILL_ROLES = ("loan_id", "region", "balance_current", "balance_original",
                "valuation_original", "borrower_id", "borrower_age_youngest")


def compute_eligibility_loans(output_root, client_id: str,
                              to_run_id: Optional[str], status: str, *,
                              scope=None, max_rows: int = 500) -> Dict[str, Any]:
    """The loans carrying one governed eligibility status. Never raises."""
    from mi_agent.borrowing_base.eligibility import (
        eligibility_available,
        status_mask,
    )
    from mi_agent.borrowing_base.models import (
        ELIGIBILITY_STATUSES,
        FIELD_ELIGIBILITY_REASON,
        FIELD_ELIGIBILITY_STATUS,
    )
    from mi_agent.borrowing_base.config import load_facility

    status = str(status or "").strip().upper()
    if status not in ELIGIBILITY_STATUSES:
        return {"available": False,
                "reason": f"'{status}' is not a governed eligibility status.",
                "rows": [], "columns": []}

    from . import concentration_tests_api as conc_mod
    try:
        df, _prior, reporting_date, _pd_, run_id = conc_mod._resolve_frames(
            output_root, client_id, to_run_id, scope=scope)
    except Exception as exc:  # noqa: BLE001 - never 500
        return {"available": False,
                "reason": f"The governed funded frames could not be resolved: {exc}",
                "rows": [], "columns": []}
    if df is None or getattr(df, "empty", True):
        return {"available": False,
                "reason": "No funded data for this period.",
                "rows": [], "columns": []}
    if not eligibility_available(df):
        return {"available": False,
                "reason": ("No governed eligibility determination exists for "
                           "this book, so there is nothing to drill into."),
                "rows": [], "columns": []}

    facility = load_facility(client_id)
    subset = df[status_mask(df, status, facility)]
    columns: List[str] = []
    try:
        from mi_agent.concentration_tests.library import load_library
        from mi_agent.concentration_tests.metrics import resolve_role_column
        lib = load_library()
        for role in _DRILL_ROLES:
            col = resolve_role_column(df, lib, role)
            if col and col not in columns:
                columns.append(col)
    except Exception:  # noqa: BLE001 - context is a nicety, not the answer
        pass
    for col in (FIELD_ELIGIBILITY_STATUS, FIELD_ELIGIBILITY_REASON):
        if col in subset.columns and col not in columns:
            columns.append(col)
    present = [c for c in columns if c in subset.columns]
    rows = subset[present].head(max_rows)
    return {
        "available": True,
        "status": status,
        "facilityId": getattr(facility, "facility_id", None),
        "columns": present,
        "rows": rows.where(rows.notna(), None).to_dict(orient="records"),
        "rowCount": int(len(subset)),
        "truncated": bool(len(subset) > max_rows),
        "reportingDate": reporting_date,
        "toRunId": run_id,
    }


@_perf.stage_fn("borrowing_base")
def compute_borrowing_base(output_root, client_id: str,
                           to_run_id: Optional[str] = None, *,
                           scope=None) -> Dict[str, Any]:
    """The standalone borrowing-base envelope. Never raises.

    Reads the block the concentration-test service already produced rather
    than resolving the run and recalculating: the standalone route and the
    Eligibility & Concentrations tab are then the same numbers by
    construction, not by two implementations agreeing.
    """
    from . import concentration_tests_api as conc_mod
    try:
        envelope = conc_mod.compute_concentration_tests(
            output_root, client_id, to_run_id, scope=scope)
    except Exception as exc:  # noqa: BLE001 - never 500
        logger.warning("borrowing-base resolution failed for %s: %s",
                       client_id, exc)
        return {"available": False, "portfolioId": client_id,
                "reason": f"The governed funded frames could not be resolved: {exc}",
                "measures": {}, "measureDefinitions": measure_definitions()}
    return envelope.get("borrowingBase") or {
        "available": False, "portfolioId": client_id,
        "reason": "The borrowing base could not be calculated.",
        "measures": {}, "measureDefinitions": measure_definitions()}


__all__ = ["compute_borrowing_base", "compute_from_frames",
           "compute_eligibility_loans"]

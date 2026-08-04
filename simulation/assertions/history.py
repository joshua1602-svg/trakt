"""simulation.assertions.history — governed historical funded-snapshot integration.

Runs the monthly canonical snapshots through the production
``snapshot.store.SnapshotStore`` (filesystem adapter — no Azure resource is
required) and verifies the properties the historical layer exists to guarantee:

* reporting dates are ordered, and every prior snapshot stays reachable;
* ``resolve_latest`` / ``resolve_as_of`` / ``resolve_range`` / ``resolve_compare``
  return the right cut, not the most recently uploaded one;
* client tenancy is respected — another client's history is invisible;
* the funding structure (portfolio / cohort) stays intact across the history;
* opening-to-closing reconciles across consecutive snapshots;
* a duplicate-period upload follows the production rule (idempotent for
  identical content, a conflict for different content) rather than silently
  overwriting;
* a restatement is visible as a restatement;
* no synthetic fallback dataset is silently selected in place of the real one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..models import STAGE_HISTORY, Assertion, FailureClass
from . import check, compare

_DEFECT = FailureClass.PRODUCTION_DEFECT


def assert_registration(registrations: List[Dict[str, Any]], truth: Dict[str, Any]
                        ) -> List[Assertion]:
    out: List[Assertion] = []
    dates = [r["reporting_date"] for r in registrations]
    out.append(check(STAGE_HISTORY, "every period registered",
                     len(registrations) == len(truth["reporting_dates"]),
                     expected=len(truth["reporting_dates"]),
                     actual=len(registrations), failure_class=_DEFECT))
    out.append(check(STAGE_HISTORY, "reporting dates ascending and complete",
                     dates == sorted(dates) == truth["reporting_dates"],
                     expected=truth["reporting_dates"], actual=dates,
                     failure_class=_DEFECT))
    ids = [r["snapshot_id"] for r in registrations]
    out.append(check(STAGE_HISTORY, "each period has a distinct snapshot id",
                     len(set(ids)) == len(ids), actual=len(set(ids)),
                     failure_class=_DEFECT))
    return out


def assert_resolution(store, case, truth: Dict[str, Any]) -> List[Assertion]:
    """Temporal resolution must return the cut the caller asked for."""
    from snapshot.model import SnapshotNotFoundError

    out: List[Assertion] = []
    dates = truth["reporting_dates"]

    latest = store.resolve_latest(case.client_id, route="mi")
    out.append(check(STAGE_HISTORY, "resolve_latest returns the final period",
                     latest.reporting_date == dates[-1], expected=dates[-1],
                     actual=latest.reporting_date, failure_class=_DEFECT))

    for date in dates:
        header = store.resolve_as_of(case.client_id, date, route="mi")
        out.append(check(STAGE_HISTORY, f"resolve_as_of {date}",
                         header.reporting_date == date, expected=date,
                         actual=header.reporting_date, failure_class=_DEFECT))

    window = store.resolve_range(case.client_id, dates[0], dates[-1], route="mi")
    out.append(check(STAGE_HISTORY, "resolve_range covers the whole history",
                     [h.reporting_date for h in window] == dates,
                     expected=dates, actual=[h.reporting_date for h in window],
                     failure_class=_DEFECT))

    baseline, current = store.resolve_compare(case.client_id, dates[0], dates[-1],
                                              route="mi")
    out.append(check(STAGE_HISTORY, "resolve_compare picks both endpoints",
                     (baseline.reporting_date, current.reporting_date)
                     == (dates[0], dates[-1]),
                     expected=(dates[0], dates[-1]),
                     actual=(baseline.reporting_date, current.reporting_date),
                     failure_class=_DEFECT))

    # A prior snapshot must remain LOADABLE, not merely listed.
    first = store.resolve_as_of(case.client_id, dates[0], route="mi")
    frame = store.load_loans(first.snapshot_id)
    out.append(check(STAGE_HISTORY, "the earliest snapshot is still loadable",
                     len(frame) == truth["periods"][0]["reported_row_count"],
                     expected=truth["periods"][0]["reported_row_count"],
                     actual=len(frame), failure_class=_DEFECT))

    # Tenancy: a client that registered nothing must resolve to nothing.
    other = f"{case.client_id}_not_a_tenant"
    try:
        store.resolve_latest(other, route="mi")
        refused = False
    except SnapshotNotFoundError:
        refused = True
    out.append(check(STAGE_HISTORY, "another client's history is not visible",
                     refused, detail="Tenancy is enforced by the store, not by "
                                     "the caller filtering afterwards.",
                     failure_class=_DEFECT))
    return out


def assert_reconciliation(store, case, truth: Dict[str, Any]) -> List[Assertion]:
    """Opening-to-closing must reconcile from the STORED snapshots.

    The balances are re-summed from what the store returns, so this checks the
    round-trip through registration and storage, not the generator's arithmetic.
    """
    out: List[Assertion] = []
    balance_col = "current_outstanding_balance"
    previous: Optional[float] = None
    for period in truth["periods"]:
        header = store.resolve_as_of(case.client_id, period["reporting_date"],
                                     route="mi")
        frame = store.load_loans(header.snapshot_id)
        stored = float(pd.to_numeric(frame.get(balance_col), errors="coerce")
                       .fillna(0.0).sum()) if balance_col in frame.columns else None
        out.append(compare(
            STAGE_HISTORY, f"{period['reporting_date']}: stored closing balance",
            stored, period["closing_balance"], failure_class=_DEFECT))
        if previous is not None:
            out.append(compare(
                STAGE_HISTORY,
                f"{period['reporting_date']}: opening equals prior closing",
                previous, period["opening_balance"], failure_class=_DEFECT))
        previous = stored

        # Scope: the funding structure must not drift across the history.
        if "source_portfolio_id" in frame.columns:
            ids = set(frame["source_portfolio_id"].dropna().astype(str))
            out.append(check(
                STAGE_HISTORY,
                f"{period['reporting_date']}: SPV / portfolio scope intact",
                ids == {case.portfolio_id}, expected={case.portfolio_id},
                actual=ids, failure_class=_DEFECT))
    return out


def assert_duplicate_period_policy(store, canonical, case, reporting_date: str
                                   ) -> List[Assertion]:
    """A re-upload of the same period must follow the production rule.

    Identical content re-registers idempotently; different content for the same
    logical slot is a conflict. Neither may silently overwrite the stored cut.
    """
    from snapshot.model import SnapshotConflictError

    from ..pipeline import register_period

    out: List[Assertion] = []
    repeat = register_period(store, canonical, case, reporting_date=reporting_date)
    out.append(check(
        STAGE_HISTORY, f"{reporting_date}: identical re-upload is idempotent",
        repeat["idempotent"] and not repeat["created"], actual=repeat,
        detail="Re-registering the same source for the same period must not "
               "mint a second snapshot.", failure_class=_DEFECT))

    conflicted = False
    try:
        register_period(store, canonical, case, reporting_date=reporting_date,
                        source_file_id=f"{case.case_id}:{reporting_date}:variant")
    except SnapshotConflictError:
        conflicted = True
    except Exception:  # noqa: BLE001 - any other error is not the policy
        conflicted = False
    out.append(check(
        STAGE_HISTORY,
        f"{reporting_date}: a different source for the same period conflicts",
        conflicted, detail="Two different files for one reporting period is a "
                           "governance event, not a silent overwrite.",
        failure_class=_DEFECT))
    return out


def assert_no_fallback_dataset(store, case, truth: Dict[str, Any]) -> List[Assertion]:
    """The history answered from must be THIS case's data and nothing else."""
    out: List[Assertion] = []
    headers = store.list_snapshots(case.client_id, route="mi")
    tagged = [h for h in headers
              if (h.metadata or {}).get("case_id") == case.case_id]
    out.append(check(
        STAGE_HISTORY, "every stored snapshot belongs to this case",
        len(tagged) == len(headers) and bool(headers),
        expected=len(headers), actual=len(tagged),
        detail="A snapshot without this case's tag would mean a fallback or "
               "demonstration dataset had been selected.",
        failure_class=_DEFECT))
    seeds = {(h.metadata or {}).get("seed") for h in headers}
    out.append(check(
        STAGE_HISTORY, "every stored snapshot carries this case's seed",
        seeds == {case.seed}, expected={case.seed}, actual=seeds,
        failure_class=_DEFECT))
    return out


def assert_restatement(truth: Dict[str, Any], case) -> List[Assertion]:
    """A declared restatement must be visible AS a restatement."""
    from ..models import EVOLVE_RESTATE

    declared = [e for e in case.schema_evolution if e.kind == EVOLVE_RESTATE]
    if not declared:
        return []
    restated = [p for p in truth["periods"] if abs(p["restatement"]) > 0]
    return [check(
        STAGE_HISTORY, "declared restatement is reported as a movement",
        bool(restated), expected="a non-zero restatement movement",
        actual=[p["reporting_date"] for p in restated],
        detail="A correction to a previously reported balance is disclosed as "
               "its own movement, never absorbed into interest or redemptions.",
        failure_class=FailureClass.SIMULATION_DEFECT)]


__all__ = ["assert_duplicate_period_policy", "assert_no_fallback_dataset",
           "assert_reconciliation", "assert_registration", "assert_resolution",
           "assert_restatement"]

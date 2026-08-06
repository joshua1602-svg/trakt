"""simulation.assertions.platform — SPV-level consolidation of funded sources.

Checks what ``engine/platform_assembler.py`` exists to guarantee when one
client delivers its book as more than one funded population inside ONE SPV:

* both deliveries appear in one platform view, at the same reporting date;
* the SPV balance and loan count equal direct plus acquired, measured against
  the INDEPENDENT expected truth rather than against the platform's own output;
* ``direct`` / ``acquired`` provenance survives assembly on the production
  provenance fields, so the governed scope resolver can still group on it;
* nothing is duplicated and nothing is omitted — the composite key
  ``source_portfolio_id + loan_identifier`` is unique, every input row is
  present exactly once, and no row appears that no input contributed;
* the assembled cut belongs to the reporting date it claims.

The expected figures come from :func:`simulation.reference_truth.combine_truths`,
which ADDS the per-source truths. Nothing here reads a platform figure to decide
what a platform figure should be.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from ..models import STAGE_ASSEMBLE, Assertion, FailureClass
from . import check, compare

_DEFECT = FailureClass.PRODUCTION_DEFECT

#: The provenance columns assembly must not disturb.
_PROVENANCE = ("source_portfolio_id", "source_portfolio_type")
_BALANCE = "current_outstanding_balance"
_LOAN_ID = "loan_identifier"


def _frame(path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _balance(frame: pd.DataFrame) -> float:
    if _BALANCE not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[_BALANCE], errors="coerce").fillna(0.0).sum())


def assert_assembly(case, truth: Dict[str, Any],
                    assembled: Dict[str, Any],
                    manifests: Dict[str, Any]) -> List[Assertion]:
    """Every SPV-level property, period by period."""
    out: List[Assertion] = []
    dates = truth["reporting_dates"]
    expected_ids = list(truth["source_portfolio_ids"])
    expected_types = dict(truth["source_portfolio_types"])

    out.append(check(
        STAGE_ASSEMBLE, "every reporting period was assembled",
        sorted(assembled) == sorted(dates), expected=dates,
        actual=sorted(assembled), failure_class=_DEFECT))

    for period in truth["periods"]:
        date = period["reporting_date"]
        path = assembled.get(date)
        if path is None:
            continue
        frame = _frame(path)
        tag = f"{date}: "

        # -- one SPV view containing BOTH deliveries ---------------------- #
        present = sorted(set(frame.get("source_portfolio_id", pd.Series(dtype=str))
                             .dropna().astype(str)))
        out.append(check(
            STAGE_ASSEMBLE, f"{tag}both funded sources are in one platform view",
            present == sorted(expected_ids), expected=sorted(expected_ids),
            actual=present,
            detail="A platform canonical missing a delivery is a silently "
                   "under-reported book, not a smaller one.",
            failure_class=_DEFECT))

        # -- SPV totals equal direct + acquired ---------------------------- #
        out.append(compare(
            STAGE_ASSEMBLE, f"{tag}SPV loan count equals the sum of its sources",
            len(frame), period["reported_row_count"], failure_class=_DEFECT))
        out.append(compare(
            STAGE_ASSEMBLE, f"{tag}SPV balance equals the sum of its sources",
            _balance(frame), period["closing_balance"], failure_class=_DEFECT))

        # -- provenance survives ------------------------------------------- #
        for column in _PROVENANCE:
            out.append(check(
                STAGE_ASSEMBLE, f"{tag}{column} survives assembly",
                column in frame.columns and not frame[column].isna().any(),
                actual=(int(frame[column].isna().sum())
                        if column in frame.columns else "column absent"),
                failure_class=_DEFECT))
        if set(_PROVENANCE) <= set(frame.columns):
            mapping = (frame[list(_PROVENANCE)].drop_duplicates()
                       .set_index("source_portfolio_id")["source_portfolio_type"]
                       .to_dict())
            out.append(check(
                STAGE_ASSEMBLE, f"{tag}each source keeps its direct/acquired type",
                {str(k): str(v) for k, v in mapping.items()} == expected_types,
                expected=expected_types, actual=mapping,
                detail="The governed scope resolver groups on this field; if "
                       "assembly rewrote it, 'direct' would stop meaning direct.",
                failure_class=_DEFECT))

            # -- per-source figures inside the assembled book -------------- #
            for pid, expected in (period.get("by_source") or {}).items():
                subset = frame[frame["source_portfolio_id"].astype(str) == pid]
                out.append(compare(
                    STAGE_ASSEMBLE, f"{tag}{pid} contributes its own loan count",
                    len(subset), expected["loan_count"], failure_class=_DEFECT))
                out.append(compare(
                    STAGE_ASSEMBLE, f"{tag}{pid} contributes its own balance",
                    _balance(subset), expected["closing_balance"],
                    failure_class=_DEFECT))

        # -- no duplication, no omission ----------------------------------- #
        if _LOAN_ID in frame.columns:
            composite = (frame["source_portfolio_id"].astype(str) + "|"
                         + frame[_LOAN_ID].astype(str))
            duplicated = composite[composite.duplicated()].tolist()
            out.append(check(
                STAGE_ASSEMBLE,
                f"{tag}the composite platform key is unique",
                not duplicated, actual=duplicated[:8],
                detail="source_portfolio_id + loan_identifier identifies a loan "
                       "inside the platform dataset; a duplicate is a "
                       "double-counted exposure.",
                failure_class=_DEFECT))
            out.append(check(
                STAGE_ASSEMBLE,
                f"{tag}loan identifiers stay distinct across sources",
                frame[_LOAN_ID].astype(str).is_unique,
                actual=int(frame[_LOAN_ID].astype(str).duplicated().sum()),
                failure_class=_DEFECT))

        # -- the cut belongs to the date it claims -------------------------- #
        if "reporting_date" in frame.columns:
            reported = sorted(set(frame["reporting_date"].dropna().astype(str)))
            out.append(check(
                STAGE_ASSEMBLE, f"{tag}every assembled row carries this cut's date",
                reported == [date], expected=[date], actual=reported,
                detail="Assembly selects the latest snapshot per source; a "
                       "mixed date would mean two different points in time were "
                       "consolidated into one view.",
                failure_class=_DEFECT))

    out += _assert_manifest(case, truth, manifests)
    return out


def _assert_manifest(case, truth: Dict[str, Any],
                     manifests: Dict[str, Any]) -> List[Assertion]:
    """The assembler's own lineage manifest must describe what it did."""
    out: List[Assertion] = []
    dates = truth["reporting_dates"]
    out.append(check(
        STAGE_ASSEMBLE, "the assembler wrote a lineage manifest per period",
        sorted(manifests) == sorted(dates), expected=dates,
        actual=sorted(manifests), failure_class=_DEFECT))

    for date, manifest in sorted(manifests.items()):
        out.append(check(
            STAGE_ASSEMBLE, f"{date}: the manifest names this run deterministically",
            manifest.get("assembler_run_id") == f"asm_{case.case_id}_{date}",
            expected=f"asm_{case.case_id}_{date}",
            actual=manifest.get("assembler_run_id"),
            detail="A wall-clock run id would make the assembled evidence "
                   "irreproducible.",
            failure_class=FailureClass.SIMULATION_DEFECT))
        out.append(check(
            STAGE_ASSEMBLE, f"{date}: the manifest records the client",
            manifest.get("client_id") == case.client_id,
            expected=case.client_id, actual=manifest.get("client_id"),
            failure_class=_DEFECT))
    return out


def assert_scope_isolation(answers: Dict[str, Dict[str, Any]],
                           truth: Dict[str, Any]) -> List[Assertion]:
    """Total, direct-only and acquired-only must reconcile and not leak.

    ``answers`` is ``{scope: {"balance": float, "count": int}}`` measured
    through the production scope resolver. The properties are arithmetic:
    the parts sum to the whole, and no part equals the whole (which is what a
    scope filter silently failing open would look like).
    """
    out: List[Assertion] = []
    final = truth["periods"][-1]
    by_source = final.get("by_source") or {}
    types = truth["source_portfolio_types"]

    expected_by_type: Dict[str, Dict[str, float]] = {}
    for pid, figures in by_source.items():
        bucket = expected_by_type.setdefault(types[pid], {"balance": 0.0,
                                                          "count": 0})
        bucket["balance"] += figures["closing_balance"]
        bucket["count"] += figures["loan_count"]

    for scope, expected in expected_by_type.items():
        got = answers.get(scope)
        if got is None:
            out.append(check(STAGE_ASSEMBLE, f"scope {scope!r} was answerable",
                             False, failure_class=_DEFECT))
            continue
        out.append(compare(
            STAGE_ASSEMBLE, f"scope {scope!r} reports only its own balance",
            got["balance"], round(expected["balance"], 2),
            failure_class=_DEFECT))
        out.append(compare(
            STAGE_ASSEMBLE, f"scope {scope!r} reports only its own loan count",
            got["count"], expected["count"], failure_class=_DEFECT))

    total = answers.get("total")
    if total is not None:
        out.append(compare(
            STAGE_ASSEMBLE, "the total scope is the whole SPV",
            total["balance"], final["closing_balance"], failure_class=_DEFECT))
        parts = round(sum(a["balance"] for s, a in answers.items()
                          if s != "total"), 2)
        out.append(compare(
            STAGE_ASSEMBLE, "the provenance scopes sum to the total",
            parts, total["balance"], failure_class=_DEFECT))
        for scope, got in answers.items():
            if scope == "total":
                continue
            out.append(check(
                STAGE_ASSEMBLE, f"scope {scope!r} does not leak the whole book",
                abs(got["balance"] - total["balance"]) > 0.01,
                actual={"scope": got["balance"], "total": total["balance"]},
                detail="A filtered scope equal to the total is what a scope "
                       "filter failing OPEN looks like.",
                failure_class=_DEFECT))
    return out


__all__ = ["assert_assembly", "assert_scope_isolation"]

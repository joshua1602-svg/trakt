#!/usr/bin/env python3
"""Slice 1B: a governed plan becomes the ANSWER, for ONE allow-listed principal.

Slice 1A gave the shadow a plan, executed it, compared it and recorded it, and
the legacy answer was always what shipped. This module lets that result BE the
served response — for a principal named explicitly in configuration and for
nobody else. No new semantic capability: the interpreter, the compiler, the
eligibility perimeter and the adapter are the accepted ones, called as the
shadow calls them.

WHY A PRINCIPAL AND NOT A CLIENT. `ERE` is the only authorised production
client, so a client-level canary would be the whole book on its first day.
`ExecutionContext.actor_id` is the stable identity auth already established —
the verified Entra `oid` on the bearer path (`react_auth` refuses a token
carrying none), the platform `userId` on the Static Web Apps path — and already
what the audit trail records, so it is what this matches, exactly and
case-folded.

    MI_AGENT_PLAN_SERVE=canary               default off
    MI_AGENT_PLAN_SERVE_PRINCIPALS=<oid>     comma-separated, exact, no wildcard

No value of either setting means "everybody". `*`, `all`, `any` and `everyone`
are refused, and so are identity's own sentinels `unknown-principal` and
`local-dev` — what `actor_id` falls back to when nobody was identified, so
allow-listing one would serve every unidentified caller.

IT CANNOT READ THE QUESTION TO DECIDE ANYTHING. `handles` sees the context and
nothing else, and after the plan exists no semantic decision comes from the
sentence: no `ParsedQuestion.parse`, `llm_query_parser`,
`question_interpretation`, recogniser, router, or raw-text lens or dimension
reconstruction. The text reaches two presentational places only — the envelope's
`question` echo, which the legacy envelope also carries, and the record.

THE LEGACY ANSWER IS COMPUTED FIRST AND KEPT. This runs where the shadow runs,
after the legacy answer is complete. That costs the canary principal both paths
and buys the property that matters more: the fallback cannot fail, because what
is fallen back to is already in hand. Interpreter failure, clarify, refuse,
ineligibility, execution fault, reconciliation failure and any unexpected
exception all end with the legacy envelope returned unchanged and a record
saying which.

WHAT IS NOT RE-RUN, AND WHY THAT IS SAFE. Four of the five guards between legacy
execution and return re-read the question, which is the re-reading this
architecture exists to remove. They are unnecessary here because the eligibility
perimeter refuses — structurally, before execution — every shape they catch: a
stated historical period is PERIOD_NOT_CURRENT, a governed geography axis is
GEOGRAPHY_REQUESTED, an explicit lens is EXPLICIT_LENS, a specialist capability
or non-generic operation is CAPABILITY_NOT_GENERIC / OPERATION_NOT_GENERIC. Each
returns the question to the legacy path, where those guards run as they always
have. What the perimeter cannot see is whether the EXECUTOR did what the plan
said, so `reconcile` checks that before serving, from the receipt and the plan
and no text at all.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

from mi_agent import plan_runtime_adapter as adapter
from mi_agent import plan_shadow_evidence as evidence
from mi_agent import plan_shadow_wiring as wiring

logger = logging.getLogger("mi_agent.plan_serving_canary")

#: The serving flag, following the estate's `MI_AGENT_PLAN_*` convention. Two
#: states only, and the default is the one that serves nothing new.
SERVE_ENV_VAR = "MI_AGENT_PLAN_SERVE"
SERVE_OFF = "off"
SERVE_CANARY = "canary"

#: The allow-list: comma-separated PRINCIPAL ids, matched exactly.
PRINCIPALS_ENV_VAR = "MI_AGENT_PLAN_SERVE_PRINCIPALS"

#: Refused as allow-list entries. The first four are the wildcards this setting
#: deliberately does not support; the last two are `mi_agent_api.identity`'s own
#: fallbacks for "nobody was identified", which must never name a canary.
FORBIDDEN_PRINCIPAL_TOKENS = frozenset({
    "*", "all", "any", "everyone", "unknown-principal", "local-dev"})

#: What the response that reached the caller actually came from.
SERVED_NEW = "NEW"
SERVED_LEGACY = "LEGACY_FALLBACK"

# Why legacy served instead. Stable strings: the evidence groups by them.
INTERPRETER_FAILED = "INTERPRETER_FAILURE"
CLARIFY_NOT_SERVED = "CLARIFY_NOT_SERVED_IN_THIS_SLICE"
REFUSE_NOT_SERVED = "REFUSE_NOT_SERVED_IN_THIS_SLICE"
INELIGIBLE = "INELIGIBLE"
EXECUTION_FAILED = "EXECUTION_FAILED"
RECONCILIATION_FAILED = "PLAN_RECEIPT_RECONCILIATION_FAILED"
RENDER_FAILED = "RENDER_FAILED"
UNEXPECTED_ERROR = "UNEXPECTED_ERROR"


# --------------------------------------------------------------------------- #
# the canary
# --------------------------------------------------------------------------- #

def serve_mode() -> str:
    """`off` (the default) or `canary`. Anything unrecognised reads as off."""
    value = str(os.environ.get(SERVE_ENV_VAR) or "").strip().lower()
    return SERVE_CANARY if value == SERVE_CANARY else SERVE_OFF


def serve_principals() -> FrozenSet[str]:
    """The allow-listed principal ids, trimmed and case-folded."""
    found = set()
    for part in str(os.environ.get(PRINCIPALS_ENV_VAR) or "").split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in FORBIDDEN_PRINCIPAL_TOKENS:
            logger.warning("%s contains %r, which names no individual and is "
                           "ignored; name each canary principal explicitly",
                           PRINCIPALS_ENV_VAR, token)
            continue
        found.add(token)
    return frozenset(found)


def principal_of(context: Any) -> str:
    """The authenticated principal id, as the allow-list compares it.

    `ExecutionContext.actor_id` and nothing else: the verified `oid` on the
    bearer path, the platform `userId` on the header path. Never a display name
    or an email — identity that survives a rename is the whole point.
    """
    return str(getattr(context, "actor_id", "") or "").strip().lower()


def handles(context: Any) -> bool:
    """Whether THIS request's principal may be served the new path. Fail-closed.

    The only inputs are configuration and the authenticated identity. Nothing
    about what was asked reaches this decision, and an unset or empty allow-list
    serves nobody.
    """
    try:
        if serve_mode() != SERVE_CANARY:
            return False
        allowed = serve_principals()
        if not allowed:
            return False
        principal = principal_of(context)
        if not principal or principal in FORBIDDEN_PRINCIPAL_TOKENS:
            return False
        return principal in allowed
    except Exception:                                                # noqa: BLE001
        logger.warning("serving canary membership could not be decided; legacy "
                       "serves", exc_info=True)
        return False


# --------------------------------------------------------------------------- #
# plan -> receipt reconciliation, the one gate the perimeter cannot provide
# --------------------------------------------------------------------------- #

def reconcile(spec: Any, result: Any) -> Tuple[bool, str]:
    """Did the executor do what the plan said? `(ok, why_not)`.

    Reads the bound spec and the executor's own receipt. No question, no plan
    re-interpretation: the spec was bound from the plan by the accepted adapter,
    so comparing the spec with what executed compares the plan with what
    executed.

    Row counts are not used as a correctness criterion — the executor's own
    `predicate_evidence` explains why, and a predicate that correctly matches
    nothing has still executed. An EMPTY MEASURED POPULATION is nevertheless not
    served, and that is a PRESENTATION decision rather than a verdict on the
    arithmetic: "0 loans · £0" is the right answer to a question about a
    population this book does not contain, and the legacy path already owns a
    proven controlled way to say so. This slice adds no second one.

    `filtered_row_count`, not the result frame, is what says the population was
    empty. A `count` over an empty book still produces a one-row SUMMARY holding
    zero, so a result-frame emptiness check passed it and the canary served the
    zero — found by this module's own tests, not by reading the code.
    """
    metadata = dict(getattr(result, "metadata", None) or {})

    applied = {str(entry.get("field")) for entry in
               (metadata.get("applied_predicates") or ())
               if isinstance(entry, Mapping)}
    for field_name in (getattr(spec, "filters", None) or {}):
        if str(field_name) not in applied:
            return False, f"predicate on {field_name!r} was not applied"

    grouped = {str(key) for key in (metadata.get("group_field_keys") or ())}
    for axis in (getattr(spec, "dimensions", None) or ()):
        if str(axis) not in grouped:
            return False, f"axis {axis!r} was not grouped"

    frame = getattr(result, "data", None)
    if frame is None or getattr(frame, "empty", True):
        return False, "the execution produced no rows"
    measured = metadata.get("filtered_row_count")
    if isinstance(measured, int) and measured <= 0:
        return False, "the measured population is empty"
    return True, ""


# --------------------------------------------------------------------------- #
# rendering, through the response contract the legacy path already uses
# --------------------------------------------------------------------------- #

def render(spec: Any, result: Any, semantics: Any, frame: Any, *, question: str,
           portfolio_id: Optional[str], as_of: Optional[str]) -> Dict[str, Any]:
    """`(spec, MIQueryResult)` -> the existing React/API envelope.

    `mi_agent_api.adapters.adapt_workflow_result` is the contract every channel
    already renders, and it is given exactly what it needs: the spec, the
    executed result, the dataset's display hints, and the spec-derived
    interpretation block. Its own `_answer` composes the lead sentence from the
    executed rows and the spec — not from the question — so the prose here is
    the estate's existing prose, not a second narrative owner.
    """
    from mi_agent.mi_agent_workflow import describe_spec
    from mi_agent.mi_dataset_profile import profile_dataset
    from mi_agent_api.adapters import adapt_workflow_result

    from mi_agent.mi_agent_workflow import _RENDERABLE

    warnings = list(getattr(result, "warnings", None) or ())
    chart_result = None
    if spec.chart_type in _RENDERABLE:
        try:
            from mi_agent.mi_chart_factory import create_mi_chart
            chart_result = create_mi_chart(result, semantics)
            warnings.extend(chart_result.warnings)
        except Exception as exc:                                     # noqa: BLE001
            warnings.append(f"Chart not rendered: {exc}")

    workflow = {
        "ok": True,
        "error": None,
        # PRESENTATION ONLY. The legacy envelope echoes the question too; no
        # decision on this path is taken from it.
        "question": question,
        "parser_mode": "governed_plan",
        "spec": spec.to_dict(),
        "interpreted": describe_spec(spec, semantics,
                                     parser_mode="governed_plan"),
        "validation": {"ok": True, "errors": [], "warnings": [],
                       "resolved_fields": dict(
                           getattr(result, "resolved_fields", None) or {})},
        "display_hints": (profile_dataset(frame, semantics) or {}).get(
            "display_hints", {}),
        "query_result": result,
        "chart_result": chart_result,
        "warnings": warnings,
        "metadata": {},
    }
    return adapt_workflow_result(workflow, portfolio_id=portfolio_id, as_of=as_of)


# --------------------------------------------------------------------------- #
# the production entry point
# --------------------------------------------------------------------------- #

def serve(*, question: str, context: Any, client_id: Optional[str] = None,
          run_id: Optional[str] = None, legacy_result: Any, frame: Any,
          semantics: Any, view: Optional[str] = None,
          portfolio_id: Optional[str] = None,
          render_portfolio_id: Optional[str] = None,
          as_of: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The new envelope to serve, or None meaning "legacy serves".

    Raises nothing: a serving canary that could fail a request would be worse
    than no canary, and the legacy envelope the caller is holding is always a
    complete answer.

    `handles` IS RE-CHECKED HERE rather than trusted. The call site asks it and
    branches on it, so this is redundant today — and it is the difference between
    a future caller that forgets and a future caller that serves the new path to
    everybody. A serving decision is not a thing to leave to a docstring.
    """
    if not handles(context):
        return None
    cid = evidence.correlation_id()
    body = evidence.new_record(correlation_id=cid, question=question,
                               client_id=client_id, run_id=run_id, view=view,
                               portfolio_id=portfolio_id)
    payload: Optional[Dict[str, Any]] = None
    reason = UNEXPECTED_ERROR
    try:
        payload, reason = _attempt(
            body, question=question, frame=frame, semantics=semantics,
            render_portfolio_id=(render_portfolio_id if render_portfolio_id
                                 is not None else portfolio_id), as_of=as_of)
    except Exception as exc:                                         # noqa: BLE001
        payload, reason = None, UNEXPECTED_ERROR
        body["disposition"] = evidence.ORCHESTRATION_ERROR
        body["orchestration_error"] = f"{type(exc).__name__}: {exc}"[:300]
        logger.warning("serving canary failed; legacy serves", exc_info=True)

    # RECORDING CANNOT COST THE ANSWER. `evidence.write` swallows its own
    # faults, but this tail is still guarded: a recorder that raises anyway —
    # a future sink, a patched one — must not turn a good answer into a 500.
    # The test that proves it found this unguarded.
    try:
        legacy_ok = bool(isinstance(legacy_result, Mapping)
                         and legacy_result.get("ok"))
        body["serving"] = {
            "mode": serve_mode(),
            "principal_matched": True,
            "principal_id": principal_of(context),
            "new_path_eligible": bool(
                (body.get("eligibility") or {}).get("eligible")),
            "decision": SERVED_NEW if payload is not None else SERVED_LEGACY,
            "reason": "" if payload is not None else reason,
            "plan_id": (body.get("compiler") or {}).get("plan_id"),
            # WHICH ENVELOPE THE CALLER ACTUALLY RETURNED. Recorded as the
            # decision this module made and the caller honours unconditionally,
            # so the record states the served provenance rather than implying it.
            "response_served_from": (SERVED_NEW if payload is not None
                                     else SERVED_LEGACY),
            "legacy_result_available": legacy_ok,
            "legacy_value": adapter._old_value(legacy_result),
            "legacy_ok": legacy_ok,
            "new_value": (body.get("execution") or {}).get("value"),
        }
        evidence.write(body)
    except Exception:                                                # noqa: BLE001
        logger.warning("the serving record could not be completed; the answer "
                       "stands", exc_info=True)
    return payload


def _attempt(body: Dict[str, Any], *, question: str, frame: Any, semantics: Any,
             render_portfolio_id: Optional[str], as_of: Optional[str]
             ) -> Tuple[Optional[Dict[str, Any]], str]:
    """One serving attempt. `(payload or None, reason)`; fills `body` as it goes."""
    from mi_agent.interpretation_v2.outcomes import (OUTCOME_CLARIFY, OUTCOME_PLAN,
                                                     OUTCOME_REFUSE)

    outcome, compiled = wiring.build_plan(question)
    wiring.record_plan_stages(body, outcome, compiled)

    if not outcome.ok:
        body["disposition"] = evidence.INTERPRETER_FAILURE
        return None, INTERPRETER_FAILED
    if compiled.outcome == OUTCOME_CLARIFY:
        body["disposition"] = evidence.CLARIFY
        return None, CLARIFY_NOT_SERVED
    if compiled.outcome == OUTCOME_REFUSE:
        body["disposition"] = evidence.REFUSE
        return None, REFUSE_NOT_SERVED
    if compiled.outcome != OUTCOME_PLAN:                             # unreachable
        body["disposition"] = evidence.ORCHESTRATION_ERROR
        return None, UNEXPECTED_ERROR

    # From here the accepted slice 1 perimeter owns every decision, and nothing
    # below edits the plan the compiler emitted.
    plan = compiled.plan.to_dict()
    eligible, why, detail = adapter.check_eligibility(plan)
    body["eligibility"] = {"eligible": eligible, "reason": why, "detail": detail}
    if not eligible:
        body["execution"] = {"attempted": False, "why_not": f"{why}: {detail}"[:300]}
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{why}"

    from mi_agent.mi_query_executor import execute_mi_query
    spec = adapter.spec_for_plan(plan)
    body["execution"] = {"attempted": True, "bound_spec": spec.to_dict(),
                         "requested_semantics": adapter.requested_semantics(plan)}
    try:
        result = execute_mi_query(spec, frame, semantics)
    except Exception as exc:                                         # noqa: BLE001
        body["execution"]["error"] = f"{type(exc).__name__}: {exc}"[:300]
        body["disposition"] = evidence.EXECUTION_ERROR
        return None, EXECUTION_FAILED

    dimensions = list(getattr(spec, "dimensions", None) or ())
    cells, note = (evidence.cells_of(getattr(result, "data", None), dimensions,
                                     f"{spec.metric}_{spec.aggregation}")
                   if dimensions else (None, None))
    body["execution"].update({
        "value": adapter._scalar_of(result, spec),
        "receipt": dict(getattr(result, "metadata", None) or {}),
        "warnings": [str(w)[:200] for w in (getattr(result, "warnings", None) or ())],
        "row_count": getattr(result, "row_count", None),
        "result_type": getattr(result, "result_type", None),
        "grouped_cells": cells,
        "grouped_note": note,
    })
    body["disposition"] = evidence.EXECUTED

    reconciled, why_not = reconcile(spec, result)
    body["execution"]["reconciled"] = reconciled
    body["execution"]["reconciliation_note"] = why_not
    if not reconciled:
        return None, f"{RECONCILIATION_FAILED}:{why_not}"[:300]

    try:
        payload = render(spec, result, semantics, frame, question=question,
                         portfolio_id=render_portfolio_id, as_of=as_of)
    except Exception as exc:                                         # noqa: BLE001
        body["execution"]["render_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return None, RENDER_FAILED
    if not isinstance(payload, Mapping) or not payload.get("ok"):
        body["execution"]["render_error"] = "the rendered envelope was not ok"
        return None, RENDER_FAILED
    return dict(payload), ""

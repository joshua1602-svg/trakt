#!/usr/bin/env python3
"""The one missing seam: a live request, and a governed plan built for it.

WHAT WAS WRONG. `plan_runtime_adapter.observe` can compare a governed plan with
the served answer, but nothing in production ever gave it a plan — the call site
passed none and `set_plan_provider` had no production caller — so
`MI_AGENT_PLAN_SHADOW=shadow` produced no plan, no execution and no ledger row.
This module is the missing half, and nothing more: it adds PLAN CREATION to the
existing shadow and no semantic capability of any kind.

WHAT IT IS NOT ALLOWED TO BE. It hands the question verbatim to the FROZEN
`interpretation_v2` and takes what comes back. It never reads the question
itself, never calls a legacy parser, recogniser or router, never fills a gap in a
plan from a legacy result, and never edits an emitted plan. The eligibility
perimeter and the adapter it calls are the accepted slice 1 ones, byte-unchanged.

THREE BOUNDS ON WHAT IT CAN COST, because a feature flag that means "call Opus
for every production request" would be a liability rather than a canary:

    OFF                  zero work — no interpreter is even constructed
    ON, outside canary   zero model calls, zero evidence; the canary is an
                         explicit allow-list of CLIENT ids, and it is
                         fail-closed: an unset list shadows nobody
    ON, in canary        at most `MAX_IN_FLIGHT` shadows at once (default ONE),
                         so concurrent canary traffic is skipped-and-recorded
                         rather than queued into an unbounded model bill

THE CANARY CANNOT READ THE QUESTION. It sees the client id and nothing else; a
test asserts this module's canary function touches no question text. Scoping a
rollout by what was asked would make the shadow's coverage depend on semantics,
which is exactly the confusion this architecture exists to remove.

IT RUNS OFF THE REQUEST PATH. The shadow is dispatched to a daemon thread, so the
served response is unaffected in CONTENT and in LATENCY alike — a live
interpretation measured 10-26 seconds in the live sign-off, and adding that to a
served response would be "influencing" it by any honest reading, even with the
answer identical. The served answer is already fully computed before this is
called. Dispatch is a module seam so tests can run it inline and deterministically.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional, Tuple

from mi_agent import plan_runtime_adapter as adapter
from mi_agent import plan_shadow_evidence as evidence

logger = logging.getLogger("mi_agent.plan_shadow_wiring")

#: The canary allow-list: comma-separated CLIENT ids, following the estate's
#: existing app-setting convention (`react_auth._dashboard_directories`). Unset
#: or empty means nobody, never everybody.
CANARY_ENV_VAR = "MI_AGENT_PLAN_SHADOW_CLIENTS"

#: The wildcard is deliberately NOT supported. There is no value of this setting
#: that means "every client": widening the canary is an act of configuration per
#: client, so a rollout cannot happen by accident.
FORBIDDEN_CANARY_TOKENS = frozenset({"*", "all", "any", "everyone"})

#: How many shadows may be in flight at once. One by default: strictly serial
#: shadowing bounds the model spend to one interpretation at a time.
MAX_IN_FLIGHT_ENV_VAR = "MI_AGENT_PLAN_SHADOW_MAX_IN_FLIGHT"
DEFAULT_MAX_IN_FLIGHT = 1

_in_flight = 0
_in_flight_lock = threading.Lock()

#: Built once, on the first canary request. Constructing these loads the governed
#: vocabulary, which must not happen merely because the module was imported.
_interpreter_cache: Optional[Any] = None
_compiler_cache: Optional[Any] = None
_interpreter_factory: Optional[Callable[[], Any]] = None


# --------------------------------------------------------------------------- #
# the canary
# --------------------------------------------------------------------------- #

def canary_clients() -> FrozenSet[str]:
    """The allow-listed client ids, lower-cased and trimmed."""
    raw = str(os.environ.get(CANARY_ENV_VAR) or "")
    found = set()
    for part in raw.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in FORBIDDEN_CANARY_TOKENS:
            logger.warning("%s contains %r, which is not a client id and is "
                           "ignored; name each canary client explicitly",
                           CANARY_ENV_VAR, token)
            continue
        found.add(token)
    return frozenset(found)


def in_canary(client_id: Optional[str]) -> bool:
    """Whether this CLIENT is shadowed. Fail-closed, and semantics-blind.

    The only input is the authenticated request's client id. No part of what was
    asked reaches this decision.
    """
    allowed = canary_clients()
    if not allowed:
        return False
    return str(client_id or "").strip().lower() in allowed


def max_in_flight() -> int:
    try:
        value = int(str(os.environ.get(MAX_IN_FLIGHT_ENV_VAR) or
                        DEFAULT_MAX_IN_FLIGHT).strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_IN_FLIGHT
    return max(1, value)


# --------------------------------------------------------------------------- #
# the frozen interpreter, built lazily
# --------------------------------------------------------------------------- #

def set_interpreter_factory(factory: Optional[Callable[[], Any]]) -> None:
    """Inject an interpreter. For tests and offline replay only.

    Production leaves this None, and the factory below builds the frozen
    `OpusInterpreter` over the real client.
    """
    global _interpreter_factory, _interpreter_cache
    _interpreter_factory = factory
    _interpreter_cache = None


def _default_interpreter() -> Any:
    """The frozen interpreter, configured exactly as the sign-off configured it.

    `AnthropicInterpreterClient`'s own default model is the configured one, and
    `CompilerContext()` is built with no narrowing — the same construction
    `interpretation_v2.benchmark.run_benchmark` used for the 135-question
    sign-off and the live integration run. That is deliberate: a deployed plan
    must be the plan those runs proved, and narrowing the vocabulary to the
    book would be a semantic change, which this slice is not.
    """
    from mi_agent.interpretation_v2.opus_interpreter import (
        AnthropicInterpreterClient, OpusInterpreter)
    return OpusInterpreter(AnthropicInterpreterClient())


def _interpreter() -> Any:
    global _interpreter_cache
    if _interpreter_cache is None:
        factory = _interpreter_factory or _default_interpreter
        _interpreter_cache = factory()
    return _interpreter_cache


def _compiler() -> Any:
    global _compiler_cache
    if _compiler_cache is None:
        from mi_agent.interpretation_v2.compiler import (CompilerContext,
                                                         DeterministicCompiler)
        _compiler_cache = DeterministicCompiler(CompilerContext())
    return _compiler_cache


def build_plan(question: str) -> Tuple[Any, Any]:
    """`question -> (InterpretationOutcome, CompileResult)`. One attempt.

    Exactly the sequence `run_benchmark` uses, and exactly one interpretation per
    shadowed request: no retry, no reworded second try, no repair loop. Transport
    retries remain the model client's own business.
    """
    from mi_agent.interpretation_v2.outcomes import refuse

    compiler = _compiler()
    outcome = _interpreter().interpret(question)
    compiled = (compiler.compile(outcome.intent) if outcome.ok
                else refuse(outcome.reason, compiler_version=compiler.version))
    return outcome, compiled


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

def _dispatch_in_background(work: Callable[[], Any]) -> Optional[Dict[str, Any]]:
    """Run the shadow off the request path. Production default."""
    thread = threading.Thread(target=work, name="mi-plan-shadow", daemon=True)
    thread.start()
    return None


def _dispatch_inline(work: Callable[[], Any]) -> Optional[Dict[str, Any]]:
    """Run the shadow in the caller's thread and hand back its record. Tests."""
    return work()


_DISPATCH: Callable[[Callable[[], Any]], Optional[Dict[str, Any]]] = \
    _dispatch_in_background


def set_dispatch(dispatch: Optional[Callable]) -> None:
    """Swap the dispatcher. For tests; production keeps the background one."""
    global _DISPATCH
    _DISPATCH = dispatch or _dispatch_in_background


# --------------------------------------------------------------------------- #
# the shadow
# --------------------------------------------------------------------------- #

def _record_spec(spec: Any) -> Optional[Dict[str, Any]]:
    try:
        return spec.to_dict()
    except Exception:                                                # noqa: BLE001
        return None


def record_plan_stages(body: Dict[str, Any], outcome: Any, compiled: Any) -> None:
    """Fill the model / interpretation / compiler stages of one record.

    ONE OWNER FOR THE RECORD SHAPE. The slice 1B serving canary records the same
    three stages, and the deployed acceptance adjudicates on these exact keys —
    two copies of this that drifted would corrupt the evidence corpus rather than
    merely duplicate code, so both callers come here.
    """
    body["model"] = {
        "model_id": outcome.model_id,
        "usage": dict(outcome.usage or {}),
        "raw_payload": outcome.raw_payload,
        "metadata_calls": [dict(call) for call in
                           (outcome.metadata_calls or ())],
        "failure": (outcome.reason.to_dict()
                    if (outcome.reason is not None and not outcome.ok)
                    else None),
    }
    # The ambiguities are read off the SERIALISED intent rather than off the
    # dataclass: `CandidateIntent.to_dict` is an `asdict`, so it already
    # carries them whole, and `Ambiguity` has no `to_dict` of its own — asking
    # it for one raised and lost the entire record. Caught by these tests.
    intent_body = (outcome.intent.to_dict()
                   if outcome.intent is not None else None)
    body["interpretation"] = {
        "produced_intent": bool(outcome.ok),
        "candidate_intent": intent_body,
        "ambiguities": list((intent_body or {}).get("ambiguity") or ()),
    }
    body["compiler"] = dict(compiled.to_dict(),
                            reason_codes=list(compiled.codes()),
                            plan_id=(compiled.plan.plan_id
                                     if compiled.is_plan else None))


def _shadow(*, question: str, client_id: Optional[str], run_id: Optional[str],
            result: Any, frame: Any, semantics: Any, view: Optional[str],
            portfolio_id: Optional[str], cid: str) -> Dict[str, Any]:
    """One shadowed request, start to finish. Always returns a record."""
    from mi_agent.interpretation_v2.outcomes import (OUTCOME_CLARIFY,
                                                     OUTCOME_PLAN,
                                                     OUTCOME_REFUSE)
    body = evidence.new_record(correlation_id=cid, question=question,
                               client_id=client_id, run_id=run_id, view=view,
                               portfolio_id=portfolio_id)
    try:
        outcome, compiled = build_plan(question)
        record_plan_stages(body, outcome, compiled)

        if not outcome.ok:
            body["disposition"] = evidence.INTERPRETER_FAILURE
            return _finish(body)
        if compiled.outcome == OUTCOME_CLARIFY:
            body["disposition"] = evidence.CLARIFY
            return _finish(body)
        if compiled.outcome == OUTCOME_REFUSE:
            body["disposition"] = evidence.REFUSE
            return _finish(body)
        if compiled.outcome != OUTCOME_PLAN:                          # unreachable
            body["disposition"] = evidence.ORCHESTRATION_ERROR
            return _finish(body)

        # From here the accepted slice 1 perimeter owns every decision. The plan
        # is handed over as the compiler emitted it; nothing below edits it.
        plan = compiled.plan.to_dict()
        eligible, reason, detail = adapter.check_eligibility(plan)
        body["eligibility"] = {
            "eligible": eligible, "reason": reason, "detail": detail,
            "gate_capability": adapter.ELIGIBLE_CAPABILITY,
            "gate_operations": sorted(adapter.ELIGIBLE_OPERATIONS),
            "gate_period_forms": sorted(adapter.ELIGIBLE_PERIOD_FORMS),
            "gate_max_dimensions": adapter.MAX_DIMENSIONS,
        }

        comparison = adapter.observe(result=result, frame=frame,
                                    semantics=semantics, plan=plan,
                                    case_id=cid, view=view,
                                    portfolio_id=portfolio_id)
        body["legacy_control"] = {
            "served_ok": (comparison or {}).get("control_ok"),
            "served_value": (comparison or {}).get("control_value"),
            "served_route": (comparison or {}).get("control_route"),
            "classification": (comparison or {}).get("classification"),
            "note": (comparison or {}).get("note"),
        }

        if not eligible:
            body["execution"] = {"attempted": False,
                                 "why_not": f"{reason}: {detail}"[:300]}
            body["disposition"] = evidence.INELIGIBLE
            return _finish(body)

        body["execution"] = _execute_evidence(plan, frame, semantics, comparison)
        body["disposition"] = (evidence.EXECUTION_ERROR
                               if body["execution"].get("error")
                               else evidence.EXECUTED)
        return _finish(body)
    except Exception as exc:                                         # noqa: BLE001
        body["disposition"] = evidence.ORCHESTRATION_ERROR
        body["orchestration_error"] = f"{type(exc).__name__}: {exc}"[:300]
        logger.warning("plan shadow orchestration failed", exc_info=True)
        return _finish(body)


def _execute_evidence(plan: Mapping[str, Any], frame: Any, semantics: Any,
                      comparison: Optional[Mapping[str, Any]]
                      ) -> Dict[str, Any]:
    """The requested contract, the bound contract, and what the engine produced.

    `adapter.observe` above already executed this plan through the accepted
    path, and its record carries the figure and the receipt. What it cannot
    carry is the grouped GRID — `ShadowOutcome` deliberately holds no frame —
    so for a grouped plan only, the adapter's OWN spec is run once more through
    the same deterministic executor purely to capture the cells. Same spec, same
    frame, same engine: nothing is re-derived here, and the reconciliation
    fields below let a reader check the two agree.
    """
    record: Dict[str, Any] = {
        "attempted": True,
        "requested_semantics": adapter.requested_semantics(plan),
        "bound_spec": None,
        "value": (comparison or {}).get("new_value"),
        "receipt": dict((comparison or {}).get("new_receipt") or {}),
        "warnings": list((comparison or {}).get("new_warnings") or ()),
        "error": (comparison or {}).get("new_error") or "",
        "grouped_cells": None,
        "grouped_note": None,
        "cell_capture_reran_the_adapters_own_spec": False,
    }
    try:
        spec = adapter.spec_for_plan(plan)
        record["bound_spec"] = _record_spec(spec)
        dimensions = list(getattr(spec, "dimensions", None) or ())
        if dimensions and not record["error"]:
            from mi_agent.mi_query_executor import execute_mi_query
            result = execute_mi_query(spec, frame, semantics)
            cells, note = evidence.cells_of(
                getattr(result, "data", None), dimensions,
                f"{spec.metric}_{spec.aggregation}")
            record["grouped_cells"] = cells
            record["grouped_note"] = note
            record["cell_capture_reran_the_adapters_own_spec"] = True
            record["cell_capture_row_count"] = getattr(result, "row_count", None)
    except Exception as exc:                                         # noqa: BLE001
        record["grouped_note"] = (f"cell capture failed: "
                                  f"{type(exc).__name__}: {exc}")[:300]
    return record


def _finish(body: Dict[str, Any]) -> Dict[str, Any]:
    evidence.write(body)
    return body


# --------------------------------------------------------------------------- #
# the production entry point
# --------------------------------------------------------------------------- #

def observe_request(*, question: str, client_id: Optional[str],
                    run_id: Optional[str] = None, result: Any, frame: Any,
                    semantics: Any, view: Optional[str] = None,
                    portfolio_id: Optional[str] = None
                    ) -> Optional[Dict[str, Any]]:
    """The call `mi_service` makes. Returns a record only under inline dispatch.

    Swallows everything. The served answer has already been computed when this
    runs and this cannot touch it; the background dispatcher means it cannot
    delay it either.
    """
    global _in_flight
    try:
        if adapter.shadow_mode() != adapter.SHADOW_ON:
            return None
        if not in_canary(client_id):
            # Nothing is written and no interpreter is built. The skeleton is
            # returned unpersisted so an operator or a test can see WHY this
            # request was not shadowed.
            skipped = evidence.new_record(
                correlation_id=evidence.correlation_id(), question=question,
                client_id=client_id, run_id=run_id, view=view,
                portfolio_id=portfolio_id)
            skipped["disposition"] = evidence.OUTSIDE_CANARY
            skipped["evidence_persisted"] = False
            return skipped

        cid = evidence.correlation_id()
        limit = max_in_flight()
        with _in_flight_lock:
            if _in_flight >= limit:
                busy = evidence.new_record(
                    correlation_id=cid, question=question, client_id=client_id,
                    run_id=run_id, view=view, portfolio_id=portfolio_id)
                busy["disposition"] = evidence.SHADOW_SKIPPED_BUSY
                busy["in_flight_limit"] = limit
                evidence.write(busy)
                return busy
            _in_flight += 1

        def work() -> Dict[str, Any]:
            global _in_flight
            try:
                return _shadow(question=question, client_id=client_id,
                               run_id=run_id, result=result, frame=frame,
                               semantics=semantics, view=view,
                               portfolio_id=portfolio_id, cid=cid)
            finally:
                with _in_flight_lock:
                    _in_flight -= 1

        return _DISPATCH(work)
    except Exception:                                                # noqa: BLE001
        logger.warning("plan shadow could not be started", exc_info=True)
        return None


def in_flight() -> int:
    """For tests and operational visibility."""
    return _in_flight

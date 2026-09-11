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
from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent import plan_temporal_runtime as temporal

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
#: A temporal plan arrived and the caller supplied no governed snapshot
#: catalogue. Not an error and not a refusal: the legacy path serves, exactly as
#: it does for any plan this canary cannot take. Production supplies no store
#: today, so this is the state every production temporal plan is in — and it is
#: why wiring this seam changes nothing a production caller can observe.
TEMPORAL_STORE_UNAVAILABLE = "TEMPORAL_STORE_UNAVAILABLE"
TEMPORAL_NOT_RESOLVED = "TEMPORAL_NOT_RESOLVED"


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
    # THE STRUCTURAL HALF LIVES IN THE ADAPTER. It is the same question the
    # temporal runtime asks of each snapshot, and two copies of it that drifted
    # would let one path serve what the other refuses. Behaviour here is
    # unchanged: the same two checks, in the same order, before the same rows
    # and emptiness rulings below.
    structural_ok, why_not = adapter.reconcile_receipt(spec, result)
    if not structural_ok:
        return False, why_not

    metadata = dict(getattr(result, "metadata", None) or {})
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
           portfolio_id: Optional[str], as_of: Optional[str],
           requested: Optional[Mapping[str, Any]] = None,
           executed: Optional[Mapping[str, Any]] = None,
           execution_population: Optional[str] = None) -> Dict[str, Any]:
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
    payload = adapt_workflow_result(workflow, portfolio_id=portfolio_id,
                                    as_of=as_of)
    # THE TWO GOVERNED OBJECTS, CARRIED SO THE COVERAGE OWNER CAN RECONCILE THEM.
    #
    # Everything downstream of here used to have only one way to ask "was the LTV
    # threshold applied": re-read the question and look for a legacy-shaped
    # receipt. This envelope carries neither, so the owner found the facet
    # unaccounted and converted three correct figures into UNSUPPORTED_QUESTION —
    # A01/A02/A03 measured live on d360bead, each with the right number in the
    # evidence and no number in the response.
    #
    # So the answer states WHAT WAS ASKED (the plan's own transcription, which the
    # adapter read off the GovernedQueryPlan and never from the sentence) and WHAT
    # RAN (the deterministic executor's own receipt). Additive metadata only: no
    # pre-existing key changes, and a legacy answer never carries this block, so
    # the legacy path cannot see any of it.
    meta = payload.setdefault("metadata", {})
    if isinstance(meta, dict):
        receipt = dict(getattr(result, "metadata", None) or {})
        # `executed` IS SUPPLIED BY THE TEMPORAL PATH and by nothing else. One
        # snapshot's receipt cannot describe a series, so the temporal runtime
        # hands its own per-snapshot evidence in here rather than growing a
        # second renderer. A slice 1 answer passes None and is byte-identical to
        # what it always was.
        executed_block = dict(executed) if executed is not None else {
            "applied_predicates": receipt.get("applied_predicates") or [],
            "group_field_keys": list(receipt.get("group_field_keys") or ()),
            "aggregation": receipt.get("aggregation"),
            "balance_field_used": receipt.get("balance_field_used"),
            "percent_scale_detected": receipt.get("percent_scale_detected"),
            "filtered_row_count": receipt.get("filtered_row_count"),
        }
        # WHICH POPULATION ACTUALLY RAN, declared by the runtime that resolved
        # the frame. The receipt above proves which PREDICATES ran; nothing in
        # it names the population they ran over, and a filter correctly applied
        # to the wrong dataset is still the wrong answer. Absent, the coverage
        # ledger finds the population unaccounted and refuses — which is the
        # right failure, because the alternative is assuming funded.
        if execution_population:
            executed_block["population_base"] = str(execution_population)
        meta["governedPlan"] = {
            "requested": dict(requested or {}),
            "executed": executed_block,
        }
    return payload


# --------------------------------------------------------------------------- #
# the production entry point
# --------------------------------------------------------------------------- #

def serve(*, question: str, context: Any, client_id: Optional[str] = None,
          run_id: Optional[str] = None, legacy_result: Any, frame: Any,
          semantics: Any,
          # THE EXECUTED POPULATION. `funded` | `pipeline` | `forecast` — the
          # governed dataset identity the caller resolved `frame` with. It was
          # already supplied for the evidence record; it is now the fact the
          # perimeter reconciles the plan's `population.base` against, and an
          # unset one serves nothing.
          view: Optional[str] = None,
          portfolio_id: Optional[str] = None,
          render_portfolio_id: Optional[str] = None,
          as_of: Optional[str] = None,
          snapshot_store: Any = None,
          snapshot_client_id: Optional[str] = None,
          snapshot_route: Optional[str] = None,
          source_registry: Any = None,
          # THE PIPELINE OWNERS' INPUTS, resolved by the caller that already owns
          # discovery. This module loads nothing and discovers nothing: it is
          # handed the governed source scope, the weekly history root and the
          # client whose extracts they are, exactly as it is handed the funded
          # frame and the funded snapshot catalogue.
          pipeline_source: Any = None, pipeline_root: Any = None,
          pipeline_client_id: Optional[str] = None,
          pipeline_history: Any = None) -> Optional[Dict[str, Any]]:
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
                                 is not None else portfolio_id), as_of=as_of,
            snapshot_store=snapshot_store,
            snapshot_client_id=snapshot_client_id,
            snapshot_route=snapshot_route,
            pipeline_source=pipeline_source, pipeline_root=pipeline_root,
            pipeline_client_id=pipeline_client_id,
            pipeline_history=pipeline_history, pipeline_run_id=run_id,
            # `view` IS THE EXECUTED POPULATION, and it is load-bearing here
            # rather than decorative. It is the governed dataset identity the
            # caller resolved `frame` with — `mi_service` passes the same string
            # it called `datasets._resolve_query_frame(view, …)` with — so it
            # states what was LOADED. A SECOND parameter carrying the same fact
            # is not added on purpose: two fields that must agree is the defect
            # class this gate exists to close.
            execution_population=view)
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


def _attempt_pipeline(body: Dict[str, Any], *, plan: Mapping[str, Any],
                      question: str, render_portfolio_id: Optional[str],
                      as_of: Optional[str], pipeline_source: Any,
                      pipeline_root: Any, pipeline_client_id: Optional[str],
                      pipeline_history: Any, pipeline_run_id: Optional[str]
                      ) -> Tuple[Optional[Dict[str, Any]], str]:
    """One PIPELINE serving attempt. Same contract as `_attempt`: `(payload, reason)`.

    Stage for stage the slice 1 attempt: perimeter, prove the population,
    execute, render, record. What differs is whose perimeter and whose runtime —
    the pipeline runtime declares its own executable population and calls the
    Pipeline owners that already compute these figures.

    THE POPULATION IS PROVED AGAINST THE PIPELINE RUNTIME'S OWN DECLARATION, not
    a merged list. `adapter.check_population_base` takes the executable set as an
    argument for exactly this: the funded runtime says it executes funded, this
    one says it executes pipeline, and a plan is refused by whichever runtime it
    reached if the two do not agree. One global list saying both runtimes execute
    both populations would prove nothing.
    """
    eligible, why, detail = pipeline_rt.check_eligibility(plan)
    body["eligibility"] = {"eligible": eligible, "reason": why, "detail": detail,
                           "perimeter": "pipeline_specialist"}
    if not eligible:
        body["execution"] = {"attempted": False, "why_not": f"{why}: {detail}"[:300]}
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{why}"

    base_ok, base_why, base_detail = adapter.check_population_base(
        plan, pipeline_rt.EXECUTION_POPULATION,
        executable=pipeline_rt.EXECUTABLE_POPULATIONS)
    if not base_ok:
        body["eligibility"] = {"eligible": False, "reason": base_why,
                               "detail": base_detail,
                               "perimeter": "pipeline_population"}
        body["execution"] = {"attempted": False,
                             "why_not": f"{base_why}: {base_detail}"[:300]}
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{base_why}"

    temporal_plan = pipeline_rt.is_temporal(plan)
    body["execution"] = {"attempted": True,
                         "runtime": ("pipeline_temporal" if temporal_plan
                                     else "pipeline_current"),
                         "requested_semantics": adapter.requested_semantics(plan)}
    try:
        if temporal_plan:
            outcome = pipeline_rt.execute_temporal(
                plan, root=pipeline_root, client_id=pipeline_client_id or "",
                to_run_id=pipeline_run_id, history_model=pipeline_history)
        else:
            outcome = pipeline_rt.execute_current(
                plan, source=pipeline_source, history_model=pipeline_history)
    except Exception as exc:                                         # noqa: BLE001
        body["execution"]["error"] = f"{type(exc).__name__}: {exc}"[:300]
        body["disposition"] = evidence.EXECUTION_ERROR
        return None, EXECUTION_FAILED
    if not outcome.ok:
        body["execution"].update({"attempted": False,
                                  "why_not": f"{outcome.reason}: {outcome.detail}"[:300]})
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{outcome.reason}"

    body["execution"].update({
        "value": outcome.value,
        "grouped_cells": outcome.cells,
        "receipt": dict(outcome.receipt),
        "row_count": (len(outcome.cells) if outcome.cells is not None else None),
    })
    body["disposition"] = evidence.EXECUTED

    try:
        payload = render_pipeline(plan, outcome, question=question,
                                  portfolio_id=render_portfolio_id, as_of=as_of)
    except Exception as exc:                                         # noqa: BLE001
        body["execution"]["render_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return None, RENDER_FAILED
    if not isinstance(payload, Mapping) or not payload.get("ok"):
        body["execution"]["render_error"] = "the rendered envelope was not ok"
        return None, RENDER_FAILED
    return dict(payload), ""


def render_pipeline(plan: Mapping[str, Any], outcome: Any, *, question: str,
                    portfolio_id: Optional[str], as_of: Optional[str]
                    ) -> Dict[str, Any]:
    """The pipeline result, in the envelope every channel already renders.

    THE ARTEFACT BUILDERS ARE THE ESTATE'S, not a second presenter. The accepted
    pipeline routes already publish through `chat_routing`'s KPI, chart and
    envelope builders, so a governed pipeline answer is shaped the same way the
    legacy one is and no channel learns a new form. Only the BUILDERS are
    borrowed — `try_route`, the router itself, is never called, and the forbidden
    entry-point guard asserts it.
    """
    import uuid
    from datetime import datetime, timezone

    def _uid() -> str:
        return f"art_{uuid.uuid4().hex[:8]}"

    def _artefact(kind: str, title: str, **rest) -> Dict[str, Any]:
        """One artefact in the estate's published shape.

        BUILT HERE RATHER THAN BORROWED, and not by preference. `chat_routing`
        owns identical builders, but the serving path may not import that module
        — `test_the_module_names_no_legacy_semantic_owner_in_its_code` bans it,
        and rightly: importing the legacy router to borrow a presenter drags the
        whole legacy semantic estate into the governed path. The keys below are
        the same contract every channel already renders, so nothing downstream
        learns a new form; only the construction is local.
        """
        return {"id": _uid(), "type": kind, "title": title,
                "source": {"engine": "mi_agent.governed_plan",
                           "label": f"MI Agent · {kind}", "spec": spec_dict,
                           "asOf": as_of, "portfolio": portfolio_id},
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "mock": False, **rest}

    receipt = dict(outcome.receipt)
    measure = str(receipt.get("measure_concept") or "")
    is_amount = receipt.get("measure_kind") == "amount"
    axis = (receipt.get("group_field_keys") or [None])[0]
    spec_dict = {"capability": receipt.get("capability"),
                 "population": receipt.get("population_base"),
                 "measure": measure, "dimensions": receipt.get("group_field_keys")}
    shape = str(receipt.get("result_shape") or "")

    if shape == "scalar":
        value = float(outcome.value)
        kpis = [{"field": measure, "label": _PIPELINE_LABELS.get(measure, measure),
                 "value": (f"£{value:,.0f}" if is_amount else f"{value:,.0f}"),
                 "rawValue": value}]
        artefacts = [_artefact("kpi", "Pipeline", kpis=kpis,
                               description="Governed pipeline extract.")]
        answer = (f"Pipeline {_PIPELINE_LABELS.get(measure, measure).lower()} is "
                  f"{kpis[0]['value']}.")
    elif shape == "grouped":
        rows = [{str(axis): c[axis], "value": c["value"]} for c in outcome.cells]
        artefacts = [_artefact(
            "table", "Pipeline by stage", rows=rows,
            columns=[{"key": str(axis), "label": "Stage"},
                     {"key": "value",
                      "label": _PIPELINE_LABELS.get(measure, measure)}],
            description=f"{len(rows)} rows.")]
        answer = f"Pipeline by {axis} across {len(rows)} governed stage(s)."
    else:
        # A SERIES, weekly. One row per governed extract; a grouped series gets
        # one column per stage, which is the shape the accepted evolution route
        # already publishes.
        periods = sorted({str(c["period"]) for c in outcome.cells})
        if axis:
            stages = sorted({str(c[axis]) for c in outcome.cells})
            rows = [{"period": per,
                     **{st: sum(float(c["value"] or 0) for c in outcome.cells
                                if str(c["period"]) == per and str(c[axis]) == st)
                        for st in stages}}
                    for per in periods]
            series = [{"key": st, "label": st} for st in stages]
            answer = (f"Pipeline by {axis} across {len(periods)} governed weekly "
                      f"extract(s): {', '.join(stages)}.")
        else:
            rows = [{"period": str(c["period"]), "value": c["value"]}
                    for c in outcome.cells]
            series = [{"key": "value",
                       "label": _PIPELINE_LABELS.get(measure, measure)}]
            answer = (f"Pipeline {_PIPELINE_LABELS.get(measure, measure).lower()} "
                      f"across {len(periods)} governed weekly extract(s).")
        artefacts = [_artefact(
            "chart", "Pipeline over time", chartType="line", xKey="period",
            rows=rows, series=series,
            valueFormat=("gbp" if is_amount else "number"), displayHints={})]

    reconciliation = {"dataset": "pipeline", "coverage_by_balance_pct": 100.0}
    for artefact in artefacts:
        artefact.setdefault("reconciliation", reconciliation)
    payload: Dict[str, Any] = {
        "ok": True, "error": None, "question": question, "answer": answer,
        "interpreted": "", "spec": spec_dict,
        "validation": {"ok": True, "errors": [], "warnings": [],
                       "resolved_fields": {}},
        "artifacts": artefacts, "reconciliation": reconciliation,
        "sourceNotes": [], "warnings": [], "diagnostics": [], "assumptions": [],
        "metadata": {"engine": "mi_agent", "source": "python", "mock": False,
                     "route": "governed_plan_pipeline", "lensApplied": None},
    }
    meta = payload.setdefault("metadata", {})
    if isinstance(meta, dict):
        meta["parserMode"] = "governed_plan"
        # THE TWO GOVERNED OBJECTS, exactly as the slice 1 renderer carries them,
        # so the coverage owner reconciles a pipeline answer by the same reading
        # it applies to a funded one.
        meta["governedPlan"] = {
            "requested": dict(adapter.requested_semantics(plan)),
            "executed": receipt,
        }
    return payload


#: Reader-facing names for the governed pipeline measures. Presentation only.
_PIPELINE_LABELS = {
    "pipeline_amount": "Pipeline amount",
    "pipeline_case_count": "Pipeline case count",
    "loan": "Pipeline case count",
    "loan_count": "Pipeline case count",
}


def _attempt_temporal(body: Dict[str, Any], *, plan: Mapping[str, Any],
                      question: str, semantics: Any, store: Any,
                      snapshot_client_id: Optional[str],
                      snapshot_route: Optional[str],
                      render_portfolio_id: Optional[str], as_of: Optional[str],
                      execution_population: Optional[str] = None
                      ) -> Tuple[Optional[Dict[str, Any]], str]:
    """One temporal serving attempt. Same contract as `_attempt`.

    Mirrors the slice 1 attempt stage for stage — perimeter, execute,
    reconcile, render, record — with the slice 2 perimeter and the slice 2
    runtime in place of slice 1's, and the SAME renderer and the SAME evidence
    recorder. It adds no calculation and no interpretation; every figure came
    from `execute_mi_query`, once per snapshot, inside
    `temporal.execute_temporal_plan`.

    THE CATALOGUE IS THE CALLER'S TO SUPPLY, exactly as the frame is on the
    slice 1 path. When none is supplied this returns None and the legacy
    envelope serves — which is what every production request does today, since
    `mi_service` wires no `SnapshotStore`. Choosing a catalogue here would be
    this module deciding which book a question is about.
    """
    eligible, why, detail = temporal.check_temporal_eligibility(plan)
    body["eligibility"] = {"eligible": eligible, "reason": why, "detail": detail,
                           "perimeter": "slice2_temporal"}
    if not eligible:
        body["execution"] = {"attempted": False, "why_not": f"{why}: {detail}"[:300]}
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{why}"

    if store is None or not snapshot_client_id:
        body["execution"] = {"attempted": False,
                             "why_not": "no governed snapshot catalogue was "
                                        "supplied for this request"}
        body["disposition"] = evidence.INELIGIBLE
        return None, TEMPORAL_STORE_UNAVAILABLE

    outcome = temporal.execute_temporal_plan(
        plan, store=store, client_id=snapshot_client_id, semantics=semantics,
        route=snapshot_route)
    body["execution"] = {"attempted": True,
                         "bound_spec": (outcome.spec.to_dict()
                                        if outcome.spec is not None else None),
                         "requested_semantics": dict(outcome.requested),
                         "temporal": outcome.to_dict()}

    if outcome.error:
        body["disposition"] = evidence.EXECUTION_ERROR
        return None, EXECUTION_FAILED
    if outcome.reason:
        # A period the catalogue cannot honour, a label it cannot settle, a
        # cadence it does not keep. Fail closed and let the legacy path answer;
        # nothing here narrows the window or substitutes a period.
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{TEMPORAL_NOT_RESOLVED}:{outcome.reason}"
    if not outcome.executed or not outcome.reconciled:
        body["disposition"] = evidence.EXECUTION_ERROR
        return None, RECONCILIATION_FAILED
    body["disposition"] = evidence.EXECUTED

    try:
        frame = temporal.series_frame(outcome, outcome.spec)
        # THE DISPLAY HINTS COME FROM THE BOOK, not from the series. The
        # renderer profiles a frame to learn that a balance is currency and an
        # LTV a percentage, and the stacked series carries only the aggregated
        # column. One selected snapshot is loaded for its SCHEMA; no figure in
        # the answer comes from it, and every figure was computed already.
        hints_frame = store.load_loans(outcome.points[-1].snapshot_id)
        result = _temporal_result(outcome, frame)
        payload = render(outcome.spec, result, semantics, hints_frame,
                         question=question, portfolio_id=render_portfolio_id,
                         as_of=as_of, requested=dict(outcome.requested),
                         executed=temporal.served_evidence(outcome),
                         execution_population=execution_population)
    except Exception as exc:                                         # noqa: BLE001
        body["execution"]["render_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return None, RENDER_FAILED
    if not isinstance(payload, Mapping) or not payload.get("ok"):
        body["execution"]["render_error"] = "the rendered envelope was not ok"
        return None, RENDER_FAILED
    return dict(payload), ""


def _temporal_result(outcome: Any, frame: Any) -> Any:
    """The stacked series, in the result type the response contract already takes.

    A `MIQueryResult` is a dataclass, and every field here is read off the
    governed outcome. The metadata is the UNION of what each snapshot applied,
    so the envelope's receipt says what ran on the series rather than on one
    period of it.
    """
    from mi_agent.mi_query_executor import MIQueryResult

    predicates: List[Dict[str, Any]] = []
    grouped: List[str] = []
    for point in outcome.points:
        for entry in (point.receipt.get("applied_predicates") or ()):
            if entry not in predicates:
                predicates.append(dict(entry))
        for key in (point.receipt.get("group_field_keys") or ()):
            if key not in grouped:
                grouped.append(str(key))
    first = outcome.points[0].receipt if outcome.points else {}
    return MIQueryResult(
        spec=outcome.spec,
        result_type="table",
        data=frame,
        row_count=int(len(frame)),
        metadata={
            "aggregation": first.get("aggregation"),
            "applied_predicates": predicates,
            "group_field_keys": grouped,
            "balance_field_used": first.get("balance_field_used"),
            "filtered_row_count": sum(
                int(p.receipt.get("filtered_row_count") or 0)
                for p in outcome.points),
            "temporal": temporal.served_evidence(outcome),
        })


def _attempt(body: Dict[str, Any], *, question: str, frame: Any, semantics: Any,
             render_portfolio_id: Optional[str], as_of: Optional[str],
             snapshot_store: Any = None,
             snapshot_client_id: Optional[str] = None,
             snapshot_route: Optional[str] = None,
             source_registry: Any = None,
             execution_population: Optional[str] = None,
             pipeline_source: Any = None, pipeline_root: Any = None,
             pipeline_client_id: Optional[str] = None,
             pipeline_history: Any = None,
             pipeline_run_id: Optional[str] = None
             ) -> Tuple[Optional[Dict[str, Any]], str]:
    """One serving attempt. `(payload or None, reason)`; fills `body` as it goes."""
    from mi_agent.interpretation_v2.outcomes import (OUTCOME_CLARIFY, OUTCOME_PLAN,
                                                     OUTCOME_REFUSE)

    # THIS CLIENT'S GOVERNED SOURCE PORTFOLIOS, and no other client's. Supplied
    # by the caller that authorised the request; `build_plan` uses it for one
    # compilation and never caches it. With none supplied a question naming a
    # portfolio is refused, which is where every caller was until the production
    # seam started passing one.
    outcome, compiled = wiring.build_plan(question,
                                          source_registry=source_registry)
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

    # SPECIALIST CAPABILITY DISPATCH, FIRST, and from the plan alone.
    #
    # A pipeline plan is not the funded runtime's to refuse: the population gate
    # below speaks for the FUNDED runtime and would answer
    # POPULATION_NOT_EXECUTABLE for a plan that has a perfectly good owner. So
    # the capability is read off the plan and dispatched before any funded
    # perimeter runs. `claims` reads `plan.capability` and nothing else — no
    # question, no dataset, no recogniser — and the six specialist capabilities
    # that have NOT been migrated match nothing here and fall through exactly as
    # they did, refusing with CAPABILITY_NOT_GENERIC.
    if pipeline_rt.claims(plan):
        return _attempt_pipeline(
            body, plan=plan, question=question,
            render_portfolio_id=render_portfolio_id, as_of=as_of,
            pipeline_source=pipeline_source, pipeline_root=pipeline_root,
            pipeline_client_id=pipeline_client_id,
            pipeline_history=pipeline_history, pipeline_run_id=pipeline_run_id)

    # WHICH POPULATION, BEFORE WHICH RUNTIME. Placed above the dispatch because
    # it is true of both: the temporal runtime reads the same funded route the
    # slice 1 executor reads a funded frame from, and neither may answer a
    # question about a population it did not load. Placed above EXECUTION for
    # the reason that matters — a refusal here means zero rows were touched, so
    # a pipeline question cannot produce a funded number even transiently, in
    # the evidence sink or anywhere else.
    base_ok, base_why, base_detail = adapter.check_population_base(
        plan, execution_population)
    if not base_ok:
        body["eligibility"] = {"eligible": False, "reason": base_why,
                               "detail": base_detail,
                               "perimeter": "population_base"}
        body["execution"] = {"attempted": False,
                             "why_not": f"{base_why}: {base_detail}"[:300]}
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{base_why}"

    # THE DISPATCH. One structural read of the plan's own period form, made by
    # `plan_temporal_runtime.claims`, decides which runtime owns it. The two
    # perimeters are disjoint by construction — slice 1 accepts `current` and
    # only `current`, slice 2 excludes it and only it — so this chooses between
    # them without deciding anything semantic, and no question is read to reach
    # it. A temporal plan the slice 2 contract does not admit is refused BY
    # slice 2, with a temporal reason, rather than falling through to slice 1 to
    # be refused for not being current.
    if temporal.claims(plan):
        return _attempt_temporal(
            body, plan=plan, question=question, semantics=semantics,
            store=snapshot_store, snapshot_client_id=snapshot_client_id,
            snapshot_route=snapshot_route,
            render_portfolio_id=render_portfolio_id, as_of=as_of,
            execution_population=execution_population)

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
                         portfolio_id=render_portfolio_id, as_of=as_of,
                         requested=body["execution"].get("requested_semantics"),
                         execution_population=execution_population)
    except Exception as exc:                                         # noqa: BLE001
        body["execution"]["render_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return None, RENDER_FAILED
    if not isinstance(payload, Mapping) or not payload.get("ok"):
        body["execution"]["render_error"] = "the rendered envelope was not ok"
        return None, RENDER_FAILED
    return dict(payload), ""

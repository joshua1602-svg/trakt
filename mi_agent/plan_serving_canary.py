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
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from mi_agent import plan_runtime_adapter as adapter
from mi_agent import plan_shadow_evidence as evidence
from mi_agent import plan_shadow_wiring as wiring
from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent import plan_forecast_runtime as forecast_rt
from mi_agent import plan_stage_movement_runtime as stage_rt
from mi_agent import plan_runtime_registry as runtime_registry
from mi_agent import plan_temporal_runtime as temporal
from mi_agent import plan_material_summary as material_summary
from mi_agent import plan_attribution as attribution
from mi_agent import plan_metric_delta as metric_delta

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
#: The plan names a governed field the book being answered does not carry.
FIELD_NOT_IN_BOOK = "FIELD_NOT_IN_BOOK"
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
#: A change-intelligence plan arrived and the caller supplied no governed source
#: root or no tenant. Not an error and not a refusal: the legacy path serves,
#: exactly as it does for a temporal plan with no catalogue. This module
#: discovers nothing — choosing an output root here would be it deciding which
#: client's book a question is about.
CHANGE_OWNER_INPUTS_UNAVAILABLE = "CHANGE_OWNER_INPUTS_UNAVAILABLE"


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
    # THE EXECUTION RECEIPT, from the one owner every funded answer's receipt
    # line comes from (`execution_receipt.build_receipt`): what was measured,
    # over which loans, grouped how, and AS AT the book's own cut-off — D4 on
    # the answer itself. Built from the executed result and the frame, never
    # from the question; the governed coverage owner, not the legacy semantic
    # guard, judges it. A series is described by the temporal runtime's own
    # evidence instead, so only a single-snapshot answer gets one here.
    if executed is None:
        try:
            from mi_agent import execution_receipt as receipt_mod
            from mi_agent.mi_agent_workflow import reporting_date_label
            workflow["execution_receipt"] = receipt_mod.build_receipt(
                spec=spec, query_result=result,
                semantics=dict(semantics) if isinstance(semantics, Mapping) else {},
                facets=(), dataset=execution_population,
                period=reporting_date_label(frame), frame=frame).to_dict()
        except Exception as exc:                                     # noqa: BLE001
            # A disclosure must never cost an answer that would otherwise stand.
            warnings.append(f"Receipt not rendered: {type(exc).__name__}")
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
          pipeline_history: Any = None,
          # THE GOVERNED FUNDED FRAME, as a resolver `f(client_id, run_id)` —
          # the one `mi_service` already hands the legacy analytical route. The
          # forecast composer reads the funded book through it (D6). Supplied,
          # never discovered here.
          funded_frame_resolver: Any = None,
          # THE CHANGE-INTELLIGENCE OWNERS' INPUTS, resolved by the caller that
          # already owns discovery and authorisation. `output_root` is the
          # governed source root `mi_service` already resolves for its routed
          # branch; `tenant_id` comes from the execution context and never from
          # request data; `authorised_portfolio_ids` is what the caller
          # authorised. This module discovers none of them and substitutes none
          # of them — with any of them missing the legacy envelope serves.
          output_root: Optional[str] = None,
          tenant_id: Optional[str] = None,
          authorised_portfolio_ids: Tuple[str, ...] = (),
          ) -> Optional[Dict[str, Any]]:
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
            # THE CLIENT'S GOVERNED SOURCE PORTFOLIOS. Accepted above and, from
            # e6e16c63 until this line was restored, never forwarded: every
            # production compilation ran with no registry, so a question naming
            # a portfolio was refused on the governed path however well the
            # production seam built one. A structural test now asserts that
            # every input `serve` shares with `_attempt` is passed through.
            source_registry=source_registry,
            pipeline_source=pipeline_source, pipeline_root=pipeline_root,
            pipeline_client_id=pipeline_client_id,
            pipeline_history=pipeline_history, pipeline_run_id=run_id,
            funded_frame_resolver=funded_frame_resolver,
            client_id=client_id, output_root=output_root, tenant_id=tenant_id,
            authorised_portfolio_ids=tuple(authorised_portfolio_ids),
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
                      question: str, semantics: Any,
                      render_portfolio_id: Optional[str],
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
    dated_plan = pipeline_rt.is_dated(plan)
    body["execution"] = {"attempted": True,
                         "runtime": ("pipeline_temporal" if temporal_plan
                                     else "pipeline_dated" if dated_plan
                                     else "pipeline_current"),
                         "requested_semantics": adapter.requested_semantics(plan)}
    try:
        if temporal_plan:
            outcome = pipeline_rt.execute_temporal(
                plan, root=pipeline_root, client_id=pipeline_client_id or "",
                to_run_id=pipeline_run_id, history_model=pipeline_history)
        elif dated_plan:
            # Named months (D7) and "latest against previous", from the same
            # weekly owner; the runtime only chooses which extracts.
            outcome = pipeline_rt.execute_dated(
                plan, root=pipeline_root, client_id=pipeline_client_id or "",
                to_run_id=pipeline_run_id, history_model=pipeline_history)
        else:
            outcome = pipeline_rt.execute_current(
                plan, source=pipeline_source, semantics=semantics,
                history_model=pipeline_history)
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
    # The weighted expected funded amount is money too.
    is_amount = receipt.get("measure_kind") in ("amount", "weighted")
    axis = (receipt.get("group_field_keys") or [None])[0]
    spec_dict = {"capability": receipt.get("capability"),
                 "population": receipt.get("population_base"),
                 "measure": measure, "dimensions": receipt.get("group_field_keys")}
    shape = str(receipt.get("result_shape") or "")
    # WHICH PIPELINE. The runtime answers over the live pipeline by the
    # dashboard's own rule and states it on the receipt, with the sentence the
    # Pipeline owner writes about it; this only prints what it was handed.
    scope = receipt.get("pipeline_scope") or {}
    label = _PIPELINE_LABELS.get(measure, measure)
    phrase = ("live " if scope.get("population") == "open" else "") + label.lower()

    def _shown(value: Any) -> str:
        if value is None:
            return "n/a"
        return _money(value) if is_amount else f"{float(value):,.0f}"

    # D4: THE EXTRACT THE FIGURE WAS READ FROM, on the sentence. A current
    # figure is as at one weekly extract — the dataset the runtime read.
    extract = str((receipt.get("dataset") or {}).get("as_of_date") or "")
    as_at = f", as at the weekly extract of {extract}" if extract else ""

    if shape == "scalar":
        value = float(outcome.value)
        kpis = [{"field": measure, "label": label,
                 "value": (f"£{value:,.0f}" if is_amount else f"{value:,.0f}"),
                 "rawValue": value}]
        artefacts = [_artefact("kpi", "Pipeline", kpis=kpis,
                               description="Governed pipeline extract.")]
        timing = receipt.get("timing") or {}
        if timing:
            # D11: the bucket, stated against the extract's month.
            answer = (f"The {phrase} {_timing_phrase(timing)} is "
                      f"{_shown(value)}{as_at}.")
        else:
            answer = f"The {phrase} is {_shown(value)}{as_at}."
    elif shape == "grouped":
        # Months read in date order; every other breakdown largest first, so
        # "which is largest" is answered by the sentence itself.
        rows = sorted(({str(axis): c[axis], "value": c["value"]}
                       for c in outcome.cells),
                      key=(lambda r: r[str(axis)])
                      if axis == "expected_completion_month"
                      else (lambda r: -(r["value"] or 0.0)))
        axis_label = _PIPELINE_AXES.get(str(axis), str(axis).replace("_", " "))
        artefacts = [_artefact(
            "table", f"Pipeline by {axis_label}", rows=rows,
            columns=[{"key": str(axis), "label": axis_label.capitalize()},
                     {"key": "value", "label": label}],
            description=f"{len(rows)} rows.")]
        # The sentence names the leaders; a long breakdown (hundreds of
        # brokers) is capped in the sentence and complete in the table.
        named = (_stage_name if axis == "pipeline_stage" else str)
        shown = rows[:_SENTENCE_ROWS]
        answer = (f"The {phrase} by {axis_label}: "
                  + ", ".join(f"{named(r[str(axis)])} {_shown(r['value'])}"
                              for r in shown)
                  + (f", and {len(rows) - len(shown):,} more"
                     if len(rows) > len(shown) else "") + f"{as_at}.")
    elif shape in ("dated", "grouped_dated"):
        # THE PIPELINE AT NAMED DATES. D4: the measure and every extract date
        # are in the sentence; the rule that chose each extract (D7 for a
        # month) is in the source notes. No change between them is stated,
        # because none was computed — the plan asked for the dates' figures.
        dates = [str(r.get("extract_date"))
                 for r in (receipt.get("period_resolution") or ())]
        if shape == "dated":
            rows = [{"period": str(c["period"]), "value": c["value"]}
                    for c in outcome.cells]
            values = "; ".join(f"{r['period']} {_shown(r['value'])}" for r in rows)
            answer = f"The {phrase} at each weekly extract — {values}."
            columns = [{"key": "period", "label": "Weekly extract"},
                       {"key": "value", "label": label}]
        else:
            stages = sorted({str(c[axis]) for c in outcome.cells})
            by_period: Dict[str, Dict[str, Any]] = {}
            for c in outcome.cells:
                by_period.setdefault(str(c["period"]), {"period": str(c["period"])})[
                    str(c[axis])] = c["value"]
            rows = [by_period[d] for d in sorted(by_period)]
            answer = (f"The {phrase} by stage at the weekly extracts of "
                      f"{' and '.join(dates)}: "
                      f"{', '.join(_stage_name(st) for st in stages)}.")
            columns = ([{"key": "period", "label": "Weekly extract"}]
                       + [{"key": st, "label": st} for st in stages])
        artefacts = [_artefact("table", "Pipeline at the named dates",
                               rows=rows, columns=columns,
                               description=f"{len(rows)} weekly extract(s).")]
    else:
        # A SERIES, weekly. One row per governed extract; a grouped series gets
        # one column per stage, which is the shape the accepted evolution route
        # already publishes.
        periods = sorted({str(c["period"]) for c in outcome.cells})
        # A MONTHLY series is one point per month — D7's extract, the last
        # weekly one dated within the month — and says so; the rule is in the
        # source notes. Otherwise the series is weekly and says that.
        span = (f"{len(periods)} month(s), at the last weekly extract of each"
                if receipt.get("grain") == "monthly"
                else f"{len(periods)} weekly extract(s)")
        if axis:
            stages = sorted({str(c[axis]) for c in outcome.cells})
            rows = [{"period": per,
                     **{st: sum(float(c["value"] or 0) for c in outcome.cells
                                if str(c["period"]) == per and str(c[axis]) == st)
                        for st in stages}}
                    for per in periods]
            series = [{"key": st, "label": st} for st in stages]
            # The first and last extract, so the series says when it spans.
            window = (f" ({periods[0]} to {periods[-1]})"
                      if len(periods) > 1 else f" ({periods[0]})" if periods else "")
            answer = (f"The {phrase} by stage across {span}{window}: "
                      f"{', '.join(_stage_name(st) for st in stages)}.")
        else:
            rows = sorted(({"period": str(c["period"]), "value": c["value"]}
                           for c in outcome.cells), key=lambda r: r["period"])
            series = [{"key": "value", "label": label}]
            answer = f"The {phrase} across {span}"
            if rows:
                answer += (f", from {_shown(rows[0]['value'])} at {rows[0]['period']} "
                           f"to {_shown(rows[-1]['value'])} at {rows[-1]['period']}")
            answer += "."
        artefacts = [_artefact(
            "chart", "Pipeline over time", chartType="line", xKey="period",
            rows=rows, series=series,
            valueFormat=("gbp" if is_amount else "number"), displayHints={})]

    reconciliation = {"dataset": "pipeline", "coverage_by_balance_pct": 100.0}
    for artefact in artefacts:
        artefact.setdefault("reconciliation", reconciliation)
    # WHICH EXTRACT ANSWERED WHICH REQUESTED DATE, and by what rule — D7 for a
    # named month. Empty for every shape that selects no dates.
    source_notes = [{"field": f"period: {row.get('requested')}",
                     "note": f"{row.get('extract_date')} — {row.get('rule')}"}
                    for row in (receipt.get("period_resolution") or ())]
    if receipt.get("month_rule"):
        source_notes.append({"field": "grain: monthly",
                             "note": str(receipt["month_rule"])})
    # WHAT THE FIGURE LEAVES OUT, on the sentence and in the notes — the same
    # disclosure the Pipeline tab makes, in the Pipeline owner's words.
    if scope.get("note"):
        answer = f"{answer} {scope['note']}."
        source_notes.append({"field": "population", "note": scope["note"]})
    # WHICH REGIONS: the client's reporting taxonomy, and the live cases whose
    # extract region it could not place — the tab's own disclosure.
    region = receipt.get("region_basis") or {}
    if region:
        note = (f"Regions are the client's reporting regions"
                + (f" ({region['taxonomy']})" if region.get("taxonomy") else ""))
        if region.get("unmappedCaseCount"):
            note += (f"; {int(region['unmappedCaseCount']):,} case(s) "
                     f"({_money(region.get('unmappedAmount') or 0.0)}) whose "
                     f"extract region has no governed mapping are in no region")
        answer = f"{answer} {note}."
        source_notes.append({"field": "region", "note": note})
    # The expected-completion view counts the cases carrying a forecast; say so.
    if receipt.get("completion_basis"):
        answer = f"{answer} Counted over {receipt['completion_basis']}."
        source_notes.append({"field": "expected completion",
                             "note": str(receipt["completion_basis"])})
    payload: Dict[str, Any] = {
        "ok": True, "error": None, "question": question, "answer": answer,
        "interpreted": "", "spec": spec_dict,
        "validation": {"ok": True, "errors": [], "warnings": [],
                       "resolved_fields": {}},
        "artifacts": artefacts, "reconciliation": reconciliation,
        "sourceNotes": source_notes, "warnings": [], "diagnostics": [],
        "assumptions": [],
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


def _attempt_stage_movement(body: Dict[str, Any], *, plan: Mapping[str, Any],
                            question: str, pipeline_root: Any,
                            pipeline_client_id: Optional[str],
                            pipeline_history: Any,
                            render_portfolio_id: Optional[str],
                            as_of: Optional[str]
                            ) -> Tuple[Optional[Dict[str, Any]], str]:
    """One STAGE MOVEMENT serving attempt. Same contract as `_attempt`.

    Perimeter, prove the population, execute, render, record — with the stage
    movement runtime's own perimeter and declaration, and the pipeline inputs
    the legacy stage movement route hands the same owner.
    """
    eligible, why, detail = stage_rt.check_eligibility(plan)
    body["eligibility"] = {"eligible": eligible, "reason": why, "detail": detail,
                           "perimeter": "stage_movement_specialist"}
    if not eligible:
        body["execution"] = {"attempted": False, "why_not": f"{why}: {detail}"[:300]}
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{why}"

    base_ok, base_why, base_detail = adapter.check_population_base(
        plan, stage_rt.EXECUTION_POPULATION,
        executable=stage_rt.EXECUTABLE_POPULATIONS)
    if not base_ok:
        body["eligibility"] = {"eligible": False, "reason": base_why,
                               "detail": base_detail,
                               "perimeter": "stage_movement_population"}
        body["execution"] = {"attempted": False,
                             "why_not": f"{base_why}: {base_detail}"[:300]}
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{base_why}"

    body["execution"] = {"attempted": True, "runtime": "stage_movement",
                         "requested_semantics": adapter.requested_semantics(plan)}
    try:
        outcome = stage_rt.execute(plan, root=pipeline_root,
                                   client_id=pipeline_client_id or "",
                                   history_model=pipeline_history)
    except Exception as exc:                                         # noqa: BLE001
        body["execution"]["error"] = f"{type(exc).__name__}: {exc}"[:300]
        body["disposition"] = evidence.EXECUTION_ERROR
        return None, EXECUTION_FAILED
    if not outcome.ok:
        body["execution"].update({"attempted": False,
                                  "why_not": f"{outcome.reason}: {outcome.detail}"[:300]})
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{outcome.reason}"

    body["execution"].update({"receipt": dict(outcome.receipt),
                              "grouped_cells": outcome.rows,
                              "row_count": len(outcome.rows)})
    body["disposition"] = evidence.EXECUTED
    try:
        payload = render_stage_movement(plan, outcome, question=question,
                                        portfolio_id=render_portfolio_id,
                                        as_of=as_of)
    except Exception as exc:                                         # noqa: BLE001
        body["execution"]["render_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return None, RENDER_FAILED
    return dict(payload), ""


def render_stage_movement(plan: Mapping[str, Any], outcome: Any, *,
                          question: str, portfolio_id: Optional[str],
                          as_of: Optional[str]) -> Dict[str, Any]:
    """The stage movement result, in the envelope every channel renders.

    THE SENTENCE IS THE OWNER'S. `stage_movement_query.compose` worded it from
    the governed payload, and it already states the measure and the window
    ("between <prior extract> and <latest extract>") — D4 on the sentence.
    """
    import uuid
    from datetime import datetime, timezone

    receipt = dict(outcome.receipt)
    dataset = receipt.get("dataset") or {}
    spec_dict = {"capability": receipt.get("capability"),
                 "population": receipt.get("population_base"),
                 "operation": receipt.get("operation"),
                 "measure": receipt.get("measure_concept")}
    artefacts: List[Dict[str, Any]] = []
    if outcome.rows:
        artefacts.append({
            "id": f"art_{uuid.uuid4().hex[:8]}", "type": "table",
            "title": "Governed stage movement", "rows": outcome.rows,
            "columns": outcome.columns,
            "description": f"{len(outcome.rows)} row(s).",
            "source": {"engine": "mi_agent.governed_plan",
                       "label": "MI Agent · table", "spec": spec_dict,
                       "asOf": dataset.get("as_of_date") or as_of,
                       "portfolio": portfolio_id},
            "createdAt": datetime.now(timezone.utc).isoformat(), "mock": False})
    reconciliation = {"dataset": "pipeline", "coverage_by_balance_pct": 100.0}
    for artefact in artefacts:
        artefact["reconciliation"] = reconciliation
    notes = [{"field": "stage_movement",
              "note": (f"{receipt.get('execution_owner')}: weekly extracts "
                       f"{dataset.get('comparison_date')} and "
                       f"{dataset.get('as_of_date')}")}]
    return {
        "ok": True, "error": None, "question": question, "answer": outcome.answer,
        "interpreted": "", "spec": spec_dict,
        "validation": {"ok": True, "errors": [], "warnings": [],
                       "resolved_fields": {}},
        "artifacts": artefacts, "reconciliation": reconciliation,
        "sourceNotes": notes, "warnings": [], "diagnostics": [],
        "assumptions": [],
        "metadata": {"engine": "mi_agent", "source": "python", "mock": False,
                     "route": "governed_plan_stage_movement", "lensApplied": None,
                     "parserMode": "governed_plan",
                     "governedPlan": {
                         "requested": dict(adapter.requested_semantics(plan)),
                         "executed": receipt}},
    }


def _attempt_forecast(body: Dict[str, Any], *, plan: Mapping[str, Any],
                      question: str, client_id: Optional[str],
                      output_root: Optional[str], pipeline_root: Any,
                      pipeline_history: Any, run_id: Optional[str],
                      render_portfolio_id: Optional[str], as_of: Optional[str],
                      funded_frame_resolver: Any = None, semantics: Any = None
                      ) -> Tuple[Optional[Dict[str, Any]], str]:
    """One FORECAST serving attempt. Same contract as `_attempt`.

    Stage for stage the pipeline attempt — perimeter, prove the population,
    execute, render, record — with the forecast runtime's perimeter and its own
    population declaration. The inputs are the ones the legacy forecast route
    hands the same owner: the governed funded root, the pipeline discovery root,
    the client's historical model and the selected run. None is discovered here,
    and with any missing the runtime refuses and the legacy envelope serves.
    """
    eligible, why, detail = forecast_rt.check_eligibility(plan)
    body["eligibility"] = {"eligible": eligible, "reason": why, "detail": detail,
                           "perimeter": "forecast_specialist"}
    if not eligible:
        body["execution"] = {"attempted": False, "why_not": f"{why}: {detail}"[:300]}
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{why}"

    base_ok, base_why, base_detail = adapter.check_population_base(
        plan, forecast_rt.execution_population(plan),
        executable=forecast_rt.EXECUTABLE_POPULATIONS)
    if not base_ok:
        body["eligibility"] = {"eligible": False, "reason": base_why,
                               "detail": base_detail,
                               "perimeter": "forecast_population"}
        body["execution"] = {"attempted": False,
                             "why_not": f"{base_why}: {base_detail}"[:300]}
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{base_why}"

    body["execution"] = {"attempted": True, "runtime": "forecast",
                         "requested_semantics": adapter.requested_semantics(plan)}
    try:
        outcome = forecast_rt.execute(
            plan, output_root=output_root, pipeline_root=pipeline_root,
            client_id=client_id or "", run_id=run_id,
            history_model=pipeline_history,
            funded_frame_resolver=funded_frame_resolver,
            semantics=semantics if isinstance(semantics, Mapping) else None)
    except Exception as exc:                                         # noqa: BLE001
        body["execution"]["error"] = f"{type(exc).__name__}: {exc}"[:300]
        body["disposition"] = evidence.EXECUTION_ERROR
        return None, EXECUTION_FAILED
    if not outcome.ok:
        body["execution"].update({"attempted": False,
                                  "why_not": f"{outcome.reason}: {outcome.detail}"[:300]})
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{outcome.reason}"

    body["execution"].update({"value": outcome.value,
                              "receipt": dict(outcome.receipt)})
    body["disposition"] = evidence.EXECUTED

    try:
        payload = render_forecast(plan, outcome, question=question,
                                  portfolio_id=render_portfolio_id, as_of=as_of)
    except Exception as exc:                                         # noqa: BLE001
        body["execution"]["render_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return None, RENDER_FAILED
    if not isinstance(payload, Mapping) or not payload.get("ok"):
        body["execution"]["render_error"] = "the rendered envelope was not ok"
        return None, RENDER_FAILED
    return dict(payload), ""


def _money(value: Any) -> str:
    """A reader-facing GBP amount. Presentation only: no figure is derived."""
    amount = float(value)
    for size, unit in ((1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if abs(amount) >= size:
            return f"£{amount / size:,.1f}{unit}"
    return f"£{amount:,.0f}"


def _as_at_clause(inputs: Mapping[str, Any]) -> str:
    """D4 + D1: the as-at date of EVERY input the figure used, on the sentence.

    One input, one date. Two inputs, both dates, each named — a forecast that
    composes a funded snapshot with a pipeline extract and states one date has
    not said what it is as at.
    """
    parts = [f"{row.get('label') or name} {row.get('as_of')}"
             for name, row in sorted(inputs.items())]
    return "As at " + " and ".join(parts) + "."


def render_forecast(plan: Mapping[str, Any], outcome: Any, *, question: str,
                    portfolio_id: Optional[str], as_of: Optional[str]
                    ) -> Dict[str, Any]:
    """The forecast result, in the envelope every channel already renders.

    D4: the MEASURE and its AS-AT are in the sentence itself, every time. D1:
    when the figure composes two datasets, both vintages are. Everything else
    the owner said — the completion signal it used, its scenario basis, its
    caveats, the path of the figure in its output — is published in
    `sourceNotes` and `warnings`, not dropped.

    THE WORDS COME FROM THE RECEIPT. A milestone's state is the owner's
    `milestone_answer`; every other figure's label, unit and companion figures
    are the semantic model's, and its value the owner's. Nothing is re-derived
    here, including the verb: "already reached", "around <date>" and "beyond
    the projection horizon" map one-to-one onto the owner's states.
    """
    import uuid
    from datetime import datetime, timezone

    receipt = dict(outcome.receipt)
    kind = str(receipt.get("measure_kind") or "")
    measure = str(receipt.get("measure_concept") or "")
    inputs = dict(receipt.get("inputs") or {})
    as_at = _as_at_clause(inputs)
    signal = (receipt.get("completion_signal") or {}).get("description") or ""
    spec_dict = {"capability": receipt.get("capability"),
                 "population": receipt.get("population_base"),
                 "measure": measure, "operation": receipt.get("operation"),
                 "dimensions": receipt.get("group_field_keys") or []}

    def _artefact(kind_: str, title: str, **rest: Any) -> Dict[str, Any]:
        return {"id": f"art_{uuid.uuid4().hex[:8]}", "type": kind_,
                "title": title,
                "source": {"engine": "mi_agent.governed_plan",
                           "label": f"MI Agent · {kind_}", "spec": spec_dict,
                           "asOf": as_of, "portfolio": portfolio_id},
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "mock": False, **rest}

    warnings = [str(c) for c in (receipt.get("caveats") or ())]
    notes: List[Dict[str, str]] = []
    if kind == forecast_rt.KIND_MILESTONE:
        answer, kpis, title = _milestone_sentence(receipt, outcome, as_at)
        artefacts = [_artefact("kpi", title, kpis=kpis,
                               description="Governed forecast.")]
        if kpis[0]["rawValue"] is not None:
            # Only a PROJECTED date has bands to qualify; "already reached" and
            # "beyond the horizon" state no banded figure.
            warnings.append(_BANDS_WARNING)
    else:
        answer, artefacts, banded = _catalogue_answer(receipt, outcome, as_at,
                                                      _artefact)
        if banded:
            warnings.append(_BANDS_WARNING)
        notes.extend(_catalogue_notes(receipt))

    if signal:
        notes.insert(0, {"field": "completion_signal", "note": signal})
    if receipt.get("scenario_basis"):
        notes.append({"field": "scenario_basis",
                      "note": str(receipt.get("scenario_basis"))})
    for name, row in sorted(inputs.items()):
        notes.append({"field": f"input:{name}",
                      "note": f"{row.get('owner')} as at {row.get('as_of')}"})
    if receipt.get("input_vintage_skew_days") is not None:
        notes.append({"field": "input_vintage_skew",
                      "note": (f"{receipt['input_vintage_skew_days']} days between "
                               f"inputs; ceiling "
                               f"{receipt.get('input_vintage_ceiling_days')} days")})

    reconciliation = {"dataset": "forecast",
                      "inputs": sorted(inputs), "coverage_by_balance_pct": 100.0}
    for artefact in artefacts:
        artefact["reconciliation"] = reconciliation
    payload: Dict[str, Any] = {
        "ok": True, "error": None, "question": question, "answer": answer,
        "interpreted": "", "spec": spec_dict,
        "validation": {"ok": True, "errors": [], "warnings": [],
                       "resolved_fields": {}},
        "artifacts": artefacts, "reconciliation": reconciliation,
        "sourceNotes": notes, "warnings": warnings, "diagnostics": [],
        "assumptions": [],
        "metadata": {"engine": "mi_agent", "source": "python", "mock": False,
                     "route": "governed_plan_forecast", "lensApplied": None,
                     "parserMode": "governed_plan",
                     "governedPlan": {
                         "requested": dict(adapter.requested_semantics(plan)),
                         "executed": receipt}},
    }
    return payload


_BANDS_WARNING = ("Downside/base/upside are indicative scenario bands, not "
                  "statistically validated confidence intervals.")


def _milestone_sentence(receipt: Mapping[str, Any], outcome: Any, as_at: str
                        ) -> Tuple[str, List[Dict[str, Any]], str]:
    """A milestone for a stated threshold, in the owner's own state."""
    from mi_agent_api import forecast_extrapolation as fx_mod

    measure = str(receipt.get("measure_concept") or "")
    base_rate = receipt.get("base_monthly_run_rate")
    target = receipt.get("target") or {}
    scale = receipt.get("scale") or None
    threshold = _money(receipt.get("threshold_applied", target.get("value")))
    state = str(receipt.get("milestone_state") or "")
    row = receipt.get("milestone") or {}
    label = f"Forecast milestone date (funded balance reaching {threshold})"
    # D9: a question about SCALE is answered against the portfolio's own
    # threshold, and the answer says which rule set it and what it is
    # measured on — the reader never has to know the number to ask.
    if scale:
        label = (f"Scale ({scale.get('decision')}): for a "
                 f"{scale.get('stage_label')}, scale is {threshold}, measured "
                 f"on {scale.get('measured_on')}")
    if state == fx_mod.MILESTONE_ALREADY_REACHED:
        answer = (f"{label}: already reached — the funded balance is "
                  f"{_money(receipt.get('current_funded_balance'))}"
                  + (", so the portfolio is at scale" if scale else "")
                  + f". {as_at}")
        kpi_value = "at scale" if scale else "reached"
    elif state == fx_mod.MILESTONE_PROJECTED:
        to_go = (f", {_money(receipt.get('gap_to_threshold'))} to go"
                 if receipt.get("gap_to_threshold") else "")
        answer = (f"{label}: around {row.get('baseDate')} at the base "
                  f"completion run-rate of {_money(base_rate)}/month "
                  f"(downside {row.get('downsideDate')}, upside "
                  f"{row.get('upsideDate')}), from a funded balance of "
                  f"{_money(receipt.get('current_funded_balance'))}{to_go}. "
                  f"{as_at}")
        kpi_value = str(row.get("baseDate"))
    else:
        answer = (f"{label}: beyond the projection horizon, so no date is "
                  f"given. The funded balance is "
                  f"{_money(receipt.get('current_funded_balance'))}. {as_at}")
        kpi_value = "beyond horizon"
    kpis = [{"field": measure, "label": label, "value": kpi_value,
             "rawValue": outcome.value}]
    return answer, kpis, "Forecast milestone"


#: Reader-facing names for the axes a forecast figure is broken down by.
#: Presentation only.
_FORECAST_AXES = {
    "forecast_component": "component", "forecast_scenario": "scenario",
    "funding_threshold": "funding threshold",
    "weighting_exclusion_reason": "reason",
    "canonical_region_reporting": "region", "ltv_bucket": "LTV band",
}


def _shown_in(unit: str, value: Any) -> str:
    """A figure in its semantic-model unit. Presentation only."""
    if value is None:
        return "no date" if unit == "month" else "n/a"
    if unit == "gbp":
        return _money(value)
    if unit == "gbp_per_month":
        return f"{_money(value)}/month"
    if unit == "gbp_per_year":
        return f"{_money(value)}/year"
    if unit == "count":
        return f"{float(value):,.0f}"
    return str(value)


def _explained(receipt: Mapping[str, Any]) -> str:
    """The semantic model's `explain` sentence, with the owner's companion
    figures in it — money, or a count for a `*_count` figure."""
    import string

    template = str(receipt.get("explain") or "")
    if not template:
        return ""
    companions = receipt.get("context") or {}
    values = {}
    for _, key, _, _ in string.Formatter().parse(template):
        if key:
            raw = companions.get(key)
            values[key] = ("n/a" if raw is None
                           else f"{float(raw):,.0f}" if key.endswith("_count")
                           else _money(raw))
    return template.format(**values)


def _words(value: Any) -> str:
    return str(value).replace("_", " ")


def _catalogue_answer(receipt: Mapping[str, Any], outcome: Any, as_at: str,
                      make: Any) -> Tuple[str, List[Dict[str, Any]], bool]:
    """A semantic-model figure: `(sentence, artefacts, banded)`.

    D4 — the measure and its as-at in the sentence. The label and unit are the
    semantic model's; the numbers the owner's; the order of a breakdown the
    owner's (a month-ordered curve, the ladder low to high)."""
    label = str(receipt.get("measure_label") or receipt.get("measure_concept"))
    unit = str(receipt.get("unit") or "")
    shape = str(receipt.get("result_shape") or "")
    member = receipt.get("member") or None
    axis = (receipt.get("group_field_keys") or [None])[0]
    banded = False

    if shape == "series":
        columns = list(receipt.get("series_columns") or ())
        rows = [dict(c) for c in (outcome.cells or ())]
        first, last = rows[0], rows[-1]
        bands = ", ".join(f"{_words(c)} {_shown_in(unit, last.get(c))}"
                          for c in columns)
        horizon = receipt.get("horizon") or {}
        span = (f"over the next {horizon.get('periods_ahead')} month(s)"
                if horizon.get("periods_ahead") else
                f"over the owner's {horizon.get('published_months')}-month horizon")
        rate = receipt.get("base_monthly_run_rate")
        answer = (f"{label} {span}, {first.get('period')} to "
                  f"{last.get('period')}: {bands} by {last.get('period')}"
                  + (f", at a base completion run-rate of "
                     f"{_shown_in('gbp_per_month', rate)}" if rate is not None else "")
                  + f". {as_at}")
        artefacts = [make("chart", label, chartType="line", xKey="period",
                          rows=rows,
                          series=[{"key": c, "label": _words(c).capitalize()}
                                  for c in columns],
                          valueFormat="gbp")]
        return answer, artefacts, len(columns) > 1 or bool(member)

    if shape == "grouped":
        axis_label = _FORECAST_AXES.get(str(axis), _words(axis))
        rows = [dict(c) for c in (outcome.cells or ())]
        parts = ", ".join(f"{_words(r.get(axis))} {_shown_in(unit, r.get('value'))}"
                          for r in rows)
        answer = f"{label} by {axis_label}: {parts}."
        basis = receipt.get("axis_basis") or {}
        if basis.get("field"):
            unplaced = basis.get("unplacedForecastAmount") or 0.0
            answer = (f"{answer} Regions are the client's reporting regions"
                      + (f"; {_money(unplaced)} of the forecast has no region "
                         f"and is in none" if unplaced else "") + ".")
        answer = f"{answer} {as_at}"
        also = [k for k in (rows[0] if rows else {}) if k not in (axis, "value")]
        columns = ([{"key": str(axis), "label": axis_label.capitalize()},
                    {"key": "value", "label": label}]
                   + [{"key": k, "label": _words(k)} for k in also])
        artefacts = [make("table", f"{label} by {axis_label}", rows=rows,
                          columns=columns, description=f"{len(rows)} rows.")]
        banded = axis == "forecast_scenario" or axis == "funding_threshold"
        return answer, artefacts, banded

    value = outcome.value
    if member:
        answer = (f"{label} ({_FORECAST_AXES.get(member['dimension'], _words(member['dimension']))}: "
                  f"{_words(member['value'])}): {_shown_in(unit, value)}. {as_at}")
        banded = member.get("dimension") == "forecast_scenario"
    else:
        explained = _explained(receipt)
        answer = (f"{label}: {_shown_in(unit, value)}"
                  + (f" — {explained}" if explained else "") + f". {as_at}")
    kpis = [{"field": str(receipt.get("measure_concept")), "label": label,
             "value": _shown_in(unit, value), "rawValue": value}]
    artefacts = [make("kpi", label, kpis=kpis, description="Governed forecast.")]
    return answer, artefacts, banded


def _catalogue_notes(receipt: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Where the figure came from: the owner, the path, the model entry."""
    notes = [{"field": "owner",
              "note": f"{receipt.get('execution_owner')} → {receipt.get('read_path')}"},
             {"field": "semantic_model",
              "note": f"{receipt.get('semantic_model')}: {receipt.get('measure_concept')}"}]
    if receipt.get("definition_decision"):
        notes.append({"field": "definition",
                      "note": f"owner decision {receipt.get('definition_decision')}"})
    return notes


#: Reader-facing names for the governed pipeline measures. Presentation only.
def _stage_name(stage: Any) -> str:
    """A stage as a reader writes it: KFI, Application, Offer."""
    text = str(stage or "").strip().upper()
    return text if text == "KFI" else text.title()


#: How a pipeline breakdown's axis reads in a sentence.
_PIPELINE_AXES = {
    "pipeline_stage": "stage", "broker_channel": "broker",
    "erm_product_type": "product", "ltv_bucket": "LTV band",
    "canonical_region_reporting": "region",
    "expected_completion_month": "expected completion month"}

#: At most this many groups are named in a sentence; the table has them all.
_SENTENCE_ROWS = 10


def _timing_phrase(timing: Mapping[str, Any]) -> str:
    """D11's bucket in words, against the extract's month."""
    as_of = timing.get("as_of_month")
    value = timing.get("value")
    if value == "overdue":
        return f"overdue (expected to complete before {as_of})"
    if value == "current_month":
        return f"expected to complete this month ({as_of})"
    nxt = timing.get("next_month")
    return (f"expected to complete next month ({nxt})" if nxt
            else "expected to complete after this month (no later month "
                 "carries a completion)")


_PIPELINE_LABELS = {
    "pipeline_amount": "Pipeline amount",
    "pipeline_case_count": "Pipeline case count",
    "weighted_expected_funded_amount": "Pipeline weighted expected funded amount",
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


def _change_form_owner(plan: Mapping[str, Any]) -> Any:
    """The adapter that CLAIMS this plan's analytical form, or None.

    Each adapter's `claims` reads `provenance.compiler_bindings["change_form"]`
    and nothing else — the COMPILER's derived reading, never the model's raw
    claim and never the question. The two forms are disjoint by construction
    because one plan carries one form, so this chooses between them without
    deciding anything semantic.

    `level_comparison` is deliberately absent, and stays absent. It is already
    served by `plan_temporal_runtime`, which the live canary confirmed on two
    questions; claiming it here for symmetry would reorganise a working path.

    `metric_delta` was absent for one sprint and is present now. The live canary
    measured what its absence cost: a flawless reading — the form, the capability,
    the measure and a stated pair — compiled to a plan, claimed by the temporal
    runtime on period form alone, and refused CAPABILITY_NOT_GENERIC by a runtime
    that speaks for generic single-measure evaluation and never for its owner.
    """
    for owner in (material_summary, attribution, metric_delta):
        if owner.claims(plan):
            return owner
    return None


def _attempt_change_form(body: Dict[str, Any], *, plan: Mapping[str, Any],
                         owner: Any, question: str,
                         client_id: Optional[str], output_root: Optional[str],
                         tenant_id: Optional[str],
                         authorised_portfolio_ids: Tuple[str, ...],
                         run_id: Optional[str],
                         render_portfolio_id: Optional[str],
                         as_of: Optional[str]
                         ) -> Tuple[Optional[Dict[str, Any]], str]:
    """One CHANGE-INTELLIGENCE serving attempt. Same contract as `_attempt`.

    Stage for stage the other two attempts: perimeter, execute, render, record.
    What differs is whose perimeter and whose owner — the adapter declares its
    own, and every figure comes back from
    `period_change.workflow.run_period_change_analysis`, which is where every
    other governed period-change figure in this estate comes from.

    NOTHING IS CALCULATED, COMPOSED OR RENDERED HERE. The adapter owns the
    translation, the deterministic owner the arithmetic, `insight_funded` the
    composition where there is one, and `period_change_route._render` the
    envelope. This function chooses between two adapters and records what
    happened.

    THE OWNER'S INPUTS ARE THE CALLER'S TO SUPPLY, exactly as the funded frame
    and the snapshot catalogue are. With no governed source root or no tenant
    this returns None and the legacy envelope serves.
    """
    eligible, why, detail = owner.check_eligibility(plan)
    body["eligibility"] = {"eligible": eligible, "reason": why, "detail": detail,
                           "perimeter": f"change_form_{owner.CHANGE_FORM}"}
    if not eligible:
        body["execution"] = {"attempted": False, "why_not": f"{why}: {detail}"[:300]}
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{why}"

    if not client_id or not tenant_id:
        body["execution"] = {"attempted": False,
                             "why_not": "no governed source root or tenant was "
                                        "supplied for this request"}
        body["disposition"] = evidence.INELIGIBLE
        return None, CHANGE_OWNER_INPUTS_UNAVAILABLE

    body["execution"] = {"attempted": True, "runtime": owner.CHANGE_FORM,
                         "requested_semantics": material_summary.plan_provenance(
                             plan)}
    try:
        if owner is material_summary:
            outcome = owner.execute(
                plan, client_id=client_id, output_root=output_root,
                tenant_id=tenant_id, run_id=run_id,
                authorised_portfolio_ids=tuple(authorised_portfolio_ids))
        else:
            # `run_id` is not passed because this form composes nothing: it
            # tags a finding with the run it was generated from, and an
            # attribution answer IS the governed decomposition. Accepting it here
            # to make one call serve both forms would mean an input that
            # silently does nothing.
            outcome = owner.execute(
                plan, client_id=client_id, output_root=output_root,
                tenant_id=tenant_id,
                authorised_portfolio_ids=tuple(authorised_portfolio_ids))
    except Exception as exc:                                         # noqa: BLE001
        # EVERY GOVERNED REFUSAL THE OWNER RAISES LANDS HERE, classified by the
        # exception it raised: `PeriodChangeFailure` for a book with fewer than
        # two snapshots or no eligible field, `LensNotApplied` for a scope that
        # could not be applied to every snapshot compared. The legacy envelope
        # then serves, which is where those refusals are already worded. Nothing
        # is widened and no date is invented in either case.
        body["execution"]["error"] = f"{type(exc).__name__}: {exc}"[:300]
        body["disposition"] = evidence.EXECUTION_ERROR
        return None, EXECUTION_FAILED

    if not outcome.get("ok"):
        body["execution"].update({
            "attempted": False,
            "why_not": f"{outcome.get('reason')}: {outcome.get('detail')}"[:300]})
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{outcome.get('reason')}"

    receipt = dict(outcome.get("receipt") or {})
    body["execution"].update({"receipt": receipt,
                              "value": receipt.get("period_to")})
    body["disposition"] = evidence.EXECUTED

    try:
        if owner is material_summary:
            payload = owner.envelope(plan, outcome["result"], outcome["brief"],
                                     question=question,
                                     portfolio_id=render_portfolio_id,
                                     as_of=as_of)
        else:
            payload = owner.envelope(plan, outcome["result"], question=question,
                                     portfolio_id=render_portfolio_id,
                                     as_of=as_of)
    except Exception as exc:                                         # noqa: BLE001
        body["execution"]["render_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return None, RENDER_FAILED
    if not isinstance(payload, Mapping) or not payload.get("ok"):
        body["execution"]["render_error"] = "the rendered envelope was not ok"
        return None, RENDER_FAILED
    return dict(payload), ""


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
             pipeline_run_id: Optional[str] = None,
             funded_frame_resolver: Any = None,
             client_id: Optional[str] = None,
             output_root: Optional[str] = None,
             tenant_id: Optional[str] = None,
             authorised_portfolio_ids: Tuple[str, ...] = (),
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
            body, plan=plan, question=question, semantics=semantics,
            render_portfolio_id=render_portfolio_id, as_of=as_of,
            pipeline_source=pipeline_source, pipeline_root=pipeline_root,
            pipeline_client_id=pipeline_client_id,
            pipeline_history=pipeline_history, pipeline_run_id=pipeline_run_id)

    # STAGE MOVEMENT, a second pipeline owner, above the gate for the same
    # reason: its plans are about the pipeline and the funded runtimes would
    # refuse them for a population they never execute.
    if stage_rt.claims(plan):
        return _attempt_stage_movement(
            body, plan=plan, question=question,
            pipeline_root=pipeline_root, pipeline_client_id=pipeline_client_id,
            pipeline_history=pipeline_history,
            render_portfolio_id=render_portfolio_id, as_of=as_of)

    # FORECAST, the derived population, likewise above the gate: a forecast plan
    # is not the funded runtimes' to refuse. It reads the funded book and the
    # pipeline as INPUTS and executes neither — `forecast_rt` declares
    # `forecast` alone, so this branch cannot carry a funded plan past the gate.
    if forecast_rt.claims(plan):
        return _attempt_forecast(
            body, plan=plan, question=question, client_id=client_id,
            output_root=output_root, pipeline_root=pipeline_root,
            pipeline_history=pipeline_history, run_id=pipeline_run_id,
            funded_frame_resolver=funded_frame_resolver, semantics=semantics,
            render_portfolio_id=render_portfolio_id, as_of=as_of)

    # WHICH POPULATION, BEFORE WHICH RUNTIME. Placed above the dispatch because
    # it is true of both: the temporal runtime reads the same funded route the
    # slice 1 executor reads a funded frame from, and neither may answer a
    # question about a population it did not load. Placed above EXECUTION for
    # the reason that matters — a refusal here means zero rows were touched, so
    # a pipeline question cannot produce a funded number even transiently, in
    # the evidence sink or anywhere else.
    #
    # WHOSE DECLARATION THIS CHECKS, named rather than inherited. The gate speaks
    # for the funded runtimes below it, so it admits exactly what they declare —
    # `runtime_registry.FUNDED_GATE_POPULATIONS`, derived from those runtimes and
    # nothing else. A runtime owning another population is dispatched ABOVE this
    # line from `POPULATION_OWNING_RUNTIMES`; one that is not is refused here.
    base_ok, base_why, base_detail = adapter.check_population_base(
        plan, execution_population,
        executable=runtime_registry.FUNDED_GATE_POPULATIONS)
    if not base_ok:
        body["eligibility"] = {"eligible": False, "reason": base_why,
                               "detail": base_detail,
                               "perimeter": "population_base"}
        body["execution"] = {"attempted": False,
                             "why_not": f"{base_why}: {base_detail}"[:300]}
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{base_why}"

    # CHANGE-INTELLIGENCE DISPATCH, from the plan's own analytical form.
    #
    # Placed AFTER the population gate because both forms are funded-book forms
    # and the gate speaks for exactly that — a plan asking about the pipeline
    # extract must not reach a funded snapshot comparison. Placed BEFORE the
    # temporal dispatch because a `material_summary` or `attribution` plan
    # stating a relative pair IS claimed by `plan_temporal_runtime.claims`, which
    # reads the period form alone; reaching it first, both forms were refused
    # CAPABILITY_NOT_GENERIC by a runtime that speaks for single-measure generic
    # evaluation and never for these owners. That refusal is what this sprint
    # closes, and it is the whole behavioural change here: a plan that served
    # nothing now reaches the owner that already computes it.
    change_owner = _change_form_owner(plan)
    if change_owner is not None:
        return _attempt_change_form(
            body, plan=plan, owner=change_owner, question=question,
            client_id=client_id, output_root=output_root, tenant_id=tenant_id,
            authorised_portfolio_ids=authorised_portfolio_ids,
            run_id=pipeline_run_id, render_portfolio_id=render_portfolio_id,
            as_of=as_of)

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
    # A FIELD THIS BOOK DOES NOT CARRY is a fact about the book, not a failed
    # execution: the executor's own validator says which governed fields have
    # no column here, and the attempt is recorded as that — never as a crash,
    # and never answered from a neighbouring field.
    missing = adapter.fields_not_in_book(spec, semantics,
                                         getattr(frame, "columns", None))
    if missing:
        body["execution"] = {"attempted": False,
                             "why_not": (f"{FIELD_NOT_IN_BOOK}: this book carries "
                                         f"no {', '.join(missing)}")[:300],
                             "fields_not_in_book": missing}
        body["disposition"] = evidence.INELIGIBLE
        return None, f"{INELIGIBLE}:{FIELD_NOT_IN_BOOK}"
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

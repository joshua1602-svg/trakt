#!/usr/bin/env python3
"""mi_agent_api/concept_merge_arm.py — the split, wired, behind one flag.

The model proposes concepts in registered vocabulary; the registry binds them;
the merge decides what may reach the contract. This module is the ONE place
that connects those three to a live request, so `chat_routing` gains a single
call and everything about the arm — the flag, the vocabulary, the model call,
the merge, what it is allowed to change on the spec, and what it publishes —
lives together.

OFF BY DEFAULT AND INDEPENDENT OF THE FREE-FORM PARSER.
`MI_AGENT_CONCEPT_MERGE=on` turns this on; it is not implied by a key being
present. That separation is not cosmetic. The shipped free-form arm
(`MI_AGENT_LLM_PARSER`) emits a whole `MIQuerySpec` and is the arrangement this
split exists to REPLACE; running both would measure their sum. A measurement of
this arm sets `MI_AGENT_LLM_PARSER=off` and `MI_AGENT_CONCEPT_MERGE=on`.

WHAT IT MAY CHANGE, AND WHAT IT MAY NOT.
Only what `claim_merge` reports as `filled_by_model` — a slot that was EMPTY.
A slot the reader filled, a slot the caller filled, and a slot carrying a
governed DEFAULT are all untouched, which is what keeps "Show me the trend."
refusing: that guard fires on `subject.provenance == PROV_DEFAULT`, and a
default is a filled slot.

Every fill is published under `metadata.conceptMerge` with the provenance
`model_inferred`, so nothing downstream can mistake it for something the reader
said. `stated_by_user` is False for that provenance on every claim that exposes
it.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Built once per (registry, book columns). Building it probes every candidate
#: term through the binder, which is cheap once and wasteful per request.
_VOCAB_CACHE: Dict[Tuple[Any, ...], Any] = {}


#: THE AVAILABILITY STATE. A call that did not happen, could not be read, or
#: came back as an error. It is NEVER inferred from an empty proposal list: a
#: model that successfully proposes nothing returns `no_change` with
#: `proposed: []`, and the two are different events with different consequences.
#: Named here so producer and consumer share one string.
PROPOSAL_UNAVAILABLE = "proposal_unavailable"


#: A recorded proposal per question, used INSTEAD of a live call when set.
#:
#: The same injection seam `parse_with_repair` already offers as `llm_callable`,
#: and for the same reason: a measurement has to be reproducible without paying
#: for the model again, and a review pack that re-derives its numbers from a
#: SECOND live call is a review of a different run. Set by a harness, never in
#: serving — `apply` consults it only when it is non-empty, and records in the
#: evidence that the proposal was replayed rather than asked for.
_REPLAY: Dict[str, Any] = {}


def set_replay(proposals_by_question: Optional[Dict[str, Any]]) -> None:
    """Replay recorded proposals instead of calling the model. ``None`` clears."""
    _REPLAY.clear()
    if proposals_by_question:
        _REPLAY.update(proposals_by_question)


def enabled() -> bool:
    """Is the concept-merge arm on for this process?"""
    mode = os.environ.get("MI_AGENT_CONCEPT_MERGE", "off").strip().lower()
    if mode not in ("on", "1", "true", "yes"):
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def model_name() -> str:
    from mi_agent.llm_query_parser import DEFAULT_MODEL

    return os.environ.get("MI_AGENT_CONCEPT_MERGE_MODEL") \
        or os.environ.get("MI_AGENT_LLM_MODEL") or DEFAULT_MODEL


def _vocabulary(semantics, available_values, available_columns):
    from question_interpretation import concept_proposal as CP

    # KEYED ON WHAT THE VOCABULARY IS BUILT FROM, never on `id()`. CPython
    # reuses an id once an object is collected, so an id key can serve the
    # vocabulary of one book for a request against another — and `mi_service`
    # loads the semantics fresh per request, so an id key would also never hit.
    key = (frozenset(available_columns or ()),
           len((semantics or {}).get("fields") or {}),
           tuple(sorted(str(f) for f in (available_values or {}))))
    if key not in _VOCAB_CACHE:
        _VOCAB_CACHE[key] = CP.vocabulary(
            semantics, available_values=available_values,
            available_columns=available_columns)
    return _VOCAB_CACHE[key]


def _apply_to_spec(spec, filled) -> List[Dict[str, Any]]:
    """Put the model's fills on the spec. Additive only, by construction.

    A threshold arrives with an operator and becomes `{"op": ..., "value": ...}`
    — the shape `population.predicate_of` normalises and both executors read. A
    categorical arrives without one and becomes a bare value, which is the same
    shape the deterministic parser produces for a categorical filter.
    """
    from question_interpretation import claim_merge as CM

    applied: List[Dict[str, Any]] = []
    filters = dict(getattr(spec, "filters", None) or {})
    dimensions = list(getattr(spec, "dimensions", None) or [])

    for slot in filled:
        if slot.slot == CM.SLOT_ROW_PREDICATES and slot.key:
            if slot.key in filters:
                continue        # never overwrite; the merge already said so
            filters[slot.key] = ({"op": slot.operator, "value": slot.value}
                                 if slot.operator else slot.value)
            applied.append({"kind": "filter", "field": slot.key,
                            "operator": slot.operator, "value": slot.value})
        elif slot.slot == CM.SLOT_DIMENSIONS and slot.key:
            if slot.key in dimensions or getattr(spec, "dimension", None) == slot.key:
                continue
            dimensions.append(slot.key)
            applied.append({"kind": "dimension", "field": slot.key})
        elif slot.slot == CM.SLOT_SUBJECT and not getattr(spec, "metric", None):
            spec.metric = slot.value
            applied.append({"kind": "measure", "field": slot.value})
        elif slot.slot == CM.SLOT_SOURCE_SCOPE and slot.value:
            # A SCOPE THE MERGE FILLED AND NOTHING CARRIED. This branch did not
            # exist, so `source_book` was the one proposal kind that could be
            # proposed, bound by the registry, accepted by the merge, recorded
            # as `filled_by_model` — and then dropped on the floor. The receipt
            # then correctly reported `source_portfolio_type` as a lost
            # narrowing and refused, which is why four questions naming the
            # Direct book were declined while their siblings answered.
            #
            # Applied through the governed lens owner, exactly as the
            # deterministic path applies one: the model proposed the TERM, the
            # registry resolved it to a lens, and the lens decides what it
            # narrows to. Nothing here chooses a field or a value.
            from mi_agent import portfolio_lens as _PL

            lens = _PL.lens_from_term(str(slot.value))
            if not getattr(lens, "filters", None):
                continue        # an unresolved book narrows nothing; say nothing
            if any(k in filters for k in lens.filters):
                continue        # never overwrite; the merge already said so
            _PL.apply_lens(spec, lens)
            filters = dict(getattr(spec, "filters", None) or {})
            applied.append({"kind": "scope", "field": "portfolio_lens",
                            "value": getattr(lens, "name", None),
                            "narrows": sorted(lens.filters)})

    if applied:
        spec.filters = filters
        if dimensions:
            spec.dimensions = dimensions
            # A single governed axis also travels on `spec.dimension`, which is
            # what the point-in-time executor groups by. Setting the list alone
            # produced a contract that named an axis nothing grouped on.
            if len(dimensions) == 1 and not getattr(spec, "dimension", None):
                spec.dimension = dimensions[0]
    return applied


def apply(question: str, spec: Any, semantics: Dict[str, Any], *,
          interpretation: Any, available_values: Any = None,
          available_columns: Any = None) -> Optional[Dict[str, Any]]:
    """Propose, bind, merge, and apply. Returns the evidence, or ``None``.

    NEVER RAISES INTO A REQUEST. A model that is unreachable, a reply that
    cannot be read, a binder that refuses everything — each leaves the
    deterministic contract exactly as it was, which is the arm being off for
    that question rather than the question failing.
    """
    if interpretation is None:
        return None
    from mi_agent import llm_query_parser as LQ
    from question_interpretation import claim_merge as CM
    from question_interpretation import concept_proposal as CP

    replayed = question in _REPLAY
    try:
        vocab = _vocabulary(semantics, available_values, available_columns)
        if replayed:
            usage = {}
            proposals = [CP.ProposedConcept(
                p["kind"], p["term"], p.get("covers") or "",
                p.get("comparator"), p.get("value")) for p in _REPLAY[question]]
        else:
            prompt = CP.build_proposal_prompt(question, vocab)
            text, usage, _cached = LQ._call_llm(prompt, model_name())
            proposals = CP.parse_proposal_response(text)
    except Exception as exc:  # noqa: BLE001 - the arm degrades, the request lives
        logger.info("concept proposal unavailable for %r: %s: %s",
                    question, type(exc).__name__, exc)
        return {"status": PROPOSAL_UNAVAILABLE,
                "detail": "%s: %s" % (type(exc).__name__, str(exc)[:200])}

    bound, rejected = CP.bind(proposals, vocab)
    # The governed operation the contract ALREADY selected, so the merge can
    # check a proposed role rather than trust it. Built from `spec` before any
    # fill touches it — a profile read after the merge would be circular.
    result = CM.merge(CM.deterministic_slots(interpretation), bound, rejected,
                      profile=CM.operation_profile(spec))
    applied = _apply_to_spec(spec, result.filled_by_model)

    return {
        "status": "applied" if applied else "no_change",
        "source": "replayed" if replayed else "model",
        "model": None if replayed else model_name(),
        "provenance": CM.PROV_MODEL_INFERRED,
        "proposed": [p.as_dict() for p in proposals],
        "bound": [b.as_dict() for b in bound],
        "rejected": [r.as_dict() for r in rejected],
        "applied": applied,
        "findings": [f.as_dict() for f in result.findings],
        "conflicts": len(result.conflicts),
        "usage": dict(usage or {}),
        # WHAT THE ANSWER COST, on the answer. The arm published raw token
        # counts and nothing priced them, so cost per question could only be
        # reconstructed later by whoever still had the envelopes — and the
        # stability, surface and CR4 harnesses had already discarded them.
        # Priced by `llm_query_parser.estimate_cost`, the estate's existing
        # owner of the pricing table, so there is no second opinion about what
        # a token costs. A replayed proposal is not a call and is not priced.
        "cost": _priced(model_name(), usage) if not replayed else None,
    }


def _priced(model: str, usage: Any) -> Optional[Dict[str, Any]]:
    """The cost of a call, or ``None``. NEVER RAISES INTO A REQUEST.

    TELEMETRY MAY NOT DECIDE WHETHER AN ANSWER IS GIVEN. `apply` is called
    inside a `try` whose `except` records `proposal_unavailable`, and an
    unavailable proposal is a controlled REFUSAL. So an exception here — a
    pricing table that cannot read a token count, a usage shape a future SDK
    changes — would travel that path and turn a delivered analytical answer
    into "I could not complete the language-understanding step". Measured
    before this guard existed: a malformed usage record flipped a correct
    39-group heatmap into a refusal.

    That is the wrong trade in both directions. Cost is an observation ABOUT an
    answer; it cannot be allowed to become a fact about whether the reader gets
    one. A pricing failure costs the telemetry and nothing else.

    `estimate_cost` already reports an unpriced model as status "unknown"
    rather than a silent zero, and that path is untouched: it is a value, not a
    failure.
    """
    from mi_agent import llm_query_parser as _LQ

    try:
        return _LQ.estimate_cost(model, usage)
    except Exception as exc:  # noqa: BLE001 - an answer never fails for a price
        logger.info("cost telemetry unavailable for model %r: %s: %s",
                    model, type(exc).__name__, exc)
        return None

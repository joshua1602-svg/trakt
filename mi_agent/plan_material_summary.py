#!/usr/bin/env python3
"""Plan connectivity for `change_form = material_summary`. Offline only.

WHAT THIS IS. The thin adapter between a compiled `GovernedQueryPlan` that was
read as "tell me what materially changed" and the two governed owners that
answer it. It translates a plan into one call and hands the result to a
composition. It is about sixty lines of translation with nothing else in it.

WHAT IT IS NOT. It is not a calculation owner, not a composition owner and not a
materiality rule. Every figure comes from
`period_change.workflow.run_period_change_analysis`; every finding is composed
by `mi_agent_api.insight_funded`, which is the EXISTING insight engine's
generator layer; every threshold is read by that layer through
`mi_agent_api.insight_config`. Nothing here decides whether a movement matters.

THE SAME THREE OWNERSHIP RULES SLICE 2 IS SUBORDINATE TO.

    1. The PLAN owns requested semantics. Nothing here reads a question: it is
       not a parameter of any function in this module, and the module imports no
       parser, recogniser, router or `re`, so it cannot pattern-match a sentence
       even by accident. It does not read the model's raw claim either — the
       dispatch reads `provenance.compiler_bindings["change_form"]`, which is
       the COMPILER's derived reading, so no serving decision is ever taken from
       what the model said.
    2. `period_change.periods.resolve_periods` owns which two snapshots a
       comparison is between. This module chooses no date, no snapshot and no
       file; it passes the plan's period binding through and the resolver
       resolves.
    3. The RECEIPT owns what actually executed. `receipt()` records the mode,
       the resolved pair, the governed owner and the configuration the findings
       were gated by, so an auditor can see what ran without re-running it.

FAIL CLOSED, NEVER NARROW. A plan outside this perimeter is refused with a
reason that names the perimeter it failed. It is never answered by the nearest
available analysis — a balance delta, a portfolio overview or a bridge in place
of "what changed?" is the exact defect the `change_form` slot exists to end, and
this module is the thing that would commit it if it guessed.

NOTHING HERE SERVES A USER. There is no flag, no envelope and no call site in
`mi_service` or `plan_serving_canary`. Like slice 2 before it, this is an
offline capability with a fixture bank; wiring it to a request is a later
decision made with evidence in hand.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from mi_agent import plan_runtime_adapter as adapter

#: The form this runtime owns. One value, because one analytical form maps to
#: one composition; a second form here would mean this module had started
#: deciding which question it was answering.
CHANGE_FORM = "material_summary"

#: The governed mode the shared owner runs in for this form. `metric_delta` runs
#: the same owner in `requested_metric`; the mode is the whole distinction, and
#: it is the workflow's own concept, not one invented here.
WORKFLOW_MODE = "portfolio_overview"

#: The funded book, and only the funded book. A material-change summary is over
#: governed portfolio snapshots; the pipeline extract is a different population
#: with a different owner and its own weekly brief.
POPULATION_BASES = frozenset({"funded"})

#: THE ONE OPERATION THIS FORM EXECUTES AS. A reader asking "what changed" says
#: it as `movement`, `compare` or `summary` — one analytical form, three
#: spellings of its action — and the canonicalisation seam
#: (`normalise.py` rule 5, over `vocabulary.CHANGE_FORM_OPERATION_VARIANTS`)
#: collapses them to `summary` BEFORE the plan exists. So this perimeter sees
#: exactly one shape and admits exactly one.
#:
#: A non-canonical operation reaching here means the plan was built without that
#: seam. It is refused rather than re-canonicalised: a second place that rewrites
#: an operation is a second place the two could disagree, and fail-closed is the
#: right answer to a plan that skipped a governed step.
OPERATIONS = frozenset({"summary"})

# Ineligibility reasons. Stable strings: a bank groups on them and a report
# counts them.
FORM_NOT_CLAIMED = "FORM_NOT_CLAIMED"
OPERATION_NOT_ADMITTED = "OPERATION_NOT_ADMITTED"
POPULATION_NOT_FUNDED = "POPULATION_NOT_FUNDED"
MEASURE_NARROWS_THE_SUMMARY = "MEASURE_NARROWS_THE_SUMMARY"
PERIOD_NOT_A_PAIR = "PERIOD_NOT_A_PAIR"
PERIOD_GRAIN_UNTRANSLATABLE = "PERIOD_GRAIN_UNTRANSLATABLE"
# Slots the deterministic owner cannot honour. Refused, never dropped: a
# governed narrowing the reader stated and the analysis did not apply is the
# whole-book answer published under the narrow label, and it is
# indistinguishable from a correct one once rendered.
DIMENSION_NARROWS_THE_SUMMARY = "DIMENSION_NARROWS_THE_SUMMARY"
FILTER_NOT_SUPPORTED = "FILTER_NOT_SUPPORTED"
GEOGRAPHY_NOT_SUPPORTED = "GEOGRAPHY_NOT_SUPPORTED"
TARGET_NOT_SUPPORTED = "TARGET_NOT_SUPPORTED"
COMPARISON_NOT_SUPPORTED = "COMPARISON_NOT_SUPPORTED"
SCOPE_NOT_EXPRESSIBLE = "SCOPE_NOT_EXPRESSIBLE"

#: Period forms that resolve to TWO snapshots on their own. A material summary is
#: a statement about a movement, so it needs a pair. `series` names many periods
#: and is not a pair, and is not silently turned into one.
PAIR_PERIOD_FORMS = frozenset({
    "explicit_period", "range", "previous_reporting_period", "relative_pair"})

#: Period forms that name ONE ANCHOR, which THIS FORM completes into a pair.
#:
#: THE TEMPORAL AUTHORITY RULE. The reader supplies the anchor or an explicit
#: pair; the analytical owner supplies any comparison state the FORM itself
#: intrinsically requires. `material_summary` means "which governed changes are
#: materially relevant", so a comparison state is part of what the form IS — the
#: reader naming only "this period" has asked a complete question, and making the
#: interpreter also author the other end would be the compiler demanding an
#: implementation detail the model was deliberately never shown.
#:
#: The live v1 gate measured exactly that: a reading anchored at `current` was
#: refused here as PERIOD_NOT_A_PAIR even though its form was right, its scope
#: survived and the owner already has a method for the adjacent pair.
#:
#: `current` is therefore admitted as an ANCHOR and NOT added to
#: `PAIR_PERIOD_FORMS` — it still does not denote two snapshots, and the
#: distinction is what keeps the provenance honest: the INTERPRETED form stays
#: `current`, and the RESOLUTION METHOD is the owner's. Nothing rewrites the
#: intent to pretend the model emitted a pair.
ANCHOR_PERIOD_FORMS = frozenset({"current"})

#: The forms this composition can start from at all: a pair, or an anchor it
#: completes. Kept as the union so the two reasons stay distinguishable above.
ADMITTED_PERIOD_FORMS = PAIR_PERIOD_FORMS | ANCHOR_PERIOD_FORMS

#: A `relative_pair` states a GRAIN and a DISTANCE; the resolver states the same
#: idea as a relative METHOD. Both vocabularies are governed, and this is the
#: whole translation between them — written out rather than derived, so a
#: combination with no method is refused instead of being rounded to the nearest
#: one.
#:
#: THE ABSENT GRAIN IS NOT A GAP. The compiler's own normal form produces
#: `relative_pair` + `periods_back=1` with no grain for the commonest request in
#: this family, and states what it means: "a pair with no distance IS the
#: adjacent pair". The resolver has a method for exactly that, and it is not a
#: calendar month — `current_vs_previous` compares the two adjacent GOVERNED
#: SNAPSHOTS, whatever cadence the book reports on. Mapping an absent grain to
#: `month_on_month` would assume a monthly book; mapping it to nothing would
#: refuse the question this contract was built for.
#:
#: `daily` and `weekly` are absent because the resolver publishes no such method.
RELATIVE_MODE_BY_GRAIN: Mapping[Optional[str], str] = {
    None: "current_vs_previous",
    "monthly": "month_on_month",
    "quarterly": "quarter_on_quarter",
    "annual": "year_on_year",
}

#: Distances a relative method can express. Each method above IS "one period
#: back" at its own grain; the resolver publishes no method for "three periods
#: back", and this module may not synthesise one — computing the dates itself
#: would be exactly the period arithmetic the resolver owns. So a longer
#: distance is refused and says why, rather than being approximated by the
#: nearest method that happens to span a similar number of months.
RELATIVE_DISTANCES = frozenset({None, 1})


def period_defaulted_to(plan: Any, owner: str) -> bool:
    """Whether the DETERMINISTIC layer supplied a window ``owner`` owns.

    One structural read of the plan's period provenance. True means the reading
    stated no temporal form and `change_form` authorised the default; it never
    means the reader asked for the current state.

    ``owner`` IS A PARAMETER BECAUSE TWO FORMS SHARE THE CONTRACT, not because
    this module decides which. `vocabulary.CHANGE_FORM_ABSENT_PERIOD_DEFAULT`
    authorises `current_vs_previous` for `material_summary` AND for
    `attribution`, and the compiler stamps which form it applied; a second copy
    of this read in the attribution adapter would be a second place the two
    could come to disagree about what the compiler wrote.
    """
    period = adapter._as_mapping(plan).get("period") or {}
    return (bool(period.get("defaulted"))
            and period.get("default_method") == "current_vs_previous"
            and period.get("default_owner") == owner)


def _period_defaulted_to_owner(plan: Any) -> bool:
    """This form's own face on :func:`period_defaulted_to`."""
    return period_defaulted_to(plan, CHANGE_FORM)


def _change_form_binding(plan: Any) -> Mapping[str, Any]:
    """The COMPILER's reading of the analytical form, from the plan itself."""
    provenance = adapter._as_mapping(plan).get("provenance") or {}
    bindings = provenance.get("compiler_bindings") or {}
    return bindings.get("change_form") or {}


def claims(plan: Any) -> bool:
    """Whether this runtime owns the plan. One structural read, no judgement.

    True does NOT mean eligible. A material-summary plan outside this contract is
    claimed here and then refused by `check_eligibility`, which is the point: it
    is refused with a reason about THIS form rather than falling through to a
    runtime that would refuse it for not being something else.
    """
    return _change_form_binding(plan).get("form") == CHANGE_FORM


def check_eligibility(plan: Any) -> Tuple[bool, str, str]:
    """``(eligible, reason, detail)``. Every refusal names its perimeter."""
    body = adapter._as_mapping(plan)
    if not claims(plan):
        return False, FORM_NOT_CLAIMED, (
            f"the compiler read this plan's analytical form as "
            f"{_change_form_binding(plan).get('form')!r}, not {CHANGE_FORM!r}")

    operation = body.get("operation")
    if operation not in OPERATIONS:
        return False, OPERATION_NOT_ADMITTED, (
            f"{operation!r} states a result shape this composition does not "
            f"produce; it is refused rather than flattened into a summary")

    base = (body.get("population") or {}).get("base")
    if base not in POPULATION_BASES:
        return False, POPULATION_NOT_FUNDED, (
            f"population base {base!r}: a material-change summary is over "
            f"governed funded snapshots")

    # A plan that names a measure is a `metric_delta`, whatever form the model
    # claimed. Running a portfolio overview for it would answer a broader
    # question than the reader asked and report it as theirs.
    named = [m.get("concept")
             for output in (body.get("outputs") or ())
             for m in (output.get("measures") or ())]
    if named:
        return False, MEASURE_NARROWS_THE_SUMMARY, (
            f"the plan names measure(s) {sorted(str(n) for n in named)}; a named "
            f"measure is a metric delta, which has its own owner and its own mode")

    # A DIMENSION IS A NARROWER QUESTION, like a measure. `portfolio_overview`
    # selects every governed dimension the registry admits and reports the
    # composition shifts it finds; running it for a plan that asked about ONE
    # dimension would answer more than the reader asked and publish it as theirs.
    grouped = [d.get("concept")
               for output in (body.get("outputs") or ())
               for d in (output.get("dimensions") or ())]
    if grouped:
        return False, DIMENSION_NARROWS_THE_SUMMARY, (
            f"the plan groups by {sorted(str(g) for g in grouped)}; this "
            f"composition selects its own dimensions and a named one is a "
            f"different question with a different owner")

    ok, why, detail = check_slots_honoured(plan)
    if not ok:
        return ok, why, detail

    return check_period(plan, owner=CHANGE_FORM)


def check_slots_honoured(plan: Any) -> Tuple[bool, str, str]:
    """Every plan slot `analyse_period_change` CANNOT honour, refused by name.

    WHY THIS EXISTS AT ALL. The entry point this family executes through takes a
    client, an output root, a mode, a period request and a LENS. It takes no row
    predicates: `period_change_route.build_snapshots` has a `population=`
    parameter and `analyse_period_change` does not pass one, so a governed filter
    on the plan would reach neither snapshot. Left unchecked, a plan saying "what
    changed in the back book" would have been answered over the whole book and
    labelled with the reader's words — the defect class the population ledger was
    built to close, arriving through a different door.

    So the perimeter is drawn at what the owner can honour, and everything else
    REFUSES with the slot named. Widening is never the fallback, and neither is
    silently dropping the slot: both publish a different question's answer.

    Shared by `material_summary` and by `plan_attribution`, which execute through
    the same entry point and therefore have exactly the same limits. The SCOPE
    half is delegated to `mi_agent_api.contract_scope.lens_from_plan`, which is
    already the estate's only mapping from a transcribed request to a
    `portfolio_lens`; resolving a role here would make this a second owner of
    what "acquired" means.
    """
    from mi_agent_api import contract_scope

    body = adapter._as_mapping(plan)
    outputs = tuple(body.get("outputs") or ())

    # THE TWO FILTER SLOTS, AND NOT THE THIRD. `plan_runtime_adapter.
    # plan_predicates` reads three — plan `filters`, output `filters` and
    # `population.scope_predicates` — and the third is where an explicit
    # Direct/Acquired role and a named book land. Those two ARE honoured, as the
    # owner's lens; the other two are row predicates and are not.
    restrictions = (tuple(body.get("filters") or ())
                    + tuple(f for output in outputs
                            for f in (output.get("filters") or ())))
    if restrictions:
        fields = sorted({str(f.get("canonical_field") or f.get("concept") or "")
                         for f in restrictions})
        return False, FILTER_NOT_SUPPORTED, (
            f"the plan restricts on {fields}; this owner applies no row "
            f"predicates, so the restriction would reach neither snapshot and "
            f"the movement reported would be the whole book's")

    if body.get("geography") or any(output.get("geography") for output in outputs):
        return False, GEOGRAPHY_NOT_SUPPORTED, (
            "the plan carries a governed geography contract; this owner selects "
            "its own dimensions and binds no geography axis")

    if body.get("target"):
        return False, TARGET_NOT_SUPPORTED, (
            f"the plan states target {body.get('target')!r}; this owner "
            f"publishes no threshold assessment")

    if str(body.get("comparison_kind") or "none") != "none":
        return False, COMPARISON_NOT_SUPPORTED, (
            f"the plan compares populations ({body.get('comparison_kind')!r}); "
            f"this owner compares two reporting states of ONE population")

    ok, _lens, detail = contract_scope.lens_from_plan(plan)
    if not ok:
        return False, SCOPE_NOT_EXPRESSIBLE, detail
    return True, "", ""


def check_period(plan: Any, *, owner: str) -> Tuple[bool, str, str]:
    """The TEMPORAL half of the perimeter, for any form that shares it.

    Called by this form and by `plan_attribution`, which admits the same three
    temporal states for the same reason: a pair the reader named, an anchor the
    form completes, or an absent window the compiler defaulted on the form's
    behalf. The reason codes are this module's and stay this module's, so the
    two forms refuse an untranslatable window under one name.
    """
    period = adapter._as_mapping(plan).get("period") or {}
    form = period.get("form")
    # AN UNSTATED WINDOW THIS FORM OWNS. The compiler marks the binding
    # `defaulted` when the reading stated no temporal form and `change_form` owns
    # an authorised default; `form` then still carries its construction value, so
    # admitting it on `form` alone would be reading the masquerade. Admit it on
    # the PROVENANCE instead, which is the thing that actually says the owner
    # supplied the window.
    if period_defaulted_to(plan, owner):
        return True, "", ""
    if form not in ADMITTED_PERIOD_FORMS:
        return False, PERIOD_NOT_A_PAIR, (
            f"period form {form!r} neither resolves to two governed snapshots "
            f"nor anchors a pair this form completes, and a movement cannot be "
            f"stated without a pair")

    if form == "relative_pair":
        grain = period.get("grain")
        if grain not in RELATIVE_MODE_BY_GRAIN:
            return False, PERIOD_GRAIN_UNTRANSLATABLE, (
                f"grain {grain!r} has no governed relative method on the period "
                f"resolver, and rounding it to the nearest one would answer a "
                f"different question than the reader asked")
        distance = period.get("periods_back")
        if distance not in RELATIVE_DISTANCES:
            return False, PERIOD_GRAIN_UNTRANSLATABLE, (
                f"a distance of {distance!r} periods has no governed relative "
                f"method; computing the two dates here would be the period "
                f"arithmetic the resolver owns")

    return True, "", ""


def period_request(plan: Any) -> Any:
    """This form's own face on :func:`period_request_for`."""
    return period_request_for(plan, owner=CHANGE_FORM)


def period_request_for(plan: Any, *, owner: str) -> Any:
    """The plan's period binding, as the resolver's own request object.

    TRANSLATION ONLY, between two governed vocabularies. Every date this puts on
    a request was read out of the plan; none is computed from a calendar, no
    month is stepped and no window is widened. The RESOLVER decides which
    snapshots those dates land on, whether either end had to be adjusted, and
    whether the gap between them is within policy.

    A single explicit period is passed as the END of the pair with no start, so
    the resolver chooses the snapshot to compare it against. That is its
    decision to make and it records it: choosing the earlier snapshot here would
    be this module resolving a period it was asked only to pass on.
    """
    from mi_agent.period_change.periods import PeriodRequest

    period = adapter._as_mapping(plan).get("period") or {}
    form = period.get("form")
    labels = [str(x) for x in (period.get("labels") or ())]

    # ONE ANCHOR, AND THE OWNER SUPPLIES THE OTHER END. `previous_reporting_period`
    # names the earlier state and `current` the later one; either way the reader
    # has named one end of a comparison this form intrinsically requires, and
    # `current_vs_previous` is the resolver's own method for the two adjacent
    # GOVERNED snapshots — not a calendar month, and not a date computed here.
    # Which state the comparison runs against is never selected by the model.
    # The authorised default comes first: an unstated window this form owns
    # resolves through the same governed method an anchor does, and for the same
    # reason — the form, not the reader, owns the comparison state.
    if period_defaulted_to(plan, owner):
        return PeriodRequest(relative_mode="current_vs_previous")
    if form in ("previous_reporting_period",) or form in ANCHOR_PERIOD_FORMS:
        return PeriodRequest(relative_mode="current_vs_previous")
    if form == "relative_pair":
        grain = period.get("grain")
        return PeriodRequest(relative_mode=RELATIVE_MODE_BY_GRAIN[
            str(grain) if grain is not None else None])
    if len(labels) >= 2:
        return PeriodRequest(requested_start=labels[0], requested_end=labels[-1])
    if labels:
        return PeriodRequest(requested_end=labels[0])
    # No label and no relative mode: the resolver's documented fallback is the
    # latest available pair, "the only defensible reading of a bare 'what
    # changed?'". Stated by it, not decided here.
    return PeriodRequest()


#: THE DETERMINISTIC OWNER EVERY FIGURE IN THIS FAMILY COMES FROM. Recorded as a
#: string in the receipt so a reader of the evidence can name the owner without
#: re-deriving it from a workflow id, and stated once so two forms cannot record
#: two different answers to "who calculated this".
CALCULATION_OWNER = "mi_agent.period_change.workflow.run_period_change_analysis"


def change_receipt(plan: Any, result: Any, *, change_form: str,
                   calculation_owner: str = CALCULATION_OWNER) -> Dict[str, Any]:
    """The evidence EVERY connected change form must record. Shared, so one
    form cannot come to receipt less than another.

    Read from the plan (what was REQUESTED) and from the governed result (what
    actually RAN); nothing is re-derived and no figure is recomputed. The period
    pair is the resolver's own, the scope is the workflow's own reference, and
    the owner is named rather than implied.
    """
    body = adapter._as_mapping(plan)
    population = body.get("population") or {}
    period = body.get("period") or {}
    resolution = result.period_resolution.to_dict()
    # THE EXECUTED SCOPE, FROM THE RESOLUTION THAT WAS ALREADY READ. The
    # workflow hands one `PortfolioScopeRef` to `resolve_periods` and records
    # the same object on its result, so the resolution already carries the
    # registry's explicit portfolio-id list. Reading it from there rather than
    # from a second attribute keeps one fact with one reader.
    scope = dict(resolution.get("portfolio_scope") or {})
    return {
        "plan_id": body.get("plan_id"),
        "change_form": change_form,
        "capability": body.get("capability"),
        "operation": body.get("operation"),
        "mode": (dict(getattr(result, "request_interpretation", None)
                      or {}).get("mode")),
        "population_base": population.get("base"),
        # THE SCOPE, REQUESTED AND EXECUTED, SIDE BY SIDE. The requested side is
        # the plan's own transcription; the executed side is the portfolio ids
        # `build_snapshots` actually narrowed every compared snapshot to. An
        # audit that held only one of them could not answer whether the book the
        # reader named is the book the numbers came from.
        "direct_acquired_scope": population.get("lens"),
        "source_scope": population.get("source_reference"),
        "source_portfolio_id": population.get("source_portfolio_id"),
        "executed_scope": scope,
        # WHO DECIDED THE PERIOD, kept as two separate facts. The INTERPRETED form
        # is what the reader's request carried — `current` stays `current`, and
        # nothing rewrote the intent to claim a pair was asked for. The
        # RESOLUTION is the owner's: its method, and the two governed snapshots
        # it actually read. An audit that cannot tell these apart cannot answer
        # whether the model selected the comparison state. It did not.
        # THREE STATES, KEPT APART. A reading that STATED a form; a reading that
        # stated an ANCHOR the form completes; and a reading that stated NOTHING,
        # where the deterministic layer applied the form's authorised default.
        # The first two are the reader's words; the third is this system's policy,
        # and an audit that could not tell them apart could not answer who chose
        # the comparison state.
        "interpreted_time_present": bool(period.get("stated")),
        "interpreted_period_form": (period.get("form")
                                    if period.get("stated") else None),
        "temporal_default_applied": period_defaulted_to(plan, change_form),
        "temporal_default_method": period.get("default_method") or None,
        "temporal_default_owner": period.get("default_owner") or None,
        "period_completed_by_form": (
            period.get("stated", False)
            and period.get("form") in ANCHOR_PERIOD_FORMS),
        # THE TWO SNAPSHOTS, AS THE RESOLVER NAMED THEM. Lifted to the top of the
        # receipt because FROM/TO is the first question asked of any movement,
        # and read out of the resolution rather than recomputed from dates.
        "period_from": (resolution.get("resolved_start_snapshot") or {}
                        ).get("reporting_date"),
        "period_to": (resolution.get("resolved_end_snapshot") or {}
                      ).get("reporting_date"),
        "snapshot_references": [resolution.get("resolved_start_snapshot"),
                                resolution.get("resolved_end_snapshot")],
        "period_resolution": resolution,
        "calculation_owner": calculation_owner,
        "workflow_owner": getattr(result, "workflow_id", None),
    }


def receipt(plan: Any, result: Any, brief: Mapping[str, Any]) -> Dict[str, Any]:
    """What actually ran, from the governed outputs rather than from intent."""
    body = adapter._as_mapping(plan)
    return {
        **change_receipt(plan, result, change_form=CHANGE_FORM),
        "change_form": CHANGE_FORM,
        "operation": body.get("operation"),
        "population_base": (body.get("population") or {}).get("base"),
        "workflow_owner": getattr(result, "workflow_id", None),
        "workflow_mode": (dict(getattr(result, "request_interpretation", None)
                               or {}).get("mode")),
        # WHO DECIDED THE PERIOD, kept as two separate facts. The INTERPRETED form
        # is what the reader's request carried — `current` stays `current`, and
        # nothing rewrote the intent to claim a pair was asked for. The
        # RESOLUTION is the owner's: its method, and the two governed snapshots
        # it actually read. An audit that cannot tell these apart cannot answer
        # whether the model selected the comparison state. It did not.
        # THREE STATES, KEPT APART. A reading that STATED a form; a reading that
        # stated an ANCHOR the form completes; and a reading that stated NOTHING,
        # where the deterministic layer applied the form's authorised default.
        # The first two are the reader's words; the third is this system's policy,
        # and an audit that could not tell them apart could not answer who chose
        # the comparison state.
        "interpreted_time_present": bool(
            (body.get("period") or {}).get("stated")),
        "interpreted_period_form": (
            (body.get("period") or {}).get("form")
            if (body.get("period") or {}).get("stated") else None),
        "temporal_default_applied": _period_defaulted_to_owner(plan),
        "temporal_default_method": (
            (body.get("period") or {}).get("default_method") or None),
        "temporal_default_owner": (
            (body.get("period") or {}).get("default_owner") or None),
        "period_completed_by_form": (
            (body.get("period") or {}).get("stated", False)
            and (body.get("period") or {}).get("form") in ANCHOR_PERIOD_FORMS),
        "period_resolution": result.period_resolution.to_dict(),
        "composition_owner": "mi_agent_api.insight_funded.compose",
        "materiality_config": brief.get("config_source"),
        "finding_count": brief.get("insight_count"),
        "omission_count": len(brief.get("omitted") or ()),
        "status": brief.get("status"),
    }


def analyse(plan: Any, *, client_id: str, output_root: Optional[str],
            tenant_id: str, mode: Optional[str],
            authorised_portfolio_ids: Tuple[str, ...] = ()) -> Any:
    """ONE call to the deterministic owner, for any form in this family.

    Shared by `material_summary` and `plan_attribution` so there is exactly one
    place a change-form plan becomes a period-change request. The SCOPE comes
    from the plan and from nowhere else — no caller may supply a lens, because a
    caller-supplied scope is a second transcription of the reader's request and
    the two could disagree about which book the numbers are from.

    Raises `PeriodChangeFailure` and `chat_routing.LensNotApplied` unchanged:
    both are the owner's own governed refusals, and swallowing either here would
    replace a classified failure with an unclassified one.
    """
    from mi_agent_api import contract_scope
    from mi_agent_api.period_change_route import analyse_period_change

    ok, lens, detail = contract_scope.lens_from_plan(plan)
    if not ok:                                                  # pragma: no cover
        # Unreachable through `check_slots_honoured`, which refuses first. Kept
        # because "the perimeter already checked" is how an unchecked path gets
        # added later, and widening here would be silent.
        raise ValueError(f"scope not expressible: {detail}")
    return analyse_period_change(
        client_id=client_id, output_root=output_root,
        mode=mode, period_request=period_request_for(
            plan, owner=_change_form_binding(plan).get("form") or CHANGE_FORM),
        scope=lens, tenant_id=tenant_id,
        authorised_portfolio_ids=tuple(authorised_portfolio_ids),
        include_bridge=True)


def plan_provenance(plan: Any) -> Dict[str, Any]:
    """What the PLAN asked for, in the `spec` slot every artefact already carries.

    PROVENANCE, NOT A SPEC. The legacy route passes its parsed `spec` dict here
    and the artefact builders only stamp it into `source.spec` for display; there
    is no parsed spec on this path and none is synthesised. These five fields are
    the plan's own, so an artefact says which governed request produced it.
    """
    body = adapter._as_mapping(plan)
    binding = _change_form_binding(plan)
    return {"change_form": binding.get("form"),
            "capability": body.get("capability"),
            "operation": body.get("operation"),
            "population": (body.get("population") or {}).get("base"),
            "lens": (body.get("population") or {}).get("lens"),
            "plan_id": body.get("plan_id")}


def governed_envelope(plan: Any, result: Any, receipt_block: Mapping[str, Any], *,
                      question: str, portfolio_id: Optional[str],
                      as_of: Optional[str]) -> Dict[str, Any]:
    """The governed period-change envelope, FROM THE ROUTE THAT ALREADY OWNS IT.

    NO SECOND PRESENTER. `period_change_route._render` builds the KPI block, the
    metric-movement table, the composition-shift table and the balance-bridge
    table from a `PeriodChangeResult`, and `build_answer` states the summary from
    `result.summary` alone. Rebuilding any of that here would put a second
    renderer downstream of one calculation, and the two would drift. So the
    existing renderer is called, and this adds exactly one thing: the governed
    plan's own receipt, in the `metadata.governedPlan` slot the slice 1 and
    pipeline paths already publish it in.

    `question` IS NOT READ, it is ECHOED. The renderer puts it in the envelope's
    `question` field, which is what every channel displays back to the reader;
    no branch anywhere below depends on its content.
    """
    from mi_agent_api.period_change_route import _render

    payload = _render(result, question, plan_provenance(plan), portfolio_id,
                      as_of)
    meta = payload.setdefault("metadata", {})
    if isinstance(meta, dict):
        meta["parserMode"] = "governed_plan"
        meta["route"] = f"governed_plan_{receipt_block.get('change_form')}"
        meta["governedPlan"] = {"requested": plan_provenance(plan),
                                "executed": dict(receipt_block)}
    return payload


def envelope(plan: Any, result: Any, brief: Mapping[str, Any], *, question: str,
             portfolio_id: Optional[str], as_of: Optional[str]
             ) -> Dict[str, Any]:
    """The material-change answer: the governed tables, and the COMPOSITION's
    own findings as the answer.

    WHY THE ANSWER IS THE BRIEF'S AND NOT THE RENDERER'S. This form exists
    because `insight_funded.compose` decides which governed changes are
    materially relevant — B10 proved the composition owns that candidate set.
    Publishing the renderer's portfolio narrative instead would mean the
    composition ran and was discarded, and the reader who asked "what changed
    materially" would get "here is everything that changed".

    No sentence is written here. Each finding's own `summary` is the prose its
    generator produced, and they are joined in the order `insight_engine.select`
    put them in. With no material finding the deterministic narrative stands and
    `insight_count` records the zero, rather than this module asserting that
    nothing mattered.

    The brief travels WHOLE, in the envelope `insight_contract.build_brief`
    already publishes on `/mi/insights/weekly-brief`, so no channel learns a new
    shape and the omissions and thresholds reach the reader with the findings.
    """
    payload = governed_envelope(plan, result, receipt(plan, result, brief),
                               question=question, portfolio_id=portfolio_id,
                               as_of=as_of)
    findings = list(brief.get("insights") or ())
    if findings:
        payload["answer"] = " ".join(
            _as_sentence(str(finding.get("summary")
                             or finding.get("headline") or ""))
            for finding in findings).strip()
    meta = payload.setdefault("metadata", {})
    if isinstance(meta, dict):
        meta["materialChangeBrief"] = dict(brief)
    return payload


def _as_sentence(text: str) -> str:
    """One finding's own prose, terminated. Presentation only."""
    stripped = text.strip()
    if not stripped or stripped.endswith((".", "!", "?")):
        return stripped
    return stripped + "."


def execute(plan: Any, *, client_id: str, output_root: Optional[str],
            tenant_id: str,
            limit_envelope: Optional[Mapping[str, Any]] = None,
            run_id: Optional[str] = None,
            authorised_portfolio_ids: Tuple[str, ...] = (),
            ) -> Dict[str, Any]:
    """One governed analysis, composed into findings. Raises nothing it hides.

    ``question`` is absent from this signature deliberately: the owner defaults
    it to ``""`` and the only read of it inside `period_change` echoes it into
    provenance, so a material-change summary needs no sentence to produce.

    ``scope`` is absent for the reason :func:`analyse` states: the plan is the
    transcription of what was asked, so it is the only place a scope may come
    from. ``portfolio_context`` is absent for the same reason with one more step:
    it is part of an insight's identity hash, so a caller-supplied one could give
    two briefs over two different books the same finding ids. It is read below
    off the scope the analysis ACTUALLY RAN FOR — `scope_ref_from_lens`'s own
    `context_id`, which is `portfolio_lens.context_id` and not a second opinion.
    """
    from mi_agent_api import insight_funded

    eligible, why, detail = check_eligibility(plan)
    if not eligible:
        return {"ok": False, "eligible": False, "reason": why, "detail": detail}

    result = analyse(plan, client_id=client_id, output_root=output_root,
                     tenant_id=tenant_id, mode=WORKFLOW_MODE,
                     authorised_portfolio_ids=authorised_portfolio_ids)

    brief = insight_funded.compose(
        result, tenant_id=tenant_id, portfolio_id=client_id,
        portfolio_context=(result.portfolio_scope.context_id or "total"),
        limit_envelope=limit_envelope, run_id=run_id)

    return {"ok": True, "eligible": True, "result": result, "brief": brief,
            "receipt": receipt(plan, result, brief)}

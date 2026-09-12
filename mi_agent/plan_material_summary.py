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

    period = body.get("period") or {}
    form = period.get("form")
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


def receipt(plan: Any, result: Any, brief: Mapping[str, Any]) -> Dict[str, Any]:
    """What actually ran, from the governed outputs rather than from intent."""
    body = adapter._as_mapping(plan)
    return {
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
        "interpreted_period_form": (body.get("period") or {}).get("form"),
        "period_completed_by_form": (
            (body.get("period") or {}).get("form") in ANCHOR_PERIOD_FORMS),
        "period_resolution": result.period_resolution.to_dict(),
        "composition_owner": "mi_agent_api.insight_funded.compose",
        "materiality_config": brief.get("config_source"),
        "finding_count": brief.get("insight_count"),
        "omission_count": len(brief.get("omitted") or ()),
        "status": brief.get("status"),
    }


def execute(plan: Any, *, client_id: str, output_root: Optional[str],
            tenant_id: str, portfolio_context: str = "total",
            limit_envelope: Optional[Mapping[str, Any]] = None,
            scope: Any = None, run_id: Optional[str] = None,
            authorised_portfolio_ids: Tuple[str, ...] = (),
            ) -> Dict[str, Any]:
    """One governed analysis, composed into findings. Raises nothing it hides.

    ``question`` is absent from this signature deliberately: the owner defaults
    it to ``""`` and the only read of it inside `period_change` echoes it into
    provenance, so a material-change summary needs no sentence to produce.
    """
    from mi_agent_api import insight_funded
    from mi_agent_api.period_change_route import analyse_period_change

    eligible, why, detail = check_eligibility(plan)
    if not eligible:
        return {"ok": False, "eligible": False, "reason": why, "detail": detail}

    result = analyse_period_change(
        client_id=client_id, output_root=output_root,
        mode=WORKFLOW_MODE, period_request=period_request(plan),
        scope=scope, tenant_id=tenant_id,
        authorised_portfolio_ids=tuple(authorised_portfolio_ids),
        include_bridge=True)

    brief = insight_funded.compose(
        result, tenant_id=tenant_id, portfolio_id=client_id,
        portfolio_context=portfolio_context, limit_envelope=limit_envelope,
        run_id=run_id)

    return {"ok": True, "eligible": True, "brief": brief,
            "receipt": receipt(plan, result, brief)}

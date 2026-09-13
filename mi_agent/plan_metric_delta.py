#!/usr/bin/env python3
"""Plan connectivity for `change_form = metric_delta`. THE LAST MISSING ARROW.

WHAT THIS CLOSES, measured live rather than supposed. The ten-question serving
canary asked "By how much has outstanding balance moved from the preceding
reporting date to the latest one?" and the interpretation was flawless —
`metric_delta`, `period_movement`/`movement`, `current_outstanding_balance`,
`relative_pair` with the pair stated, compiled to a plan. It then served nothing:
no adapter claimed the form, `plan_temporal_runtime` claimed it on period form
alone, and refused `CAPABILITY_NOT_GENERIC` because it speaks for single-measure
GENERIC evaluation and never for this owner. The legacy path refused too.

So this adds the adapter and nothing else. The calculation owner already exists
and is the one every other governed period-change figure comes from:
`period_change.workflow.run_period_change_analysis`, run in the mode
`vocabulary.CHANGE_FORM_MODE` names for this form.

WHAT IT SHARES. The temporal perimeter, the relative-grain translation, the slot
perimeter, the one call to the owner and the receipt core all live in
`plan_material_summary` and are called from here, exactly as `plan_attribution`
calls them. A third copy would be a third place they could disagree.

WHERE IT DIFFERS, AND WHY EACH DIFFERENCE IS FORCED BY THE OWNER.

    1. IT NAMES A MEASURE, and the other two do not. `requested_metric` mode
       analyses `requested_fields` — governed FIELD names from the BSR registry —
       so the plan's `canonical_field` bindings are what is passed. A measure
       with no canonical field (the row itself, a capability-owned quantity) has
       no field to select and is REFUSED rather than approximated.
    2. IT ADMITS NO ANCHOR. `CHANGE_FORM_ABSENT_PERIOD_DEFAULT` authorises a
       `current_vs_previous` default for `material_summary` and `attribution` and
       withholds one here, because this owner requires the pair to be named.
       Admitting a lone `current` would be this module inventing the comparison
       state that table deliberately withholds.
    3. IT RECONCILES THE STATISTIC AFTER THE FACT. The owner does not take a
       caller-supplied aggregation: `calculations` reads
       `entry.default_aggregation` from the registry and follows it exactly. So a
       statistic the READER stated explicitly is checked against the one actually
       applied, and a divergence REFUSES rather than publishing a weighted
       average under the word "sum". Where the compiler defaulted the statistic
       there is no reader intent to violate and the registry's own default
       stands — which is the common case and is recorded as such.

NOTHING HERE CALCULATES. No delta, no aggregation, no period arithmetic and no
row filtering. Every figure comes back from the owner. Nothing here reads a
question either: it is not a parameter of any function in this module except the
envelope builder, which echoes it, and the module imports no parser, recogniser,
router or `re`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from mi_agent import plan_runtime_adapter as adapter
from mi_agent import plan_material_summary as shared
from mi_agent.interpretation_v2.vocabulary import CHANGE_FORM_MODE

CHANGE_FORM = "metric_delta"

#: The capability `vocabulary.CHANGE_FORM_CAPABILITY` maps this form to.
CAPABILITY = "period_movement"

#: The governed mode the shared owner runs in, read from the table that owns it
#: rather than restated. `material_summary` runs the same owner in
#: `portfolio_overview`; the mode is the whole distinction between them.
WORKFLOW_MODE: Optional[str] = CHANGE_FORM_MODE.get(CHANGE_FORM)

#: The funded book, and only the funded book.
POPULATION_BASES = frozenset({"funded"})

#: The one result shape this form executes as.
#:
#: THIS COMMENT USED TO SAY THE CANONICALISATION SEAM HAD ALREADY COLLAPSED A
#: READER'S SPELLING BEFORE A PLAN EXISTS, AND IT WAS NOT TRUE OF THIS FORM.
#: `CHANGE_FORM_OPERATION_VARIANTS` carried `material_summary` alone, so a
#: metric delta stated as `compare` arrived here intact and was refused. It is
#: true now: `metric_delta` canonicalises `compare` to `movement` in
#: `normalise.py` rule 5, over the central table, and this set is what it always
#: should have been — the defence in depth for a plan built without that seam,
#: exactly as `material_summary`'s single-operation set is.
#:
#: `period_movement` also admits `rank`, `breakdown`, `series` and `summary`.
#: Each states a shape the requested-metric mode does not produce, so each is
#: refused here rather than flattened into a movement.
OPERATIONS = frozenset({"movement"})

#: The deterministic owner of every figure this form publishes.
CALCULATION_OWNER = shared.CALCULATION_OWNER

# Ineligibility reasons. Stable strings: a bank groups on them and a report
# counts them.
FORM_NOT_CLAIMED = "FORM_NOT_CLAIMED"
CAPABILITY_NOT_PERIOD_MOVEMENT = "CAPABILITY_NOT_PERIOD_MOVEMENT"
OPERATION_NOT_ADMITTED = "OPERATION_NOT_ADMITTED"
POPULATION_NOT_FUNDED = "POPULATION_NOT_FUNDED"
MEASURE_NOT_NAMED = "MEASURE_NOT_NAMED"
MEASURE_HAS_NO_GOVERNED_FIELD = "MEASURE_HAS_NO_GOVERNED_FIELD"
DIMENSION_NOT_SUPPORTED = "DIMENSION_NOT_SUPPORTED"
MEASURE_NOT_ELIGIBLE = "MEASURE_NOT_ELIGIBLE"
STATISTIC_NOT_HONOURED = "STATISTIC_NOT_HONOURED"

#: Reasons this module does not own and therefore does not define.
PERIOD_NOT_A_PAIR = shared.PERIOD_NOT_A_PAIR
PERIOD_GRAIN_UNTRANSLATABLE = shared.PERIOD_GRAIN_UNTRANSLATABLE
FILTER_NOT_SUPPORTED = shared.FILTER_NOT_SUPPORTED
GEOGRAPHY_NOT_SUPPORTED = shared.GEOGRAPHY_NOT_SUPPORTED
TARGET_NOT_SUPPORTED = shared.TARGET_NOT_SUPPORTED
COMPARISON_NOT_SUPPORTED = shared.COMPARISON_NOT_SUPPORTED
SCOPE_NOT_EXPRESSIBLE = shared.SCOPE_NOT_EXPRESSIBLE

#: Plan statistic -> the owner's aggregation vocabulary, for the RECONCILIATION
#: only. Nothing here chooses an aggregation; this is how the two names for one
#: idea are compared after the owner has applied its own.
_AGGREGATION_NAME = {"sum": "sum", "average": "average",
                     "weighted_average": "weighted_average", "share": "share"}


def claims(plan: Any) -> bool:
    """Whether this runtime owns the plan. One structural read, no judgement."""
    return shared._change_form_binding(plan).get("form") == CHANGE_FORM


def _measures(plan: Any) -> Tuple[Dict[str, Any], ...]:
    body = adapter._as_mapping(plan)
    return tuple(measure for output in (body.get("outputs") or ())
                 for measure in (output.get("measures") or ()))


def requested_fields(plan: Any) -> Tuple[str, ...]:
    """The governed FIELDS the owner is asked to analyse, from the plan alone.

    `canonical_field` and never `concept`: `_select_requested` looks each name up
    in the BSR registry, which is keyed by field. The compiler already resolved
    concept to field, and resolving it again here would be a second owner of that
    mapping.
    """
    return tuple(str(measure.get("canonical_field"))
                 for measure in _measures(plan)
                 if measure.get("canonical_field"))


def check_eligibility(plan: Any) -> Tuple[bool, str, str]:
    """``(eligible, reason, detail)``. Every refusal names its perimeter."""
    body = adapter._as_mapping(plan)
    if not claims(plan):
        return False, FORM_NOT_CLAIMED, (
            f"the compiler read this plan's analytical form as "
            f"{shared._change_form_binding(plan).get('form')!r}, not "
            f"{CHANGE_FORM!r}")

    if body.get("capability") != CAPABILITY:
        return False, CAPABILITY_NOT_PERIOD_MOVEMENT, (
            f"capability {body.get('capability')!r} does not implement this "
            f"form; `vocabulary.CHANGE_FORM_CAPABILITY` maps {CHANGE_FORM!r} to "
            f"{CAPABILITY!r}")

    operation = body.get("operation")
    if operation not in OPERATIONS:
        # THE SECOND CLAUSE OF THIS MESSAGE USED TO SAY THE OPERATION BELONGED TO
        # "a different form with a different mode", AND FOR `compare` THAT WAS
        # WRONG. `level_comparison` owns level-against-level comparison and runs
        # `generic_analysis`; no form owns a `compare` over `period_movement`
        # with a named measure, which is why `compare` is now a variant that
        # canonicalises to `movement` upstream instead of arriving here. What is
        # left here genuinely states a shape this owner does not produce.
        return False, OPERATION_NOT_ADMITTED, (
            f"{operation!r} states a result shape the requested-metric mode does "
            f"not produce; only the shapes in "
            f"vocabulary.CHANGE_FORM_OPERATION_VARIANTS[{CHANGE_FORM!r}] "
            f"canonicalise to {sorted(OPERATIONS)[0]!r}")

    base = (body.get("population") or {}).get("base")
    if base not in POPULATION_BASES:
        return False, POPULATION_NOT_FUNDED, (
            f"population base {base!r}: this owner compares governed funded "
            f"snapshots")

    measures = _measures(plan)
    if not measures:
        return False, MEASURE_NOT_NAMED, (
            "the plan names no measure; a movement with no named quantity is a "
            "material summary, which has its own owner and its own mode")
    fieldless = sorted({str(m.get("concept") or "") for m in measures
                        if not m.get("canonical_field")})
    if fieldless:
        return False, MEASURE_HAS_NO_GOVERNED_FIELD, (
            f"measure(s) {fieldless} carry no governed field, and this owner "
            f"selects registry FIELDS; there is nothing for it to analyse and a "
            f"substitute would answer a different question")

    # THE OWNER GROUPS BY NOTHING IN THIS MODE. `_select_requested` routes a
    # requested field with a dimension role into `dimensions`, not `measures`, so
    # a grouped movement is a different output shape than this form publishes.
    grouped = sorted({str(d.get("concept") or "") for output in
                      (body.get("outputs") or ())
                      for d in (output.get("dimensions") or ())})
    if grouped:
        return False, DIMENSION_NOT_SUPPORTED, (
            f"the plan groups the movement by {grouped}; this form publishes a "
            f"movement in the named measure and this owner groups by nothing in "
            f"requested-metric mode")

    ok, why, detail = shared.check_slots_honoured(plan)
    if not ok:
        return ok, why, detail

    # NO ANCHOR. See `shared.check_period` — the comparison state is the reader's
    # to name for this form, because its owner does not supply one.
    return shared.check_period(plan, owner=CHANGE_FORM, admit_anchor=False)


def period_request(plan: Any) -> Any:
    """This form's own face on the shared period translation."""
    return shared.period_request_for(plan, owner=CHANGE_FORM)


def check_owner_honoured(plan: Any, result: Any) -> Tuple[bool, str, str]:
    """Did the owner analyse what the plan asked for? Read from ITS outputs.

    THE PERIMETER CANNOT SEE THIS AND THAT IS WHY IT EXISTS. `_select_requested`
    excludes a field the registry will not admit, with a reason, and carries on;
    the workflow then reports a movement for whatever survived. Serving that
    would answer about the fields that happened to be eligible under the label of
    the field the reader named. So the selection and the applied aggregation are
    reconciled against the plan AFTER execution, exactly as the slice 1 path
    reconciles its bound spec against the executor's receipt.
    """
    selection = result.field_selection.to_dict()
    selected = set(selection.get("selected_measures") or ())
    missing = [field for field in requested_fields(plan) if field not in selected]
    if missing:
        excluded = {str(entry.get("field")): entry.get("reason")
                    for entry in (selection.get("excluded_candidates") or ())}
        return False, MEASURE_NOT_ELIGIBLE, (
            f"the owner did not analyse {missing}: "
            f"{ {field: excluded.get(field, 'not selected') for field in missing} }")

    # AN EXPLICITLY STATED STATISTIC IS THE READER'S AND MUST SURVIVE. Where the
    # compiler DEFAULTED it there is no reader intent to violate, and the
    # registry's `default_aggregation` — which this owner follows exactly — is
    # the governed answer.
    applied = {change.field: change.aggregation for change in result.metric_changes}
    for measure in _measures(plan):
        if measure.get("statistic_defaulted"):
            continue
        field = str(measure.get("canonical_field"))
        wanted = _AGGREGATION_NAME.get(str(measure.get("statistic") or ""))
        got = applied.get(field)
        if wanted and got and wanted != got:
            return False, STATISTIC_NOT_HONOURED, (
                f"the reader asked for {wanted!r} on {field!r} and the governed "
                f"registry aggregates it as {got!r}; publishing the second under "
                f"the first would be a different number wearing the same label")
    return True, "", ""


def receipt(plan: Any, result: Any) -> Dict[str, Any]:
    """What actually ran, from the governed outputs rather than from intent."""
    selection = result.field_selection.to_dict()
    return {
        **shared.change_receipt(plan, result, change_form=CHANGE_FORM,
                                calculation_owner=CALCULATION_OWNER),
        # NO COMPOSITION OWNER. A metric delta IS the governed movement; nothing
        # composes it into findings, and naming a composition that did not run
        # would be provenance for work nobody did.
        "composition_owner": None,
        "requested_fields": list(requested_fields(plan)),
        "selected_measures": selection.get("selected_measures"),
        "excluded_candidates": selection.get("excluded_candidates"),
        "field_selection_policy": (f"{selection.get('policy_name')} "
                                   f"v{selection.get('policy_version')}"),
        # THE MOVEMENT ITSELF, as the owner stated it. Copied, never recomputed:
        # which aggregation was applied is the registry's answer and the receipt's
        # job is to carry it so a reader can see it beside what was asked.
        #
        # AND THE AVAILABILITY EVIDENCE BESIDE IT. `status` alone says a figure
        # was qualified and not why: the live canary published
        # `partially_available` on an interest-rate movement and this receipt
        # could not say how many rows were excluded or against which weight, so
        # neither a reader nor a certification test could tell a sound
        # qualification from a defect. The counts, the weight field and the
        # owner's own notes are its answer; nothing here derives them.
        "metric_movements": [
            {"field": change.field, "aggregation": change.aggregation,
             "weight_field": change.weight_field,
             "start_value": change.start_value, "end_value": change.end_value,
             "movement_value": change.movement_value,
             "movement_unit": change.movement_unit, "status": change.status,
             "valid_population": {"start": change.start.valid_population,
                                  "end": change.end.valid_population},
             "excluded_population": {"start": change.start.excluded_population,
                                     "end": change.end.excluded_population},
             "notes": list(change.notes)}
            for change in result.metric_changes],
        # WHAT THE READER ACTUALLY GOT, named by the owner's summary rather than
        # re-derived here. A certification test that reads only the route and the
        # owner cannot tell a served answer from a served silence; this is the
        # fact that separates them.
        "requested_metric_disposition": [
            {"canonical_field": row.get("canonical_field"),
             "disposition": row.get("disposition"),
             "status": row.get("status")}
            for row in (result.summary.get("requested_metrics") or ())],
    }


def envelope(plan: Any, result: Any, *, question: str,
             portfolio_id: Optional[str], as_of: Optional[str]) -> Dict[str, Any]:
    """The movement, in the envelope the period-change route already publishes.

    `_render` builds the metric-movement table from `result.metric_changes` and
    `build_answer` states the movement from `result.summary` alone, so the answer
    is the estate's own prose over the owner's own figures. Formatting them a
    second time here would be a second chance to publish a different number.
    """
    return shared.governed_envelope(plan, result, receipt(plan, result),
                                    question=question,
                                    portfolio_id=portfolio_id, as_of=as_of)


def execute(plan: Any, *, client_id: str, output_root: Optional[str],
            tenant_id: str,
            authorised_portfolio_ids: Tuple[str, ...] = (),
            ) -> Dict[str, Any]:
    """One governed metric movement. Raises nothing it hides.

    ``question`` is absent from this signature deliberately, and ``scope`` is
    absent because the PLAN carries it — both for the reasons
    `plan_material_summary` states for the same two omissions.
    """
    eligible, why, detail = check_eligibility(plan)
    if not eligible:
        return {"ok": False, "eligible": False, "reason": why, "detail": detail}

    result = shared.analyse(plan, client_id=client_id, output_root=output_root,
                            tenant_id=tenant_id, mode=WORKFLOW_MODE,
                            requested_fields=requested_fields(plan),
                            authorised_portfolio_ids=authorised_portfolio_ids)

    honoured, why, detail = check_owner_honoured(plan, result)
    if not honoured:
        return {"ok": False, "eligible": True, "reason": why, "detail": detail,
                "result": result, "receipt": receipt(plan, result)}

    return {"ok": True, "eligible": True, "result": result,
            "receipt": receipt(plan, result)}

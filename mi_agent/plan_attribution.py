#!/usr/bin/env python3
"""Plan connectivity for `change_form = attribution`. THE MISSING ARROW ONLY.

WHAT THIS IS. The thin adapter between a compiled `GovernedQueryPlan` that was
read as "what drove the movement" and the governed owner that already answers
it. Previous evidence recorded the state this closes exactly:

    ATTRIBUTION_TEMPORAL_CALCULATION_EXISTS  = YES
    ATTRIBUTION_CURRENT_VS_PREVIOUS_SUPPORTED = YES
    ATTRIBUTION_PLAN_CONNECTIVITY_EXISTS     = NO

So this module adds the third line and nothing else. It contains no bridge
arithmetic, no movement calculation, no period arithmetic and no balance, count
or residual of its own — every figure it publishes comes back from
`mi_agent.period_change.bridge.balance_bridge`, reached through
`run_period_change_analysis` with `include_bridge=True`, which is where every
other governed period-change figure comes from.

WHY THAT OWNER AND NOT THE OTHER ONE. The estate holds two bridges.
`mi_agent_api.evolution.funded_bridge` attributes a balance movement ACROSS A
DIMENSION and resolves its own opening period from `start_period` /
`window_periods`; `period_change.bridge.balance_bridge` decomposes the movement
BY LOAN IDENTITY — new, exited and continuing — over the pair
`period_change.periods.resolve_periods` selected. Only the second one resolves
its window through the governed period resolver, and the resolver is the thing
that makes the authorised `current_vs_previous` default for this form mean the
two adjacent GOVERNED snapshots rather than a calendar step. A plan whose window
the reader left to the owner can therefore only be honoured by that owner, so
that owner is this form's.

WHAT IT SHARES, AND WHY IT SHARES IT. The temporal perimeter, the relative-grain
translation, the slot perimeter, the period request and the receipt core all live
in `plan_material_summary`, and this module calls them. `attribution` and
`material_summary` are the two forms `vocabulary.CHANGE_FORM_ABSENT_PERIOD_DEFAULT`
authorises a `current_vs_previous` default for, and they execute through one
entry point, so they have one temporal contract and one set of limits. A second
copy here would be a second place the two could come to disagree about what the
compiler wrote.

WHAT IT MAY NOT DO. Nothing here reads a question: it is not a parameter of any
function in this module, and the module imports no parser, recogniser, router or
`re`. It does not read the model's raw claim either — `claims` reads
`provenance.compiler_bindings["change_form"]`, the COMPILER's derived reading.
It chooses no date, no snapshot and no file. A plan slot the owner cannot honour
REFUSES with that slot named; it is never dropped and the scope is never widened
to the whole book.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from mi_agent import plan_runtime_adapter as adapter
from mi_agent import plan_material_summary as shared
from mi_agent.interpretation_v2.vocabulary import (CHANGE_FORM_MODE,
                                                  SPECIALIST_MEASURES)

#: The form this runtime owns.
CHANGE_FORM = "attribution"

#: The capability `vocabulary.CHANGE_FORM_CAPABILITY` maps this form to. Asserted
#: against the plan rather than assumed: a plan claiming this form under another
#: capability was not built by the compiler this adapter is downstream of.
CAPABILITY = "funded_bridge"

#: The governed mode the shared owner runs in. `CHANGE_FORM_MODE` carries an
#: entry only where the capability alone does not separate two forms, and this
#: form is absent from it — so it states no mode and the owner's own default
#: applies, which is the table's documented meaning rather than a choice here.
WORKFLOW_MODE: Optional[str] = CHANGE_FORM_MODE.get(CHANGE_FORM)

#: The funded book, and only the funded book.
POPULATION_BASES = frozenset({"funded"})

#: Result shapes this form executes as. Both are `funded_bridge`'s own governed
#: operations (`vocabulary.CAPABILITY_OPERATIONS`): `bridge` asks for the
#: decomposition and `movement` for the net figure it reconciles to. The same
#: owner produces both, so both are admitted and neither is rewritten.
OPERATIONS = frozenset({"bridge", "movement"})

#: Measures this form may name, READ FROM THE VOCABULARY that owns them. The
#: capability's specialist measures are `funded_balance_movement` (the quantity
#: decomposed) and `bridge_component` (the decomposition itself); a plan naming
#: anything else is asking a question the bridge does not answer.
ADMITTED_MEASURES = frozenset(SPECIALIST_MEASURES[CAPABILITY])

# Ineligibility reasons. Stable strings: a bank groups on them and a report
# counts them.
FORM_NOT_CLAIMED = "FORM_NOT_CLAIMED"
CAPABILITY_NOT_BRIDGE = "CAPABILITY_NOT_BRIDGE"
OPERATION_NOT_ADMITTED = "OPERATION_NOT_ADMITTED"
POPULATION_NOT_FUNDED = "POPULATION_NOT_FUNDED"
MEASURE_NOT_A_BRIDGE_TARGET = "MEASURE_NOT_A_BRIDGE_TARGET"
DIMENSION_NOT_BRIDGEABLE = "DIMENSION_NOT_BRIDGEABLE"
BRIDGE_NOT_AVAILABLE = "BRIDGE_NOT_AVAILABLE"

#: Reasons this module does not define because it does not own them. Re-exported
#: so a caller reading an attribution refusal needs one import, and so the two
#: forms cannot come to refuse an untranslatable window under two names.
PERIOD_NOT_A_PAIR = shared.PERIOD_NOT_A_PAIR
PERIOD_GRAIN_UNTRANSLATABLE = shared.PERIOD_GRAIN_UNTRANSLATABLE
FILTER_NOT_SUPPORTED = shared.FILTER_NOT_SUPPORTED
GEOGRAPHY_NOT_SUPPORTED = shared.GEOGRAPHY_NOT_SUPPORTED
TARGET_NOT_SUPPORTED = shared.TARGET_NOT_SUPPORTED
COMPARISON_NOT_SUPPORTED = shared.COMPARISON_NOT_SUPPORTED
SCOPE_NOT_EXPRESSIBLE = shared.SCOPE_NOT_EXPRESSIBLE

#: The deterministic owner of every figure this form publishes.
CALCULATION_OWNER = "mi_agent.period_change.bridge.balance_bridge"


def claims(plan: Any) -> bool:
    """Whether this runtime owns the plan. One structural read, no judgement.

    True does NOT mean eligible. An attribution plan outside this contract is
    claimed here and then refused by `check_eligibility`, so it is refused with a
    reason about THIS form rather than falling through to a runtime that would
    refuse it for not being something else.
    """
    return shared._change_form_binding(plan).get("form") == CHANGE_FORM


def check_eligibility(plan: Any) -> Tuple[bool, str, str]:
    """``(eligible, reason, detail)``. Every refusal names its perimeter."""
    body = adapter._as_mapping(plan)
    if not claims(plan):
        return False, FORM_NOT_CLAIMED, (
            f"the compiler read this plan's analytical form as "
            f"{shared._change_form_binding(plan).get('form')!r}, not "
            f"{CHANGE_FORM!r}")

    if body.get("capability") != CAPABILITY:
        return False, CAPABILITY_NOT_BRIDGE, (
            f"capability {body.get('capability')!r} does not implement this "
            f"form; `vocabulary.CHANGE_FORM_CAPABILITY` maps {CHANGE_FORM!r} to "
            f"{CAPABILITY!r} and a plan disagreeing with it was not compiled")

    operation = body.get("operation")
    if operation not in OPERATIONS:
        return False, OPERATION_NOT_ADMITTED, (
            f"{operation!r} states a result shape the balance bridge does not "
            f"produce; it is refused rather than flattened into a bridge")

    base = (body.get("population") or {}).get("base")
    if base not in POPULATION_BASES:
        return False, POPULATION_NOT_FUNDED, (
            f"population base {base!r}: this bridge reconciles governed funded "
            f"snapshots over a stable loan identifier")

    outputs = tuple(body.get("outputs") or ())
    named = [str(m.get("concept") or "") for output in outputs
             for m in (output.get("measures") or ())]
    stray = sorted(set(named) - ADMITTED_MEASURES)
    if stray:
        return False, MEASURE_NOT_A_BRIDGE_TARGET, (
            f"the plan names measure(s) {stray}; this owner decomposes the "
            f"funded balance movement and publishes no other quantity, so a "
            f"different measure is a different question with a different owner")

    # THE BRIDGE DECOMPOSES BY LOAN IDENTITY, NOT BY A CATEGORY. Its components
    # are new, exited and continuing loans; it has no group-by. A plan asking to
    # attribute the movement across a governed dimension is asking for
    # `evolution.funded_bridge`, whose window this form's temporal contract
    # cannot reach, so it is REFUSED with the dimension named rather than
    # answered by the decomposition the reader did not ask for.
    grouped = [str(d.get("concept") or "") for output in outputs
               for d in (output.get("dimensions") or ())]
    if grouped:
        return False, DIMENSION_NOT_BRIDGEABLE, (
            f"the plan attributes across {sorted(set(grouped))}; this owner "
            f"decomposes the movement by loan identity and groups by nothing")

    ok, why, detail = shared.check_slots_honoured(plan)
    if not ok:
        return ok, why, detail

    return shared.check_period(plan, owner=CHANGE_FORM)


def period_request(plan: Any) -> Any:
    """This form's own face on the shared period translation."""
    return shared.period_request_for(plan, owner=CHANGE_FORM)


def receipt(plan: Any, result: Any) -> Dict[str, Any]:
    """What actually ran, from the governed outputs rather than from intent."""
    bridge = getattr(result, "balance_bridge", None)
    return {
        **shared.change_receipt(plan, result, change_form=CHANGE_FORM,
                                calculation_owner=CALCULATION_OWNER),
        # NO COMPOSITION OWNER. `material_summary` records one because a brief is
        # composed from the analysis; an attribution answer IS the governed
        # decomposition, and naming a composition that does not exist would be
        # provenance for work nobody did.
        "composition_owner": None,
        # THE BRIDGE'S OWN STATUS AND ITS OWN RECONCILIATION. Copied, not
        # recomputed: whether the components sum to the movement within
        # tolerance is the owner's finding and the receipt's job is to carry it.
        "bridge_status": getattr(bridge, "status", None),
        "bridge_reconciles": getattr(bridge, "reconciles", None),
        "bridge_residual": getattr(bridge, "residual", None),
        "bridge_balance_field": getattr(bridge, "balance_field", None),
        "bridge_identifier_fields": list(
            getattr(bridge, "identifier_fields", None) or ()),
    }


def envelope(plan: Any, result: Any, *, question: str,
             portfolio_id: Optional[str], as_of: Optional[str]
             ) -> Dict[str, Any]:
    """The attribution answer, in the envelope the period-change route publishes.

    THE RENDERER IS NOT REPLACED AND THE ANSWER IS NOT REWRITTEN. `_render`
    already builds the "Balance bridge" table — opening balance, new-loan
    balance, exited-loan balance, movement on continuing loans, closing balance —
    from `result.balance_bridge`, which IS the decomposition this form asked for,
    and `build_answer` states the movement from `result.summary` alone. Writing a
    bridge sentence here would mean formatting the owner's figures a second time,
    and a second formatting of a published number is a second chance to publish a
    different one.

    So this adds the receipt and nothing else. What makes the answer an
    ATTRIBUTION rather than an overview is which table leads and which owner the
    receipt names, not prose composed here.
    """
    return shared.governed_envelope(plan, result, receipt(plan, result),
                                    question=question,
                                    portfolio_id=portfolio_id, as_of=as_of)


def execute(plan: Any, *, client_id: str, output_root: Optional[str],
            tenant_id: str,
            authorised_portfolio_ids: Tuple[str, ...] = (),
            ) -> Dict[str, Any]:
    """One governed bridge. Raises nothing it hides.

    ``question`` is absent from this signature deliberately, and ``scope`` is
    absent because the PLAN carries it — both for the reasons
    `plan_material_summary` states for the same two omissions.
    """
    from mi_agent.period_change.models import BRIDGE_STATUS_AVAILABLE

    eligible, why, detail = check_eligibility(plan)
    if not eligible:
        return {"ok": False, "eligible": False, "reason": why, "detail": detail}

    result = shared.analyse(plan, client_id=client_id, output_root=output_root,
                            tenant_id=tenant_id, mode=WORKFLOW_MODE,
                            authorised_portfolio_ids=authorised_portfolio_ids)

    # A BRIDGE THAT DID NOT RECONCILE IS NOT AN ATTRIBUTION. The owner publishes
    # its own status for every way the decomposition can fail — no balance
    # field, no stable identifier, duplicate identifiers, mixed currency, or a
    # residual outside tolerance — and each of those makes the components an
    # incomplete account of the movement. Serving one anyway would publish an
    # attribution that does not add up, so the owner's status is honoured.
    bridge = result.balance_bridge
    status = getattr(bridge, "status", None)
    if bridge is None or status != BRIDGE_STATUS_AVAILABLE:
        return {"ok": False, "eligible": True, "reason": BRIDGE_NOT_AVAILABLE,
                "detail": (f"the governed balance bridge reported "
                           f"{status!r}; its components are not a complete "
                           f"account of the movement"),
                "result": result, "receipt": receipt(plan, result)}

    return {"ok": True, "eligible": True, "result": result,
            "receipt": receipt(plan, result)}

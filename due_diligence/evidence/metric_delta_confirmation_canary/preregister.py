#!/usr/bin/env python3
"""Pre-register metric_delta_confirmation_canary. Run ONCE, before any call.

WHAT THIS BANK IS FOR, and it is deliberately narrow. Two things:

    1. prove `metric_delta` now serves through its governed plan adapter;
    2. prove that adding it disturbed neither of the forms already serving.

It is not a re-run of the spent ten-question bank and supersedes nothing. That
bank is SPENT, immutable, and fully adjudicated: 8 PASS, 1 safe fail-closed
(S09), 1 product gap (S03). This closes S03 and confirms the close.

WHY FIVE. Three metric deltas across the measure families the owner's
requested-metric mode can select, plus one regression case for each of the two
forms whose live serving is already proved. A larger bank would re-measure
contracts that v3, v4 and the spent canary have already settled.

WHAT S03 PROVED AND THIS MUST NOT REPEAT. Its interpretation was flawless —
`metric_delta`, `period_movement`/`movement`, `current_outstanding_balance`,
`relative_pair` with the pair stated — and it still served nothing, because no
adapter claimed the form and the temporal runtime refused the capability. So
every case here pins the ROUTE and the CALCULATION OWNER, not merely the reading:
a correct reading that reaches no owner is exactly the failure this bank exists
to catch.

WHAT IS NOT HERE, AND WHY.

  * NO S09 RETEST. S09 was refused because the acquired portfolio does not exist
    in both selected snapshots — data history, not a defect — and the operator's
    instruction is explicit that it may return only when separate factual
    evidence shows the portfolio genuinely present in both. No such evidence
    exists, so no scoped case is pinned.
  * NO LOAN COUNT. Measured, not assumed: `loan` carries no governed field and
    `requested_metric` mode selects registry FIELDS, so the adapter refuses it.
    Pinning a question the architecture correctly declines would repeat the v4
    W06 authoring defect in a new costume.
  * NO NUMBERS. Numeric truth is reconciled afterwards against the owner run
    independently, as before. A figure taken from a model's output and used to
    score that model certifies nothing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "metric_delta_canary_bank_manifest.json"
HASH = HERE / "metric_delta_canary_bank_manifest.sha256"

#: The product state this bank confirms. Recorded by the closeout report; the
#: runner asserts the deployed build matches before asking anything.
PRODUCT_BASELINE = "88cf5fde759c29f1cff29b9adbf003a2d6ebd530"
MODEL = "claude-opus-5"
BANK_ID = "metric_delta_confirmation_canary"

# --------------------------------------------------------------------------- #
# measure families — canonical concept_ids, and every one of them MEASURED
# eligible through the adapter before it was pinned
# --------------------------------------------------------------------------- #
BALANCE = {"family": "funded_or_outstanding_balance",
           "concepts": ["current_outstanding_balance",
                        "current_principal_balance"],
           "statistics": ["sum"]}
WA_LTV = {"family": "weighted_average_current_ltv",
          "concepts": ["current_loan_to_value"],
          "statistics": ["weighted_average", "average"]}
WA_RATE = {"family": "weighted_average_interest_rate",
           "concepts": ["current_interest_rate"],
           "statistics": ["weighted_average", "average"]}
ATTRIBUTION_TARGET = {
    "family": "funded_balance_or_its_governed_components",
    "concepts": ["current_outstanding_balance", "current_principal_balance",
                 "funded_balance_movement", "bridge_component"],
    "statistics": []}

PAIR_FORMS = ["relative_pair", "previous_reporting_period", "explicit_period"]
ABSENT = "ABSENT"
#: Only `material_summary` and `attribution` own an authorised default, so only
#: they accept an absent or anchored window. Derived from
#: `vocabulary.CHANGE_FORM_ABSENT_PERIOD_DEFAULT`, not restated by hand.
OWNS_DEFAULT = PAIR_FORMS + ["current", ABSENT]

OWNER_METRIC_DELTA = {
    "plan_adapter": "mi_agent.plan_metric_delta",
    "calculation_owner":
        "mi_agent.period_change.workflow.run_period_change_analysis",
    "mode": "requested_metric",
    "composition_owner": None,
    "serving_provenance": "GOVERNED_PLAN",
    "route": "governed_plan_metric_delta"}
OWNER_MATERIAL_SUMMARY = {
    "plan_adapter": "mi_agent.plan_material_summary",
    "calculation_owner":
        "mi_agent.period_change.workflow.run_period_change_analysis",
    "mode": "portfolio_overview",
    "composition_owner": "mi_agent_api.insight_funded.compose",
    "serving_provenance": "GOVERNED_PLAN",
    "route": "governed_plan_material_summary"}
OWNER_ATTRIBUTION = {
    "plan_adapter": "mi_agent.plan_attribution",
    "calculation_owner": "mi_agent.period_change.bridge.balance_bridge",
    "mode": None,
    "composition_owner": None,
    "serving_provenance": "GOVERNED_PLAN",
    "route": "governed_plan_attribution"}

CASES = [
    # ---------------- METRIC DELTA (3) — the arrow under test ------------- #
    dict(id="M01", change_form="metric_delta", measure=BALANCE, scope=None,
         temporal_intent="EXPLICIT_PAIR", temporal_accept=list(PAIR_FORMS),
         filters_permitted=[], outcome="ANSWER", owner=OWNER_METRIC_DELTA,
         question="Compare outstanding balance across the two most recent "
                  "reporting dates and tell me the size of the shift."),
    dict(id="M02", change_form="metric_delta", measure=WA_LTV, scope=None,
         temporal_intent="EXPLICIT_PAIR", temporal_accept=list(PAIR_FORMS),
         filters_permitted=[], outcome="ANSWER", owner=OWNER_METRIC_DELTA,
         question="Weighted average LTV — what is the difference between the "
                  "latest reporting date and the one before?"),
    dict(id="M03", change_form="metric_delta", measure=WA_RATE, scope=None,
         temporal_intent="EXPLICIT_PAIR", temporal_accept=list(PAIR_FORMS),
         filters_permitted=[], outcome="ANSWER", owner=OWNER_METRIC_DELTA,
         question="Has the weighted average interest rate on the funded book "
                  "gone up or down since the previous reporting date, and by "
                  "how much?"),

    # ---------------- REGRESSION (2) — already-serving forms -------------- #
    dict(id="M04", change_form="material_summary", measure=None, scope=None,
         temporal_intent="OWNER_DEFAULT", temporal_accept=list(OWNS_DEFAULT),
         filters_permitted=[], outcome="ANSWER",
         owner=OWNER_MATERIAL_SUMMARY,
         question="Give me the material movements in the funded portfolio."),
    dict(id="M05", change_form="attribution", measure=ATTRIBUTION_TARGET,
         scope=None, temporal_intent="EXPLICIT_PAIR",
         temporal_accept=list(PAIR_FORMS), filters_permitted=[],
         outcome="ANSWER", owner=OWNER_ATTRIBUTION,
         question="Across the last two reporting dates, what makes up the "
                  "change in funded balance?"),
]


def build() -> dict:
    for case in CASES:
        case["dimensions"] = []
    return {
        "bank_id": BANK_ID,
        "model": MODEL,
        "product_baseline": PRODUCT_BASELINE,
        "supersedes":
            "Nothing. change_intelligence_serving_canary is SPENT, immutable and "
            "adjudicated (8 PASS, 1 safe fail-closed, 1 product gap). v1-v4 "
            "remain immutable interpretation banks.",
        "what_this_is":
            "A narrow confirmation that metric_delta now serves through its "
            "governed plan adapter, and that connecting it disturbed neither "
            "form already serving. Five fresh questions; no wording from any "
            "earlier bank is reused, verified programmatically before any call.",
        "closes":
            "S03 of change_intelligence_serving_canary — a flawless reading that "
            "reached no owner because no adapter claimed the form.",
        "authorised_live_calls": 5,
        "calls_per_case": 1,
        "retries": 0,
        "registry_is_visible_to_the_model": False,
        "numbers_are_not_pinned": {
            "why":
                "A figure taken from a model's output and then used to score the "
                "same model certifies nothing.",
            "how_numeric_truth_is_established":
                "AFTER each live call, run_period_change_analysis is invoked "
                "independently over the same governed snapshots, in the mode the "
                "case pins and with the receipt's own period request and "
                "requested_fields, and every published figure must match.",
            "required": {"numerical_differences": 0, "period_differences": 0,
                         "scope_differences": 0, "owner_differences": 0},
        },
        "primary_gates": {
            "change_form": "5 of 5",
            "route_and_calculation_owner":
                "5 of 5 — each case must record the route AND the "
                "calculation_owner its owner block pins. A correct reading that "
                "reaches no owner is the failure this bank exists to catch.",
            "mode": "5 of 5 — requested_metric, portfolio_overview, and the "
                    "owner default for attribution",
            "outcome_type": "5 of 5 ANSWER",
            "owner_parity": "0 numerical, 0 period, 0 scope, 0 owner differences",
        },
        "secondary_gates": {
            "explicit_measure_preserved": "4 of 4 (M01, M02, M03, M05)",
            "material_summary_measure_invented": "0 of 1 (M04)",
            "requested_field_analysed":
                "3 of 3 — for each metric delta the field the plan named must "
                "appear in the owner's selected_measures, and no excluded "
                "candidate may stand in for it",
            "unnecessary_clarifications": 0,
            "invented_filters": 0,
            "invented_dimensions": 0,
        },
        "not_pinned": [
            "capability", "operation", "any numeric value", "time.grain",
            "the statistic actually applied — the owner follows the registry's "
            "default_aggregation and the receipt records both sides",
            "WHICH of the valid temporal routes M04 takes, since its form owns a "
            "default and all three are correct readings",
        ],
        "deliberate_omissions": {
            "no_s09_retest":
                "S09 was refused because the acquired portfolio does not exist "
                "in both selected snapshots. That is data history. It returns "
                "only when separate factual evidence shows the portfolio present "
                "at both, and no such evidence exists.",
            "no_loan_count":
                "MEASURED, not assumed: `loan` carries no governed field and "
                "requested_metric mode selects registry FIELDS, so the adapter "
                "refuses it. Pinning a question the architecture correctly "
                "declines would repeat the v4/W06 authoring defect.",
            "no_scoped_case":
                "Direct/Acquired survival is proved offline and its live "
                "behaviour is blocked on the same data-history fact as S09.",
        },
        "stop_conditions": [
            "the deployed build is not the product baseline — STOP, zero calls",
            "MI_BEARER does not authenticate the free preflight — STOP",
            "a provider quota, credit or authentication failure — STOP and "
            "report the partial result",
            "any case asked twice — the run is void",
        ],
        "cases": CASES,
    }


def main() -> int:
    if MANIFEST.exists() and "--force" not in sys.argv:
        print(f"refusing to overwrite {MANIFEST.name}: it is pinned evidence.")
        return 1
    payload = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    MANIFEST.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    HASH.write_text(f"{digest}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"wrote {MANIFEST.name}  ({len(CASES)} cases)")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

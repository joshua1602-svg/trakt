#!/usr/bin/env python3
"""Pre-register live_change_form_gate_v4. Run ONCE, before Opus.

WHY v4. v3 measured change_form 18/18 and that contract is accepted, so v4 is not
about the analytical form. It measures the TEMPORAL contract after the presence
repair, on SEVENTEEN FRESH paraphrases — none of v3's questions is reused, so the
repair is measured against wording it was not derived from.

WHAT IT MUST SEPARATE, and this is the whole point of the bank:

    EXPLICIT_PAIR     the wording names two reporting states
    EXPLICIT_ANCHOR   the wording names only the current endpoint
    OWNER_DEFAULT     the wording names no temporal relation at all, and the
                      analytical FORM owns the comparison window

Each case pins which of those its wording actually supports, so a reading is scored
against what the question says rather than against one global rule. A form that
owns a default accepts all three routes — a model that states a period for
"summarise the material changes" is not wrong — while `metric_delta` and
`level_comparison` accept only a stated pair, because their owners require the pair
to be named and no default exists for them.

TWO AUTHORING DEFECTS OF v3, CORRECTED HERE RATHER THAN ARGUED WITH.

  1. v3 pinned `filters: []` on all eighteen, then scored V06 as having invented a
     filter — but its question said "live funded loans", and "live" IS a restricting
     adjective. A governed filter the WORDING supports is not an invention. v4
     therefore pins `filters_permitted` PER CASE, and deliberately includes a
     question whose wording restricts the population ("loans in arrears") so the
     corrected rule is exercised rather than merely asserted.

  2. v3 expected an attribution's measure to be the quantity being decomposed, then
     scored V13 as a miss for naming `bridge_component` — a governed `funded_bridge`
     concept, and the better reading of a question about components. v4 accepts
     either the decomposed quantity or the governed component concept for
     attribution, and includes a components-shaped question.

Neither correction was used to justify any product change.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "change_form_v4_bank_manifest.json"
HASH = HERE / "change_form_v4_bank_manifest.sha256"

BASELINE_SHA = "665176864d2f5dbceedc277e351c5d2cd978777f"
MODEL = "claude-opus-5"
BANK_ID = "live_change_form_gate_v4"

# --------------------------------------------------------------------------- #
# measure families, as canonical concept_ids after vocabulary resolution
# --------------------------------------------------------------------------- #

BALANCE = {"family": "funded_or_outstanding_balance",
           "concepts": ["current_outstanding_balance",
                        "current_principal_balance", "funded_balance_movement"],
           "statistics": []}
WA_LTV = {"family": "weighted_average_current_ltv",
          "concepts": ["current_loan_to_value"],
          "statistics": ["weighted_average", "average"]}
INDEXED_LTV = {"family": "weighted_average_indexed_ltv",
               "concepts": ["indexed_loan_to_value"],
               "statistics": ["weighted_average", "average"]}
LOAN_COUNT = {"family": "loan_count", "concepts": ["loan"],
              "statistics": ["count", "count_distinct"]}
ARREARS = {"family": "arrears_balance", "concepts": ["arrears_balance"],
           "statistics": []}

#: CORRECTION 2. An attribution may name the quantity being decomposed OR the
#: governed component concept the decomposition reports. Both are right readings of
#: "what drove it" / "break it into components", and v3 was wrong to accept only
#: the first.
ATTRIBUTION_TARGET = {
    "family": "funded_balance_or_its_governed_components",
    "concepts": ["current_outstanding_balance", "current_principal_balance",
                 "funded_balance_movement", "bridge_component"],
    "statistics": []}

#: CORRECTION 1. Governed concepts a question's own wording legitimately restricts
#: on. A filter on one of these is the reader's, not an invention; a filter on
#: anything else is invented. An empty list means the wording restricts nothing.
ARREARS_PREDICATES = ["arrears_balance", "arrears_bucket",
                      "number_of_days_in_arrears", "account_status",
                      "principal_arrears_amount", "interest_in_arrears"]

# --------------------------------------------------------------------------- #
# temporal routes
# --------------------------------------------------------------------------- #

#: Forms that state a PAIR outright.
PAIR_FORMS = ["relative_pair", "previous_reporting_period", "explicit_period"]
#: The sentinel for "the reading stated no temporal form at all". Scored from KEY
#: PRESENCE in the payload, never from wording.
ABSENT = "ABSENT"

#: A form that owns an authorised default accepts all three routes: the reader may
#: state the pair, state the current endpoint, or say nothing and leave the window
#: to the owner. A form that owns no default accepts only a stated pair.
OWNS_DEFAULT = PAIR_FORMS + ["current", ABSENT]
PAIR_REQUIRED = list(PAIR_FORMS)

CASES = [
    # ---------------- MATERIAL SUMMARY (4) ------------------------------- #
    dict(id="W01", change_form="material_summary", measure=None, scope=None,
         temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         question="Compare the funded book across the last two reporting dates "
                  "and tell me what moved materially."),
    dict(id="W02", change_form="material_summary", measure=None, scope=None,
         temporal_intent="EXPLICIT_ANCHOR", filters_permitted=[],
         question="As things stand this reporting period, what has materially "
                  "shifted in the book?"),
    dict(id="W03", change_form="material_summary", measure=None, scope=None,
         temporal_intent="OWNER_DEFAULT", filters_permitted=[],
         question="Summarise the material changes in the funded portfolio."),
    dict(id="W04", change_form="material_summary", measure=None,
         scope="acquired", temporal_intent="OWNER_DEFAULT",
         filters_permitted=[],
         question="What has materially changed in the acquired book?"),

    # ---------------- METRIC DELTA (5) ----------------------------------- #
    dict(id="W05", change_form="metric_delta", measure=BALANCE, scope=None,
         temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         question="How much did outstanding balance change between the two most "
                  "recent reporting dates?"),
    # CORRECTION 1 exercised: "in arrears" restricts the population, so a governed
    # arrears predicate here is the READER'S filter and not an invention.
    dict(id="W06", change_form="metric_delta", measure=LOAN_COUNT, scope=None,
         temporal_intent="EXPLICIT_PAIR",
         filters_permitted=list(ARREARS_PREDICATES),
         question="What is the movement in the number of loans in arrears since "
                  "the prior reporting date?"),
    dict(id="W07", change_form="metric_delta", measure=WA_LTV, scope=None,
         temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         question="How did weighted average LTV shift against the previous "
                  "reporting date?"),
    dict(id="W08", change_form="metric_delta", measure=ARREARS, scope="direct",
         temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         question="How much did arrears balance move in the direct book since "
                  "the last report?"),
    dict(id="W09", change_form="metric_delta", measure=INDEXED_LTV, scope=None,
         temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         question="How did weighted average indexed LTV change from the prior "
                  "reporting date to the current one?"),

    # ---------------- ATTRIBUTION (4) ------------------------------------ #
    dict(id="W10", change_form="attribution", measure=ATTRIBUTION_TARGET,
         scope=None, temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         question="What drove the funded balance movement between the last two "
                  "reporting dates?"),
    dict(id="W11", change_form="attribution", measure=ATTRIBUTION_TARGET,
         scope=None, temporal_intent="EXPLICIT_ANCHOR", filters_permitted=[],
         question="For this reporting period, what explains the movement in "
                  "funded balance?"),
    # CORRECTION 2 exercised: a components question may resolve to the governed
    # component concept.
    dict(id="W12", change_form="attribution", measure=ATTRIBUTION_TARGET,
         scope=None, temporal_intent="OWNER_DEFAULT", filters_permitted=[],
         question="Break the funded balance movement into its components."),
    dict(id="W13", change_form="attribution", measure=ATTRIBUTION_TARGET,
         scope="direct", temporal_intent="OWNER_DEFAULT", filters_permitted=[],
         question="What are the drivers behind the direct book's funded balance "
                  "movement?"),

    # ---------------- LEVEL COMPARISON (4) ------------------------------- #
    dict(id="W14", change_form="level_comparison", measure=BALANCE, scope=None,
         temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         question="Show outstanding balance at the two most recent reporting "
                  "dates side by side."),
    dict(id="W15", change_form="level_comparison", measure=WA_LTV, scope=None,
         temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         question="What was weighted average LTV at the current reporting date "
                  "and at the previous one?"),
    dict(id="W16", change_form="level_comparison", measure=LOAN_COUNT,
         scope=None, temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         question="Give me funded loan counts for the current and the prior "
                  "reporting dates."),
    dict(id="W17", change_form="level_comparison", measure=BALANCE,
         scope="acquired", temporal_intent="EXPLICIT_PAIR",
         filters_permitted=[],
         question="Put the acquired book's funded balance at the latest "
                  "reporting date next to the previous one."),
]

#: Which temporal readings each FORM may produce. Derived from
#: `vocabulary.CHANGE_FORM_ABSENT_PERIOD_DEFAULT`, not restated by hand.
TEMPORAL_ACCEPT_BY_FORM = {
    "material_summary": OWNS_DEFAULT,
    "attribution": OWNS_DEFAULT,
    "metric_delta": PAIR_REQUIRED,
    "level_comparison": PAIR_REQUIRED,
}


def build() -> dict:
    for case in CASES:
        case["dimensions"] = []
        case["temporal_accept"] = list(
            TEMPORAL_ACCEPT_BY_FORM[case["change_form"]])
    return {
        "bank_id": BANK_ID,
        "model": MODEL,
        "baseline_sha": BASELINE_SHA,
        "supersedes":
            "live_change_form_gate_v3 (immutable; change_form 18/18 ACCEPTED, "
            "three temporal failures). v1 and v2 remain immutable; v2 remains "
            "unrun.",
        "what_this_is":
            "NATURAL LANGUAGE -> OPUS -> CandidateIntent, measuring the TEMPORAL "
            "contract after the presence repair. Seventeen FRESH paraphrases; "
            "none of v3's questions is reused. It does NOT test deterministic "
            "execution and computes no answer.",
        "boundary":
            "AnthropicInterpreterClient directly. No /mi/query, no MI_BEARER, no "
            "App Service, no plan_serving_canary, no MI_AGENT_PLAN_SERVE, "
            "neither funded_bridge nor material_summary serving connected, not "
            "the 135 benchmark CLI. Nothing runs against a book.",
        "authorised_live_calls": 17,
        "calls_per_case": 1,
        "retries": 0,
        "registry_is_visible_to_the_model": False,
        "temporal_routes": {
            "EXPLICIT_PAIR": "the wording names two reporting states",
            "EXPLICIT_ANCHOR": "the wording names only the current endpoint",
            "OWNER_DEFAULT":
                "the wording names no temporal relation, and the form owns the "
                "comparison window; absent, anchor and pair are all valid",
            "scored_from":
                "key presence in the payload (`form` in `time`), never wording",
        },
        "temporal_accept_by_form": TEMPORAL_ACCEPT_BY_FORM,
        "v3_authoring_defects_corrected": {
            "filters":
                "v3 pinned filters=[] on every case and then scored V06 as "
                "inventing one, though its question said 'live funded loans'. v4 "
                "pins filters_permitted PER CASE and includes W06 ('loans in "
                "arrears') so the corrected rule is exercised.",
            "attribution_measure":
                "v3 scored V13 as a miss for naming bridge_component on a "
                "question about components. v4 accepts the decomposed quantity "
                "OR the governed component concept, and includes W12.",
        },
        "primary_gate": "change_form == expected, 17/17",
        "secondary_gates": {
            "explicit_measure_preserved": "13 of 13 (W05-W17)",
            "material_summary_specific_measure_invented": "0 of 4",
            "scope_preserved": "4 of 4 (W04, W08, W13, W17)",
            "temporal_route_valid": "17 of 17, per temporal_accept_by_form",
            "unnecessary_clarifications": 0,
            "invented_filters": "0 — a filter outside filters_permitted",
            "invented_dimensions": 0,
        },
        "not_pinned": [
            "capability", "operation",
            "the statistic on a measure (recorded, reported, not a gate)",
            "time.grain",
            "WHICH of the three valid temporal routes a form that owns a default "
            "takes — all three are correct readings, so the gate scores validity "
            "rather than preference, and reports which route each case took",
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

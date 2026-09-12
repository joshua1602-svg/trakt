#!/usr/bin/env python3
"""Pre-register live_change_form_gate_v2. Run ONCE, before Opus.

WHY A SECOND BANK. v1 measured 17/18 and is immutable historical evidence: it
recorded the defect, and re-asking its eighteen questions after the repair would
measure the repair against the sample that motivated it. v2 is fresh paraphrases
of the same four analytical forms, so the repair is measured against wording it
was not derived from.

WHAT CHANGED IN THE QUESTION SET, DELIBERATELY. v1's single `metric_delta` miss
was a weighted-average-LTV question. v2 carries SIX metric deltas across FIVE
distinct governed quantities — outstanding balance, live loan count, arrears
balance, weighted-average current LTV and weighted-average indexed LTV — so a
pass cannot come from having learned one metric. Nothing in the model contract
mentions LTV, a metric name, or any of these phrasings.

TEMPORAL IS RECORDED AND NOT GATED, and this is the one scored difference from
v1. v1 gated on the period pair and two readings failed it by anchoring at
`current` and letting the capability own the comparison — a reading the
deterministic owners do NOT currently honour. That is an independent
temporal-contract defect, reported separately and deliberately NOT repaired in
this sprint. Gating v2 on a contract known to be defective would mix the two and
make a temporal failure look like a change_form failure, so the temporal reading
is recorded in full, reported, and excluded from the verdict.

THE COMPILE OUTCOME IS NOW EVIDENCE. With the completeness gate in place, an
intent that omits `change_form` on a change request CLARIFIES instead of
planning. So each case's compile outcome is carried: a PLAN proves the reading was
complete, and a CLARIFY naming `change_form` proves the gate caught an incomplete
one rather than letting it through.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "change_form_v2_bank_manifest.json"
HASH = HERE / "change_form_v2_bank_manifest.sha256"

BASELINE_SHA = "665176864d2f5dbceedc277e351c5d2cd978777f"
MODEL = "claude-opus-5"
BANK_ID = "live_change_form_gate_v2"

BALANCE = {"family": "funded_or_outstanding_balance",
           "concepts": ["current_outstanding_balance",
                        "current_principal_balance",
                        "funded_balance_movement"],
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

#: Recorded, NOT gated. See the module docstring.
ADJACENT_PAIR = ["relative_pair", "previous_reporting_period", "explicit_period"]

CASES = [
    # ------------------ MATERIAL SUMMARY (4) ----------------------------- #
    dict(id="V01", change_form="material_summary", measure=None, scope=None,
         question="Walk me through what has moved in the funded book since the "
                  "last report."),
    dict(id="V02", change_form="material_summary", measure=None, scope=None,
         question="Is there anything I should be aware of in how the portfolio "
                  "has shifted this period?"),
    dict(id="V03", change_form="material_summary", measure=None,
         scope="acquired",
         question="Summarise the notable movements in the acquired funded book "
                  "against the prior reporting date."),
    dict(id="V04", change_form="material_summary", measure=None, scope=None,
         question="What stands out as having changed across the book between "
                  "the two most recent reporting dates?"),

    # ------------------ METRIC DELTA (6, five distinct metrics) ---------- #
    dict(id="V05", change_form="metric_delta", measure=BALANCE, scope=None,
         question="By how much did outstanding balance shift between the last "
                  "two reporting dates?"),
    dict(id="V06", change_form="metric_delta", measure=LOAN_COUNT, scope=None,
         question="What is the change in the number of live funded loans since "
                  "the prior report?"),
    dict(id="V07", change_form="metric_delta", measure=ARREARS, scope=None,
         question="How much did arrears balance move over the latest reporting "
                  "period?"),
    dict(id="V08", change_form="metric_delta", measure=WA_LTV, scope=None,
         question="Did weighted average LTV rise or fall against the previous "
                  "reporting date?"),
    dict(id="V09", change_form="metric_delta", measure=INDEXED_LTV, scope=None,
         question="How has weighted average indexed LTV changed since the last "
                  "report?"),
    dict(id="V10", change_form="metric_delta", measure=BALANCE, scope="direct",
         question="How much did funded balance move in the direct book over the "
                  "latest period?"),

    # ------------------ ATTRIBUTION (4) --------------------------------- #
    dict(id="V11", change_form="attribution", measure=BALANCE, scope=None,
         question="What accounts for the movement in funded balance between the "
                  "last two reporting dates?"),
    dict(id="V12", change_form="attribution", measure=BALANCE, scope=None,
         question="Break the change in funded balance into its drivers for the "
                  "latest period."),
    dict(id="V13", change_form="attribution", measure=BALANCE, scope=None,
         question="Which components explain why the funded balance ended where "
                  "it did this period?"),
    dict(id="V14", change_form="attribution", measure=BALANCE, scope="acquired",
         question="Give me the drivers behind the acquired book's funded balance "
                  "movement since the prior report."),

    # ------------------ LEVEL COMPARISON (4) ---------------------------- #
    dict(id="V15", change_form="level_comparison", measure=BALANCE, scope=None,
         question="Put outstanding balance at the latest reporting date "
                  "alongside the previous one."),
    dict(id="V16", change_form="level_comparison", measure=WA_LTV, scope=None,
         question="What were the weighted average LTV figures at the current "
                  "and the prior reporting dates?"),
    dict(id="V17", change_form="level_comparison", measure=LOAN_COUNT,
         scope=None,
         question="Show funded loan counts for the two most recent reporting "
                  "dates side by side."),
    dict(id="V18", change_form="level_comparison", measure=BALANCE,
         scope="direct",
         question="Set the direct book's funded balance at the current "
                  "reporting date against the prior one."),
]


def build() -> dict:
    for case in CASES:
        case["filters"] = []
        case["dimensions"] = []
        case["temporal_forms"] = list(ADJACENT_PAIR)
    return {
        "bank_id": BANK_ID,
        "model": MODEL,
        "baseline_sha": BASELINE_SHA,
        "supersedes": "live_change_form_gate_v1 (immutable, 17/18)",
        "what_this_is":
            "NATURAL LANGUAGE -> OPUS -> CandidateIntent, re-measured on fresh "
            "paraphrases after the change_form completeness repair. It does NOT "
            "test deterministic execution and computes no answer.",
        "boundary":
            "AnthropicInterpreterClient directly. No /mi/query, no MI_BEARER, no "
            "App Service, no plan_serving_canary, no MI_AGENT_PLAN_SERVE, not "
            "the 135 benchmark CLI. Nothing runs against a book.",
        "authorised_live_calls": 18,
        "calls_per_case": 1,
        "retries": 0,
        "registry_is_visible_to_the_model": False,
        "metric_spread":
            "Six metric deltas over five distinct governed quantities "
            "(outstanding balance, loan count, arrears balance, current LTV, "
            "indexed LTV) so a pass cannot come from one learned metric. v1's "
            "only metric_delta miss was an LTV question.",
        "not_pinned": [
            "capability", "operation",
            "the statistic on a measure (recorded, reported, not a gate)",
            "time.grain",
            "THE TEMPORAL FORM — recorded and reported, explicitly NOT gated: "
            "the deterministic owners do not currently honour a `current` "
            "anchor for material_summary or attribution, which is an "
            "independent temporal-contract defect reported separately and not "
            "repaired in this sprint. Gating on it would make a temporal "
            "failure read as a change_form failure.",
        ],
        "primary_gate": "change_form == expected, 18/18",
        "secondary_gates": {
            "material_summary_specific_measure_invented": "0 of 4",
            "explicit_measure_preserved": "14 of 14 (V05-V18)",
            "scope_preserved": "4 of 4 (V03, V10, V14, V18)",
            "unnecessary_clarifications": 0,
            "invented_filters": 0,
            "invented_dimensions": 0,
        },
        "compile_outcome_is_evidence":
            "With the completeness gate in place, an intent omitting "
            "change_form on a change request CLARIFIES rather than planning. "
            "Each case carries its compile outcome: PLAN proves the reading was "
            "complete; CLARIFY naming change_form proves the gate caught an "
            "incomplete reading instead of letting it through.",
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

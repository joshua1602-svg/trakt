#!/usr/bin/env python3
"""Pre-register the live change_form interpretation bank. Run ONCE, before Opus.

This is the ONLY file that writes `change_form_bank_manifest.json` and its
sha256. The runner verifies the hash and never writes either, so expectations
cannot be edited after a result is seen — the failure mode that makes a
favourable measurement worthless.

WHAT IS PINNED. The eighteen questions, the expected `change_form`, the expected
measure FAMILY where the question names one, the expected Direct/Acquired scope
where the question names one, and the expected temporal relationship. Plus the
model id and the product baseline SHA the measurement stands against.

WHY A MEASURE *FAMILY* AND NOT ONE STRING. The governed vocabulary resolves
aliases: `balance`, `outstanding_balance` and `current_outstanding` all resolve
to `current_outstanding_balance`, and "how many loans" is `loan` + `count`
rather than a `loan_count` concept, which does not exist. Scoring a canonical
concept_id after resolution measures whether the reading is usable by the
governed layer; scoring a literal string would measure whether the model typed
what the author typed. The existing slice 3 harnesses make the same choice and
say so.

WHAT IS DELIBERATELY NOT PINNED. `capability` and `operation`. Sprint A made the
compiler authoritative over execution ownership precisely so the model does not
have to choose it, and `change_form` is the slot under measurement. They are
RECORDED for every case and scored as secondary observations only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "change_form_bank_manifest.json"
HASH = HERE / "change_form_bank_manifest.sha256"

BASELINE_SHA = "665176864d2f5dbceedc277e351c5d2cd978777f"
MODEL = "claude-opus-5"
BANK_ID = "live_change_form_gate_v1"

# --------------------------------------------------------------------------- #
# measure families, as canonical concept_ids after vocabulary resolution
# --------------------------------------------------------------------------- #

#: "funded balance" / "outstanding balance". `funded_balance_movement` is the
#: governed movement concept for the same quantity and is accepted: a question
#: about the CHANGE in funded balance may name either the level or the movement.
BALANCE = {
    "family": "funded_or_outstanding_balance",
    "concepts": ["current_outstanding_balance", "current_principal_balance",
                 "funded_balance_movement"],
    "statistics": [],
}

#: "weighted average LTV". The registry's default statistic for LTV already IS
#: `weighted_average` with an exposure weight, so the statistic is recorded and
#: reported but the CONCEPT is what the measure check turns on.
WA_LTV = {
    "family": "weighted_average_ltv",
    "concepts": ["current_loan_to_value", "indexed_loan_to_value"],
    "statistics": ["weighted_average", "average"],
}

#: "number of funded loans" / "funded loan count". There is no `loan_count`
#: concept: the governed expression is the `loan` concept counted, which is what
#: the signed-off 135 bank uses (`measures: [loan]`, `statistic: [count]`).
LOAN_COUNT = {
    "family": "loan_count",
    "concepts": ["loan"],
    "statistics": ["count", "count_distinct"],
}

#: Every question in this bank asks about a change between two reporting states.
#: These are the time forms that PRESERVE that relationship. `current` alone,
#: `series`, `range` and `forward_looking` do not, and are recorded as a
#: temporal failure rather than quietly accepted.
ADJACENT_PAIR = ["relative_pair", "previous_reporting_period", "explicit_period"]

CASES = [
    # ---------------- MATERIAL SUMMARY (5) ------------------------------- #
    dict(id="CF01", change_form="material_summary", measure=None, scope=None,
         question="What are the main changes in the funded portfolio since the "
                  "previous reporting period?"),
    dict(id="CF02", change_form="material_summary", measure=None, scope=None,
         question="Give me the material changes in the book over the latest "
                  "month."),
    dict(id="CF03", change_form="material_summary", measure=None, scope=None,
         question="What should I know about how the portfolio has changed since "
                  "last month?"),
    dict(id="CF04", change_form="material_summary", measure=None, scope=None,
         question="Have there been any important changes in the funded book "
                  "since the last reporting date?"),
    dict(id="CF05", change_form="material_summary", measure=None, scope="direct",
         question="What materially changed in the direct funded portfolio over "
                  "the latest reporting period?"),

    # ---------------- METRIC DELTA (5) ----------------------------------- #
    dict(id="CF06", change_form="metric_delta", measure=BALANCE, scope=None,
         question="How much did funded balance change since last month?"),
    dict(id="CF07", change_form="metric_delta", measure=WA_LTV, scope=None,
         question="How did weighted average LTV move over the latest reporting "
                  "period?"),
    dict(id="CF08", change_form="metric_delta", measure=LOAN_COUNT, scope=None,
         question="How much has the number of funded loans changed since the "
                  "previous report?"),
    dict(id="CF09", change_form="metric_delta", measure=BALANCE, scope=None,
         question="What was the change in outstanding balance between this "
                  "month and last month?"),
    dict(id="CF10", change_form="metric_delta", measure=BALANCE,
         scope="acquired",
         question="How did funded balance change in the acquired portfolio over "
                  "the latest month?"),

    # ---------------- ATTRIBUTION (4) ------------------------------------ #
    # `target` in the gate brief means the measure being DECOMPOSED. It is not
    # `CandidateIntent.target`, which is a threshold/goal slot. So it is scored
    # as the measure, which is why EXPLICIT_MEASURE_PRESERVED spans CF06-CF18.
    dict(id="CF11", change_form="attribution", measure=BALANCE, scope=None,
         question="What drove the change in funded balance this month?"),
    dict(id="CF12", change_form="attribution", measure=BALANCE, scope=None,
         question="Explain the main contributors to the movement in funded "
                  "balance since the previous reporting period."),
    dict(id="CF13", change_form="attribution", measure=BALANCE, scope=None,
         question="Bridge the funded balance movement from the previous month "
                  "to the current month."),
    dict(id="CF14", change_form="attribution", measure=BALANCE, scope="direct",
         question="Why did the direct funded book balance move over the latest "
                  "reporting period?"),

    # ---------------- LEVEL COMPARISON (4) ------------------------------- #
    dict(id="CF15", change_form="level_comparison", measure=BALANCE, scope=None,
         question="Compare funded balance this month with the previous month."),
    dict(id="CF16", change_form="level_comparison", measure=WA_LTV, scope=None,
         question="What was weighted average LTV in the current reporting "
                  "period versus the previous one?"),
    dict(id="CF17", change_form="level_comparison", measure=LOAN_COUNT,
         scope=None,
         question="Show me the funded loan count at the current reporting date "
                  "and at the prior reporting date."),
    dict(id="CF18", change_form="level_comparison", measure=BALANCE,
         scope="acquired",
         question="Compare the acquired book's funded balance at the latest "
                  "reporting date with the previous reporting date."),
]


def build() -> dict:
    for case in CASES:
        # Not one question in this bank states a predicate or asks for a
        # grouping, so the expectation is empty for all eighteen and ANY filter
        # or dimension is an invention. Pinned per case so the runner reads it
        # from the manifest rather than assuming it.
        case["filters"] = []
        case["dimensions"] = []
        case["temporal_forms"] = list(ADJACENT_PAIR)
    return {
        "bank_id": BANK_ID,
        "model": MODEL,
        "baseline_sha": BASELINE_SHA,
        "what_this_is":
            "NATURAL LANGUAGE -> OPUS -> CandidateIntent. Whether live Opus "
            "assigns the Sprint A first-class `change_form` to eighteen "
            "natural-language change questions. It does NOT test deterministic "
            "execution, and no answer is computed.",
        "boundary":
            "Calls the Anthropic API directly via AnthropicInterpreterClient. "
            "No /mi/query, no MI_BEARER, no Azure App Service, no "
            "plan_serving_canary, no MI_AGENT_PLAN_SERVE, and not the 135 "
            "benchmark CLI. Nothing is executed against a book.",
        "authorised_live_calls": 18,
        "calls_per_case": 1,
        "retries": 0,
        "registry_is_visible_to_the_model": False,
        "not_pinned": [
            "capability", "operation",
            "the statistic on a measure (recorded, reported, not a gate)",
            "time.grain (a question naming no grain may leave it absent)",
        ],
        "primary_gate": "change_form == expected, 18/18",
        "secondary_gates": {
            "material_summary_specific_measure_invented": "0 of 5",
            "explicit_measure_preserved": "13 of 13 (CF06-CF18)",
            "scope_preserved": "4 of 4 (CF05, CF10, CF14, CF18)",
            "unnecessary_clarifications": 0,
            "invented_filters": 0,
            "invented_dimensions": 0,
            "silent_semantic_drops": 0,
        },
        "note_on_scope_count":
            "The gate brief states SCOPE_PRESERVED = 3/3. The question set "
            "contains FOUR scope-bearing questions: CF05 direct, CF10 "
            "acquired, CF14 direct, CF18 acquired. All four are pinned and "
            "scored. The stricter reading is used deliberately; a gate is not "
            "relaxed to fit an arithmetic slip, and the discrepancy is "
            "reported rather than resolved by dropping a case.",
        "cases": CASES,
    }


def main() -> int:
    if MANIFEST.exists() and "--force" not in sys.argv:
        print(f"refusing to overwrite {MANIFEST.name}: it is pinned evidence. "
              f"--force only if no model call has been made against it.")
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

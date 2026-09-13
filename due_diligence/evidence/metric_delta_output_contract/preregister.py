#!/usr/bin/env python3
"""Pre-register metric_delta_output_contract_canary. Run ONCE, before any call.

WHAT THIS BANK IS FOR. Two repairs, and the confirmation that neither broke a
form already serving:

    1. a metric delta stated with a COMPARISON-shaped operation now reaches its
       owner, because the central change-form contract canonicalises the variant
       — proved live, not merely offline;
    2. a requested metric is ANSWERED, not merely executed.

WHY (2) NEEDED A BANK OF ITS OWN. The previous bank scored M03 a PASS. Its route,
owner, mode and requested field were all correct and its published answer did not
contain the movement its own receipt held. Every case here therefore pins an
EXPECTED USER-FACING OUTCOME alongside the owner, and the runner reads the served
answer for the figure the receipt records. A case whose owner ran perfectly and
whose reader learned nothing FAILS.

THE OPERATION IS PINNED AS A CHAIN, NOT A VALUE. The model may spell a metric
delta's action `compare` or `movement`; both are correct readings and the
contract collapses them. So each metric-delta case pins the CANONICAL operation
the plan must carry (`movement`) and accepts either spelling upstream. Pinning
the model's spelling would be pinning a coin toss; pinning the canonical form is
pinning the contract.

ONE DISPOSITION IS AN ACCEPT-SET, AND THAT IS DELIBERATE. N03's metric came back
`partially_available` on this book in the last live run, which is why it is here.
But availability is a fact about the data on the day, not about the code, and a
book that improves between runs would turn a correct `ANSWERED` into a false
FAIL. So N03 accepts either disposition and pins the invariant that actually
belongs to this sprint: whichever it is, the metric must be named and its figure
stated, and a `QUALIFIED` one must carry its caveat. Pinning a data condition as
if it were a contract is the v4/W06 authoring defect, and it is not repeated
here.

NO NUMBERS. Numeric truth is reconciled afterwards against the owner run
independently. A figure taken from a model's output and used to score that model
certifies nothing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
MANIFEST = HERE / "delta_contract_bank_manifest.json"
HASH = HERE / "delta_contract_bank_manifest.sha256"

#: THE MANIFEST IS NOT CALLED `output_...` FOR A REASON. `.gitignore` carries
#: `*out*.json`, which silently matched `output_contract_bank_manifest.json`:
#: the hash file would have been committed and the manifest it pins would not,
#: so a fresh CI checkout could not have verified the bank at all. Caught before
#: anything was spent against it. Any future manifest in this programme must be
#: checked against `git check-ignore` before it is hashed.
PRODUCT_BASELINE = "5d84a6a31ccf0f255c6c9170de97b01532fa35da"
MODEL = "claude-opus-5"
BANK_ID = "metric_delta_output_contract_canary"

#: Every bank whose wording this one must not reuse.
HISTORICAL = [
    ("live_change_form_gate", "change_form_bank_manifest"),
    ("live_change_form_gate_v2", "change_form_v2_bank_manifest"),
    ("live_change_form_gate_v3", "change_form_v3_bank_manifest"),
    ("live_change_form_gate_v4", "change_form_v4_bank_manifest"),
    ("change_intelligence_serving_canary", "serving_canary_bank_manifest"),
    ("metric_delta_confirmation_canary", "metric_delta_canary_bank_manifest"),
]

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

PAIR_FORMS = ["relative_pair", "previous_reporting_period", "explicit_period"]
OWNS_DEFAULT = PAIR_FORMS + ["current", "ABSENT"]

#: Both spellings of the one action. The contract collapses them; which one the
#: model emits is not this bank's business.
DELTA_OPERATIONS = ["movement", "compare"]

OWNER_METRIC_DELTA = {
    "plan_adapter": "mi_agent.plan_metric_delta",
    "calculation_owner":
        "mi_agent.period_change.workflow.run_period_change_analysis",
    "mode": "requested_metric",
    "composition_owner": None,
    "serving_provenance": "GOVERNED_PLAN",
    "route": "governed_plan_metric_delta",
    "canonical_operation": "movement",
    "candidate_operation_accept": list(DELTA_OPERATIONS)}
OWNER_MATERIAL_SUMMARY = {
    "plan_adapter": "mi_agent.plan_material_summary",
    "calculation_owner":
        "mi_agent.period_change.workflow.run_period_change_analysis",
    "mode": "portfolio_overview",
    "composition_owner": "mi_agent_api.insight_funded.compose",
    "serving_provenance": "GOVERNED_PLAN",
    "route": "governed_plan_material_summary",
    "canonical_operation": "summary",
    "candidate_operation_accept": ["summary", "movement", "compare"]}

#: What the READER must end up with. This is the half the previous bank lacked.
STATES_THE_METRIC = (
    "the served answer must name the requested metric and state the movement "
    "the receipt records, formatted by the estate's own formatter")
OVERVIEW_NARRATIVE = (
    "the governed overview narrative; no requested-metric block, because no "
    "metric was named")

CASES = [
    # -- 1. the M01 repair, live: a comparison-shaped metric delta ---------- #
    dict(id="N01", change_form="metric_delta", measure=BALANCE, scope=None,
         temporal_intent="EXPLICIT_PAIR", temporal_accept=list(PAIR_FORMS),
         filters_permitted=[], outcome="ANSWER", owner=OWNER_METRIC_DELTA,
         expected_disposition=["ANSWERED"],
         expected_user_facing_outcome=STATES_THE_METRIC,
         why="M01 was read perfectly, compiled to a plan, and served nothing "
             "because `compare` was not a governed variant of this form. This "
             "is the same shape of request in words M01 did not use.",
         question="Set the funded book's outstanding balance at the newest "
                  "reporting date against the one before it and tell me how "
                  "far it has moved."),

    # -- 2. a fully available weighted average ------------------------------ #
    dict(id="N02", change_form="metric_delta", measure=WA_LTV, scope=None,
         temporal_intent="EXPLICIT_PAIR", temporal_accept=list(PAIR_FORMS),
         filters_permitted=[], outcome="ANSWER", owner=OWNER_METRIC_DELTA,
         expected_disposition=["ANSWERED"],
         expected_user_facing_outcome=STATES_THE_METRIC,
         why="The control case for the output contract: an unqualified metric "
             "must be stated plainly, with no caveat invented for it.",
         question="Where does the funded book's weighted average loan-to-value "
                  "stand now compared with the prior reporting date?"),

    # -- 3. the availability policy, exercised ------------------------------ #
    dict(id="N03", change_form="metric_delta", measure=WA_RATE, scope=None,
         temporal_intent="EXPLICIT_PAIR", temporal_accept=list(PAIR_FORMS),
         filters_permitted=[], outcome="ANSWER", owner=OWNER_METRIC_DELTA,
         expected_disposition=["QUALIFIED", "ANSWERED"],
         expected_user_facing_outcome=(
             STATES_THE_METRIC + "; and where the disposition is QUALIFIED the "
             "answer must also carry the owner's exclusion counts"),
         why="This metric returned `partially_available` on this book in the "
             "last live run, which is why it is here. The disposition is an "
             "accept-set because availability is a fact about the data on the "
             "day; the invariant pinned is that the figure reaches the reader "
             "either way. M03 is the case this replaces.",
         question="Tell me where the funded book's weighted average interest "
                  "rate sat at each of the two most recent reporting dates and "
                  "what the change between them was."),

    # -- 4. a second comparison-shaped delta, different measure ------------- #
    dict(id="N04", change_form="metric_delta", measure=BALANCE, scope=None,
         temporal_intent="EXPLICIT_PAIR", temporal_accept=list(PAIR_FORMS),
         filters_permitted=[], outcome="ANSWER", owner=OWNER_METRIC_DELTA,
         expected_disposition=["ANSWERED"],
         expected_user_facing_outcome=STATES_THE_METRIC,
         why="A second, independent sample of the canonicalisation on a "
             "different measure and a different verb, so N01 passing is not "
             "one lucky reading.",
         question="Contrast the principal balance on the funded book between "
                  "the two latest reporting dates."),

    # -- 5. regression: the form whose answer must NOT have changed --------- #
    dict(id="N05", change_form="material_summary", measure=None, scope=None,
         temporal_intent="OWNER_DEFAULT", temporal_accept=list(OWNS_DEFAULT),
         filters_permitted=[], outcome="ANSWER",
         owner=OWNER_MATERIAL_SUMMARY,
         expected_disposition=[],
         expected_user_facing_outcome=OVERVIEW_NARRATIVE,
         why="The requested-metric lead was added to the one presenter both "
             "forms share. This proves it did not edit the overview a reader "
             "who named no metric still gets.",
         question="Which movements in the funded book stand out this period?"),
]


def _historical_questions():
    seen = {}
    for directory, manifest in HISTORICAL:
        path = EVIDENCE / directory / f"{manifest}.json"
        pinned = (EVIDENCE / directory / f"{manifest}.sha256").read_text().split()[0]
        body = path.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if digest != pinned:
            raise SystemExit(f"{directory} was edited after hashing")
        for case in json.loads(body).get("cases") or ():
            seen[str(case.get("question") or "").strip().lower()] = (
                f"{directory}:{case.get('id')}")
    return seen


def _normalised(text):
    return " ".join(str(text).strip().lower().split())


def build():
    history = _historical_questions()
    for case in CASES:
        key = _normalised(case["question"])
        if key in history:
            raise SystemExit(f"{case['id']} reuses wording from {history[key]}")
    return {
        "bank_id": BANK_ID,
        "model": MODEL,
        "product_baseline": PRODUCT_BASELINE,
        "authorised_live_calls": len(CASES),
        "calls_per_case": 1,
        "retries": 0,
        "registry_is_visible_to_the_model": False,
        "what_this_is":
            "A confirmation that a comparison-shaped metric delta now reaches "
            "its owner through the central change-form operation contract, and "
            "that a requested metric is ANSWERED rather than merely executed. "
            "Five fresh questions; no wording from any earlier bank is reused, "
            "verified programmatically against every historical manifest before "
            "this file was written.",
        "supersedes":
            "Nothing. metric_delta_confirmation_canary is SPENT, immutable and "
            "adjudicated (4 PASS, 1 FAIL on M01). This bank closes M01 and the "
            "M03 output-contract gap that bank's scorer could not see.",
        "closes": [
            "M01 — OPERATION_NOT_ADMITTED on a correctly read metric delta",
            "M03 — a requested metric computed, published in the table, and "
            "absent from the answer",
        ],
        "primary_gates": {
            "change_form": "5 of 5",
            "canonical_operation":
                "4 of 4 metric deltas must carry `movement` in the compiled "
                "plan, whichever spelling the model emitted",
            "route_and_calculation_owner":
                "5 of 5 — a correct reading that reaches no owner is a FAIL",
            "mode": "5 of 5",
            "requested_metric_answered":
                "4 of 4 — the served answer must name the requested metric and "
                "state the figure the receipt records. THIS IS THE GATE THE "
                "PREVIOUS BANK LACKED.",
            "outcome_type": "5 of 5 ANSWER",
        },
        "secondary_gates": {
            "requested_field_analysed":
                "4 of 4 — the field the plan named must appear in the owner's "
                "selected_measures AND in the served metric table",
            "disposition_recorded":
                "4 of 4 — every requested metric must carry a disposition in "
                "ANSWERED / QUALIFIED / GOVERNED_REFUSAL",
            "no_bridge_substitution":
                "0 — no requested-metric answer may consist of the balance "
                "bridge without its own metric",
            "overview_unchanged":
                "N05 must carry no requested-metric block at all",
            "invented_filters": 0,
            "invented_dimensions": 0,
            "unnecessary_clarifications": 0,
        },
        "not_pinned": [
            "capability",
            "WHICH spelling of the action the model emits — the contract "
            "collapses `compare` and `movement`, and pinning the model's choice "
            "would pin a coin toss",
            "any numeric value",
            "time.grain",
            "the statistic actually applied — the owner follows the registry's "
            "default_aggregation and the receipt records both sides",
            "N03's availability, which is a fact about the book on the day",
            "WHICH of the valid temporal routes N05 takes, since its form owns "
            "a default and all of them are correct readings",
        ],
        "deliberate_omissions": {
            "no_unavailable_metric_case":
                "The anti-substitution rule for a metric present at only one "
                "date is proved offline against a fixture built for it. No "
                "field is KNOWN to be single-dated on this book, and pinning a "
                "live case on a guess about the data would be pinning a "
                "condition rather than a contract.",
            "no_s09_retest":
                "Unchanged: the acquired portfolio is absent from one of the "
                "two selected snapshots. Data history, not a defect, and no new "
                "factual evidence has appeared.",
            "no_attribution_case":
                "Attribution passed live in the previous bank and nothing in "
                "this sprint touches its adapter, its owner or its envelope. "
                "material_summary is the regression case because it SHARES the "
                "presenter this sprint changed; attribution does not.",
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

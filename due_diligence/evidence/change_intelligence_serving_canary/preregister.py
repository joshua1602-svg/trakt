#!/usr/bin/env python3
"""Pre-register change_intelligence_serving_canary. Run ONCE, before any call.

WHAT THIS BANK IS, AND HOW IT DIFFERS FROM v1-v4. Those four measured
INTERPRETATION only: natural language -> Opus -> `CandidateIntent`, with no MI
API, no book and no answer. This one is END TO END — `POST /api/mi/query`, a
governed plan, a deterministic owner, a real book, a rendered envelope — so for
the first time in this programme a pinned expectation can be wrong because
SERVING is wrong rather than because a reading was.

WHAT IT MEASURES, AND WHAT IT DELIBERATELY DOES NOT. It measures, per question:
the analytical FORM the compiler derived, the measure semantics, the scope, the
temporal route, WHICH DETERMINISTIC OWNER executed, and whether the outcome was
an ANSWER or a governed REFUSAL. It pins NO NUMBER. Numeric truth is reconciled
AFTERWARDS by running the named deterministic owner independently over the same
governed snapshots and requiring every published figure to match — the same
owner-parity rule the offline controls apply, moved to live inputs. A bank that
pinned a figure from a model's output would be certifying the model against
itself.

THE THREE AUTHORING DEFECTS OF THE EARLIER BANKS, AVOIDED RATHER THAN REPEATED.
Each was mine, and each surfaced while correcting the one before it.

  1. v3/V06 — a RESTRICTING ADJECTIVE is not an invented filter ("live funded
     loans"). So `filters_permitted` is pinned per case here, as in v4.
  2. v3/V13 — `bridge_component` is a governed `funded_bridge` concept and a
     legitimate attribution target. So `ATTRIBUTION_TARGET` admits it.
  3. v4/W06 — a governed concept may be PERMITTED as a filter target and still
     publish no value vocabulary, so the restriction is not expressible and the
     only correct outcome is to clarify. Pinning zero clarifications while
     including such a question is a bank defect. So NO QUESTION HERE RESTRICTS ON
     ARREARS, or on any other concept whose governed value list is empty, and the
     one question that cannot be satisfied is pinned as a REFUSAL rather than as
     an answer.

WHAT IS FROZEN. Every semantic contract: `change_form`, the temporal presence and
default rules, `interpretation_v2`, the Insight Engine and the materiality
policy. This sprint connected two arrows and this bank measures those arrows. A
miss here is a serving defect or a bank defect, and in neither case is it licence
to reopen a semantic that v3 and v4 measured at 18/18 and 17/17.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "serving_canary_bank_manifest.json"
HASH = HERE / "serving_canary_bank_manifest.sha256"

#: The APPROVED PRIOR PRODUCT STATE this sprint started from — Sprint A, Sprint B,
#: change_form completeness, the current-anchor contract and the temporal presence
#: repair, with change_form measured live at 18/18 and the temporal contract at
#: 17/17.
PRODUCT_BASELINE = "f22d0e467bec66daa2cd3177d7e5c260d847c4e7"

#: WHICH COMMIT THE CANARY MAY RUN AGAINST, stated as a rule rather than as a
#: literal, for the reason the v1 gate found the hard way: a bank cannot name its
#: own commit, because committing the bank changes it. The runner resolves this
#: against the repository and the deployed `/health` build, and refuses with zero
#: questions asked on any mismatch.
PRODUCT_HEAD = "88cf5fde759c29f1cff29b9adbf003a2d6ebd530"

DEPLOYMENT_SHA_RULE = {
    "must_equal": PRODUCT_HEAD,
    "what_that_is":
        "the commit every offline gate in this sprint passed on — the product "
        "change itself, with no evidence file and no CI plumbing in it",
    "asserted_by":
        "GET /api/health -> build.commit, compared before any question is asked",
    "tolerated_difference":
        "none in product code. The approved head must be an ancestor of the "
        "deployed build, and the only files differing between them may be this "
        "evidence directory and the workflow that runs it — asserted by diffing, "
        "not claimed.",
    "on_mismatch": "STOP. Zero live calls, zero spend.",
}
MODEL = "claude-opus-5"
BANK_ID = "change_intelligence_serving_canary"

# --------------------------------------------------------------------------- #
# measure families — canonical concept_ids after vocabulary resolution
# --------------------------------------------------------------------------- #
BALANCE = {"family": "funded_or_outstanding_balance",
           "concepts": ["current_outstanding_balance",
                        "current_principal_balance", "funded_balance_movement"],
           "statistics": []}
WA_LTV = {"family": "weighted_average_current_ltv",
          "concepts": ["current_loan_to_value"],
          "statistics": ["weighted_average", "average"]}
LOAN_COUNT = {"family": "loan_count", "concepts": ["loan"],
              "statistics": ["count", "count_distinct"]}

#: Defect 2, carried forward: an attribution may name the quantity decomposed OR
#: the governed component concept the decomposition reports.
ATTRIBUTION_TARGET = {
    "family": "funded_balance_or_its_governed_components",
    "concepts": ["current_outstanding_balance", "current_principal_balance",
                 "funded_balance_movement", "bridge_component"],
    "statistics": []}

# --------------------------------------------------------------------------- #
# temporal routes — unchanged from v4, because the contract is unchanged
# --------------------------------------------------------------------------- #
PAIR_FORMS = ["relative_pair", "previous_reporting_period", "explicit_period"]
ABSENT = "ABSENT"
OWNS_DEFAULT = PAIR_FORMS + ["current", ABSENT]
PAIR_REQUIRED = list(PAIR_FORMS)

TEMPORAL_ACCEPT_BY_FORM = {
    "material_summary": OWNS_DEFAULT,
    "attribution": OWNS_DEFAULT,
    "metric_delta": PAIR_REQUIRED,
    "level_comparison": PAIR_REQUIRED,
}

# --------------------------------------------------------------------------- #
# execution owners — what this sprint connected, and what it left alone
# --------------------------------------------------------------------------- #
#: THE TWO ARROWS THIS SPRINT ADDED. A served answer for these forms must carry
#: `metadata.governedPlan.executed.calculation_owner` naming the owner below, and
#: `metadata.route` naming the governed change-form route.
OWNER_MATERIAL_SUMMARY = {
    "plan_adapter": "mi_agent.plan_material_summary",
    "calculation_owner":
        "mi_agent.period_change.workflow.run_period_change_analysis",
    "composition_owner": "mi_agent_api.insight_funded.compose",
    "serving_provenance": "GOVERNED_PLAN",
    "route": "governed_plan_material_summary"}
OWNER_ATTRIBUTION = {
    "plan_adapter": "mi_agent.plan_attribution",
    "calculation_owner": "mi_agent.period_change.bridge.balance_bridge",
    "composition_owner": None,
    "serving_provenance": "GOVERNED_PLAN",
    "route": "governed_plan_attribution"}

#: THE TWO FORMS THIS SPRINT DID NOT TOUCH. Their serving provenance is NOT
#: pinned, and that is deliberate: this sprint established nothing about it, so
#: pinning it would be pinning an unmeasured fact. What IS pinned is the
#: negative — neither may arrive through a change-form route that does not claim
#: it — plus the form, the measure and the temporal route, exactly as v4 pinned
#: them.
OWNER_METRIC_DELTA = {
    "plan_adapter": None,
    "calculation_owner":
        "mi_agent.period_change.workflow.run_period_change_analysis",
    "composition_owner": None,
    "serving_provenance": "NOT_PINNED_UNCHANGED_BY_THIS_SPRINT",
    "route_must_not_be": ["governed_plan_material_summary",
                          "governed_plan_attribution"]}
OWNER_LEVEL_COMPARISON = {
    "plan_adapter": "mi_agent.plan_temporal_runtime (pre-existing)",
    "calculation_owner": "mi_agent.mi_query_executor.execute_mi_query",
    "composition_owner": None,
    "serving_provenance": "NOT_PINNED_UNCHANGED_BY_THIS_SPRINT",
    "route_must_not_be": ["governed_plan_material_summary",
                          "governed_plan_attribution"]}

CASES = [
    # ---------------- MATERIAL SUMMARY (2) ------------------------------- #
    dict(id="S01", change_form="material_summary", measure=None, scope=None,
         temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         outcome="ANSWER", owner=OWNER_MATERIAL_SUMMARY,
         question="Taking the two most recent reporting dates, where did the "
                  "funded portfolio shift significantly?"),
    dict(id="S02", change_form="material_summary", measure=None, scope=None,
         temporal_intent="OWNER_DEFAULT", filters_permitted=[],
         outcome="ANSWER", owner=OWNER_MATERIAL_SUMMARY,
         question="Tell me which changes in the portfolio are worth my "
                  "attention."),

    # ---------------- METRIC DELTA (2) ----------------------------------- #
    dict(id="S03", change_form="metric_delta", measure=BALANCE, scope=None,
         temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         outcome="ANSWER", owner=OWNER_METRIC_DELTA,
         question="By how much has outstanding balance moved from the preceding "
                  "reporting date to the latest one?"),
    dict(id="S04", change_form="metric_delta", measure=WA_LTV, scope=None,
         temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         outcome="ANSWER", owner=OWNER_METRIC_DELTA,
         question="Quantify the move in weighted average LTV between this "
                  "reporting date and the last one."),

    # ---------------- ATTRIBUTION (2) ------------------------------------ #
    dict(id="S05", change_form="attribution", measure=ATTRIBUTION_TARGET,
         scope=None, temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         outcome="ANSWER", owner=OWNER_ATTRIBUTION,
         question="Which parts of the book account for the funded balance "
                  "movement across the two most recent reporting dates?"),
    dict(id="S06", change_form="attribution", measure=ATTRIBUTION_TARGET,
         scope=None, temporal_intent="OWNER_DEFAULT", filters_permitted=[],
         outcome="ANSWER", owner=OWNER_ATTRIBUTION,
         question="Explain what sits behind the change in funded balance."),

    # ---------------- LEVEL COMPARISON (2) ------------------------------- #
    dict(id="S07", change_form="level_comparison", measure=BALANCE, scope=None,
         temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         outcome="ANSWER", owner=OWNER_LEVEL_COMPARISON,
         question="I want outstanding balance as two figures: one for the "
                  "current reporting date and one for the date before it."),
    dict(id="S08", change_form="level_comparison", measure=LOAN_COUNT,
         scope=None, temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         outcome="ANSWER", owner=OWNER_LEVEL_COMPARISON,
         question="How many funded loans were there at this reporting date, "
                  "and how many at the one before it?"),

    # ---------------- SCOPED CHANGE QUESTION (1) -------------------------- #
    # THE SCOPE MUST SURVIVE TO THE NUMBERS, not only to the label. The gate
    # requires `executed_scope.portfolio_ids` to be the registry's list for the
    # acquired role and non-empty, because the measured historical defect was a
    # whole-book movement published under the word "Acquired" with a receipt that
    # declared the scope it had not applied.
    dict(id="S09", change_form="material_summary", measure=None,
         scope="acquired", temporal_intent="EXPLICIT_PAIR",
         filters_permitted=[], outcome="ANSWER", owner=OWNER_MATERIAL_SUMMARY,
         scope_must_narrow=True,
         question="In the acquired book specifically, what changed between "
                  "this reporting date and the last one?"),

    # ---------------- GOVERNED NEGATIVE CONTROL (1) ----------------------- #
    # A PORTFOLIO THE GOVERNED SOURCE REGISTRY DOES NOT HOLD. The only correct
    # outcome is to decline: answering it over the whole book, under the name the
    # reader used, is the failure the scope axis exists to prevent and is
    # indistinguishable from a correct answer once rendered. Either a governed
    # REFUSAL or a CLARIFICATION satisfies this case; an ANSWER fails it, and an
    # answer whose scope silently widened fails it twice.
    dict(id="S10", change_form=None, measure=None, scope="unresolvable",
         temporal_intent="EXPLICIT_PAIR", filters_permitted=[],
         outcome="REFUSE", owner=None, must_not_answer=True,
         question="How did the Meridian Trust portfolio change since the "
                  "preceding reporting date?"),
]


def build() -> dict:
    for case in CASES:
        case["dimensions"] = []
        case["temporal_accept"] = (
            list(TEMPORAL_ACCEPT_BY_FORM[case["change_form"]])
            if case["change_form"] else [])
    return {
        "bank_id": BANK_ID,
        "model": MODEL,
        "product_baseline": PRODUCT_BASELINE,
        "product_head": PRODUCT_HEAD,
        "deployment_sha_rule": DEPLOYMENT_SHA_RULE,
        "supersedes":
            "Nothing. v1, v2, v3 and v4 remain immutable INTERPRETATION banks; "
            "v2 remains unrun. This is the first END-TO-END bank in the "
            "programme and replaces none of them.",
        "what_this_is":
            "NATURAL LANGUAGE -> POST /api/mi/query -> interpretation_v2 -> "
            "GovernedQueryPlan -> deterministic owner -> rendered envelope. Ten "
            "fresh questions; none of v1's, v3's or v4's wording is reused, "
            "verified programmatically by the runner before any call.",
        "boundary":
            "The deployed MI API at the pinned product SHA, with "
            "MI_AGENT_PLAN_SERVE=canary and the canary principal allow-listed. "
            "It does NOT run the 135 benchmark, does not touch the historical "
            "banks, and changes no configuration itself.",
        "authorised_live_calls": 10,
        "calls_per_case": 1,
        "retries": 0,
        "registry_is_visible_to_the_model": False,
        "numbers_are_not_pinned": {
            "why":
                "A figure taken from a model's output and then used to score the "
                "same model certifies nothing. No numeric answer is pinned.",
            "how_numeric_truth_is_established":
                "AFTER each live call, the deterministic owner named in the "
                "case's `owner` block is invoked independently over the same "
                "governed snapshots, with the period request the served receipt "
                "records, and every published figure — metric movements, "
                "composition shifts, balance-bridge components, finding metrics "
                "— must match exactly. This is the offline OWNER PARITY GATE "
                "applied to live inputs.",
            "required": {"numerical_differences": 0, "semantic_differences": 0,
                         "scope_differences": 0},
        },
        "temporal_routes": {
            "EXPLICIT_PAIR": "the wording names two reporting states",
            "EXPLICIT_ANCHOR": "the wording names only the current endpoint",
            "OWNER_DEFAULT":
                "the wording names no temporal relation, and the form owns the "
                "comparison window; absent, anchor and pair are all valid",
            "scored_from":
                "the receipt's own `interpreted_time_present` / "
                "`temporal_default_applied`, never from wording",
        },
        "temporal_accept_by_form": TEMPORAL_ACCEPT_BY_FORM,
        "primary_gates": {
            "change_form": "9 of 9 (S01-S09); S10 states no form",
            "execution_owner":
                "5 of 5 — S01, S02, S05, S06 and S09 must record the "
                "calculation_owner their case pins, and S03, S04, S07 and S08 "
                "must NOT arrive through a change-form route",
            "outcome_type": "10 of 10 ANSWER / REFUSE as pinned",
            "owner_parity": "0 numerical, 0 semantic, 0 scope differences",
        },
        "secondary_gates": {
            "explicit_measure_preserved":
                "6 of 6 (S03, S04, S05, S06, S07, S08)",
            "material_summary_measure_invented": "0 of 3 (S01, S02, S09)",
            "scope_preserved_and_narrowed": "1 of 1 (S09)",
            "unnecessary_clarifications": 0,
            "invented_filters": "0 — a filter outside filters_permitted",
            "invented_dimensions": 0,
            "legacy_envelope_never_damaged":
                "every case's response is a complete answer or a governed "
                "decline; a canary failure must show as LEGACY_FALLBACK and "
                "never as an error",
        },
        "not_pinned": [
            "capability", "operation",
            "the statistic on a measure (recorded, reported, not a gate)",
            "time.grain",
            "any numeric value",
            "WHICH of the three valid temporal routes a form that owns a default "
            "takes",
            "the serving provenance of metric_delta and level_comparison — this "
            "sprint established nothing about it",
        ],
        "authoring_defects_avoided": {
            "v3_V06_restricting_adjective":
                "filters_permitted is pinned per case.",
            "v3_V13_attribution_measure":
                "ATTRIBUTION_TARGET admits bridge_component.",
            "v4_W06_no_governed_value_list":
                "no question restricts on arrears or on any concept with an "
                "empty governed value list; the one unsatisfiable question (S10) "
                "is pinned as a REFUSAL, not as an answer with zero "
                "clarifications.",
        },
        "stop_conditions": [
            "the deployed build does not satisfy deployment_sha_rule — STOP "
            "with zero questions asked",
            "MI_AGENT_PLAN_SERVE cannot be read from the authoritative source — "
            "STOP before enabling serving",
            "a provider quota, credit or authentication failure — STOP "
            "immediately and report the partial result",
            "any case is asked twice — the run is void",
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

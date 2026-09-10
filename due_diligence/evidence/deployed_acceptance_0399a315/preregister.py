#!/usr/bin/env python3
"""Phase 0 provenance and the Phase 3 acceptance bank, both offline.

Written before any deployment and before any shadow model call. Two artefacts:

    PROVENANCE     the blob identities and freeze proofs a deployment operator
                   must check against whatever they actually deploy, plus the
                   no-wildcard and default-off proofs, run rather than asserted.

    THE BANK       12 pre-registered requests, every question taken verbatim
                   from the successful local live-integration bank, with the
                   expected interpretation, eligibility, gate reason and shadow
                   disposition read out of the FROZEN run-8 sign-off.

WHAT CANNOT BE PRE-REGISTERED, AND WHY IT IS SAID HERE RATHER THAN DISCOVERED
LATER. Independent numerical truth is NOT available for the deployed portfolio:
this harness cannot see production data, so there is no oracle to compute. Every
eligible case is therefore pre-registered as `TRUTH_UNAVAILABLE` for its figure,
with the checks that DO adjudicate named explicitly — the interpretation against
the frozen expectation, the eligibility against the gate, the plan-to-spec facet
comparison, and the grouped grid reconciled against the execution receipt's own
row counts. A figure that merely matches the legacy answer is not a pass, because
the legacy answer is not the oracle.

Run: `python due_diligence/evidence/deployed_acceptance_0399a315/preregister.py`
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_runtime_adapter as adapter                  # noqa: E402
from mi_agent import plan_shadow_wiring as wiring                    # noqa: E402

CANDIDATE_SHA = "0399a315"
ACCEPTED_SLICE1_SHA = "6f42df67"
RUN8_SIGNOFF_SHA = "2b00172"
QUARANTINED_SHA = "3399d2ef"

FROZEN = (_REPO_ROOT / "mi_agent" / "interpretation_v2" / "evidence"
          / "run8_135_signoff_2b00172.json")
HERE = Path(__file__).resolve().parent

#: The files whose identity IS this candidate.
CANDIDATE_FILES = (
    "mi_agent/plan_runtime_adapter.py",
    "mi_agent/plan_shadow_wiring.py",
    "mi_agent/plan_shadow_evidence.py",
    "mi_agent_api/mi_service.py",
)

#: The bank. `(case_id, frozen question id, what it covers)`. Eight eligible,
#: four ineligible across four distinct gate reasons. Questions are verbatim from
#: the local live-integration bank; none was invented to fill a category.
BANK = (
    ("A01", "Q01A", ["count", "two_numeric_filters", "scalar"]),
    ("A02", "Q02A", ["sum_balance", "two_numeric_filters", "scalar"]),
    ("A03", "Q03A", ["count", "categorical_filter_governed_value", "numeric_filter"]),
    ("A04", "Q12A", ["sum_balance", "two_dimensions", "grouped"]),
    ("A05", "Q11A", ["sum_balance", "two_dimensions", "ticket_bucket",
                     "absent_from_the_local_oracle_book"]),
    ("A06", "Q13A", ["sum_balance", "two_dimensions", "interest_rate_bucket",
                     "absent_from_the_local_oracle_book"]),
    ("A07", "Q01B", ["count", "paraphrase_stability_pair_with_A01"]),
    ("A08", "Q02C", ["sum_balance", "paraphrase_stability_pair_with_A02"]),
    ("B01", "Q04A", ["ineligible", "explicit_direct_lens"]),
    ("B02", "Q14A", ["ineligible", "governed_geography_axis"]),
    ("B03", "Q19A", ["ineligible", "specialist_capability", "historical_temporal"]),
    ("B04", "Q12C", ["ineligible", "unsupported_shape_distribution",
                     "same_canonical_as_A04"]),
)

#: Fields the LOCAL oracle book does not carry. Whether the DEPLOYED funded frame
#: carries them is a dataset fact this harness cannot know, so both branches are
#: pre-registered with their adjudication.
LOCALLY_ABSENT_FIELDS = ("ticket_bucket", "interest_rate_bucket")

POLLING_POLICY = {
    "why": "the HTTP response returns before the daemon shadow thread finishes, "
           "so absence immediately after the response is not a failure",
    "interval_seconds": 2,
    "timeout_seconds": 60,
    "on_timeout": "ASYNC_EVIDENCE_LOST — makes acceptance INCONCLUSIVE for that "
                  "case, never a silent pass and never a semantic FAIL",
}

EVIDENCE_ASSOCIATION = {
    "how": "the evidence record's request.question plus request.client_id, "
           "within the polling window",
    "why_not_the_correlation_id": "the correlation id is generated server-side "
                                  "inside the shadow and is NOT returned on the "
                                  "HTTP response, so the caller cannot quote it",
    "unambiguous_for_this_bank": "all 12 questions are distinct strings, so "
                                 "question + client identifies one record",
    "limitation": "a bank that repeated a question would need a time-window "
                  "tiebreak; this one does not",
}


def git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=_REPO_ROOT, text=True,
                          capture_output=True).stdout.strip()


def provenance() -> Dict[str, Any]:
    for key in ("MI_AGENT_PLAN_SHADOW", "MI_AGENT_PLAN_SHADOW_CLIENTS"):
        os.environ.pop(key, None)

    wildcard_attempts = {}
    for attempt in ("*", "all", "any", "everyone", "*,acme", " ALL , Any ", "**"):
        os.environ["MI_AGENT_PLAN_SHADOW_CLIENTS"] = attempt
        wildcard_attempts[attempt] = {
            "allow_list": sorted(wiring.canary_clients()),
            "an_unlisted_client_is_shadowed":
                wiring.in_canary("some-unlisted-client"),
        }
    os.environ.pop("MI_AGENT_PLAN_SHADOW_CLIENTS", None)

    dirty = [line for line in git("status", "--porcelain").splitlines()
             if line and "deployed_acceptance_0399a315" not in line]

    return {
        "deploy_candidate_sha": git("rev-parse", "HEAD"),
        "deploy_candidate_sha_matches_brief":
            git("rev-parse", "HEAD").startswith(CANDIDATE_SHA),
        "tree_clean_outside_this_evidence_directory": dirty == [],
        "candidate_blob_identities": {
            path: git("rev-parse", f"HEAD:{path}") for path in CANDIDATE_FILES},
        "adapter_identical_to_accepted_slice1": (
            git("rev-parse", f"HEAD:mi_agent/plan_runtime_adapter.py") ==
            git("rev-parse",
                f"{ACCEPTED_SLICE1_SHA}:mi_agent/plan_runtime_adapter.py")),
        "interpretation_v2_code_frozen_since_run8": git(
            "diff", "--stat", RUN8_SIGNOFF_SHA, "HEAD", "--",
            "mi_agent/interpretation_v2/",
            ":!mi_agent/interpretation_v2/evidence") == "",
        "quarantined_branch_is_ancestor": subprocess.run(
            ("git", "merge-base", "--is-ancestor", QUARANTINED_SHA, "HEAD"),
            cwd=_REPO_ROOT, capture_output=True).returncode == 0,
        "shadow_mode_with_var_unset": adapter.shadow_mode(),
        "canary_with_var_unset": sorted(wiring.canary_clients()),
        "wildcard_attempts": wildcard_attempts,
        "no_wildcard_opens_the_canary": all(
            not v["an_unlisted_client_is_shadowed"]
            for v in wildcard_attempts.values()),
        "note_on_double_star":
            "'**' is not in FORBIDDEN_CANARY_TOKENS so it survives as a LITERAL "
            "client id, and matches nothing: the canary compares exact strings "
            "and does no globbing, so an unlisted client is still not shadowed",
        "gate_perimeter": {
            "capability": adapter.ELIGIBLE_CAPABILITY,
            "operations": sorted(adapter.ELIGIBLE_OPERATIONS),
            "period_forms": sorted(adapter.ELIGIBLE_PERIOD_FORMS),
            "population_lens": sorted(adapter.ELIGIBLE_POPULATION_LENS),
            "max_dimensions": adapter.MAX_DIMENSIONS,
        },
    }


def _semantics(plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not plan:
        return {}
    output = (tuple(plan.get("outputs") or ()) or ({},))[0]
    measure = (tuple(output.get("measures") or ()) or ({},))[0]
    geography = plan.get("geography") or {}
    return {
        "capability": plan.get("capability"),
        "operation": plan.get("operation"),
        "statistic": measure.get("statistic"),
        "measure_field": measure.get("canonical_field"),
        "weight_field": measure.get("weight_field"),
        "dimensions": [d["canonical_field"]
                       for d in (output.get("dimensions") or ())],
        "filters": sorted((f["canonical_field"],
                           str(f.get("comparator") or "eq"), f.get("value"))
                          for f in (tuple(plan.get("filters") or ())
                                    + tuple(output.get("filters") or ()))),
        "population_base": (plan.get("population") or {}).get("base"),
        "population_lens": (plan.get("population") or {}).get("lens"),
        "period_form": (plan.get("period") or {}).get("form"),
        "comparison_kind": plan.get("comparison_kind"),
        "geography_requested": bool(geography),
        "output_count": len(plan.get("outputs") or ()),
    }


def bank() -> List[Dict[str, Any]]:
    frozen = json.loads(FROZEN.read_text())
    by_id = {r["question_id"]: r for r in frozen["results"]}
    cases: List[Dict[str, Any]] = []

    for case_id, question_id, coverage in BANK:
        record = by_id[question_id]
        plan = record.get("plan")
        eligible, reason, detail = adapter.check_eligibility(plan)
        semantics = _semantics(plan)
        locally_absent = sorted(
            f for f in LOCALLY_ABSENT_FIELDS
            if f in semantics.get("dimensions", []))

        if not eligible:
            disposition = "INELIGIBLE"
            adjudication = (
                "the gate must return exactly the reason below; an ineligible "
                "control becoming eligible is an ELIGIBILITY_DEFECT, and the "
                "adapter must not have executed")
        elif locally_absent:
            disposition = ("EXECUTED if the deployed funded frame carries "
                           f"{locally_absent}, otherwise EXECUTION_ERROR naming "
                           f"the missing column")
            adjudication = (
                "both branches are acceptable and neither is a product defect: "
                "whether the deployed frame carries these governed fields is a "
                "DATASET fact. EXECUTION_ERROR here is TRUTH_UNAVAILABLE, not a "
                "DETERMINISTIC_EXECUTION_DEFECT")
        else:
            disposition = "EXECUTED"
            adjudication = (
                "the plan must reach the spec with no measure, filter or "
                "dimension dropped; a grouped result must reconcile against the "
                "receipt's own row counts")

        cases.append({
            "case_id": case_id,
            "frozen_question_id": question_id,
            "question": record["question"],
            "coverage": coverage,
            "expected_interpretation_disposition": record["outcome"],
            "expected_semantics": semantics,
            "expected_slice1_eligible": eligible,
            "expected_ineligible_reason": reason,
            "expected_shadow_disposition": disposition,
            "expected_frozen_plan_id": record.get("plan_id"),
            "independent_numerical_truth": None,
            "independent_numerical_truth_status":
                "TRUTH_UNAVAILABLE — this harness cannot see the deployed "
                "portfolio, so no oracle figure exists. The legacy answer is NOT "
                "the oracle; a figure matching it is a signal, not a pass",
            "adjudication_rule": adjudication,
            "provenance": f"{FROZEN.name} :: {question_id} (frozen sign-off; no "
                          f"deployed answer consulted)",
        })
    return cases


def main() -> int:
    cases = bank()
    manifest = {
        "phase": "Deployed slice 1 shadow acceptance — phase 0 provenance and "
                 "phase 3 pre-registration",
        "written_before_any_deployment_and_any_shadow_model_call": True,
        "deploy_candidate_sha": CANDIDATE_SHA,
        "frozen_source": str(FROZEN.relative_to(_REPO_ROOT)),
        "frozen_source_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "required_model": "claude-opus-5",
        "required_configuration": {
            "MI_AGENT_PLAN_SHADOW": "shadow",
            "MI_AGENT_PLAN_SHADOW_CLIENTS": "<exactly ONE authorised test client "
                                            "id; no wildcard>",
            "MI_AGENT_PLAN_SHADOW_EVIDENCE": "<a path outside any git checkout, "
                                             "verified writable AND readable "
                                             "before the bank runs>",
            "MI_AGENT_PLAN_SHADOW_MAX_IN_FLIGHT": "1 (the default)",
        },
        "polling_policy": POLLING_POLICY,
        "evidence_association": EVIDENCE_ASSOCIATION,
        "bank_size": len(cases),
        "eligible_expected": sum(1 for c in cases
                                 if c["expected_slice1_eligible"]),
        "ineligible_expected": sum(1 for c in cases
                                   if not c["expected_slice1_eligible"]),
        "distinct_ineligible_reasons": sorted(
            {c["expected_ineligible_reason"] for c in cases
             if not c["expected_slice1_eligible"]}),
        "provenance": provenance(),
        "cases": cases,
    }

    body = json.dumps(manifest, indent=2, default=str) + "\n"
    out = HERE / "acceptance_manifest.json"
    out.write_text(body)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    (HERE / "acceptance_manifest.sha256").write_text(f"{digest}  {out.name}\n")

    p = manifest["provenance"]
    print(f"DEPLOY_CANDIDATE_SHA              {p['deploy_candidate_sha']}")
    print(f"  matches the brief               {p['deploy_candidate_sha_matches_brief']}")
    print(f"TREE_CLEAN                        "
          f"{p['tree_clean_outside_this_evidence_directory']}")
    print(f"ADAPTER == ACCEPTED SLICE 1        "
          f"{p['adapter_identical_to_accepted_slice1']}")
    print(f"INTERPRETATION_V2 FROZEN           "
          f"{p['interpretation_v2_code_frozen_since_run8']}")
    print(f"QUARANTINED BRANCH PRESENT         "
          f"{p['quarantined_branch_is_ancestor']}")
    print(f"SHADOW DEFAULT                     {p['shadow_mode_with_var_unset']}")
    print(f"NO WILDCARD OPENS THE CANARY       "
          f"{p['no_wildcard_opens_the_canary']}")
    for path, blob in p["candidate_blob_identities"].items():
        print(f"  {path:<40} {blob}")
    print()
    print(f"BANK_SIZE                         {manifest['bank_size']}")
    print(f"ELIGIBLE_EXPECTED                 {manifest['eligible_expected']}")
    print(f"INELIGIBLE_EXPECTED               {manifest['ineligible_expected']}")
    print(f"DISTINCT_INELIGIBLE_REASONS       "
          f"{manifest['distinct_ineligible_reasons']}")
    print()
    for case in cases:
        flag = ("ELIGIBLE" if case["expected_slice1_eligible"]
                else case["expected_ineligible_reason"])
        print(f"  {case['case_id']}  {case['frozen_question_id']:<6} {flag:<24} "
              f"{case['expected_shadow_disposition'][:58]}")
    print()
    print(f"MANIFEST_SHA256                   {digest}")
    print(f"WRITTEN                           {out.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

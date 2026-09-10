#!/usr/bin/env python3
"""Phase 1: pre-register the live bank, OFFLINE, before any model call.

Every expectation in the manifest this writes is read out of the FROZEN run-8
sign-off evidence (`run8_135_signoff_2b00172.json`) or computed by the
independent control that Gate 5 already used. Nothing here consults a live
answer, because no live answer exists yet when this runs — which is the whole
point of pre-registration.

WHAT THE BANK CAN AND CANNOT COVER. The brief asks for eligible positives
covering sum, count, mean, weighted mean, one filter, conjunctive filters, one
dimension, two dimensions, the configured core-region dimension and a grouped
result, "where available". Measured against the frozen bank, four of those are
NOT available as eligible positives at this SHA, and the manifest records each
gap with its reason rather than inventing question wording the sign-off never
adjudicated:

    mean                 no bank question compiles to statistic `average`
    weighted mean        only Q05 does, and Q05 states the Direct lens, so it is
                         an EXPLICIT_LENS control rather than a positive
    one filter alone     every eligible bank case carries two predicates or none
    one dimension alone  every eligible breakdown carries two axes
    core-region axis     Q14/Q16 request a governed geography binding, which is a
                         slice 1 ineligibility as of the geography fix — so the
                         region axis is a CONTROL here, not a positive

Likewise only five of the gate's fifteen ineligibility reasons are reachable from
this bank: CAPABILITY_NOT_GENERIC, EXPLICIT_LENS, GEOGRAPHY_REQUESTED,
OPERATION_NOT_GENERIC and NOT_A_PLAN. PERIOD_NOT_CURRENT is unreachable (no bank
question is generic_analysis with a non-current period), and NOT_SINGLE_OUTPUT
and TOO_MANY_DIMENSIONS are masked: Q17 carries three outputs and three axes but
also states the Direct lens, and the lens check is declared first. Q17A is
pre-registered anyway, expecting EXPLICIT_LENS, because a gate that returned
NOT_SINGLE_OUTPUT instead would be a mismatch worth catching.

Run: `python due_diligence/evidence/live_slice1_signoff/preregister.py`
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_runtime_adapter as adapter                  # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth           # noqa: E402

# The independent control Gate 5 used, reused rather than rewritten: one control
# implementation means the live run and the replay cannot disagree by accident.
sys.path.insert(0, str(_REPO_ROOT / "due_diligence" / "evidence"
                      / "plan_shadow_slice1"))
import corpus_replay as control                                      # noqa: E402

FROZEN = (_REPO_ROOT / "mi_agent" / "interpretation_v2" / "evidence"
          / "run8_135_signoff_2b00172.json")
OUT = Path(__file__).resolve().parent / "live_bank_manifest.json"

#: The bank. `(case_id, frozen question id, what it is here to cover)`.
#: Question strings are taken verbatim from the frozen sign-off, never rewritten.
BANK: List[tuple] = [
    # ---- eligible positives -------------------------------------------------
    ("E01", "Q01A", ["count", "conjunctive_numeric_filters", "scalar"]),
    ("E02", "Q01B", ["count", "paraphrase_of_E01"]),
    ("E03", "Q02A", ["sum", "conjunctive_numeric_filters", "scalar"]),
    ("E04", "Q02C", ["sum", "paraphrase_of_E03"]),
    ("E05", "Q03A", ["count", "categorical_filter_governed_value_vs_display_form",
                     "numeric_filter"]),
    ("E06", "Q03B", ["count", "paraphrase_of_E05"]),
    ("E07", "Q12A", ["sum", "two_dimensions", "grouped_result", "cell_for_cell"]),
    ("E08", "Q12B", ["sum", "two_dimensions", "paraphrase_of_E07"]),
    ("E09", "Q11A", ["sum", "two_dimensions", "expected_fixture_gap_ticket_bucket"]),
    ("E10", "Q13A", ["sum", "two_dimensions",
                     "expected_fixture_gap_interest_rate_bucket"]),
    # ---- ineligible controls ------------------------------------------------
    ("I01", "Q04A", ["explicit_direct_lens"]),
    ("I02", "Q05A", ["explicit_direct_lens", "weighted_mean_exists_only_here"]),
    ("I03", "Q14A", ["configured_core_region_axis", "geography_binding"]),
    ("I04", "Q12C", ["distribution_operation", "same_canonical_as_E07"]),
    ("I05", "Q17A", ["explicit_direct_lens", "also_three_outputs",
                     "also_three_dimensions"]),
    ("I06", "Q06A", ["specialist_portfolio_summary"]),
    ("I07", "Q19A", ["specialist_period_movement", "historical_temporal"]),
    ("I08", "Q23A", ["specialist_forecast", "forward_temporal"]),
    ("I09", "SM01A", ["specialist_pipeline_stage_movement"]),
    ("I10", "NL1A", ["clarify_has_no_plan"]),
]

#: Coverage the frozen bank cannot supply as an eligible positive, with why.
COVERAGE_NOT_AVAILABLE = {
    "mean": "no question in the frozen bank compiles to statistic 'average'",
    "weighted_mean": "only Q05 compiles to 'weighted_average', and it states the "
                     "Direct lens, so it is pre-registered as an EXPLICIT_LENS "
                     "control (I02) rather than an eligible positive",
    "single_filter": "every slice-1-eligible bank case carries two predicates or "
                     "none",
    "single_dimension": "every slice-1-eligible breakdown in the bank carries two "
                        "axes",
    "configured_core_region_dimension":
        "Q14/Q16 request a governed geography binding, which is a slice 1 "
        "ineligibility as of the geography fix, so the region axis appears as "
        "control I03 rather than as a positive",
}

#: Gate reasons no bank question can reach, with why. Covered by the focused unit
#: suite instead, which puts synthetic plans to `check_eligibility` directly.
REASONS_NOT_REACHABLE = {
    "PERIOD_NOT_CURRENT": "no bank question is generic_analysis with a "
                          "non-current period",
    "NOT_SINGLE_OUTPUT": "the only multi-output generic plans (Q17A/Q17C) also "
                         "state the Direct lens, and the lens check is declared "
                         "first",
    "TOO_MANY_DIMENSIONS": "the only three-axis plan (Q17B) also states the "
                           "Direct lens",
    "COMPARISON_REQUESTED": "every comparing question is a specialist capability",
    "TARGET_REQUESTED": "every target question is a specialist capability",
    "MEASURE_NOT_GENERIC / MEASURE_UNBOUND / DIMENSION_UNBOUND / "
    "FILTER_UNBOUND / FILTER_NOT_EXPRESSIBLE / NO_MEASURE":
        "the compiler refuses or clarifies before producing such a plan",
}


def _semantics(plan: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The governed semantics a live plan must preserve. Read off the frozen plan."""
    if not plan:
        return {}
    output = (tuple(plan.get("outputs") or ()) or ({},))[0]
    measure = (tuple(output.get("measures") or ()) or ({},))[0]
    geography = plan.get("geography") or {}
    return {
        "capability": plan.get("capability"),
        "operation": plan.get("operation"),
        "statistic": measure.get("statistic"),
        "measure_concept": measure.get("concept"),
        "measure_field": measure.get("canonical_field"),
        "weight_field": measure.get("weight_field"),
        "dimensions": [d["canonical_field"] for d in (output.get("dimensions") or ())],
        "filters": sorted((f["canonical_field"], str(f.get("comparator") or "eq"),
                           f.get("value"))
                          for f in (tuple(plan.get("filters") or ())
                                    + tuple(output.get("filters") or ()))),
        "population_base": (plan.get("population") or {}).get("base"),
        "population_lens": (plan.get("population") or {}).get("lens"),
        "period_form": (plan.get("period") or {}).get("form"),
        "comparison_kind": plan.get("comparison_kind"),
        "geography_requested": bool(geography),
        "geography_group_by": bool(geography.get("group_by")),
        "output_count": len(plan.get("outputs") or ()),
    }


def build() -> Dict[str, Any]:
    frozen = json.loads(FROZEN.read_text())
    by_id = {r["question_id"]: r for r in frozen["results"]}
    book = truth.canonical_book()
    columns = set(book.columns)

    cases: List[Dict[str, Any]] = []
    for case_id, question_id, coverage in BANK:
        record = by_id[question_id]
        plan = record.get("plan")
        eligible, reason, detail = adapter.check_eligibility(plan)
        semantics = _semantics(plan)

        expected_value: Optional[float] = None
        expected_cells: Optional[Dict[str, float]] = None
        if not eligible:
            disposition = "NOT_EXECUTED_INELIGIBLE"
        else:
            missing = sorted(control._referenced_fields(plan) - columns)
            if missing:
                disposition = "UNEXECUTABLE_FIXTURE_GAP"
            elif semantics["dimensions"]:
                disposition = "EXECUTABLE_GROUPED"
                expected_cells = {
                    " | ".join(key): round(value, 4) for key, value in
                    control.control_cells(book, plan,
                                          semantics["dimensions"]).items()}
            else:
                disposition = "EXECUTABLE_SCALAR"
                expected_value = round(control.control_scalar(book, plan), 4)

        cases.append({
            "case_id": case_id,
            "frozen_question_id": question_id,
            "question": record["question"],
            "coverage": coverage,
            "expected_interpretation_disposition": record["outcome"],
            "expected_governed_semantics": semantics,
            "expected_slice1_eligible": eligible,
            "expected_ineligible_reason": reason,
            "expected_ineligible_detail_contains":
                detail.split(" ")[0] if detail else "",
            "expected_deterministic_disposition": disposition,
            "independent_expected_value": expected_value,
            "independent_expected_cells": expected_cells,
            "expected_frozen_plan_id": record.get("plan_id"),
            "provenance": f"{FROZEN.name} :: {question_id} (frozen sign-off "
                          f"adjudication; no live answer consulted)",
        })

    manifest = {
        "phase": "Slice 1 live Opus integration sign-off — Phase 1 "
                 "pre-registration",
        "written_before_any_live_call": True,
        "frozen_source": str(FROZEN.relative_to(_REPO_ROOT)),
        "frozen_source_sha256":
            hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "replay_frame": "mi_agent/tests/portfolio_truth_oracle.py :: "
                        "canonical_book() — 400 rows, seed 20260905",
        "independent_control": "due_diligence/evidence/plan_shadow_slice1/"
                               "corpus_replay.py (the Gate 5 control, reused)",
        "model_required": "claude-opus-5",
        "authorised_live_calls": 25,
        "bank_size": len(cases),
        "eligible_expected": sum(1 for c in cases if c["expected_slice1_eligible"]),
        "ineligible_expected": sum(1 for c in cases
                                   if not c["expected_slice1_eligible"]),
        "executable_expected": sum(
            1 for c in cases if c["expected_deterministic_disposition"].startswith(
                "EXECUTABLE")),
        "fixture_gaps_expected": sum(
            1 for c in cases
            if c["expected_deterministic_disposition"] == "UNEXECUTABLE_FIXTURE_GAP"),
        "coverage_not_available_from_the_frozen_bank": COVERAGE_NOT_AVAILABLE,
        "gate_reasons_not_reachable_from_the_frozen_bank": REASONS_NOT_REACHABLE,
        "cases": cases,
    }
    return manifest


def main() -> int:
    manifest = build()
    body = json.dumps(manifest, indent=2, default=str, sort_keys=False) + "\n"
    OUT.write_text(body)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    (OUT.parent / "live_bank_manifest.sha256").write_text(
        f"{digest}  {OUT.name}\n")

    print(f"BANK_SIZE            {manifest['bank_size']}")
    print(f"ELIGIBLE_EXPECTED    {manifest['eligible_expected']}")
    print(f"INELIGIBLE_EXPECTED  {manifest['ineligible_expected']}")
    print(f"EXECUTABLE_EXPECTED  {manifest['executable_expected']}")
    print(f"FIXTURE_GAPS         {manifest['fixture_gaps_expected']}")
    print()
    for case in manifest["cases"]:
        flag = ("ELIGIBLE" if case["expected_slice1_eligible"]
                else case["expected_ineligible_reason"])
        print(f"  {case['case_id']}  {case['frozen_question_id']:<6} "
              f"{case['expected_interpretation_disposition']:<8} {flag:<24} "
              f"{case['expected_deterministic_disposition']}")
    print()
    print(f"MANIFEST_SHA256      {digest}")
    print(f"WRITTEN              {OUT.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

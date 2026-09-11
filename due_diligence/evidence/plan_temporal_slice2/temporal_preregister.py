#!/usr/bin/env python3
"""Phase 5A: pre-register the temporal live bank, OFFLINE, before any model call.

WHAT IS BEING PROVED, AND WHAT IS NOT. The Slice 2 offline acceptance measured
the compiler, the perimeter, the temporal resolver, the snapshot selector and
the executor — with intents this repository authored. It could not say whether
**Opus** states temporal intent in a shape the deterministic layer can act on.
That is the one open question Slice 2 ended on, and it is the only question this
bank exists to answer.

So the boundary under test is exactly:

    raw question -> Opus -> CandidateIntent -> compiler -> GovernedQueryPlan
                 -> slice 2 perimeter -> SnapshotSelector -> resolved snapshots

and it stops there. Nothing is executed against a book, no MI API is called, no
bearer is needed, and no product module changes.

WHAT IS PINNED, AND WHAT DELIBERATELY IS NOT. A pre-registration that demanded
the model emit a particular `time.form`, or `operation`, or statistic, would be
scoring Opus against this file's guess at its wording rather than against
whether the governed layer can act on what it said. Two readings are routinely
both correct: `series` and `range` are both span forms the perimeter admits, and
`series` and `breakdown` are both operations it admits.

    PINNED    the plan must be slice 2 ELIGIBLE;
              the resolver must land on the pre-registered SNAPSHOTS;
              the measure concept, the statistic (from an allowed set), the
              filters and the dimensions must survive;
              no physical date or snapshot id may appear anywhere.

    NOT PINNED  which span form, which operation, which words the model chose.

`expected_snapshots` is written out by hand against the eight-month fixture
catalogue (2025-11-30 … 2026-06-30) and is never read back from the product, so
a wrong selection fails here rather than agreeing with itself. Every relative
window resolves against the CATALOGUE's latest period, not against wall-clock,
so this bank produces the same expectation whenever it is run.

Run: `python due_diligence/evidence/plan_temporal_slice2/temporal_preregister.py`
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "temporal_bank_manifest.json"
MANIFEST_HASH = HERE / "temporal_bank_manifest.sha256"

#: The fixture catalogue every expectation below is written against.
CATALOGUE = ["2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28",
             "2026-03-31", "2026-04-30", "2026-05-31", "2026-06-30"]

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"

#: An average of a rate is genuinely two defensible readings — the simple mean
#: and the balance-weighted mean — and the governed registry permits both. A
#: pre-registration that insisted on one would be scoring a coin toss.
ANY_AVERAGE = ["average", "weighted_average"]


def case(case_id: str, question: str, category: str, **overrides) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "id": case_id,
        "question": question,
        "category": category,
        # PLAN / CLARIFY / REFUSE at the compiler.
        "expected_interpretation_disposition": "PLAN",
        # Whether the plan's period form must be something other than `current`.
        # A temporal question answered with a current-period plan is the model
        # dropping the time constraint, which is the first thing this bank looks
        # for.
        "temporal_required": True,
        # ELIGIBLE, or the slice 2 ineligibility reason expected.
        "expected_eligibility": "ELIGIBLE",
        # Reporting dates the resolver must land on. None when the case is not
        # expected to reach resolution.
        "expected_snapshots": None,
        # Governed semantics that must survive the model boundary.
        "expected_measure_concept": BALANCE,
        "expected_statistics": ["sum"],
        "expected_filters": [],
        "expected_dimensions": [],
        "expected_population_base": "funded",
        "note": "",
    }
    body.update(overrides)
    return body


CASES: List[Dict[str, Any]] = [

    # ---- the ten required slice 2 categories, as positives ---------------- #

    case("T01", "How has funded balance changed over the last six months?",
         "last_n_months", expected_snapshots=CATALOGUE[-6:],
         note="The span is stated as a COUNT of months. The resolver can only "
              "honour it if the model supplies periods_back=6; a labels-only "
              "interpretation clarifies, and that would be the finding."),

    case("T02", "What was the funded book balance in each of the last three months?",
         "last_n_months", expected_snapshots=CATALOGUE[-3:]),

    case("T03", "What was funded balance each month?",
         "each_month", expected_snapshots=list(CATALOGUE),
         note="No count and no anchor: the whole available series. Needs a "
              "label the governed span vocabulary recognises."),

    case("T04", "How many loans were on the book each month?",
         "count_each_month", expected_snapshots=list(CATALOGUE),
         expected_measure_concept="loan", expected_statistics=["count"]),

    case("T05", "How has average LTV changed over the last four months?",
         "average_over_time", expected_snapshots=CATALOGUE[-4:],
         expected_measure_concept=LTV, expected_statistics=ANY_AVERAGE),

    case("T06", "How many drawdown loans were there each month?",
         "filtered_evolution", expected_snapshots=list(CATALOGUE),
         expected_measure_concept="loan", expected_statistics=["count"],
         expected_filters=[["erm_product_type", "eq", "drawdown"]]),

    case("T07", "Show loan count by LTV bucket each month.",
         "time_by_one_dimension", expected_snapshots=list(CATALOGUE),
         expected_measure_concept="loan", expected_statistics=["count"],
         expected_dimensions=["ltv_bucket"]),

    case("T08", "Show loan count by LTV bucket and product type for the last "
                "two months.",
         "time_by_two_dimensions", expected_snapshots=CATALOGUE[-2:],
         expected_measure_concept="loan", expected_statistics=["count"],
         expected_dimensions=["erm_product_type", "ltv_bucket"]),

    case("T09", "What was funded balance in April?",
         "explicit_period", expected_snapshots=["2026-04-30"],
         note="A named month with no year. The catalogue carries exactly one "
              "April, so it resolves; a book with two would clarify."),

    case("T10", "How has funded balance changed since March?",
         "since_period", expected_snapshots=CATALOGUE[-4:]),

    case("T11", "What was funded balance this month versus last month?",
         "current_versus_prior", expected_snapshots=CATALOGUE[-2:]),

    # ---- controls: each must fail closed somewhere on the boundary -------- #

    case("T12", "What will funded balance be over the next three months?",
         "future_period", expected_interpretation_disposition="REFUSE",
         temporal_required=False, expected_eligibility="NOT_REACHED",
         expected_measure_concept=None, expected_statistics=[],
         note="A forward-looking period on a capability that owns no forward "
              "methodology. Expected to refuse at the compiler; if the model "
              "instead routes it to the forecast capability, that is a "
              "JUSTIFIED_REFUSE at the perimeter and is recorded as such."),

    case("T13", "How has funded balance changed over the last twenty-four months?",
         "unavailable_period", expected_eligibility="ELIGIBLE",
         expected_snapshots=[],
         note="The book carries eight periods. The resolver must clarify with "
              "PERIOD_NOT_AVAILABLE and must not return a shorter series."),

    case("T14", "Why did funded balance increase last month?",
         "movement_cause", expected_interpretation_disposition="ANY",
         temporal_required=False, expected_eligibility="INELIGIBLE",
         expected_measure_concept=None, expected_statistics=[],
         note="Movement attribution is out of the slice 2 contract. Whether it "
              "arrives as period_movement (CAPABILITY_NOT_GENERIC) or as a "
              "generic movement operation (OPERATION_NOT_TEMPORAL), it must "
              "not be answered as an evaluation across snapshots."),

    case("T15", "Give me balance, loan count and average LTV for each of the "
                "last three months.",
         "multi_output_temporal", expected_interpretation_disposition="ANY",
         expected_eligibility="INELIGIBLE",
         expected_measure_concept=None, expected_statistics=[],
         note="Three independent measures in one request. Must not be silently "
              "simplified to one."),
]


def main() -> int:
    manifest = {
        "bank_id": "slice2_temporal_live_v1",
        "purpose": ("prove the Opus -> temporal CandidateIntent -> "
                    "GovernedQueryPlan -> slice 2 perimeter -> SnapshotSelector "
                    "boundary on fresh live interpretations"),
        "boundary_under_test": [
            "raw question", "OpusInterpreter.interpret", "CandidateIntent",
            "DeterministicCompiler.compile", "GovernedQueryPlan",
            "plan_temporal_runtime.check_temporal_eligibility",
            "plan_temporal_runtime.resolve_temporal", "SnapshotSelector",
            "resolved snapshot headers",
        ],
        "stops_before": ["execute_mi_query", "any MI API call",
                         "any serving surface", "any deployment"],
        "model": "claude-opus-5",
        "authorised_live_calls": len(CASES),
        # STATED AS IMPLEMENTED, not as permitted. `temporal_live_run.run_case`
        # calls the interpreter exactly once and records whatever comes back,
        # including a transport failure. It does not retry, reword or probe.
        "retry_policy": ("exactly one call per case; a failure is RECORDED, "
                         "never retried, reworded or probed"),
        "fixture_catalogue": CATALOGUE,
        "fixture_cadence": "monthly",
        "relative_windows_resolve_against": ("the catalogue's latest reporting "
                                             "period, never wall-clock"),
        "what_is_pinned": [
            "slice 2 eligibility", "the resolved snapshot set",
            "measure concept", "statistic (from the allowed set)", "filters",
            "dimensions", "population base",
            "the absence of any physical date or snapshot id",
        ],
        "what_is_not_pinned": [
            "which span form the model chose (series or range)",
            "which operation the model chose (series, breakdown or point_in_time)",
            "the model's own wording of the period label",
        ],
        "cases": CASES,
    }
    body = json.dumps(manifest, indent=2, sort_keys=False)
    MANIFEST.write_text(body + "\n", encoding="utf-8")
    digest = hashlib.sha256((body + "\n").encode("utf-8")).hexdigest()
    MANIFEST_HASH.write_text(f"{digest}  {MANIFEST.name}\n", encoding="utf-8")

    print("=== SLICE 2 TEMPORAL LIVE BANK — PRE-REGISTERED (no model call)")
    print(f"  cases                  {len(CASES)}")
    print(f"  authorised live calls  {len(CASES)}")
    print(f"  positives              {sum(1 for c in CASES if c['expected_eligibility'] == 'ELIGIBLE' and c['expected_snapshots'])}")
    print(f"  controls               {sum(1 for c in CASES if c['expected_eligibility'] != 'ELIGIBLE' or not c['expected_snapshots'])}")
    print(f"  catalogue              {CATALOGUE[0]} .. {CATALOGUE[-1]} "
          f"({len(CATALOGUE)} monthly periods)")
    print(f"  manifest               {MANIFEST.relative_to(_REPO_ROOT)}")
    print(f"  sha256                 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

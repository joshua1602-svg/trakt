#!/usr/bin/env python3
"""migration_phase0/freeze_baseline.py — the Phase 0 migration baseline.

READ-ONLY. Produces `migration_phase0/BASELINE.json`: the state the migration is
measured against, recorded BY NAME wherever a name exists.

The purpose is ATTRIBUTION. A later migration failure must be separable from
behaviour that was already there, so this records four distinct things and never
collapses them:

    delivered          behaviour that answers today
    governed refusal   behaviour that correctly declines today
    known failure      a test that fails today, for a reason already understood
    known defect       a governance gap that is real, open, and NOT the
                       migration's fault when it is later found

The fixture is hashed. If the hashes move, the baseline is void and nothing
measured against it means anything.

    python -m migration_phase0.freeze_baseline
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "migration_phase0" / "BASELINE.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fixture_hashes() -> Dict[str, Any]:
    root = _REPO / "demo_platform/workspace/store/processed/platform/alderbridge"
    files: Dict[str, str] = {}
    for period in sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        tape = root / period / "platform_canonical_typed.csv"
        if tape.exists():
            files[f"{period}/platform_canonical_typed.csv"] = _sha256(tape)
    return {"root": str(root), "files": files}



def _frozen_git() -> Dict[str, Any]:
    """The frozen git block: preserved if one already exists, else taken now."""
    if OUT.exists() and "--refreeze" not in sys.argv:
        try:
            existing = json.loads(OUT.read_text()).get("git")
            if existing:
                return existing
        except (OSError, ValueError):
            pass
    return {
        "headSha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "scopingStudyCommit": "9f2d256",
        "measurementBaseCommit": "42cef00",
        "productCodeIdenticalToBase": _git(
            "diff", "--name-only", "42cef00", "HEAD",
            "--", ":!docs", ":!compositional_plan_scoping",
            ":!migration_phase0", ":!tests/test_migration_preregistered.py") == "",
        "workingTreeClean": _git("status", "--porcelain") == "",
    }


def main() -> int:
    baseline: Dict[str, Any] = {
        "artefact": "MI compositional migration — Phase 0 baseline",
        "purpose": ("Attribution. Later migration movement must be separable from "
                    "behaviour that was already present at this commit."),
        # THE FREEZE IS THE POINT. This block records the state the baseline was
        # frozen at, and re-running this script must NOT re-stamp it to whatever
        # HEAD happens to be — a "Phase 0 baseline" that silently follows HEAD
        # cannot attribute anything. An existing block is preserved verbatim;
        # only a first run writes it. Use --refreeze to deliberately re-take it.
        "git": _frozen_git(),
        "fixture": fixture_hashes(),
        "environment": {
            "note": ("Measured with TRAKT_RUNTIME_MODE=development and the LLM "
                     "parser OFF. Every figure below is the deterministic arm."),
            "pandas": __import__("pandas").__version__,
        },

        # ---- delivered / refusing behaviour, by surface --------------------
        "surfaces": {
            "calibration_bank": {
                "runner": "pytest mi_agent/tests/test_mi_calibration_bank.py",
                "bank": "config/mi/golden_questions/ere_mi_calibration_250.yaml",
                "result": {"passed": 267, "failed": 0},
            },
            "robustness_44": {
                "runner": "python -m question_interpretation.run_robustness_deterministic",
                "record": "migration_phase0/robustness_deterministic.json",
                "result": {"CORRECT": 32, "UNHELPFUL_REFUSAL": 6,
                           "SAFE_REFUSAL": 4,
                           "CORRECT_WITH_DISCLOSED_LIMITATION": 2},
                "seasoning_families_by_name": {
                    "Q1": {"CORRECT": 4}, "Q7": {"CORRECT": 4}, "Q8": {"CORRECT": 12}},
                "by_intent": {
                    "Q1": {"CORRECT": 4},
                    "Q2": {"CORRECT": 3, "UNHELPFUL_REFUSAL": 1},
                    "Q3": {"CORRECT_WITH_DISCLOSED_LIMITATION": 2, "UNHELPFUL_REFUSAL": 2},
                    "Q4": {"CORRECT": 2, "UNHELPFUL_REFUSAL": 2},
                    "Q5": {"CORRECT": 3, "UNHELPFUL_REFUSAL": 1},
                    "Q6": {"CORRECT": 4},
                    "Q7": {"CORRECT": 4},
                    "Q8": {"CORRECT": 12},
                    "Q9": {"SAFE_REFUSAL": 4}},
            },
            "shipped_shapes": {
                "runner": "python -m question_interpretation.shipped_shapes",
                "record": "migration_phase0/shipped_shapes.json",
                "result": {"correct": 15, "wrong_answer": 0, "honest_refusal": 0,
                           "unhelpful_refusal": 0, "total": 15},
                "note": ("Cases C1-C5 are the arity-2 grouped shape "
                         "(ltv_bucket x ticket_bucket, 50 cells). They pass, AND "
                         "they are the shape the known-open arity-2 disclosure "
                         "defect below applies to."),
            },
            "routed_surface": {
                "runner": "python -m question_interpretation.routed_surface",
                "record": "migration_phase0/routed_surface.json",
                "result": {"passed": 31, "failed": 1},
                "failing_case_names": ["rt_004"],
            },
            "recognition_diagnosis_61": {
                "runner": "python -m question_interpretation.mi_recognition_diagnosis",
                "record": "migration_phase0/recognition_diagnosis.json",
                "result": {"DELIVERED": 15, "WORDING": 7, "UNPARSED": 10,
                           "CAPABILITY": 29, "reached_no_route": 13},
                "by_shape_delivered": {"T1": 6, "T2": 1, "T3": 0, "T4": 0,
                                       "T5": 0, "T6": 0, "T7": 3, "T8": 5},
            },
            "time_series_surface": {
                "runner": "python -m question_interpretation.time_series_surface",
                "record": "migration_phase0/time_series_surface.json",
                "ratings": {"T1": "PROVEN", "T2": "PARTIAL", "T3": "ABSENT",
                            "T4": "ABSENT", "T5": "ABSENT", "T6": "ABSENT",
                            "T7": "ABSENT", "T8": "ABSENT"},
                "silent_drops": 0,
                "honest_refusals": "20 of 29 runs",
                "note": ("SILENT DROPS = 0 is the P0 property. It must remain 0 "
                         "at every migration step. A migration that reintroduces "
                         "one has failed regardless of its economics."),
            },
        },

        # ---- known failures: real, understood, NOT the migration's ---------
        "known_failures": [
            {
                "name": "tests/test_analytical_capability_layer.py::TestSecondBookAcceptance"
                        "::test_q7_compares_the_two_governed_sides_and_reconciles",
                "symptom": "assert answer['ok'] is True -> assert False is True",
                "attribution": ("Verified to fail identically on a clean tree at "
                                "42cef00, before any migration work. Pre-existing."),
                "status": "known-open, unattributed to this programme",
            },
            {
                "name": "question_interpretation.routed_surface::rt_004",
                "question": "funded balance by quarter",
                "symptom": ("expects route=None / verdict=None; now reaches "
                            "route='evolution' with verdict='refuse'"),
                "attribution": ("The rt_004 expectation was last edited at 1b90fe4 "
                                "and pinned as a deliberate 'before'. Commit "
                                "42cef00 widened the time-axis vocabulary, which "
                                "moved the behaviour. The expectation was not "
                                "updated with it."),
                "status": "known-open, attributed to 42cef00, NOT to the migration",
            },
        ],

        # ---- known governance defects: open before the migration starts ----
        "known_defects": [
            {
                "id": "arity-2-disclosure",
                "title": "Thin-sample and denominator disclosure are arity-1 only",
                "location": "mi_agent/mi_query_executor.py::_execute_grouped",
                "mechanism": ("Both the `loan_count` denominator column and the "
                              "thin-sample warning sit inside "
                              "`if len(group_cols) == 1:`. A grouped answer with "
                              "two or more dimensions omits both."),
                "measured": {
                    "arity_1 collateral_geography": {"groups": 12, "thin": 1,
                                                     "disclosed": True},
                    "arity_2 collateral_geography x ltv_bucket": {
                        "groups": 88, "thin": 16, "disclosed": False},
                    "arity_2 ltv_bucket x ticket_bucket": {
                        "groups": 50, "thin": 11, "disclosed": False},
                },
                "live_on": ("shipped_shapes C1-C5, which currently PASS — the "
                            "defect is on a shipped, passing shape"),
                "secondary": ("_execute_grouped_measure_set attaches loan_count at "
                              "every arity but raises the thin-sample warning at "
                              "NO arity; mi_query_executor.py:1045 "
                              "(contribution) resolves only group_keys[0], a "
                              "latent arity assumption currently unreachable "
                              "because the 2-dimension contribution question "
                              "refuses upstream."),
                "status": ("KNOWN-OPEN AT PHASE 0. Pre-registered so it cannot be "
                           "discovered mid-migration and attributed to it. "
                           "Expectations are committed, declared failing, at "
                           "tests/test_migration_preregistered.py::"
                           "TestArityIndependentDisclosure."),
                "policy_note": ("LOW_GROUP_COUNT = 5 and the avg/weighted_avg "
                                "restriction are POLICY. This programme "
                                "generalises where the policy applies; it does "
                                "not change what the policy is."),
            },
            {
                "id": "interpretation-lens-gap",
                "title": "The interpretation contract cannot carry a portfolio lens",
                "location": "question_interpretation/schema.py + projection.py",
                "mechanism": ("QuestionInterpretation.population is EMPTY for "
                              "'Summarise the acquired book', while "
                              "mi_agent.portfolio_lens.resolve_lens(question) "
                              "resolves source_portfolio_type=acquired from the "
                              "RAW QUESTION. Seasoning populations ARE carried; "
                              "the source-portfolio lens is not."),
                "consequence": ("Blocks the portfolio_summary shadow plan on 9 of "
                                "9 surface cases. An empty population list cannot "
                                "be read as 'Total' while the lens is "
                                "unrepresentable."),
                "status": "KNOWN-OPEN AT PHASE 0; blocks the first migration slice",
            },
            {
                "id": "filter-clause-join",
                "title": "Filter clauses reach the contract as unjoined halves",
                "location": "question_interpretation/schema.py::FilterClaim.clause_id",
                "mechanism": ("71 questions carry a wording-only claim and a "
                              "binding-only claim; wording half located by span "
                              "71/71, binding half 0/71, clause_id set on 0. "
                              "Recorded in docs/mi_question_interpretation_stage2.md."),
                "consequence": ("Blocks `select population` from a row filter. Does "
                                "NOT block portfolio_summary, which applies none."),
                "status": "KNOWN-OPEN, pre-dates this programme",
            },
        ],

        "ships_to_clients_in_this_phase": "NOTHING",
        "capability_gates_closed": ["T3", "T4", "T5", "T6", "T7"],
    }

    estate = _REPO / "migration_phase0" / "estate.json"
    if estate.exists():
        baseline["surfaces"]["estate"] = json.loads(estate.read_text())

    OUT.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(_REPO)}")
    print(f"  frozen at       : {baseline['git']['headSha'][:12]}"
          f"  (preserved; --refreeze to re-take)")
    print(f"  product == base : {baseline['git']['productCodeIdenticalToBase']}")
    print(f"  fixture files   : {len(baseline['fixture']['files'])} hashed")
    print(f"  known failures  : {len(baseline['known_failures'])}")
    print(f"  known defects   : {len(baseline['known_defects'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

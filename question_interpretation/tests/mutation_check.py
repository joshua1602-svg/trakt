#!/usr/bin/env python3
"""question_interpretation/tests/mutation_check.py

Standing condition 4, discharged by evidence rather than by assertion.

Five times in this programme an instrument reproduced the exact defect it was
built to find. A passing test suite is not evidence that the suite would notice
a regression. So this applies real mutations to `schema.py` — each one a defect
this programme has actually seen — runs the suite against the mutated file, and
requires that a NAMED test fails. A mutation that leaves the suite green is
reported as an undetected defect class.

    python -m question_interpretation.tests.mutation_check

Restores the file afterwards, including on failure.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).resolve().parent
_SCHEMA = _HERE.parent / "schema.py"
_REPO = _HERE.parents[2]

#: (name, why it matters, find, replace, the test that must fail)
MUTATIONS: List[Tuple[str, str, str, str, str]] = [
    (
        "execution vocabulary leaks into the states",
        "`applied` carries execution evidence a pre-execution object cannot "
        "have. This is the conflation the whole contract turns on.",
        'STATES = (FILLED, EMPTY, UNRESOLVABLE)',
        'STATES = (FILLED, EMPTY, UNRESOLVABLE, "applied")',
        "test_no_execution_vocabulary_on_the_object",
    ),
    (
        "dimension role defaults to grouping",
        "This is KIND_GROUPING exactly: every named dimension silently becomes "
        "an axis, and 'balance by region for joint borrowers' loses its filter.",
        "    role: str = UNRESOLVED_ROLE",
        "    role: str = GROUPING",
        "test_role_defaults_to_unresolved_not_grouping",
    ),
    (
        "dimension role validation removed",
        "Without it an unknown role is stored and the conflation returns by a "
        "different name.",
        '        if self.role not in DIMENSION_ROLES:\n'
        '            raise ValueError("unknown dimension role %r" % (self.role,))',
        "        pass",
        "test_an_unknown_role_is_rejected",
    ),
    (
        "slot state validation removed",
        "An unresolvable slot could then be stored as any string, and the "
        "explicit-unresolvable guarantee stops being enforceable.",
        '        if self.state not in STATES:\n'
        '            raise ValueError("unknown slot state %r" % (self.state,))',
        "        pass",
        "test_an_unknown_state_is_rejected",
    ),
    (
        "target source shadows the interpreter source",
        "One field cannot mean both 'where the threshold came from' and 'which "
        "interpreter said so'. This was a real bug in the first draft.",
        "    target_source: Optional[str] = None\n\n"
        "    def __post_init__(self) -> None:\n"
        "        super().__post_init__()\n"
        "        if self.target_source is not None and self.target_source not in TARGET_SOURCES:\n"
        "            raise ValueError(\"unknown target source %r\" % (self.target_source,))",
        "    source: Optional[str] = None\n\n"
        "    def __post_init__(self) -> None:\n"
        "        super().__post_init__()\n"
        "        if self.source is not None and self.source not in TARGET_SOURCES:\n"
        "            raise ValueError(\"unknown target source %r\" % (self.source,))",
        "test_target_source_does_not_shadow_the_interpreter_source",
    ),
    (
        "grain and window collapse into one slot",
        "'balance by month over the last 6 months' then cannot be represented, "
        "which is the Stage 5 blocker.",
        "    comparison_period: Slot = field(default_factory=Slot)\n"
        "    requested_grain: Slot = field(default_factory=Slot)\n"
        "    trend_window: Slot = field(default_factory=Slot)",
        "    comparison_period: Slot = field(default_factory=Slot)\n"
        "    trend_window: Slot = field(default_factory=Slot)\n\n"
        "    @property\n"
        "    def requested_grain(self) -> Slot:\n"
        "        return self.trend_window",
        "test_grain_and_window_are_separate_slots",
    ),
    (
        "unresolvable slots stop being reported",
        "A slot that is unresolvable but unreported is the silent omission the "
        "state exists to prevent.",
        "    def unresolvable_slots(self) -> List[Tuple[str, Slot]]:\n"
        "        out: List[Tuple[str, Slot]] = []",
        "    def unresolvable_slots(self) -> List[Tuple[str, Slot]]:\n"
        "        return []\n"
        "        out: List[Tuple[str, Slot]] = []",
        "test_unresolvable_slots_reports_every_slot_kind",
    ),
    (
        "a resolved field key is admitted onto the object",
        "Resolution belongs to ResolvedConcept. Admitting field_key here is how "
        "the layers re-merge.",
        "    candidate_concept: Optional[str] = None\n\n    def as_dict(self) -> Dict[str, Any]:\n"
        "        d = super().as_dict()\n        d[\"candidate_concept\"] = self.candidate_concept",
        "    candidate_concept: Optional[str] = None\n    field_key: Optional[str] = None\n\n"
        "    def as_dict(self) -> Dict[str, Any]:\n        d = super().as_dict()\n"
        "        d[\"candidate_concept\"] = self.candidate_concept",
        "test_no_slot_carries_a_resolved_field_key",
    ),
    (
        "coverage is re-added to the operation vocabulary",
        "Correction 3. An operation type nothing supplies invites someone to "
        "populate it by intuition — the failure the corpus-first rule prevents.",
        "OPERATION_TYPES = (COUNT, AMOUNT, AVERAGE, MOVEMENT, RANKING, FORWARD, NEUTRAL)",
        "COVERAGE = \"coverage\"\n"
        "OPERATION_TYPES = (COUNT, AMOUNT, AVERAGE, MOVEMENT, RANKING, FORWARD,\n"
        "                   NEUTRAL, COVERAGE)",
        "test_coverage_is_not_an_operation_type",
    ),
    (
        "a conflicted dimension role is invented",
        "Correction 5. Zero of 690 questions evidence it. Adding it is exactly "
        "the intuition-shaped slot the contract warns about.",
        "DIMENSION_ROLES = (GROUPING, FILTER, UNRESOLVED_ROLE)",
        "CONFLICTED_ROLE = \"conflicted\"\n"
        "DIMENSION_ROLES = (GROUPING, FILTER, UNRESOLVED_ROLE, CONFLICTED_ROLE)",
        "test_no_conflicted_role_was_invented",
    ),
    (
        "a filter claim stops declaring what it carries",
        "Correction 1. Inferring a half-claim from which attributes are None "
        "cannot distinguish 'this interpreter does not supply it' from 'this "
        "interpreter looked and found nothing'.",
        "    provides: Tuple[str, ...] = ()",
        "    provides: Tuple[str, ...] = (WORDING, BOUND, FIELD)",
        "test_a_filter_claim_declares_nothing_by_default",
    ),
    (
        "claim contents stop being validated",
        "Correction 1. An unchecked `provides` lets a claim assert a half that "
        "does not exist.",
        "        unknown = set(self.provides) - set(CLAIM_CONTENTS)\n"
        "        if unknown:\n"
        "            raise ValueError(\"unknown claim contents %s\" % sorted(unknown))",
        "        pass",
        "test_unknown_claim_contents_are_rejected",
    ),
    (
        "span presence stops being reportable",
        "Correction 2. 170 claims across 690 questions have no span; a "
        "consumer that cannot tell is a consumer that will assume one.",
        "    @property\n"
        "    def has_span(self) -> bool:\n"
        "        \"\"\"Whether this claim can be located in the question at all.\"\"\"\n"
        "        return self.span is not None",
        "    @property\n"
        "    def has_span(self) -> bool:\n"
        "        return True",
        "test_a_claim_without_a_span_is_valid_and_says_so",
    ),
    (
        "the configured target source stops being marked unsupplied",
        "Correction 4. Silently treating it as supplied is how a slot nothing "
        "populates comes to look populated.",
        "UNSUPPLIED_TARGET_SOURCES = (CONFIGURED,)",
        "UNSUPPLIED_TARGET_SOURCES = ()",
        "test_the_configured_target_source_is_recorded_as_unsupplied",
    ),
]


def _run_suite() -> Tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest",
         str(_HERE / "test_schema.py"), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(_REPO), capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    original = _SCHEMA.read_text(encoding="utf-8")
    backup = Path(tempfile.mkdtemp()) / "schema.py.bak"
    shutil.copy2(_SCHEMA, backup)

    print("=" * 76)
    print("mutation check — every guarantee must be shown to be enforceable")
    print("=" * 76)

    code, out = _run_suite()
    if code != 0:
        print("BASELINE SUITE IS NOT GREEN — nothing below is meaningful\n" + out[-2000:])
        return 2
    print("baseline: green\n")

    undetected: List[str] = []
    misapplied: List[str] = []
    try:
        for name, why, find, replace, must_fail in MUTATIONS:
            if find not in original:
                misapplied.append(name)
                print("  %-46s MUTATION DID NOT APPLY" % name[:46])
                continue
            _SCHEMA.write_text(original.replace(find, replace, 1), encoding="utf-8")
            code, out = _run_suite()
            caught = code != 0 and must_fail in out
            status = "caught by %s" % must_fail if caught else (
                "SUITE STILL GREEN" if code == 0 else "failed, but NOT via %s" % must_fail)
            print("  %-46s %s" % (name[:46], status))
            if not caught:
                undetected.append("%s -> %s" % (name, status))
                print("      why it matters: %s" % why)
    finally:
        shutil.copy2(backup, _SCHEMA)

    code, _ = _run_suite()
    print("\nrestored: %s" % ("green" if code == 0 else "NOT GREEN — INVESTIGATE"))

    print("\n%d mutations, %d caught, %d undetected, %d misapplied"
          % (len(MUTATIONS), len(MUTATIONS) - len(undetected) - len(misapplied),
             len(undetected), len(misapplied)))
    for line in undetected:
        print("  UNDETECTED: %s" % line)
    return 1 if (undetected or misapplied or code != 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())

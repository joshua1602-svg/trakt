"""The permanent adversarial compound-question canary — structure, and movement.

Two guards, and the split between them is the point.

`test_bank_*`      STRUCTURAL. Fast, always run. They protect the bank from
                   being quietly weakened: a family deleted, a case removed, an
                   invariant dropped, an element left with no grading rule.

`test_canary_*`    EXECUTED. Runs all 33 cases through the live `/mi/query`
                   path and compares the grades to the frozen baseline in
                   `migration_phase0/COMPOUND_CANARY_FREEZE.json`.

WHAT THE EXECUTED GUARD ASSERTS, AND WHAT IT DELIBERATELY DOES NOT
------------------------------------------------------------------
It asserts MOVEMENT, not STATE. The baseline records four known defects
(D1–D4). Asserting them as expected output is exactly the mistake C6 made with
`test_the_stage_the_shipped_route_cannot_name`, which pinned a defect and had to
be retired the moment the product fixed it. An estate must not assert behaviour
the product has fixed.

So: any grade that CHANGES is reported, with its direction. A grade that improves
(DROPPED -> HONOURED) fails the test just as loudly as one that degrades — not
because improvement is bad, but because unexplained capability movement is the
thing this programme stops on. The fix is to attribute the movement and re-freeze
the baseline in the same commit, never to edit the bank into agreement.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from migration_phase0 import compound_canary as canary

BASELINE = Path(__file__).resolve().parent.parent / "migration_phase0" / \
    "COMPOUND_CANARY_FREEZE.json"

#: Floors, not targets. The bank may grow; it may not shrink past these without
#: a deliberate edit to this line, which is what makes the shrink visible.
MIN_FAMILIES = 10
MIN_CASES = 33
MIN_INVARIANTS = 7


@pytest.fixture(scope="module")
def bank():
    return canary.load_bank()


# --------------------------------------------------------------------------- #
# Structural
# --------------------------------------------------------------------------- #
def test_bank_is_not_vacuous(bank):
    families = bank["families"]
    cases = canary.cases(bank)
    assert len(families) >= MIN_FAMILIES, "families were removed from the bank"
    assert len(cases) >= MIN_CASES, "cases were removed from the bank"
    assert len(bank["invariants"]) >= MIN_INVARIANTS, "an invariant was dropped"
    for family in families:
        assert family["cases"], f"family {family['id']} has no cases"
        assert family.get("hazard"), (
            f"family {family['id']} states no hazard — a canary whose hazard is "
            f"not written down cannot be reviewed")


def test_every_case_id_is_unique(bank):
    ids = [c["id"] for c in canary.cases(bank)]
    assert len(ids) == len(set(ids)), "duplicate canary ids"


def test_every_declared_element_has_a_grading_rule(bank):
    """An element with no rule would grade silently, which is the defect class
    this bank exists to catch, reproduced inside the instrument itself."""
    declared = {e for c in canary.cases(bank) for e in c["declares"]}
    assert declared <= set(bank["elements"]), (
        f"undeclared elements: {sorted(declared - set(bank['elements']))}")
    empty = {"metadata": {}, "artifacts": [], "ok": True, "answer": ""}
    for element in sorted(declared):
        canary._grade_element(element, empty)   # raises if there is no rule


def test_every_paraphrase_and_lattice_reference_resolves(bank):
    ids = {c["id"] for c in canary.cases(bank)}
    for family in bank["families"]:
        for group in family.get("paraphrase_sets") or []:
            assert set(group) <= ids, f"{family['id']} paraphrase set dangles"
    for lat in bank.get("lattices") or []:
        assert lat["base"] in ids, f"lattice {lat['id']} base dangles"
        for step in lat["steps"]:
            assert step["case"] in ids, f"lattice {lat['id']} step dangles"


def test_known_defects_are_recorded_as_defects_not_as_expectations(bank):
    """The C6 lesson, made structural.

    The freeze observations may describe a defect. They may not be phrased as
    the bank's expected behaviour, and no invariant may be written to accommodate
    one.
    """
    obs = bank["freeze_observations"]
    assert obs["known_defects_at_freeze"], "the freeze recorded no defects at all"
    for defect in obs["known_defects_at_freeze"]:
        assert defect["breaches"] in bank["invariants"], (
            f"{defect['id']} breaches {defect['breaches']!r}, which is not an "
            f"invariant this bank defines")
        assert defect["canaries"], f"{defect['id']} names no canary"


def test_the_frozen_defects_are_never_edited_out_of_history(bank):
    """A fixed defect stays in the record.

    `COMPOUND_CANARY_FREEZE.json` advances — it must, or the movement detector
    fails forever on an already-attributed movement and can never catch the
    next one. What must NOT advance is the history: D1-D4 stand as the record
    of what was true at freeze. This pins that, and pins the rule that the JSON
    only advances alongside a ledger entry explaining what moved.
    """
    frozen = {d["id"] for d in
              bank["freeze_observations"]["known_defects_at_freeze"]}
    assert {"D1", "D2", "D3", "D4"} <= frozen, (
        "a defect was edited out of the freeze observations")
    for move in bank.get("authorised_movements") or []:
        assert move["defect"] in frozen, (
            f"{move['id']} attributes a movement to {move['defect']}, which the "
            f"freeze observations do not record")
        assert move["breaches_after"] == move["breaches_before"] - len(
            move["cleared"]) + move["new_breaches"], (
            f"{move['id']}'s arithmetic does not close")


def test_a_baseline_that_moved_carries_a_ledger_entry(bank):
    """The JSON may only sit ahead of the freeze if something explains why."""
    current = len(json.loads(BASELINE.read_text())["breaches"])
    moves = bank.get("authorised_movements") or []
    at_freeze = (moves[0]["breaches_before"] if moves else current)
    if current != at_freeze:
        assert moves, (
            f"the baseline carries {current} breaches but the bank froze at "
            f"{at_freeze} and no authorised movement explains the difference")
        assert moves[-1]["breaches_after"] == current, (
            f"the last ledger entry ends at {moves[-1]['breaches_after']} "
            f"breaches; the baseline carries {current}")


# --------------------------------------------------------------------------- #
# Executed
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def executed():
    pytest.importorskip("fastapi")
    pytest.importorskip("pandas")
    return canary.run()


def test_canary_measurement_is_complete(executed, bank):
    """Zero readings is not zero findings. A canary that could not run must fail
    loudly rather than report an empty, passing table."""
    assert len(executed["rows"]) == len(canary.cases(bank))
    assert all(r["grades"] for r in executed["rows"])


def test_canary_families_are_exercised(executed, bank):
    """I6. A family answered entirely by refusals proves nothing, so the count of
    unexercised families is pinned — it may not grow silently."""
    verdict = canary.evaluate(executed)
    frozen = json.loads(BASELINE.read_text())["unexercised_families"]
    assert verdict["unexercised_families"] == frozen, (
        f"the set of families that cannot be exercised MOVED: "
        f"{frozen} -> {verdict['unexercised_families']}")


def test_canary_grades_have_not_moved(executed):
    """The movement detector. Improvements fail as loudly as regressions."""
    frozen = {r["id"]: r["grades"] for r in json.loads(BASELINE.read_text())["rows"]}
    moved = []
    for row in executed["rows"]:
        before = frozen.get(row["id"])
        if before is None:
            moved.append(f"{row['id']}: NEW case, not in the frozen baseline")
            continue
        for element, grade in row["grades"].items():
            was = before.get(element)
            if was != grade:
                moved.append(f"{row['id']} {element}: {was} -> {grade}")
        for element in set(before) - set(row["grades"]):
            moved.append(f"{row['id']} {element}: {before[element]} -> (no longer declared)")
    assert not moved, (
        "canary grades moved. Attribute every line below and re-freeze "
        "COMPOUND_CANARY_FREEZE.json in the SAME commit — do not edit the bank "
        "to agree with the code:\n  " + "\n  ".join(moved))


def test_canary_breach_count_has_not_moved(executed):
    """The four known defects are NOT asserted individually — asserting a defect
    is what C6 had to retire. Only the count and the identity of the breaching
    (case, element, invariant) triples are pinned, so a fix shows up as a
    movement to be attributed rather than as a test that was always wrong."""
    def key(b):
        return (b["invariant"], b["case"], b["element"])

    now = sorted(key(b) for b in canary.evaluate(executed)["breaches"])
    frozen = sorted(key(b) for b in json.loads(BASELINE.read_text())["breaches"])
    assert now == frozen, (
        f"invariant breaches moved.\n"
        f"  newly breaching : {sorted(set(now) - set(frozen))}\n"
        f"  no longer breaching: {sorted(set(frozen) - set(now))}")

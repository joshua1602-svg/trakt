"""The coverage inventory must be able to produce every result it reports.

Standing rule 4: no instrument ships without a test proving it can fail. This
one carries a heavier burden than most, because its FIRST draft reported eleven
false holes from a one-size evidence bundle, and its second reported a false
hole on the routed population cell from a malformed envelope. Both were caught
by the self-test rather than by reading the output, which is the only reason
they are corrections rather than findings.
"""
from __future__ import annotations

import pytest

from mi_agent import execution_receipt as R
from question_interpretation import stamp_coverage as SC


@pytest.fixture(scope="module")
def result():
    return SC.scan()


def test_the_instrument_self_test_passes():
    assert SC.self_test() == 0


# --------------------------------------------------------------------------- #
# All four cell values must be producible
# --------------------------------------------------------------------------- #
def test_it_finds_the_hole_it_was_built_to_find(result):
    """KIND_POPULATION on the point-in-time path — the e35a01b defect."""
    cell = result["matrix"]["(point-in-time)"][R.KIND_POPULATION]
    assert cell["cell"] == "no-branch"
    assert not cell.get("designed")


def test_it_can_see_a_stamped_cell(result):
    """Otherwise "everything is a hole" would pass the test above."""
    assert result["matrix"]["(point-in-time)"][R.KIND_GROUPING]["cell"] == "stamped"


def test_it_tells_a_correct_non_application_from_a_missing_branch(result):
    """A comparison period on a risk-limit schedule was genuinely not compared.

    Blocking there is right. An instrument that called it a hole would bury the
    one real hole in fifteen false ones — which is exactly what the first draft
    did.
    """
    assert (result["matrix"]["risk_limits"][R.KIND_COMPARISON_PERIOD]["cell"]
            == "route-bound")
    assert (result["matrix"]["period_change_analysis"][R.KIND_COMPARISON_PERIOD]["cell"]
            == "stamped")


def test_it_marks_a_kind_that_cannot_be_raised_on_a_route(result):
    cell = result["matrix"]["risk_limits"][R.KIND_UNRESOLVED_ROLE]
    assert cell["cell"] == "unreachable"
    assert "reconcile_facets" in cell["why"]


# --------------------------------------------------------------------------- #
# The designed/live split is derived, not asserted in prose
# --------------------------------------------------------------------------- #
def test_exactly_one_hole_is_live(result):
    assert result["live_holes"] == ["(point-in-time)/row_population"]


def test_a_designed_hole_carries_its_reason(result):
    cell = result["matrix"]["(point-in-time)"][R.KIND_UNRESOLVED_MEASURE]
    assert cell["cell"] == "no-branch"
    assert cell["designed"]


def test_an_unresolved_measure_is_adjudicated_at_construction():
    """Why that hole is designed: the detector already stamped it, with a reason.

    This is the property the classification rests on, asserted against the code
    rather than against the table. If the detector stopped setting a reason, the
    hole would become live and this test would say so.
    """
    from mi_agent.mi_query_validator import load_mi_semantics
    from pathlib import Path
    semantics = load_mi_semantics(
        Path(__file__).resolve().parents[2] / "mi_agent"
        / "mi_semantics_field_registry.yaml")
    facets = R.detect_requested_facets(
        "balance and total flimflam by region", semantics, frame=None,
        requested_dimensions=[])
    unresolved = [f for f in facets if f.kind == R.KIND_UNRESOLVED_MEASURE]
    if not unresolved:
        pytest.skip("this phrasing raises no unresolved-measure slot")
    assert all(f.status == R.LOST and f.reason for f in unresolved)

"""Every kind a facet can be MOVED into must have a reconciler that receives it.

The closed-class assertion. Two regressions shared a precondition — a facet
reclassified after detection into a kind the path it arrived on could not
confirm — while having different causes:

  32c263a  the facet reached a branch that READ IT WRONGLY (a field name against
           governed predicate text). A comparison defect.
  e35a01b  the facet reached NO BRANCH AT ALL. An absence.

A better comparison could not have prevented an absent branch, so "the same
defect twice" is the wrong lesson and would have produced the wrong fix. What
generalises is the precondition, and this closes it: the moves are discovered
from the code, every move must be registered, and every registered target must
have a receiver or a stated reason why none is needed.

Finding the third one here is cheaper than finding it in production, which is
where the first two were found.
"""
from __future__ import annotations

import pytest

from mi_agent import execution_receipt as R
from question_interpretation import stamp_coverage as SC


class _Spec:
    """No filters and no axes: the shape that sends a facet down every branch
    of the reclassifier, so no variant's target is missed."""

    filters: dict = {}
    dimensions: list = []
    dimension = None


class _SpecWithFilter:
    filters = {"probe_field": "probe value"}
    dimensions: list = []
    dimension = None


def _grouping():
    return R.RequestedFacet(
        kind=R.KIND_GROUPING, label="probe term", field_key="probe_field",
        alt_keys=(), concepts=("probe term",), status=R.LOST, reason="")


def _observed_targets() -> set:
    """The kinds the reclassifier ACTUALLY produces, discovered by running it.

    Not read from a list in the source. A registry checked against a hand-kept
    list would pass while the code produced something else — which is the shape
    of half the instrument defects in this programme.
    """
    seen = set()
    original = R.UNRESOLVED_ROLE_DEFAULT
    try:
        for variant in R.UNRESOLVED_ROLE_VARIANTS:
            R.UNRESOLVED_ROLE_DEFAULT = variant
            for spec in (_Spec(), _SpecWithFilter()):
                for facet in R._split_named_dimension_roles([_grouping()], spec):
                    if facet.kind != R.KIND_GROUPING:
                        seen.add(facet.kind)
    finally:
        R.UNRESOLVED_ROLE_DEFAULT = original
    return seen


# --------------------------------------------------------------------------- #
# The assertion
# --------------------------------------------------------------------------- #
def test_every_move_the_code_performs_is_registered():
    """A new reclassification target must be declared before it can ship."""
    unregistered = _observed_targets() - set(R.RECLASSIFICATION_TARGETS)
    assert not unregistered, (
        "these kinds are produced by reclassification but are not in "
        "RECLASSIFICATION_TARGETS, so nothing checks that a reconciler can "
        "receive them: %s" % sorted(unregistered))


def test_every_registered_target_has_a_receiver_or_a_stated_reason():
    """The closed class. This is the test e35a01b would have failed."""
    matrix = SC.scan()["matrix"]
    missing = []
    for kind, exemption in R.RECLASSIFICATION_TARGETS.items():
        if exemption:
            continue
        cell = matrix["(point-in-time)"][kind]
        if cell["cell"] == "no-branch":
            missing.append(kind)
    assert not missing, (
        "reconcile_facets has no branch for these reclassification targets, so a "
        "facet moved into one keeps its default LOST status whatever execution "
        "did, and blocks: %s" % sorted(missing))


def test_an_exemption_must_say_why():
    for kind, exemption in R.RECLASSIFICATION_TARGETS.items():
        if exemption:
            assert len(exemption) > 40, (
                "%s is exempted from needing a receiver with no real reason "
                "given; an exemption without a reason is how the next one gets "
                "waved through" % kind)


# --------------------------------------------------------------------------- #
# Can-fail: the assertion must be able to catch the thing it exists for
# --------------------------------------------------------------------------- #
def test_the_assertion_catches_an_unregistered_target(monkeypatch):
    """Deregister KIND_POPULATION and the discovery test must fail.

    Without this, an `_observed_targets` that silently returned nothing would
    pass every test above and assert nothing at all.
    """
    monkeypatch.setattr(R, "RECLASSIFICATION_TARGETS", {})
    with pytest.raises(AssertionError):
        test_every_move_the_code_performs_is_registered()


def test_the_assertion_catches_a_target_with_no_receiver(monkeypatch):
    """Register a kind that has no branch and the closure test must fail.

    KIND_UNRESOLVED_MEASURE is the honest probe: it genuinely has no branch on
    either reconciler, by design, so requiring a receiver for it must fail.
    """
    monkeypatch.setattr(R, "RECLASSIFICATION_TARGETS",
                        {R.KIND_UNRESOLVED_MEASURE: ""})
    with pytest.raises(AssertionError):
        test_every_registered_target_has_a_receiver_or_a_stated_reason()


def test_the_discovery_actually_finds_the_moves():
    """`_observed_targets` returning an empty set would make everything pass."""
    targets = _observed_targets()
    assert R.KIND_POPULATION in targets
    assert R.KIND_UNRESOLVED_ROLE in targets

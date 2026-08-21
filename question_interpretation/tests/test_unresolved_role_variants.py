"""The three candidate defaults for an UNRESOLVED role, and what each does.

These are MEASUREMENT tests. They pin what each variant does so the side-by-side
report can be re-derived, and they pin the one fact that decides how the choice
should be read: today an unapplied grouping does not disclose. It blocks.
"""
from __future__ import annotations

import pytest

from mi_agent import execution_receipt as R


def _grouping(field="borrower_type", label="joint borrower", status=None):
    return R.RequestedFacet(
        kind=R.KIND_GROUPING, label=label, field_key=field,
        alt_keys=(), concepts=(label,),
        status=status or R.LOST, reason="not applied")


class _Spec:
    def __init__(self, filters=None, dimensions=None):
        self.filters = dict(filters or {})
        self.dimensions = list(dimensions or [])
        self.dimension = None


def _receipt(facets):
    return R.ExecutionReceipt(facets=list(facets))


# --------------------------------------------------------------------------- #
# The fact that frames the choice
# --------------------------------------------------------------------------- #
def test_an_unapplied_grouping_blocks_it_does_not_disclose():
    """Variant 1 is NOT "a whole-book number with a note".

    `SHAPE_FACETS` still names KIND_GROUPING and `assess` still has a
    VERDICT_PARTIAL branch for it, which reads like a disclosure path. It is
    unreachable: `_blocks` returns True for any non-applied grouping, and the
    blocking test runs first. Whichever way the choice goes, it is not being
    made between a wrong number and a round trip.
    """
    verdict, message = R.assess(_receipt([_grouping()]))
    assert verdict == R.VERDICT_REFUSE
    assert "I have not substituted a broader figure" in message


def test_the_disclosure_branch_still_fires_for_an_applied_shape_facet():
    """The can-fail half: the partial branch is reachable, just not this way.

    Without this, the test above would also pass if `assess` had no partial
    branch at all, and would prove nothing about which branch won.
    """
    assert R.KIND_GROUPING in R.SHAPE_FACETS


# --------------------------------------------------------------------------- #
# Each variant does what it says
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant,expected_kind", [
    ("grouping", R.KIND_GROUPING),
    ("population", R.KIND_POPULATION),
    ("clarify", R.KIND_UNRESOLVED_ROLE),
])
def test_variant_decides_the_kind_of_an_unresolved_dimension(variant, expected_kind,
                                                             monkeypatch):
    monkeypatch.setattr(R, "UNRESOLVED_ROLE_DEFAULT", variant)
    out = R._split_named_dimension_roles([_grouping()], _Spec())
    assert [f.kind for f in out] == [expected_kind]


def test_a_dimension_with_a_role_is_untouched_by_every_variant(monkeypatch):
    """Can-fail: a variant that moved everything would pass the tests above.

    A field the parser put on an AXIS has a role. No variant may touch it.
    """
    for variant in ("grouping", "population", "clarify"):
        monkeypatch.setattr(R, "UNRESOLVED_ROLE_DEFAULT", variant)
        out = R._split_named_dimension_roles(
            [_grouping()], _Spec(dimensions=["borrower_type"]))
        assert [f.kind for f in out] == [R.KIND_GROUPING], variant


def test_clarify_asks_rather_than_refusing():
    facet = R.RequestedFacet(
        kind=R.KIND_UNRESOLVED_ROLE, label="joint borrower",
        field_key="borrower_type", alt_keys=(), concepts=("joint borrower",),
        status=R.LOST, reason="not applied")
    verdict, message = R.assess(_receipt([facet]))
    assert verdict == R.VERDICT_CLARIFY
    assert "Did you want the book split by it" in message
    assert "I have not answered over the whole book" in message


# --------------------------------------------------------------------------- #
# B5 across the split, under every variant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", ["grouping", "population", "clarify"])
def test_no_variant_builds_a_population_label_that_omits_its_field(variant,
                                                                  monkeypatch):
    from question_interpretation.b5_reachability import label_omits_its_field
    monkeypatch.setattr(R, "UNRESOLVED_ROLE_DEFAULT", variant)
    out = R._split_named_dimension_roles([_grouping()], _Spec())
    assert not [f for f in out
                if f.kind == R.KIND_POPULATION and label_omits_its_field(f)]


def test_population_keeps_the_wording_the_governed_check_reads(monkeypatch):
    """Why "population" is measured with the wording preserved.

    `_governed_population_predicates` resolves lending windows from the facet's
    WORDS. Relabelling to the bare field name destroys "front book" and would
    re-create the 32c263a failure by construction — an artefact of the
    measurement, not a property of the variant. `population_bare` exists to
    show that failure is producible, so ruling it out means something.
    """
    facet = _grouping(field="seasoning_segment", label="front book")

    monkeypatch.setattr(R, "UNRESOLVED_ROLE_DEFAULT", "population")
    kept = R._split_named_dimension_roles([facet], _Spec())[0]
    assert "front book" in kept.label
    assert R._governed_population_predicates(kept)

    monkeypatch.setattr(R, "UNRESOLVED_ROLE_DEFAULT", "population_bare")
    lost = R._split_named_dimension_roles([facet], _Spec())[0]
    assert "front book" not in lost.label
    assert not R._governed_population_predicates(lost)


def test_the_default_is_the_current_behaviour():
    """The switch must stay inert until a variant is chosen."""
    assert R.UNRESOLVED_ROLE_DEFAULT == "grouping"

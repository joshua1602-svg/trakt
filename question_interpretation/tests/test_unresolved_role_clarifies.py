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
# What the settled rule does
# --------------------------------------------------------------------------- #
def test_an_unresolved_role_becomes_a_question():
    out = R._split_named_dimension_roles([_grouping()], _Spec())
    assert [f.kind for f in out] == [R.KIND_UNRESOLVED_ROLE]


def test_a_dimension_with_a_role_is_untouched():
    """Can-fail: a rule that moved everything would pass the test above.

    A field the parser put on an AXIS has a role and must be left alone; a field
    it put in FILTERS has one too, and becomes a population rather than a
    question.
    """
    axis = R._split_named_dimension_roles(
        [_grouping()], _Spec(dimensions=["borrower_type"]))
    assert [f.kind for f in axis] == [R.KIND_GROUPING]

    filtered = R._split_named_dimension_roles(
        [_grouping()], _Spec(filters={"borrower_type": "joint"}))
    assert [f.kind for f in filtered] == [R.KIND_POPULATION]


# --------------------------------------------------------------------------- #
# B5 across the split
# --------------------------------------------------------------------------- #
def test_the_split_builds_no_population_label_that_omits_its_field():
    from question_interpretation.b5_reachability import label_omits_its_field
    for spec in (_Spec(), _Spec(filters={"borrower_type": "joint"}),
                 _Spec(dimensions=["borrower_type"])):
        out = R._split_named_dimension_roles([_grouping()], spec)
        assert not [f for f in out
                    if f.kind == R.KIND_POPULATION and label_omits_its_field(f)]


def test_the_governed_check_reads_the_facet_s_WORDING_not_its_field():
    """Why a population facet must never be relabelled to a bare field name.

    `_governed_population_predicates` resolves lending windows from the facet's
    WORDS. This was measured through a fourth, deliberately-broken arm
    (`population_bare`) while the three variants were being compared; that arm
    is gone with the switch, so the property is asserted directly here. Losing
    it would re-create the 32c263a failure by construction.
    """
    named = R.RequestedFacet(
        kind=R.KIND_POPULATION, label="the population seasoning_segment: front book",
        field_key="seasoning_segment", alt_keys=(), concepts=("front book",),
        status=R.LOST, reason="")
    assert R._governed_population_predicates(named)

    bare = R.RequestedFacet(
        kind=R.KIND_POPULATION, label="the population seasoning_segment",
        field_key="seasoning_segment", alt_keys=(), concepts=(),
        status=R.LOST, reason="")
    assert not R._governed_population_predicates(bare)


def test_the_measurement_switch_is_gone():
    """One variant was chosen; the other two and the switch were deleted."""
    assert not hasattr(R, "UNRESOLVED_ROLE_DEFAULT")
    assert not hasattr(R, "UNRESOLVED_ROLE_VARIANTS")


# --------------------------------------------------------------------------- #
# The role must be read by EVERY key that satisfies the facet
# --------------------------------------------------------------------------- #
def test_an_axis_named_by_an_alt_key_still_counts_as_a_role():
    """The defect the three-variant measurement did not see.

    A generic term resolves to different concrete fields depending on what the
    book carries: "region" is `collateral_geography` on a real tape and
    `geographic_region_obligor` on the generated fixture, and the parser may put
    EITHER on the axis. Reading only `field_key` made a facet whose axis was
    named by an alt key look as though no source had given it a role, and
    "balance by borrower type by region" became a clarification instead of a
    two-dimensional answer — 26 tests, none of them on either measurement
    surface, because on the real tape the two names coincide.
    """
    facet = R.RequestedFacet(
        kind=R.KIND_GROUPING, label="region", field_key="collateral_geography",
        alt_keys=("geographic_region_obligor",), concepts=("region",),
        status=R.LOST, reason="")
    spec = _Spec(dimensions=["geographic_region_obligor"])
    assert [f.kind for f in R._split_named_dimension_roles([facet], spec)] \
        == [R.KIND_GROUPING]


def test_a_filter_named_by_an_alt_key_still_counts_as_a_role():
    """The same resolution on the other slot, and the predicate must be built
    on the key the parser actually filtered by — not on the facet's own."""
    facet = R.RequestedFacet(
        kind=R.KIND_GROUPING, label="region", field_key="collateral_geography",
        alt_keys=("geographic_region_obligor",), concepts=("region",),
        status=R.LOST, reason="")
    spec = _Spec(filters={"geographic_region_obligor": "London"})
    out = R._split_named_dimension_roles([facet], spec)
    assert [f.kind for f in out] == [R.KIND_POPULATION]
    assert out[0].field_key == "geographic_region_obligor"
    assert "geographic_region_obligor" in out[0].label


def test_an_unrelated_key_is_not_mistaken_for_a_role():
    """Can-fail: matching on ANY key would make every facet look resolved."""
    facet = R.RequestedFacet(
        kind=R.KIND_GROUPING, label="region", field_key="collateral_geography",
        alt_keys=("geographic_region_obligor",), concepts=("region",),
        status=R.LOST, reason="")
    spec = _Spec(dimensions=["borrower_type"], filters={"ltv_bucket": "60-70"})
    assert [f.kind for f in R._split_named_dimension_roles([facet], spec)] \
        == [R.KIND_UNRESOLVED_ROLE]


# --------------------------------------------------------------------------- #
# A clarification is only worth asking when answering it changes something
# --------------------------------------------------------------------------- #
def test_a_certain_refusal_beats_a_question():
    """Ordering, and the reason it is not arbitrary.

    "Which region contributes most to the weighted average LTV?" answered by a
    plain weighted-average spec must say the contribution was not computed. If
    the clarification ran first it would ask whether region meant a breakdown or
    a filter — a question whose answer changes nothing, in place of the P0
    message that names the actual obstacle. Six refusals were pre-empted this
    way before the order was corrected.
    """
    unresolved = R.RequestedFacet(
        kind=R.KIND_UNRESOLVED_ROLE, label="region", field_key="collateral_geography",
        alt_keys=(), concepts=("region",), status=R.LOST, reason="not applied")
    blocking = R.RequestedFacet(
        kind=R.KIND_CONTRIBUTION, label="the contribution to the portfolio "
        "weighted average", field_key="current_loan_to_value", alt_keys=(),
        concepts=(), status=R.LOST, reason="each group's own value was calculated")

    verdict, message = R.assess(_receipt([unresolved, blocking]))
    assert verdict == R.VERDICT_REFUSE
    assert "contribution to the portfolio weighted average" in message


def test_it_still_asks_when_nothing_else_blocks():
    """Can-fail: an order that never clarified would pass the test above."""
    unresolved = R.RequestedFacet(
        kind=R.KIND_UNRESOLVED_ROLE, label="region", field_key="collateral_geography",
        alt_keys=(), concepts=("region",), status=R.LOST, reason="not applied")
    verdict, _ = R.assess(_receipt([unresolved]))
    assert verdict == R.VERDICT_CLARIFY


def test_a_field_the_book_cannot_express_is_not_asked_about():
    """Its role is moot, and the unavailability is the useful reply.

    "How many joint borrowers are there" on a book with no borrower column can
    neither split nor narrow. Asking which was meant invites an answer that
    cannot be honoured either way; naming the absent field can be acted on.
    """
    facet = _grouping(field="borrower_type", label="joint borrower")
    semantics = {"fields": {"borrower_type": {"canonical_field": "borrower_type"}}}
    out = R._split_named_dimension_roles(
        [facet], _Spec(), semantics, available_columns={"current_outstanding_balance"})
    assert [f.kind for f in out] == [R.KIND_GROUPING]


def test_a_field_the_book_does_carry_is_still_asked_about():
    """Can-fail: a guard that suppressed every clarification would pass above."""
    facet = _grouping(field="borrower_type", label="joint borrower")
    semantics = {"fields": {"borrower_type": {"canonical_field": "borrower_type"}}}
    out = R._split_named_dimension_roles(
        [facet], _Spec(), semantics, available_columns={"borrower_type"})
    assert [f.kind for f in out] == [R.KIND_UNRESOLVED_ROLE]


def test_a_governed_seasoning_population_still_refuses_rather_than_asking():
    """`_blocks` claims an unresolved-role facet whose wording names a governed
    seasoning window, and the refusal names the population.

    Asserted because the RECLASSIFICATION_TARGETS exemption for this kind
    originally said `assess` reads it "before the blocking test and never
    consults its status". Both halves were wrong once the order was corrected,
    and the exemption's reasoning is what licenses this kind having no
    reconciler branch — so it has to be accurate.
    """
    facet = R.RequestedFacet(
        kind=R.KIND_UNRESOLVED_ROLE, label="front book",
        field_key="seasoning_segment", alt_keys=(), concepts=("front book",),
        status=R.LOST, reason="not applied")
    verdict, message = R.assess(_receipt([facet]))
    assert verdict == R.VERDICT_REFUSE
    assert "front book" in message

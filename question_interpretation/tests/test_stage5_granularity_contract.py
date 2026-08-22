"""Change 5 — a facet raisable for a request that SUCCEEDED, not only one that failed.

`granularity_disclosure` used to return a facet only on a MISMATCH, filed as
`KIND_GROUPING`, with `status=UNAVAILABLE` written at detection. Three
consequences, and they had to be fixed together:

  * a reporting LEVEL was described to the reader as a breakdown;
  * the status was decided before execution, so no reconciler ever weighed
    evidence — and in `mi_service` the facet was appended AFTER
    `reconcile_routed_facets`, which is why it had to be;
  * a grain the route DID honour raised nothing, so nothing downstream could act
    on a correct request.

A rule can only be enforced on a request that is represented. This is the change
that makes the time axis enforceable in the first place.
"""
from __future__ import annotations

import pytest

from mi_agent import execution_receipt as R


def _reconcilers():
    return ("point_in_time", "routed")


def _run(facet, which):
    if which == "point_in_time":
        class _Spec:
            filters, dimensions, dimension, aggregation = {}, [], None, None

        class _Result:
            metadata, data, result_type = {}, None, None

        return R.reconcile_facets([facet], spec=_Spec(), query_result=_Result(),
                                  semantics={"fields": {}},
                                  available_columns=None, route="geo_exposure")[0]
    return R.reconcile_routed_facets([facet], route="geo_exposure",
                                     semantics={"fields": {}},
                                     available_columns=None, envelope={})[0]


def _facet(asked, reported):
    return R.RequestedFacet(kind=R.KIND_GRANULARITY, label=asked,
                            concepts=(asked, reported))


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #
def test_the_detector_no_longer_decides_the_outcome():
    """The status must arrive at reconciliation undecided."""
    facet = R.granularity_disclosure("show exposure by postcode", "geo_exposure")
    assert facet is not None
    assert facet.kind == R.KIND_GRANULARITY
    assert facet.status == R.LOST          # the default, not a verdict
    assert not facet.reason


def test_a_grain_the_route_honours_raises_a_facet_at_all():
    """The half that did not exist. Without it there is nothing to enforce on."""
    facet = R.granularity_disclosure("exposure by ITL3 area", "geo_exposure")
    assert facet is not None
    assert facet.concepts == ("ITL3 area", "ITL3 area")


def test_a_question_naming_no_grain_still_raises_nothing():
    """Can-fail: raising one unconditionally would pass both tests above."""
    assert R.granularity_disclosure("geographic exposure", "geo_exposure") is None
    assert R.granularity_disclosure("balance by region", "point_in_time_kpi") is None


@pytest.mark.parametrize("which", _reconcilers())
def test_both_reconcilers_stamp_an_honoured_grain_applied(which):
    assert _run(_facet("ITL3 area", "ITL3 area"), which).status == R.APPLIED


@pytest.mark.parametrize("which", _reconcilers())
def test_both_reconcilers_stamp_a_substituted_grain_unsupported(which):
    out = _run(_facet("postcode", "ITL3 area"), which)
    assert out.status == R.UNSUPPORTED
    assert out.reason == "this answer is reported at ITL3 area level, not by postcode"


# --------------------------------------------------------------------------- #
# A substituted grain BLOCKS — it is not a shortfall
# --------------------------------------------------------------------------- #
def test_a_substituted_grain_refuses():
    """Honour-or-clarify. A different level measured and presented as the answer
    is the same shape as a period or a population that was not applied."""
    facet = _facet("postcode", "ITL3 area")
    facet.status, facet.reason = R.UNSUPPORTED, "reported at ITL3 area level"
    verdict, message = R.assess(R.ExecutionReceipt(facets=[facet]))
    assert verdict == R.VERDICT_REFUSE
    assert "postcode" in message


def test_an_honoured_grain_does_not_refuse():
    """Can-fail: a kind that always blocked would pass the test above."""
    facet = _facet("ITL3 area", "ITL3 area")
    facet.status, facet.reason = R.APPLIED, ""
    assert R.assess(R.ExecutionReceipt(facets=[facet]))[0] == R.VERDICT_OK


# --------------------------------------------------------------------------- #
# The kind is not a dimension
# --------------------------------------------------------------------------- #
def test_a_grain_carries_no_field_key():
    """It is a reporting LEVEL, not a registry field. There is no `postcode`
    column to be applied or unavailable, which is why filing it as a grouping
    made the receipt describe it as a breakdown."""
    facet = R.granularity_disclosure("show exposure by postcode", "geo_exposure")
    assert facet.field_key is None
    assert facet.kind != R.KIND_GROUPING

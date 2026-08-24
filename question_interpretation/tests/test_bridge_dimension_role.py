"""The interpretation contract carries the parser's bridge attribution dimension.

On a bridge question the parser already resolves the attribution axis into
`spec.bridge_dimension` (a governed field key, populated for bridge questions
only). Before this fix the projection emitted that claim's role as `unresolved`,
so a fact the parser had settled was lost downstream. The role split now projects
it as `grouping` — projection, not reinterpretation:

  * the trigger is the existing parser field `spec.bridge_dimension`, not a
    reread of the question, a phrase list, or a route-name check;
  * the match is governed-key to governed-key;
  * it fires for exactly the one claim whose key equals `spec.bridge_dimension`,
    and changes no other claim's role.

The negative tests are the point of the fix as much as the positive ones: a
dimension is only promoted when the PARSER put it in `bridge_dimension`, never
because a question merely contains a dimension.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mi_agent.mi_query_validator import load_mi_semantics
from question_interpretation import projection
from question_interpretation.schema import GROUPING, UNRESOLVED_ROLE, FILTER

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_REPO / "mi_agent" / "mi_semantics_field_registry.yaml")


def _roles(semantics, question):
    qi = projection.project(question, semantics=semantics)
    return {d.candidate_concept: (d.role, d.source) for d in qi.dimensions}


# --------------------------------------------------------------------------- #
# Positive: the bridge attribution dimension is carried as a grouping
# --------------------------------------------------------------------------- #
def test_bridge_by_region_carries_the_grouping_role(semantics):
    roles = _roles(semantics, "Funded balance bridge by region")
    assert roles["collateral_geography"] == (GROUPING, "parser.bridge_dimension")


def test_bridge_by_product_carries_the_grouping_role(semantics):
    roles = _roles(semantics, "Bridge the funded balance by product")
    assert roles["erm_product_type"] == (GROUPING, "parser.bridge_dimension")


def test_bridge_by_ltv_band_carries_the_grouping_role(semantics):
    roles = _roles(semantics, "balance bridge by LTV band")
    assert roles["ltv_bucket"] == (GROUPING, "parser.bridge_dimension")


def test_only_the_bridge_dimension_moves_not_a_companion(semantics):
    """The whole reason `unresolved` existed: do not promote every dimension.

    'by region for joint borrowers' — region is the attribution axis, borrower
    type is not the bridge dimension, so only region becomes a grouping and
    borrower type keeps its existing (unresolved) role.
    """
    roles = _roles(semantics, "Bridge the funded balance by region for joint borrowers")
    assert roles["collateral_geography"] == (GROUPING, "parser.bridge_dimension")
    assert roles["borrower_type"][0] == UNRESOLVED_ROLE
    assert roles["borrower_type"][1] != "parser.bridge_dimension"


# --------------------------------------------------------------------------- #
# Negative / zero-blast: nothing else acquires the role
# --------------------------------------------------------------------------- #
def test_ordinary_grouping_is_unchanged(semantics):
    """`spec.bridge_dimension` is None here — the parser puts the dimension in
    `spec.dimension`, so it is a grouping via the ordinary source, unchanged."""
    roles = _roles(semantics, "Show funded balance by region")
    assert roles["collateral_geography"] == (GROUPING, "parser.dimension")


def test_trend_grouping_does_not_acquire_bridge_semantics(semantics):
    """No `bridge_dimension`, so a dimension that reaches the projection only
    through the facet layer stays `unresolved` — it is NOT promoted merely
    because the question contains a dimension."""
    roles = _roles(semantics, "Show funded balance over time by region")
    assert roles["collateral_geography"][0] == UNRESOLVED_ROLE


def test_two_dimensions_without_a_bridge_keep_their_own_roles(semantics):
    """No role is guessed from ordering; both are parser groupings, unchanged."""
    roles = _roles(semantics, "Balance by region and ticket size")
    assert roles["collateral_geography"] == (GROUPING, "parser.dimension")
    assert roles["ticket_bucket"] == (GROUPING, "parser.dimension")


def test_bridge_with_no_resolved_bridge_dimension_stays_as_it_was(semantics):
    """A bridge phrasing that names no dimension resolves no `bridge_dimension`,
    so nothing is promoted — absence is not turned into inference."""
    qi = projection.project("funded balance bridge", semantics=semantics)
    # No dimension claim is promoted to grouping via the bridge source.
    assert not any(d.source == "parser.bridge_dimension" for d in qi.dimensions)

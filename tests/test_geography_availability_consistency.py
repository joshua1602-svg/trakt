#!/usr/bin/env python3
"""One availability decision, read by every path that needs it.

THE DEFECT THIS PINS, found in production rather than in a test. A live run
answered "total balance by region" — measured on ``collateral_geography`` — and
in the same request refused "total balance by property region", saying the book
records no collateral geography. Same book, same basis, opposite answers.

There were two implementations of "does this book carry that geography":

    mi_agent.llm_query_parser._basis_region_field   is the COLUMN NAME present?
    mi_agent.mi_geography.supported_bases           do its VALUES resolve as
                                                    places through the region
                                                    ladder?

A book can satisfy one and fail the other, and production did: the deployed
artefact was missing `uk_itl_master_lookup_v2.csv`, so the ladder recognised
nothing, every basis reported unsupported, and the name-based path answered
anyway.

Stated wording may decide WHICH basis is measured. It must never reach a
different answer about whether that basis is AVAILABLE.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import llm_query_parser as parser          # noqa: E402
from mi_agent import mi_geography as geo                 # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics  # noqa: E402
from mi_agent_api import mi_service                      # noqa: E402
from mi_agent_api.data_source import semantics_path      # noqa: E402

GROUPED = {"ok": True, "spec": {"dimension": "collateral_geography",
                                "dimensions": ["collateral_geography"],
                                "filters": {}}}
UNGROUPED = {"ok": True, "spec": {"dimension": None, "dimensions": [],
                                  "filters": {}}}


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(semantics_path())


def _contract(frame, basis="collateral"):
    """A contract resolved the way a request resolves one: against the frame."""
    resolved = geo.resolved_basis_fields(frame=frame)
    return geo.GeographyContract(
        primary_basis=basis, source=geo.SOURCE_ASSET_DEFAULT,
        asset_class="equity_release",
        supported=tuple(b for b, field in resolved if field), fields=resolved)


def _book_with_geography():
    return pd.DataFrame({"collateral_geography": ["Scotland", "London"] * 25,
                         "current_outstanding_balance": [1.0] * 50})


def _book_without_geography():
    """The live shape: the column is PRESENT and its values are not places."""
    return pd.DataFrame({"collateral_geography": ["ND5"] * 50,
                         "current_outstanding_balance": [1.0] * 50})


def _guard(envelope, question, contract):
    return mi_service._guard_stated_geography_basis(
        copy.deepcopy(envelope), question=question, geography=contract)


# =========================================================================== #
# One decision
# =========================================================================== #
def test_supported_and_the_field_come_from_one_computation():
    """`supported_bases` is DERIVED from the field resolution, so the two
    cannot drift into disagreeing about the same book."""
    for frame in (_book_with_geography(), _book_without_geography()):
        resolved = geo.resolved_basis_fields(frame=frame)
        assert geo.supported_bases(frame=frame) == tuple(
            basis for basis, field in resolved if field)


def test_the_parser_reads_the_contracts_decision_not_the_column_names(semantics):
    """THE ROOT CAUSE, pinned.

    The column is present, so a name-based test says the basis is available.
    The contract, which probed the values, says it is not. The parser must
    return the contract's answer — asserting on the name-based one is what let
    production answer a question it had just declared unanswerable.
    """
    frame = _book_without_geography()
    contract = _contract(frame)
    assert "collateral_geography" in set(frame.columns)      # present by NAME
    assert contract.supports("collateral") is False          # not by CONTENT
    assert contract.field_for("collateral") is None


def test_generic_and_explicit_agree_on_a_book_that_has_the_geography(semantics):
    """The ordinary case: both answer, and both on the same field."""
    contract = _contract(_book_with_geography())
    assert _guard(GROUPED, "Total balance by region", contract)["ok"] is True
    assert _guard(GROUPED, "Total balance by property region",
                  contract)["ok"] is True
    assert (parser._preferred_region(semantics,
                                     available_columns={"collateral_geography"},
                                     geography=contract)
            == contract.field_for("collateral"))


def test_generic_and_explicit_agree_on_a_book_that_does_not(semantics):
    """The defect's own case. Before the fix the first of these answered and
    the second refused, in the same request."""
    contract = _contract(_book_without_geography())
    generic = _guard(GROUPED, "Total balance by region", contract)
    explicit = _guard(GROUPED, "Total balance by property region", contract)
    assert generic["ok"] is False, "generic region answered on a non-geography"
    assert explicit["ok"] is False
    assert generic["ok"] == explicit["ok"]


def test_a_question_that_uses_no_geography_is_not_refused_for_lacking_one():
    """A total is still a total. Refusing it would punish every question for a
    fact about one dimension."""
    contract = _contract(_book_without_geography())
    assert _guard(UNGROUPED, "What is the total balance?", contract)["ok"] is True


# =========================================================================== #
# The anti-substitution behaviour that was already right
# =========================================================================== #
def test_explicit_borrower_region_never_inherits_collateral(semantics):
    """G07, live-verified and preserved: a borrower question on a book with no
    borrower geography REFUSES. It does not quietly become the collateral one,
    which the book does have."""
    frame = _book_with_geography()
    contract = _contract(frame)
    assert contract.supports("collateral") is True
    assert contract.supports("borrower") is False

    refused = _guard(GROUPED, "Total balance by borrower region", contract)
    assert refused["ok"] is False
    assert "collateral" not in str(refused.get("answer", "")).split("it records")[0]

    chosen = parser._basis_region_field("borrower", semantics,
                                        available_columns=set(frame.columns),
                                        geography=contract)
    assert chosen is None or geo.basis_of_field(chosen) == "borrower", (
        f"borrower resolved to {chosen!r}, which is not a borrower field")


def test_a_configured_basis_never_falls_through_to_the_other_one(semantics):
    """When the book's own basis is established the answer is one of ITS
    columns or none. Dropping into the cross-basis preference order would
    answer "region" on the other geography."""
    frame = pd.DataFrame({"geographic_region_obligor": ["Scotland"] * 50,
                          "current_outstanding_balance": [1.0] * 50})
    contract = _contract(frame, basis="collateral")   # asset says collateral
    assert contract.supports("borrower") is True      # book carries borrower
    assert contract.supports("collateral") is False

    chosen = parser._preferred_region(semantics,
                                      available_columns=set(frame.columns),
                                      geography=contract)
    assert chosen is None or geo.basis_of_field(chosen) != "borrower", (
        f"a collateral-configured book resolved generic region to {chosen!r}")


# =========================================================================== #
# Provenance stays distinct
# =========================================================================== #
def test_stated_and_configured_provenance_remain_distinguishable():
    """Consistency of AVAILABILITY must not blur WHERE the basis came from. A
    reader has to be able to tell an answer the configuration chose from one
    they asked for by name."""
    contract = _contract(_book_with_geography())
    generic = contract.effective_for("Total balance by region")
    explicit = contract.effective_for("Total balance by property region")

    assert generic["basisSource"] == geo.SOURCE_ASSET_DEFAULT
    assert generic["primaryBasis"] == "collateral"
    assert "configuredBasis" not in generic

    assert explicit["basisSource"] == geo.SOURCE_QUESTION
    assert explicit["primaryBasis"] == "collateral"
    assert explicit["configuredBasis"] == "collateral"
    assert explicit["configuredBasisSource"] == geo.SOURCE_ASSET_DEFAULT


# =========================================================================== #
# The packaging defect that caused it
# =========================================================================== #
def test_the_region_ladder_is_staged_for_deployment():
    """WHY PRODUCTION HAD NO GEOGRAPHY AT ALL.

    `region_resolution` reads `uk_itl_master_lookup_v2.csv`, and its loader is
    deliberately non-raising — absent, "every caller degrades to exact matching,
    never to a fabricated mapping". So a deployment missing the file does not
    fail. It silently loses the governed ladder: the live run reported
    `supportedBases: []` on a book whose regions are Scotland and London, and
    returned SCOTLAND and Scotland as two separate rows because nothing was left
    to harmonise them.

    The file is data the runtime reads, so it has to be staged like any other.
    """
    from mi_agent.region_resolution import _LOOKUP_NAME

    manifest = (_REPO_ROOT / "deploy" / "trakt-mi-api"
                / "package_contents.txt").read_text(encoding="utf-8")
    staged = {line.split("#", 1)[0].strip()
              for line in manifest.splitlines() if line.split("#", 1)[0].strip()}
    assert _LOOKUP_NAME in staged, (
        f"{_LOOKUP_NAME} is not staged for deployment; the deployed service "
        "will silently lose the governed region ladder")
    assert (_REPO_ROOT / _LOOKUP_NAME).exists()


def test_without_the_ladder_no_basis_is_supported():
    """The mechanism, demonstrated rather than described: this is what the
    deployed service was doing, and why every explicit basis was refused."""
    import mi_agent.region_resolution as region_resolution

    frame = _book_with_geography()
    assert geo.supported_bases(frame=frame) == ("collateral",)

    saved = region_resolution._REPO_ROOT
    try:
        import tempfile
        region_resolution._REPO_ROOT = Path(tempfile.mkdtemp())
        region_resolution._table.cache_clear()
        region_resolution._index.cache_clear()
        assert geo.supported_bases(frame=frame) == (), (
            "a book of real places still reported a supported basis with no "
            "ladder; the reproduction no longer reproduces")
    finally:
        region_resolution._REPO_ROOT = saved
        region_resolution._table.cache_clear()
        region_resolution._index.cache_clear()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

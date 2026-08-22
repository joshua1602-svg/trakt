"""A region term reaches the rows whatever representation the book stores.

The defect this pins
--------------------
"exposure to London" could not be answered on a book whose region column holds
``TLI43``. Not for want of London exposure — the demonstration book carries
1,380 London loans — but because the value was matched literally, matched
nothing, and the filter was dropped.

The framing that made this look like a choice between a code field and a name
field was wrong. Raw loan tapes do not generally carry ITL codes at all: they
carry a partial postcode, a county, or the lender's own grouping, and the code is
DERIVED by the canonical transformation from ``uk_itl_master_lookup_v2.csv``.
So the dimension must accept whatever the user says and resolve it through the
mapping that transformation already produced.

Commercially this is the whole slice-and-dice capability on client data: no
client will ever type a statistical code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mi_agent import region_resolution as rr  # noqa: E402

#: What a book might actually store in its region column.
CODES = ["TLI43", "TLI44", "TLJ25", "TLM50", "TLE31"]
NAMES = ["London", "South East", "Scotland", "Yorkshire and The Humber"]
POSTCODES = ["SE1 4AA", "SE15 2BB", "GU1 1AA", "AB10 7XX"]


@pytest.mark.parametrize("term", [
    "London", "Greater London", "london", "  LONDON  ",
])
def test_common_names_and_aliases_reach_the_code_column(term):
    assert "TLI43" in rr.resolve(term, CODES)


def test_a_code_typed_verbatim_resolves_to_itself_only():
    assert rr.resolve("TLI43", CODES) == ["TLI43"]


def test_a_postcode_district_resolves_through_the_governed_mapping():
    """No client types a statistical code; plenty would type a postcode."""
    assert rr.resolve("SE1", CODES) == ["TLI44"]


def test_a_postcode_reaches_a_column_that_stores_postcodes():
    got = rr.resolve("SE1", POSTCODES)
    assert "SE1 4AA" in got and "SE15 2BB" in got
    assert "GU1 1AA" not in got


@pytest.mark.parametrize("term", ["South East", "the south east", "SOUTH EAST"])
def test_definite_articles_and_case_do_not_change_the_answer(term):
    """Near-identical phrasings handled differently is worse than either choice."""
    assert rr.resolve(term, NAMES) == ["South East"]


def test_a_name_column_is_matched_directly_without_going_through_codes():
    assert rr.resolve("Scotland", NAMES) == ["Scotland"]


def test_yorkshire_reaches_the_longer_official_name():
    assert rr.resolve("Yorkshire", NAMES) == ["Yorkshire and The Humber"]


def test_an_unknown_term_resolves_to_nothing_and_fabricates_nothing():
    assert rr.resolve("Narnia", CODES) == []
    assert rr.resolve("Narnia", NAMES) == []
    assert not rr.looks_like_region_term("Narnia")


def test_an_empty_term_is_not_treated_as_a_wildcard():
    assert rr.resolve("", CODES) == []
    assert rr.resolve(None, CODES) == []


def test_a_resolution_is_always_describable():
    """A resolution the reader cannot see is a substitution."""
    resolved = rr.resolve("London", CODES)
    text = rr.describe("London", resolved)
    assert "London" in text and "governed UK ITL mapping" in text
    assert "no matching value" in rr.describe("Narnia", [])


def test_a_broad_term_denotes_many_codes_and_a_narrow_one_denotes_itself():
    assert len(rr.codes_for("London")) > 1
    assert rr.codes_for("TLI43") == {"TLI43"}


@pytest.fixture(scope="module")
def books():
    """The committed demonstration tape, prepared exactly as the MI path does."""
    pd = pytest.importorskip("pandas")
    tape = (_REPO / "demo_platform" / "workspace" / "store" / "processed"
            / "platform" / "alderbridge" / "2026-06-30"
            / "platform_canonical_typed.csv")
    if not tape.exists():
        pytest.skip("demonstration tape not materialised in this tree")
    from mi_agent_api.funded_prep import prepare_funded_mi_dataset
    prepared, _ = prepare_funded_mi_dataset(pd.read_csv(tape, low_memory=False))
    return prepared


class TestAgainstARealBook:
    """The end-to-end property, on the committed demonstration tape."""

    def test_the_mapping_reproduces_the_name_based_answer_exactly(self, books):
        """The strongest available check: two independent representations of the
        same book must give the same row count for the same term."""
        by_name = books[books["collateral_geography"] == "London"]
        codes = rr.resolve("London", books["geographic_region_obligor"]
                           .dropna().unique().tolist())
        by_code = books[books["geographic_region_obligor"].isin(codes)]
        assert len(by_name) == len(by_code) > 0

    def test_the_filter_applies_on_a_book_that_stores_codes_only(self, books):
        from mi_agent.mi_query_executor import _apply_filters
        from mi_agent.mi_query_spec import MIQuerySpec
        from mi_agent.mi_query_validator import load_mi_semantics

        semantics = load_mi_semantics(
            str(_REPO / "mi_agent" / "mi_semantics_field_registry.yaml"))
        codes_only = books.drop(columns=["collateral_geography"])
        warnings = []
        out = _apply_filters(
            codes_only.copy(),
            MIQuerySpec(metric="current_outstanding_balance",
                        filters={"geographic_region_obligor": "London"}),
            semantics, warnings)
        assert 0 < len(out) < len(codes_only)
        assert any("ITL mapping" in w for w in warnings), \
            "the resolution must appear in the receipt"

    def test_an_unmatched_term_yields_no_rows_rather_than_a_dropped_filter(self, books):
        from mi_agent.mi_query_executor import _apply_filters
        from mi_agent.mi_query_spec import MIQuerySpec
        from mi_agent.mi_query_validator import load_mi_semantics

        semantics = load_mi_semantics(
            str(_REPO / "mi_agent" / "mi_semantics_field_registry.yaml"))
        out = _apply_filters(
            books.copy(),
            MIQuerySpec(metric="current_outstanding_balance",
                        filters={"collateral_geography": "Narnia"}),
            semantics, [])
        assert len(out) == 0, "a term that reaches nothing must not broaden to everything"

#!/usr/bin/env python3
"""The geography contract, asserted end to end through the governed MI service.

    "balance by region"           the book's CONFIGURED primary geography
    "balance by borrower region"  the borrower's geography, always
    "balance by property region"  the collateral's geography, always
    a stated basis the book lacks REFUSED, never silently substituted

WHY THE FIRST LINE IS THE HARD ONE. A loan has a borrower's region and a
property's region. They are different facts, they disagree on any buy-to-let and
on every auto loan, and an unqualified "by region" does not say which is meant.
Three separate layers used to answer that by taking whichever geography column
was populated first — the preparation layer gap-filled the borrower column from
the property's, the harmonisation layer resolved from whichever source led its
list, and the parser bound the head of a fixed preference order. None of that is
a semantics; it is the absence of one, and its output was a borrower region
column full of collateral geography.

What decides now is a property of the ASSET: a lifetime mortgage's regional
concentration is a concentration of houses, an auto book's is a concentration of
people. Onboarding establishes the asset class, the asset class carries a
governed default basis, the portfolio registry records it, and MI reads it. See
``mi_agent/mi_geography.py`` and ``tests/test_mi_geography_basis_lifecycle.py``.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import mi_geography as geo

BORROWER_COL = "geographic_region_obligor"
COLLATERAL_COL = "collateral_geography"

#: Real ITL3 codes, so the governed region ladder recognises them as places —
#: which is how MI tells a geography from a value that merely occupies the
#: column. Hartlepool/Stockton, Bournemouth, Aberdeen.
_CODES = ["TLC31", "TLK25", "TLM50", "TLC31", "TLK25"]
_NAMES = ["London", "South West", "Scotland", "London", "South West"]
_BALANCES = [100000.0, 200000.0, 300000.0, 400000.0, 500000.0]


def _write_tape(path: Path, *, borrower, collateral: bool) -> pd.DataFrame:
    frame = pd.DataFrame({
        "loan_identifier": [f"L{i}" for i in range(5)],
        "current_outstanding_balance": _BALANCES,
        "current_loan_to_value": [40.0, 45.0, 50.0, 55.0, 60.0],
        "source_portfolio_id": ["book_001"] * 5,
        "source_portfolio_type": ["direct"] * 5,
        "reporting_date": ["2026-06-30"] * 5,
    })
    if borrower == "declared_only":
        # Populated on every row, and not with a geography: the shape a
        # regulatory no-data declaration takes. Deliberately not spelled with a
        # regulatory code here — MI must reach the same conclusion for ANY value
        # its region owner does not recognise as a place, and encoding one
        # regulatory vocabulary in an MI test would be the first copy of it.
        frame[BORROWER_COL] = ["NOT COLLECTED"] * 5
    elif borrower:
        frame[BORROWER_COL] = _CODES
    if collateral:
        frame[COLLATERAL_COL] = _NAMES
    frame.to_csv(path, index=False)
    return frame


def _write_registry(path: Path, asset_class: str) -> None:
    path.write_text(yaml.safe_dump({"portfolios": [
        {"source_portfolio_id": "book_001", "asset_class": asset_class,
         geo.REGISTRY_KEY: {geo.REGISTRY_BASIS_KEY:
                            geo.default_primary_basis(asset_class)}},
    ]}), encoding="utf-8")


def _ask(question: str, *, asset_class: str = "equity_release",
         borrower=True, collateral: bool = True):
    from trakt_core.context import ExecutionContext
    from mi_agent_api import data_source
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    root = Path(tempfile.mkdtemp())
    tape = root / "tape.csv"
    registry = root / "portfolio_registry.yaml"
    _write_tape(tape, borrower=borrower, collateral=collateral)
    _write_registry(registry, asset_class)

    saved = {k: os.environ.get(k) for k in
             ("MI_AGENT_DATA_CSV", "MI_AGENT_DATA_CACHE_TTL",
              "TRAKT_RUNTIME_MODE", "MI_AGENT_AUTH_ENABLED",
              "TRAKT_PORTFOLIO_REGISTRY")}
    os.environ.update({
        "MI_AGENT_DATA_CSV": str(tape),
        "MI_AGENT_DATA_CACHE_TTL": "0",
        "TRAKT_RUNTIME_MODE": "test",
        "MI_AGENT_AUTH_ENABLED": "false",
        "TRAKT_PORTFOLIO_REGISTRY": str(registry),
    })
    data_source.reset_cache()
    try:
        result = execute_governed_mi_query(
            MiQueryRequest(question=question),
            context=ExecutionContext.for_internal("ERE"))
    finally:
        data_source.reset_cache()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return (result.result if not isinstance(result, dict) else result) or {}


def _region_fields(envelope) -> set:
    spec = envelope.get("spec") or {}
    keys = set(spec.get("filters") or {})
    for key in ("dimension", "x"):
        if spec.get(key):
            keys.add(spec[key])
    for key in (spec.get("dimensions") or []):
        keys.add(key)
    return {k for k in keys if geo.basis_of_field(k)}


class TestGenericRegionMeansTheConfiguredBasis(unittest.TestCase):

    def test_a_collateral_book_measures_region_on_the_property(self):
        envelope = _ask("balance by region", asset_class="equity_release")
        self.assertTrue(envelope.get("ok"), envelope.get("error"))
        self.assertEqual(_region_fields(envelope), {COLLATERAL_COL})

    def test_a_borrower_book_measures_region_on_the_obligor(self):
        envelope = _ask("balance by region", asset_class="auto_finance")
        self.assertTrue(envelope.get("ok"), envelope.get("error"))
        self.assertEqual(_region_fields(envelope), {BORROWER_COL})

    def test_the_same_question_on_two_books_reads_two_different_columns(self):
        """The whole architecture in one assertion. Identical question, identical
        data, different asset class — different answer, correctly."""
        collateral = _region_fields(_ask("balance by region",
                                         asset_class="equity_release"))
        borrower = _region_fields(_ask("balance by region",
                                       asset_class="auto_finance"))
        self.assertNotEqual(collateral, borrower)

    def test_a_named_region_narrows_on_the_configured_basis(self):
        """Generic VALUE language too, not just the axis: "the Scottish balance"
        states no basis and is measured on the book's own."""
        envelope = _ask("what is the Scottish balance", asset_class="equity_release")
        self.assertTrue(envelope.get("ok"), envelope.get("error"))
        self.assertEqual(_region_fields(envelope), {COLLATERAL_COL})


class TestAStatedBasisIsHonouredWhicheverTheBookReportsOn(unittest.TestCase):

    def test_borrower_region_reads_the_borrower_on_a_collateral_book(self):
        envelope = _ask("balance by borrower region", asset_class="equity_release")
        self.assertTrue(envelope.get("ok"), envelope.get("error"))
        self.assertEqual(_region_fields(envelope), {BORROWER_COL})

    def test_property_region_reads_the_collateral_on_a_borrower_book(self):
        envelope = _ask("balance by property region", asset_class="auto_finance")
        self.assertTrue(envelope.get("ok"), envelope.get("error"))
        self.assertEqual(_region_fields(envelope), {COLLATERAL_COL})

    def test_obligor_region_is_the_same_statement_as_borrower_region(self):
        envelope = _ask("balance by obligor region", asset_class="equity_release")
        self.assertEqual(_region_fields(envelope), {BORROWER_COL})


class TestAStatedBasisTheBookLacksIsRefused(unittest.TestCase):
    """Never a silent substitution. An answer labelled "borrower region" that was
    measured on collateral is worse than no answer, because the reader cannot
    tell — so the two honest outcomes are that geography or none."""

    def _refused(self, envelope):
        self.assertFalse(envelope.get("ok"),
                         "answered a basis the book does not record")
        self.assertEqual(envelope.get("artifacts") or [], [])
        return str(envelope.get("answer") or envelope.get("error") or "").lower()

    def test_a_book_with_no_borrower_geography_refuses_the_borrower_question(self):
        envelope = _ask("balance by borrower region", asset_class="equity_release",
                        borrower=False)
        text = self._refused(envelope)
        self.assertIn("not available in this dataset", text)
        self.assertIn("no value was fabricated", text)

    def test_the_refusal_does_not_quietly_substitute_the_other_geography(self):
        envelope = _ask("balance by borrower region", asset_class="equity_release",
                        borrower=False)
        self._refused(envelope)
        self.assertNotIn(COLLATERAL_COL, _region_fields(envelope))

    def test_a_book_with_no_collateral_geography_refuses_the_property_question(self):
        envelope = _ask("balance by property region", asset_class="auto_finance",
                        collateral=False)
        self._refused(envelope)
        self.assertNotIn(BORROWER_COL, _region_fields(envelope))

    def test_a_column_that_records_no_geography_refuses_rather_than_grouping_it(self):
        """THE CASE ONLY THE CONTRACT CATCHES. A regulatory geography field can
        be populated on every row with a value declaring the geography was never
        collected. The column is PRESENT, so nothing about column availability
        objects, and the executor cheerfully returns one bar labelled with the
        declaration — five loans, one "region", a number that is right and a
        category that is a fiction.

        The book is asked instead whether the column carries a PLACE, using the
        governed region ladder that already answers that question everywhere else
        in MI. It does not, so the borrower basis is not supported and the stated
        question is refused."""
        envelope = _ask("balance by borrower region", asset_class="equity_release",
                        borrower="declared_only")
        text = self._refused(envelope)
        self.assertTrue(envelope.get("controlledRefusal"))
        self.assertEqual((envelope.get("metadata") or {}).get("refusalClass"),
                         "DATA_UNAVAILABLE")
        self.assertIn("borrower", text)
        # And it names what the book DOES hold rather than quietly using it.
        self.assertIn("collateral", text)
        self.assertNotIn(COLLATERAL_COL, _region_fields(envelope))

    def test_a_generic_region_question_still_answers_on_the_same_book(self):
        """Only the STATED basis is refused. The book still has a geography and
        still answers the question that does not name one."""
        envelope = _ask("balance by region", asset_class="equity_release",
                        borrower=False)
        self.assertTrue(envelope.get("ok"), envelope.get("error"))
        self.assertEqual(_region_fields(envelope), {COLLATERAL_COL})


class TestTheAnswerDisclosesWhatItWasMeasuredOn(unittest.TestCase):

    def test_the_envelope_publishes_the_geography_contract(self):
        envelope = _ask("balance by region", asset_class="equity_release")
        block = (envelope.get("metadata") or {}).get("geographyBasis") or {}
        self.assertEqual(block.get("primaryBasis"), geo.BASIS_COLLATERAL)
        self.assertEqual(block.get("basisSource"), geo.SOURCE_PORTFOLIO)
        self.assertEqual(sorted(block.get("supportedBases") or []),
                         [geo.BASIS_BORROWER, geo.BASIS_COLLATERAL])

    def test_it_is_published_on_a_question_that_never_mentioned_geography(self):
        """A reader comparing two answers needs to know they were measured on the
        same basis, and the answer that did not mention geography is exactly the
        one where that is easy to get wrong."""
        envelope = _ask("what is the total balance", asset_class="auto_finance")
        block = (envelope.get("metadata") or {}).get("geographyBasis") or {}
        self.assertEqual(block.get("primaryBasis"), geo.BASIS_BORROWER)


if __name__ == "__main__":
    raise SystemExit(unittest.main())

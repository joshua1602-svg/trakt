"""A filter value is checked against the values the book records (twins run
2026-10-01).

"Show loan count by LTV bucket for lifetime mortgages" and "What is the
balance of active loans?" were read correctly and declined as "could not be
produced reliably": the filter's value is the governed registry's
(`lifetime_mortgage`, `Active`) and the production book records neither, so
the execution matched nothing and the reconciliation withheld it. That is a
fact about the book, and it is now said as one, naming the values the book
does record — and a value the book merely spells differently is matched.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mi_agent import plan_decline as decline
from mi_agent import plan_runtime_adapter as adapter
from mi_agent.mi_query_spec import MIQuerySpec
from tests.interpretation_v2.test_funded_breadth import (  # noqa: F401
    _intent, _served, book, semantics)

_BALANCE = [{"concept": "current_outstanding_balance", "statistic": "sum"}]


def _status(value):
    return [{"concept": "account_status", "comparator": "eq", "value": value}]


def _spec(filters):
    return MIQuerySpec(metric="current_outstanding_balance", aggregation="sum",
                       filters=filters)


# --------------------------------------------------------------------------- #
# the check
# --------------------------------------------------------------------------- #

def test_a_value_the_book_spells_differently_is_the_books_value(semantics):
    frame = pd.DataFrame({"erm_product_type": ["Lifetime Mortgage", "Drawdown"]})
    spec, absent = adapter.filter_values_in_book(
        _spec({"erm_product_type": "lifetime_mortgage"}), semantics, frame)
    assert absent == []
    assert spec.filters == {"erm_product_type": "Lifetime Mortgage"}


def test_a_value_the_book_does_not_record_is_named_with_what_it_does(semantics):
    frame = pd.DataFrame({"account_status": ["Live", "Redeemed", None]})
    spec, absent = adapter.filter_values_in_book(
        _spec({"account_status": "Active"}), semantics, frame)
    assert absent == [{"field": "account_status", "label": absent[0]["label"],
                       "values": ["Active"], "book_values": ["Live", "Redeemed"]}]


def test_a_field_of_many_values_is_not_listed_back(semantics):
    frame = pd.DataFrame({"account_status": [f"S{i:03d}" for i in range(40)]})
    _spec_out, absent = adapter.filter_values_in_book(
        _spec({"account_status": "Active"}), semantics, frame)
    assert absent[0]["book_values"] is None


def test_a_value_the_executor_translates_is_left_to_it(semantics):
    """A region name on a book that records codes is the executor's own domain
    resolution ("London" -> TLI43), not a missing value."""
    frame = pd.DataFrame({"geographic_region_obligor": ["TLI43", "TLK11"]})
    spec, absent = adapter.filter_values_in_book(
        _spec({"geographic_region_obligor": "London"}), semantics, frame)
    assert absent == []
    assert spec.filters == {"geographic_region_obligor": "London"}


def test_a_column_the_book_lacks_or_a_numeric_bound_is_not_checked(semantics):
    frame = pd.DataFrame({"current_loan_to_value": [0.4, 0.5]})
    filters = {"account_status": "Active",
               "current_loan_to_value": {"op": "gt", "value": 0.45}}
    spec, absent = adapter.filter_values_in_book(_spec(filters), semantics, frame)
    assert absent == [] and spec.filters == filters


# --------------------------------------------------------------------------- #
# served
# --------------------------------------------------------------------------- #

def test_a_value_the_book_records_is_answered(monkeypatch, book, semantics):
    payload, record, _ = _served(
        _intent(operation="point_in_time", measures=_BALANCE,
                filters=_status("Active")), monkeypatch, book, semantics)
    assert payload is not None and payload.get("ok"), record.get("execution")
    expected = book.loc[book["account_status"] == "Active",
                        "current_outstanding_balance"].sum()
    assert record["execution"]["value"] == pytest.approx(float(expected))


def test_a_value_the_book_does_not_record_is_declined_in_words(monkeypatch, book,
                                                              semantics):
    other = book.copy()
    other["account_status"] = other["account_status"].map(
        {"Active": "Live", "Redeemed": "Redeemed"})
    payload, record, _ = _served(
        _intent(operation="point_in_time", measures=_BALANCE,
                filters=_status("Active")), monkeypatch, other, semantics)
    assert record["execution"]["attempted"] is False
    assert record["execution"]["filter_values_not_in_book"][0]["book_values"] == [
        "Live", "Redeemed"]
    # Worded from the record exactly as production words a decline
    # (`respond` -> `plan_decline.envelope`); this harness serves without it.
    answer = decline.envelope(question="q", body=record,
                              reason="INELIGIBLE:FILTER_VALUE_NOT_IN_BOOK",
                              view="funded")["answer"]
    assert "records" in answer and "Live, Redeemed" in answer
    assert "'Active' is not one of them" in answer
    assert "could not be produced reliably" not in answer


def test_the_decline_wording_without_a_list():
    body = {"execution": {"filter_values_not_in_book": [
        {"field": "broker_channel", "label": "Broker", "values": ["Acme"],
         "book_values": None}]}}
    text = decline.message(body, "INELIGIBLE:FILTER_VALUE_NOT_IN_BOOK")
    assert "no loan in this book has Broker 'Acme'" in text


def test_a_registry_value_is_said_in_words_and_an_empty_field_as_unrecorded():
    """Twins run 2026-10-01 on 2e9e1cc4: "no loan in this book has Product
    Type 'lifetime_mortgage'" quoted the registry's code. The value is said
    as the reading above it says it, and a field the book holds no value for
    at all is said to be unrecorded."""
    body = {"execution": {"filter_values_not_in_book": [
        {"field": "erm_product_type", "label": "Product Type",
         "values": ["lifetime_mortgage"], "book_values": []}]}}
    text = decline.message(body, "INELIGIBLE:FILTER_VALUE_NOT_IN_BOOK")
    assert "this book does not record Product Type for any loan" in text
    many = {"execution": {"filter_values_not_in_book": [
        {"field": "erm_product_type", "label": "Product Type",
         "values": ["lifetime_mortgage"], "book_values": None}]}}
    text = decline.message(many, "INELIGIBLE:FILTER_VALUE_NOT_IN_BOOK")
    assert "no loan in this book has Product Type 'lifetime mortgage'" in text

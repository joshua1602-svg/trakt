"""Eleven questions, and not one said which file it was about.

What the operator saw, for a field that IS mapped::

    The field 'erm product type' could not be filled automatically.
    How should Trakt treat it?

    [ Accept the suggested match ] [ Choose a different match ]
    [ Combine the overlapping sources ] [ Decide later ]

and said the only sensible thing: "I don't know which tape these fields
emanate from because some were confirmed not mapped, and some were mapped."

TWO FAULTS IN ONE SENTENCE.

It is untrue. `erm_product_type` was filled — from `Product Category` in
`LoanExtract One`. The decision is a `source_priority_confirmation`: FOUR files
offer that field and Trakt wants to know which is authoritative. "Could not be
filled automatically" describes a different decision entirely, and the options
beneath it ("Combine the overlapping sources") belong to the real one.

And it is unanswerable. ERE's pack is three workbooks that overlap — an
interest rate in all three, a balance in two. Without the file and column, an
operator cannot say which source to keep, which is the whole purpose of asking.

THE RIGHT QUESTION ALREADY EXISTED. `target_coverage` writes one that fits the
decision::

    operator_question = (f"Which source column is the authoritative source "
                         f"for '{tf['target_field']}'?")

It is carried into 28c and reaches this adapter, which threw it away and
substituted the generic sentence for every decision that is not a missing
field. One fact produced in one place and discarded in another — the same
shape as most of what went wrong on this delivery.
"""

from __future__ import annotations

import pytest

from operations_control import language
from operations_control.adapters import _chosen_source_sentence

AUTHORITATIVE = ("Which source column is the authoritative source for "
                 "'erm_product_type'?")


def _decision(**over):
    d = {"decision_type": "source_priority_confirmation",
         "target_field": "erm_product_type",
         "source_column": "Product Category",
         "source_file": "LoanExtract One - OMNI 2026_09_01.xlsx",
         "operator_question": AUTHORITATIVE}
    d.update(over)
    return d


class TestTheQuestionNamesItsSource:

    def test_it_names_the_column_and_the_file(self):
        said = _chosen_source_sentence(_decision())
        assert "Product Category" in said
        assert "LoanExtract One - OMNI 2026_09_01.xlsx" in said

    def test_a_decision_with_no_file_still_names_the_column(self):
        said = _chosen_source_sentence(_decision(source_file=""))
        assert "Product Category" in said

    def test_a_field_with_no_source_at_all_says_nothing(self):
        """`reporting_date` has none. Inventing one is worse than silence."""
        assert _chosen_source_sentence(
            _decision(source_column="", source_file="")) == ""

    @pytest.mark.parametrize("missing", [{}, {"source_column": None},
                                         {"source_column": "   "}])
    def test_it_never_invents_a_source(self, missing):
        d = {"target_field": "reporting_date"}
        d.update(missing)
        assert _chosen_source_sentence(d) == ""


class TestTheLenderSOwnWordsAreAllowed:
    """`language` admits an operator's file and column names through `allow`:
    their data is their vocabulary rather than ours."""

    def test_the_sentence_passes_the_operator_contract(self):
        d = _decision()
        said = f"{_chosen_source_sentence(d)} {d['operator_question']}"
        assert language.is_operator_safe(
            said, allow=(d["source_file"], d["source_column"]))

    def test_it_carries_no_path_or_artefact_code(self):
        said = _chosen_source_sentence(_decision())
        for banned in ("/tmp/", "/home/", "blob://", "28a_", "28c_"):
            assert banned not in said


class TestTheEnginesOwnQuestionIsUsed:
    """The adapter must stop substituting a generic sentence for a specific
    one the engine already wrote."""

    def test_the_generic_sentence_is_only_a_fallback(self):
        """The generic wording must sit inside the `if not question:` branch —
        reachable only when the engine supplied nothing."""
        import inspect
        from operations_control import adapters
        src = inspect.getsource(adapters)
        assert 'd.get("operator_question")' in src, (
            "the engine's question must be read")
        guard = src.find("if not question:")
        generic = src.find('"The field \'{friendly}\' could not be filled ')
        if generic == -1:                      # f-string, quoted either way
            generic = src.find("could not be filled ", guard)
        assert guard != -1 and generic > guard, (
            "the generic sentence must only be a fallback")

    def test_the_authoritative_question_survives_to_the_operator(self):
        d = _decision()
        question = str(d.get("operator_question") or "")
        chosen = _chosen_source_sentence(d)
        full = f"{chosen} {question}" if chosen else question
        assert "authoritative" in full
        assert "could not be filled" not in full


class TestTheLivePackReadsSensibly:

    @pytest.mark.parametrize("field,column,file_name", [
        ("current_interest_rate", "Current Interest Rate",
         "LoanExtract One - OMNI 2026_09_01.xlsx"),
        ("current_valuation_amount", "Latest Valuation",
         "PropertyExtract - Omni 2026_09_01.xlsx"),
        ("origination_date", "Policy Completion Date",
         "LoanExtract One - OMNI 2026_09_01.xlsx"),
    ])
    def test_each_question_says_where_the_field_comes_from(self, field, column,
                                                           file_name):
        said = _chosen_source_sentence(
            _decision(target_field=field, source_column=column,
                      source_file=file_name))
        assert column in said and file_name in said

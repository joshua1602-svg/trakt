"""One governed Predicate, one execution meaning.

The invariant:

    For any frame and any governed `Predicate(field, op, value)`, the shipped
    point-in-time executor (`_apply_filters`) and the reusable population
    executor (`apply_population`) must either select the SAME ROWS, or fail in
    the SAME GOVERNED WAY. There must be no case where one narrows while the
    other silently widens.

Before this, `apply_population` reused the COMPARATOR — it called
`_apply_numeric_op` — and reimplemented none of the resolution or normalisation
around it. Its docstring claimed the two paths "cannot disagree". They disagreed
five ways, and only two were visible in the 119-question corpus: a corpus can
only exercise the predicates it happens to contain.

Every test here drives BOTH executors and compares. A test that only asserted
what `apply_population` returns would pass just as happily against a second
divergent implementation.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from mi_agent.mi_query_executor import MIQueryExecutionError, _apply_filters
from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.population import (Predicate, apply_population, material_predicates,
                                 predicate_of)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    return load_mi_semantics(
        _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


@pytest.fixture
def book():
    """A small governed book: a percent-FRACTION field, a percent-POINTS field,
    a plain numeric, two categoricals, and a null in every column."""
    return pd.DataFrame({
        "current_loan_to_value": [0.30, 0.55, 0.72, None, 0.95],
        "youngest_borrower_age": [55, 68, 75, 82, None],
        "collateral_geography": ["South East", "London", "South West", None,
                                 "south east"],
        "borrower_type": ["Single", "Joint", "Single", "Joint", None],
        "current_interest_rate": [3.5, 4.25, 5.0, 6.75, None],
        "months_on_book": [1, 3, 12, 36, None],
    })


def _shipped(frame, filters, semantics):
    try:
        return "rows", sorted(_apply_filters(
            frame.copy(), MIQuerySpec(filters=dict(filters)), semantics, [], []).index)
    except MIQueryExecutionError as exc:
        return "refused", str(exc)


def _reusable(frame, filters, semantics):
    out, evidence = apply_population(
        frame, material_predicates(filters, semantics), semantics)
    if evidence.unavailable or evidence.blocked_reason:
        # A frame handed back here would BE the silent-widening defect.
        assert out is None, "an unexecutable predicate must not return a frame"
        return "refused", evidence.blocked_reason or "; ".join(evidence.unavailable)
    return "rows", sorted(out.index)


def _both_agree(frame, filters, semantics):
    """The invariant, as one call. Returns the shared outcome."""
    shipped, reusable = _shipped(frame, filters, semantics), _reusable(frame, filters, semantics)
    assert shipped[0] == reusable[0], (
        f"one path {shipped[0]}, the other {reusable[0]}: {filters}")
    if shipped[0] == "rows":
        assert shipped[1] == reusable[1], (
            f"different rows for {filters}: {len(shipped[1])} vs {len(reusable[1])}")
    return shipped


# --------------------------------------------------------------------------- #
# Class 1 — percent normalisation
# --------------------------------------------------------------------------- #
def test_a_percent_threshold_in_points_selects_the_same_rows(book, semantics):
    """LTV is stored 0.30..0.95. "over 50" means 50 POINTS, not 50x.

    `_apply_filters` divides by 100 once, in the single percent-scale source of
    truth. `apply_population` did not, so it selected NOTHING where the executor
    selected 1,889 loans on the real book.
    """
    kind, rows = _both_agree(book, {"current_loan_to_value": {"op": "gt", "value": 50}},
                             semantics)
    assert kind == "rows" and len(rows) == 3


def test_a_percent_range_rescales_both_bounds(book, semantics):
    kind, rows = _both_agree(
        book, {"current_loan_to_value": {"op": "between", "value": [40, 60]}}, semantics)
    assert kind == "rows" and len(rows) == 1


def test_an_operand_already_in_the_stored_scale_is_left_alone(book, semantics):
    """0.5 is below the 1.5 guard, so it is already a fraction. Rescaling it
    would be the mirror defect."""
    kind, rows = _both_agree(book, {"current_loan_to_value": {"op": "gt", "value": 0.5}},
                             semantics)
    assert kind == "rows" and len(rows) == 3


def test_a_percent_points_column_is_not_rescaled(book, semantics):
    """`current_interest_rate` is `format: percent` but STORED IN POINTS. The
    scale is decided per column at execution time, never from the field name."""
    kind, rows = _both_agree(book, {"current_interest_rate": {"op": "gt", "value": 4}},
                             semantics)
    assert kind == "rows" and len(rows) == 3


def test_a_non_percent_numeric_is_never_rescaled(book, semantics):
    kind, rows = _both_agree(book, {"youngest_borrower_age": {"op": "gt", "value": 70}},
                             semantics)
    assert kind == "rows" and len(rows) == 2


# --------------------------------------------------------------------------- #
# Class 2 — operator aliases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", [">", "above", "over", "gte", "greater_than_or_equal",
                                "at_least", "<=", "below"])
def test_an_operator_alias_means_the_same_thing_to_both_paths(book, semantics, op):
    """`_OP_ALIASES` lived in the executor alone. The reusable path handed the
    raw operator to `_apply_numeric_op`, which KeyError'd, which became
    "unavailable", which returned the WHOLE BOOK."""
    kind, rows = _both_agree(book, {"youngest_borrower_age": {"op": op, "value": 70}},
                             semantics)
    assert kind == "rows"
    assert len(rows) < 5, f"{op!r} narrowed nothing — that is the widening defect"


# --------------------------------------------------------------------------- #
# Class 3 — a predicate that cannot execute must fail closed
# --------------------------------------------------------------------------- #
def test_a_missing_column_refuses_on_both_paths(book, semantics):
    kind, _ = _both_agree(book.drop(columns=["current_loan_to_value"]),
                          {"current_loan_to_value": {"op": "gt", "value": 50}}, semantics)
    assert kind == "refused"


def test_an_unknown_semantic_field_refuses_on_both_paths(book, semantics):
    kind, _ = _both_agree(book, {"not_a_governed_field": {"op": "gt", "value": 1}},
                          semantics)
    assert kind == "refused"


def test_a_value_the_operator_cannot_compare_refuses_on_both_paths(book, semantics):
    kind, _ = _both_agree(book, {"youngest_borrower_age": {"op": "gt", "value": "abc"}},
                          semantics)
    assert kind == "refused"


def test_an_unexecutable_predicate_returns_no_frame_at_all(book, semantics):
    """THE anti-widening guarantee, asserted directly rather than through
    agreement: there must be no frame to mistake for a narrowed one."""
    frame, evidence = apply_population(
        book.drop(columns=["borrower_type"]),
        [Predicate("borrower_type", "eq", "Joint")], semantics)
    assert frame is None
    assert evidence.unavailable and evidence.blocked_reason
    assert evidence.is_usable is False


def test_a_predicate_that_narrows_nothing_is_still_a_success(book, semantics):
    """The distinction the caller must be able to make: executed-and-unchanged
    is NOT could-not-execute."""
    frame, evidence = apply_population(
        book, [Predicate("youngest_borrower_age", "gt", 0)], semantics)
    assert frame is not None and len(frame) == 4       # the null row drops out
    assert evidence.is_usable and not evidence.unavailable


# --------------------------------------------------------------------------- #
# Class 4 — value-domain resolution
# --------------------------------------------------------------------------- #
def test_an_exact_region_name_matches_case_insensitively_on_both_paths(book, semantics):
    kind, rows = _both_agree(book, {"collateral_geography": "South East"}, semantics)
    assert kind == "rows" and len(rows) == 2


@pytest.mark.parametrize("term", ["the South East", "Greater London"])
def test_a_region_term_resolves_through_its_governed_domain_on_both_paths(
        book, semantics, term):
    """The executor does not know what a region is. It asks the semantics which
    DOMAIN the field's values are drawn from and lets the domain resolve the
    term. The reusable path did an exact match, found nothing, and returned an
    EMPTY population — the mirror of silent widening, and just as wrong.

    Non-vacuity: an exact match must fail first, or this proves nothing.
    """
    exact = (book["collateral_geography"].astype(str).str.strip().str.casefold()
             == term.strip().casefold())
    assert not exact.any(), f"{term!r} matches exactly — the domain is not exercised"
    kind, rows = _both_agree(book, {"collateral_geography": term}, semantics)
    assert kind == "rows" and rows, f"{term!r} resolved to nothing on both paths"


# --------------------------------------------------------------------------- #
# Class 5 — one Predicate, one meaning, whatever shape the spec used
# --------------------------------------------------------------------------- #
def test_the_two_spec_shapes_for_one_category_mean_the_same_thing(book, semantics):
    """RULED product semantics: a bare categorical value and an explicit
    equality predicate are identical, and the input shape carries no intended
    business meaning.

    `material_predicates` already normalised `{"op":"eq","value":"Joint"}` and a
    bare `"Joint"` to the same Predicate. `_apply_filters` did not: it sent the
    first through the numeric comparator (which raised) and the second through
    the categorical branch, so one Predicate had two shipped meanings and the
    invariant was unstatable. The shape is normalised away, NOT carried."""
    bare = _both_agree(book, {"borrower_type": "Joint"}, semantics)
    dictish = _both_agree(book, {"borrower_type": {"op": "eq", "value": "Joint"}},
                          semantics)
    assert bare == dictish
    assert bare[0] == "rows" and len(bare[1]) == 2


def test_predicate_of_is_the_one_normaliser_for_all_three_shapes():
    assert predicate_of("f", {"op": "gt", "value": 5}) == Predicate("f", "gt", 5)
    assert predicate_of("f", ["a", "b"]) == Predicate("f", "in", ["a", "b"])
    assert predicate_of("f", "a") == Predicate("f", "eq", "a")


# --------------------------------------------------------------------------- #
# Nulls, membership, and the multi-predicate case
# --------------------------------------------------------------------------- #
def test_nulls_are_excluded_identically_by_both_paths(book, semantics):
    kind, rows = _both_agree(book, {"youngest_borrower_age": {"op": "le", "value": 200}},
                             semantics)
    assert kind == "rows" and len(rows) == 4


def test_membership_selects_identically(book, semantics):
    kind, rows = _both_agree(book, {"borrower_type": ["Joint"]}, semantics)
    assert kind == "rows" and len(rows) == 2


def test_two_predicates_compose_identically(book, semantics):
    """Multiple filters keep TODAY's semantics — conjunctive, applied in order.
    This task does not redesign that."""
    kind, rows = _both_agree(book, {"current_loan_to_value": {"op": "gt", "value": 50},
                                    "youngest_borrower_age": {"op": "gt", "value": 70}},
                             semantics)
    assert kind == "rows" and len(rows) == 1


# --------------------------------------------------------------------------- #
# Deliberately PRESERVED imperfections — not fixed here
# --------------------------------------------------------------------------- #
def test_a_repeated_field_still_collapses_to_the_last_predicate(semantics):
    """`spec.filters` is a dict, so "LTV above 50 and LTV below 80" keeps only
    the second. `Predicate[]` could express both; being ABLE to is not
    authorisation to change the shipped product mid-migration."""
    spec = {"current_loan_to_value": {"op": "gt", "value": 50}}
    spec["current_loan_to_value"] = {"op": "lt", "value": 80}
    assert material_predicates(spec, semantics) == [
        Predicate("current_loan_to_value", "lt", 80)]


def test_a_source_portfolio_scope_is_still_not_a_row_predicate(semantics):
    """P1I-A: that phrase family is SCOPE and travels on the lens."""
    assert material_predicates({"source_portfolio_id": "nbs"}, semantics) == []

def test_the_predicate_carries_no_shape_or_provenance_state():
    """The ruling was that the shape has no business meaning — so it must not
    survive as state either. `Predicate` is field, operator, value and nothing
    else, and the two shapes produce an EQUAL object, not merely an
    equivalent one."""
    from dataclasses import fields
    assert [f.name for f in fields(Predicate)] == ["field", "op", "value"]
    assert (predicate_of("borrower_type", {"op": "eq", "value": "Joint"})
            == predicate_of("borrower_type", "Joint"))


def test_the_executor_recognises_no_borrower_type_vocabulary():
    """Field binding stays upstream. The executor dispatches on the OPERAND'S
    TYPE, so a governed string field it has never heard of behaves identically
    to `borrower_type` — which is the proof that no vocabulary was added."""
    import pandas as pd
    from mi_agent.mi_query_executor import (PREDICATE_CATEGORICAL,
                                            governed_predicate_mask)
    frame = pd.DataFrame({"anything_at_all": ["Joint", "Single", "Joint"]})
    execution = governed_predicate_mask(frame, "anything_at_all", "eq", "joint", {})
    assert execution.kind == PREDICATE_CATEGORICAL
    assert list(execution.mask) == [True, False, True]

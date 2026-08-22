#!/usr/bin/env python3
"""Stage 4, step 2: a named dimension gets the role the sentence gave it.

`KIND_GROUPING` meant "a dimension the question named", with no
grouping-versus-filter distinction: "balance by region for joint borrowers"
raised it for the axis and for the filter alike, and the reader was told
borrower type was a breakdown when it was a selector.

The role comes from the parser's own slot assignment. Only a POSITIVELY
identified filter moves — a dimension no source assigned a role to stays exactly
as it is, which is the opposite conservatism from `32c263a`.

Test shape carried over from the population fix: three tests establishing the
property, and a fourth proving the reclassifier can still fail.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

warnings.simplefilter("ignore")
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import execution_receipt as R  # noqa: E402


class _Spec:
    def __init__(self, filters=None, dimension=None):
        self.filters = filters or {}
        self.dimension = dimension
        self.dimensions = []


def _grouping(field, label=None):
    return R.RequestedFacet(kind=R.KIND_GROUPING, field_key=field,
                            label=label or field.replace("_", " "))


# -- 1: a filter-role dimension MOVES ---------------------------------------
def test_a_dimension_the_parser_put_in_filters_becomes_a_population():
    spec = _Spec(filters={"borrower_type": "Joint"}, dimension="collateral_geography")
    out = R._split_named_dimension_roles([_grouping("borrower_type", "joint borrower")], spec)
    assert len(out) == 1
    assert out[0].kind == R.KIND_POPULATION
    assert out[0].field_key == "borrower_type"


# -- 2: a grouping-role dimension does NOT move -----------------------------
def test_a_dimension_the_parser_put_on_the_axis_stays_a_grouping():
    spec = _Spec(filters={"borrower_type": "Joint"}, dimension="collateral_geography")
    out = R._split_named_dimension_roles([_grouping("collateral_geography", "region")], spec)
    assert out[0].kind == R.KIND_GROUPING


def test_both_roles_in_one_sentence_are_told_apart():
    """The case KIND_GROUPING could not distinguish."""
    spec = _Spec(filters={"borrower_type": "Joint"}, dimension="collateral_geography")
    out = R._split_named_dimension_roles(
        [_grouping("collateral_geography", "region"),
         _grouping("borrower_type", "joint borrower")], spec)
    assert [f.kind for f in out] == [R.KIND_GROUPING, R.KIND_POPULATION]


# -- 3: an unresolved-role dimension does NOT move --------------------------
def test_a_dimension_with_no_role_from_any_source_becomes_a_question():
    """Superseded by the settled clarify rule, and the reason matters.

    This asserted the facet STAYS PUT — the conservatism the split shipped with,
    chosen against 32c263a, which let anything unjustifiable fall to POPULATION,
    the side that blocks, at a cost of 160 runs.

    Staying put turned out not to be the safe option it looked like: an
    unapplied grouping refuses too, because `_blocks` returns True for any of
    them. The measured choice was never between a whole-book number and a
    refusal; it was between a refusal and a question. It is now a question.

    What is preserved from 32c263a is the thing that actually mattered: an
    ambiguous dimension does not become a POPULATION. Falling to the blocking
    side is still wrong; the correction is that staying put was blocking too.
    """
    spec = _Spec(filters={}, dimension=None)
    out = R._split_named_dimension_roles([_grouping("broker_channel", "broker")], spec)
    assert out[0].kind == R.KIND_UNRESOLVED_ROLE
    assert out[0].kind != R.KIND_POPULATION


# -- 4: the reclassifier can fail -------------------------------------------
def test_the_reclassifier_does_not_simply_move_everything():
    """Proof the three above are detecting the slot assignment.

    Same facets, same call, a spec whose filters name a DIFFERENT field: neither
    becomes a POPULATION. A reclassifier that moved on the presence of any
    filter at all would fail here and pass every test above.

    They become questions rather than staying groupings — no source gave either
    one a role — which is the settled rule, not a failure of this check. What
    this proves is that the FILTER slot is being read, not merely counted.
    """
    spec = _Spec(filters={"collateral_geography": "London"}, dimension=None)
    facets = [_grouping("borrower_type", "joint borrower"),
              _grouping("broker_channel", "broker")]
    out = R._split_named_dimension_roles(facets, spec)
    assert [f.kind for f in out] == [R.KIND_UNRESOLVED_ROLE, R.KIND_UNRESOLVED_ROLE]

    # And the same two fields WITH a role are read correctly, which is what
    # makes the assertion above about the slot rather than about the default.
    on_axis = _Spec(filters={}, dimension=None)
    on_axis.dimensions = ["borrower_type", "broker_channel"]
    assert [f.kind for f in R._split_named_dimension_roles(facets, on_axis)] \
        == [R.KIND_GROUPING, R.KIND_GROUPING]


def test_no_other_facet_kind_is_touched():
    """The pre-registered prediction, asserted."""
    spec = _Spec(filters={"current_loan_to_value": {"op": "gt", "value": 50}})
    others = [R.RequestedFacet(kind=k, field_key="current_loan_to_value", label=k)
              for k in (R.KIND_THRESHOLD, R.KIND_GEOGRAPHIC_SCOPE, R.KIND_RANKING,
                        R.KIND_COMPARISON_PERIOD, R.KIND_STATISTIC, R.KIND_SHARE)]
    out = R._split_named_dimension_roles(others, spec)
    assert [f.kind for f in out] == [f.kind for f in others]


# -- B5 must stay unreachable across the split ------------------------------
def test_a_reclassified_label_names_its_field():
    """Condition 1. Reclassifying without relabelling is what would make B5
    live: the literal guard splits the label on the field name, and an empty
    value accepts any predicate naming that field — including the wrong
    population."""
    spec = _Spec(filters={"borrower_type": "Joint"})
    out = R._split_named_dimension_roles([_grouping("borrower_type", "joint borrower")], spec)
    assert "borrower_type" in out[0].label
    assert out[0].label == "the population borrower_type = Joint"


def test_that_b5_guard_can_fail():
    """The original label omits the field — which is exactly the shape the
    relabelling exists to prevent, and it is shown to be detectable."""
    from question_interpretation.b5_reachability import label_omits_its_field
    assert label_omits_its_field(_grouping("borrower_type", "joint borrower")) is True
    spec = _Spec(filters={"borrower_type": "Joint"})
    out = R._split_named_dimension_roles([_grouping("borrower_type", "joint borrower")], spec)
    assert label_omits_its_field(out[0]) is False


def test_a_comparator_filter_relabels_through_the_same_renderer():
    spec = _Spec(filters={"current_loan_to_value": {"op": "gt", "value": 50.0}})
    out = R._split_named_dimension_roles([_grouping("current_loan_to_value", "ltv")], spec)
    assert out[0].label == "the population current_loan_to_value gt 50.0"


# -- the seasoning families must not be reclassified -------------------------
def test_no_seasoning_facet_is_reclassified_on_the_robustness_bank():
    """Verified rather than assumed, as the brief requires: the population
    over-assignment that caused the 160-run regression must not recur.

    Counts moves to POPULATION specifically. Under the settled clarify rule an
    unresolved dimension does move — to KIND_UNRESOLVED_ROLE, which asks rather
    than blocks — and counting every kind change would make this fail for the
    one reclassification it was never about.
    """
    sys.path.insert(0, str(_REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"))
    import nl_bank
    from mi_agent import mi_calibration as CAL
    from mi_agent.llm_query_parser import _deterministic_parse
    from mi_agent.mi_query_validator import load_mi_semantics

    tape = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
            / "platform" / "alderbridge" / "2026-06-30"
            / "platform_canonical_typed.csv")
    if not tape.exists():
        pytest.skip("calibration book not built")
    semantics = load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    df = CAL.default_bank_frame(tape)
    columns = list(df.columns)

    moved = 0
    seasoning = [r for r in nl_bank.bank("alderbridge")
                 if r["intent"] in ("Q1", "Q7", "Q8")]
    assert len(seasoning) == 20
    for row in seasoning:
        q = row["question"]
        for cols in (set(columns), None):
            spec, _ = _deterministic_parse(q, semantics, available_columns=cols)
            dims = R.requested_dimension_terms(q, semantics, columns)
            detected = R.detect_requested_facets(q, semantics, frame=df,
                                                 requested_dimensions=dims)
            after = R._split_named_dimension_roles(detected, spec)
            moved += sum(1 for a, b in zip(detected, after)
                         if a.kind != b.kind and b.kind == R.KIND_POPULATION)
    assert moved == 0

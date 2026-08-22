"""D2 — one owner for "is this named dimension an axis or a filter".

Four readers had an opinion and the census found them disagreeing on 37 of 693
questions. `dimension_role` is now the only one that decides; the rest consume
it. These tests pin each of its five sources, and they pin two facts about the
corpus that the surfaces cannot show because the surfaces measure ANSWERS:

  * every dimension facet reaches the owner with `span=None`, so the sentence's
    role markers have to be located rather than carried;
  * the deterministic parser resolves NO categorical filter, so the owner's
    FILTER answer is unreachable through the corpus on this arm. That is why
    the coverage below is constructed and not drawn from 693 questions.

`test_d2_routed_role_carriage.py` is the other half — the same decision arriving
on the routed path, and the two failure modes that arrival could have
reintroduced.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mi_agent import execution_receipt as R
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent import mi_calibration as CAL


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
         / "platform" / "alderbridge" / "2026-06-30"
         / "platform_canonical_typed.csv")


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(
        _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


@pytest.fixture(scope="module")
def frame():
    if not _TAPE.exists():
        pytest.skip("calibration book not built")
    return CAL.default_bank_frame(_TAPE)


def _facet(label, field, alts=()):
    return R.RequestedFacet(kind=R.KIND_GROUPING, label=label, field_key=field,
                            alt_keys=tuple(alts))


# --------------------------------------------------------------------------- #
# The five sources, in order
# --------------------------------------------------------------------------- #
def test_source_1_a_parser_filter_slot_makes_it_a_filter(semantics):
    role, key = R.dimension_role(
        _facet("account status", "account_status"),
        question="balance where account status is active",
        filters={"account_status": "Active"}, axes=set(),
        columns={"account_status"}, fields=semantics.get("fields", {}))
    assert (role, key) == (R.ROLE_FILTER, "account_status")


def test_source_1_reads_every_key_the_facet_resolves_to(semantics):
    """Not just `field_key`. A generic term resolves to different concrete
    fields depending on the book, and the parser may have slotted either."""
    role, key = R.dimension_role(
        _facet("region", "collateral_geography",
               alts=("geographic_region_obligor",)),
        question="balance for the region",
        filters={"geographic_region_obligor": "South East"}, axes=set(),
        columns={"geographic_region_obligor"},
        fields=semantics.get("fields", {}))
    assert (role, key) == (R.ROLE_FILTER, "geographic_region_obligor")


def test_source_2_a_parser_axis_slot_makes_it_an_axis(semantics):
    role, key = R.dimension_role(
        _facet("region", "collateral_geography"),
        question="balance by region", filters={},
        axes={"collateral_geography"}, columns={"collateral_geography"},
        fields=semantics.get("fields", {}))
    assert (role, key) == (R.ROLE_AXIS, None)


def test_source_3_the_sentence_marks_an_axis_when_no_slot_did(semantics):
    """THE NEW SOURCE. `lexical.grouping_cut` already owns where a grouping
    clause opens; until now the role decision asked it nothing, so a parser that
    failed to fill a slot made "by region" look ambiguous."""
    role, key = R.dimension_role(
        _facet("region", "collateral_geography"),
        question="What is the pipeline amount by region?",
        filters={}, axes=set(), columns={"collateral_geography"},
        fields=semantics.get("fields", {}))
    assert (role, key) == (R.ROLE_AXIS, None)


def test_source_3_does_not_fire_where_the_mention_precedes_the_cut(semantics):
    """"region" here is the SUBJECT, not the axis, and the axis is LTV bucket.
    A source that fired on the presence of "by" anywhere would answer AXIS for
    both and the ordering would mean nothing."""
    role, _ = R.dimension_role(
        _facet("region", "collateral_geography"),
        question="regional balance by ltv bucket",
        filters={}, axes=set(), columns={"collateral_geography"},
        fields=semantics.get("fields", {}))
    assert role == R.ROLE_UNRESOLVED


def test_source_4_a_field_the_book_cannot_express_stays_an_axis(semantics):
    """Reader 2's rule, unchanged. "Did you mean a breakdown or a filter?"
    invites an answer that cannot be honoured either way, so it is left an axis
    and the grouping branch says the thing the reader can act on."""
    role, _ = R.dimension_role(
        _facet("broker", "broker_channel"),
        question="show broker concentration", filters={}, axes=set(),
        columns={"current_outstanding_balance"},
        fields=semantics.get("fields", {}))
    assert role == R.ROLE_AXIS


def test_source_5_nothing_settles_it(semantics):
    role, _ = R.dimension_role(
        _facet("region", "collateral_geography"),
        question="show me interesting regions", filters={}, axes=set(),
        columns={"collateral_geography"}, fields=semantics.get("fields", {}))
    assert role == R.ROLE_UNRESOLVED


def test_the_order_is_filter_then_axis_then_sentence(semantics):
    """A field in BOTH slots is a filter. The parser positively identified a
    narrowing; a "by" elsewhere in the sentence does not undo that."""
    role, key = R.dimension_role(
        _facet("account status", "account_status"),
        question="balance by region where account status is active",
        filters={"account_status": "Active"}, axes={"account_status"},
        columns={"account_status"}, fields=semantics.get("fields", {}))
    assert (role, key) == (R.ROLE_FILTER, "account_status")


# --------------------------------------------------------------------------- #
# Source 3 answers AXIS only — stated as a test, not as a comment
# --------------------------------------------------------------------------- #
def test_the_sentence_cannot_make_something_a_filter(semantics):
    """A grouping reclassified to a filter must carry a resolved PREDICATE:
    `_analytical_population_satisfies` recovers the value by splitting the label
    on the field name, and a population facet with no value accepts any
    predicate naming that field (B5). Reader 1 is the only source of a value, so
    where the sentence marks a selector the parser did not resolve, the owner
    must NOT invent one.

    This is the live gap D2 cannot close: the deterministic parser resolves no
    categorical filter at all, so "where account status is active" reaches here
    with empty filters and no source can say "filter"."""
    role, key = R.dimension_role(
        _facet("account status", "account_status"),
        question="total balance where account status is active",
        filters={}, axes=set(), columns={"account_status"},
        fields=semantics.get("fields", {}))
    assert role != R.ROLE_FILTER
    assert key is None


# --------------------------------------------------------------------------- #
# Two measured facts about the corpus, asserted rather than assumed
# --------------------------------------------------------------------------- #
def test_dimension_facets_carry_no_span_so_the_term_must_be_located(semantics, frame):
    """`requested_dimension_terms` returns the matched TERM and no offsets. The
    owner recovers the span; if that ever changes to a carried span, this fails
    and the recovery can be deleted rather than left as dead code."""
    columns = list(frame.columns)
    question = "What is the pipeline amount by region?"
    dims = R.requested_dimension_terms(question, semantics, columns)
    facets = R.detect_requested_facets(question, semantics, frame=frame,
                                       requested_dimensions=dims)
    dimension_facets = [f for f in facets if f.kind == R.KIND_GROUPING]
    assert dimension_facets, "the probe must raise at least one dimension"
    assert all(f.span is None for f in dimension_facets)
    assert R._named_term_span(question, "region") == (31, 37)
    assert question[31:37] == "region"


def test_no_corpus_question_slots_a_named_dimension_as_a_filter(semantics, frame):
    """The reason the routed coverage is constructed.

    Measured across all 693 corpus questions: 81 carry a non-empty
    `spec.filters`, and the intersection of "field the parser slotted as a
    filter" with "dimension the detector named" is EMPTY. Five fields ever
    appear as filters — `current_loan_to_value`, `youngest_borrower_age`,
    `current_outstanding_balance`, `current_interest_rate` and
    `collateral_geography` — and the first four are numeric bounds on measures,
    which no dimension term resolves to.

    `collateral_geography` is the one categorical filter the parser does
    resolve, 9 times. It never overlaps because naming the geography DIMENSION
    suppresses the value binding: "exposure to London" binds the filter and
    names no dimension, while "regional exposure to London" names the dimension
    and binds nothing. The one constructed phrasing that produces the overlap —
    "how many loans are in the South East by region" — binds the fabricated
    value "South East By", which is B1.

    This is a statement about the CORPUS, not about the code. The FILTER branch
    is live and reachable through the drill-through API — see
    `test_the_filter_branch_is_reachable_through_the_drill_through_api`, which
    exercises it rather than asserting anything about it. The D2 report said
    "unreachable through a well-formed question on this arm" and did not check
    the other way in; that was wrong, and a branch carried on an assertion of
    its own unreachability is the position that produced e35a01b.

    Kept because it says why the routed-surface coverage is constructed: the day
    a question's TEXT reaches the branch, this fails and the construction can be
    replaced by a real case.
    """
    from mi_agent.llm_query_parser import _deterministic_parse

    columns = set(frame.columns)
    for question in ("balance by region where account status is active",
                     "total balance where repayment type is interest roll-up",
                     "show funded exposure by region for joint borrowers",
                     "regional exposure to London",
                     "Show geographic exposure in the South East."):
        spec, _ = _deterministic_parse(question, semantics,
                                       available_columns=columns)
        filters = getattr(spec, "filters", None) or {}
        named = {k for k, _t, _a in R.requested_dimension_terms(
            question, semantics, list(columns))}
        assert not (set(filters) & named), (
            f"{question!r} now slots a named dimension as a filter: "
            f"{sorted(set(filters) & named)}")


# --------------------------------------------------------------------------- #
# The FILTER branch is reachable. Exercised, not asserted about.
# --------------------------------------------------------------------------- #
def test_the_filter_branch_is_reachable_through_the_drill_through_api():
    """`MiQueryRequest.filters` — a UI drill from a breakdown into one group.

    Caller-supplied filters merge into `spec.filters` on both paths before the
    guard runs, so a drill from a "balance by region" table into South East is
    exactly the shape the role owner's FILTER branch exists for: the question
    NAMES region and the spec now SLOTS it as a filter.

    Three cases, because one would not distinguish the branch from the
    drill-through: two fields, and the same question without a drill.
    """
    if not _TAPE.exists():
        pytest.skip("calibration book not built")
    import os

    from demo_platform import config as cfg

    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)

    def kinds(question, filters=None):
        res = execute_governed_mi_query(
            MiQueryRequest(question=question, filters=filters), ctx).result or {}
        return [(f.get("kind"), f.get("field"), f.get("status"))
                for f in ((res.get("executionSummary") or {}).get("facets") or [])]

    assert ("row_population", "collateral_geography", "applied") in kinds(
        "balance by region", {"collateral_geography": "South East"})
    assert ("row_population", "vintage_year", "applied") in kinds(
        "balance by vintage", {"vintage_year": 2024})
    # The can-fail: without the drill the SAME question keeps its axis role, so
    # the assertions above are about the branch and not about the API.
    assert ("grouping_dimension", "collateral_geography", "applied") in kinds(
        "balance by region")

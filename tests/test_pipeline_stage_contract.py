"""tests/test_pipeline_stage_contract.py — PIPELINE_STAGE as a governed claim.

The last C6 prerequisite was that the interpretation contract could not express
Pipeline Stage, so `_route_evolution` re-read the raw sentence to pick between
ordinary evolution, stage evolution and the per-stage funnel series — three
semantic decisions with no structural representation anywhere.

`pipeline_stage` was already governed on the DATA side: `role: dimension` in
`config/mi/pipeline_field_contract.yaml`, categorical over `total_pipeline` in
`config/mi/stratification_catalogue.yaml`, with one normalisation map in
`pipeline_prep._STAGE_CANON`. What was missing was any way for a QUESTION to
name it. These tests pin the reader that closes that, and the two composed
claims it feeds — no new claim type, no funnel-specific field, no second
conversion calculator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from question_interpretation import lexical as L

#: The governed set. Five, in funnel order, terminal states last.
CANONICAL = ("KFI", "APPLICATION", "OFFER", "COMPLETED", "WITHDRAWN")


def test_the_canonical_stage_set_is_five_in_funnel_order():
    assert L.canonical_pipeline_stages() == CANONICAL


def test_the_vocabulary_is_derived_from_the_one_governed_map():
    """Not redeclared here. If the product adds a stage spelling, the reader
    gets it without this file changing."""
    from mi_agent_api.pipeline_prep import _STAGE_CANON
    vocab = L.pipeline_stage_vocabulary()
    assert vocab, "vocabulary is empty"
    assert set(vocab).issubset(set(_STAGE_CANON))
    assert set(vocab.values()) == set(CANONICAL)


def test_a_data_value_map_is_not_a_question_vocabulary():
    """`_STAGE_CANON` maps "funded" onto COMPLETED, which is right for a tape
    cell and catastrophic for a sentence: it would give the single most ordinary
    question in the corpus a COMPLETED stage. Dropped by view-name collision."""
    from mi_agent_api.pipeline_prep import _STAGE_CANON
    assert _STAGE_CANON["funded"] == "COMPLETED"
    assert "funded" not in L.pipeline_stage_vocabulary()
    assert L.pipeline_stage_request("Show funded balance evolution by month.") == (None, False)


def test_word_fragments_are_dropped_by_rule_not_by_hand():
    """"complete" is a prefix of "completed"/"completion" for the same stage, so
    it is a fragment rather than a stage noun — and it matched five corpus
    questions about DATA COMPLETENESS. "app" is the same shape. The canonical
    token itself is always kept even when it prefixes a longer spelling."""
    vocab = L.pipeline_stage_vocabulary()
    assert "complete" not in vocab and "app" not in vocab
    assert "completed" in vocab and "completion" in vocab and "application" in vocab
    assert "offer" in vocab and "offer issued" in vocab   # canonical token survives
    assert L.pipeline_stage_request("How complete is interest rate?") == (None, False)


# --------------------------------------------------------------------------- #
# Every canonical stage is namable — including the one the route cannot reach
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stage,question", [
    ("KFI", "Show the KFI trend by week."),
    ("APPLICATION", "How have application-stage cases changed?"),
    ("OFFER", "How have offer-stage cases changed?"),
    ("COMPLETED", "Show the completion trend by week."),
    ("WITHDRAWN", "Show withdrawn cases over time."),
])
def test_every_canonical_stage_is_representable(stage, question):
    assert L.pipeline_stage_request(question)[0] == stage


def test_the_governed_vocabulary_reaches_every_canonical_stage():
    """THE CURRENT PRODUCT INVARIANT: a question can name any of the five.

    This test used to assert the opposite — that the shipped route could reach
    only four, WITHDRAWN being unnameable from any question. That was a pinned
    record of a legacy defect in `chat_routing._FUNNEL_KEYWORDS`, a five-substring
    map that C6 retired. An estate must not go on asserting behaviour the product
    has fixed, so the requirement is inverted here and the historical fact is kept
    where history belongs: `docs/mi_conversion6_stop_conditions.md`, the C6 report,
    and the commit that removed the map.

    The defect was worse than a missing spelling. A stage the map did not
    recognise did not refuse — it fell through to the funded series, so
    *"Show the illustration trend"* answered with GBP1.96bn of whole-book funded
    balance. Correcting that is why the five affected questions are classified
    AUTHORISED H4 — LEGACY WRONG-DELIVERY CORRECTION rather than as equivalence.

    NON-VACUITY is derived, not hand-asserted: the spellings the retired map
    could not match are computed from the governed vocabulary and must be
    non-empty and must include WITHDRAWN, the stage it could not reach at all.
    """
    for stage in CANONICAL:
        spellings = [k for k, v in L.pipeline_stage_vocabulary().items() if v == stage]
        assert spellings, stage
        for spelling in spellings:
            assert L.pipeline_stage_request(f"Show {spelling} cases.")[0] == stage, spelling

    # What the five-substring map could NOT have matched. Computed from the
    # governed map so this stays true as the product adds spellings.
    retired = ("kfi", "application", "offer", "completion", "completed")
    missed = {k: v for k, v in L.pipeline_stage_vocabulary().items()
              if not any(sub in k.lower() for sub in retired)}
    assert missed, "the governed vocabulary adds nothing the retired map lacked"
    assert "WITHDRAWN" in set(missed.values()), (
        "WITHDRAWN was the stage the retired map could not reach at all; if no "
        "WITHDRAWN spelling is beyond it, this test has stopped proving anything")
    for spelling, stage in missed.items():
        assert L.pipeline_stage_request(f"Show {spelling} cases.")[0] == stage, spelling


def test_stage_temporal_execution_is_fixture_proven_only():
    """AND THE LIMIT OF THAT EVIDENCE, stated where it cannot be overlooked.

    The stage vocabulary resolving is one thing; a stage SERIES executing across
    real weekly history is another. The latter is proved only against
    `tests/fixtures/pipeline_history_5w`, because the configured production
    discovery root currently contains ZERO weekly extracts — every pipeline,
    stage and funnel question there answers "No weekly pipeline extracts are
    available", with ok=True.

    So Pipeline Stage temporal execution is FIXTURE-PROVEN,
    PRODUCTION-DATA-UNEXERCISED. This test fails if that stops being true in
    either direction: if the fixture disappears, or if production acquires
    extracts and this caveat is silently left standing.
    """
    fixture = _ROOT / "tests" / "fixtures" / "pipeline_history_5w"
    assert fixture.is_dir(), "the five-week fixture is the only stage-history evidence"
    weeks = sorted(p.name for p in fixture.iterdir() if p.is_dir())
    assert len(weeks) == 5, weeks

    from mi_agent_api import datasets as datasets_mod, pipeline_contract as pc
    from demo_platform import config as cfg

    import os
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    os.environ.update(cfg.mi_env(period_role="current"))
    production = pc.weekly_extract_inventory(
        datasets_mod._pipeline_discovery_root(), cfg.CLIENT_ID)
    assert production.get("uniqueWeeklyExtractsUsed") == 0, (
        "production now carries weekly extracts — the fixture-only caveat in the "
        "C6 record is out of date and must be re-measured against real data")


def test_governed_aliases_normalise_to_the_canonical_token():
    """A consumer never spell-matches: the reader returns OFFER, not "offer
    issued"."""
    for spelling, canon in (("declined", "WITHDRAWN"), ("cancelled", "WITHDRAWN"),
                            ("illustration", "KFI"), ("quote", "KFI"),
                            ("offer issued", "OFFER"), ("drawdown", "COMPLETED")):
        assert L.pipeline_stage_request(f"Show {spelling} cases.")[0] == canon


# --------------------------------------------------------------------------- #
# The axis, and what must NOT acquire it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "Show pipeline amount by stage over time.",
    "How has the pipeline changed by stage?",
    "Show pipeline stage balances over time.",
    "Show pipeline stage migration.",
    "What is the pipeline stage distribution?",
])
def test_the_stage_axis_is_named(question):
    stage, axis = L.pipeline_stage_request(question)
    assert axis is True and stage is None, (question, stage, axis)


def test_the_bare_word_stage_is_not_self_evidencing():
    """A canonical stage NAME is its own evidence; the word "stage" is ordinary
    English and needs an axis marker or a pipeline dataset behind it."""
    assert L.pipeline_stage_request("What stage is the securitisation at?") == (None, False)


@pytest.mark.parametrize("question", [
    "Show funded balance evolution by month.",
    "Which region gained the most cases since last month?",
    "How has the pipeline changed over time?",
    "Show pipeline amount evolution by week.",
    "What is the total funded balance?",
])
def test_unrelated_questions_acquire_no_stage_semantics(question):
    assert L.pipeline_stage_request(question) == (None, False), question


def test_a_disclaimed_stage_does_not_select():
    """Same bar the governed dataset owner already uses."""
    assert L.pipeline_stage_request("Show the balance, ignoring withdrawn cases.")[0] is None


def test_naming_both_a_stage_and_the_axis_narrows_rather_than_splits():
    stage, axis = L.pipeline_stage_request("How have offer-stage cases changed?")
    assert stage == "OFFER" and axis is True


def test_two_stages_resolve_in_funnel_order():
    """Matches the shipped handler's own precedence, from the governed bucket
    order rather than a dict's insertion order."""
    q = "How much is sitting at offer today and how much will complete?"
    assert L.pipeline_stage_request(q)[0] == "OFFER"


# --------------------------------------------------------------------------- #
# The claims the projection composes — existing structures, no new claim type
# --------------------------------------------------------------------------- #
def _interpret(question: str):
    import logging
    import os
    import warnings
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    from mi_agent_api import mi_service
    sem = {}
    for name in ("load_semantics", "_load_semantics", "semantics_for"):
        fn = getattr(mi_service, name, None)
        if callable(fn):
            try:
                sem = fn(cfg.CLIENT_ID) or {}
                break
            except TypeError:
                try:
                    sem = fn() or {}
                    break
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
    from question_interpretation import projection as proj
    return proj.project(question, semantics=sem, frame=None)


def test_a_specific_stage_is_carried_as_a_governed_filter_value():
    qi = _interpret("How have offer-stage cases changed?")
    dims = {(d.candidate_concept, d.role) for d in (qi.dimensions or [])}
    assert ("pipeline_stage", "filter") in {(k, str(r)) for k, r in dims}, dims
    values = {f.categorical_value for f in (qi.filters or [])}
    assert "OFFER" in values, values


def test_the_stage_axis_is_carried_as_a_grouping_dimension():
    qi = _interpret("Show pipeline amount by stage over time.")
    roles = {d.candidate_concept: str(d.role) for d in (qi.dimensions or [])}
    assert roles.get("pipeline_stage") == "grouping", roles
    assert not [f for f in (qi.filters or [])
                if str(f.categorical_value or "").upper() in CANONICAL]


def test_no_new_claim_type_was_introduced():
    """The representation is a COMPOSITION of the dimension and filter claims
    every other axis already uses. A funnel-shaped field would have made the
    contract route-specific, which is the thing the migration is removing."""
    from question_interpretation import schema as S
    assert not hasattr(S, "PipelineStageClaim")
    assert not hasattr(S, "FunnelClaim")
    qi = _interpret("Show the KFI trend by week.")
    assert not hasattr(qi, "funnel")
    assert not hasattr(qi, "pipeline_stage")


def test_an_unrelated_question_gains_no_stage_claim():
    qi = _interpret("Show funded balance evolution by month.")
    assert "pipeline_stage" not in {d.candidate_concept for d in (qi.dimensions or [])}

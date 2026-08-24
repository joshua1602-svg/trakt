"""Target-state closure — the two semantic facts the contract did not carry.

The compositional target says interpretation owns meaning once and nothing
downstream rereads the question. `migration_phase0/semantic_owner_inventory.py`
finds every place that still does, statically, across the whole generic
funded-book estate rather than one route at a time.

It found six concepts re-decided downstream. Four were already accounted for:

    source scope + caller precedence   carried since Phase 1G; the downstream
                                       reads are duplicates a conversion removes
    route shape                        recognition, which is a route's own claim
    ranking subject + direction        ONE route needs it -> left for migration
    whole-question delegation          two routes hand the sentence to a
                                       workflow that re-interprets it

Two were genuine generic gaps, and this module pins their closure:

    time window MAGNITUDE   the contract carried the WORDING of a window and
                            not how far back it reaches, so `period_movement`
                            and `period_change_analysis` both asked the owner
                            again for the number.
    dataset selection       nothing carried which TAPE the answer is built
                            from, so `chat_routing._dataset_for` re-derived it
                            over a wider vocabulary than the owner's — a
                            duplicate its own docstring names "THE SECOND
                            OWNER".
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@pytest.fixture(scope="module")
def env():
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    try:
        yield cfg.CLIENT_ID
    finally:
        os.environ.clear()
        os.environ.update(before)


@pytest.fixture(scope="module")
def semantics(env):
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.datasets import semantics_path
    return load_mi_semantics(semantics_path())


def project(question, semantics, **kw):
    from question_interpretation import projection
    return projection.project(question, semantics=semantics, **kw)


class TestTheTimeWindowCarriesItsMagnitude:
    """`trend_window` said a window was named. It did not say which one."""

    @pytest.mark.parametrize("question,periods", [
        ("Show balance over the last 6 months", 6),
        ("Show balance over the last 3 months", 3),
        ("How has the funded balance moved this year?", 12),
    ])
    def test_the_window_reaches_a_stated_number_of_periods(
            self, semantics, question, periods):
        qi = project(question, semantics)
        assert qi.time.trend_window.state == "filled"
        assert qi.time.window_periods == periods

    def test_a_governed_recency_is_flagged_as_governed(self, semantics):
        """"Recently" is not a countable span. A governed convention settles it,
        and the answer owes the reader a disclosure — which is not derivable
        from the period count alone, so it is carried separately."""
        qi = project("what changed recently?", semantics)
        assert qi.time.window_periods is not None
        assert qi.time.window_governed is True

    def test_a_stated_span_is_not_flagged_as_governed(self, semantics):
        qi = project("Show balance over the last 6 months", semantics)
        assert qi.time.window_governed is False

    def test_a_question_naming_no_window_carries_none(self, semantics):
        qi = project("What is the funded balance?", semantics)
        assert qi.time.trend_window.state == "empty"
        assert qi.time.window_periods is None

    def test_the_owner_and_the_contract_agree(self, semantics):
        """The contract carries the OWNER's number, not its own reading."""
        from mi_agent import period_request
        for question in ("Show balance over the last 6 months",
                         "How has the funded balance moved this year?",
                         "what changed recently?"):
            span = period_request.requested_span(question)
            qi = project(question, semantics)
            assert qi.time.window_periods == span.periods, question
            assert qi.time.window_governed is bool(span.governed), question

    def test_a_window_shorter_than_one_period_is_refused(self):
        from question_interpretation.schema import TimeClaim
        with pytest.raises(ValueError):
            TimeClaim(window_periods=0)


class TestTheDatasetClaimSaysWhichTape:
    """A question picks a TAPE, and separately a portfolio scope within it."""

    @pytest.mark.parametrize("question,dataset", [
        ("How many pipeline cases are there?", "pipeline"),
        ("What is the funded balance?", "funded"),
    ])
    def test_a_named_dataset_is_explicit(self, semantics, question, dataset):
        qi = project(question, semantics)
        assert qi.dataset.state == "filled"
        assert qi.dataset.dataset == dataset
        assert qi.dataset.provenance == "explicit_user"
        assert qi.dataset.stated_by_user is True

    def test_the_workspace_tab_no_longer_applies_when_the_question_is_silent(
            self, semantics):
        """RETIRED BEHAVIOUR, replaced deliberately.

        This asserted the tab filled the gap when the question named no dataset,
        and `caller_context` provenance recorded that it had. Natural-language
        MI is self-contained now: the governed default applies on every tab, and
        `caller_context` is unreachable on this axis.
        """
        for tab in (None, "pipeline", "forecast", "funded"):
            qi = project("Show balance by region", semantics, caller_dataset=tab)
            assert qi.dataset.dataset == "funded", tab
            assert qi.dataset.provenance == "default", tab
            assert qi.dataset.stated_by_user is False

    def test_neither_gives_the_funded_default(self, semantics):
        qi = project("Show balance by region", semantics)
        assert qi.dataset.dataset == "funded"
        assert qi.dataset.provenance == "default"

    def test_the_question_overrides_the_tab(self, semantics):
        qi = project("What is the funded balance?", semantics,
                     caller_dataset="pipeline")
        assert qi.dataset.dataset == "funded"
        assert qi.dataset.provenance == "explicit_user"

    def test_a_disclaimed_view_does_not_choose_the_tape(self, semantics):
        """B21's defect, at the contract boundary: the clause RULING A VIEW OUT
        was what selected it. "ignoring the forecast" loaded the forecast frame,
        which carries 12 of the funded book's 71 columns."""
        qi = project("the balance by vintage, ignoring the forecast", semantics)
        assert qi.dataset.dataset == "funded"

    def test_the_contract_agrees_with_the_owner(self, semantics):
        from mi_agent_api.workspace import resolve_dataset
        for question, tab in (("How many pipeline cases are there?", None),
                              ("Show balance by region", "pipeline"),
                              ("Show balance by region", None),
                              ("How many applications are there?", "funded"),
                              ("the balance by vintage, ignoring the forecast", None)):
            qi = project(question, semantics, caller_dataset=tab)
            # The tab is passed and must make no difference: the contract
            # carries the OWNER's answer, and the owner has no tab to read.
            assert qi.dataset.dataset == resolve_dataset(question), (question, tab)

    def test_the_dataset_axis_is_not_the_portfolio_axis(self, semantics):
        """Conflating them is how "the balance by seasoning segment excluding
        pipeline cases" reached a route narrowed to the very thing it excluded.
        A question can name one, the other, or both."""
        qi = project("How many pipeline cases are in the acquired book?", semantics)
        assert qi.dataset.dataset == "pipeline"
        assert qi.source_scope.scope == "acquired"

    def test_an_unknown_dataset_is_refused_by_the_schema(self):
        from question_interpretation.schema import DatasetClaim
        with pytest.raises(ValueError):
            DatasetClaim(state="filled", dataset="ledger")

    def test_a_filled_dataset_claim_must_name_one(self):
        from question_interpretation.schema import DatasetClaim
        with pytest.raises(ValueError):
            DatasetClaim(state="filled")


class TestTheOwnerStaysSingle:
    """The closure must not create the duplicate it removes."""

    def test_the_view_reading_lives_in_one_place(self):
        """One owner reads the vocabulary; everything else delegates to it.

        `resolve_dataset` is that owner. `resolve_active_view` is now a shim
        that forwards to it and reads nothing itself — so the assertion moved
        from "the resolver calls the extracted helper" to "the shim decides
        nothing at all", which is the stronger property.
        """
        text = (_REPO / "mi_agent_api/workspace.py").read_text(encoding="utf-8")
        shim = text[text.index("def resolve_active_view("):]
        shim = shim[:shim.index("\ndef ", 1)]
        assert "resolve_dataset(question)" in shim
        assert shim.count("undisclaimed_mention") == 0

        owner = text[text.index("def resolve_dataset("):]
        owner = owner[:owner.index("\ndef ", 1)]
        assert "view_named_by_question(question)" in owner

    def test_the_projection_reads_no_question_vocabulary(self):
        """The contract carries the owners' answers. It matches no phrases."""
        text = (_REPO / "question_interpretation/projection.py").read_text(
            encoding="utf-8")
        code = "\n".join(line for line in text.splitlines()
                         if not line.lstrip().startswith("#"))
        for word in ("pipeline", "forecast", "acquired", "vintage of"):
            assert f'"{word}"' not in code, word

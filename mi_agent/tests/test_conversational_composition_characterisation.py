#!/usr/bin/env python3
"""mi_agent/tests/test_conversational_composition_characterisation.py

CHARACTERISATION, not specification. Every assertion here pins what the estate
does TODAY so that the conversational analytical-composition sprint starts from
a measured baseline rather than a remembered one. Nothing in production changed
to make these pass; several of them pin behaviour the review classifies as a
DEFECT, and each of those says so in its docstring.

Read with ``migration_phase0/MI_CONVERSATIONAL_READINESS.md``. The harness that
produced the review's numbers is
``migration_phase0/conversational_readiness_probe.py``; this module asserts the
handful of facts the review's verdict actually rests on, so a change that moves
one of them fails here rather than silently invalidating the review.

Four groups:

  A. SCOPE RECONSTRUCTION — a governed population can be rebuilt from ``spec``
     and ``query_result.metadata`` alone and replayed on the unchanged
     deterministic executor. This is the GO criterion.

  B. MULTI-OUTPUT — ``metadata.measures_executed`` is machine-readable proof
     that every requested measure ran, and the grouped and pipeline paths honour
     a measure set whole.

  C. COMPOSITION DEFECTS — two seams where a combined request does not answer
     the question its atoms answer. Pinned so the fix is visible when it lands.

  D. GOVERNANCE — the two safety properties the review's verdict depends on:
     a bare money reference never becomes a predicate, and a follow-up turn
     that inherits nothing is never presented as if it had.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from migration_phase0 import conversational_readiness_probe as probe  # noqa: E402


@pytest.fixture(scope="module")
def funded():
    return probe.funded_tape()


@pytest.fixture(scope="module")
def pipeline():
    return probe.pipeline_tape()


def _ask(question, frame, dataset="funded"):
    return probe.observed(probe.ask(question, frame, dataset=dataset))


# --------------------------------------------------------------------------- #
# A. SCOPE RECONSTRUCTION — the readiness criterion
# --------------------------------------------------------------------------- #
class TestScopeReconstruction:
    """Can the population an answer was calculated over be rebuilt from the
    answer's own governed objects, without the question?"""

    def test_the_executed_population_is_on_the_spec_not_only_in_the_prose(self, funded):
        """`spec.filters` carries the population as field -> condition.

        The receipt's ``filtersApplied`` is PROSE ("London", "Current LTV > 40")
        and is not re-parseable; the spec is the machine form, and it is the one
        the API already returns to the caller.
        """
        result = probe.ask(
            "What is the funded balance for joint borrowers in the London "
            "region with LTV above 40%?", funded)
        scope = probe.reconstruct_scope(result)
        assert scope["population_filters"] == {
            "geographic_region_obligor": "London",
            "current_loan_to_value": {"op": "gt", "value": 40.0},
            "borrower_type": "Joint"}
        assert scope["dataset"] == "funded"
        assert scope["measure"] == "current_outstanding_balance"
        assert scope["aggregation"] == "sum"

    def test_execution_declares_which_fields_actually_narrowed(self, funded):
        """`applied_filter_fields` is EXECUTION evidence, not spec echo.

        An inheriting layer must inherit what RAN, never what was merely parsed,
        so the seam it reads has to be this one.
        """
        obs = _ask("What is the funded balance for joint borrowers in the "
                   "London region with LTV above 40%?", funded)
        assert set(obs["applied_filter_fields"]) == {
            "geographic_region_obligor", "current_loan_to_value", "borrower_type"}

    def test_a_reconstructed_population_replays_on_the_unchanged_executor(self, funded):
        """The whole GO question, in one assertion.

        A spec assembled from the reconstructed scope — carrying no natural
        language at all — executes on ``execute_mi_query`` and lands on exactly
        the rows the original question landed on, with a DIFFERENT measure.
        """
        result = probe.ask(
            "What is the funded balance for joint borrowers in the London "
            "region with LTV above 40%?", funded)
        original = probe.observed(result)["population"]
        scope = probe.reconstruct_scope(result)
        replay = probe.replay_scope(scope, funded,
                                    metric="current_loan_to_value",
                                    aggregation="weighted_avg")
        assert replay["loan_count"] == original
        assert set(replay["applied_filter_fields"]) == {
            "geographic_region_obligor", "current_loan_to_value", "borrower_type"}

    @pytest.mark.parametrize("case,mutation,expect_fields", [
        ("ADD", {"extra_filters": {"current_loan_to_value":
                                   {"op": "gt", "value": 80.0}}},
         {"geographic_region_obligor", "current_loan_to_value", "borrower_type"}),
        ("RESET", {"reset": True}, set()),
        ("GROUP", {"grouping": ["age_bucket"]},
         {"geographic_region_obligor", "current_loan_to_value", "borrower_type"}),
    ])
    def test_the_four_follow_up_operations_are_expressible_as_spec_edits(
            self, funded, case, mutation, expect_fields):
        """INHERIT / ADD / MODIFY / RESET need no new execution primitive.

        Each is an edit to ``spec.filters`` / ``spec.dimensions`` followed by the
        existing executor. That is why the review's estimate carries no
        calculation work.
        """
        result = probe.ask(
            "What is the funded balance for joint borrowers in the London "
            "region with LTV above 40%?", funded)
        scope = probe.reconstruct_scope(result)
        if mutation.pop("reset", False):
            scope = {**scope, "population_filters": {}}
        replay = probe.replay_scope(scope, funded,
                                    metric="current_outstanding_balance",
                                    aggregation="sum", **mutation)
        assert set(replay["applied_filter_fields"]) == expect_fields

    def test_two_bounds_on_one_field_need_folding_not_a_new_predicate_kind(
            self, funded):
        """`spec.filters` is field -> ONE condition, and `between` is governed.

        So ADD on an already-bounded field is a FOLD the resolver must perform
        (`> 40` ∧ `< 80` -> `between [40, 80]`), never two entries — the dict
        cannot hold two. The executor already runs the folded form, which is why
        the fold is a resolver invariant rather than an executor change.
        """
        scope = {"dataset": "funded",
                 "population_filters": {"borrower_type": "Joint"},
                 "grouping": []}
        replay = probe.replay_scope(
            scope, funded, metric="current_outstanding_balance",
            aggregation="sum",
            extra_filters={"current_loan_to_value":
                           {"op": "between", "value": [40, 80]}})
        assert replay["ok"]
        assert "current_loan_to_value" in replay["applied_filter_fields"]


# --------------------------------------------------------------------------- #
# B. MULTI-OUTPUT — what already composes
# --------------------------------------------------------------------------- #
class TestMultiOutputThatWorks:
    """Same-turn multi-measure over one population is shipped, and it publishes
    machine-readable proof that every requested output ran."""

    def test_a_measure_set_reports_what_it_executed(self, funded):
        obs = _ask("For loans in the London region, give me count, balance and "
                   "weighted average LTV.", funded)
        assert obs["ok"]
        assert obs["measures_requested"] == [
            "loan_count", "current_outstanding_balance", "current_loan_to_value"]
        assert obs["measures_executed"] == [
            "loan_count", "current_outstanding_balance", "current_loan_to_value"]
        assert obs["measures_unavailable"] == []

    def test_a_measure_set_honours_the_population(self, funded):
        obs = _ask("For loans in the London region, give me count, balance and "
                   "weighted average LTV.", funded)
        assert obs["applied_filter_fields"] == ["geographic_region_obligor"]

    def test_a_grouped_measure_set_composes(self, funded):
        obs = _ask("By borrower type, show count, balance and WA LTV.", funded)
        assert obs["ok"]
        assert obs["measures_executed"] == [
            "loan_count", "current_outstanding_balance", "current_loan_to_value"]
        assert obs["dimension"] == "borrower_type"

    def test_the_pipeline_uses_the_same_measure_set_machinery(self, pipeline):
        """The executor is dataset-parametric: nothing about the measure set is
        funded-specific, and the receipt names the pipeline as the population."""
        obs = _ask("For pipeline loans at the OFFER stage, give me case count, "
                   "balance and weighted average LTV.", pipeline,
                   dataset="pipeline")
        assert obs["ok"]
        assert obs["dataset"] == "pipeline"
        assert obs["measures_executed"] == [
            "loan_count", "current_outstanding_balance", "current_loan_to_value"]
        assert "pipeline_stage" in obs["applied_filter_fields"]


# --------------------------------------------------------------------------- #
# C. COMPOSITION DEFECTS — pinned, not fixed
# --------------------------------------------------------------------------- #
class TestCompositionDefects:
    """Two seams where a combined request does not answer what its atoms answer.

    Both are recorded in ``MI_CONVERSATIONAL_READINESS.md`` §C. Neither is fixed
    here: this sprint is a readiness review, and a finding that a function could
    be widened is a finding, not permission.
    """

    def test_atoms_are_green_so_these_are_composition_failures_not_atomic_ones(
            self, funded):
        for question in ("How many loans have a joint borrower type?",
                         "What is the funded balance for joint borrowers?",
                         "What is the weighted average LTV for joint borrowers?"):
            assert _ask(question, funded)["ok"], question

    def test_the_measure_set_path_drops_the_borrower_population(self, funded):
        """DEFECT (fails closed). `_measure_set_recognizer` resolves filters with
        `_parse_filters` and `_parse_categorical_filter`; neither owns the
        joint/sole vocabulary. `_borrower_structure_filter` does, and has three
        call sites — none of them this one — so the population is never asked
        for. Geography and numeric thresholds survive the same path.

        It REFUSES rather than answering the whole book, because the population
        facet is raised from the question and then found unapplied.
        """
        obs = _ask("For joint borrowers, give me loan count and funded balance.",
                   funded)
        assert obs["measures_executed"] == ["loan_count",
                                            "current_outstanding_balance"]
        assert obs["filters"] == {}, "borrower population dropped (the defect)"
        assert not obs["ok"], "but it must not answer over the whole book"
        assert obs["guard"] == "clarify"

    def test_a_clause_scoped_filter_is_promoted_to_the_shared_population(
            self, funded):
        """DEFECT (SILENT). `MIQuerySpec` carries ONE filter set, so a predicate
        belonging to the third clause has nowhere to live but the population all
        three clauses share.

        The atoms are 69 / 69 / 39 loans. The combined request answers count and
        balance over 39 — the LTV>40 cohort — and never produces the third
        figure at all, with ``ok`` true and a receipt that names the extra
        filter without saying it was only ever asked for once.

        This is the one shape in the review that fails SILENTLY, and it is why
        the verdict is CONDITIONAL rather than GO.
        """
        shared = _ask("How many loans are in the London region?", funded)
        narrow = _ask("What is the balance in the London region with LTV above "
                      "40%?", funded)
        combined = _ask("For loans in the London region, what is the loan count, "
                        "the balance, and how much of that balance has LTV "
                        "above 40%?", funded)
        assert shared["population"] != narrow["population"]
        assert combined["ok"], "it answers"
        assert combined["population"] == narrow["population"], (
            "over the clause-scoped population, not the shared one")
        assert "current_loan_to_value" in combined["applied_filter_fields"]
        assert len(combined["measures_executed"]) == 2, (
            "the third requested output was never executed")

    def test_a_bare_place_name_beside_another_predicate_is_dropped(self, funded):
        """ATOMIC defect, recorded so it is not read as a compositional one.

        "in the London region" binds; "in London" beside a borrower predicate
        does not. It fails closed — the geographic facet is raised and lost — so
        no broadened figure is published. Belongs to the atomic remediation
        sprint, not to this one.
        """
        assert _ask("What is the balance in London?", funded)["filters"] == {
            "geographic_region_obligor": "London"}
        beside = _ask("What is the funded balance for joint borrowers in "
                      "London?", funded)
        assert "geographic_region_obligor" not in beside["filters"]
        assert not beside["ok"], "the drop must fail closed"


# --------------------------------------------------------------------------- #
# D. GOVERNANCE — the safety properties the verdict rests on
# --------------------------------------------------------------------------- #
class TestGovernanceProperties:

    @pytest.mark.parametrize("question", [
        "Of the £38m, what is the weighted average LTV?",
        "Of the 38 million, what is the weighted average LTV?",
        "What is the weighted average LTV of the £38m?",
        "Of the 43 loans, what is the weighted average LTV?",
        "Of the £38m, how many loans are there?",
    ])
    def test_a_money_reference_never_becomes_a_predicate(self, funded, question):
        """"Of the £38m" is a REFERENCE to a prior population, and the parser
        already refuses to read it as a filter: a bare number with no governed
        comparator phrase beside it produces no predicate.

        Governance failure mode F is therefore structurally prevented BEFORE the
        conversational layer exists, rather than being something it must add.
        """
        assert _ask(question, funded)["filters"] == {}

    def test_an_explicit_threshold_still_binds(self, funded):
        """The control. Without it the test above proves only that the parser is
        deaf to numbers."""
        assert _ask("What is the balance for loans above £250,000?",
                    funded)["filters"] == {
            "current_outstanding_balance": {"op": "gt", "value": 250000.0}}

    def test_a_follow_up_turn_today_answers_the_whole_book(self, funded):
        """The BASELINE the sprint has to improve on, stated as a fact.

        The API is stateless, so "what is their WA LTV?" carries no population
        and is answered over the entire funded book. The receipt says so —
        "entire funded portfolio" — which is the only reason this is a
        disclosed broadening rather than a silent one.
        """
        first = _ask("What is the funded balance for joint borrowers in the "
                     "London region?", funded)
        follow = _ask("What is their weighted average LTV?", funded)
        assert first["applied_filter_fields"]
        assert follow["ok"]
        assert follow["applied_filter_fields"] == []
        assert follow["population"] == follow["population_total"]
        assert "entire funded portfolio" in (follow["receipt"] or "")

    def test_the_receipt_names_the_dataset_it_broadened_over(self, pipeline):
        """The same broadening on the pipeline names the pipeline, so a
        conversational turn that crosses datasets cannot be misread as funded."""
        follow = _ask("What is the weighted average LTV?", pipeline,
                      dataset="pipeline")
        assert "entire pipeline" in (follow["receipt"] or "")


# --------------------------------------------------------------------------- #
# E. PRESENTATION — the renderer reads the result, never the economics
# --------------------------------------------------------------------------- #
class TestPresentationIsDownstream:

    def test_the_chart_factory_receives_only_the_executed_result(self, funded):
        """`create_mi_chart(result, semantics)` takes no question, no frame and
        no filters. A conversational "chart that" therefore needs no
        re-execution of economics — only a spec whose ``chart_type`` differs.
        """
        import inspect

        from mi_agent.mi_chart_factory import create_mi_chart

        params = list(inspect.signature(create_mi_chart).parameters)
        assert params[:2] == ["result", "semantics"]
        assert "question" not in params and "filters" not in params

    def test_one_result_already_produces_several_typed_artifacts(self, funded):
        """The envelope's ``artifacts`` is a LIST of typed presentations of ONE
        governed result — so text / table / chart is already a renderer choice
        downstream of the same deterministic figure."""
        from mi_agent_api.adapters import adapt_workflow_result

        workflow = probe.ask("What is the balance by borrower type?", funded)
        envelope = adapt_workflow_result(workflow, portfolio_id="client_001",
                                         as_of=None)
        kinds = {a.get("type") for a in envelope["artifacts"]}
        assert {"chart", "table"} <= kinds
        assert envelope["answer"]

    def test_a_presentation_verb_is_read_as_a_measure_today(self, funded):
        """DEFECT (fails closed). "put it in a table" reaches the measure
        resolver as a requested measure and is refused as ungoverned.

        Recorded because it is the one place a presentation request touches the
        ECONOMICS path, and the review's Stage 3 has to move that decision above
        the parser rather than teach the parser more verbs.
        """
        obs = _ask("For loans in the London region, give me the balance and put "
                   "it in a table.", funded)
        assert not obs["ok"]
        assert "put it" in (obs["error"] or "")

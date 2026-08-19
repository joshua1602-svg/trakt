"""The governed ANALYTICAL INTENT BOUNDARY — families, the lending ruling, and
the fail-closed safety rule.

Three properties are asserted here, and they are the three the boundary exists
to hold:

1. **It introduces no analytic.** Every capability and route the six families
   declare already exists. The boundary can route a question to a governed
   answer; it can never invent one.
2. **The lending ruling is governed, contextual and config-driven.** NEW / RECENT
   / FRONT BOOK / BACK BOOK are thresholds in configuration, resolved to a
   POPULATION or an ORIGINATION FLOW by analytical context — never by a global
   mapping and never by matching a sentence.
3. **It fails closed.** A materially analytical question that no governed route
   owns is refused, not answered from one snapshot of the loan tape. A question
   that is NOT materially analytical keeps exactly the answer it had.
"""

from __future__ import annotations

import pytest

from mi_agent import seasoning as season
from mi_workflows.analytical import intent as intent_mod
from mi_workflows.analytical import populations as pops
from mi_workflows.analytical.registry import CAPABILITIES


class _Spec:
    """The two governed flags the boundary is allowed to settle, and no more."""

    def __init__(self, **kw):
        self.risk_limit_query = kw.get("risk_limit_query", False)
        self.forecast_mode = kw.get("forecast_mode")
        self.metric = kw.get("metric")
        self.dimension = kw.get("dimension")


# --------------------------------------------------------------------------- #
# §1 / §2 / §5 — six families, governed operations, existing owners
# --------------------------------------------------------------------------- #
class TestFamilyDeclarations:

    def test_exactly_the_six_governed_families(self):
        assert set(intent_mod.FAMILIES) == set(intent_mod.ALL_FAMILIES)
        assert len(intent_mod.ALL_FAMILIES) == 6

    def test_every_family_declares_at_least_one_operation(self):
        for family in intent_mod.FAMILIES.values():
            assert family.operations, f"{family.name} governs no operation"

    def test_every_declared_capability_already_exists(self):
        """The guarantee that no new mathematics enters through this door."""
        known = {c.id for c in CAPABILITIES.all()}
        for family in intent_mod.FAMILIES.values():
            unknown = [c for c in family.capabilities if c not in known]
            assert not unknown, (
                f"{family.name} names capabilities that do not exist: {unknown}")

    def test_every_declared_route_already_exists(self):
        from mi_agent_api.chat_routing import REGISTRY

        known = set(REGISTRY.names())
        for family in intent_mod.FAMILIES.values():
            unknown = [r for r in family.routes if r not in known]
            assert not unknown, (
                f"{family.name} names routes that do not exist: {unknown}")

    def test_the_families_a_question_names_are_the_families_it_is_about(self):
        cases = {
            "What is the mix of the book by region?": intent_mod.FAMILY_MIX_PROFILE,
            "How much is at offer?": intent_mod.FAMILY_PIPELINE,
            "Which limits have the least headroom?":
                intent_mod.FAMILY_LIMITS_CONCENTRATION,
            "What will the funded balance be next quarter?":
                intent_mod.FAMILY_FORECAST_PROJECTION,
            "How has the balance moved?": intent_mod.FAMILY_MOVEMENT_TREND,
            "Which vintage is weakest?": intent_mod.FAMILY_VINTAGE_COHORT,
        }
        for question, family in cases.items():
            assert family in intent_mod.classify(question).families, question


# --------------------------------------------------------------------------- #
# §4 — the governed lending ruling
# --------------------------------------------------------------------------- #
class TestLendingRuling:

    def test_the_four_windows_carry_the_ruling_thresholds(self):
        config = season.load_seasoning_config()
        windows = {w.key: w for w in config.lending_windows()}
        assert windows[season.LENDING_NEW].max_months == 1
        assert windows[season.LENDING_RECENT].max_months == 3
        assert windows[season.LENDING_FRONT_BOOK].max_months == 12
        assert windows[season.LENDING_BACK_BOOK].after_months == 12

    def test_the_thresholds_come_from_configuration(self, tmp_path):
        path = tmp_path / "buckets.yaml"
        path.write_text(
            "seasoning:\n"
            "  front_book_max_months: 18\n"
            "  lending_windows:\n"
            "    new_max_months: 2\n"
            "    recent_max_months: 6\n"
            "  buckets: []\n", encoding="utf-8")
        config = season.load_seasoning_config(path)
        windows = {w.key: w for w in config.lending_windows()}
        assert windows[season.LENDING_NEW].max_months == 2
        assert windows[season.LENDING_RECENT].max_months == 6
        assert windows[season.LENDING_BACK_BOOK].after_months == 18

    def test_the_windows_are_nested_not_a_partition(self):
        """Every NEW loan is also RECENT and also FRONT BOOK. That is the point:
        "new lending" and "the front book" are different questions."""
        config = season.load_seasoning_config()
        windows = {w.key: w for w in config.lending_windows()}
        assert (windows[season.LENDING_NEW].max_months
                < windows[season.LENDING_RECENT].max_months
                < windows[season.LENDING_FRONT_BOOK].max_months)

    def test_front_and_back_keep_the_predicate_they_already_had(self):
        """A question that resolved to one of them before resolves to exactly
        the same rows now — the binary partition is untouched."""
        for key in (season.LENDING_FRONT_BOOK, season.LENDING_BACK_BOOK):
            spec = pops.lending_window_population(key)
            assert set(spec.filters) == {season.SEASONING_SEGMENT_FIELD}

    def test_new_lending_is_not_globally_mapped_to_a_segment(self):
        """§4 is explicit: do NOT globally map "new lending" to one role. The
        segment vocabulary — which selects a population EVERYWHERE in the stack —
        must not have gained it."""
        assert season.segments_named("our new lending this month") == []
        assert season.segments_named("recent lending") == []
        assert season.lending_windows_named("our new lending this month") == [
            season.LENDING_NEW]

    def test_windows_are_ordered_by_position_in_the_question(self):
        """The first population named is the subject, the second the comparand.
        Reversing them reverses the sign of every delta reported."""
        assert season.lending_windows_named(
            "how does our new lending compare with the back book?") == [
                season.LENDING_NEW, season.LENDING_BACK_BOOK]
        assert season.lending_windows_named(
            "how does the back book compare with our new lending?") == [
                season.LENDING_BACK_BOOK, season.LENDING_NEW]

    @pytest.mark.parametrize("question", [
        "How has the risk profile of our new lending changed?",
        "What are the characteristics of recent lending?",
        "Compare the credit profile of new business with the back book.",
    ])
    def test_a_profile_context_makes_lending_a_population(self, question):
        assert intent_mod.classify(question).lending_role == intent_mod.ROLE_POPULATION

    @pytest.mark.parametrize("question", [
        "What is our new lending run rate?",
        "What is the volume of new lending per month?",
        "What is the volume of new business each month?",
    ])
    def test_a_flow_context_makes_lending_an_origination_flow(self, question):
        assert intent_mod.classify(question).lending_role == intent_mod.ROLE_FLOW

    def test_the_role_is_not_settled_by_matching_the_sentence(self):
        """The SAME words carry different roles in different analytical contexts,
        which is the whole reason the ruling is contextual."""
        population = intent_mod.classify(
            "What is the risk profile of our new lending?")
        flow = intent_mod.classify("What is our new lending run rate?")
        assert population.lending_windows == flow.lending_windows
        assert population.lending_role != flow.lending_role


# --------------------------------------------------------------------------- #
# §6 — materially analytical, and what is deliberately not
# --------------------------------------------------------------------------- #
class TestMateriality:

    @pytest.mark.parametrize("question", [
        "balance by region",
        "What is the total balance?",
        "weighted average LTV by region",
        "Show me balance by LTV band",
        "What is the largest single loan exposure?",
        "How many loans are on the book?",
        "What is the balance of the front book?",
    ])
    def test_a_point_in_time_question_is_not_materially_analytical(self, question):
        """These keep exactly the answer they have always had. A boundary that
        refused them would have traded a working capability for safety."""
        assert not intent_mod.classify(question).materially_analytical

    @pytest.mark.parametrize("question", [
        "How many loans are we completing at the moment?",
        "What completion rate are we running at?",
        "Where are we closest to our limits?",
        "Which of our limits are most at risk?",
        "How has the profile of our new lending changed?",
        "When will we reach £100m?",
        "Are direct and acquired balances developing differently over time?",
    ])
    def test_an_analytical_question_is_recognised_as_one(self, question):
        assert intent_mod.classify(question).materially_analytical


# --------------------------------------------------------------------------- #
# §5 / §8 — governed flag normalisation
# --------------------------------------------------------------------------- #
class TestGovernedFlags:

    def test_a_limits_question_reaches_the_limits_route(self):
        spec = _Spec()
        reading, applied = intent_mod.settle("Where are we closest to our limits?",
                                             spec)
        assert applied == {intent_mod.FLAG_RISK_LIMIT: True}
        assert spec.risk_limit_query is True

    def test_a_run_rate_question_reaches_the_run_rate_route(self):
        spec = _Spec()
        _reading, applied = intent_mod.settle(
            "What completion rate are we currently running at?", spec)
        assert applied == {intent_mod.FLAG_FORECAST_MODE: "extrapolation"}

    def test_a_settled_parse_is_never_second_guessed(self):
        spec = _Spec(forecast_mode="milestone")
        _reading, applied = intent_mod.settle(
            "At the current run rate, when do we reach £100m?", spec)
        assert intent_mod.FLAG_FORECAST_MODE not in applied
        assert spec.forecast_mode == "milestone"

    def test_a_count_question_is_not_handed_to_a_currency_capability(self):
        """The governed run-rate capability produces a currency rate and nothing
        else. Answering "how many loans are we completing?" with a pounds figure
        would be a measure substitution, so the boundary declines and the
        fail-closed rule applies instead."""
        spec = _Spec()
        reading, applied = intent_mod.settle(
            "How many loans are we completing at the moment?", spec)
        assert reading.counts_requested is True
        assert applied == {}

    def test_a_non_analytical_question_settles_nothing(self):
        spec = _Spec()
        _reading, applied = intent_mod.settle("balance by region", spec)
        assert applied == {}


# --------------------------------------------------------------------------- #
# §7 — the fail-closed safety rule
# --------------------------------------------------------------------------- #
class TestFailClosed:

    #: What the generic point-in-time executor always produces: one governed
    #: snapshot of the funded tape, no forecast, no limits, no pipeline.
    POINT_IN_TIME = {"dataset": "funded", "periods": 1, "forecast": False,
                     "limits": False, "grouping": "", "populations": 0}

    @pytest.mark.parametrize("question", [
        "How many loans are we completing at the moment?",
        "What completion rate are we running at?",
        "Where are we closest to our limits?",
        "Which of our limits are most at risk?",
    ])
    def test_the_four_measured_failures_are_refused(self, question):
        """Every one of these returned a confident, plausible, wrong figure with
        ok=True and a green guard. None may do so again."""
        reading = intent_mod.classify(question)
        unmet = intent_mod.unmet_requirements(reading, evidence=self.POINT_IN_TIME)
        assert unmet, f"{question!r} would still fall through"

    def test_a_refusal_says_what_was_understood_and_what_was_missing(self):
        reading = intent_mod.classify("Where are we closest to our limits?")
        unmet = intent_mod.unmet_requirements(reading, evidence=self.POINT_IN_TIME)
        message = intent_mod.refusal_message(reading, unmet)
        assert "limits concentration" in message
        assert "limit schedule" in message
        assert "NOT substituted" in message

    def test_a_comparison_the_executor_really_did_make_is_not_refused(self):
        """Grouped on the governed dimension that partitions the two populations,
        with both sides present, IS the comparison asked for — reached by another
        mechanism. Refusing it would lose a capability the product has."""
        reading = intent_mod.classify(
            "How different is the risk profile of the front book "
            "versus the back book?")
        assert intent_mod.REQ_POPULATION_COMPARISON in reading.requirements
        evidence = dict(self.POINT_IN_TIME,
                        grouping=season.SEASONING_SEGMENT_FIELD)
        assert intent_mod.unmet_requirements(reading, evidence=evidence) == []

    def test_a_pipeline_question_is_never_satisfied_by_the_funded_tape(self):
        reading = intent_mod.classify("How much is at offer right now?")
        assert intent_mod.REQ_PIPELINE_DATASET in reading.requirements
        assert intent_mod.unmet_requirements(
            reading, evidence=dict(self.POINT_IN_TIME, grouping="anything"))

    def test_a_pipeline_answer_from_the_pipeline_extract_satisfies_it(self):
        reading = intent_mod.classify("How much is at offer right now?")
        assert intent_mod.unmet_requirements(
            reading, evidence={"dataset": "pipeline", "periods": 2}) == []

    def test_a_rate_always_needs_two_snapshots(self):
        reading = intent_mod.classify("What is our completion run rate?")
        assert intent_mod.REQ_PERIOD_COMPARISON in reading.requirements

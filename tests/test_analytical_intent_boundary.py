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

    def test_a_rate_question_another_route_owns_is_not_diverted(self):
        """`forecast_mode` is a flag three other recognisers DECLINE on, so
        setting it on a question they own would divert it to a route that cannot
        answer it. Only a PIPELINE run rate — the completion flow — is handed to
        the completion-run-rate capability."""
        spec = _Spec()
        reading, applied = intent_mod.settle(
            "How has concentration by region changed per month?", spec)
        assert intent_mod.SIGNAL_RUN_RATE in reading.signals
        assert intent_mod.FAMILY_PIPELINE not in reading.families
        assert applied == {}
        assert spec.forecast_mode is None


class TestStructuralInvariants:

    QUESTIONS = [
        "balance by region",
        "How has the profile of our new lending changed over the last few months?",
        "How much is at offer and how much completes, and when?",
        "Where are we closest to our limits?",
        "When do we reach £100m at the current run rate?",
        "Are direct and acquired balances developing differently over time?",
        "How does the front book compare with our older lending on risk?",
        "How has concentration by region changed per month?",
        "What is our new lending run rate?",
        "Which vintage is weakest?",
    ]

    @pytest.mark.parametrize("question", QUESTIONS)
    def test_no_operation_is_reported_outside_its_family(self, question):
        """An operation names something a family governs. Reporting one whose
        family was not recognised would be describing an analysis nobody owns."""
        reading = intent_mod.classify(question)
        governed = set()
        for family in reading.families:
            governed |= set(intent_mod.FAMILIES[family].operations)
        stray = set(reading.operations) - governed
        assert not stray, f"{question!r} reports ungoverned operation(s) {stray}"

    @pytest.mark.parametrize("question", QUESTIONS)
    def test_a_recognised_question_always_names_a_governed_owner(self, question):
        reading = intent_mod.classify(question)
        if reading.recognised:
            assert reading.owners

    @pytest.mark.parametrize("question", QUESTIONS)
    def test_classification_is_deterministic(self, question):
        assert intent_mod.classify(question) == intent_mod.classify(question)


class TestOwnershipDeference:
    """§5 — existing route ownership stays valid. The family layer must not force
    a question through the analytical layer unnecessarily."""

    def _plan(self, question, spec=None, frame=None):
        from mi_workflows.analytical import planner as planner_mod

        return planner_mod.plan_for(question, spec=spec, frame=frame)

    def test_a_single_measure_series_stays_with_the_evolution_route(self):
        """"Show the balance evolution for the front book" asks for one measure
        across every retained period, and `evolution` answers exactly that. A
        two-snapshot movement is LESS than a series."""
        spec = _Spec(metric="current_outstanding_balance")
        spec.chart_type = "line"
        spec.x = "reporting_date"
        assert self._plan("Show the balance evolution for the front book",
                          spec=spec) is None

    def test_a_series_question_that_also_wants_the_composition_does_not_defer(self):
        """A single-measure series does not carry what the book is made of."""
        spec = _Spec(metric="current_outstanding_balance")
        spec.chart_type = "line"
        spec.x = "reporting_date"
        plan = self._plan(
            "How has the profile of our new lending changed over time?", spec=spec)
        assert plan is not None
        assert plan.intent == "origination_profile_change"

    def test_a_comparison_the_parse_already_resolved_stays_with_the_executor(self):
        """A measure grouped on the dimension that partitions the two named
        populations IS the comparison asked for. Replacing it with a narrative
        would be a trade, not an improvement."""
        spec = _Spec(metric="current_loan_to_value",
                     dimension=season.SEASONING_SEGMENT_FIELD)
        assert self._plan("Is the credit quality of new origination better or "
                          "worse than the back book?", spec=spec) is None

    def test_that_deference_lifts_when_the_question_needs_more_than_one_snapshot(self):
        """The executor reads ONE snapshot. A question that also needs two is not
        answerable by grouping, however the parse was resolved."""
        spec = _Spec(metric="current_outstanding_balance",
                     dimension=season.SEASONING_SEGMENT_FIELD)
        plan = self._plan("How has the front book balance moved relative to the "
                          "back book over the last few months?", spec=spec)
        assert plan is not None
        assert plan.intent == "population_movement_comparison"


class TestOperationCoverage:
    """Every governed operation §2 declares must be reachable from a question.

    A declaration nothing can produce is decoration. This is the check that the
    operation set is a description of what the boundary recognises rather than a
    wish list — and it fails loudly if a future edit strands one.
    """

    #: One ordinary question per governed operation. These are NOT production
    #: vocabulary and nothing reads them at runtime; they exist so the
    #: declarations in FAMILIES cannot quietly become unreachable.
    REACHES = {
        intent_mod.OP_SNAPSHOT: "How does the risk profile of the front book compare with the back book?",
        intent_mod.OP_COMPOSITION: "What is the mix of the book?",
        intent_mod.OP_COMPARISON: "How does the front book profile compare with the back book?",
        intent_mod.OP_CHANGE: "How has the profile of the book changed?",
        intent_mod.OP_DIVERGENCE: "How has the front book profile changed relative to the back book?",
        intent_mod.OP_ATTRIBUTION: "How has the mix changed and what drove it?",
        intent_mod.OP_STOCK: "How much is at offer?",
        intent_mod.OP_MOVEMENT: "How many loans are we completing at the moment?",
        intent_mod.OP_CONVERSION: "How much of the pipeline is expected to complete?",
        intent_mod.OP_RUN_RATE: "What is our completion run rate?",
        intent_mod.OP_EXPECTED_COMPLETION: "How much of the offer pipeline do we expect to complete?",
        intent_mod.OP_TIMING: "When do we expect the offer pipeline to complete?",
        intent_mod.OP_MIX: "What is the profile of the offer pipeline?",
        intent_mod.OP_CONCENTRATION: "Are we within our concentration limits?",
        intent_mod.OP_STATUS: "Are we within our concentration limits?",
        intent_mod.OP_HEADROOM: "Which limits have the least headroom?",
        intent_mod.OP_RANKING: "Which limits have the least headroom?",
        intent_mod.OP_FORECAST_BREACH: "Are we likely to breach any limits next quarter?",
        intent_mod.OP_PROJECT_VALUE: "What will the funded balance be?",
        intent_mod.OP_MILESTONE: "When will we reach £100m?",
        intent_mod.OP_HORIZON: "When will we reach £100m?",
        intent_mod.OP_SCENARIO: "What if conversion improved, when would we reach £100m?",
        intent_mod.OP_DELTA: "How has the balance changed?",
        intent_mod.OP_TREND: "What is the balance trend?",
        intent_mod.OP_ACCELERATION: "Is balance growth accelerating?",
        intent_mod.OP_EVOLUTION: "How have the vintages evolved?",
    }

    def test_every_declared_operation_is_declared_by_some_family(self):
        declared = set()
        for family in intent_mod.FAMILIES.values():
            declared |= set(family.operations)
        assert set(self.REACHES) == declared, (
            "the coverage table and the family declarations disagree: "
            f"{set(self.REACHES) ^ declared}")

    @pytest.mark.parametrize("operation", sorted(REACHES))
    def test_every_declared_operation_is_reachable(self, operation):
        question = self.REACHES[operation]
        assert operation in intent_mod.classify(question).operations, (
            f"{operation} is declared but no question reaches it")

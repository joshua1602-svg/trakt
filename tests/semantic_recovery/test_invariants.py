"""SEMANTIC RECOVERY — the architectural invariants, written FIRST.

Each class pins one invariant from the recovery sprint (I1–I7). They were
written before any consolidation and are EXPECTED TO FAIL at the recovery
baseline (harness 748c6f3 / product ea8c65b): a passing test here is a test of
the consolidated owner, not of the current code. Where a test already passes
at baseline that is stated in its docstring and reported, not hidden.

None of these tests reads the frozen 135-question bank. The questions below are
either plain lender wording or the sprint's own worked examples; none is copied
from the bank as a target and no test asserts a bank verdict.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from tests.semantic_recovery.conftest import contract_for, frames_for

_REPO = Path(__file__).resolve().parent.parent.parent


def _fn_source(module, name):
    return inspect.getsource(getattr(module, name))


def _names_referenced(module, name):
    tree = ast.parse(inspect.getsource(getattr(module, name)))
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}


def _calls(module, name):
    tree = ast.parse(inspect.getsource(getattr(module, name)))
    return {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
            for c in ast.walk(tree) if isinstance(c, ast.Call)}


def _string_constants(module, name):
    tree = ast.parse(inspect.getsource(getattr(module, name)))
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


# =========================================================================== #
# I1 — ONE normaliser. "loan-to-value" == "loan to value" everywhere, and a row
#      noun INSIDE a bound measure never forces a COUNT.
# =========================================================================== #
class TestI1OneNormaliser:
    def test_the_owner_exists_and_is_offset_preserving(self):
        from question_interpretation import normalise as N
        out = N.normalise_question("Loan-to-Value  drift?")
        assert out.startswith("loan to value")
        # Offsets are the currency every span owner trades in, so the owner
        # may not shift them: a hyphen becomes one space, in place.
        assert len(out) == len("Loan-to-Value  drift?")

    def test_hyphen_and_space_spellings_parse_identically(self, parse):
        a = parse("Has the book's loan-to-value drifted across reporting periods?").spec
        b = parse("Has the book's loan to value drifted across reporting periods?").spec
        assert (a.metric, a.aggregation, a.intent, a.dimension) == \
               (b.metric, b.aggregation, b.intent, b.dimension)
        assert a.metric == "current_loan_to_value"
        assert a.aggregation != "count"

    def test_a_row_noun_inside_a_bound_measure_is_not_a_count(self, semantics):
        from mi_agent import llm_query_parser as P
        assert P._counts_a_row_noun(
            "has the book's loan to value drifted across reporting periods?",
            semantics) is False
        assert P._counts_a_row_noun("how many loans are in the book?", semantics) is True

    def test_every_reader_delegates_to_the_one_owner(self):
        """`borrowing_base_query._normalise` and `intent.normalise` each had a
        private hyphen rule. They must now be the owner's, so no reader can
        see a hyphen the parser did not."""
        from mi_agent_api import borrowing_base_query as BB
        from mi_workflows.analytical import intent as I
        assert "loan to value" in BB._normalise("Loan-to-Value")
        assert " loan to value " in I.normalise("loan-to-value")
        assert "normalise_question" in _calls(BB, "_normalise")
        assert "normalise_question" in _calls(I, "normalise")
        assert "normalise_question" in _calls(
            __import__("mi_agent.llm_query_parser", fromlist=["x"]),
            "_deterministic_parse_unchecked")


# =========================================================================== #
# I2 — ONE period-pair owner for funded_bridge / period_movement /
#      temporal_compare. Explicit wins; "last month to this month" is the
#      previous observation → current under gap governance; never the earliest.
# =========================================================================== #
class TestI2OnePeriodPairOwner:
    @staticmethod
    def _unspecified(interpret):
        _, qi = interpret("How has the funded balance moved?")
        return qi

    def test_unspecified_movement_is_current_versus_previous(self, interpret):
        from mi_agent.period_change.models import METHOD_CURRENT_VS_PREVIOUS
        from mi_agent_api import analytical_plan as plan
        frames = frames_for(["2025-06-30", "2025-07-31", "2025-09-30", "2025-12-31"])
        start, end, res = plan.resolve_period_pair(frames, self._unspecified(interpret))
        assert start is frames[-2] and end is frames[-1]
        assert res.resolution_method == METHOD_CURRENT_VS_PREVIOUS

    def test_last_month_to_this_month_is_the_previous_observation_disclosed(self, interpret):
        from mi_agent.period_change.models import METHOD_MONTH_ON_MONTH
        from mi_agent_api import analytical_plan as plan
        _, qi = interpret("Bridge the movement in the book from last month to this month.")
        frames = frames_for(["2025-06-30", "2025-07-31", "2025-09-30", "2025-12-31"])
        start, end, res = plan.resolve_period_pair(frames, qi)
        assert end is frames[-1]
        assert start is frames[-2], "never the earliest, never a wider window"
        assert res.resolution_method == METHOD_MONTH_ON_MONTH
        # 2025-09-30 is not the month before 2025-12-31: the adjustment is
        # DISCLOSED on the resolution, not silently absorbed.
        assert res.start_adjusted is True
        assert res.adjustment_notes

    def test_last_month_beyond_the_gap_ceiling_refuses_not_widens(self, interpret):
        from mi_agent.period_change.models import PeriodChangeFailure
        from mi_agent_api import analytical_plan as plan
        _, qi = interpret("Bridge the movement in the book from last month to this month.")
        frames = frames_for(["2025-06-30", "2025-07-31", "2025-09-30", "2025-12-31"])
        with pytest.raises(PeriodChangeFailure):
            plan.resolve_period_pair(frames, qi, max_gap_days=30)

    def test_explicit_periods_beat_any_window(self, interpret):
        from mi_agent.period_change.models import METHOD_EXPLICIT_DATES
        from mi_agent_api import analytical_plan as plan
        _, qi = interpret("Compare the funded balance between June 2025 and September 2025")
        frames = frames_for(["2025-06-30", "2025-07-31", "2025-09-30", "2025-12-31"])
        start, end, res = plan.resolve_period_pair(frames, qi)
        assert (start["reporting_date"], end["reporting_date"]) == ("2025-06-30", "2025-09-30")
        assert res.resolution_method == METHOD_EXPLICIT_DATES

    def test_the_bridge_executor_runs_on_the_owner(self, interpret, monkeypatch):
        """End to end: the composed bridge for "last month" on an irregular tape
        opens at the previous observation, says so, and refuses under a
        ceiling — it never opens at the earliest period."""
        from mi_agent_api import analytical_plan as plan
        from mi_agent_api import evolution as evo
        frames = frames_for(["2025-06-30", "2025-07-31", "2025-09-30", "2025-12-31"])
        monkeypatch.setattr(evo, "funded_frames", lambda *a, **k: list(frames))
        _, qi = interpret("Bridge the movement in the book from last month to this month.")
        out = plan.funded_bridge("/nonexistent", "client_001", interpretation=qi,
                                 dimension_columns=["collateral_geography"],
                                 dimension_key="collateral_geography",
                                 dimension_label="Region")
        assert out["available"] is True
        assert out["start"]["period"] == "2025-09"
        assert out["end"]["period"] == "2025-12"
        assert out["periodResolution"]["start_adjusted_to_available_snapshot"] is True
        refused = plan.funded_bridge("/nonexistent", "client_001", interpretation=qi,
                                     dimension_columns=["collateral_geography"],
                                     dimension_key="collateral_geography",
                                     dimension_label="Region", max_gap_days=30)
        assert refused["available"] is False
        assert "2025-06" not in str(refused)

    def test_the_three_executors_all_ask_the_owner(self):
        from mi_agent_api import analytical_plan as plan
        for name in ("funded_bridge", "period_movement", "temporal_compare"):
            assert "resolve_period_pair" in _calls(plan, name), name

    def test_the_duplicate_owners_are_gone(self):
        from mi_agent_api import analytical_plan as plan
        from mi_agent_api import evolution as evo
        assert "window_periods" not in inspect.signature(evo.funded_bridge).parameters
        assert not hasattr(plan, "DEFAULT_SPAN_PERIODS")
        # The index arithmetic `scoped[len - 1 - window]` and the silent
        # `scoped[0]` fallback were the second and third period owners.
        src = _fn_source(evo, "funded_bridge")
        assert "scoped[0]" not in src
        assert "window_periods" not in src


# =========================================================================== #
# I3 — the claimant ASKS the owners. No exemption list.
# =========================================================================== #
class TestI3OwnerAwareClaiming:
    @pytest.mark.parametrize("token", ["prior", "previous", "live", "new", "exited", "churn"])
    def test_a_word_an_owner_reads_is_claimed(self, token, semantics, columns, book_values):
        from mi_agent import llm_query_parser as P
        assert P._claimed_by_an_owner(token, semantics, columns, book_values) is True

    @pytest.mark.parametrize("token", ["platinum", "atlantis"])
    def test_a_word_no_owner_reads_is_still_unclaimed(self, token, semantics, columns, book_values):
        from mi_agent import llm_query_parser as P
        assert P._claimed_by_an_owner(token, semantics, columns, book_values) is False

    def test_no_exemption_list(self):
        """The claim is made by ASKING owners, so none of these words may be a
        literal inside the claimant."""
        from mi_agent import llm_query_parser as P
        assert not ({"prior", "previous", "live", "new", "exited", "churn"}
                    & _string_constants(P, "_claimed_by_an_owner"))

    def test_end_to_end_no_false_unknown_category(self, parse):
        from mi_agent import llm_query_parser as P
        for q in ("How many live loans do we have?",
                  "What is the balance of new loans this month?",
                  "What was the funded balance in the prior period?"):
            notes = parse(q).spec.unavailable_filters or []
            assert P.unknown_category_refusal(notes) is None, (q, notes)
        notes = parse("How many platinum loans do we have?").spec.unavailable_filters or []
        assert P.unknown_category_refusal(notes), "a real unknown category still refuses"


# =========================================================================== #
# I4 — the FINAL interpretation includes the model merge; fill-only rule kept.
# =========================================================================== #
def _route_with_capture(question, *, parsed, semantics, frame, registry_extra=None):
    """Run `try_route` with a capture recogniser that runs FIRST and records
    what a route would see. Returns (captured, envelope)."""
    from mi_agent_api import chat_routing as CR
    from mi_agent_api.recogniser_registry import (Recogniser, RecogniserRegistry,
                                                  Recognition)
    captured = {}

    def _recognise(request):
        return Recognition.yes(1.0, "capture")

    def _handle(request):
        qi = request.resolve_interpretation()
        captured["metric"] = getattr(request.spec, "metric", None)
        captured["aggregation"] = getattr(request.spec, "aggregation", None)
        captured["subject"] = getattr(getattr(qi, "subject", None), "candidate_concept", None)
        captured["subject_provenance"] = getattr(getattr(qi, "subject", None), "provenance", None)
        captured["parse_meta"] = dict(request.parse_meta or {})
        return {"ok": True, "answer": "captured", "metadata": {"route": "capture"}}

    reg = RecogniserRegistry()
    reg.register(Recogniser(name="capture", priority=-10**6, recognise=_recognise,
                            handle=_handle, description="test capture"))
    env = CR.try_route(question, portfolio_id=None, view="funded",
                       output_root=None, pipeline_root=None, semantics=semantics,
                       parsed=parsed, registry=reg,
                       base_frame_resolver=lambda view, pid: frame)
    return captured, env


class TestI4FinalInterpretationIncludesTheMerge:
    QUESTION = "How big is the book by region?"

    @pytest.fixture
    def merge_on(self, monkeypatch):
        from mi_agent_api import concept_merge_arm as arm
        monkeypatch.setenv("MI_AGENT_CONCEPT_MERGE", "on")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        arm.set_replay({self.QUESTION: [
            {"kind": "measure", "term": "balance", "covers": "big"}]})
        yield
        arm.set_replay(None)

    def test_a_route_sees_the_merged_subject_with_its_provenance(
            self, merge_on, parse, semantics, frame):
        from question_interpretation.schema import PROV_MODEL_INFERRED
        parsed = parse(self.QUESTION)
        assert parsed.spec.metric is None, "the deterministic parse leaves the subject empty"
        captured, env = _route_with_capture(self.QUESTION, parsed=parsed,
                                            semantics=semantics, frame=frame)
        assert env and env.get("answer") == "captured"
        assert parsed.meta["conceptMerge"]["status"] == "applied"
        assert captured["metric"] == "current_outstanding_balance"
        assert captured["subject"] == "current_outstanding_balance"
        # The contract says WHERE the subject came from. A model fill that
        # re-projects as a deterministic reading has lost its provenance.
        assert captured["subject_provenance"] == PROV_MODEL_INFERRED
        # And the operation follows the measure: a balance filled onto a parse
        # that had fallen back to COUNT must not be counted.
        assert captured["aggregation"] == "sum"

    def test_an_asked_count_is_a_claim_the_model_cannot_overwrite(
            self, monkeypatch, parse, semantics, frame):
        """The fill-only rule, kept: "how many" is the reader's own subject.
        (Passes at baseline — the rule already holds; kept as the guard for
        the projection change I4 needs.)"""
        from mi_agent_api import concept_merge_arm as arm
        from question_interpretation.schema import PROV_MODEL_INFERRED
        q = "How many loans do we have by region?"
        monkeypatch.setenv("MI_AGENT_CONCEPT_MERGE", "on")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        arm.set_replay({q: [{"kind": "measure", "term": "balance", "covers": "loans"}]})
        try:
            parsed = parse(q)
            captured, _ = _route_with_capture(q, parsed=parsed, semantics=semantics,
                                              frame=frame)
        finally:
            arm.set_replay(None)
        assert captured["aggregation"] == "count"
        assert captured["subject"] == "loan_count"
        assert captured["subject_provenance"] != PROV_MODEL_INFERRED

    def test_the_merge_never_overwrites_a_deterministic_subject(
            self, monkeypatch, parse, semantics, frame):
        from mi_agent_api import concept_merge_arm as arm
        q = "What is the total funded balance by region?"
        monkeypatch.setenv("MI_AGENT_CONCEPT_MERGE", "on")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        arm.set_replay({q: [{"kind": "measure", "term": "ltv", "covers": "balance"}]})
        try:
            parsed = parse(q)
            captured, _ = _route_with_capture(q, parsed=parsed, semantics=semantics,
                                              frame=frame)
        finally:
            arm.set_replay(None)
        assert captured["metric"] == "current_outstanding_balance"
        assert captured["subject"] == "current_outstanding_balance"


# =========================================================================== #
# I5 — ONE pre-execution semantic claims ledger; guards compare CLAIMED with
#      EXECUTED; no second raw reader may veto.
# =========================================================================== #
class TestI5SemanticClaimsLedger:
    LEVEL_Q = "On a balance-weighted basis, what LTV is the portfolio running at?"

    def test_the_ledger_is_built_once_and_carried_on_the_parse(self, parse, semantics, frame):
        from mi_agent import semantic_claims as SC
        parsed = parse(self.LEVEL_Q)
        captured, _ = _route_with_capture(self.LEVEL_Q, parsed=parsed,
                                          semantics=semantics, frame=frame)
        ledger = captured["parse_meta"].get("semanticClaims")
        assert isinstance(ledger, dict)
        assert ledger == parsed.meta["semanticClaims"]
        assert ledger["temporal_aspect"] == "level"
        assert "period_comparison" not in (ledger.get("requirements") or [])
        assert SC.from_dict(ledger).temporal_aspect == "level"

    def test_a_level_question_is_not_refused_as_a_trend(self, interpret, semantics):
        """D3: `running at` reads as MOVEMENT_TREND to the raw intent reader,
        while the temporal owner says LEVEL. The guard may only assert what the
        ledger CLAIMS, so the level answer stands."""
        from mi_agent import semantic_claims as SC
        from mi_agent_api import mi_service as S
        parsed, qi = interpret(self.LEVEL_Q)
        claims = SC.build(self.LEVEL_Q, interpretation=qi, spec=parsed.spec,
                          parse_meta=parsed.meta)
        result = {"ok": True, "answer": "58.1%", "spec": parsed.spec.to_dict(),
                  "metadata": {}, "artifacts": []}
        out = S._fail_closed_analytical(result, question=self.LEVEL_Q, view="funded",
                                        claims=claims)
        assert out.get("ok") is True and not out.get("controlledRefusal")

    def test_a_claimed_but_unexecuted_requirement_still_refuses(self, interpret):
        from mi_agent import semantic_claims as SC
        from mi_agent_api import mi_service as S
        q = "How has the weighted average LTV changed month on month?"
        parsed, qi = interpret(q)
        claims = SC.build(q, interpretation=qi, spec=parsed.spec, parse_meta=parsed.meta)
        assert "period_comparison" in claims.requirements
        result = {"ok": True, "answer": "58.1%", "spec": parsed.spec.to_dict(),
                  "metadata": {}, "artifacts": []}
        out = S._fail_closed_analytical(result, question=q, view="funded", claims=claims)
        assert out.get("ok") is False and out.get("controlledRefusal") is True

    def test_the_guard_reads_the_ledger_not_the_sentence(self):
        from mi_agent_api import mi_service as S
        assert "classify" not in _calls(S, "_fail_closed_analytical")


# =========================================================================== #
# I6 — capability ownership beats a generic measure collision.
# =========================================================================== #
class TestI6CapabilityArbitration:
    BORROW_Q = "How much collateral value are we able to borrow against?"
    DRAW_Q = "How much can we draw against the facility?"
    VALUATION_Q = "What is the total collateral valuation?"

    def test_the_borrowing_base_owner_claims_a_borrow_verb_over_collateral(self):
        from mi_agent import capability_ownership as CO
        claims = CO.claims(self.BORROW_Q)
        assert any(c.capability == "borrowing_base" and c.concept == "borrowing_capacity"
                   for c in claims)
        assert CO.claims(self.VALUATION_Q) == ()

    def test_the_parser_does_not_bind_a_measure_inside_a_claimed_span(self, parse):
        parsed = parse(self.BORROW_Q)
        assert parsed.spec.metric != "current_valuation_amount"
        assert parsed.meta.get("capability_claims")
        assert parsed.meta["capability_claims"][0]["capability"] == "borrowing_base"

    def test_the_route_honours_the_claim(self):
        from mi_agent_api import borrowing_base_query as BB
        assert BB.read(self.BORROW_Q).matched is True
        assert BB.read(self.DRAW_Q).matched is True

    def test_valuation_stays_valuation(self, parse):
        from mi_agent_api import borrowing_base_query as BB
        parsed = parse(self.VALUATION_Q)
        assert parsed.spec.metric == "current_valuation_amount"
        assert not parsed.meta.get("capability_claims")
        assert BB.read(self.VALUATION_Q).matched is False

    def test_two_owners_on_one_span_is_unresolvable(self):
        from mi_agent import capability_ownership as CO
        a = CO.CapabilityClaim("borrowing_base", "borrowing_capacity", 0, 10, "x")
        b = CO.CapabilityClaim("risk_limits", "headroom", 0, 10, "y")
        verdict = CO.arbitrate((a, b))
        assert verdict.winner is None and verdict.conflicts
        assert CO.arbitrate((a,)).winner is a


# =========================================================================== #
# I7 — ONE geography field owner.
# =========================================================================== #
class TestI7OneGeographyFieldOwner:
    def test_the_owner_answers_the_code_field_too(self, two_basis_frame):
        from mi_agent import mi_geography as G
        assert G.code_field_for_basis("borrower", frame=two_basis_frame) == \
            "geographic_region_obligor_itl3"
        assert G.code_field_for_basis("collateral", frame=two_basis_frame) == \
            "geographic_region_collateral_itl3"

    def test_every_chooser_returns_the_contracts_field(self, two_basis_frame, semantics):
        from mi_agent_api import chat_routing as CR
        from mi_agent_api import evolution as evo
        from mi_agent_api import movement_summary as MS
        contract = contract_for("borrower", two_basis_frame)
        assert MS._region_column(two_basis_frame, geography=contract) == "geographic_region_obligor"
        assert evo._region_breakdown_column(two_basis_frame, geography=contract) == \
            "geographic_region_obligor"
        _key, cols, _label = CR._bridge_dimension("collateral_geography", semantics,
                                                  geography=contract)
        cols = [cols] if isinstance(cols, str) else list(cols)
        assert cols == ["geographic_region_obligor"], \
            "the bridge may not cross bases by offering the whole family"

    def test_itl3_exposure_is_measured_on_the_contracts_basis(self, two_basis_frame):
        from mi_agent_api import geo
        contract = contract_for("borrower", two_basis_frame)
        out = geo.exposure_by_itl3(two_basis_frame, geography=contract)
        assert out["available"] is True
        assert out["basis"] == "obligor"
        assert out["resolvedFromItl3Field"] == 4

    def test_the_duplicate_choosers_no_longer_choose(self):
        from mi_agent_api import chat_routing as CR
        from mi_agent_api import evolution as evo
        from mi_agent_api import geo
        from mi_agent_api import movement_summary as MS
        assert "REGION_COLUMNS" not in _names_referenced(MS, "_region_column")
        assert "_ITL3_FIELDS" not in _names_referenced(geo, "exposure_by_itl3")
        assert "_REGION_FAMILY" not in _names_referenced(CR, "_bridge_dimension")
        assert "_REGION_PREFERENCE" not in _names_referenced(evo, "_region_breakdown_column")

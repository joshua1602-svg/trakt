"""The model proposes; a person decides.

``engine/gate_1_alignment/llm_mapper_agent.py`` has existed all along, with its
redaction, its canonical-only enforcement and its human review session.
``config/system/onboarding_agent.yaml`` has carried a full budget policy for it.
Nothing in the OCC Agent ever called it. A real client's hundred-column tape
arrived, thirty columns matched nothing, and those thirty were reported as
unreadable with no proposal against any of them — leaving an operator to name
each canonical field from memory. "The LLM must be enabled. That is the whole
point of the model."

THE CONTRACT IS THE ENGINE'S OWN: "The LLM only reviews unresolved ambiguity and
never writes final mappings." Enabling a model is only safe if that holds, so it
is what this file is mostly about:

  * a column the deterministic tiers SETTLED is never sent. Zero-cost first is
    not a preference, it is the reason a model costs nothing on a clean pack;
  * what comes back never enters the resolved mappings and never reaches the
    tape. It becomes a question;
  * the question says a model proposed it. A suggestion dressed as a match
    would be the model writing mappings by another route, with the human
    confirmation step preserved in form and defeated in substance;
  * the budget is the configuration's. Calls, items, sample values, the model
    itself and the skip threshold all come from the policy file, and no default
    here is more permissive than that file;
  * where it does not run, it says why. A model that quietly did not run and a
    model that had nothing to say look identical from the outside, and they are
    not the same thing to anyone deciding whether to trust the mapping.

Every test drives the real wiring with a stub in place of the network call, so
what is proven is the path a client's tape takes, not a restatement of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
import yaml

from operations_control.occ_agent import llm_mapping as llm

POLICY = llm.Policy(enabled=True, model="claude-haiku-4-5-20251001",
                    max_calls=10, max_items_per_call=10, settled_above=0.85,
                    max_samples=3, unresolved_ceiling=25)


class _Suggested:
    """What ``LLMFieldMapper.suggest_mappings`` returns, in shape."""

    def __init__(self, header, field_name="", confidence=0.0, reasoning="",
                 alternative=None):
        self.raw_header = header
        self.suggested_field = field_name or None
        self.confidence = confidence
        self.reasoning = reasoning
        self.alternative_field = alternative


def stub_mapper(monkeypatch, answers: Dict[str, Any], seen: List[Any]):
    """Stand in for the network call, recording exactly what was asked."""
    import engine.gate_1_alignment.llm_mapper_agent as agent

    class _Stub:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def suggest_mappings(self, headers, frame, deterministic_report=None):
            seen.append({"headers": list(headers)})
            return [answers.get(h, _Suggested(h)) for h in headers]

    monkeypatch.setattr(agent, "LLMFieldMapper", _Stub)
    return seen


def report(*rows):
    """A mapping report in the Agent's own shape."""
    out = []
    for column, tier, confidence in rows:
        out.append({"source_file": "tape.xlsx", "source_column": column,
                    "canonical_field": "", "tier": tier,
                    "confidence": confidence, "primary": True,
                    "source_sheet": ""})
    return out


# --------------------------------------------------------------------------- #
# Only what deterministic matching could not settle
# --------------------------------------------------------------------------- #

class TestZeroCostFirst:

    def test_a_contract_backed_match_is_never_sent(self):
        """An alias match is the platform's own recorded answer. Asking a model
        about it spends budget to be told what is already known."""
        rows = report(("Month Run", "alias", 1.0),
                      ("Loan ID", "exact", 1.0),
                      ("Cur Bal", "normalized", 1.0))
        assert llm.unresolved(rows, POLICY) == []

    def test_a_confident_fuzzy_match_is_never_sent(self):
        """0.85 is the configuration's threshold, not this module's."""
        rows = report(("Prp Val", "fuzz_token_set", 0.92))
        assert llm.unresolved(rows, POLICY) == []

    def test_an_unmatched_column_is_what_gets_sent(self):
        rows = report(("Xyz Ref 7", "unmapped", 0.0))
        assert llm.unresolved(rows, POLICY) == ["Xyz Ref 7"]

    def test_a_weak_match_is_sent_too(self):
        rows = report(("Prp Ref", "fuzz_ratio_norm", 0.41))
        assert llm.unresolved(rows, POLICY) == ["Prp Ref"]

    def test_a_column_an_operator_already_answered_is_not_sent(self):
        rows = report(("Prp Ref", "operator_approved", 1.0))
        assert llm.unresolved(rows, POLICY) == []

    def test_a_column_from_a_file_the_tape_is_not_built_from_is_not_sent(self):
        """An operator's answer there resolves nothing, so a question about it
        is a question with no answer."""
        rows = report(("Xyz Ref 7", "unmapped", 0.0))
        rows[0]["primary"] = False
        assert llm.unresolved(rows, POLICY) == []


# --------------------------------------------------------------------------- #
# A suggestion is never a mapping
# --------------------------------------------------------------------------- #

class TestItProposesAndDoesNotDecide:

    def test_the_question_says_a_model_proposed_it(self):
        """A suggestion dressed as a deterministic match would preserve the
        human confirmation step in form and defeat it in substance."""
        decision = llm.decision(
            llm.Suggestion("Prp Valn Amt", "current_valuation_amount", 0.91,
                           "A property valuation amount in pounds."),
            source_file="tape.xlsx", populated=560, rows=568)

        assert "a model suggests" in decision["issue"].lower()
        assert decision["basis"] == llm.BASIS
        assert decision["status"] == "pending"
        assert decision["blocking"] is True

    def test_the_question_names_the_field_in_words(self):
        decision = llm.decision(
            llm.Suggestion("Prp Valn Amt", "current_valuation_amount", 0.91),
            source_file="tape.xlsx", populated=560, rows=568)
        assert "current valuation amount" in decision["issue"]

    def test_the_model_s_reason_travels_with_it(self):
        decision = llm.decision(
            llm.Suggestion("Prp Valn Amt", "current_valuation_amount", 0.91,
                           "A property valuation amount in pounds."),
            source_file="tape.xlsx", populated=560, rows=568)
        assert "A property valuation amount in pounds." \
            in decision["evidence_summary"]

    def test_rejecting_it_is_one_of_the_offered_answers(self):
        decision = llm.decision(
            llm.Suggestion("Prp Valn Amt", "current_valuation_amount", 0.91),
            source_file="tape.xlsx", populated=560, rows=568)
        assert "choose_alternative" in decision["available_actions"]
        assert "mark_unavailable" in decision["available_actions"]


# --------------------------------------------------------------------------- #
# The budget is the configuration's
# --------------------------------------------------------------------------- #

class TestTheBudgetIsGoverned:

    def test_it_is_read_from_the_policy_file(self):
        p = llm.Policy.load()
        assert p.model == "claude-haiku-4-5-20251001"
        assert p.max_calls == 10 and p.max_items_per_call == 10
        assert p.settled_above == 0.85
        assert p.max_samples == 3
        assert p.allow_frontier_escalation is False

    def test_the_model_is_on(self):
        """The whole point of the model. Every guard that makes it safe is
        asserted in this file; this asserts it is actually running."""
        assert llm.Policy.load().enabled is True

    def test_an_unreadable_policy_leaves_it_off(self, tmp_path):
        """A missing budget is not an unlimited one."""
        bad = tmp_path / "nope.yaml"
        assert llm.Policy.load(bad).enabled is False

    def test_no_more_columns_are_sent_than_the_budget_allows(self, monkeypatch):
        seen: List[Any] = []
        stub_mapper(monkeypatch, {}, seen)
        tight = llm.Policy(enabled=True, model="m", max_calls=2,
                           max_items_per_call=3, settled_above=0.85,
                           max_samples=3, unresolved_ceiling=0)
        columns = [f"C{i}" for i in range(20)]
        outcome = llm.suggest(columns, pd.DataFrame({c: [1] for c in columns}),
                              policy=tight, registry_path=Path("r.yaml"),
                              aliases_dir=Path("a"), asset_type="equity_release",
                              api_key="k")
        assert outcome.asked == 6

    def test_the_configured_model_is_the_one_asked(self, monkeypatch):
        seen: List[Any] = []
        stub_mapper(monkeypatch, {}, seen)
        llm.suggest(["C1"], pd.DataFrame({"C1": [1]}), policy=POLICY,
                    registry_path=Path("r.yaml"), aliases_dir=Path("a"),
                    asset_type="equity_release", api_key="k")
        assert seen[0]["model"] == "claude-haiku-4-5-20251001"

    def test_the_sample_limit_is_passed_through(self, monkeypatch):
        """Redacted samples only, and no more of them than the policy allows."""
        seen: List[Any] = []
        stub_mapper(monkeypatch, {}, seen)
        llm.suggest(["C1"], pd.DataFrame({"C1": [1]}), policy=POLICY,
                    registry_path=Path("r.yaml"), aliases_dir=Path("a"),
                    asset_type="equity_release", api_key="k")
        assert seen[0]["max_sample_values"] == 3

    def test_the_engine_honours_that_limit(self):
        """Asserted against the engine's own envelope builder, not against a
        restatement of it: a limit the caller states and the mapper ignores is
        client data leaving in a quantity nobody approved."""
        from engine.gate_1_alignment.llm_mapper_agent import LLMFieldMapper
        mapper = LLMFieldMapper.__new__(LLMFieldMapper)
        mapper.max_sample_values = 3
        frame = pd.DataFrame({"C1": [f"v{i}" for i in range(10)]})
        envelope = LLMFieldMapper._build_envelope(mapper, "C1", frame)
        assert len(envelope["samples"]) == 3

    def test_it_never_escalates_to_a_frontier_model(self, monkeypatch):
        seen: List[Any] = []
        stub_mapper(monkeypatch, {}, seen)
        escalating = llm.Policy(enabled=True, model="m", max_calls=10,
                               max_items_per_call=10, settled_above=0.85,
                               max_samples=3, unresolved_ceiling=25,
                               allow_frontier_escalation=True)
        outcome = llm.suggest(["C1"], pd.DataFrame({"C1": [1]}),
                              policy=escalating, registry_path=Path("r.yaml"),
                              aliases_dir=Path("a"), asset_type="equity_release",
                              api_key="k")
        assert outcome.asked == 0
        assert "escalate" in outcome.skipped_because

    def test_past_the_uncertainty_ceiling_the_question_goes_to_a_person(
            self, monkeypatch):
        """The configuration's own instruction: past this many unsettled
        columns, stop asking a model and ask a person."""
        seen: List[Any] = []
        stub_mapper(monkeypatch, {}, seen)
        outcome = llm.suggest([f"C{i}" for i in range(26)], pd.DataFrame(),
                              policy=POLICY, registry_path=Path("r.yaml"),
                              aliases_dir=Path("a"), asset_type="equity_release",
                              api_key="k")
        assert outcome.asked == 0
        assert "put to you" in outcome.skipped_because


# --------------------------------------------------------------------------- #
# Where it does not run, it says so
# --------------------------------------------------------------------------- #

class TestSilenceIsNeverAmbiguous:

    def test_switched_off_says_switched_off(self):
        outcome = llm.suggest(["C1"], pd.DataFrame(),
                              policy=llm.Policy(enabled=False),
                              registry_path=Path("r.yaml"), aliases_dir=Path("a"),
                              asset_type="equity_release", api_key="k")
        assert outcome.skipped_because == ("the model is switched off for this "
                                           "environment")

    def test_no_key_says_no_access(self, monkeypatch):
        monkeypatch.delenv(llm.API_KEY_ENV, raising=False)
        outcome = llm.suggest(["C1"], pd.DataFrame(), policy=POLICY,
                              registry_path=Path("r.yaml"), aliases_dir=Path("a"),
                              asset_type="equity_release")
        assert outcome.skipped_because == "no model access is configured here"

    def test_a_failure_is_reported_and_never_raised(self, monkeypatch):
        """A model that cannot be reached must not take a client's onboarding
        down with it."""
        import engine.gate_1_alignment.llm_mapper_agent as agent

        def _boom(**_kwargs):
            raise RuntimeError("no route to host")

        monkeypatch.setattr(agent, "LLMFieldMapper", _boom)
        outcome = llm.suggest(["C1"], pd.DataFrame(), policy=POLICY,
                              registry_path=Path("r.yaml"), aliases_dir=Path("a"),
                              asset_type="equity_release", api_key="k")
        assert outcome.suggestions == []
        assert "could not be reached" in outcome.skipped_because

    def test_a_column_the_model_could_not_place_is_not_a_proposal(
            self, monkeypatch):
        """A null answer is an answer. It must not become a question reading
        "a model suggests it is nothing"."""
        seen: List[Any] = []
        stub_mapper(monkeypatch, {"C1": _Suggested("C1")}, seen)
        outcome = llm.suggest(["C1"], pd.DataFrame({"C1": [1]}), policy=POLICY,
                              registry_path=Path("r.yaml"), aliases_dir=Path("a"),
                              asset_type="equity_release", api_key="k")
        assert outcome.by_column == {}
        assert outcome.asked == 1


# --------------------------------------------------------------------------- #
# The wiring itself, against the real onboard stage
# --------------------------------------------------------------------------- #

from .conftest import ACTOR, TENANT_A                        # noqa: E402

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")

#: One column the mapper matches on its own, and one it cannot place at all.
PACK = "loan_id,Zxq Internal Ref 7\nL1,abc\nL2,def\n"


def _spec():
    from engine.orchestrator_agent.adapters import PortfolioSpec
    return PortfolioSpec(source_portfolio_id="direct_101", input="")


class TestTheWiring:
    """Driven through the real ``onboard`` stage, with a stub in place of the
    network call — so what is proven is the path a client's tape takes."""

    def _adapters(self, service, monkeypatch, answers):
        from operations_control.occ_agent.execution import (
            SyntheticOnboardingAdapters,
        )
        seen: List[Any] = []
        stub_mapper(monkeypatch, answers, seen)
        monkeypatch.setenv(llm.API_KEY_ENV, "test-key")
        agent_case = service.create_case(tenant=TENANT_A,
                                         initiating_user=ACTOR,
                                         instruction=OPENING)
        sandbox = service.store.case_dir(TENANT_A, agent_case.case_ref)
        path = sandbox / "loan_extract.csv"
        path.write_text(PACK, encoding="utf-8")
        adapters = SyntheticOnboardingAdapters(
            artefact_paths=[path], policy=service.policy, sandbox=sandbox,
            llm_policy=POLICY, case_id=agent_case.case_ref, tenant=TENANT_A)
        return adapters, sandbox, seen

    def test_only_the_unmatched_column_reaches_the_model(
            self, service, monkeypatch):
        """``loan_id`` matches exactly. Asking about it would spend a client's
        budget to be told what the registry already says."""
        adapters, sandbox, seen = self._adapters(service, monkeypatch, {})
        adapters.onboard(_spec(), sandbox / "work")
        asked = next(s["headers"] for s in seen if "headers" in s)
        assert asked == ["Zxq Internal Ref 7"]

    def test_a_suggestion_becomes_a_question_not_a_mapping(
            self, service, monkeypatch):
        adapters, sandbox, seen = self._adapters(service, monkeypatch, {
            "Zxq Internal Ref 7": _Suggested(
                "Zxq Internal Ref 7", "loan_identifier", 0.88,
                "An internal loan reference.")})
        result = adapters.onboard(_spec(), sandbox / "work")

        assert result.blocking is True, "the run did not stop to ask"
        decisions = yaml.safe_load(
            (sandbox / "work" / "34_target_first_decisions.yaml")
            .read_text(encoding="utf-8"))["decisions"]
        mine = [d for d in decisions if d.get("basis") == llm.BASIS]
        assert len(mine) == 1
        assert mine[0]["source_column"] == "Zxq Internal Ref 7"
        assert mine[0]["target_field"] == "loan_identifier"

    def test_the_suggestion_never_reaches_the_tape(self, service, monkeypatch):
        """The engine's own contract: the model never writes final mappings.
        The run halts on the question, so no tape is built — and if one ever
        is built past this point, the column must not be in it."""
        adapters, sandbox, seen = self._adapters(service, monkeypatch, {
            "Zxq Internal Ref 7": _Suggested(
                "Zxq Internal Ref 7", "loan_identifier", 0.99)})
        adapters.onboard(_spec(), sandbox / "work")
        tape = sandbox / "work" / "18_central_lender_tape.csv"
        assert not tape.exists(), "a tape was built from an unconfirmed guess"

    def test_it_is_recorded_on_the_run(self, service, monkeypatch):
        adapters, sandbox, seen = self._adapters(service, monkeypatch, {
            "Zxq Internal Ref 7": _Suggested(
                "Zxq Internal Ref 7", "loan_identifier", 0.88)})
        adapters.onboard(_spec(), sandbox / "work")
        assert adapters.llm["asked"] == 1
        assert adapters.llm["model"] == POLICY.model
        assert adapters.llm["suggestions"][0]["basis"] == llm.BASIS

    def test_the_mapping_row_says_the_model_proposed_it(
            self, service, monkeypatch):
        """So the table an operator checks can show the basis for every row,
        deterministic or proposed."""
        adapters, sandbox, seen = self._adapters(service, monkeypatch, {
            "Zxq Internal Ref 7": _Suggested(
                "Zxq Internal Ref 7", "loan_identifier", 0.88)})
        adapters.onboard(_spec(), sandbox / "work")
        row = next(r for r in adapters.mapping_report
                   if r["source_column"] == "Zxq Internal Ref 7")
        assert row["llm_field"] == "loan_identifier"
        assert row["canonical_field"] == "", \
            "a suggestion was written into the deterministic mapping"

    def test_a_run_with_no_model_still_completes(self, service, monkeypatch):
        """Switched off, unreachable or unfunded, the onboarding goes on."""
        from operations_control.occ_agent.execution import (
            SyntheticOnboardingAdapters,
        )
        monkeypatch.delenv(llm.API_KEY_ENV, raising=False)
        agent_case = service.create_case(tenant=TENANT_A,
                                         initiating_user=ACTOR,
                                         instruction=OPENING)
        sandbox = service.store.case_dir(TENANT_A, agent_case.case_ref)
        path = sandbox / "loan_extract.csv"
        path.write_text("loan_id,current_principal_balance\nL1,100\n",
                        encoding="utf-8")
        adapters = SyntheticOnboardingAdapters(
            artefact_paths=[path], policy=service.policy, sandbox=sandbox,
            llm_policy=POLICY, case_id=agent_case.case_ref, tenant=TENANT_A)
        result = adapters.onboard(_spec(), sandbox / "work")
        assert result.ok is True


# --------------------------------------------------------------------------- #
# The table says where a suggestion came from
# --------------------------------------------------------------------------- #

class _Run:
    def __init__(self, mapping_report, open_decisions=()):
        self.mapping_report = list(mapping_report)
        self.open_decisions = list(open_decisions)


class TestTheTableShowsTheBasis:
    """"Suggested mapping" with no basis beside it invites an operator to
    accept a model's proposal on the same footing as a contract-backed match."""

    def _row(self, **over):
        base = {"source_file": "tape.xlsx", "source_column": "Zxq Ref",
                "canonical_field": "", "tier": "unmapped", "confidence": 0.0,
                "primary": True}
        base.update(over)
        return base

    def test_a_proposal_is_marked_as_a_model_s(self):
        from operations_control.occ_agent import mapping_view
        view = mapping_view.overview(_Run([self._row(
            llm_field="loan_identifier", llm_confidence=0.88,
            llm_reasoning="An internal loan reference.")]))
        row = view["rows"][0]
        assert row["basis"] == mapping_view.BASIS_MODEL
        assert row["suggested_field"] == "loan_identifier"
        assert row["suggested_label"] == "loan identifier"
        assert "not yet confirmed" in row["basis_label"]

    def test_a_deterministic_match_is_marked_as_trakt_s_own(self):
        from operations_control.occ_agent import mapping_view
        view = mapping_view.overview(_Run([self._row(
            canonical_field="loan_identifier", tier="alias", confidence=1.0)]))
        assert view["rows"][0]["basis"] == mapping_view.BASIS_DETERMINISTIC

    def test_what_an_operator_confirmed_is_marked_as_theirs(self):
        from operations_control.occ_agent import mapping_view
        view = mapping_view.overview(_Run([self._row(
            canonical_field="loan_identifier", tier="operator_approved",
            confidence=1.0)]))
        assert view["rows"][0]["basis"] == mapping_view.BASIS_OPERATOR

    def test_a_proposal_does_not_count_as_a_mapped_column(self):
        """Counting it would make a blocked run read as more complete than a
        finished one."""
        from operations_control.occ_agent import mapping_view
        view = mapping_view.overview(_Run([self._row(
            llm_field="loan_identifier", llm_confidence=0.99)]))
        assert view["counts"]["mapped"] == 0

    def test_the_canonical_field_stays_empty_until_someone_confirms(self):
        from operations_control.occ_agent import mapping_view
        view = mapping_view.overview(_Run([self._row(
            llm_field="loan_identifier", llm_confidence=0.99)]))
        assert view["rows"][0]["canonical_field"] == ""


# --------------------------------------------------------------------------- #
# The operator contract says where the proposal came from
# --------------------------------------------------------------------------- #

class TestTheOperatorContractSaysWhoProposed:
    """``extract_mapping_decisions`` labelled every ``recommended_action`` it
    found "deterministic". A model's suggestion travelling under that label
    would tell an operator a guess was a contract-backed match — the one thing
    the human confirmation step exists to prevent."""

    def _extracted(self, tmp_path, decision):
        from operations_control.adapters import extract_mapping_decisions
        from operations_control.contracts import WorkflowRun
        work = tmp_path / "work"
        work.mkdir()
        (work / "34_target_first_decisions.yaml").write_text(
            yaml.safe_dump({"decisions": [decision]}, sort_keys=False),
            encoding="utf-8")
        run = WorkflowRun(workflow_id="wf1", client_id="c", portfolio_id="p",
                          outcome="mi", workflow_type="new_client",
                          delivery={"files": [], "input_path": str(work)})
        return extract_mapping_decisions(work, run)

    def test_a_model_s_proposal_is_labelled_as_a_model_s(self, tmp_path):
        out = self._extracted(tmp_path, llm.decision(
            llm.Suggestion("Zxq Ref", "loan_identifier", 0.88),
            source_file="tape.xlsx", populated=560, rows=568))
        assert out[0].recommendation["source"] == "llm"

    def test_the_model_s_confidence_travels_with_it(self, tmp_path):
        out = self._extracted(tmp_path, llm.decision(
            llm.Suggestion("Zxq Ref", "loan_identifier", 0.88),
            source_file="tape.xlsx", populated=560, rows=568))
        assert out[0].recommendation["confidence"] == 0.88

    def test_a_deterministic_proposal_is_still_labelled_deterministic(
            self, tmp_path):
        """Nothing here relabels what the mapper itself proposed."""
        out = self._extracted(tmp_path, {
            "decision_id": "map_prp_val", "decision_type": "mapping_confirmation",
            "target_field": "current_valuation_amount",
            "source_column": "Prp Val", "status": "pending", "blocking": True,
            "recommended_action": "accept_mapping", "confidence": 0.71,
            "issue": "a weak match", "evidence_summary": "tier fuzz_token_set"})
        assert out[0].recommendation["source"] == "deterministic"

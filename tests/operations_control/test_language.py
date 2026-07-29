"""Failure translation into plain-English outcomes and the operator-safety
contract: no paths, tracebacks, schema codes, artefact numbers or container
names ever reach the operator-facing fields."""

from __future__ import annotations

from operations_control import language
from operations_control.adapters import translate_run_state
from operations_control.contracts import WorkflowRun
from engine.orchestrator_agent.state import PortfolioState, RunState, StepState


RAW_BLOCKERS = [
    "onboarding not ready_for_transformation_validation (unresolved mapping "
    "gaps / blocking decisions) — pending review",
    "full pipeline requested but onboarding produced no handoff manifest "
    "(Gate 2 has no contract to transform)",
    "validation has blocking exceptions (not ready_for_validation_complete)",
    "regime projection failed (rc=2)",
    "assembler: KeyError 'loan_identifier'",
    "provenance: acquisition_date missing for acquired portfolio pf9",
    "onboarding did not produce a central lender tape",
    "TypeError: cannot unpack non-sequence at /home/user/trakt/engine/foo.py:42",
]


class TestHumanise:
    def test_every_known_blocker_translates_to_safe_text(self):
        for raw in RAW_BLOCKERS:
            out = language.humanise_blocker(raw)
            assert language.is_operator_safe(out), f"unsafe: {out!r}"
            assert len(out) < 220

    def test_unknown_junk_falls_back_to_generic(self):
        out = language.humanise_blocker(
            "XYZfailure blob://trakt-state/runs/x/gates/g1/diagnostics.json")
        assert out == language.GENERIC_PROBLEM
        assert language.is_operator_safe(out)

    def test_dedupe(self):
        out = language.humanise_blockers([RAW_BLOCKERS[0], RAW_BLOCKERS[0]])
        assert len(out) == 1


class TestSafetyPredicate:
    def test_rejects_paths_uris_and_artefact_codes(self):
        for bad in ("/home/user/trakt/out", "blob://processed-v2/x",
                    "see 24_onboarding_handoff_manifest.json",
                    "Traceback (most recent call last)",
                    "sha256:abc", "check the run_manifest"):
            assert not language.is_operator_safe(bad)

    def test_accepts_plain_language(self):
        for ok in ("Three mappings need your confirmation.",
                   "The report is ready and waiting for your approval.",
                   language.GENERIC_PROBLEM):
            assert language.is_operator_safe(ok)


class TestTranslatedResults:
    def _run(self) -> WorkflowRun:
        return WorkflowRun(workflow_id="wf_t", client_id="c", portfolio_id="p",
                           outcome="mi", workflow_type="new_client",
                           delivery={"files": [{"name": "loans.csv"}],
                                     "input_path": "/x"})

    def _state(self, onboard_status: str, blockers=None) -> RunState:
        state = RunState(run_id="orun_t", client_id="c", target="mi",
                         out_root="/tmp/x", created_at="2026-01-01T00:00:00Z")
        p = PortfolioState(source_portfolio_id="p", source_portfolio_type="direct")
        p.steps["onboard"] = StepState("onboard", status=onboard_status,
                                       blockers=list(blockers or []))
        state.portfolios.append(p)
        return state

    def test_failed_step_produces_safe_operator_fields(self, tmp_path):
        state = self._state("failed",
                            ["TypeError: boom at /home/user/x.py:1"])
        results = translate_run_state(self._run(), state, tmp_path)
        for gar in results.values():
            assert language.is_operator_safe(gar.summary), gar.summary
            for b in gar.blockers:
                assert language.is_operator_safe(b), b
            for w in gar.warnings:
                assert language.is_operator_safe(w), w

    def test_halted_without_decisions_is_safe(self, tmp_path):
        state = self._state("halted", [RAW_BLOCKERS[6]])
        results = translate_run_state(self._run(), state, tmp_path)
        under = results["understanding"]
        assert under.status == "blocked"
        for b in under.blockers:
            assert language.is_operator_safe(b)

    def test_needs_configuration_presents_as_setup_task(self, tmp_path):
        """A pre-existing regime-config gap (onboarding status
        NEEDS_CONFIGURATION) is presented as an administrator set-up task, not
        a dead end or a file problem."""
        import json as _json
        (tmp_path / "40_operator_workflow_summary.json").write_text(
            _json.dumps({"status": "NEEDS_CONFIGURATION"}), encoding="utf-8")
        state = self._state("halted", ["onboarding not "
                                       "ready_for_transformation_validation"])
        results = translate_run_state(self._run(), state, tmp_path)
        assert results["understanding"].status == "completed"
        mapping = results["mapping"]
        assert mapping.status == "blocked"
        assert "administrator" in mapping.blockers[0]
        for text in (mapping.summary, mapping.why_it_matters,
                     *mapping.blockers):
            assert language.is_operator_safe(text), text

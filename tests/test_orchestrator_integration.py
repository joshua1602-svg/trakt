#!/usr/bin/env python3
"""tests/test_orchestrator_integration.py

REAL-agent integration smoke for the Agentic Orchestration: runs the actual
Onboarding / Transformation / Validation agents through the orchestrator on the
committed ``synthetic_demo/input`` pack (no stubs).

  * MI path (mi_only): green end-to-end — onboarding → central tape → stamp →
    assemble → MI route, with provenance on the central canonical.
  * Regime path (regulatory_mi): runs the real regulatory chain and HALTS at the
    transformation gate (the synthetic pack has blocking regulatory issues incl.
    mandatory ESMA Annex 2 fields) — i.e. governed auto-halt on real agent output.

These are slower (real agents); they skip if the pack/deps are unavailable.

Run: python -m unittest tests.test_orchestrator_integration
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_PACK = _REPO / "synthetic_demo" / "input"
_REGISTRY = str(_REPO / "config" / "system" / "fields_registry.yaml")
_NOW = "2026-09-01T00:00:00+00:00"


@unittest.skipUnless(_PACK.is_dir(), "synthetic_demo/input pack not present")
class TestOrchestratorRealAgents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        try:
            from engine.orchestrator_agent import run_orchestration  # noqa: F401
            from engine.orchestrator_agent.adapters import RealAgentAdapters  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"orchestrator/agent deps unavailable: {exc}")

    def _run(self, *, target, mode, regime=None, full_pipeline=False,
             processing_mode="source_onboarding", reporting_period=None):
        from engine.orchestrator_agent import run_orchestration
        from engine.orchestrator_agent.adapters import RealAgentAdapters, PortfolioSpec
        ad = RealAgentAdapters(registry=_REGISTRY, client_name="ERE_IT",
                               onboarding_mode=mode, processing_mode=processing_mode,
                               full_pipeline=full_pipeline,
                               reporting_period=reporting_period)
        # Keep the output dir alive for the test body (assertions read artifacts).
        td = tempfile.mkdtemp(prefix="orch_it_")
        self.addCleanup(lambda: __import__("shutil").rmtree(td, ignore_errors=True))
        return run_orchestration(
            "ERE", [PortfolioSpec("direct_001", str(_PACK),
                                  source_portfolio_label="Direct Book")],
            target=target, regime=regime, out_root=td, adapters=ad,
            full_pipeline=full_pipeline, created_at=_NOW, run_id="orun_it")

    def test_mi_full_pipeline_uses_mi_contract_not_annex2(self):
        # Funded MI full pipeline: MI contract (mi_semantics), Gate 2/3 attempted,
        # governed halt on this intentionally-incomplete pack — NOT Annex 2.
        from engine.orchestrator_agent.state import STEP_DONE, STEP_HALTED, STEP_FAILED
        state = self._run(target="mi", mode="mi_only", full_pipeline=True)
        onboard = state.portfolios[0].step("onboard")
        self.assertEqual(onboard.status, STEP_DONE)
        # MI contract used — not the ESMA Annex 2 contract.
        self.assertEqual(onboard.readiness.get("target_contract"), "mi_semantics")
        self.assertTrue(onboard.manifest_path)                 # MI handoff produced
        # The FULL pipeline was attempted (transform reached), not lean onboard→stamp.
        self.assertIn(state.portfolios[0].step("transform").status, (STEP_HALTED, STEP_FAILED, STEP_DONE))
        self.assertNotEqual(state.portfolios[0].step("transform").status, "pending")
        # Incomplete pack → fail-closed: nothing published.
        if state.status != STEP_DONE:
            self.assertIsNone(state.central_canonical_path)

    def test_deterministic_mi_full_pipeline_emits_handoff(self):
        # THE reported bug, end-to-end: a KNOWN source processed DETERMINISTICALLY
        # (no LLM review), target=mi, full_pipeline=True. Onboarding must still emit
        # the MI-contract handoff so Gate 2 has a contract to transform — previously
        # it produced NO handoff manifest and transform halted with "produced no
        # handoff manifest".
        import json
        from engine.orchestrator_agent.state import STEP_DONE
        state = self._run(target="mi", mode="mi_only", full_pipeline=True,
                          processing_mode="deterministic")
        onboard = state.portfolios[0].step("onboard")
        self.assertEqual(onboard.status, STEP_DONE)
        # Handoff manifest IS emitted now (was None on the deterministic path).
        self.assertTrue(onboard.manifest_path)
        self.assertEqual(onboard.readiness.get("target_contract"), "mi_semantics")
        # Gate 2 receives the MI contract, NOT Annex 2, and there are no registry gaps.
        m = json.loads(Path(onboard.manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(m.get("target_contract_id"), "mi_semantics_field_registry")
        self.assertEqual(m.get("registry_gap_count"), 0)
        # The "produced no handoff manifest" halt no longer occurs; transform reached
        # its readiness gate (any residual halt is a governed decision, not a missing
        # contract).
        self.assertFalse(any("produced no handoff manifest" in b.lower()
                             for b in state.blockers))
        self.assertNotEqual(state.portfolios[0].step("transform").status, "pending")

    def test_cli_vs_azure_onboarding_parity(self):
        # STEP-1 PROOF: the headless Azure onboarding produces the SAME output as the
        # Codespaces CLI (run_operator_workflow) for identical inputs — same 28a
        # coverage per field and the same handoff readiness/contract. The runtime is
        # the same code; only the approve→rerun bridge differed.
        import json, tempfile
        from engine.onboarding_agent import workflow as _wf, onboarding_handoff
        from engine.orchestrator_agent.adapters import RealAgentAdapters, PortfolioSpec

        def _cov(pdir):
            d = json.loads((Path(pdir) / "28a_target_coverage_matrix.json").read_text())
            return {r["target_field"]: r.get("coverage_status") for r in d["rows"]}

        # CLI onboarding, then build the handoff with the SAME function the adapter uses.
        cli = Path(tempfile.mkdtemp(prefix="cli_it_")) / "p"
        self.addCleanup(lambda: __import__("shutil").rmtree(cli.parent, ignore_errors=True))
        _wf.run_operator_workflow(
            input_dir=str(_PACK), client_name="ERE_IT", client_id="direct_001", run_id="run",
            project_dir=str(cli), mode="mi_only", registry=_REGISTRY, aliases_dir="config/system",
            enable_mapping_review=True, reporting_date="2025-11-30", target_first_decisions="")
        h_cli = onboarding_handoff.build_handoff_package(
            str(cli), Path(cli) / "output", client_id="direct_001", client_name="ERE_IT",
            run_id="run", mode="mi_only", registry=_REGISTRY, aliases_dir="config/system",
            reporting_date="2025-11-30")["manifest"]

        # Azure headless onboarding via the orchestrator adapter.
        az = Path(tempfile.mkdtemp(prefix="az_it_")) / "p"
        self.addCleanup(lambda: __import__("shutil").rmtree(az.parent, ignore_errors=True))
        ad = RealAgentAdapters(registry=_REGISTRY, client_name="ERE_IT", onboarding_mode="mi_only",
                               processing_mode="deterministic", full_pipeline=True,
                               reporting_period="2025-11-30")
        res = ad.onboard(PortfolioSpec("direct_001", str(_PACK), source_portfolio_type="direct",
                                       allow_unknown_acquisition_date=True), az)
        h_az = json.loads(Path(res.manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(_cov(cli), _cov(az))            # identical coverage, every field
        for k in ("target_contract_id", "ready_for_transformation_validation",
                  "blocking_decision_count", "registry_gap_count",
                  "source_mapped_count", "source_absent_count", "target_field_count"):
            self.assertEqual(h_cli.get(k), h_az.get(k), k)

    def test_reporting_date_derived_from_folder_period(self):
        # THE fix: with the folder reporting_period supplied, the MI-contract
        # portfolio-level reporting_date is DERIVED from it (the raw pack has no
        # reporting_date column), so the onboarding handoff is READY with no
        # blocking decisions — no manual approval per monthly pack — and every loan
        # is retained (period fed via context, NOT run_id → no eligibility filter).
        import json
        from engine.orchestrator_agent.state import STEP_DONE
        state = self._run(target="mi", mode="mi_only", full_pipeline=True,
                          processing_mode="deterministic", reporting_period="2025-11-30")
        onboard = state.portfolios[0].step("onboard")
        self.assertEqual(onboard.status, STEP_DONE)
        self.assertEqual(onboard.readiness.get("loan_count"), 36)      # loans preserved
        m = json.loads(Path(onboard.manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(m.get("target_contract_id"), "mi_semantics_field_registry")
        self.assertEqual(m.get("registry_gap_count"), 0)
        self.assertEqual(m.get("blocking_decision_count"), 0)          # reporting_date no longer blocks
        self.assertTrue(m.get("ready_for_transformation_validation")) # handoff READY
        # reporting_date resolved from the run period, non-blocking.
        cov = json.loads(Path(str(m["target_coverage_matrix_path"])
                              .replace(".csv", ".json")).read_text(encoding="utf-8"))
        rd = next(r for r in cov["rows"] if r.get("target_field") == "reporting_date")
        self.assertEqual(rd.get("selected_value"), "2025-11-30")
        self.assertFalse(rd.get("blocking"))
        self.assertEqual(rd.get("coverage_basis"), "run_context_period_inference")

    def test_deterministic_mi_halt_diagnostics_are_actionable(self):
        # End-to-end with real agents: a not-ready handoff (no reporting_period, so
        # reporting_date blocks) must surface WHICH readiness gate failed + the actual
        # blocking decision, and — because the transform halted at its GUARD — the
        # run summary re-points the failed gate to ONBOARDING with a specific,
        # non-mapping inspect_onboarding action.
        from apps.blob_trigger_app.orchestrator_invoke import _run_diagnostics
        from apps.blob_trigger_app.ops_advice import (
            next_operator_action, ACT_INSPECT_ONBOARDING)
        state = self._run(target="mi", mode="mi_only", full_pipeline=True,
                          processing_mode="deterministic")
        d = _run_diagnostics(state)
        self.assertEqual(d["registry_gap_count"], 0)
        self.assertEqual(d["mapping_recommendations"], [])
        hr = d["handoff_readiness"]
        self.assertFalse(hr["ready_for_transformation_validation"])
        self.assertTrue(hr["failed_readiness_gates"])              # explains the failed gate
        self.assertGreaterEqual(hr["blocking_decision_count"], 1)
        self.assertIn("reporting_date", hr["missing_target_fields"])
        self.assertTrue(any(b["target_field"] == "reporting_date"
                            for b in hr["blocking_decisions"]))
        self.assertTrue(hr.get("handoff_manifest"))               # embedded for durability
        # run summary + gate observability
        self.assertEqual(d["run_summary"]["failed_gate"], "onboarding")   # re-pointed
        self.assertEqual(d["run_summary"]["gate_status"]["transform"], "halted")
        na = next_operator_action({
            "event_decision": "known_source_halted", "status": "halted",
            "pack_key": "pk", "source_portfolio_id": "direct_001",
            "orchestrator_run_id": "orun_it", "orchestrator_diagnostics": d})
        self.assertEqual(na["action"], ACT_INSPECT_ONBOARDING)
        self.assertIn("reporting_date", na["summary"])

    def test_mi_path_real_agents_green(self):
        import pandas as pd
        from engine.orchestrator_agent.state import STEP_DONE
        state = self._run(target="mi", mode="mi_only")
        self.assertEqual(state.status, STEP_DONE, state.blockers)
        self.assertEqual(state.portfolios[0].step("onboard").status, STEP_DONE)
        self.assertEqual(state.portfolios[0].step("stamp").status, STEP_DONE)
        self.assertEqual(state.assemble.status, STEP_DONE)
        self.assertEqual(state.route.status, STEP_DONE)
        self.assertTrue(state.central_canonical_path)
        df = pd.read_csv(state.central_canonical_path)
        self.assertIn("source_portfolio_id", df.columns)
        self.assertEqual(set(df["source_portfolio_id"].unique()), {"direct_001"})
        self.assertGreater(len(df), 0)

    def test_regime_path_real_agents_runs_and_gates(self):
        from engine.orchestrator_agent.state import STEP_DONE, STEP_HALTED
        state = self._run(target="regime", mode="regulatory_mi", regime="ESMA_Annex2")
        # The real regulatory chain executes; onboarding produces the handoff and
        # transformation runs. On this pack it gates (mandatory regulatory fields),
        # so the run halts — governed auto-halt on real agent output.
        self.assertEqual(state.portfolios[0].step("onboard").status, STEP_DONE)
        self.assertIn(state.status, (STEP_HALTED, STEP_DONE))
        if state.status == STEP_HALTED:
            self.assertTrue(state.blockers)
            self.assertTrue(any("transform" in b.lower() or "valid" in b.lower()
                                for b in state.blockers))
        else:  # if a future pack passes cleanly, projection must have produced output
            self.assertEqual(state.project.status, STEP_DONE)


if __name__ == "__main__":
    unittest.main()

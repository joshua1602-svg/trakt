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

    def _run(self, *, target, mode, regime=None):
        from engine.orchestrator_agent import run_orchestration
        from engine.orchestrator_agent.adapters import RealAgentAdapters, PortfolioSpec
        ad = RealAgentAdapters(registry=_REGISTRY, client_name="ERE_IT",
                               onboarding_mode=mode)
        # Keep the output dir alive for the test body (assertions read artifacts).
        td = tempfile.mkdtemp(prefix="orch_it_")
        self.addCleanup(lambda: __import__("shutil").rmtree(td, ignore_errors=True))
        return run_orchestration(
            "ERE", [PortfolioSpec("direct_001", str(_PACK),
                                  source_portfolio_label="Direct Book")],
            target=target, regime=regime, out_root=td, adapters=ad,
            created_at=_NOW, run_id="orun_it")

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

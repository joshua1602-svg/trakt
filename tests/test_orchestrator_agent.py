#!/usr/bin/env python3
"""tests/test_orchestrator_agent.py

Governed Agentic Orchestration conductor: per-portfolio fan-out, readiness-flag
gates (governed auto-halt + resume), real per-portfolio provenance stamping,
real Assembler consolidation and real MI routing. Onboarding / Transformation /
Validation are stubbed (their internals are tested elsewhere); the stub inherits
the REAL stamp/assemble/route from AgentAdapters.

Run: python -m unittest tests.test_orchestrator_agent
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.orchestrator_agent import (
    AgentAdapters, PortfolioSpec, RunState, run_orchestration,
    STEP_DONE, STEP_HALTED,
)
from engine.orchestrator_agent.adapters import StepResult

_NOW = "2026-09-01T00:00:00+00:00"


class StubAdapters(AgentAdapters):
    """Stubs onboard/transform/validate; inherits REAL stamp/assemble/route."""

    def __init__(self, *, blocking_onboard=(), blocking_validate=(), loans=None):
        self.blocking_onboard = set(blocking_onboard)
        self.blocking_validate = set(blocking_validate)
        self.loans = loans or {}

    def onboard(self, spec: PortfolioSpec, work_dir: Path) -> StepResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        manifest = work_dir / "24_onboarding_handoff_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        if spec.source_portfolio_id in self.blocking_onboard:
            return StepResult(ok=False, blocking=True,
                              blockers=["mapping review pending"],
                              message="onboarding not ready")
        return StepResult(ok=True, manifest_path=str(manifest),
                          readiness={"ready_for_transformation_validation": True})

    def transform(self, spec, handoff_manifest, work_dir) -> StepResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        loans = self.loans.get(spec.source_portfolio_id, ["L1", "L2"])
        csv = work_dir / "31_transformed_canonical_tape.csv"
        pd.DataFrame({
            "loan_identifier": loans,
            "current_outstanding_balance": [100.0] * len(loans),
        }).to_csv(csv, index=False)
        manifest = work_dir / "30_transformation_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return StepResult(ok=True, output_path=str(csv), manifest_path=str(manifest),
                          readiness={"ready_for_validation": True})

    def validate(self, spec, transformation_manifest, work_dir) -> StepResult:
        csv = Path(transformation_manifest).parent / "31_transformed_canonical_tape.csv"
        if spec.source_portfolio_id in self.blocking_validate:
            return StepResult(ok=False, blocking=True,
                              blockers=["blocking validation exceptions"],
                              message="validation blocked")
        return StepResult(ok=True, output_path=str(csv),
                          readiness={"ready_for_validation_complete": True})


def _specs():
    return [
        PortfolioSpec("direct_001", "in/direct", source_portfolio_label="Direct Book"),
        PortfolioSpec("acquired_001", "in/acq1", acquisition_date="2026-08-15",
                      seller_name="Seller A", source_portfolio_label="Acquired Portfolio 1"),
    ]


class TestOrchestratorHappyPath(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_full_mi_pipeline(self):
        state = run_orchestration("ERE", _specs(), target="mi", out_root=self.out,
                                  adapters=StubAdapters(loans={"direct_001": ["L1", "L2"],
                                                               "acquired_001": ["L1", "L9"]}),
                                  created_at=_NOW, run_id="orun_test")
        self.assertEqual(state.status, STEP_DONE)
        for p in state.portfolios:
            self.assertEqual(p.status, STEP_DONE)
            for s in ("onboard", "transform", "validate", "stamp"):
                self.assertEqual(p.step(s).status, STEP_DONE)
        self.assertEqual(state.assemble.status, STEP_DONE)
        self.assertEqual(state.route.status, STEP_DONE)
        # Central canonical exists, combines both portfolios, provenance stamped.
        df = pd.read_csv(state.central_canonical_path)
        self.assertEqual(sorted(df["source_portfolio_id"].unique()),
                         ["acquired_001", "direct_001"])
        self.assertEqual(len(df), 4)
        # direct_001/L1 and acquired_001/L1 coexist (composite key, no collision).
        self.assertIn("direct_001/L1", set(df["platform_loan_key"]))
        self.assertIn("acquired_001/L1", set(df["platform_loan_key"]))
        # MI routing points at the central canonical.
        self.assertEqual(state.route.readiness["MI_AGENT_PLATFORM_CANONICAL"],
                         state.central_canonical_path)

    def test_single_portfolio(self):
        state = run_orchestration(
            "ERE", [PortfolioSpec("direct_001", "in/d")], target="mi",
            out_root=self.out, adapters=StubAdapters(), created_at=_NOW, run_id="orun_one")
        self.assertEqual(state.status, STEP_DONE)
        df = pd.read_csv(state.central_canonical_path)
        self.assertEqual(list(df["source_portfolio_id"].unique()), ["direct_001"])

    def test_acquired_provenance_stamped(self):
        state = run_orchestration("ERE", _specs(), target="mi", out_root=self.out,
                                  adapters=StubAdapters(), created_at=_NOW, run_id="orun_prov")
        df = pd.read_csv(state.central_canonical_path)
        acq = df[df["source_portfolio_id"] == "acquired_001"].iloc[0]
        self.assertEqual(acq["source_portfolio_type"], "acquired")
        self.assertEqual(str(acq["acquisition_date"]), "2026-08-15")
        self.assertEqual(acq["portfolio_cohort"], "acquired_001")


class TestGovernedAutoHalt(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_halts_on_blocking_onboard_and_resumes(self):
        # First run: acquired_001 onboarding is blocking → run halts.
        state = run_orchestration(
            "ERE", _specs(), target="mi", out_root=self.out,
            adapters=StubAdapters(blocking_onboard={"acquired_001"}),
            created_at=_NOW, run_id="orun_halt")
        self.assertEqual(state.status, STEP_HALTED)
        self.assertEqual(state.assemble.status, "pending")  # never reached
        self.assertTrue(any("acquired_001/onboard" in b for b in state.blockers))
        # State persisted + resumable.
        self.assertTrue(Path(state.state_path()).exists())

        # Operator resolves the blocker; resume with a non-blocking adapter.
        reloaded = RunState.load(state.state_path())
        resumed = run_orchestration(
            "ERE", [], target="mi", out_root=self.out, adapters=StubAdapters(),
            created_at=_NOW, resume_state=reloaded)
        self.assertEqual(resumed.status, STEP_DONE)
        df = pd.read_csv(resumed.central_canonical_path)
        self.assertEqual(sorted(df["source_portfolio_id"].unique()),
                         ["acquired_001", "direct_001"])

    def test_halts_on_blocking_validation(self):
        state = run_orchestration(
            "ERE", _specs(), target="mi", out_root=self.out,
            adapters=StubAdapters(blocking_validate={"direct_001"}),
            created_at=_NOW, run_id="orun_valhalt")
        self.assertEqual(state.status, STEP_HALTED)
        self.assertTrue(any("validation" in b.lower() for b in state.blockers))

    def test_completed_steps_not_rerun_on_resume(self):
        # Halt at validation; onboarding+transform already done must be skipped.
        state = run_orchestration(
            "ERE", [PortfolioSpec("direct_001", "in/d")], target="mi",
            out_root=self.out, adapters=StubAdapters(blocking_validate={"direct_001"}),
            created_at=_NOW, run_id="orun_skip")
        self.assertEqual(state.status, STEP_HALTED)
        self.assertEqual(state.portfolios[0].step("onboard").status, STEP_DONE)
        self.assertEqual(state.portfolios[0].step("transform").status, STEP_DONE)
        resumed = run_orchestration(
            "ERE", [], target="mi", out_root=self.out, adapters=StubAdapters(),
            created_at=_NOW, resume_state=RunState.load(state.state_path()))
        self.assertEqual(resumed.status, STEP_DONE)


class TestRunStateSerialization(unittest.TestCase):

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            state = run_orchestration("ERE", _specs(), target="mi", out_root=td,
                                      adapters=StubAdapters(), created_at=_NOW,
                                      run_id="orun_ser")
            reloaded = RunState.load(state.state_path())
            self.assertEqual(reloaded.run_id, state.run_id)
            self.assertEqual(len(reloaded.portfolios), 2)
            self.assertEqual(reloaded.status, STEP_DONE)
            self.assertEqual(reloaded.central_canonical_path, state.central_canonical_path)


if __name__ == "__main__":
    unittest.main()

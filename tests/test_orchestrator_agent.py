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
    """Stubs onboard/transform/validate; inherits REAL stamp/assemble/route/project.

    onboard writes a canonical CSV (the MI central tape, used directly by the MI
    path) AND a handoff manifest (used by the regulatory transform→validate path),
    so the same stub serves both target pipelines.
    """

    def __init__(self, *, blocking_onboard=(), blocking_validate=(), loans=None,
                 project_result=None):
        self.blocking_onboard = set(blocking_onboard)
        self.blocking_validate = set(blocking_validate)
        self.loans = loans or {}
        self._project_result = project_result

    def _canonical(self, spec, work_dir) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        loans = self.loans.get(spec.source_portfolio_id, ["L1", "L2"])
        csv = work_dir / "canonical.csv"
        pd.DataFrame({
            "loan_identifier": loans,
            "current_outstanding_balance": [100.0] * len(loans),
        }).to_csv(csv, index=False)
        return csv

    def onboard(self, spec: PortfolioSpec, work_dir: Path) -> StepResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        manifest = work_dir / "24_onboarding_handoff_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        if spec.source_portfolio_id in self.blocking_onboard:
            return StepResult(ok=False, blocking=True,
                              blockers=["mapping review pending"],
                              message="onboarding not ready")
        # MI canonical (central tape) lives at output_path; handoff at manifest_path.
        csv = self._canonical(spec, work_dir / "central")
        return StepResult(ok=True, output_path=str(csv), manifest_path=str(manifest),
                          readiness={"ready_for_transformation_validation": True})

    def transform(self, spec, handoff_manifest, work_dir) -> StepResult:
        csv = self._canonical(spec, Path(handoff_manifest).parent / "transform")
        manifest = Path(handoff_manifest).parent / "transform" / "30_transformation_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return StepResult(ok=True, output_path=str(csv), manifest_path=str(manifest),
                          readiness={"ready_for_validation": True})

    def validate(self, spec, transformation_manifest, work_dir) -> StepResult:
        csv = Path(transformation_manifest).parent / "canonical.csv"
        if spec.source_portfolio_id in self.blocking_validate:
            return StepResult(ok=False, blocking=True,
                              blockers=["blocking validation exceptions"],
                              message="validation blocked")
        return StepResult(ok=True, output_path=str(csv),
                          readiness={"ready_for_validation_complete": True})

    def project(self, central_canonical, out_dir, regime) -> StepResult:
        # Avoid invoking the heavy regime projector in unit tests.
        if self._project_result is not None:
            return self._project_result
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        proj = Path(out_dir) / f"central_{regime}_projected.csv"
        proj.write_text("RREL1\n", encoding="utf-8")
        return StepResult(ok=True, output_path=str(proj),
                          readiness={"regime": regime, "projected_csv": str(proj)})


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
            # MI path runs onboard → stamp only (transform/validate are the
            # regulatory chain).
            for s in ("onboard", "stamp"):
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
        # Validation only runs on the regulatory (regime) path.
        state = run_orchestration(
            "ERE", _specs(), target="regime", regime="ESMA_Annex2", out_root=self.out,
            adapters=StubAdapters(blocking_validate={"direct_001"}),
            created_at=_NOW, run_id="orun_valhalt")
        self.assertEqual(state.status, STEP_HALTED)
        self.assertTrue(any("validation" in b.lower() for b in state.blockers))

    def test_completed_steps_not_rerun_on_resume(self):
        # Halt at validation; onboarding+transform already done must be skipped.
        state = run_orchestration(
            "ERE", [PortfolioSpec("direct_001", "in/d")], target="regime",
            regime="ESMA_Annex2", out_root=self.out,
            adapters=StubAdapters(blocking_validate={"direct_001"}),
            created_at=_NOW, run_id="orun_skip")
        self.assertEqual(state.status, STEP_HALTED)
        self.assertEqual(state.portfolios[0].step("onboard").status, STEP_DONE)
        self.assertEqual(state.portfolios[0].step("transform").status, STEP_DONE)
        resumed = run_orchestration(
            "ERE", [], target="regime", regime="ESMA_Annex2", out_root=self.out,
            adapters=StubAdapters(), created_at=_NOW,
            resume_state=RunState.load(state.state_path()))
        self.assertEqual(resumed.status, STEP_DONE)


class TestRegimeTarget(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_regime_runs_full_chain_then_projects(self):
        state = run_orchestration(
            "ERE", _specs(), target="regime", regime="ESMA_Annex2", out_root=self.out,
            adapters=StubAdapters(), created_at=_NOW, run_id="orun_regime")
        self.assertEqual(state.status, STEP_DONE)
        for p in state.portfolios:
            for s in ("onboard", "transform", "validate", "stamp"):
                self.assertEqual(p.step(s).status, STEP_DONE)
        self.assertEqual(state.assemble.status, STEP_DONE)
        self.assertEqual(state.project.status, STEP_DONE)   # regime projection ran
        self.assertEqual(state.route.status, "pending")      # MI route not on regime-only
        self.assertEqual(state.project.readiness.get("regime"), "ESMA_Annex2")

    def test_all_target_routes_mi_and_projects(self):
        state = run_orchestration(
            "ERE", [PortfolioSpec("direct_001", "in/d")], target="all",
            regime="ESMA_Annex2", out_root=self.out, adapters=StubAdapters(),
            created_at=_NOW, run_id="orun_all")
        self.assertEqual(state.status, STEP_DONE)
        self.assertEqual(state.route.status, STEP_DONE)
        self.assertEqual(state.project.status, STEP_DONE)


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


class TestFullPipelineMI(unittest.TestCase):
    """Funded MI runs the FULL production path (onboard→transform→validate→stamp)
    — the same steps as the regulatory/CLI flow — routing to MI, not the lean
    mi_only shortcut."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_steps_parity_with_cli_regulatory_path(self):
        from engine.orchestrator_agent.orchestrator import (
            steps_for_target, onboarding_mode_for_target)
        # Lean MI shortcut (unchanged default).
        self.assertEqual(tuple(steps_for_target("mi")), ("onboard", "stamp"))
        # Full-pipeline MI == the regulatory/CLI steps.
        self.assertEqual(tuple(steps_for_target("mi", full_pipeline=True)),
                         ("onboard", "transform", "validate", "stamp"))
        self.assertEqual(tuple(steps_for_target("mi", full_pipeline=True)),
                         tuple(steps_for_target("regime")))
        # Onboarding mode: full MI uses the regulatory onboarding (emits handoff).
        self.assertEqual(onboarding_mode_for_target("mi"), "mi_only")
        self.assertEqual(onboarding_mode_for_target("mi", full_pipeline=True),
                         "regulatory_mi")

    def test_full_pipeline_runs_gate2_gate3_and_routes_to_mi(self):
        state = run_orchestration(
            "ERE", _specs(), target="mi", out_root=self.out, adapters=StubAdapters(),
            created_at=_NOW, run_id="orun_full", full_pipeline=True)
        self.assertEqual(state.status, STEP_DONE)
        self.assertTrue(state.full_pipeline)
        for p in state.portfolios:
            # Gate 2 (transform) + Gate 3 (validate) DID run, then stamp.
            for s in ("onboard", "transform", "validate", "stamp"):
                self.assertEqual(p.step(s).status, STEP_DONE, s)
        self.assertEqual(state.assemble.status, STEP_DONE)
        self.assertEqual(state.route.status, STEP_DONE)        # routed to MI
        self.assertFalse(state.project.done)                   # no regime projection
        self.assertIsNotNone(state.central_canonical_path)

    def test_validation_failure_blocks_publish(self):
        state = run_orchestration(
            "ERE", _specs(), target="mi", out_root=self.out,
            adapters=StubAdapters(blocking_validate={"direct_001"}),
            created_at=_NOW, run_id="orun_block", full_pipeline=True)
        self.assertEqual(state.status, STEP_HALTED)            # halts at validate
        self.assertIsNone(state.central_canonical_path)        # NOT published
        self.assertFalse(state.assemble.done)

    def test_force_publish_overrides_validation_failure(self):
        state = run_orchestration(
            "ERE", _specs(), target="mi", out_root=self.out,
            adapters=StubAdapters(blocking_validate={"direct_001"}),
            created_at=_NOW, run_id="orun_force", full_pipeline=True,
            force_publish=True)
        self.assertEqual(state.status, STEP_DONE)              # proceeds past validation
        self.assertIsNotNone(state.central_canonical_path)     # published anyway
        self.assertTrue(any("FORCE-PUBLISHED" in b for b in state.blockers))

    def test_lean_mi_unchanged_without_full_pipeline(self):
        state = run_orchestration(
            "ERE", _specs(), target="mi", out_root=self.out, adapters=StubAdapters(),
            created_at=_NOW, run_id="orun_lean")
        self.assertEqual(state.status, STEP_DONE)
        for p in state.portfolios:
            self.assertFalse(p.step("transform").done)         # lean path: no Gate 2/3
            self.assertFalse(p.step("validate").done)


if __name__ == "__main__":
    unittest.main()

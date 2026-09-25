"""A pipeline delivery takes the orchestrator's pipeline route.

ERE's first pipeline delivery stopped at validation with "The first step did
not finish cleanly". Onboarding a pipeline pack builds the central PIPELINE
tape and, by design, none of the funded contract's hand-off — but the
Operations Control Centre ran every delivery through Transformation and
Validation, which need that hand-off, and never told the orchestrator the
delivery was pipeline, so its pipeline branch (the pipeline tape as the
canonical, the funded assembler skipped) was never reached.

A pipeline run now goes onboard -> stamp with the pipeline tape as the
canonical, and a run that already stopped resumes on that route.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from engine.orchestrator_agent import (PortfolioSpec, STEP_DONE,
                                       run_orchestration)
from engine.orchestrator_agent.adapters import StepResult
from engine.orchestrator_agent.state import STEP_PENDING
from tests.test_orchestrator_agent import StubAdapters

_NOW = "2026-09-25T00:00:00+00:00"


class PipelineStub(StubAdapters):
    """Onboarding produces a pipeline tape and no hand-off, as the real one
    does for a pipeline pack; Transformation fails without the hand-off."""

    def __init__(self):
        super().__init__()
        self.transform_calls = 0

    def onboard(self, spec, work_dir):
        work_dir.mkdir(parents=True, exist_ok=True)
        tape = work_dir / "18a_central_pipeline_tape.csv"
        pd.DataFrame({"application_id": ["C1", "C2"],
                      "loan_amount": [100000, 50000]}).to_csv(tape, index=False)
        return StepResult(ok=True, output_path=str(tape), manifest_path=None,
                          readiness={"central_pipeline_tape": str(tape)})

    def transform(self, spec, handoff_manifest, work_dir):
        self.transform_calls += 1
        return StepResult(ok=False, blocking=True,
                          blockers=["missing onboarding handoff"],
                          message="no handoff")


def _spec(root: Path):
    return [PortfolioSpec("direct_001", str(root / "in"),
                          source_portfolio_label="Direct Book")]


def test_a_pipeline_run_goes_straight_to_the_pipeline_tape():
    with tempfile.TemporaryDirectory() as td:
        a = PipelineStub()
        state = run_orchestration("ERE", _spec(Path(td)), target="mi",
                                  out_root=td, adapters=a, created_at=_NOW,
                                  full_pipeline=False, dataset="pipeline")
        assert a.transform_calls == 0
        assert state.assemble.status == STEP_DONE
        assert state.central_canonical_path
        assert (state.assemble.readiness or {}).get("pipeline_dataset") is True


def test_a_stopped_pipeline_run_resumes_on_the_pipeline_route():
    with tempfile.TemporaryDirectory() as td:
        a = PipelineStub()
        first = run_orchestration("ERE", _spec(Path(td)), target="mi",
                                  out_root=td, adapters=a, created_at=_NOW,
                                  full_pipeline=True)
        assert first.central_canonical_path in (None, "")
        stopped = first.portfolios[0].step("transform")
        assert stopped.status != STEP_DONE      # the funded chain stopped here
        again = run_orchestration("ERE", _spec(Path(td)), target="mi",
                                  out_root=td, adapters=a, created_at=_NOW,
                                  resume_state=first, full_pipeline=False,
                                  dataset="pipeline")
        assert a.transform_calls == 0
        assert again.assemble.status == STEP_DONE
        assert again.central_canonical_path
        p = again.portfolios[0]
        assert p.step("transform").status == STEP_PENDING


def test_the_engine_sends_a_pipeline_run_down_the_pipeline_route():
    import inspect
    from operations_control import engine
    src = inspect.getsource(engine.OpsEngine._execute)
    assert "full_pipeline=not _is_pipeline(run)" in src
    assert 'dataset=("pipeline" if _is_pipeline(run) else "")' in src

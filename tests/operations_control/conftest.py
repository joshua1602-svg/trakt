"""Shared fixtures for the Operations Control Centre test suite.

Everything runs file-backed in a tmp directory: storage, staging, memory,
source registry. Agent execution uses stub adapters (subclassing the SAME
``AgentAdapters`` seam production wires ``RealAgentAdapters`` into), so the
conductor logic, governed translation, persistence and gating are exercised
end-to-end without the heavy agents.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pytest

from apps.blob_trigger_app.source_registry import SourceRegistry
from apps.blob_trigger_app.storage import Storage
from engine.orchestrator_agent.adapters import AgentAdapters, PortfolioSpec, StepResult

from operations_control.contracts import OUTCOME_MI
from operations_control.engine import OpsEngine
from operations_control.stores import OpsLayout, OpsStore

OP_ALL = "tok-all"
OP_A = "tok-client-a"
OP_B = "tok-client-b"
OP_ADMIN = "tok-admin"

OPERATORS = {
    OP_ALL: {"name": "Root Operator", "clients": ["*"]},
    OP_A: {"name": "Alice", "clients": ["client_a"]},
    OP_B: {"name": "Bob", "clients": ["client_b"]},
    OP_ADMIN: {"name": "Administrator", "clients": ["*"], "role": "admin"},
}


@pytest.fixture()
def ops_env(tmp_path, monkeypatch):
    """File-backed environment: blob root, staging, memory, registry, auth."""
    blob_root = tmp_path / "blob"
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(blob_root))
    monkeypatch.setenv("TRAKT_OPS_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("TRAKT_OPS_STAGING_ROOT", str(tmp_path / "staging"))
    monkeypatch.setenv("TRAKT_OPS_OPERATORS", json.dumps(OPERATORS))
    monkeypatch.setenv("TRAKT_SOURCE_REGISTRY_URI",
                       "blob://trakt-state/registry/source_registry.yaml")
    return {"tmp": tmp_path, "blob_root": blob_root}


@pytest.fixture()
def storage(ops_env) -> Storage:
    return Storage(ops_env["blob_root"])


@pytest.fixture()
def store(storage) -> OpsStore:
    return OpsStore(storage, OpsLayout("operations-control"))


@pytest.fixture()
def source_registry(storage) -> SourceRegistry:
    return SourceRegistry("blob://trakt-state/registry/source_registry.yaml",
                          storage=storage)


@pytest.fixture()
def delivery_dir(tmp_path) -> Path:
    d = tmp_path / "delivery"
    d.mkdir()
    (d / "loans.csv").write_text(
        "loan_ref,balance,rate\nL1,1000,0.05\nL2,2000,0.06\n", encoding="utf-8")
    return d


class ScriptedAdapters(AgentAdapters):
    """Deterministic stand-in for the real agents, driven by a scenario:

      * ``happy``   — every step succeeds;
      * ``mapping_halt`` — onboarding halts with a pending decisions template
        until an approved decisions file exists in the work dir, then succeeds;
      * ``validation_halt`` — validation blocks unless force-published.
    """

    def __init__(self, scenario: str = "happy"):
        self.scenario = scenario
        self.calls: list = []

    # -- helpers -------------------------------------------------------- #
    def _tape(self, work_dir: Path, name: str = "tape.csv") -> str:
        work_dir.mkdir(parents=True, exist_ok=True)
        out = work_dir / name
        out.write_text("loan_identifier,balance\nL1,1000\nL2,2000\n",
                       encoding="utf-8")
        return str(out)

    def onboard(self, spec: PortfolioSpec, work_dir: Path) -> StepResult:
        self.calls.append("onboard")
        work_dir = Path(work_dir)
        approved = work_dir / "34_target_first_decisions_approved.yaml"
        if self.scenario == "mapping_halt" and not approved.exists():
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "34_target_first_decisions.yaml").write_text(
                "decisions:\n"
                "- decision_id: dec-001\n"
                "  target_field: current_valuation_indexed\n"
                "  decision_type: mapping_gap\n"
                "  status: pending\n"
                "  blocking: true\n"
                "  recommended_action: accept_mapping\n"
                "  available_actions: [accept_mapping, mark_unavailable]\n",
                encoding="utf-8")
            return StepResult(ok=False, blocking=True,
                              blockers=["onboarding not "
                                        "ready_for_transformation_validation "
                                        "(mapping review / blocking decision "
                                        "pending)"],
                              message="mapping review pending")
        work_dir.mkdir(parents=True, exist_ok=True)
        handoff = work_dir / "handoff.json"
        handoff.write_text(
            json.dumps({"ready_for_transformation_validation": True}),
            encoding="utf-8")
        return StepResult(ok=True, output_path=self._tape(work_dir),
                          manifest_path=str(handoff),
                          readiness={"loan_count": 2}, message="onboarded")

    def transform(self, spec, handoff_manifest, work_dir) -> StepResult:
        self.calls.append("transform")
        return StepResult(ok=True, output_path=self._tape(Path(work_dir), "t.csv"),
                          manifest_path=str(Path(work_dir) / "m.json"),
                          message="transformed")

    def validate(self, spec, transformation_manifest, work_dir) -> StepResult:
        self.calls.append("validate")
        if self.scenario == "validation_halt":
            return StepResult(ok=False, blocking=True,
                              blockers=["validation has blocking exceptions "
                                        "(not ready_for_validation_complete)"],
                              message="validation blocked")
        return StepResult(ok=True, output_path=self._tape(Path(work_dir), "v.csv"),
                          message="validated")

    def stamp_provenance(self, spec, validated_csv, out_dir) -> StepResult:
        self.calls.append("stamp")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{spec.source_portfolio_id}_canonical_typed.csv"
        out.write_text(Path(validated_csv).read_text(encoding="utf-8"),
                       encoding="utf-8")
        return StepResult(ok=True, output_path=str(out), message="stamped")

    def assemble(self, stamped_paths, out_dir, client_id, target, *,
                 regime=None) -> StepResult:
        self.calls.append("assemble")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "platform_canonical_typed.csv"
        out.write_text(Path(stamped_paths[0]).read_text(encoding="utf-8"),
                       encoding="utf-8")
        return StepResult(ok=True, output_path=str(out),
                          readiness={"total_rows": 2, "portfolio_count": 1},
                          message="assembled")

    def route_mi(self, central_canonical) -> StepResult:
        self.calls.append("route")
        return StepResult(ok=True, output_path=central_canonical, message="routed")

    def project(self, central_canonical, out_dir, regime) -> StepResult:
        self.calls.append("project")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "central_projected.csv"
        out.write_text("a,b\n1,2\n", encoding="utf-8")
        return StepResult(ok=True, output_path=str(out), message="projected")


class StubAnnex2Stages:
    """Stub for the governed Annex 2 delivery chain seam. Same method
    signatures as operations_control.annex2.stages.Annex2Stages."""

    def __init__(self, scenario: str = "happy"):
        self.scenario = scenario
        self.calls: list = []

    def run_projection(self, *, central_canonical, out_dir, client_config=None):
        from operations_control.annex2.stages import StageOutcome
        self.calls.append(("projection", str(client_config or "")))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "annex2_ESMA_Annex2_projected.csv"
        p.write_text("RREL1,RREL2\nSECID,L1\n", encoding="utf-8")
        return StageOutcome(ok=True, summary="Regulatory data projected.",
                            artefacts={"projected_csv": str(p)},
                            metrics={"records": 1, "fields": 2})

    def run_normalisation(self, *, projected_csv, out_dir, rules_path=None):
        from operations_control.annex2.stages import StageOutcome
        self.calls.append("normalisation")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.scenario == "normalise_block":
            return StageOutcome(
                ok=False, blocking=True,
                summary="Annex 2 preparation found problems that need your "
                        "review.",
                blockers=["A required value is missing from the data. "
                          "(2 instances)"])
        p = out_dir / "annex2_delivery_ready.csv"
        p.write_text("RREL1,RREL2\nSECID,L1\n", encoding="utf-8")
        return StageOutcome(
            ok=True,
            summary="Annex 2 data has been prepared for XML generation.",
            artefacts={"delivery_ready_csv": str(p)},
            metrics={"input_records": 1, "output_records": 1,
                     "input_fields": 2, "output_fields": 2,
                     "rules_version": "rules@test"})

    def run_xml(self, *, delivery_csv, out_dir, xsd_path=None):
        from operations_control.annex2.stages import StageOutcome
        self.calls.append("xml")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.scenario == "xsd_fail":
            return StageOutcome(
                ok=False, blocking=True,
                summary="The Annex 2 XML could not be created or did not "
                        "pass the regulator's format check.",
                blockers=["The generated file did not pass the regulator's "
                          "format check."])
        p = out_dir / "annex2_submission.xml"
        p.write_text("<Document/>", encoding="utf-8")
        iv = out_dir / "builder_interventions.json"
        iv.write_text('{"interventions": []}', encoding="utf-8")
        return StageOutcome(
            ok=True,
            summary="Annex 2 XML passed schema validation, but automatic "
                    "values require review.",
            warnings=["Automatic values were inserted in 144 field "
                      "instances. Review is required before publication."],
            artefacts={"xml": str(p), "interventions": str(iv)},
            metrics={"records": 1, "fields": 2, "xml_bytes": 11,
                     "xsd": "xsd@test", "xsd_result": "PASSED",
                     "xml_sha256": "sha256:stub",
                     "nd_injected": 144, "review_required_instances": 144})


def make_client_config(tmp_path: Path, *, valid: bool = True) -> Path:
    p = tmp_path / "client_config.yaml"
    lei = "STUBLEI0000000000042" if valid else "NOT_A_VALID_LEI"
    p.write_text(
        f"client:\n  country: GB\n  base_currency: GBP\n"
        f"annex2:\n  originator_legal_entity_identifier: \"{lei}\"\n"
        f"  originator_establishment_country: \"GB\"\n", encoding="utf-8")
    return p


def make_engine(store: OpsStore, source_registry: SourceRegistry,
                scenario: str = "happy",
                adapters: Optional[AgentAdapters] = None,
                annex2_scenario: str = "happy",
                annex2_stages=None,
                client_config: Optional[Path] = None) -> OpsEngine:
    shared = adapters or ScriptedAdapters(scenario)
    a2 = annex2_stages or StubAnnex2Stages(annex2_scenario)
    if client_config is None:
        import tempfile
        client_config = make_client_config(
            Path(tempfile.mkdtemp(prefix="ops-cfg-")))
    return OpsEngine(store,
                     adapter_factory=lambda run: shared,
                     source_registry_factory=lambda: source_registry,
                     annex2_stages=a2,
                     client_config_path=str(client_config))


@pytest.fixture()
def engine(store, source_registry) -> OpsEngine:
    return make_engine(store, source_registry)


def wait_for(predicate, timeout: float = 15.0, interval: float = 0.05):
    """Poll until ``predicate()`` is truthy (returning its value) or fail."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = predicate()
        if v:
            return v
        time.sleep(interval)
    raise AssertionError("condition not reached within timeout")


def start_and_wait(engine: OpsEngine, run, statuses=("needs_review", "blocked",
                                                     "awaiting_publication",
                                                     "failed")):
    engine.start(run, actor="test")
    return wait_for(lambda: (
        (lambda r: r if r and r.status in statuses else None)(
            engine.store.load_workflow(run.client_id, run.workflow_id))))


def register_and_create(engine: OpsEngine, delivery_dir: Path,
                        client_id: str = "client_a",
                        portfolio_id: str = "pf1",
                        outcome: str = OUTCOME_MI,
                        period: str = "2026-06-30",
                        workflow_type: Optional[str] = None):
    d = engine.register_delivery(client_id=client_id, portfolio_id=portfolio_id,
                                 input_path=str(delivery_dir),
                                 reporting_period=period,
                                 registered_by="test")
    return engine.create_workflow(client_id=client_id,
                                  delivery_id=d["delivery_id"],
                                  outcome=outcome,
                                  workflow_type=workflow_type,
                                  created_by="test")

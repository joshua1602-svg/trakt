"""Tests for the annex-neutral half of the Annex Delivery Agent.

Everything here runs against a *stub* provider registered into a private
registry. That is the point: if these pass without importing Annex 2, the shared
agent genuinely does not depend on any one annex, and a future Annex 3 inherits
the lifecycle, checkpointing, tenancy and publication gating for free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pytest

from trakt_core.context import ExecutionContext
from trakt_core.errors import ErrorCode, TraktError
from trakt_core.tenancy import TenantRecord, TenantRegistry

from engine.annex_delivery_agent.agent import AnnexDeliveryAgent
from engine.annex_delivery_agent.contracts import (
    AnnexDefinition, ArtefactRef, DeliveryRequest, DeliveryResult, DeliveryStatus,
    SchemaValidationResult, STAGE_BUILD, STAGE_NORMALISE, STAGE_PREFLIGHT,
)
from engine.annex_delivery_agent.evidence import (
    AutomaticFill, Coercion, TransformationEvidence,
)
from engine.annex_delivery_agent.occ_adapter import (
    STAGE_OPERATOR_REVIEW, STAGE_PUBLICATION, STATE_ACTION_REQUIRED, STATE_DONE,
    operator_view,
)
from engine.annex_delivery_agent.persistence import (
    Checkpoint, DeliveryRunStore, StagedFile, atomic_write_json,
)
from engine.annex_delivery_agent.providers.base import (
    AnnexProvider, BuildOutcome, NormalisationOutcome, ResolvedInput,
)
from engine.annex_delivery_agent.publication import FilesystemPublicationSink
from engine.annex_delivery_agent.registry import AnnexRegistry, default_registry

# --------------------------------------------------------------------------- #
# A minimal test annex — deliberately outside the production registry
# --------------------------------------------------------------------------- #

STUB_DEFINITION = AnnexDefinition(
    annex_id="TEST_AnnexZ",
    annex_name="Test Annex Z",
    jurisdiction="TEST",
    regulatory_regime="test regime",
    report_version="1.0.0",
    artefact_type="txt",
    authoritative_projector="tests/stub_projector",
    normaliser="tests/stub_normaliser",
    builder="tests/stub_builder",
    validator="tests/stub_validator",
    schema_reference="tests/stub_schema",
)


class StubProvider(AnnexProvider):
    """A provider that exercises the lifecycle without any regulatory logic."""

    definition = STUB_DEFINITION

    # Class-level switches so a test can steer one run.
    fail_configuration = False
    not_applicable = False
    missing_input = False
    normalisation_blocking = 0
    raise_in = ""
    schema_valid = True
    evidence_warnings = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def check_configuration(self, request, report) -> Mapping[str, str]:
        self.calls.append("check_configuration")
        if self.fail_configuration:
            report.block("stub_config",
                         "Test Annex Z cannot be prepared because the test "
                         "configuration has not been supplied.")
        return {"stub_rules": "fingerprint-1"}

    def check_applicability(self, request, report) -> bool:
        self.calls.append("check_applicability")
        if self.not_applicable:
            report.block("report_applicability",
                         "Test Annex Z is not enabled for this client.")
            return False
        return True

    def resolve_input(self, request, report) -> Optional[ResolvedInput]:
        self.calls.append("resolve_input")
        path = Path(request.projected_input)
        if self.missing_input or not path.is_file():
            report.block("projected_input",
                         "Test Annex Z cannot be prepared because the projected "
                         "data for this reporting period was not found.")
            return None
        rows = [line for line in path.read_text().splitlines() if line.strip()]
        return ResolvedInput(path=path, row_count=max(len(rows) - 1, 0),
                             field_count=len(rows[0].split(",")) if rows else 0,
                             field_codes=tuple(rows[0].split(",")) if rows else (),
                             content_hash="stub-hash")

    def preflight(self, request, resolved, report) -> None:
        self.calls.append("preflight")

    def normalise(self, request, resolved, work_dir) -> NormalisationOutcome:
        self.calls.append("normalise")
        if self.raise_in == "normalise":
            raise RuntimeError("stub normaliser exploded")
        target = work_dir / "normalised.csv"
        with StagedFile(target) as staged:
            staged.temp_path.write_text(resolved.path.read_text(), encoding="utf-8")
        return NormalisationOutcome(
            artefact=ArtefactRef.of_file("delivery_ready_csv", target),
            row_count=resolved.row_count, field_count=resolved.field_count,
            blocking_issue_count=self.normalisation_blocking,
            issues=[{"severity": "error", "field": "X", "message": "bad"}]
            if self.normalisation_blocking else [])

    def build(self, request, normalised, work_dir) -> BuildOutcome:
        self.calls.append("build")
        if self.raise_in == "build":
            raise RuntimeError("stub builder exploded")
        target = work_dir / "artefact.txt"
        with StagedFile(target) as staged:
            staged.temp_path.write_text("stub artefact\n", encoding="utf-8")
        return BuildOutcome(artefact=ArtefactRef.of_file("artefact", target),
                            record_count=normalised.row_count,
                            field_count=normalised.field_count,
                            parsed=None, detail={"stub": True})

    def validate(self, request, built) -> SchemaValidationResult:
        self.calls.append("validate")
        if self.raise_in == "validate":
            raise RuntimeError("stub validator exploded")
        return SchemaValidationResult(
            validator="stub", schema_reference="tests/stub_schema",
            schema_sha256="abc", valid=self.schema_valid, executed=True,
            error_count=0 if self.schema_valid else 3,
            errors=() if self.schema_valid else ({"line": 1, "message": "bad"},))

    def inspect(self, request, resolved, normalised, built) -> TransformationEvidence:
        self.calls.append("inspect")
        evidence = TransformationEvidence()
        if self.evidence_warnings:
            evidence.automatic_fills.append(AutomaticFill(
                field_code="ZZ1", xml_path="Rec/ZZ1", records_affected=7,
                original_state="blank", generated_value="ND5", reason="test rule"))
            evidence.coercions.append(Coercion(
                field_code="ZZ2", records_affected=2,
                original_value_sample=("x",), generated_value="2026",
                reason="test coercion"))
        return evidence

    def publication_metadata(self, request, resolved, normalised, built,
                             validation) -> Dict[str, Any]:
        return {"annex_id": self.definition.annex_id}

    def component_versions(self) -> Mapping[str, str]:
        return {"stub_provider": "v1"}


@pytest.fixture(autouse=True)
def _reset_stub():
    StubProvider.fail_configuration = False
    StubProvider.not_applicable = False
    StubProvider.missing_input = False
    StubProvider.normalisation_blocking = 0
    StubProvider.raise_in = ""
    StubProvider.schema_valid = True
    StubProvider.evidence_warnings = True
    yield


@pytest.fixture
def registry() -> AnnexRegistry:
    reg = AnnexRegistry()
    reg.register(STUB_DEFINITION, StubProvider)
    return reg


@pytest.fixture
def tenants() -> TenantRegistry:
    return TenantRegistry({
        "acme": TenantRecord(tenant_id="acme", default_portfolio_id="acme",
                             portfolio_ids=frozenset({"acme", "book_a"})),
        "globex": TenantRecord(tenant_id="globex", default_portfolio_id="globex",
                               portfolio_ids=frozenset({"globex"})),
    }, configured=True)


@pytest.fixture
def projected(tmp_path: Path) -> Path:
    path = tmp_path / "projected.csv"
    path.write_text("A,B,C\n1,2,3\n4,5,6\n", encoding="utf-8")
    return path


@pytest.fixture
def agent(registry, tenants, tmp_path) -> AnnexDeliveryAgent:
    return AnnexDeliveryAgent(registry=registry,
                              store=DeliveryRunStore(tmp_path / "store"),
                              tenant_registry=tenants)


def _request(projected: Path, **overrides) -> DeliveryRequest:
    base = dict(annex_id="TEST_AnnexZ", reporting_date="2026-01-31",
                projected_input=str(projected), output_root="unused")
    base.update(overrides)
    return DeliveryRequest(**base)


def _ctx(tenant: str = "acme") -> ExecutionContext:
    return ExecutionContext.for_internal(tenant, actor_id=f"{tenant}-operator")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class TestRegistry:
    def test_registers_and_resolves_by_id(self, registry):
        provider = registry.resolve("TEST_AnnexZ")
        assert isinstance(provider, StubProvider)
        assert registry.annex_ids() == ["TEST_AnnexZ"]

    def test_resolves_explicit_version(self, registry):
        assert registry.resolve("TEST_AnnexZ", "1.0.0") is not None

    def test_unsupported_annex_is_a_readable_refusal(self, registry):
        with pytest.raises(TraktError) as exc:
            registry.resolve("ESMA_Annex7")
        assert exc.value.code == ErrorCode.UNSUPPORTED_QUESTION
        assert "not a report this service can prepare" in exc.value.message
        assert "TEST_AnnexZ" in exc.value.message

    def test_unsupported_version_names_what_is_available(self, registry):
        with pytest.raises(TraktError) as exc:
            registry.resolve("TEST_AnnexZ", "9.9.9")
        assert "9.9.9" in exc.value.message
        assert "1.0.0" in exc.value.message

    def test_later_registration_becomes_the_default_version(self, registry):
        from dataclasses import replace
        v2 = replace(STUB_DEFINITION, report_version="2.0.0")
        registry.register(v2, StubProvider)
        assert registry.definition("TEST_AnnexZ").report_version == "2.0.0"
        assert registry.definition("TEST_AnnexZ", "1.0.0").report_version == "1.0.0"

    def test_production_registry_carries_annex2_and_no_placeholders(self):
        ids = default_registry().annex_ids()
        assert "ESMA_Annex2" in ids
        assert "ESMA_Annex3" not in ids, "an unimplemented annex must not be registered"
        assert "ESMA_Annex9" not in ids

    def test_stub_never_leaks_into_the_production_registry(self, registry):
        assert "TEST_AnnexZ" not in default_registry().annex_ids()


# --------------------------------------------------------------------------- #
# Lifecycle and state transitions
# --------------------------------------------------------------------------- #


class TestLifecycle:
    def test_happy_path_reaches_prepared_with_warnings(self, agent, projected):
        governed = agent.prepare(_ctx(), _request(projected))
        result = DeliveryResult.from_dict(governed.result)
        assert result.status == DeliveryStatus.PREPARED_WITH_WARNINGS
        assert result.record_count == 2
        assert result.artefact is not None
        assert Path(result.artefact.path).is_file()
        assert result.schema_validation.valid is True
        assert result.publication_eligible is True

    def test_clean_run_reaches_prepared(self, agent, projected):
        StubProvider.evidence_warnings = False
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        assert result.status == DeliveryStatus.PREPARED
        assert result.warnings == ()

    def test_stage_order_is_the_shared_lifecycle(self, agent, projected, registry):
        seen: list[str] = []

        class Recording(StubProvider):
            def __init__(self) -> None:
                super().__init__()
                self.calls = seen

        registry.register(STUB_DEFINITION, Recording)
        agent.prepare(_ctx(), _request(projected))
        assert seen == ["check_configuration", "check_configuration",
                        "check_applicability", "resolve_input", "preflight",
                        "normalise", "build", "validate", "inspect"]

    def test_preflight_block_stops_before_any_build(self, agent, projected):
        StubProvider.fail_configuration = True
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        assert result.status == DeliveryStatus.PREFLIGHT_BLOCKED
        assert result.artefact is None
        assert result.publication_eligible is False
        assert "test configuration has not been supplied" in result.summary

    def test_inapplicable_report_is_blocked_not_built(self, agent, projected):
        StubProvider.not_applicable = True
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        assert result.status == DeliveryStatus.PREFLIGHT_BLOCKED
        assert "not enabled for this client" in result.summary

    def test_missing_input_is_blocking_and_plain_english(self, agent, tmp_path):
        result = DeliveryResult.from_dict(
            agent.prepare(_ctx(), _request(tmp_path / "nope.csv")).result)
        assert result.status == DeliveryStatus.PREFLIGHT_BLOCKED
        assert "was not found" in result.summary
        assert "Traceback" not in result.summary

    def test_normalisation_blocking_issues_stop_the_build(self, agent, projected):
        StubProvider.normalisation_blocking = 4
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        assert result.status == DeliveryStatus.NORMALISATION_FAILED
        assert result.artefact is None
        assert "4 value(s)" in result.summary

    def test_schema_failure_blocks_publication(self, agent, projected):
        StubProvider.schema_valid = False
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        assert result.status == DeliveryStatus.VALIDATION_FAILED
        assert result.publication_eligible is False
        assert result.artefact is not None, "the invalid file is kept as evidence"

    @pytest.mark.parametrize("stage,status", [
        ("normalise", DeliveryStatus.NORMALISATION_FAILED),
        ("build", DeliveryStatus.BUILD_FAILED),
        ("validate", DeliveryStatus.VALIDATION_FAILED),
    ])
    def test_provider_exception_becomes_a_plain_english_failure(
            self, agent, projected, stage, status):
        StubProvider.raise_in = stage
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        assert result.status == status
        assert "stub" not in result.summary.lower(), "raw exception text is not the message"
        assert result.summary.startswith("Test Annex Z could not be prepared")
        detail = result.blocking_errors[0]["technical_detail"]
        assert "RuntimeError" in detail, "the technical cause is preserved, not discarded"

    def test_timings_and_memory_are_recorded(self, agent, projected):
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        stages = {t.stage for t in result.timings}
        assert stages == {"preflight", "normalise", "build", "validate"}
        assert result.total_seconds > 0

    def test_metadata_is_deterministic_across_runs(self, agent, projected, tmp_path):
        first = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        other = AnnexDeliveryAgent(registry=agent.registry,
                                   store=DeliveryRunStore(tmp_path / "store2"),
                                   tenant_registry=agent._tenant_registry)
        second = DeliveryResult.from_dict(other.prepare(_ctx(), _request(projected)).result)
        assert first.run_id == second.run_id
        assert first.component_versions == second.component_versions
        assert first.configuration_versions == second.configuration_versions


# --------------------------------------------------------------------------- #
# Idempotency, restart and atomicity
# --------------------------------------------------------------------------- #


class TestIdempotencyAndRestart:
    def test_rerun_reuses_the_completed_artefact(self, agent, projected, registry):
        first = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)

        calls: list[str] = []

        class Counting(StubProvider):
            def __init__(self) -> None:
                super().__init__()
                self.calls = calls

        registry.register(STUB_DEFINITION, Counting)
        governed = agent.prepare(_ctx(), _request(projected))
        second = DeliveryResult.from_dict(governed.result)

        assert second.run_id == first.run_id
        assert second.artefact.path == first.artefact.path
        assert "build" not in calls, "a completed run must not rebuild"
        assert governed.result["reused_existing_run"] is True

    def test_force_rerun_rebuilds(self, agent, projected, registry):
        agent.prepare(_ctx(), _request(projected))
        calls: list[str] = []

        class Counting(StubProvider):
            def __init__(self) -> None:
                super().__init__()
                self.calls = calls

        registry.register(STUB_DEFINITION, Counting)
        agent.prepare(_ctx(), _request(projected, force_rerun=True,
                                       force_reason="operator requested"))
        assert "build" in calls

    def test_changed_configuration_moves_the_run_identity(self, agent, projected, registry):
        first = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)

        class Reconfigured(StubProvider):
            def check_configuration(self, request, report):
                return {"stub_rules": "fingerprint-2"}

        registry.register(STUB_DEFINITION, Reconfigured)
        second = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        assert second.run_id != first.run_id, (
            "a changed rules fingerprint must not resolve to the earlier artefact")

    def test_changed_input_moves_the_run_identity(self, agent, projected):
        first = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        projected.write_text("A,B,C\n9,9,9\n", encoding="utf-8")
        second = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        assert second.run_id != first.run_id

    def test_restart_resumes_from_the_last_intact_stage(self, agent, projected, registry):
        """A run interrupted after normalisation resumes without renormalising."""
        StubProvider.raise_in = "build"
        agent.prepare(_ctx(), _request(projected))

        calls: list[str] = []

        class Resuming(StubProvider):
            raise_in = ""

            def __init__(self) -> None:
                super().__init__()
                self.calls = calls

        registry.register(STUB_DEFINITION, Resuming)
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)

        assert result.status == DeliveryStatus.PREPARED_WITH_WARNINGS
        assert "normalise" not in calls, "the intact normalisation output is reused"
        assert "build" in calls

    def test_a_truncated_artefact_is_never_reused(self, agent, projected):
        first = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        Path(first.artefact.path).write_text("truncated", encoding="utf-8")
        second = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        assert Path(second.artefact.path).read_text() == "stub artefact\n"

    def test_staged_file_never_leaves_a_partial_destination(self, tmp_path):
        target = tmp_path / "artefact.xml"
        with pytest.raises(RuntimeError):
            with StagedFile(target) as staged:
                staged.temp_path.write_text("half a file", encoding="utf-8")
                raise RuntimeError("interrupted mid-write")
        assert not target.exists()
        assert list(tmp_path.glob("*.part")) == [], "the partial file is cleaned up"

    def test_staged_file_commits_atomically(self, tmp_path):
        target = tmp_path / "artefact.xml"
        target.write_text("previous", encoding="utf-8")
        with StagedFile(target) as staged:
            staged.temp_path.write_text("new", encoding="utf-8")
        assert target.read_text() == "new"

    def test_checkpoint_detects_a_missing_output(self, tmp_path):
        produced = tmp_path / "out.csv"
        produced.write_text("data", encoding="utf-8")
        checkpoint = Checkpoint(run_id="r1")
        checkpoint.mark(STAGE_NORMALISE, outputs={"delivery_ready": str(produced)})
        assert checkpoint.reusable(STAGE_NORMALISE)
        produced.unlink()
        assert not checkpoint.reusable(STAGE_NORMALISE)

    def test_checkpoint_invalidates_downstream_stages(self):
        checkpoint = Checkpoint(run_id="r1")
        for stage in (STAGE_PREFLIGHT, STAGE_NORMALISE, STAGE_BUILD):
            checkpoint.mark(stage)
        checkpoint.invalidate_from(STAGE_NORMALISE)
        assert checkpoint.completed(STAGE_PREFLIGHT)
        assert not checkpoint.completed(STAGE_NORMALISE)
        assert not checkpoint.completed(STAGE_BUILD)


# --------------------------------------------------------------------------- #
# Tenancy
# --------------------------------------------------------------------------- #


class TestTenancy:
    def test_request_cannot_claim_another_client(self, agent, projected):
        governed = agent.prepare(_ctx("acme"), _request(projected, client_id="globex"))
        assert governed.error.code == ErrorCode.TENANT_MISMATCH
        assert governed.result is None

    def test_request_cannot_name_another_tenants_portfolio(self, agent, projected):
        governed = agent.prepare(_ctx("acme"), _request(projected, portfolio_id="globex"))
        assert governed.error.code == ErrorCode.TENANT_MISMATCH

    def test_unauthorised_portfolio_within_namespace_is_refused(self, agent, projected):
        governed = agent.prepare(_ctx("acme"), _request(projected, portfolio_id="book_z"))
        assert governed.error.code == ErrorCode.PORTFOLIO_NOT_AUTHORISED

    def test_portfolio_selector_cannot_traverse_the_store(self, agent, projected):
        governed = agent.prepare(_ctx("acme"), _request(projected, portfolio_id="../../etc"))
        assert governed.error.code == ErrorCode.INVALID_INPUT

    def test_runs_are_stored_under_the_authenticated_tenant(self, agent, projected):
        agent.prepare(_ctx("acme"), _request(projected))
        assert agent.store.list_runs("acme")
        assert agent.store.list_runs("globex") == []

    def test_another_tenant_cannot_reach_the_run(self, agent, projected):
        result = DeliveryResult.from_dict(agent.prepare(_ctx("acme"), _request(projected)).result)
        with pytest.raises(TraktError) as exc:
            agent.gate_for(_ctx("globex"), _request(projected), run_id=result.run_id)
        assert exc.value.code in (ErrorCode.ARTEFACT_NOT_FOUND, ErrorCode.TENANT_MISMATCH)

    def test_result_records_the_trusted_tenant_not_the_claimed_one(self, agent, projected):
        result = DeliveryResult.from_dict(agent.prepare(_ctx("acme"), _request(projected)).result)
        assert result.tenant_id == "acme"


# --------------------------------------------------------------------------- #
# Approval and publication gating
# --------------------------------------------------------------------------- #


class TestPublicationGating:
    @pytest.fixture
    def prepared(self, agent, projected):
        governed = agent.prepare(_ctx(), _request(projected))
        return DeliveryResult.from_dict(governed.result)

    def test_valid_schema_alone_does_not_publish(self, agent, projected, prepared):
        gate = agent.gate_for(_ctx(), _request(projected), run_id=prepared.run_id)
        with pytest.raises(TraktError) as exc:
            gate.publish(_ctx())
        assert exc.value.code == ErrorCode.ARTEFACT_NOT_APPROVED
        assert "prepared, not approved" in exc.value.message

    def test_full_approval_path(self, agent, projected, prepared, tmp_path):
        agent.publication_sink = FilesystemPublicationSink(tmp_path / "published")
        gate = agent.gate_for(_ctx(), _request(projected), run_id=prepared.run_id)

        awaiting = gate.request_approval(_ctx())
        assert awaiting.status == DeliveryStatus.AWAITING_APPROVAL

        approved = gate.approve(_ctx(), note="checked the ND5 fills")
        assert approved.status == DeliveryStatus.APPROVED
        assert approved.approval.decided_by == "acme-operator"

        published = gate.publish(_ctx())
        assert published.status == DeliveryStatus.PUBLISHED
        artefact = Path(published.publication["receipt"]["artefact"])
        assert artefact.is_file()

    def test_publishing_twice_is_refused(self, agent, projected, prepared, tmp_path):
        agent.publication_sink = FilesystemPublicationSink(tmp_path / "published")
        gate = agent.gate_for(_ctx(), _request(projected), run_id=prepared.run_id)
        gate.approve(_ctx())
        gate.publish(_ctx())
        with pytest.raises(TraktError) as exc:
            gate.publish(_ctx())
        assert "already been published" in exc.value.message

    def test_rejection_requires_a_reason_and_blocks_publication(self, agent, projected, prepared):
        gate = agent.gate_for(_ctx(), _request(projected), run_id=prepared.run_id)
        with pytest.raises(TraktError):
            gate.reject(_ctx(), reason="  ")
        rejected = gate.reject(_ctx(), reason="figures look wrong")
        assert rejected.status == DeliveryStatus.REJECTED
        with pytest.raises(TraktError):
            gate.publish(_ctx())

    def test_a_failed_run_cannot_be_approved(self, agent, projected):
        StubProvider.schema_valid = False
        failed = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        gate = agent.gate_for(_ctx(), _request(projected), run_id=failed.run_id)
        with pytest.raises(TraktError):
            gate.approve(_ctx())

    def test_published_package_references_the_full_chain(self, agent, projected,
                                                         prepared, tmp_path):
        agent.publication_sink = FilesystemPublicationSink(tmp_path / "published")
        gate = agent.gate_for(_ctx(), _request(projected), run_id=prepared.run_id)
        gate.approve(_ctx())
        published = gate.publish(_ctx())

        package = published.publication["package"]
        for key in ("source_references", "projected_dataset", "normalised_dataset",
                    "artefact", "schema", "validation_report",
                    "automatic_value_evidence", "operator_approval",
                    "configuration_versions", "component_versions"):
            assert key in package, f"the published package must reference {key}"
        assert package["operator_approval"]["decided_by"] == "acme-operator"
        assert "does not confirm" in package["disclaimer"]

    def test_another_tenant_cannot_approve(self, agent, projected, prepared, tenants):
        # Reach the gate directly: even holding the path, the stored tenant wins.
        from engine.annex_delivery_agent.publication import PublicationGate
        run_dir = Path(prepared.run_directory)
        gate = PublicationGate(store=agent.store, run_dir=run_dir)
        with pytest.raises(TraktError) as exc:
            gate.approve(_ctx("globex"))
        assert exc.value.code == ErrorCode.TENANT_MISMATCH


# --------------------------------------------------------------------------- #
# Operator (OCC) view
# --------------------------------------------------------------------------- #


class TestOperatorView:
    def test_stage_labels_match_the_occ_workflow(self, agent, projected):
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        view = operator_view(result, annex_label="Annex 2")
        labels = [s["label"] for s in view["stages"]]
        assert "Annex 2 data prepared" in labels
        assert "Annex 2 XML created" in labels
        assert "XML passed schema validation" in labels
        assert "Automatic values require review" in labels
        assert "Ready for publication" in labels

    def test_review_stage_demands_action_before_publication(self, agent, projected):
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        view = operator_view(result)
        stages = {s["key"]: s for s in view["stages"]}
        assert stages[STAGE_OPERATOR_REVIEW]["state"] == STATE_ACTION_REQUIRED
        assert stages[STAGE_PUBLICATION]["state"] != STATE_DONE

    def test_view_never_implies_regulatory_correctness(self, agent, projected):
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        view = operator_view(result)
        assert "does not confirm" in view["disclaimer"]
        validation = next(s for s in view["stages"] if s["key"] == "schema_validation")
        assert "not the accuracy of its figures" in validation["detail"]

    def test_automatic_values_are_surfaced_to_the_operator(self, agent, projected):
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        view = operator_view(result)
        assert view["automatic_values"]["fills"] == 7
        assert view["automatic_values"]["coercions"] == 2
        assert view["automatic_values"]["review_required"] is True
        assert any("ZZ1" in line for line in view["automatic_values"]["lines"])


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_every_governed_document_is_written(self, agent, projected):
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        run_dir = Path(result.run_directory)
        for name in ("00_request.json", "01_preflight.json", "02_checkpoint.json",
                     "03_validation.json", "04_transformation_evidence.json",
                     "05_result.json"):
            assert (run_dir / name).is_file(), f"{name} must be persisted"

    def test_failure_evidence_is_written(self, agent, projected):
        StubProvider.raise_in = "build"
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        failure = json.loads((Path(result.run_directory) / "07_failure.json").read_text())
        assert failure["stage"] == "build"
        assert "RuntimeError" in failure["finding"]["technical_detail"]

    def test_store_refuses_to_reset_a_foreign_directory(self, tmp_path):
        store = DeliveryRunStore(tmp_path / "store")
        stranger = tmp_path / "not_ours"
        stranger.mkdir()
        (stranger / "important.txt").write_text("data", encoding="utf-8")
        with pytest.raises(ValueError):
            store.reset_run(stranger)
        assert (stranger / "important.txt").is_file()

    def test_atomic_json_replaces_cleanly(self, tmp_path):
        target = tmp_path / "doc.json"
        atomic_write_json(target, {"a": 1})
        atomic_write_json(target, {"a": 2})
        assert json.loads(target.read_text())["a"] == 2
        assert list(tmp_path.glob("*.part")) == []

    def test_result_round_trips(self, agent, projected):
        original = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(projected)).result)
        restored = agent.store.read_result(Path(original.run_directory))
        assert restored.run_id == original.run_id
        assert restored.status == original.status
        assert restored.artefact.sha256 == original.artefact.sha256

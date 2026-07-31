"""The synthetic boundary: what the OCC Agent must never be able to do.

These are the tests the feature exists to be trusted by. Each one names a
prohibited production action and proves it fails closed, returns a typed error
and leaves an audit event behind.

The one added by joining the OCC Agent to Client Onboarding is the most
important of them: a practice case may be *approved*, but it may never be
*activated* — and activation is, by Client Onboarding's own account, the only
place a client configuration is ever created.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations_control.occ_agent import policy as _policy
from operations_control.occ_agent.artefacts import ArtefactService
from operations_control.occ_agent.execution import SyntheticOnboardingAdapters
from operations_control.occ_agent.policy import (
    CAPABILITIES,
    RUNTIME_MODE_SYNTHETIC,
    SyntheticBoundaryError,
    SyntheticPolicy,
    synthetic_policy,
)
from operations_control.occ_agent.scenarios import run_scenario
from operations_control.occ_agent.store import StoreMisconfigured, SyntheticRunStore
from operations_control.onboarding.case import ACTIVATED, NO_ACTIVE_CONFIGURATION

from .conftest import ACTOR, LIVE_CONTAINER, TENANT_A, live_container_paths

CASE_REF = "ONB-2026-0001"


# --------------------------------------------------------------------------- #
# The policy itself
# --------------------------------------------------------------------------- #

def test_every_prohibited_capability_is_denied_by_construction():
    policy = synthetic_policy()
    assert policy.runtime_mode == RUNTIME_MODE_SYNTHETIC
    for capability in CAPABILITIES:
        assert policy.permits(capability) is False, capability


def test_there_is_no_supported_way_to_build_a_permissive_policy():
    """The dataclass is frozen and the constructor grants nothing.

    A caller who obtains a policy cannot widen it, and the only public factory
    hard-codes every permission to False.
    """
    policy = synthetic_policy()
    with pytest.raises(Exception):
        policy.allow_live_blob_write = True          # frozen dataclass
    assert synthetic_policy().to_dict() == policy.to_dict()


@pytest.mark.parametrize("capability", CAPABILITIES)
def test_each_guard_fails_closed_with_a_typed_error(capability):
    policy = synthetic_policy()
    with pytest.raises(SyntheticBoundaryError) as excinfo:
        policy.require(capability, case_id=CASE_REF, actor="alice")
    assert excinfo.value.capability == capability
    assert excinfo.value.code == _policy.ERROR_CODE
    assert excinfo.value.http_status == 403
    # The message is operator-safe: no path, no code, no internals.
    assert any(word in excinfo.value.message.lower()
               for word in ("synthetic", "practice"))


def test_a_refusal_produces_an_audit_event():
    events = []
    policy = SyntheticPolicy(audit_sink=events.append)
    with pytest.raises(SyntheticBoundaryError):
        policy.guard_live_blob_write(case_id=CASE_REF, actor="alice")
    assert len(events) == 1
    event = events[0]
    assert event["capability"] == _policy.CAP_LIVE_BLOB_WRITE
    assert event["execution_classification"] == "blocked"
    assert event["runtime_mode"] == RUNTIME_MODE_SYNTHETIC
    assert event["case_id"] == CASE_REF


def test_a_failing_audit_sink_does_not_swallow_the_refusal():
    def explode(_event):
        raise RuntimeError("audit is down")

    policy = SyntheticPolicy(audit_sink=explode)
    with pytest.raises(SyntheticBoundaryError):
        policy.guard_publish()


# --------------------------------------------------------------------------- #
# The named prohibited actions
# --------------------------------------------------------------------------- #

def test_activation_is_refused_so_no_configuration_is_ever_created(service):
    """The line a practice case never crosses.

    ``OnboardingService.activate()`` is the only place an active client
    configuration is written. The OCC Agent names it as a capability, refuses
    it, and audits the refusal against the case.
    """
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction="Onboard Northstar Lending.")
    with pytest.raises(SyntheticBoundaryError) as excinfo:
        service.activate(agent_case, actor=ACTOR)
    assert excinfo.value.capability == _policy.CAP_ACTIVATE_CONFIGURATION
    events = service.store.list_audit(TENANT_A, agent_case.case_ref)
    assert any(e["execution_classification"] == "blocked"
               and e["detail"]["capability"] == _policy.CAP_ACTIVATE_CONFIGURATION
               for e in events)


def test_a_completed_practice_case_created_no_configuration(service):
    """The end state: approved, ready for execution, nothing written."""
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert run.case.run.state == "ACTIVATION_CONFIRMATION_REQUIRED"
    assert run.case.case.status in NO_ACTIVE_CONFIGURATION
    assert run.case.case.status != ACTIVATED
    assert run.case.case.activated_version is None
    assert run.case.case.activated_at == ""
    # No configuration version exists for the client either.
    assert service.onboarding.cases.current(run.case.case.client_id) is None
    # And readiness says so as a criterion, not as prose.
    verdict = service.evaluate_readiness(run.case)
    written = next(c for c in verdict["criteria"]
                   if c["key"] == "no_configuration_written")
    assert written["passed"] is True


def test_readiness_fails_if_a_configuration_were_ever_activated(service,
                                                                monkeypatch):
    """The criterion is real: flip the fact and readiness refuses."""
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    run.case.case.status = ACTIVATED
    verdict = service.evaluate_readiness(run.case)
    written = next(c for c in verdict["criteria"]
                   if c["key"] == "no_configuration_written")
    assert written["passed"] is False
    assert verdict["ready"] is False


def test_no_external_email_can_be_sent(service):
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction="Onboard Northstar Lending.")
    with pytest.raises(SyntheticBoundaryError) as excinfo:
        service.send_request_by_email(agent_case, actor=ACTOR)
    assert excinfo.value.capability == _policy.CAP_EXTERNAL_EMAIL
    events = service.store.list_audit(TENANT_A, agent_case.case_ref)
    assert any(e["execution_classification"] == "blocked" for e in events)


def test_no_live_blob_write_can_occur(service):
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction="Onboard Northstar Lending.")
    artefacts = ArtefactService(service.store, service.policy)
    with pytest.raises(SyntheticBoundaryError) as excinfo:
        artefacts.assert_never_written("blob://raw-v2/anything/at/all.csv",
                                       case_id=agent_case.case_ref, actor=ACTOR)
    assert excinfo.value.capability == _policy.CAP_LIVE_BLOB_WRITE


def test_no_live_pipeline_trigger_can_occur(service, agent_env):
    adapters = SyntheticOnboardingAdapters(
        artefact_paths=[], policy=service.policy,
        sandbox=agent_env["sandbox"], case_id=CASE_REF)
    with pytest.raises(SyntheticBoundaryError) as excinfo:
        adapters.trigger_live_pipeline(actor=ACTOR)
    assert excinfo.value.capability == _policy.CAP_LIVE_PIPELINE_TRIGGER


def test_no_production_configuration_can_be_written(service):
    with pytest.raises(SyntheticBoundaryError) as excinfo:
        service.policy.guard_production_config_write(case_id=CASE_REF,
                                                     actor=ACTOR)
    assert excinfo.value.capability == _policy.CAP_PRODUCTION_CONFIG_WRITE


def test_nothing_can_be_marked_published(service):
    """There is no publish path, and the guard refuses the capability."""
    with pytest.raises(SyntheticBoundaryError):
        service.policy.guard_publish(case_id=CASE_REF, actor=ACTOR)
    # And a completed case's package says so explicitly.
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    manifest = service.readiness_package(run.case)["execution_manifest"]
    assert manifest["published"] is False
    assert manifest["execution_performed"] is False
    assert manifest["configuration_activated"] is False
    assert manifest["live_writes"] == []
    assert manifest["emails_sent"] == []


# --------------------------------------------------------------------------- #
# Isolation from live state
# --------------------------------------------------------------------------- #

def test_a_full_run_writes_nothing_to_the_live_container(service, blob_root):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert run.case.run.state == "ACTIVATION_CONFIRMATION_REQUIRED"
    assert live_container_paths(blob_root) == [], (
        "a practice case wrote into the live operations container")


def test_the_onboarding_case_itself_lands_in_the_synthetic_container(
        service, blob_root):
    """Client Onboarding is reused, but pinned to the practice container."""
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    synthetic = Path(blob_root) / "operations-control-synthetic"
    written = [p for p in synthetic.rglob("*.json") if p.is_file()]
    assert any(run.case.case_ref in p.read_text(encoding="utf-8")
               for p in written)
    assert live_container_paths(blob_root) == []


def test_a_full_run_writes_nothing_to_a_raw_delivery_container(service, blob_root):
    run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    for container in ("raw-v2", "processed", "outbound", "trakt-state"):
        assert not (Path(blob_root) / container).exists(), container


def test_the_store_refuses_the_live_container(storage, agent_env, monkeypatch):
    with pytest.raises(StoreMisconfigured):
        SyntheticRunStore(storage, container=LIVE_CONTAINER,
                          sandbox=agent_env["sandbox"])
    monkeypatch.setenv("TRAKT_OPS_CONTAINER", "some-other-live-container")
    with pytest.raises(StoreMisconfigured):
        SyntheticRunStore(storage, container="some-other-live-container",
                          sandbox=agent_env["sandbox"])


def test_intended_live_uris_are_recorded_but_never_written(service, blob_root):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    artefacts = run.case.run.received_artefacts
    assert artefacts
    for artefact in artefacts:
        assert artefact["intended_live_uri"].startswith("blob://raw-v2/")
        assert artefact["execution_status"] == "simulated_only"
        # The intended location must not exist anywhere on disk.
        key = artefact["intended_live_uri"][len("blob://"):]
        assert not (Path(blob_root) / key).exists()


def test_synthetic_mode_needs_no_live_credentials(service, monkeypatch):
    """The whole process runs with every credential variable absent."""
    for name in ("AZURE_STORAGE_CONNECTION_STRING", "AZURE_CLIENT_ID",
                 "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID",
                 "AZURE_STORAGE_ACCOUNT", "TRAKT_BLOB_CONNECTION_STRING"):
        monkeypatch.delenv(name, raising=False)
    _policy.assert_no_live_credentials_required()
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    assert run.case.run.state == "ACTIVATION_CONFIRMATION_REQUIRED"


def test_the_agent_cannot_read_files_outside_its_sandbox(service, agent_env,
                                                         tmp_path):
    """The execution adapter refuses an input path outside the case sandbox."""
    outside = tmp_path / "elsewhere.csv"
    outside.write_text("a,b\n1,2\n", encoding="utf-8")
    from operations_control.occ_agent.execution import SyntheticExecutionError
    with pytest.raises(SyntheticExecutionError):
        SyntheticOnboardingAdapters(
            artefact_paths=[outside], policy=service.policy,
            sandbox=agent_env["sandbox"] / TENANT_A / CASE_REF)


@pytest.mark.parametrize("case_ref", [
    "../../etc", "ONB-../../x", "ONB-2026/../..", "", "onb-2026-0001",
    "ONB-2026-0001/../../x",
])
def test_a_case_reference_cannot_traverse_a_path(case_ref):
    from operations_control.occ_agent.policy import UnsafePathError
    from operations_control.occ_agent.store import validate_case_ref
    with pytest.raises(UnsafePathError):
        validate_case_ref(case_ref)


def test_the_case_reference_pattern_accepts_what_client_onboarding_mints():
    """The practice store must accept a real onboarding reference, exactly."""
    from operations_control.occ_agent.store import validate_case_ref
    from operations_control.onboarding.case import mint_case_id
    assert validate_case_ref(mint_case_id(2026, 1)) == "ONB-2026-0001"


@pytest.mark.parametrize("part", ["..", "../x", "/etc/passwd", "a\\b", "\x00"])
def test_sandbox_path_refuses_anything_that_escapes(tmp_path, part):
    from operations_control.occ_agent.policy import UnsafePathError, sandbox_path
    root = tmp_path / "sandbox"
    root.mkdir()
    with pytest.raises(UnsafePathError):
        sandbox_path(root, part)


def test_sandbox_path_refuses_a_symlinked_escape(tmp_path):
    from operations_control.occ_agent.policy import UnsafePathError, sandbox_path
    root = tmp_path / "sandbox"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafePathError):
        sandbox_path(root, "link", "secret.csv")


@pytest.mark.parametrize("supplied,expected_leaf", [
    ("../../escape.csv", "escape.csv"),
    ("/etc/passwd.csv", "passwd.csv"),
    ("C:\\Windows\\evil.csv", "evil.csv"),
])
def test_a_traversing_artefact_filename_is_neutralised(service, supplied,
                                                       expected_leaf, agent_env):
    """A traversal attempt is reduced to its leaf, not merely rejected.

    The production sanitiser keeps the last segment, so the file lands inside
    the case sandbox under a safe name. What matters is that it cannot land
    anywhere else — which is what this asserts.
    """
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction="Onboard Northstar Lending.")
    artefact = service.artefacts.register(
        agent_case.run, service.facts(agent_case), filename=supplied,
        data=b"a,b\n1,2\n", provided_by=ACTOR)
    assert artefact.source_file == expected_leaf
    sandbox = Path(agent_env["sandbox"]).resolve()
    written = service.store.artefact_path(TENANT_A, agent_case.case_ref,
                                          expected_leaf)
    assert written.exists()
    assert sandbox in written.parents


@pytest.mark.parametrize("name", ["_READY.json", "notes.txt", "script.sh",
                                  "archive.zip", "."])
def test_an_unacceptable_artefact_filename_is_refused(service, name):
    """The readiness sentinel and unreadable file types are refused outright."""
    from operations_control.occ_agent.artefacts import ArtefactRejected
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction="Onboard Northstar Lending.")
    with pytest.raises(ArtefactRejected):
        service.artefacts.register(agent_case.run, service.facts(agent_case),
                                   filename=name, data=b"a,b\n1,2\n",
                                   provided_by=ACTOR)


def test_no_run_document_claims_to_be_live(service):
    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
    doc = run.case.run.to_dict()
    assert doc["synthetic"] is True
    assert doc["runtime_mode"] == RUNTIME_MODE_SYNTHETIC
    # And the persisted document on disk says the same.
    raw = json.loads(
        Path(service.store.storage._local_path(
            service.store.run_uri(TENANT_A, run.case.case_ref))).read_text(
                encoding="utf-8"))
    assert raw["synthetic"] is True
    assert raw["runtime_mode"] == RUNTIME_MODE_SYNTHETIC

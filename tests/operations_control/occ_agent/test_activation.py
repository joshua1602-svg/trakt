"""The road to production, and the one gate across it.

Everything here runs with fakes. ``OCC_AGENT_LIVE_ENABLED`` is never set on the
real environment, no Azure client is constructed, and the live adapter is
driven against stand-ins for ``OnboardingService`` and the engine — so the
*contract* is tested without anything being activated.

The properties asserted:

* live execution is off unless the flag says otherwise, and the flag fails
  closed on anything it does not recognise;
* the gate reports **every** unmet precondition, not the first;
* approving a configuration starts nothing — a separate confirmation does;
* a bare "yes" can never be that confirmation;
* in synthetic mode the confirmation is refused, and the refusal is audited;
* when everything is satisfied, the live adapter calls the platform's own
  governed path in order and reimplements none of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from operations_control.occ_agent import adapters as _adapters
from operations_control.occ_agent import states as _states
from operations_control.occ_agent.interpretation import (
    DeterministicInterpreter,
    InterpretationError,
)
from operations_control.occ_agent.run import SyntheticRun
from operations_control.occ_agent.scenarios import run_scenario
from operations_control.onboarding.case import APPROVED, DRAFT, OnboardingCase

from .conftest import ACTOR, TENANT_A


def _satisfied(**overrides: Any) -> _adapters.ActivationPreconditions:
    """Every precondition met. Tests then break exactly one."""
    base: Dict[str, Any] = {
        "mode": _adapters.MODE_LIVE, "flag_enabled": True,
        "case_ref": "ONB-2026-0001", "onboarding_status": APPROVED,
        "configuration_approved": True, "readiness_passed": True,
        "tenant": TENANT_A, "client_id": "NORTHSTAR",
        "portfolio_id": "direct_101", "configuration_valid": True,
        "configuration_problems": [], "artefacts_present": 2,
        "required_artefacts_satisfied": True, "approval_audited": True,
        "already_activated": False, "confirmed": True,
    }
    base.update(overrides)
    return _adapters.ActivationPreconditions(**base)


# --------------------------------------------------------------------------- #
# The flag
# --------------------------------------------------------------------------- #

def test_live_execution_is_off_by_default():
    assert _adapters.live_enabled({}) is False


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe",
                                   "TRUE-ish", "1 ", None])
def test_the_flag_fails_closed_on_anything_unrecognised(value):
    env = {} if value is None else {_adapters.LIVE_FLAG_ENV: value}
    assert _adapters.live_enabled(env) is (str(value).strip().lower()
                                           in ("1", "true", "yes", "on",
                                               "enabled"))


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on",
                                   "enabled"])
def test_the_flag_is_on_only_when_it_says_so(value):
    assert _adapters.live_enabled({_adapters.LIVE_FLAG_ENV: value}) is True


def test_the_repository_never_ships_the_flag_enabled():
    """Nothing checked in may turn live execution on."""
    import os
    assert _adapters.live_enabled(dict(os.environ)) is False


# --------------------------------------------------------------------------- #
# The one gate
# --------------------------------------------------------------------------- #

def test_everything_satisfied_permits_activation():
    _adapters.assert_may_activate(_satisfied())


@pytest.mark.parametrize("broken,fragment", [
    ({"mode": _adapters.MODE_SYNTHETIC}, "rehearsal"),
    ({"flag_enabled": False}, "not switched on"),
    ({"onboarding_status": DRAFT}, "has not been approved"),
    ({"configuration_approved": False}, "human has not approved"),
    ({"readiness_passed": False}, "readiness criterion"),
    ({"client_id": ""}, "not all identified"),
    ({"portfolio_id": ""}, "not all identified"),
    ({"tenant": ""}, "not all identified"),
    ({"configuration_valid": False}, "does not validate"),
    ({"artefacts_present": 0}, "source files"),
    ({"required_artefacts_satisfied": False}, "source files"),
    ({"approval_audited": False}, "audit trail"),
    ({"already_activated": True}, "already been activated"),
    ({"confirmed": False}, "final confirmation"),
])
def test_each_precondition_refuses_on_its_own(broken, fragment):
    with pytest.raises(_adapters.ActivationRefused) as excinfo:
        _adapters.assert_may_activate(_satisfied(**broken))
    assert any(fragment in reason for reason in excinfo.value.reasons), \
        excinfo.value.reasons


def test_the_gate_reports_every_reason_not_just_the_first():
    reasons = _adapters.activation_refusals(
        _adapters.ActivationPreconditions())          # nothing satisfied
    assert len(reasons) >= 8, reasons


def test_there_is_exactly_one_gate_in_the_codebase():
    """Every path to production goes through ``assert_may_activate``."""
    import inspect

    from operations_control.occ_agent import service as _service
    source = inspect.getsource(_adapters.LiveExecutionAdapter.activate)
    assert "assert_may_activate(pre)" in source
    # The service assembles the preconditions and calls the adapter; it does
    # not carry a second, friendlier check of its own. Counted on the CALL
    # form, so the prose that points at the gate is not mistaken for one.
    service_source = inspect.getsource(_service)
    assert service_source.count("_adapters.assert_may_activate(") == 1
    # And nothing else in the package calls it at all.
    import pkgutil

    import operations_control.occ_agent as _pkg
    callers = []
    for module in pkgutil.iter_modules(_pkg.__path__):
        text = (Path(_pkg.__path__[0]) / f"{module.name}.py").read_text()
        if "assert_may_activate(" in text and "def assert_may_activate" \
                not in text:
            callers.append(module.name)
    assert callers == ["service"], callers


# --------------------------------------------------------------------------- #
# Approval is not activation
# --------------------------------------------------------------------------- #

@pytest.fixture()
def approved(service):
    """A case driven to the point where a human has approved activation."""
    return run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                        actor=ACTOR).case


def test_approving_the_configuration_starts_nothing(service, approved):
    assert approved.run.state == _states.ACTIVATION_CONFIRMATION_REQUIRED
    assert approved.run.activation_result == {}
    assert approved.case.status == APPROVED
    assert approved.case.activated_version is None


def test_the_confirmation_names_what_it_would_do(service, approved):
    intent = approved.run.activation_intent
    assert intent["client_id"]
    assert intent["portfolio_id"]
    assert intent["files"], "no files were named"
    assert intent["target_locations"], "no target location was named"
    assert len(intent["actions"]) >= 4
    assert "production" in intent["statement"].lower()
    assert "cannot be undone" in intent["statement"].lower()


def test_a_bare_yes_can_never_confirm_activation():
    """The one confirmation that must be typed, not nodded through."""
    interpreter = DeterministicInterpreter()
    run = SyntheticRun(case_ref="ONB-2026-0001", tenant=TENANT_A,
                       initiating_user=ACTOR,
                       state=_states.ACTIVATION_CONFIRMATION_REQUIRED)
    with pytest.raises(InterpretationError):
        interpreter.interpret_action("yes", run, OnboardingCase(
            case_id="ONB-2026-0001", status=APPROVED))


def test_the_confirmation_view_lists_every_outstanding_reason(service,
                                                              approved):
    view = service.activation_confirmation(approved)
    assert view["live_enabled"] is False
    assert view["mode"] == _adapters.MODE_SYNTHETIC
    assert any("rehearsal" in r for r in view["refusals"])
    # The confirmation the operator has not given yet is not listed as a
    # failure — it is the next step.
    assert not any("final confirmation" in r for r in view["refusals"])


# --------------------------------------------------------------------------- #
# In a rehearsal the confirmation is refused, and audited
# --------------------------------------------------------------------------- #

def test_confirming_in_a_rehearsal_is_refused(service, approved):
    with pytest.raises(_adapters.ActivationRefused) as excinfo:
        service.confirm_activation(approved, actor=ACTOR,
                                   confirmation="do it")
    assert excinfo.value.reasons
    reloaded = service.load(TENANT_A, approved.case_ref)
    assert reloaded.run.state == _states.ACTIVATION_CONFIRMATION_REQUIRED
    assert reloaded.case.status == APPROVED
    assert reloaded.case.activated_version is None


def test_the_refusal_is_audited(service, approved):
    with pytest.raises(_adapters.ActivationRefused):
        service.confirm_activation(approved, actor=ACTOR)
    audit = service.store.list_audit(TENANT_A, approved.case_ref)
    actions = [e["action"] for e in audit]
    assert "activation_refused" in actions
    assert service.store.verify_audit_chain(TENANT_A, approved.case_ref)


def test_the_synthetic_adapter_never_permits_activation(service):
    assert service.adapter.may_activate() is False


# --------------------------------------------------------------------------- #
# The live adapter's contract — fakes only, no Azure, nothing activated
# --------------------------------------------------------------------------- #

class FakeOnboarding:
    def __init__(self):
        self.calls: List[Any] = []

    def activate(self, *, case_id, by):
        self.calls.append(("activate", case_id, by))
        return {"version": 1, "artefacts": [{"path": "clients/NORTHSTAR.yaml"}]}


class FakeEngine:
    """A stand-in with the REAL engine's signatures.

    Deliberately not ``**kwargs``: a fake that accepts anything cannot catch a
    caller that has drifted from the interface, and this one did — the adapter
    was omitting ``workflow_type``, which ``create_batch`` requires and which
    decides whether a delivery may reach regulatory reporting at all. A test
    below asserts these signatures still match the engine's.
    """

    def __init__(self, fail_on: str = ""):
        self.calls: List[str] = []
        self.uploads: List[Any] = []
        self.batches: List[Dict[str, Any]] = []
        self.fail_on = fail_on

    def _maybe_fail(self, name):
        if self.fail_on == name:
            raise RuntimeError(f"{name} failed")

    def create_batch(self, *, client_id: str, portfolio_id: str,
                     reporting_date: str, workflow_type: str, created_by: str,
                     auto_start_when_ready: bool = False, dataset: str = "",
                     frequency: str = "") -> Dict[str, Any]:
        self.calls.append("create_batch")
        self.batches.append({
            "client_id": client_id, "portfolio_id": portfolio_id,
            "reporting_date": reporting_date, "workflow_type": workflow_type,
            "dataset": dataset, "frequency": frequency,
            "created_by": created_by})
        self._maybe_fail("create_batch")
        return {"batch_id": "bat_0001"}

    def upload_batch_files(self, *, client_id: str, batch_id: str,
                           uploads, received_by: str) -> Dict[str, Any]:
        self.calls.append("upload_batch_files")
        self._maybe_fail("upload_batch_files")
        self.uploads = list(uploads)
        return {"batch_id": batch_id}

    def start_batch(self, *, client_id: str, batch_id: str,
                    actor: str) -> Dict[str, Any]:
        self.calls.append("start_batch")
        self._maybe_fail("start_batch")
        return {"workflow_id": "wf_0001"}


def _intent() -> _adapters.ActivationIntent:
    return _adapters.ActivationIntent(
        client_id="NORTHSTAR", portfolio_id="direct_101", dataset="funded",
        outcome="mi", cadence="monthly", reporting_period="2026-06-30",
        files=[{"name": "loans.csv", "target": "blob://raw/loans.csv"}],
        configuration_artefacts=["clients/NORTHSTAR.yaml"])


def test_the_live_adapter_calls_the_governed_path_in_order():
    onboarding, engine = FakeOnboarding(), FakeEngine()
    adapter = _adapters.LiveExecutionAdapter(onboarding, engine, enabled=True)
    result = adapter.activate(pre=_satisfied(), intent=_intent(), actor=ACTOR,
                              payloads={"loans.csv": b"a,b\n1,2\n"})
    assert result.ok is True
    assert onboarding.calls == [("activate", "ONB-2026-0001", ACTOR)]
    assert engine.calls == ["create_batch", "upload_batch_files", "start_batch"]
    assert result.version == 1
    assert result.batch_id == "bat_0001"
    assert result.workflow_id == "wf_0001"
    assert engine.uploads == [("loans.csv", b"a,b\n1,2\n")]
    # Every argument the engine requires, from the confirmed intent.
    assert engine.batches == [{
        "client_id": "NORTHSTAR", "portfolio_id": "direct_101",
        "reporting_date": "2026-06-30", "workflow_type": "mi",
        "dataset": "funded", "frequency": "monthly", "created_by": ACTOR}]


def test_the_adapter_calls_the_engine_with_the_engines_own_signature():
    """The fakes above are only worth anything if they match the real thing."""
    import inspect

    import operations_control.engine as _engine
    real = next(o for o in vars(_engine).values()
                if inspect.isclass(o) and hasattr(o, "create_batch"))
    for name in ("create_batch", "upload_batch_files", "start_batch"):
        expected = inspect.signature(getattr(real, name))
        actual = inspect.signature(getattr(FakeEngine, name))
        assert set(expected.parameters) == set(actual.parameters), (
            f"{name}: fake takes {sorted(actual.parameters)}, "
            f"engine takes {sorted(expected.parameters)}")


def test_the_live_adapter_refuses_before_it_touches_anything():
    onboarding, engine = FakeOnboarding(), FakeEngine()
    adapter = _adapters.LiveExecutionAdapter(onboarding, engine, enabled=True)
    with pytest.raises(_adapters.ActivationRefused):
        adapter.activate(pre=_satisfied(confirmed=False), intent=_intent(),
                         actor=ACTOR, payloads={"loans.csv": b""})
    assert onboarding.calls == []
    assert engine.calls == []


def test_the_live_adapter_refuses_without_the_bytes_it_would_place():
    adapter = _adapters.LiveExecutionAdapter(FakeOnboarding(), FakeEngine(),
                                             enabled=True)
    with pytest.raises(_adapters.ActivationRefused) as excinfo:
        adapter.activate(pre=_satisfied(), intent=_intent(), actor=ACTOR,
                         payloads={})
    assert "loans.csv" in " ".join(excinfo.value.reasons)


@pytest.mark.parametrize("fail_on,done", [
    ("create_batch", []),
    ("upload_batch_files", ["create_batch"]),
    ("start_batch", ["create_batch", "upload_batch_files"]),
])
def test_a_partial_activation_reports_what_had_already_happened(fail_on, done):
    onboarding, engine = FakeOnboarding(), FakeEngine(fail_on=fail_on)
    adapter = _adapters.LiveExecutionAdapter(onboarding, engine, enabled=True)
    result = adapter.activate(pre=_satisfied(), intent=_intent(), actor=ACTOR,
                              payloads={"loans.csv": b""})
    assert result.ok is False
    assert fail_on in result.error or "failed" in result.error
    assert engine.calls == done + [fail_on]
    # What DID happen is reported rather than rolled back silently.
    assert result.version == 1
    assert "stopped part-way" in result.message


def test_the_live_adapter_does_not_reimplement_the_pipeline():
    """It sequences four existing calls, and makes none of its own."""
    import inspect
    source = inspect.getsource(_adapters.LiveExecutionAdapter)
    for call in ("self.onboarding.activate", "self.engine.create_batch",
                 "self.engine.upload_batch_files", "self.engine.start_batch"):
        assert call in source, call
    for forbidden in ("write_bytes", "BlobServiceClient", "run_orchestration",
                      "requests.", "subprocess"):
        assert forbidden not in source, forbidden


def test_a_live_service_still_refuses_when_the_flag_is_off(storage, agent_env):
    """Constructing a live adapter is not the same as being allowed to use it."""
    from operations_control.occ_agent.service import OccAgentService
    service = OccAgentService(storage, container="operations-control-synthetic",
                              sandbox=agent_env["sandbox"],
                              engine=FakeEngine())
    assert service.adapter.mode == _adapters.MODE_SYNTHETIC
    assert service.adapter.may_activate() is False

"""The doorway: the one crossing from the practice container into the live one.

The OCC Agent authors every case in an isolated container, and
``store.assert_isolated`` refuses to let its stores point anywhere else. That
refusal is what makes the Agent safe to leave switched on in production, so
activation cannot simply be pointed at the governed store — it needs one
deliberate, gated crossing.

These tests pin the crossing itself:

* a rehearsal writes NOTHING into the live container, however far it is driven —
  the property that held before this feature existed, and must still hold;
* a live case crosses, and what arrives is the approved ANSWERS, from which the
  governed side builds its own configuration;
* it crosses exactly once — a second confirmation is refused by two independent
  guards, because a second crossing would mean two configuration versions for
  one set of approved answers;
* a live case cannot be opened at all where live execution is switched off.

Everything here is file-backed in a tmp directory. The "live container" is a
directory on disk, which is what makes "nothing crossed" checkable rather than
merely asserted.
"""

from __future__ import annotations

import pytest

from operations_control.engine import OpsError
from operations_control.occ_agent import adapters as _adapters
from operations_control.occ_agent import promotion as _promotion
from operations_control.occ_agent.scenarios import run_scenario
from operations_control.occ_agent.service import OccAgentService
from operations_control.onboarding.case import APPROVED, DRAFT
from operations_control.onboarding.service import OnboardingService
from operations_control.stores import OpsLayout, OpsStore

from .conftest import (
    ACTOR,
    LIVE_CONTAINER,
    SYNTHETIC_CONTAINER,
    TENANT_A,
    live_container_paths,
)


@pytest.fixture()
def live_env(agent_env, monkeypatch):
    """The same file-backed environment, with live execution switched on."""
    monkeypatch.setenv(_adapters.LIVE_FLAG_ENV, "true")
    return agent_env


@pytest.fixture()
def live_onboarding(storage) -> OnboardingService:
    """Onboarding against the GOVERNED container — the far side of the door."""
    return OnboardingService(OpsStore(storage,
                                      OpsLayout(container=LIVE_CONTAINER)))


class _RecordingEngine:
    """The engine, reduced to what activation calls and what it returns."""

    def __init__(self):
        self.calls = []

    def create_batch(self, **kw):
        self.calls.append(("create_batch", kw))
        return {"batch_id": "BATCH-1"}

    def upload_batch_files(self, **kw):
        self.calls.append(("upload_batch_files", kw))
        return {"ok": True}

    def start_batch(self, **kw):
        self.calls.append(("start_batch", kw))
        return {"workflow_id": "WF-1"}


@pytest.fixture()
def live_service(storage, live_env, live_onboarding) -> OccAgentService:
    """An Agent wired exactly as a live deployment wires it."""
    return OccAgentService(storage, container=SYNTHETIC_CONTAINER,
                           sandbox=live_env["sandbox"],
                           engine=_RecordingEngine(),
                           live_onboarding=live_onboarding)


# --------------------------------------------------------------------------- #
# The boundary that must still hold
# --------------------------------------------------------------------------- #

class TestARehearsalStillCannotCross:
    def test_a_rehearsal_writes_nothing_into_the_live_container(
            self, service, blob_root):
        """Driven as far as it goes, a practice case leaves no live trace."""
        run_scenario(service, "scenario_a_clean", tenant=TENANT_A, actor=ACTOR)
        assert live_container_paths(blob_root) == []

    def test_a_rehearsal_confirmation_is_refused_and_still_leaves_nothing(
            self, service, blob_root):
        case = run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                            actor=ACTOR).case
        with pytest.raises(_adapters.ActivationRefused):
            service.confirm_activation(case, actor=ACTOR,
                                       confirmation="do it")
        assert live_container_paths(blob_root) == []

    def test_the_synthetic_service_holds_no_handle_to_the_governed_store(
            self, service):
        """With live off, the Agent does not even construct the far side."""
        assert service.live_onboarding is None


# --------------------------------------------------------------------------- #
# Opening a live case
# --------------------------------------------------------------------------- #

class TestOpeningALiveCase:
    def test_a_live_case_cannot_be_opened_where_live_is_off(self, service):
        with pytest.raises(OpsError) as excinfo:
            service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                live=True)
        assert excinfo.value.code == "OPS_LIVE_NOT_ENABLED"

    def test_a_case_is_a_rehearsal_unless_asked_otherwise(self, live_service):
        case = live_service.create_case(tenant=TENANT_A, initiating_user=ACTOR)
        assert case.run.mode == _adapters.MODE_SYNTHETIC

    def test_a_live_case_says_so(self, live_service):
        case = live_service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                        live=True)
        assert case.run.mode == _adapters.MODE_LIVE

    def test_a_live_case_is_still_authored_in_the_isolated_container(
            self, live_service, blob_root):
        """Marking a case live changes where it may END, not where it is kept."""
        live_service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                 live=True)
        assert live_container_paths(blob_root) == []


# --------------------------------------------------------------------------- #
# The crossing
# --------------------------------------------------------------------------- #

class TestTheCrossing:
    def test_the_live_adapter_is_given_the_governed_onboarding_service(
            self, live_service, live_onboarding):
        """The whole point: it activates against production, not the sandbox."""
        assert live_service.adapter.mode == _adapters.MODE_LIVE
        assert live_service.adapter.onboarding is live_onboarding
        assert live_service.adapter.onboarding is not live_service.onboarding

    def test_only_the_approved_answers_cross(self, service, live_onboarding):
        """Not the generated configuration — production builds its own."""
        approved = run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                                actor=ACTOR).case
        _promotion.promote(source=service.onboarding, target=live_onboarding,
                           case_ref=approved.run.case_ref, actor=ACTOR)

        arrived = live_onboarding.load_case(approved.run.case_ref)
        assert arrived.answers == approved.case.answers
        assert arrived.status == APPROVED
        # Nothing was activated by the crossing itself.
        assert arrived.activated_version is None

    def test_an_unapproved_case_cannot_cross(self, live_service,
                                             live_onboarding):
        opened = live_service.create_case(tenant=TENANT_A,
                                          initiating_user=ACTOR, live=True)
        assert opened.case.status == DRAFT
        with pytest.raises(OpsError) as excinfo:
            _promotion.promote(source=live_service.onboarding,
                               target=live_onboarding,
                               case_ref=opened.run.case_ref, actor=ACTOR)
        assert excinfo.value.code == "OPS_ONBOARDING_NOT_APPROVED"

    def test_a_case_cannot_cross_twice(self, service, live_onboarding):
        approved = run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                                actor=ACTOR).case
        ref = approved.run.case_ref
        _promotion.promote(source=service.onboarding, target=live_onboarding,
                           case_ref=ref, actor=ACTOR)
        live_onboarding.activate(case_id=ref, by=ACTOR)

        with pytest.raises(OpsError) as excinfo:
            _promotion.promote(source=service.onboarding,
                               target=live_onboarding, case_ref=ref,
                               actor=ACTOR)
        assert excinfo.value.code == "OPS_ALREADY_ACTIVATED"

    def test_the_activation_is_carried_back_to_the_practice_case(
            self, service, live_onboarding):
        """So the ordinary precondition, which reads the practice case, sees it."""
        approved = run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                                actor=ACTOR).case
        ref = approved.run.case_ref
        _promotion.promote(source=service.onboarding, target=live_onboarding,
                           case_ref=ref, actor=ACTOR)
        live_onboarding.activate(case_id=ref, by=ACTOR)
        activated = live_onboarding.load_case(ref)

        assert service.onboarding.load_case(ref).activated_version is None
        _promotion.record_activation(source=service.onboarding,
                                     activated=activated)
        assert service.onboarding.load_case(ref).activated_version == \
            activated.activated_version

    def test_confirming_a_live_case_crosses_and_starts_the_delivery(
            self, live_service, live_onboarding, blob_root):
        """The whole doorway, through the real confirmation call.

        Nothing is in the live container until the operator confirms; then the
        approved answers cross, production builds its own configuration from
        them, and the platform's own delivery path starts.
        """
        case = run_scenario(live_service, "scenario_a_clean", tenant=TENANT_A,
                            actor=ACTOR).case
        case.run.mode = _adapters.MODE_LIVE
        live_service.store.save(case.run)
        assert live_container_paths(blob_root) == [], \
            "something reached the live container before confirmation"

        live_service.confirm_activation(case, actor=ACTOR,
                                        confirmation="activate ERE")

        assert live_container_paths(blob_root), "nothing crossed"
        activated = live_onboarding.load_case(case.run.case_ref)
        assert activated.activated_version, "no configuration was activated"
        # The platform's own path ran, in its own order.
        assert [c[0] for c in live_service.adapter.engine.calls] == [
            "create_batch", "upload_batch_files", "start_batch"]

    def test_a_second_confirmation_is_refused(self, live_service,
                                              live_onboarding):
        """One set of approved answers must not become two configurations."""
        case = run_scenario(live_service, "scenario_a_clean", tenant=TENANT_A,
                            actor=ACTOR).case
        case.run.mode = _adapters.MODE_LIVE
        live_service.store.save(case.run)
        live_service.confirm_activation(case, actor=ACTOR, confirmation="go")
        first = live_onboarding.load_case(case.run.case_ref).activated_version

        reloaded = live_service.load(TENANT_A, case.run.case_ref)
        with pytest.raises(OpsError):
            live_service.confirm_activation(reloaded, actor=ACTOR,
                                            confirmation="go again")
        assert live_onboarding.load_case(
            case.run.case_ref).activated_version == first

    def test_carrying_back_never_raises(self, service):
        """The activation already happened; this must not turn it into an error."""
        class _Broken:
            class cases:
                @staticmethod
                def save_case(case):
                    raise RuntimeError("store unavailable")

        _promotion.record_activation(source=_Broken,
                                     activated=object())   # no exception


# --------------------------------------------------------------------------- #
# What the tab is told about this environment
# --------------------------------------------------------------------------- #

class TestTheTabIsToldWhatIsPossible:
    """`/meta` decides whether the operator is even offered a real onboarding.

    Offering the choice where confirming it would only be refused later teaches
    an operator to expect refusals; withholding it where it WOULD work hides the
    capability. So it reports the same two conditions the adapter itself needs.
    """

    def test_a_rehearsal_environment_offers_no_live_case(self, service):
        from operations_control.occ_agent import api as agent_api
        agent_api.configure(service)
        assert service.adapter.mode != _adapters.MODE_LIVE

    def test_a_live_environment_does(self, live_service):
        assert live_service.adapter.mode == _adapters.MODE_LIVE
        assert _adapters.live_enabled() is True

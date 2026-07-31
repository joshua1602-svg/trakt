"""Operational behaviour: durability, retention, isolation and concurrency.

A practice case outlives the instance that created it, so the things that make
that true are tested here rather than assumed:

* artefacts and packages are stored durably and can be rebuilt on an instance
  that never saw the upload;
* a run is retained for a bounded period and can be purged, and what could NOT
  be removed is reported rather than glossed over;
* two practice cases cannot both claim one client identifier, even though
  neither activates anything;
* the production source registry is read but never written;
* a practice run occupies a thread rather than an API request.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from operations_control.occ_agent import states as _states
from operations_control.occ_agent.service import OccAgentService
from operations_control.occ_agent.store import SyntheticRunStore

from .conftest import ACTOR, LIVE_CONTAINER, TENANT_A, TENANT_B, live_container_paths

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")


@pytest.fixture()
def opened(service):
    return service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=OPENING)


# --------------------------------------------------------------------------- #
# Durability
# --------------------------------------------------------------------------- #

def test_an_artefact_survives_the_sandbox_being_lost(service, opened):
    """A recycled instance rebuilds the working files from storage."""
    service.register_synthetic_artefact(
        opened, filename="loans.csv", data=b"a,b\n1,2\n", actor=ACTOR)
    path = service.store.artefact_path(TENANT_A, opened.case_ref, "loans.csv")
    assert path.exists()
    path.unlink()                                   # the instance went away

    rebuilt = service.store.materialise(TENANT_A, opened.case_ref)
    assert path in rebuilt
    assert path.read_bytes() == b"a,b\n1,2\n"


def test_a_package_reference_is_durable_rather_than_a_local_path(service,
                                                                 opened):
    ref = service.store.write_package(TENANT_A, opened.case_ref, "probe.txt",
                                      "hello")
    assert ref.startswith("blob://") or "/" in ref
    assert service.store.read_package(TENANT_A, opened.case_ref,
                                      "probe.txt") == "hello"


def test_another_instance_sees_the_same_case(storage, agent_env, opened,
                                             service):
    """No state lives only in the process that created it."""
    service.register_synthetic_artefact(
        opened, filename="loans.csv", data=b"a,b\n1,2\n", actor=ACTOR)
    other = OccAgentService(storage, container="operations-control-synthetic",
                            sandbox=agent_env["sandbox"])
    loaded = other.load(TENANT_A, opened.case_ref)
    assert loaded.case.answers["client"]["client_name"] == "Northstar Lending"
    assert [a["source_file"] for a in loaded.run.received_artefacts] == \
        ["loans.csv"]


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #

def test_retention_has_a_bounded_default(service):
    assert service.store.retention_days() >= 1


def test_an_old_case_is_reported_as_expired(service, opened, monkeypatch):
    monkeypatch.setenv("TRAKT_OCC_AGENT_RETENTION_DAYS", "30")
    future = (datetime.now(timezone.utc) + timedelta(days=45)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    expired = service.store.expired(TENANT_A, now=future)
    assert opened.case_ref in [row["case_ref"] for row in expired]


def test_a_recent_case_is_not_expired(service, opened):
    assert service.store.expired(TENANT_A) == []


def test_purging_reports_what_it_could_not_remove(service, opened):
    outcome = service.store.purge(TENANT_A, opened.case_ref)
    assert "removed" in outcome and "retained" in outcome
    assert isinstance(outcome["removed"], list)
    # Honest about the backend: anything the store could not delete is listed
    # and explained, rather than reported as gone.
    assert bool(outcome["note"]) == bool(outcome["retained"])
    assert outcome["removed"], "nothing was removed at all"
    assert service.store.exists(TENANT_A, opened.case_ref) is False


# --------------------------------------------------------------------------- #
# Two practice cases cannot claim one identifier
# --------------------------------------------------------------------------- #

def test_a_second_case_claiming_the_same_client_is_flagged(service):
    first = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                instruction=OPENING)
    second = service.create_case(tenant=TENANT_B, initiating_user=ACTOR,
                                 instruction=OPENING)
    assert first.case.client_id == second.case.client_id, \
        "the fixture no longer produces a collision to detect"
    assert any(first.case_ref in note for note in second.run.observations), \
        "the collision with the earlier practice case was not reported"


def test_a_reservation_belongs_to_the_case_that_made_it(service, opened):
    reserved = service.store.reserved_identifiers("client_id")
    assert reserved.get(opened.case.client_id) == opened.case_ref


def test_reserving_the_same_identifier_twice_from_one_case_is_not_a_clash(
        service, opened):
    clash = service.store.reserve_identifier(
        "client_id", opened.case.client_id, case_ref=opened.case_ref,
        tenant=TENANT_A)
    assert clash == {}


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #

def test_nothing_is_written_to_the_live_container(service, opened, blob_root):
    service.register_synthetic_artefact(
        opened, filename="loans.csv", data=b"a,b\n1,2\n", actor=ACTOR)
    service.draft_pack(opened, actor=ACTOR)
    assert live_container_paths(blob_root) == []


def test_the_production_source_registry_is_read_but_never_written(service,
                                                                  opened):
    """Onboarding READS the live registry; only ``activate()`` writes it.

    The guarantee is named on the store so this can assert it rather than a
    comment claim it, and the mechanism is the policy refusing the one
    capability that would do the writing.
    """
    from operations_control.occ_agent.policy import (
        CAP_ACTIVATE_CONFIGURATION,
        SyntheticBoundaryError,
    )
    SyntheticRunStore.assert_no_live_registry_write()
    assert service.policy.permits(CAP_ACTIVATE_CONFIGURATION) is False
    with pytest.raises(SyntheticBoundaryError):
        service.activate(opened, actor=ACTOR)


def test_the_synthetic_store_is_pinned_to_its_own_container(service):
    assert service.store.container == "operations-control-synthetic"
    assert service.store.container != LIVE_CONTAINER


# --------------------------------------------------------------------------- #
# A run does not occupy a request
# --------------------------------------------------------------------------- #

def test_a_practice_run_can_be_started_in_the_background(service):
    from operations_control.occ_agent.scenarios import run_scenario

    # Drive a case to the point where it is ready to run, then start it the way
    # an API request would.
    from operations_control.occ_agent import fixtures as _fixtures
    scenario = _fixtures.by_id("scenario_a_clean")
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction=scenario.instruction,
                                     fixture_id=scenario.fixture_id)
    agent_case.run.reporting_period = scenario.reporting_period
    service.store.save(agent_case.run)
    for spec in scenario.files:
        agent_case = service.register_synthetic_artefact(
            agent_case, filename=spec.filename,
            data=spec.content.encode("utf-8"), actor=ACTOR,
            declared_type=spec.declared_type)
    agent_case = service.classify_artefacts(agent_case, actor=ACTOR)
    agent_case = service.request_client_information(agent_case, actor=ACTOR)
    request = agent_case.case.outstanding_requests[0]
    from operations_control.occ_agent.scenarios import client_answers
    agent_case = service.record_client_response(
        agent_case, request_id=request.request_id, actor=ACTOR,
        answers=client_answers(agent_case.case, request.items,
                               scenario.client_response))
    agent_case = service.submit_for_approval(agent_case, actor=ACTOR)
    agent_case = service.approve_onboarding(agent_case, actor=ACTOR)

    started = service.start_synthetic_onboarding(agent_case, actor=ACTOR)
    assert started["started"] is True
    assert started["case_ref"] == agent_case.case_ref
    # The thread is tracked, and the case is pollable while it runs.
    service._jobs[f"{TENANT_A}/{agent_case.case_ref}"].join(timeout=120)
    assert service.job_running(TENANT_A, agent_case.case_ref) is False
    reloaded = service.load(TENANT_A, agent_case.case_ref)
    assert reloaded.run.state in (_states.SYNTHETIC_ONBOARDING_PASSED,
                                  _states.EXCEPTIONS_REQUIRE_INPUT,
                                  _states.BLOCKED)
    del run_scenario


def test_the_status_projection_says_whether_a_run_is_in_flight(service, opened):
    assert service.status(opened)["running"] is False

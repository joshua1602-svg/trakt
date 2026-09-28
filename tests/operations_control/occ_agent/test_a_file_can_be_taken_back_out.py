"""An uploaded file can be removed, and a period cannot be typed wrong.

TWO HALVES OF ONE SCREEN, BOTH MISSING.

1. UPLOADING WAS A ONE-WAY DOOR.

   ``run.received_artefacts.append(...)`` was the only way the list ever
   changed. An operator who attached the wrong file — an encrypted workbook,
   a draft, last month's tape — had two options: re-upload under the same name,
   which overwrites the BYTES (``write_artefact_bytes`` writes a sanitised leaf
   with no uniquifier) but appends a second ROW, so the case then recorded two
   files where one had arrived; or cancel the case and start again.

   Neither is a correction. The first makes the record say something untrue
   about what the client sent, and ``build_intent`` would name the file twice
   at activation; the second costs the case and any pack already issued.

2. THE RUN TARGET COULD NOT BE SET FROM ANYWHERE.

   ``POST /cases/{ref}/target`` existed, ``setAgentRunTarget`` existed in the
   API client — and no screen called it. The visible symptom was a file card
   reading "Where this would be filed: —", because ``ArtefactService.
   intended_uri`` derives nothing until client, portfolio AND reporting period
   are all known, and the period is the one of the three with no way in.

   Nothing had ever reached that endpoint from a person, so it stored whatever
   it was sent. A period is a path segment and part of the pack key; "April
   2026" written into one is not an error anybody sees, it is a folder with
   that name. Now it is checked at the door, exactly as a frequency is.

WHAT REMOVAL REMOVES. The case's RECORD of the file, which is what every
consequence reads — the activation intent's file list, ``_payloads``,
classification and role readiness all iterate ``run.artefacts()``. The bytes
stay in the case's own sandbox. ``Storage`` has no delete, and giving a
governed store one is a far larger decision than letting an operator undo an
upload; the audit entry says so rather than implying the bytes are gone.
"""

from __future__ import annotations

import pytest

from operations_control.engine import OpsError
from operations_control.occ_agent.service import OccAgentService

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")

LOANS = b"loan_id,current_balance,interest_rate\nL1,100,0.05\n"
PROPERTY = b"loan_id,property_value,postcode\nL1,250000,SW1\n"


@pytest.fixture()
def opened(service):
    return service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=OPENING)


def artefacts(agent_case):
    return [a["source_file"] for a in agent_case.run.received_artefacts]


def only_artefact_id(agent_case, filename):
    return next(a["artefact_id"] for a in agent_case.run.received_artefacts
                if a["source_file"] == filename)


def _removal_entry(service, agent_case):
    entries = [e for e in service.store.list_audit(TENANT_A,
                                                   agent_case.case_ref)
               if e.get("action") == "synthetic_artefact_removed"]
    assert entries, "removing a file is audited"
    return entries[-1]


# --------------------------------------------------------------------------- #
# Taking a file back out
# --------------------------------------------------------------------------- #

class TestTheWrongFileCanBeRemoved:
    def test_it_leaves_the_pack(self, service, opened):
        service.register_synthetic_artefact(
            opened, filename="LoanExtract.xlsx", data=LOANS, actor=ACTOR)
        service.register_synthetic_artefact(
            opened, filename="encrypted.xlsx", data=b"not a workbook",
            actor=ACTOR)
        after = service.remove_synthetic_artefact(
            opened, artefact_id=only_artefact_id(opened, "encrypted.xlsx"),
            actor=ACTOR)
        assert artefacts(after) == ["LoanExtract.xlsx"]

    def test_the_other_files_are_untouched(self, service, opened):
        """Removing one file must not disturb what the client did send."""
        service.register_synthetic_artefact(
            opened, filename="LoanExtract.xlsx", data=LOANS, actor=ACTOR)
        service.register_synthetic_artefact(
            opened, filename="PropertyExtract.xlsx", data=PROPERTY,
            actor=ACTOR)
        service.register_synthetic_artefact(
            opened, filename="wrong.csv", data=b"x\n1\n", actor=ACTOR)
        after = service.remove_synthetic_artefact(
            opened, artefact_id=only_artefact_id(opened, "wrong.csv"),
            actor=ACTOR)
        assert artefacts(after) == ["LoanExtract.xlsx", "PropertyExtract.xlsx"]
        kept = after.run.received_artefacts[0]
        assert kept["artefact_type"] == "loan_extract"

    def test_the_removal_is_durable(self, service, storage, agent_env, opened):
        """Another instance must not see the file come back."""
        service.register_synthetic_artefact(
            opened, filename="wrong.csv", data=b"x\n1\n", actor=ACTOR)
        service.remove_synthetic_artefact(
            opened, artefact_id=only_artefact_id(opened, "wrong.csv"),
            actor=ACTOR)
        other = OccAgentService(storage,
                                container="operations-control-synthetic",
                                sandbox=agent_env["sandbox"])
        assert other.load(TENANT_A, opened.case_ref).run.received_artefacts \
            == []

    def test_removing_the_last_file_is_allowed(self, service, opened):
        """An operator who attached one wrong file gets back to empty."""
        service.register_synthetic_artefact(
            opened, filename="wrong.csv", data=b"x\n1\n", actor=ACTOR)
        after = service.remove_synthetic_artefact(
            opened, artefact_id=only_artefact_id(opened, "wrong.csv"),
            actor=ACTOR)
        assert artefacts(after) == []

    def test_a_file_that_is_not_there_is_refused(self, service, opened):
        with pytest.raises(OpsError) as exc:
            service.remove_synthetic_artefact(
                opened, artefact_id="sart_nothing", actor=ACTOR)
        assert exc.value.http_status == 404

    def test_the_record_says_who_removed_what(self, service, opened):
        service.register_synthetic_artefact(
            opened, filename="wrong.csv", data=b"x\n1\n", actor=ACTOR)
        service.remove_synthetic_artefact(
            opened, artefact_id=only_artefact_id(opened, "wrong.csv"),
            actor=ACTOR)
        entry = _removal_entry(service, opened)
        assert entry["actor_identity"] == ACTOR
        assert entry["input_reference"] == "wrong.csv"
        assert service.store.verify_audit_chain(TENANT_A, opened.case_ref)

    def test_the_record_does_not_claim_the_bytes_were_deleted(self, service,
                                                              opened):
        """`Storage` has no delete. The audit trail must not imply one."""
        service.register_synthetic_artefact(
            opened, filename="wrong.csv", data=b"x\n1\n", actor=ACTOR)
        service.remove_synthetic_artefact(
            opened, artefact_id=only_artefact_id(opened, "wrong.csv"),
            actor=ACTOR)
        assert _removal_entry(service, opened)["detail"]["bytes_deleted"] \
            is False


class TestWhatFollowsFromThePackIsRecomputed:
    """A stale readiness verdict is worse than none: it reads as checked."""

    def test_readiness_no_longer_counts_the_removed_file(self, service,
                                                         opened):
        service.register_synthetic_artefact(
            opened, filename="LoanExtract.xlsx", data=LOANS, actor=ACTOR)
        service.classify_artefacts(opened, actor=ACTOR)
        before = service.artefacts.readiness(opened.run, "mi")
        assert "loan_extract" in before.satisfied

        after = service.remove_synthetic_artefact(
            opened, artefact_id=only_artefact_id(opened, "LoanExtract.xlsx"),
            actor=ACTOR)
        assert "loan_extract" in service.artefacts.readiness(
            after.run, "mi").missing

    def test_the_onboarding_cases_sample_shrinks_too(self, service, opened):
        """Client Onboarding infers file names from the sample pack."""
        service.register_synthetic_artefact(
            opened, filename="LoanExtract.xlsx", data=LOANS, actor=ACTOR)
        service.register_synthetic_artefact(
            opened, filename="wrong.csv", data=b"x\n1\n", actor=ACTOR)
        service.classify_artefacts(opened, actor=ACTOR)
        after = service.remove_synthetic_artefact(
            opened, artefact_id=only_artefact_id(opened, "wrong.csv"),
            actor=ACTOR)
        names = [f["name"]
                 for f in after.case.answers["sample"]["files"]]
        assert names == ["LoanExtract.xlsx"]

    def test_activation_would_no_longer_place_the_removed_file(self, service,
                                                               opened):
        """The whole point: what activation places is the pack, so the wrong
        file must not reach production raw."""
        service.register_synthetic_artefact(
            opened, filename="LoanExtract.xlsx", data=LOANS, actor=ACTOR)
        service.register_synthetic_artefact(
            opened, filename="encrypted.xlsx", data=b"not a workbook",
            actor=ACTOR)
        after = service.remove_synthetic_artefact(
            opened, artefact_id=only_artefact_id(opened, "encrypted.xlsx"),
            actor=ACTOR)
        placed = [a.source_file for a in after.run.artefacts()]
        assert placed == ["LoanExtract.xlsx"]
        assert "encrypted.xlsx" not in service._payloads(after.run)


# --------------------------------------------------------------------------- #
# The run target
# --------------------------------------------------------------------------- #

class TestAReportingPeriodIsCheckedAtTheDoor:
    def test_a_month_is_accepted(self):
        from apps.blob_trigger_app.path_parser import canonical_period
        assert canonical_period("2026-08") == "2026-08"

    @pytest.mark.parametrize("given,expected", [
        ("2026_08", "2026-08"),
        ("2026_09_14", "2026-09-14"),
        (" 2026-09-14 ", "2026-09-14"),
        ("2026-w39", "2026-W39"),
        ("2026-q2", "2026-Q2"),
    ])
    def test_the_spellings_a_person_uses_are_canonicalised(self, given,
                                                           expected):
        from apps.blob_trigger_app.path_parser import canonical_period
        assert canonical_period(given) == expected

    @pytest.mark.parametrize("given", ["April 2026", "2026", "Q2",
                                       "next month", "2026-13-01x"])
    def test_a_period_trakt_cannot_read_is_refused(self, given):
        from apps.blob_trigger_app.path_parser import (PathParseError,
                                                       canonical_period)
        with pytest.raises(PathParseError):
            canonical_period(given)

    def test_blank_stays_blank_for_the_caller_to_default(self):
        from apps.blob_trigger_app.path_parser import canonical_period
        assert canonical_period("") == ""
        assert canonical_period(None) == ""

    def test_the_canonical_form_is_one_a_delivery_path_accepts(self):
        """The period becomes a folder; the parser must read it back."""
        from apps.blob_trigger_app.path_parser import (_PERIOD_RE,
                                                       canonical_period)
        assert _PERIOD_RE.match(canonical_period("2026_08"))


# --------------------------------------------------------------------------- #
# Over the API, which is what the screen will call
# --------------------------------------------------------------------------- #

HEADERS = {"X-Operator-Token": "tok-a"}


@pytest.fixture()
def api_client(agent_env):
    import importlib

    from fastapi.testclient import TestClient

    from apps.blob_trigger_app.storage import Storage
    import operations_control.api.app as app_module
    from operations_control.occ_agent import api as agent_api
    importlib.reload(app_module)
    agent_api.configure(OccAgentService(
        Storage(agent_env["blob_root"]),
        container="operations-control-synthetic",
        sandbox=agent_env["sandbox"]))
    return TestClient(app_module.app)


@pytest.fixture()
def case_ref(api_client):
    created = api_client.post("/ops/agent/cases", headers=HEADERS,
                              json={"instruction": OPENING})
    return created.json()["case_ref"]


def upload(api_client, case_ref, filename, data):
    return api_client.post(
        f"/ops/agent/cases/{case_ref}/artefacts", headers=HEADERS,
        files={"files": (filename, data, "text/csv")}).json()


class TestTheRoutesTheScreenCalls:
    def test_the_target_route_stores_a_canonical_period(self, api_client,
                                                        case_ref):
        body = api_client.post(f"/ops/agent/cases/{case_ref}/target",
                               headers=HEADERS,
                               json={"reporting_period": "2026_08",
                                     "dataset": "funded"}).json()
        assert body["run"]["reporting_period"] == "2026-08"
        assert body["run"]["dataset"] == "funded"

    def test_a_period_a_person_mistyped_is_refused_not_stored(self, api_client,
                                                              case_ref):
        refused = api_client.post(f"/ops/agent/cases/{case_ref}/target",
                                  headers=HEADERS,
                                  json={"reporting_period": "April 2026"})
        assert refused.status_code == 400
        after = api_client.get(f"/ops/agent/cases/{case_ref}",
                               headers=HEADERS).json()
        assert after["run"]["reporting_period"] == ""

    def test_a_book_trakt_does_not_report_on_is_refused(self, api_client,
                                                        case_ref):
        refused = api_client.post(f"/ops/agent/cases/{case_ref}/target",
                                  headers=HEADERS,
                                  json={"dataset": "securitised"})
        assert refused.status_code == 400

    def test_setting_the_period_gives_the_file_a_destination(self, api_client,
                                                             case_ref):
        """The visible symptom was "Where this would be filed: —"."""
        upload(api_client, case_ref, "LoanExtract.xlsx", LOANS)
        before = api_client.get(f"/ops/agent/cases/{case_ref}",
                                headers=HEADERS).json()
        assert before["run"]["received_artefacts"][0]["intended_live_uri"] == ""

        after = api_client.post(f"/ops/agent/cases/{case_ref}/target",
                                headers=HEADERS,
                                json={"reporting_period": "2026-08"}).json()
        assert "2026-08" in after["run"]["received_artefacts"][0][
            "intended_live_uri"]

    def test_naming_the_period_after_uploading_is_not_too_late(self,
                                                               api_client,
                                                               case_ref):
        """`intended_live_uri` is set when a file is REGISTERED, and the period
        is normally the last thing known. Upload-then-name is the ordinary
        order of work, so every file is re-derived, not only the next one."""
        upload(api_client, case_ref, "LoanExtract.xlsx", LOANS)
        upload(api_client, case_ref, "PropertyExtract.xlsx", PROPERTY)
        body = api_client.post(f"/ops/agent/cases/{case_ref}/target",
                               headers=HEADERS,
                               json={"reporting_period": "2026-08"}).json()
        assert all("2026-08" in a["intended_live_uri"]
                   for a in body["run"]["received_artefacts"])

    def test_correcting_the_period_moves_the_files_with_it(self, api_client,
                                                           case_ref):
        upload(api_client, case_ref, "LoanExtract.xlsx", LOANS)
        api_client.post(f"/ops/agent/cases/{case_ref}/target", headers=HEADERS,
                        json={"reporting_period": "2026-08"})
        body = api_client.post(f"/ops/agent/cases/{case_ref}/target",
                               headers=HEADERS,
                               json={"reporting_period": "2026-04"}).json()
        uri = body["run"]["received_artefacts"][0]["intended_live_uri"]
        assert "2026-04" in uri and "2026-08" not in uri

    def test_naming_the_target_is_recorded(self, api_client, case_ref):
        api_client.post(f"/ops/agent/cases/{case_ref}/target", headers=HEADERS,
                        json={"reporting_period": "2026-08"})
        audit = api_client.get(f"/ops/agent/cases/{case_ref}/audit",
                               headers=HEADERS).json()
        entries = [e for e in audit["events"]
                   if e.get("action") == "run_target_set"]
        assert entries and entries[-1]["detail"]["reporting_period"] \
            == "2026-08"

    def test_a_finished_case_cannot_be_retargeted(self, api_client, case_ref):
        """Renaming the period of a delivery that already ran would leave the
        record describing a period the files did not go to."""
        api_client.post(f"/ops/agent/cases/{case_ref}/cancel", headers=HEADERS,
                        json={"reason": "opened against the wrong client"})
        refused = api_client.post(f"/ops/agent/cases/{case_ref}/target",
                                  headers=HEADERS,
                                  json={"reporting_period": "2026-08"})
        assert refused.status_code == 409

    def test_a_file_can_be_removed_over_the_api(self, api_client, case_ref):
        upload(api_client, case_ref, "LoanExtract.xlsx", LOANS)
        body = upload(api_client, case_ref, "wrong.csv", b"x\n1\n")
        wrong = next(a for a in body["run"]["received_artefacts"]
                     if a["source_file"] == "wrong.csv")
        after = api_client.post(
            f"/ops/agent/cases/{case_ref}/artefacts/remove", headers=HEADERS,
            json={"artefact_id": wrong["artefact_id"]})
        assert after.status_code == 200
        assert [a["source_file"]
                for a in after.json()["run"]["received_artefacts"]] \
            == ["LoanExtract.xlsx"]

    def test_removing_a_file_that_is_not_there_answers_404(self, api_client,
                                                           case_ref):
        refused = api_client.post(
            f"/ops/agent/cases/{case_ref}/artefacts/remove", headers=HEADERS,
            json={"artefact_id": "sart_nothing"})
        assert refused.status_code == 404

    def test_another_tenant_cannot_remove_this_cases_files(self, api_client,
                                                           case_ref):
        """Same 404 as every other route: a case is invisible, not forbidden."""
        body = upload(api_client, case_ref, "wrong.csv", b"x\n1\n")
        wrong = body["run"]["received_artefacts"][0]["artefact_id"]
        denied = api_client.post(
            f"/ops/agent/cases/{case_ref}/artefacts/remove",
            headers={"X-Operator-Token": "tok-b"},
            json={"artefact_id": wrong})
        assert denied.status_code == 404
